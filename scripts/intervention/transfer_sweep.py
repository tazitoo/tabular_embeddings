#!/usr/bin/env python3
"""Cross-model virtual concept transfer: improve weak model with strong model's concepts.

The constructive complement to ablation: instead of removing the strong model's
concepts to degrade it, we transfer them to the weak model to improve it.

Pipeline:
    1. Build concept bridge: MNN-matched decoder atoms → ridge map → virtual atoms
    2. For each row where weak model underperforms strong:
       a. Encode source embeddings → extract unmatched activations
       b. Compute virtual deltas: activation × virtual_atom (in target space)
       c. Inject into target tail model
       d. Greedy accept: keep concept only if prediction moves toward strong model
    3. Find optimal k per row (fewest concepts to close the gap)

Uses the same backbone (intervene_lib) as perrow_importance and ablation_sweep.

Output:
    output/transfer_sweep/{source}_to_{target}/{dataset}.npz

Usage:
    python -m scripts.intervention.transfer_sweep --source tabpfn --target tabicl \
        --datasets credit-g --device cuda
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH, SAE_DATA_DIR,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.intervene_sae import intervene
from scripts.intervention.transfer_concepts import capture_embeddings
from scripts.intervention.transfer_virtual_nodes import (
    build_concept_bridge,
    build_mnn_matches,
    extract_decoder_atoms,
    load_cross_correlations,
    compute_local_transfer_delta,
    _encode_unmatched_activations,
    _make_virtual_delta,
    _build_full_delta_from_parts,
    compute_virtual_delta_perrow,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast", "carte", "tabula8b"]


def get_unmatched_features(
    source_model: str, target_model: str,
    concept_labels_path: Optional[Path] = None,
) -> List[int]:
    """Get source features that are NOT matched to the target model.

    These are the features we want to transfer — concepts the strong model
    has that the weak model lacks.
    """
    from scripts.intervention.intervene_lib import (
        get_alive_features, MODEL_KEY_TO_LABEL_KEY, DEFAULT_CONCEPT_LABELS,
    )

    if concept_labels_path is None:
        concept_labels_path = DEFAULT_CONCEPT_LABELS

    with open(concept_labels_path) as f:
        data = json.load(f)

    src_key = MODEL_KEY_TO_LABEL_KEY.get(source_model, source_model)
    tgt_key = MODEL_KEY_TO_LABEL_KEY.get(target_model, target_model)

    # Features matched to target via concept groups
    matched_source = set()
    for gid, group in data.get("concept_groups", {}).items():
        members = group.get("members", [])
        models_in_group = set(m for m, _ in members)
        if src_key in models_in_group and tgt_key in models_in_group:
            for m, f in members:
                if m == src_key:
                    matched_source.add(f)

    # All alive source features minus matched
    all_source = set(get_alive_features(source_model, concept_labels_path))
    unmatched = sorted(all_source - matched_source)
    return unmatched


def run_dataset_local(
    source_model: str,
    target_model: str,
    dataset: str,
    sae_source: torch.nn.Module,
    sae_target: torch.nn.Module,
    matched_pairs: List,
    unmatched_source: List[int],
    splits: dict,
    norm_stats_source: dict,
    norm_stats_target: dict,
    device: str,
) -> dict:
    """Per-row local linear transfer: no global map needed.

    For each row, fits a local ridge map on the matched features that fire,
    then applies it to unmatched source contributions to get the target delta.
    Injects via full-sequence intervene().
    """
    # Load context + query for target model
    X_train_t, y_train_t, X_query_t, y_query_t, row_indices, task = load_dataset_context(
        target_model, dataset, splits,
    )
    n_query = len(X_query_t)

    if y_train_t.dtype == np.int32:
        y_train_t = y_train_t.astype(np.int64)

    # Capture target embeddings + baseline predictions
    extraction_layer_t = get_extraction_layer_taskaware(target_model, dataset=dataset)
    t0 = time.time()
    emb_target, weak_preds = capture_embeddings(
        target_model, X_train_t, y_train_t, X_query_t,
        extraction_layer_t, device, task,
    )
    weak_preds = np.asarray(weak_preds)
    weak_loss = compute_per_row_loss(y_query_t, weak_preds, task)
    n_ctx_target = emb_target.shape[0] - n_query

    # Get strong model predictions
    imp_path = IMPORTANCE_DIR / source_model / f"{dataset}.npz"
    if imp_path.exists():
        imp = np.load(imp_path, allow_pickle=True)
        strong_preds = imp["baseline_preds"]
    else:
        X_train_s, y_train_s, X_query_s, _, _, _ = load_dataset_context(
            source_model, dataset, splits,
        )
        if y_train_s.dtype == np.int32:
            y_train_s = y_train_s.astype(np.int64)
        extraction_layer_s = get_extraction_layer_taskaware(source_model, dataset=dataset)
        _, strong_preds = capture_embeddings(
            source_model, X_train_s, y_train_s, X_query_s,
            extraction_layer_s, device, task,
        )
        strong_preds = np.asarray(strong_preds)

    strong_loss = compute_per_row_loss(y_query_t, strong_preds, task)

    logger.info(f"  Captured in {time.time() - t0:.1f}s")
    logger.info(f"  Strong ({source_model}) mean loss: {strong_loss.mean():.4f}")
    logger.info(f"  Weak ({target_model}) mean loss: {weak_loss.mean():.4f}")

    strong_wins = strong_loss < weak_loss
    n_strong_wins = strong_wins.sum()
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {"n_strong_wins": 0, "n_query": n_query}

    # Get decoder atoms
    atoms_source = extract_decoder_atoms(sae_source).numpy()
    atoms_target = extract_decoder_atoms(sae_target).numpy()

    # Encode BOTH models' SAE activations on query embeddings
    per_ds_s = load_test_embeddings(source_model)
    per_ds_t = load_test_embeddings(target_model)

    ds_mean_s, ds_std_s = norm_stats_source[dataset]
    ds_mean_t, ds_std_t = norm_stats_target[dataset]

    with torch.no_grad():
        emb_s = torch.tensor(per_ds_s[dataset], dtype=torch.float32, device=device)
        h_source = sae_source.encode(emb_s).cpu().numpy()

        emb_t = torch.tensor(per_ds_t[dataset], dtype=torch.float32, device=device)
        h_target = sae_target.encode(emb_t).cpu().numpy()

    data_std_t = torch.tensor(ds_std_t, dtype=torch.float32, device=device)

    # Per-row local transfer
    preds_transferred = weak_preds.copy()
    gap_closed = np.zeros(n_query, dtype=np.float32)
    n_matched_firing = np.zeros(n_query, dtype=np.int32)

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            continue

        target_loss_r = strong_loss[r]
        orig_gap = weak_loss[r] - target_loss_r
        if orig_gap <= 0:
            continue

        # Compute local transfer delta for this row
        delta_target = compute_local_transfer_delta(
            atoms_source, atoms_target,
            h_source[r], h_target[r],
            matched_pairs, unmatched_source,
            alpha=1.0,
        )

        if np.abs(delta_target).max() < 1e-10:
            continue

        # Denormalize to raw target space
        delta_raw = delta_target * ds_std_t

        # Build full-sequence delta (query only, zero context)
        delta_query = torch.zeros(n_query, len(delta_raw))
        delta_query[r] = torch.tensor(delta_raw, dtype=torch.float32)
        delta_ctx = torch.zeros(n_ctx_target, len(delta_raw))
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query_t,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )

        transferred_preds_r = result["ablated_preds"]
        transferred_loss_r = compute_per_row_loss(
            y_query_t[r:r+1], transferred_preds_r[r:r+1], task,
        )[0]

        # Accept only if it improves
        if transferred_loss_r < weak_loss[r]:
            preds_transferred[r] = transferred_preds_r[r]
            gap_closed[r] = min(1.0, (weak_loss[r] - transferred_loss_r) / orig_gap)
            # Count matched pairs that fired
            n_fire = sum(1 for si, ti in matched_pairs
                         if h_source[r, si] > 0 and h_target[r, ti] > 0)
            n_matched_firing[r] = n_fire

        if (r + 1) % 20 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            gc_so_far = gap_closed[strong_wins[:r+1]]
            mean_gc = gc_so_far[gc_so_far > 0].mean() if (gc_so_far > 0).any() else 0
            logger.info(f"    row {r+1}/{n_query}: gap_closed={mean_gc:.3f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    valid_gc = gap_closed[strong_wins]

    return {
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": strong_preds,
        "preds_weak": weak_preds,
        "preds_transferred": preds_transferred,
        "n_query": n_query,
        "n_strong_wins": int(n_strong_wins),
        "n_matched_firing": n_matched_firing,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "y_query": y_query_t.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
    }


def run_dataset(
    source_model: str,
    target_model: str,
    dataset: str,
    sae_source: torch.nn.Module,
    sae_target: torch.nn.Module,
    bridge: Dict,
    splits: dict,
    norm_stats_source: dict,
    norm_stats_target: dict,
    device: str,
    max_steps: int,
) -> dict:
    """Run virtual concept transfer for one dataset.

    Uses full-sequence delta injection via intervene() — both context and
    query get the virtual concept, so the target model sees it consistently
    across the entire attention window.
    """
    # Load context + query for target model
    X_train_t, y_train_t, X_query_t, y_query_t, row_indices, task = load_dataset_context(
        target_model, dataset, splits,
    )
    n_query = len(X_query_t)

    if y_train_t.dtype == np.int32:
        y_train_t = y_train_t.astype(np.int64)

    # Capture target embeddings + baseline predictions via full forward pass
    extraction_layer_t = get_extraction_layer_taskaware(target_model, dataset=dataset)
    t0 = time.time()
    emb_target, weak_preds = capture_embeddings(
        target_model, X_train_t, y_train_t, X_query_t,
        extraction_layer_t, device, task,
    )
    weak_preds = np.asarray(weak_preds)
    weak_loss = compute_per_row_loss(y_query_t, weak_preds, task)

    # Get strong model predictions
    imp_path = IMPORTANCE_DIR / source_model / f"{dataset}.npz"
    if imp_path.exists():
        imp = np.load(imp_path, allow_pickle=True)
        strong_preds = imp["baseline_preds"]
    else:
        # Capture source predictions
        X_train_s, y_train_s, X_query_s, _, _, _ = load_dataset_context(
            source_model, dataset, splits,
        )
        if y_train_s.dtype == np.int32:
            y_train_s = y_train_s.astype(np.int64)
        extraction_layer_s = get_extraction_layer_taskaware(source_model, dataset=dataset)
        _, strong_preds = capture_embeddings(
            source_model, X_train_s, y_train_s, X_query_s,
            extraction_layer_s, device, task,
        )
        strong_preds = np.asarray(strong_preds)

    strong_loss = compute_per_row_loss(y_query_t, strong_preds, task)

    logger.info(f"  Captured in {time.time() - t0:.1f}s")
    logger.info(f"  Strong ({source_model}) mean loss: {strong_loss.mean():.4f}")
    logger.info(f"  Weak ({target_model}) mean loss: {weak_loss.mean():.4f}")

    # Filter to rows where strong outperforms weak
    strong_wins = strong_loss < weak_loss
    n_strong_wins = strong_wins.sum()
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {"n_strong_wins": 0, "n_query": n_query}

    # Encode source embeddings → unmatched activations (for ALL positions)
    ds_mean_s, ds_std_s = norm_stats_source[dataset]
    data_mean_s = torch.tensor(ds_mean_s, dtype=torch.float32, device=device)
    data_std_s = torch.tensor(ds_std_s, dtype=torch.float32, device=device)

    unmatched_indices = bridge["unmatched_indices"]
    n_unmatched = len(unmatched_indices)
    virtual_atoms = bridge["virtual_atoms"]

    # Encode source SAE on source test embeddings
    per_ds_s = load_test_embeddings(source_model)
    emb_s = per_ds_s[dataset]
    with torch.no_grad():
        emb_s_t = torch.tensor(emb_s, dtype=torch.float32, device=device)
        h_source = sae_source.encode(emb_s_t)
    acts_query = h_source[:, unmatched_indices].cpu().numpy()

    # For context activations, we need the source SAE encoding of context embeddings.
    # The context embeddings come from the source model's forward pass.
    # Use the target's emb_target (which includes context + query from target model).
    # But we need source activations for the context — approximate with zeros
    # (context rows don't have source SAE activations since we only have test embeddings).
    # The old script used capture_embeddings on the source model to get source emb for all positions,
    # then encoded through source SAE.
    # For now: context delta = mean of query deltas (approximation)
    n_ctx_target = emb_target.shape[0] - n_query

    # Phase 1: Per-feature importance via full intervene()
    logger.info(f"  Phase 1: per-feature importance ({n_unmatched} virtual concepts)...")
    individual_preds = []
    t1 = time.time()

    for feat_local in range(n_unmatched):
        mask = np.zeros(n_unmatched, dtype=bool)
        mask[feat_local] = True

        # Query delta for this feature
        delta_query = _make_virtual_delta(acts_query, virtual_atoms, feature_mask=mask)
        # Context: use zero delta (we don't have source activations for context)
        delta_ctx = torch.zeros(n_ctx_target, virtual_atoms.shape[1])
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query_t,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )
        individual_preds.append(result["ablated_preds"])

        if (feat_local + 1) % 10 == 0:
            logger.info(f"    {feat_local + 1}/{n_unmatched} features")

    logger.info(f"  Phase 1 done in {time.time() - t1:.1f}s")

    # Importance: negative = transfer helped (logloss decreased)
    from scripts.intervention.intervene_sae import _perrow_importance, _perrow_rankings
    importance = _perrow_importance(weak_preds, individual_preds, y_query_t)
    importance = -importance  # negate: positive = transfer helped

    # Phase 2: Per-row ranking (only features that fire in source)
    feature_indices = list(range(n_unmatched))
    rankings = _perrow_rankings(importance, feature_indices, acts_query)
    max_k = max((len(r) for r in rankings), default=0)
    logger.info(f"  Phase 2: rankings built. Max firing/row: {max_k}")

    # Phase 3: Greedy accept/reject sweep
    logger.info(f"  Phase 3: greedy sweep k=1..{min(max_k, max_steps)}...")
    t2 = time.time()

    # Track accepted masks per row
    accepted_masks = np.zeros((n_query, n_unmatched), dtype=bool)
    # Distance to strong model: |weak_pred - strong_pred| per row
    sp1 = strong_preds[:, 1] if strong_preds.ndim == 2 else strong_preds.ravel()
    bp1 = weak_preds[:, 1] if weak_preds.ndim == 2 else weak_preds.ravel()
    best_dist = np.abs(bp1 - sp1)

    sweep_preds = []
    for k in range(1, min(max_k, max_steps) + 1):
        # Tentative: accepted + k-th ranked feature per row
        tentative_masks = accepted_masks.copy()
        for row_idx in range(n_query):
            if k - 1 < len(rankings[row_idx]):
                tentative_masks[row_idx, rankings[row_idx][k - 1]] = True

        # Per-row query delta
        delta_query_t = torch.tensor(
            compute_virtual_delta_perrow(acts_query, virtual_atoms, tentative_masks),
            dtype=torch.float32,
        )
        # Context delta: union of all tentatively active features
        ctx_mask = tentative_masks.any(axis=0)
        delta_ctx = torch.zeros(n_ctx_target, virtual_atoms.shape[1])

        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query_t, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query_t,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )
        preds_k = result["ablated_preds"]

        # Accept/reject: keep if prediction moves closer to strong
        p1_k = preds_k[:, 1] if preds_k.ndim == 2 else preds_k.ravel()
        dist_k = np.abs(p1_k - sp1)
        for row_idx in range(n_query):
            if dist_k[row_idx] < best_dist[row_idx] - 1e-8:
                accepted_masks[row_idx] = tentative_masks[row_idx]
                best_dist[row_idx] = dist_k[row_idx]

        sweep_preds.append(preds_k)

        if k % 5 == 0 or k == min(max_k, max_steps):
            n_acc = accepted_masks.sum(axis=1)
            logger.info(f"    k={k}: mean accepted={n_acc.mean():.1f}")

    logger.info(f"  Phase 3 done in {time.time() - t2:.1f}s")

    # Final predictions with accepted masks
    n_accepted = accepted_masks.sum(axis=1)
    optimal_k = n_accepted.astype(np.int32)

    # Get final predictions by re-running with accepted masks
    delta_query_final = torch.tensor(
        compute_virtual_delta_perrow(acts_query, virtual_atoms, accepted_masks),
        dtype=torch.float32,
    )
    delta_ctx_final = torch.zeros(n_ctx_target, virtual_atoms.shape[1])
    full_delta_final = _build_full_delta_from_parts(delta_ctx_final, delta_query_final, n_ctx_target)

    result_final = intervene(
        model_key=target_model,
        X_context=X_train_t, y_context=y_train_t,
        X_query=X_query_t, y_query=y_query_t,
        external_delta=full_delta_final.to(device),
        device=device, task=task,
    )
    preds_transferred = result_final["ablated_preds"]

    # Compute gap closed
    transferred_loss = compute_per_row_loss(y_query_t, preds_transferred, task)
    gap_closed = np.zeros(n_query, dtype=np.float32)
    for r in range(n_query):
        if strong_wins[r]:
            orig_gap = weak_loss[r] - strong_loss[r]
            if orig_gap > 0:
                gap_closed[r] = min(1.0, (weak_loss[r] - transferred_loss[r]) / orig_gap)

    valid_mask = strong_wins
    valid_k = optimal_k[valid_mask]
    valid_gc = gap_closed[valid_mask]

    return {
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": strong_preds,
        "preds_weak": weak_preds,
        "preds_transferred": preds_transferred,
        "n_query": n_query,
        "n_strong_wins": int(n_strong_wins),
        "mean_optimal_k": float(valid_k[valid_k > 0].mean()) if (valid_k > 0).any() else 0.0,
        "median_optimal_k": float(np.median(valid_k[valid_k > 0])) if (valid_k > 0).any() else 0.0,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "concept_map_r2": bridge["concept_map_r2"],
        "n_unmatched": len(unmatched_indices),
        "n_matched_pairs": bridge["n_matched_pairs"],
        "y_query": y_query_t.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Virtual concept transfer: improve weak model with strong model's concepts")
    parser.add_argument("--source", required=True, choices=SUPPORTED_MODELS,
                        help="Source (strong) model — concepts transferred FROM here")
    parser.add_argument("--target", required=True, choices=SUPPORTED_MODELS,
                        help="Target (weak) model — concepts transferred TO here")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-match-r", type=float, default=0.3,
                        help="Min correlation for MNN matching (higher = fewer, cleaner pairs)")
    parser.add_argument("--map-type", choices=["ridge", "mlp", "local"], default="local",
                        help="Concept map: ridge (global linear), mlp (global nonlinear), local (per-row linear)")
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAEs
    sae_source, cfg_s = load_sae(args.source, device=args.device)
    sae_source.eval()
    sae_target, cfg_t = load_sae(args.target, device=args.device)
    sae_target.eval()

    norm_stats_source = load_norm_stats_matching(args.source)
    norm_stats_target = load_norm_stats_matching(args.target)

    # Build concept bridge / matched pairs
    logger.info("Building concept bridge: %s -> %s", args.source, args.target)
    corr, indices_a, indices_b = load_cross_correlations(args.source, args.target)
    unmatched = get_unmatched_features(args.source, args.target)
    logger.info("  Cross-corr: %s, unmatched source features: %d", corr.shape, len(unmatched))

    bridge = build_concept_bridge(
        sae_source, sae_target, corr, indices_a, indices_b,
        unmatched, min_match_r=args.min_match_r, ridge_alpha=args.ridge_alpha,
        map_type=args.map_type, mlp_hidden_dim=args.mlp_hidden_dim,
    )
    r2 = bridge["concept_map_r2"]
    r2_str = f"R²={r2:.4f}" if not np.isnan(r2) else "local interpolation"
    logger.info("  Bridge: %s, %d landmarks, %d virtual atoms",
                 r2_str, bridge["n_matched_pairs"],
                 len(bridge["unmatched_indices"]))

    # Find datasets with test embeddings for both models
    per_ds_s = load_test_embeddings(args.source)
    per_ds_t = load_test_embeddings(args.target)
    available = sorted(set(per_ds_s.keys()) & set(per_ds_t.keys()))

    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    pair_name = f"{args.source}_to_{args.target}"
    out_dir = OUTPUT_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nTransfer sweep: {args.source} -> {args.target}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Max steps: {args.max_steps}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        if ds not in norm_stats_source or ds not in norm_stats_target:
            logger.info(f"  SKIP (missing norm stats)")
            continue

        try:
            result = run_dataset(
                args.source, args.target, ds,
                sae_source, sae_target, bridge, splits,
                norm_stats_source, norm_stats_target,
                args.device, args.max_steps,
            )
            np.savez_compressed(str(out_path), **result)

            if result["n_strong_wins"] > 0:
                mean_k = result.get('mean_optimal_k', 0)
                logger.info(f"  -> {out_path.name}: {result['n_strong_wins']} rows, "
                            f"gap_closed={result['mean_gap_closed']:.2f}"
                            + (f", mean_k={mean_k:.1f}" if mean_k else ""))
            else:
                logger.info(f"  -> {out_path.name}: weak model wins all rows")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
