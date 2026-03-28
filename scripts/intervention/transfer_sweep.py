#!/usr/bin/env python3
"""Cross-model concept transfer sweep: improve each model with the other's concepts.

For an unordered model pair on each dataset:
  1. Auto-detect strong/weak via dataset-level AUC/RMSE
  2. Build concept bridge: MNN-matched decoder atoms → virtual atoms
  3. Transfer strong model's unique concepts → weak model (forward)
  4. Transfer weak model's unique concepts → strong model (reverse)
  5. Per-row greedy accept: keep concept only if prediction improves

Output:
    output/transfer_sweep/{model_a}_vs_{model_b}/{dataset}.npz

    Symmetric naming (sorted). NPZ records which model was strong/weak,
    and contains both forward and reverse transfer results.

Usage:
    python -m scripts.intervention.transfer_sweep --models tabpfn tabicl --device cuda
    python -m scripts.intervention.transfer_sweep --models tabpfn tabicl --datasets credit-g
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
    load_sae, get_extraction_layer_taskaware,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_importance_metric,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching
from scripts.intervention.intervene_sae import intervene
from scripts.intervention.transfer_concepts import capture_embeddings
from scripts.intervention.transfer_virtual_nodes import (
    build_concept_bridge,
    extract_decoder_atoms,
    load_cross_correlations,
    compute_local_transfer_delta,
    _make_virtual_delta,
    _build_full_delta_from_parts,
    compute_virtual_delta_perrow,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "transfer_sweep"

SUPPORTED_MODELS = [
    "tabpfn", "tabicl", "tabicl_v2", "mitra",
    "tabdpt", "hyperfast", "carte", "tabula8b",
]


def get_unmatched_features(
    source_model: str, target_model: str,
    concept_labels_path: Optional[Path] = None,
) -> List[int]:
    """Get source features that are NOT matched to the target model."""
    from scripts.intervention.intervene_lib import (
        get_alive_features, MODEL_KEY_TO_LABEL_KEY, DEFAULT_CONCEPT_LABELS,
    )

    if concept_labels_path is None:
        concept_labels_path = DEFAULT_CONCEPT_LABELS

    with open(concept_labels_path) as f:
        data = json.load(f)

    src_key = MODEL_KEY_TO_LABEL_KEY.get(source_model, source_model)
    tgt_key = MODEL_KEY_TO_LABEL_KEY.get(target_model, target_model)

    matched_source = set()
    for gid, group in data.get("concept_groups", {}).items():
        members = group.get("members", [])
        models_in_group = set(m for m, _ in members)
        if src_key in models_in_group and tgt_key in models_in_group:
            for m, f in members:
                if m == src_key:
                    matched_source.add(f)

    all_source = set(get_alive_features(source_model, concept_labels_path))
    return sorted(all_source - matched_source)


# ── Single-direction transfer ────────────────────────────────────────────────


def _transfer_one_direction(
    source_model: str,
    target_model: str,
    dataset: str,
    sae_source: torch.nn.Module,
    sae_target: torch.nn.Module,
    bridge: Dict,
    splits: dict,
    norm_stats_source: dict,
    norm_stats_target: dict,
    target_preds: np.ndarray,
    source_preds: np.ndarray,
    y_query: np.ndarray,
    row_indices: np.ndarray,
    task: str,
    device: str,
    max_steps: int,
    map_type: str,
) -> dict:
    """Transfer source model's unique concepts into target model for one dataset.

    Returns dict with per-row transfer results (optimal_k, gap_closed, etc).
    """
    n_query = len(y_query)

    # Load context for target model (needed for intervene)
    X_train_t, y_train_t, X_query_t, y_query_t, _, _, _ = load_dataset_context(
        target_model, dataset, splits,
    )
    if y_train_t.dtype == np.int32:
        y_train_t = y_train_t.astype(np.int64)

    # Capture target embeddings (for sequence length)
    extraction_layer_t = get_extraction_layer_taskaware(target_model, dataset=dataset)
    emb_target, _ = capture_embeddings(
        target_model, X_train_t, y_train_t, X_query_t,
        extraction_layer_t, device, task,
    )
    n_ctx_target = emb_target.shape[0] - n_query

    # Per-row loss
    target_loss = compute_per_row_loss(y_query, target_preds, task)
    source_loss = compute_per_row_loss(y_query, source_preds, task)

    # Rows where source outperforms target (these are the rows we can improve)
    source_wins = source_loss < target_loss
    n_source_wins = int(source_wins.sum())

    if n_source_wins == 0:
        return {
            "optimal_k": np.zeros(n_query, dtype=np.int32),
            "gap_closed": np.zeros(n_query, dtype=np.float32),
            "preds_transferred": target_preds.copy(),
            "n_source_wins": 0,
        }

    # ── Local map type: per-row ridge, no global bridge ──────────────────
    if map_type == "local" and "matched_pairs_local" in bridge:
        return _transfer_local(
            source_model, target_model, dataset,
            sae_source, sae_target, bridge,
            splits, norm_stats_source, norm_stats_target,
            X_train_t, y_train_t, X_query_t, y_query,
            target_preds, source_preds, target_loss, source_loss,
            source_wins, n_ctx_target, extraction_layer_t, task, device,
        )

    # ── Virtual atoms (ridge / mlp / local-interpolation bridge) ─────────
    unmatched_indices = bridge["unmatched_indices"]
    n_unmatched = len(unmatched_indices)
    virtual_atoms = bridge["virtual_atoms"]

    # Encode source SAE on source test embeddings
    per_ds_s = load_test_embeddings(source_model)
    with torch.no_grad():
        emb_s_t = torch.tensor(per_ds_s[dataset], dtype=torch.float32, device=device)
        h_source = sae_source.encode(emb_s_t)
    acts_query = h_source[:, unmatched_indices].cpu().numpy()

    # Phase 1: Per-feature importance
    logger.info(f"    Phase 1: {n_unmatched} virtual concepts...")
    individual_preds = []
    t1 = time.time()

    for feat_local in range(n_unmatched):
        mask = np.zeros(n_unmatched, dtype=bool)
        mask[feat_local] = True

        delta_query = -_make_virtual_delta(acts_query, virtual_atoms, feature_mask=mask)
        delta_ctx = torch.zeros(n_ctx_target, virtual_atoms.shape[1])
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )
        individual_preds.append(result["ablated_preds"])

        if (feat_local + 1) % 10 == 0:
            logger.info(f"      {feat_local + 1}/{n_unmatched}")

    logger.info(f"    Phase 1 done in {time.time() - t1:.1f}s")

    # Importance: negate so positive = transfer helped
    from scripts.intervention.intervene_sae import _perrow_importance, _perrow_rankings
    importance = -_perrow_importance(target_preds, individual_preds, y_query)

    # Phase 2: Per-row ranking (only features that fire in source)
    feature_indices = list(range(n_unmatched))
    rankings = _perrow_rankings(importance, feature_indices, acts_query)
    max_k = max((len(r) for r in rankings), default=0)
    logger.info(f"    Phase 2: max firing/row={max_k}")

    # Phase 3: Greedy accept/reject sweep
    effective_steps = min(max_k, max_steps)
    logger.info(f"    Phase 3: greedy sweep k=1..{effective_steps}...")
    t2 = time.time()

    accepted_masks = np.zeros((n_query, n_unmatched), dtype=bool)
    sp1 = source_preds[:, 1] if source_preds.ndim == 2 else source_preds.ravel()
    tp1 = target_preds[:, 1] if target_preds.ndim == 2 else target_preds.ravel()
    best_dist = np.abs(tp1 - sp1)

    for k in range(1, effective_steps + 1):
        tentative_masks = accepted_masks.copy()
        for row_idx in range(n_query):
            if k - 1 < len(rankings[row_idx]):
                tentative_masks[row_idx, rankings[row_idx][k - 1]] = True

        delta_query_t = -torch.tensor(
            compute_virtual_delta_perrow(acts_query, virtual_atoms, tentative_masks),
            dtype=torch.float32,
        )
        delta_ctx = torch.zeros(n_ctx_target, virtual_atoms.shape[1])
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query_t, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )
        preds_k = result["ablated_preds"]

        p1_k = preds_k[:, 1] if preds_k.ndim == 2 else preds_k.ravel()
        dist_k = np.abs(p1_k - sp1)
        for row_idx in range(n_query):
            if dist_k[row_idx] < best_dist[row_idx] - 1e-8:
                accepted_masks[row_idx] = tentative_masks[row_idx]
                best_dist[row_idx] = dist_k[row_idx]

        if k % 5 == 0 or k == effective_steps:
            n_acc = accepted_masks.sum(axis=1)
            logger.info(f"      k={k}: mean accepted={n_acc.mean():.1f}")

    logger.info(f"    Phase 3 done in {time.time() - t2:.1f}s")

    # Final predictions with accepted masks
    optimal_k = accepted_masks.sum(axis=1).astype(np.int32)

    delta_query_final = -torch.tensor(
        compute_virtual_delta_perrow(acts_query, virtual_atoms, accepted_masks),
        dtype=torch.float32,
    )
    delta_ctx_final = torch.zeros(n_ctx_target, virtual_atoms.shape[1])
    full_delta_final = _build_full_delta_from_parts(
        delta_ctx_final, delta_query_final, n_ctx_target,
    )

    result_final = intervene(
        model_key=target_model,
        X_context=X_train_t, y_context=y_train_t,
        X_query=X_query_t, y_query=y_query,
        external_delta=full_delta_final.to(device),
        device=device, task=task,
        extraction_layer=extraction_layer_t,
    )
    preds_transferred = result_final["ablated_preds"]

    # Compute gap closed
    transferred_loss = compute_per_row_loss(y_query, preds_transferred, task)
    gap_closed = np.zeros(n_query, dtype=np.float32)
    for r in range(n_query):
        if source_wins[r]:
            orig_gap = target_loss[r] - source_loss[r]
            if orig_gap > 0:
                gap_closed[r] = min(1.0, (target_loss[r] - transferred_loss[r]) / orig_gap)

    return {
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "preds_transferred": preds_transferred,
        "n_source_wins": n_source_wins,
    }


def _transfer_local(
    source_model, target_model, dataset,
    sae_source, sae_target, bridge,
    splits, norm_stats_source, norm_stats_target,
    X_train_t, y_train_t, X_query_t, y_query,
    target_preds, source_preds, target_loss, source_loss,
    source_wins, n_ctx_target, extraction_layer_t, task, device,
):
    """Per-row local ridge transfer (no global map)."""
    n_query = len(y_query)
    matched_pairs = bridge["matched_pairs_local"]
    unmatched_source = bridge["unmatched_indices"]

    atoms_source = extract_decoder_atoms(sae_source).numpy()
    atoms_target = extract_decoder_atoms(sae_target).numpy()

    per_ds_s = load_test_embeddings(source_model)
    per_ds_t = load_test_embeddings(target_model)

    ds_mean_t, ds_std_t = norm_stats_target[dataset]

    with torch.no_grad():
        h_source = sae_source.encode(
            torch.tensor(per_ds_s[dataset], dtype=torch.float32, device=device)
        ).cpu().numpy()
        h_target = sae_target.encode(
            torch.tensor(per_ds_t[dataset], dtype=torch.float32, device=device)
        ).cpu().numpy()

    preds_transferred = target_preds.copy()
    gap_closed = np.zeros(n_query, dtype=np.float32)

    t0 = time.time()
    for r in range(n_query):
        if not source_wins[r]:
            continue

        orig_gap = target_loss[r] - source_loss[r]
        if orig_gap <= 0:
            continue

        delta_target = compute_local_transfer_delta(
            atoms_source, atoms_target,
            h_source[r], h_target[r],
            matched_pairs, unmatched_source, alpha=1.0,
        )
        if np.abs(delta_target).max() < 1e-10:
            continue

        delta_raw = delta_target * ds_std_t
        delta_query = torch.zeros(n_query, len(delta_raw))
        delta_query[r] = torch.tensor(delta_raw, dtype=torch.float32)
        delta_ctx = torch.zeros(n_ctx_target, len(delta_raw))
        full_delta = _build_full_delta_from_parts(delta_ctx, delta_query, n_ctx_target)

        result = intervene(
            model_key=target_model,
            X_context=X_train_t, y_context=y_train_t,
            X_query=X_query_t, y_query=y_query,
            external_delta=full_delta.to(device),
            device=device, task=task,
        )

        transferred_loss_r = compute_per_row_loss(
            y_query[r:r+1], result["ablated_preds"][r:r+1], task,
        )[0]

        if transferred_loss_r < target_loss[r]:
            preds_transferred[r] = result["ablated_preds"][r]
            gap_closed[r] = min(1.0, (target_loss[r] - transferred_loss_r) / orig_gap)

        if (r + 1) % 20 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed if elapsed > 0 else 0
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            gc_valid = gap_closed[source_wins[:r+1]]
            mean_gc = gc_valid[gc_valid > 0].mean() if (gc_valid > 0).any() else 0
            logger.info(f"      row {r+1}/{n_query}: gap_closed={mean_gc:.3f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    return {
        "optimal_k": (gap_closed > 0).astype(np.int32),
        "gap_closed": gap_closed,
        "preds_transferred": preds_transferred,
        "n_source_wins": int(source_wins.sum()),
    }


# ── Main: bidirectional transfer for one dataset ─────────────────────────────


def run_dataset(
    model_a: str,
    model_b: str,
    dataset: str,
    saes: dict,
    bridges: dict,
    splits: dict,
    norm_stats: dict,
    device: str,
    max_steps: int,
    map_type: str,
) -> dict:
    """Bidirectional transfer for one dataset.

    Auto-detects strong/weak, runs forward (strong→weak) and reverse
    (weak→strong) transfers, returns combined results.
    """
    # Get baseline predictions for both models
    preds = {}
    for m in (model_a, model_b):
        X_train, y_train, X_query, y_query, row_indices, task, _ = load_dataset_context(
            m, dataset, splits,
        )
        if y_train.dtype == np.int32:
            y_train = y_train.astype(np.int64)
        layer = get_extraction_layer_taskaware(m, dataset=dataset)
        _, p = capture_embeddings(m, X_train, y_train, X_query, layer, device, task)
        preds[m] = np.asarray(p)

    # y_query and row_indices are the same for both models (same split)
    n_query = len(y_query)

    # Determine strong/weak
    metric_a, metric_name = compute_importance_metric(y_query, preds[model_a], task)
    metric_b, _ = compute_importance_metric(y_query, preds[model_b], task)

    if metric_a >= metric_b:
        strong, weak = model_a, model_b
        metric_strong, metric_weak = metric_a, metric_b
    else:
        strong, weak = model_b, model_a
        metric_strong, metric_weak = metric_b, metric_a

    logger.info(f"  {strong} ({metric_name}={metric_strong:.4f}) > "
                f"{weak} ({metric_name}={metric_weak:.4f})")

    # Forward transfer: strong's unique concepts → weak model
    logger.info(f"  Forward: {strong} concepts -> {weak} model")
    bridge_key_fwd = f"{strong}_to_{weak}"
    fwd = _transfer_one_direction(
        source_model=strong, target_model=weak, dataset=dataset,
        sae_source=saes[strong], sae_target=saes[weak],
        bridge=bridges[bridge_key_fwd],
        splits=splits,
        norm_stats_source=norm_stats[strong],
        norm_stats_target=norm_stats[weak],
        target_preds=preds[weak], source_preds=preds[strong],
        y_query=y_query, row_indices=row_indices,
        task=task, device=device,
        max_steps=max_steps, map_type=map_type,
    )

    # Reverse transfer: weak's unique concepts → strong model
    logger.info(f"  Reverse: {weak} concepts -> {strong} model")
    bridge_key_rev = f"{weak}_to_{strong}"
    rev = _transfer_one_direction(
        source_model=weak, target_model=strong, dataset=dataset,
        sae_source=saes[weak], sae_target=saes[strong],
        bridge=bridges[bridge_key_rev],
        splits=splits,
        norm_stats_source=norm_stats[weak],
        norm_stats_target=norm_stats[strong],
        target_preds=preds[strong], source_preds=preds[weak],
        y_query=y_query, row_indices=row_indices,
        task=task, device=device,
        max_steps=max_steps, map_type=map_type,
    )

    # Combine results
    fwd_valid_k = fwd["optimal_k"][fwd["gap_closed"] > 0]
    rev_valid_k = rev["optimal_k"][rev["gap_closed"] > 0]

    return {
        # Identifiers
        "strong_model": strong,
        "weak_model": weak,
        "metric_strong": float(metric_strong),
        "metric_weak": float(metric_weak),
        "metric_name": metric_name,
        # Baseline predictions
        "preds_strong": preds[strong],
        "preds_weak": preds[weak],
        "y_query": y_query.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
        "n_query": n_query,
        # Forward: strong concepts → weak model (improves weak)
        "fwd_preds_transferred": fwd["preds_transferred"],
        "fwd_optimal_k": fwd["optimal_k"],
        "fwd_gap_closed": fwd["gap_closed"],
        "fwd_n_source_wins": fwd["n_source_wins"],
        "fwd_mean_optimal_k": float(fwd_valid_k.mean()) if len(fwd_valid_k) else 0.0,
        "fwd_mean_gap_closed": float(fwd["gap_closed"][fwd["gap_closed"] > 0].mean())
            if (fwd["gap_closed"] > 0).any() else 0.0,
        # Reverse: weak concepts → strong model (improves strong)
        "rev_preds_transferred": rev["preds_transferred"],
        "rev_optimal_k": rev["optimal_k"],
        "rev_gap_closed": rev["gap_closed"],
        "rev_n_source_wins": rev["n_source_wins"],
        "rev_mean_optimal_k": float(rev_valid_k.mean()) if len(rev_valid_k) else 0.0,
        "rev_mean_gap_closed": float(rev["gap_closed"][rev["gap_closed"] > 0].mean())
            if (rev["gap_closed"] > 0).any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional concept transfer between two models")
    parser.add_argument("--models", nargs=2, required=True, metavar="MODEL",
                        help="Two models to compare (order does not matter)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-match-r", type=float, default=0.3,
                        help="Min correlation for MNN matching")
    parser.add_argument("--map-type", choices=["ridge", "mlp", "local"],
                        default="local",
                        help="Concept map type for virtual atoms")
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    args = parser.parse_args()

    model_a, model_b = sorted(args.models)
    pair_name = f"{model_a}_vs_{model_b}"

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAEs and norm stats for both models
    saes = {}
    norm_stats = {}
    for m in (model_a, model_b):
        sae, _ = load_sae(m, device=args.device)
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)

    # Build concept bridges in both directions
    bridges = {}
    for source, target in [(model_a, model_b), (model_b, model_a)]:
        key = f"{source}_to_{target}"
        logger.info(f"Building concept bridge: {source} -> {target}")
        corr, indices_a, indices_b = load_cross_correlations(source, target)
        unmatched = get_unmatched_features(source, target)
        logger.info(f"  Cross-corr: {corr.shape}, unmatched: {len(unmatched)}")

        bridge = build_concept_bridge(
            saes[source], saes[target], corr, indices_a, indices_b,
            unmatched, min_match_r=args.min_match_r,
            ridge_alpha=args.ridge_alpha,
            map_type=args.map_type, mlp_hidden_dim=args.mlp_hidden_dim,
        )
        r2 = bridge["concept_map_r2"]
        r2_str = f"R²={r2:.4f}" if not np.isnan(r2) else "local interpolation"
        logger.info(f"  {r2_str}, {bridge['n_matched_pairs']} landmarks, "
                     f"{len(bridge['unmatched_indices'])} virtual atoms")
        bridges[key] = bridge

    # Find datasets available for both models
    per_ds_a = load_test_embeddings(model_a)
    per_ds_b = load_test_embeddings(model_b)
    available = sorted(set(per_ds_a.keys()) & set(per_ds_b.keys()))

    if args.datasets:
        datasets = [d for d in available if d in args.datasets]
    else:
        datasets = available

    out_dir = OUTPUT_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nTransfer sweep: {model_a} vs {model_b}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Max steps: {args.max_steps}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        if ds not in norm_stats[model_a] or ds not in norm_stats[model_b]:
            logger.info(f"  SKIP (missing norm stats)")
            continue

        try:
            result = run_dataset(
                model_a, model_b, ds, saes, bridges, splits,
                norm_stats, device=args.device,
                max_steps=args.max_steps, map_type=args.map_type,
            )
            np.savez_compressed(str(out_path), **result)

            logger.info(
                f"  -> {out_path.name}: "
                f"fwd({result['fwd_n_source_wins']} rows, "
                f"gc={result['fwd_mean_gap_closed']:.2f}) "
                f"rev({result['rev_n_source_wins']} rows, "
                f"gc={result['rev_mean_gap_closed']:.2f})"
            )

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
