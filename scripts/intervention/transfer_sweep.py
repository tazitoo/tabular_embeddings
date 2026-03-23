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
from scripts.intervention.transfer_virtual_nodes import (
    build_concept_bridge,
    extract_decoder_atoms,
    load_cross_correlations,
    _encode_unmatched_activations,
    _make_virtual_delta,
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
    """Run virtual concept transfer for one dataset."""

    # Load context + query for target model (the one being improved)
    X_train_t, y_train_t, X_query_t, y_query_t, row_indices, task = load_dataset_context(
        target_model, dataset, splits,
    )
    n_query = len(X_query_t)

    if y_train_t.dtype == np.int32:
        y_train_t = y_train_t.astype(np.int64)

    # Build target tail → baseline (weak) predictions
    extraction_layer_t = get_extraction_layer_taskaware(target_model, dataset=dataset)
    t0 = time.time()
    tail_t = build_tail(target_model, X_train_t, y_train_t, X_query_t,
                        extraction_layer_t, task, device)
    weak_preds = tail_t.baseline_preds
    weak_loss = compute_per_row_loss(y_query_t, weak_preds, task)

    # Load source model's importance to get strong predictions
    # (We need the source predictions to know the target for transfer)
    imp_path = IMPORTANCE_DIR / source_model / f"{dataset}.npz"
    if imp_path.exists():
        imp = np.load(imp_path, allow_pickle=True)
        strong_preds = imp["baseline_preds"]
        strong_loss = compute_per_row_loss(y_query_t, strong_preds, task)
    else:
        # Build source tail to get predictions
        X_train_s, y_train_s, X_query_s, _, _, _ = load_dataset_context(
            source_model, dataset, splits,
        )
        if y_train_s.dtype == np.int32:
            y_train_s = y_train_s.astype(np.int64)
        extraction_layer_s = get_extraction_layer_taskaware(source_model, dataset=dataset)
        tail_s = build_tail(source_model, X_train_s, y_train_s, X_query_s,
                            extraction_layer_s, task, device)
        strong_preds = tail_s.baseline_preds
        strong_loss = compute_per_row_loss(y_query_t, strong_preds, task)
        del tail_s
        torch.cuda.empty_cache()

    logger.info(f"  Tails built in {time.time() - t0:.1f}s")
    logger.info(f"  Strong ({source_model}) mean loss: {strong_loss.mean():.4f}")
    logger.info(f"  Weak ({target_model}) mean loss: {weak_loss.mean():.4f}")

    # Filter to rows where strong outperforms weak
    strong_wins = strong_loss < weak_loss
    n_strong_wins = strong_wins.sum()
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {"n_strong_wins": 0, "n_query": n_query}

    # Encode source embeddings through source SAE → unmatched activations
    per_ds_s = load_test_embeddings(source_model)
    emb_s = per_ds_s[dataset]

    ds_mean_s, ds_std_s = norm_stats_source[dataset]
    data_mean_s = torch.tensor(ds_mean_s, dtype=torch.float32, device=device)
    data_std_s = torch.tensor(ds_std_s, dtype=torch.float32, device=device)

    unmatched_indices = bridge["unmatched_indices"]
    n_unmatched = len(unmatched_indices)

    with torch.no_grad():
        emb_t_tensor = torch.tensor(emb_s, dtype=torch.float32, device=device)
        h_source = sae_source.encode(emb_t_tensor)
        # Extract activations for unmatched features only
        acts_query = h_source[:, unmatched_indices].cpu().numpy()

    virtual_atoms = bridge["virtual_atoms"]  # (n_unmatched, d_target)

    # Norm stats for target model (to denormalize virtual deltas)
    ds_mean_t, ds_std_t = norm_stats_target[dataset]
    data_std_t = torch.tensor(ds_std_t, dtype=torch.float32, device=device)

    # Per-row importance of virtual concepts (from source importance if available)
    # For transfer, importance = which source concepts help the TARGET model most
    # We approximate using the source model's importance ranking
    if imp_path.exists():
        imp_drops = imp["row_feature_drops"]
        imp_features = list(imp["feature_indices"])
        # Map unmatched features to their importance
        feat_to_imp_idx = {int(f): i for i, f in enumerate(imp_features)}
    else:
        feat_to_imp_idx = {}

    use_sequential = isinstance(tail_t, SEQUENTIAL_MODELS)

    # Per-row greedy transfer
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    n_classes = weak_preds.shape[1] if weak_preds.ndim == 2 else 1
    preds_transferred = weak_preds.copy()

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        target_loss = strong_loss[r]
        orig_gap = weak_loss[r] - target_loss
        if orig_gap <= 0:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Rank unmatched features by source importance (most helpful first)
        row_acts = acts_query[r]
        firing = [(j, row_acts[j]) for j in range(n_unmatched) if row_acts[j] > 0]
        if not firing:
            continue

        # Sort by source importance if available, else by activation magnitude
        if feat_to_imp_idx:
            firing_ranked = []
            for j, act in firing:
                global_idx = unmatched_indices[j]
                imp_idx = feat_to_imp_idx.get(global_idx)
                imp_val = imp_drops[r, imp_idx] if imp_idx is not None else 0.0
                firing_ranked.append((j, imp_val, act))
            firing_ranked.sort(key=lambda x: -x[1])
        else:
            firing_ranked = [(j, 0.0, act) for j, act in firing]
            firing_ranked.sort(key=lambda x: -x[2])

        ranked_local = [j for j, _, _ in firing_ranked[:max_steps]]
        K = len(ranked_local)

        # Compute K virtual deltas (cumulative: step k adds concepts 0..k)
        X_row = X_query_t[r:r + 1]
        virtual_deltas = []
        with torch.no_grad():
            for k in range(K):
                mask = np.zeros(n_unmatched, dtype=bool)
                for j in range(k + 1):
                    mask[ranked_local[j]] = True
                # Virtual delta = sum of (activation * virtual_atom) for masked features
                delta = np.zeros(virtual_atoms.shape[1], dtype=np.float32)
                for j in range(n_unmatched):
                    if mask[j]:
                        delta += row_acts[j] * virtual_atoms[j]
                # Denormalize to target raw space
                delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
                delta_raw = delta_t * data_std_t
                virtual_deltas.append(delta_raw)

        deltas = torch.stack(virtual_deltas)  # (K, d_target)

        # Inject into target tail — transfer ADDS to the embedding (not replace)
        if use_sequential:
            from scripts.intervention.intervene_lib import batched_ablation_sequential
            preds = batched_ablation_sequential(tail_t, X_row, deltas, query_idx=r)
        else:
            from scripts.intervention.intervene_lib import batched_ablation
            preds = batched_ablation(tail_t, X_row, deltas)

        # Greedy search: find first k where transfer closes the gap
        y_tiled = np.full(len(preds), y_query_t[r])
        step_losses = compute_per_row_loss(y_tiled, preds, task)

        # Walk curve, accept only steps that improve (decrease loss toward target)
        current_loss = weak_loss[r]
        accepted_k = 0
        accepted_pred_idx = None
        for k in range(K):
            if step_losses[k] <= current_loss:
                current_loss = step_losses[k]
                accepted_k = k + 1
                accepted_pred_idx = k
                if current_loss <= target_loss:
                    break

        if accepted_pred_idx is not None:
            optimal_k[r] = accepted_k
            gap_closed[r] = min(1.0, (weak_loss[r] - current_loss) / orig_gap) if orig_gap > 0 else 1.0
            preds_transferred[r] = preds[accepted_pred_idx]
        else:
            optimal_k[r] = 0
            gap_closed[r] = 0.0

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            valid = optimal_k[:r+1][strong_wins[:r+1]]
            mean_k = valid[valid > 0].mean() if (valid > 0).any() else 0
            logger.info(f"    row {r+1}/{n_query}: mean_optimal_k={mean_k:.1f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

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
        "mean_gap_closed": float(valid_gc[~np.isnan(valid_gc)].mean()) if (~np.isnan(valid_gc)).any() else 0.0,
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
    parser.add_argument("--map-type", choices=["ridge", "mlp"], default="ridge",
                        help="Concept map type: ridge (linear) or mlp (nonlinear)")
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

    # Build concept bridge
    logger.info("Building concept bridge: %s -> %s", args.source, args.target)
    corr, indices_a, indices_b = load_cross_correlations(args.source, args.target)
    unmatched = get_unmatched_features(args.source, args.target)
    logger.info("  Cross-corr: %s, unmatched source features: %d", corr.shape, len(unmatched))

    bridge = build_concept_bridge(
        sae_source, sae_target, corr, indices_a, indices_b,
        unmatched, min_match_r=args.min_match_r, ridge_alpha=args.ridge_alpha,
        map_type=args.map_type, mlp_hidden_dim=args.mlp_hidden_dim,
    )
    logger.info("  Bridge: R²=%.4f, %d landmarks, %d virtual atoms",
                bridge["concept_map_r2"], bridge["n_matched_pairs"],
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
                logger.info(f"  -> {out_path.name}: {result['n_strong_wins']} rows, "
                            f"mean_k={result['mean_optimal_k']:.1f}, "
                            f"gap_closed={result['mean_gap_closed']:.2f}")
            else:
                logger.info(f"  -> {out_path.name}: weak model wins all rows")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
