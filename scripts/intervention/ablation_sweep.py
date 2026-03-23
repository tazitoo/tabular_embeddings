#!/usr/bin/env python3
"""Cumulative ablation sweep: remove features in importance order, measure degradation.

For each model and dataset:
  1. Load per-row importance from perrow_importance output
  2. Build tail model (fit once)
  3. For each row: rank features by importance, cumulatively ablate top-1..top-K
  4. Measure prediction loss at each ablation step → degradation curve

Output:
    output/ablation_sweep/{model}/{dataset}.npz

Usage:
    python -m scripts.intervention.ablation_sweep --model tabpfn --device cuda
    python -m scripts.intervention.ablation_sweep --model tabicl --datasets website_phishing
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    load_sae, get_extraction_layer_taskaware, build_tail,
    load_dataset_context, load_test_embeddings,
    compute_per_row_loss, compute_feature_reconstructions,
    batched_ablation, batched_ablation_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast", "carte", "tabula8b"]


def compute_cumulative_deltas(
    sae: torch.nn.Module,
    h_row: torch.Tensor,
    ranked_features: list,
    data_std: torch.Tensor,
) -> torch.Tensor:
    """Compute deltas for cumulative ablation: step k zeros features 0..k.

    Returns:
        deltas: (K, emb_dim) where deltas[k] is the delta from zeroing
                the top-(k+1) features simultaneously.
    """
    K = len(ranked_features)
    with torch.no_grad():
        recon_full = sae.decode(h_row.unsqueeze(0))  # (1, emb_dim)

        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
        for k in range(K):
            # Zero features 0..k (cumulative)
            for j in range(k + 1):
                h_batch[k, ranked_features[j]] = 0.0

        recon_ablated = sae.decode(h_batch)
        delta_norm = recon_ablated - recon_full
        delta_raw = delta_norm * data_std.unsqueeze(0)

    return delta_raw


def compute_cumulative_reconstructions(
    sae: torch.nn.Module,
    h_row: torch.Tensor,
    ranked_features: list,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
) -> torch.Tensor:
    """For Mitra: full reconstructions with cumulative ablation."""
    K = len(ranked_features)
    with torch.no_grad():
        h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
        for k in range(K):
            for j in range(k + 1):
                h_batch[k, ranked_features[j]] = 0.0
        recon_norm = sae.decode(h_batch)
        recon_raw = recon_norm * data_std.unsqueeze(0) + data_mean.unsqueeze(0)
    return recon_raw


def run_dataset(
    model_key: str,
    dataset: str,
    sae: torch.nn.Module,
    extraction_layer: int,
    splits: dict,
    norm_stats: dict,
    device: str,
    max_K: int,
    max_steps: int,
) -> dict:
    """Run cumulative ablation sweep for one dataset."""
    # Load importance results
    imp_path = IMPORTANCE_DIR / model_key / f"{dataset}.npz"
    if not imp_path.exists():
        raise FileNotFoundError(f"No importance results: {imp_path}")

    imp = np.load(imp_path, allow_pickle=True)
    row_feature_drops = imp["row_feature_drops"]  # (n_query, n_alive)
    feature_indices = imp["feature_indices"]        # (n_alive,)
    baseline_preds_saved = imp["baseline_preds"]
    y_query_saved = imp["y_query"]
    n_query = len(y_query_saved)

    # Load context + aligned query rows
    X_train, y_train, X_query, y_query, row_indices, task = load_dataset_context(
        model_key, dataset, splits,
    )
    assert len(X_query) == n_query, f"Query count mismatch: {len(X_query)} vs {n_query}"

    # Mitra needs int64 labels
    if y_train.dtype == np.int32:
        y_train = y_train.astype(np.int64)

    # Load test embeddings and SAE encode
    per_ds = load_test_embeddings(model_key)
    emb = per_ds[dataset]
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        activations = sae.encode(emb_t)

    firing_mask = (activations > 0).cpu().numpy()

    # Norm stats
    ds_mean, ds_std = norm_stats[dataset]
    data_mean_t = torch.tensor(ds_mean, dtype=torch.float32, device=device)
    data_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    logger.info(f"  Context: {X_train.shape}, Query: {n_query}, Task: {task}")

    # Build tail ONCE
    t0 = time.time()
    tail = build_tail(model_key, X_train, y_train, X_query,
                      extraction_layer, task, device)
    baseline_preds = tail.baseline_preds
    baseline_loss = compute_per_row_loss(y_query, baseline_preds, task)
    logger.info(f"  Tail built in {time.time() - t0:.1f}s, "
                f"mean baseline loss: {baseline_loss.mean():.4f}")

    use_sequential = isinstance(tail, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail, MitraTail)

    # Per-row cumulative ablation
    # For each row, rank firing features by importance (most helpful first)
    # Then ablate top-1, top-2, ..., top-min(K, max_steps)
    n_steps_per_row = []
    # Store per-row degradation curves: (n_query, max_steps)
    curve = np.full((n_query, max_steps), np.nan, dtype=np.float32)

    t0 = time.time()
    for r in range(n_query):
        # Get this row's firing features sorted by importance (descending)
        row_drops = row_feature_drops[r]
        row_firing = [i for i, fi in enumerate(feature_indices)
                      if firing_mask[r, fi]]
        if not row_firing:
            n_steps_per_row.append(0)
            continue

        # Sort by importance (most helpful = highest positive drop)
        firing_importance = [(i, row_drops[i]) for i in row_firing]
        firing_importance.sort(key=lambda x: -x[1])

        # Take top max_steps features
        ranked = [feature_indices[i] for i, _ in firing_importance[:max_steps]]
        K = len(ranked)
        n_steps_per_row.append(K)

        h_row = activations[r]
        X_row = X_query[r:r + 1]

        if use_mitra:
            recons = compute_cumulative_reconstructions(
                sae, h_row, ranked, data_mean_t, data_std_t,
            )
            preds = batched_ablation(tail, X_row, recons, max_K=max_K)
        elif use_sequential:
            deltas = compute_cumulative_deltas(sae, h_row, ranked, data_std_t)
            preds = batched_ablation_sequential(tail, X_row, deltas, query_idx=r)
        else:
            deltas = compute_cumulative_deltas(sae, h_row, ranked, data_std_t)
            preds = batched_ablation(tail, X_row, deltas, max_K=max_K)

        y_tiled = np.full(len(preds), y_query[r])
        step_losses = compute_per_row_loss(y_tiled, preds, task)
        curve[r, :K] = step_losses - baseline_loss[r]

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            # Mean curve at step K (all features ablated)
            full_ablation = curve[r, K - 1] if K > 0 else 0
            logger.info(f"    row {r+1}/{n_query}: {K} steps, "
                        f"full_ablation={full_ablation:+.4f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # Compute mean degradation curve (averaging across rows)
    mean_curve = np.nanmean(curve, axis=0)
    # Count how many rows contribute at each step
    n_valid = np.sum(~np.isnan(curve), axis=0)

    return {
        "curve": curve,                          # (n_query, max_steps)
        "mean_curve": mean_curve,                # (max_steps,)
        "n_valid": n_valid,                      # (max_steps,)
        "baseline_loss": baseline_loss,          # (n_query,)
        "n_steps_per_row": np.array(n_steps_per_row),
        "feature_indices": feature_indices,
        "y_query": y_query.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
        "extraction_layer": np.array(extraction_layer),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative ablation sweep: degrade predictions by removing concepts")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-K", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=64,
                        help="Max ablation steps per row (default: top 64 features)")
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    sae, config = load_sae(args.model, device=args.device)
    sae.eval()
    norm_stats = load_norm_stats_matching(args.model)

    # Datasets with both test embeddings AND importance results
    per_ds = load_test_embeddings(args.model)
    available = sorted(per_ds.keys())
    has_importance = [d for d in available
                      if (IMPORTANCE_DIR / args.model / f"{d}.npz").exists()]

    if args.datasets:
        datasets = [d for d in has_importance if d in args.datasets]
    else:
        datasets = has_importance

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ablation sweep: {args.model}")
    logger.info(f"  SAE: {config.input_dim} -> {config.hidden_dim}")
    logger.info(f"  Datasets: {len(datasets)} (of {len(has_importance)} with importance)")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Device: {args.device}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        if ds not in norm_stats:
            logger.info(f"  SKIP (no norm stats)")
            continue

        try:
            extraction_layer = get_extraction_layer_taskaware(args.model, dataset=ds)
            result = run_dataset(
                args.model, ds, sae, extraction_layer,
                splits, norm_stats, args.device, args.max_K, args.max_steps,
            )
            np.savez_compressed(str(out_path), **result)

            mc = result["mean_curve"]
            valid = ~np.isnan(mc)
            if valid.any():
                last_valid = np.where(valid)[0][-1]
                logger.info(f"  -> {out_path.name}: {last_valid+1} steps, "
                            f"final degradation={mc[last_valid]:+.4f}")
            else:
                logger.info(f"  -> {out_path.name}: no valid steps")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
