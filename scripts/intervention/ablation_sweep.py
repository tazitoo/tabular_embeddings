#!/usr/bin/env python3
"""Cross-model ablation sweep: how many concepts separate two models?

For a model pair (strong vs weak) on each dataset:
  1. Load per-row importance for the stronger model
  2. Filter to rows where the stronger model outperforms the weaker
  3. Cumulatively ablate the stronger model's features in importance order
  4. Search along predictions until logloss matches the weaker model's

The number of concepts removed to reach parity = the "concept gap."

Builds on find_per_row_optimal_ablation() from plot_prediction_scatter.py,
rewritten to use the intervene_lib backbone.

Output:
    output/ablation_sweep/{strong_model}_vs_{weak_model}/{dataset}.npz

Usage:
    python -m scripts.intervention.ablation_sweep --strong tabpfn --weak tabicl --device cuda
    python -m scripts.intervention.ablation_sweep --strong tabpfn --weak tabicl --datasets website_phishing
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
    compute_per_row_loss, compute_feature_deltas, compute_feature_reconstructions,
    batched_ablation, batched_ablation_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast", "carte", "tabula8b"]


def run_dataset(
    strong_model: str,
    weak_model: str,
    dataset: str,
    sae_strong: torch.nn.Module,
    sae_weak: torch.nn.Module,
    splits: dict,
    norm_stats_strong: dict,
    norm_stats_weak: dict,
    device: str,
    max_K: int,
    max_steps: int,
) -> dict:
    """Cross-model ablation for one dataset."""

    # Load importance for the strong model
    imp_path = IMPORTANCE_DIR / strong_model / f"{dataset}.npz"
    imp = np.load(imp_path, allow_pickle=True)
    row_feature_drops = imp["row_feature_drops"]
    feature_indices = imp["feature_indices"]

    # Load context + query for strong model
    X_train_s, y_train_s, X_query_s, y_query_s, row_indices, task = load_dataset_context(
        strong_model, dataset, splits,
    )
    n_query = len(X_query_s)

    if y_train_s.dtype == np.int32:
        y_train_s = y_train_s.astype(np.int64)

    # Load test embeddings for strong model
    per_ds_s = load_test_embeddings(strong_model)
    emb_s = per_ds_s[dataset]
    with torch.no_grad():
        emb_t = torch.tensor(emb_s, dtype=torch.float32, device=device)
        activations_s = sae_strong.encode(emb_t)
    firing_mask_s = (activations_s > 0).cpu().numpy()

    # Norm stats for strong model
    ds_mean_s, ds_std_s = norm_stats_strong[dataset]
    data_mean_t_s = torch.tensor(ds_mean_s, dtype=torch.float32, device=device)
    data_std_t_s = torch.tensor(ds_std_s, dtype=torch.float32, device=device)

    # Build tail for strong model → baseline predictions
    extraction_layer_s = get_extraction_layer_taskaware(strong_model, dataset=dataset)
    t0 = time.time()
    tail_s = build_tail(strong_model, X_train_s, y_train_s, X_query_s,
                        extraction_layer_s, task, device)
    baseline_preds_s = tail_s.baseline_preds
    baseline_loss_s = compute_per_row_loss(y_query_s, baseline_preds_s, task)

    # Build tail for weak model → weak model predictions
    X_train_w, y_train_w, X_query_w, y_query_w, _, _ = load_dataset_context(
        weak_model, dataset, splits,
    )
    if y_train_w.dtype == np.int32:
        y_train_w = y_train_w.astype(np.int64)

    extraction_layer_w = get_extraction_layer_taskaware(weak_model, dataset=dataset)
    tail_w = build_tail(weak_model, X_train_w, y_train_w, X_query_w,
                        extraction_layer_w, task, device)
    weak_preds = tail_w.baseline_preds
    weak_loss = compute_per_row_loss(y_query_w, weak_preds, task)

    logger.info(f"  Tails built in {time.time() - t0:.1f}s")
    logger.info(f"  Strong ({strong_model}) mean loss: {baseline_loss_s.mean():.4f}")
    logger.info(f"  Weak ({weak_model}) mean loss: {weak_loss.mean():.4f}")

    # Filter to rows where strong outperforms weak
    strong_wins = baseline_loss_s < weak_loss
    n_strong_wins = strong_wins.sum()
    logger.info(f"  Strong wins on {n_strong_wins}/{n_query} rows")

    if n_strong_wins == 0:
        return {"n_strong_wins": 0, "n_query": n_query}

    # Free weak tail
    del tail_w
    torch.cuda.empty_cache()

    use_sequential = isinstance(tail_s, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_s, MitraTail)

    # Per-row ablation search
    optimal_k = np.zeros(n_query, dtype=np.int32)
    gap_closed = np.full(n_query, np.nan, dtype=np.float32)
    # Store intervened predictions at optimal_k for scatter plots
    n_classes = baseline_preds_s.shape[1] if baseline_preds_s.ndim == 2 else 1
    preds_intervened = baseline_preds_s.copy()  # default: unmodified

    t0 = time.time()
    for r in range(n_query):
        if not strong_wins[r]:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        target_loss = weak_loss[r]
        orig_gap = target_loss - baseline_loss_s[r]
        if orig_gap <= 0:
            optimal_k[r] = 0
            gap_closed[r] = 1.0
            continue

        # Rank this row's firing features by importance
        row_drops = row_feature_drops[r]
        row_firing = [i for i, fi in enumerate(feature_indices)
                      if firing_mask_s[r, fi]]
        if not row_firing:
            continue

        firing_importance = [(i, row_drops[i]) for i in row_firing]
        firing_importance.sort(key=lambda x: -x[1])
        ranked = [int(feature_indices[i]) for i, _ in firing_importance[:max_steps]]
        K = len(ranked)

        h_row = activations_s[r]
        X_row = X_query_s[r:r + 1]

        # Compute cumulative ablations
        if use_mitra:
            # Full reconstructions for each cumulative step
            recons_list = []
            with torch.no_grad():
                h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
                for k in range(K):
                    for j in range(k + 1):
                        h_batch[k, ranked[j]] = 0.0
                recon_norm = sae_strong.decode(h_batch)
                recons = recon_norm * data_std_t_s.unsqueeze(0) + data_mean_t_s.unsqueeze(0)
            preds = batched_ablation(tail_s, X_row, recons, max_K=max_K)
        elif use_sequential:
            with torch.no_grad():
                recon_full = sae_strong.decode(h_row.unsqueeze(0))
                h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
                for k in range(K):
                    for j in range(k + 1):
                        h_batch[k, ranked[j]] = 0.0
                recon_abl = sae_strong.decode(h_batch)
                deltas = (recon_abl - recon_full) * data_std_t_s.unsqueeze(0)
            preds = batched_ablation_sequential(tail_s, X_row, deltas, query_idx=r)
        else:
            with torch.no_grad():
                recon_full = sae_strong.decode(h_row.unsqueeze(0))
                h_batch = h_row.unsqueeze(0).expand(K, -1).clone()
                for k in range(K):
                    for j in range(k + 1):
                        h_batch[k, ranked[j]] = 0.0
                recon_abl = sae_strong.decode(h_batch)
                deltas = (recon_abl - recon_full) * data_std_t_s.unsqueeze(0)
            preds = batched_ablation(tail_s, X_row, deltas, max_K=max_K)

        # Find first k where ablated loss >= weak model's loss
        y_tiled = np.full(len(preds), y_query_s[r])
        step_losses = compute_per_row_loss(y_tiled, preds, task)  # (K,) absolute

        # Delta: how much gap remains at each step
        gap_remaining = target_loss - step_losses  # positive = still better than weak
        crossed = np.where(gap_remaining <= 0)[0]  # steps where we've matched or exceeded weak

        if len(crossed) > 0:
            best_k = int(crossed[0])
            optimal_k[r] = best_k + 1
            gap_closed[r] = 1.0
        else:
            best_k = K - 1
            optimal_k[r] = K
            gap_closed[r] = 1.0 - gap_remaining.min() / orig_gap if orig_gap > 0 else 1.0

        # Save prediction at optimal_k for scatter plots
        preds_intervened[r] = preds[best_k]

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            valid = optimal_k[:r+1][strong_wins[:r+1]]
            mean_k = valid.mean() if len(valid) else 0
            logger.info(f"    row {r+1}/{n_query}: mean_optimal_k={mean_k:.1f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # Summary stats for strong-wins rows only
    valid_mask = strong_wins
    valid_k = optimal_k[valid_mask]
    valid_gc = gap_closed[valid_mask]

    return {
        "optimal_k": optimal_k,
        "gap_closed": gap_closed,
        "strong_wins": strong_wins,
        "preds_strong": baseline_preds_s,
        "preds_weak": weak_preds,
        "preds_intervened": preds_intervened,
        "baseline_loss_strong": baseline_loss_s,
        "baseline_loss_weak": weak_loss,
        "n_query": n_query,
        "n_strong_wins": int(n_strong_wins),
        "mean_optimal_k": float(valid_k.mean()) if len(valid_k) else 0.0,
        "median_optimal_k": float(np.median(valid_k)) if len(valid_k) else 0.0,
        "mean_gap_closed": float(valid_gc.mean()) if len(valid_gc) else 0.0,
        "feature_indices": feature_indices,
        "y_query": y_query_s.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model ablation: how many concepts separate two models?")
    parser.add_argument("--strong", required=True, choices=SUPPORTED_MODELS,
                        help="Stronger model (the one being ablated)")
    parser.add_argument("--weak", required=True, choices=SUPPORTED_MODELS,
                        help="Weaker model (target performance level)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-K", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAEs for both models
    sae_strong, cfg_s = load_sae(args.strong, device=args.device)
    sae_strong.eval()
    sae_weak, cfg_w = load_sae(args.weak, device=args.device)
    sae_weak.eval()

    norm_stats_strong = load_norm_stats_matching(args.strong)
    norm_stats_weak = load_norm_stats_matching(args.weak)

    # Find datasets with importance results for the strong model
    per_ds = load_test_embeddings(args.strong)
    available = sorted(per_ds.keys())
    has_importance = [d for d in available
                      if (IMPORTANCE_DIR / args.strong / f"{d}.npz").exists()]

    if args.datasets:
        datasets = [d for d in has_importance if d in args.datasets]
    else:
        datasets = has_importance

    pair_name = f"{args.strong}_vs_{args.weak}"
    out_dir = OUTPUT_DIR / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ablation sweep: {args.strong} -> {args.weak}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Max steps: {args.max_steps}")

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        if args.resume and out_path.exists():
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists)")
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")

        if ds not in norm_stats_strong or ds not in norm_stats_weak:
            logger.info(f"  SKIP (missing norm stats)")
            continue

        try:
            result = run_dataset(
                args.strong, args.weak, ds,
                sae_strong, sae_weak, splits,
                norm_stats_strong, norm_stats_weak,
                args.device, args.max_K, args.max_steps,
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
