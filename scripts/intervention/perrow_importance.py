#!/usr/bin/env python3
"""Per-row, per-feature importance via batched SAE ablation.

For each model and dataset:
  1. Load preprocessed data, build tail model (fit once)
  2. Load test embeddings, encode through SAE
  3. For each query row: recapture with K copies, ablate each firing
     feature, measure prediction loss change

Output:
    output/perrow_importance/{model}/{dataset}.npz

Usage:
    python -m scripts.intervention.perrow_importance --model tabpfn --device cuda
    python -m scripts.intervention.perrow_importance --model mitra --datasets adult
    python -m scripts.intervention.perrow_importance --model tabpfn --resume
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
    MitraTail, SEQUENTIAL_MODELS, DATAFRAME_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "perrow_importance"

SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "hyperfast", "carte", "tabula8b"]


def run_dataset(
    model_key: str,
    dataset: str,
    sae: torch.nn.Module,
    extraction_layer: int,
    splits: dict,
    norm_stats: dict,
    device: str,
    max_K: int,
) -> dict:
    """Run per-row importance for one dataset."""
    # Load context + aligned query rows
    X_train, y_train, X_query, y_query, row_indices, task = load_dataset_context(
        model_key, dataset, splits,
    )
    n_query = len(X_query)

    # Load test embeddings for this dataset (already normalized)
    per_ds = load_test_embeddings(model_key)
    emb = per_ds[dataset]
    assert len(emb) == n_query, f"Embedding count {len(emb)} != query count {n_query}"

    # SAE encode
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        activations = sae.encode(emb_t)

    firing_mask = (activations > 0).cpu().numpy()
    alive_features = np.where(firing_mask.any(axis=0))[0].tolist()
    n_alive = len(alive_features)

    # Norm stats for denormalization
    ds_mean, ds_std = norm_stats[dataset]
    data_mean_t = torch.tensor(ds_mean, dtype=torch.float32, device=device)
    data_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    logger.info(f"  Context: {X_train.shape}, Query: {n_query}, "
                f"Alive: {n_alive}, Task: {task}")

    # Mitra cross_entropy expects int64 labels
    if y_train.dtype == np.int32:
        y_train = y_train.astype(np.int64)

    # Build tail ONCE
    cat_indices = None
    if model_key == "hyperfast":
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed("hyperfast", dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    t0 = time.time()
    tail = build_tail(model_key, X_train, y_train, X_query,
                      extraction_layer, task, device, cat_indices=cat_indices)
    baseline_preds = tail.baseline_preds
    baseline_loss = compute_per_row_loss(y_query, baseline_preds, task)
    logger.info(f"  Tail built in {time.time() - t0:.1f}s, "
                f"mean baseline loss: {baseline_loss.mean():.4f}")

    # Choose ablation strategy
    use_sequential = isinstance(tail, SEQUENTIAL_MODELS)

    # Per-row importance loop
    row_feature_drops = np.zeros((n_query, n_alive), dtype=np.float32)
    feature_n_firing = np.zeros(n_alive, dtype=np.int32)

    t0 = time.time()
    for r in range(n_query):
        # Which alive features fire on this row?
        row_firing = [i for i, fi in enumerate(alive_features)
                      if firing_mask[r, fi]]
        if not row_firing:
            continue

        for i in row_firing:
            feature_n_firing[i] += 1

        firing_feat_indices = [alive_features[i] for i in row_firing]
        h_row = activations[r]

        X_row = X_query[r:r + 1]
        if isinstance(tail, MitraTail):
            # Mitra: use deltas (not full reconstructions) to avoid 36% SAE
            # reconstruction error. batched_ablation handles Mitra via
            # hook-based delta injection.
            deltas = compute_feature_deltas(sae, h_row, firing_feat_indices, data_std_t)
            preds = batched_ablation(tail, X_row, deltas, max_K=max_K)
        elif use_sequential:
            deltas = compute_feature_deltas(sae, h_row, firing_feat_indices, data_std_t)
            preds = batched_ablation_sequential(tail, X_row, deltas, query_idx=r)
        else:
            deltas = compute_feature_deltas(sae, h_row, firing_feat_indices, data_std_t)
            preds = batched_ablation(tail, X_row, deltas, max_K=max_K)

        y_tiled = np.full(len(preds), y_query[r])
        ablated_loss = compute_per_row_loss(y_tiled, preds, task)
        for j, col_idx in enumerate(row_firing):
            row_feature_drops[r, col_idx] = ablated_loss[j] - baseline_loss[r]

        if (r + 1) % 50 == 0 or r == n_query - 1:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_query - r - 1) / rate if rate > 0 else 0
            row_drops = row_feature_drops[r]
            nz = row_drops[row_drops != 0]
            n_pos = (row_drops > 0).sum()
            mean_drop = nz.mean() if len(nz) else 0.0
            logger.info(f"    row {r+1}/{n_query}: {len(row_firing)} firing, "
                        f"{n_pos} helpful, mean={mean_drop:+.4f} "
                        f"({rate:.1f} rows/s, ETA {eta:.0f}s)")

    logger.info(f"  Done in {time.time() - t0:.1f}s")

    return {
        "row_feature_drops": row_feature_drops,
        "feature_indices": np.array(alive_features, dtype=np.int32),
        "feature_n_firing": feature_n_firing,
        "baseline_preds": baseline_preds,
        "y_query": y_query.astype(np.float32),
        "row_indices": row_indices.astype(np.int32),
        "extraction_layer": np.array(extraction_layer),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-row feature importance via batched SAE ablation")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-K", type=int, default=512)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    # Load SAE once
    sae, config = load_sae(args.model, device=args.device)
    sae.eval()
    norm_stats = load_norm_stats_matching(args.model)

    # Datasets with test embeddings
    per_ds = load_test_embeddings(args.model)
    datasets = sorted(per_ds.keys())
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Per-row importance: {args.model}")
    logger.info(f"  SAE: {config.input_dim} -> {config.hidden_dim}")
    logger.info(f"  Datasets: {len(datasets)}")
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
                splits, norm_stats, args.device, args.max_K,
            )
            np.savez_compressed(str(out_path), **result)

            rd = result["row_feature_drops"]
            nz = rd[rd != 0]
            logger.info(f"  -> {out_path.name}: {len(result['feature_indices'])} alive, "
                        f"{len(nz)} nonzero, mean={nz.mean():.4f}" if len(nz) else
                        f"  -> {out_path.name}: no nonzero importance")

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
