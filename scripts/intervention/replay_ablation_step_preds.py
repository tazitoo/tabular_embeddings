#!/usr/bin/env python3
"""Replay ablation_sweep's accepted steps and record per-step predictions.

For each row in an existing ablation_sweep output, we already know which
features were accepted (`selected_features[r, :optimal_k[r]]`). The replay
skips the expensive combinatorial greedy search and re-runs only the
cumulative ablations — one forward pass per accepted step, batched across
steps per row — to produce the true `|Δpred|` decay at every step.

Writes side-car files to `ablation_sweep_step_preds/<pair>/<dataset>.npz`
containing:
    step_preds      (n_rows, max_k, pred_dim)  cumulative pred after step k
    step_features   (n_rows, max_k)             feature idx added at step k
    row_indices     (n_rows,)                   alignment with sweep output

Usage:
    python -m scripts.intervention.replay_ablation_step_preds --models tabpfn mitra
"""
from __future__ import annotations

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
    batched_ablation, batched_ablation_sequential,
    MitraTail, SEQUENTIAL_MODELS,
)
from scripts.matching.utils import load_norm_stats as load_norm_stats_matching

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
OUT_DIR = PROJECT_ROOT / "output" / "ablation_sweep_step_preds"
IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"


def replay_dataset(
    pair_dir: Path,
    dataset: str,
    saes: dict,
    splits: dict,
    norm_stats: dict,
    test_embeddings: dict,
    device: str,
    out_dir: Path,
) -> bool:
    """Replay one (pair, dataset). Returns True on success."""
    npz_path = pair_dir / f"{dataset}.npz"
    d = np.load(npz_path, allow_pickle=True)
    if "selected_features" not in d or "preds_strong" not in d:
        return False

    strong = str(d["strong_model"])
    if strong not in norm_stats or dataset not in norm_stats[strong]:
        return False

    sel = d["selected_features"]                       # (n_rows, max_k)
    optimal_k = d["optimal_k"]
    strong_wins = d["strong_wins"]
    row_indices = d["row_indices"]
    baseline_preds_s = d["preds_strong"]
    pred_shape_suffix = baseline_preds_s.shape[1:]

    n_rows, max_k = sel.shape
    step_preds = np.full((n_rows, max_k) + pred_shape_suffix,
                         np.nan, dtype=np.float32)

    # Build strong-model tail + SAE activations (same prep as ablation_sweep)
    X_train_s, y_train_s, X_query_s, _, _, task_s = load_dataset_context(
        strong, dataset, splits)
    if y_train_s.dtype == np.int32:
        y_train_s = y_train_s.astype(np.int64)
    layer_s = get_extraction_layer_taskaware(strong, dataset=dataset)
    cat_indices = None
    if strong in ("hyperfast", "tabpfn"):
        from data.preprocessing import load_preprocessed, CACHE_DIR
        try:
            pre = load_preprocessed(strong, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass
    target_name = splits.get(dataset, {}).get("target", "target")
    tail_s = build_tail(strong, X_train_s, y_train_s, X_query_s, layer_s,
                        task_s, device, cat_indices=cat_indices,
                        target_name=target_name)

    emb_s = test_embeddings[strong][dataset]
    with torch.no_grad():
        emb_t = torch.tensor(emb_s, dtype=torch.float32, device=device)
        activations_s = saes[strong].encode(emb_t)

    ds_mean, ds_std = norm_stats[strong][dataset]
    data_std_t = torch.tensor(ds_std, dtype=torch.float32, device=device)

    use_sequential = isinstance(tail_s, SEQUENTIAL_MODELS)
    use_mitra = isinstance(tail_s, MitraTail)

    t0 = time.time()
    replayed = 0
    for r in range(n_rows):
        k = int(optimal_k[r])
        if not strong_wins[r] or k == 0:
            continue

        feats = sel[r, :k]
        # Build cumulative-ablation hidden states for each step 1..k
        with torch.no_grad():
            h_row = activations_s[r]
            recon_full = saes[strong].decode(h_row.unsqueeze(0))
            h_batch = h_row.unsqueeze(0).expand(k, -1).clone()
            for step in range(k):
                for fi in feats[:step + 1]:
                    if fi < 0:
                        continue
                    h_batch[step, int(fi)] = 0.0
            recon_batch = saes[strong].decode(h_batch)
            deltas = (recon_batch - recon_full) * data_std_t.unsqueeze(0)

        X_row = X_query_s[r:r + 1]
        if use_mitra:
            cand_preds = batched_ablation(tail_s, X_row, deltas)
        elif use_sequential:
            cand_preds = batched_ablation_sequential(tail_s, X_row, deltas,
                                                     query_idx=r)
        else:
            cand_preds = batched_ablation(tail_s, X_row, deltas)

        # cand_preds shape (k, *pred_shape_suffix)
        step_preds[r, :k] = cand_preds
        replayed += 1

    pair_name = pair_dir.name
    out_sub = out_dir / pair_name
    out_sub.mkdir(parents=True, exist_ok=True)
    out_path = out_sub / f"{dataset}.npz"
    np.savez(
        out_path,
        step_preds=step_preds.astype(np.float32),
        step_features=sel.astype(np.int32),
        optimal_k=optimal_k.astype(np.int32),
        row_indices=row_indices.astype(np.int32),
        strong_wins=strong_wins,
        strong_model=strong,
        weak_model=str(d["weak_model"]),
    )
    logger.info(f"  {pair_name}/{dataset}: replayed {replayed} rows in "
                f"{time.time() - t0:.1f}s → {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Replay ablation sweep to record per-step predictions")
    parser.add_argument("--models", nargs=2, required=True, metavar="MODEL",
                        help="Two models to compare (order doesn't matter)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets to replay (default: all)")
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Skip datasets whose side-car file already exists")
    args = parser.parse_args()

    a, b = sorted(args.models)
    pair_name = f"{a}_vs_{b}"
    pair_dir = args.ablation_dir / pair_name
    if not pair_dir.exists():
        logger.error(f"No such pair dir: {pair_dir}")
        return

    splits = json.loads(SPLITS_PATH.read_text())

    saes = {}
    norm_stats = {}
    test_embeddings = {}
    for m in (a, b):
        sae, _ = load_sae(m, device=args.device)
        sae.eval()
        saes[m] = sae
        norm_stats[m] = load_norm_stats_matching(m)
        test_embeddings[m] = load_test_embeddings(m)

    out_pair_dir = args.output_dir / pair_name
    ds_files = sorted(pair_dir.glob("*.npz"))
    if args.datasets:
        wanted = set(args.datasets)
        ds_files = [p for p in ds_files if p.stem in wanted]

    done = 0
    skipped = 0
    for npz in ds_files:
        side_car = out_pair_dir / npz.name
        if args.resume and side_car.exists():
            skipped += 1
            continue
        try:
            ok = replay_dataset(
                pair_dir, npz.stem, saes, splits, norm_stats,
                test_embeddings, args.device, args.output_dir,
            )
            if ok:
                done += 1
        except Exception as e:
            logger.error(f"  {pair_name}/{npz.stem}: FAILED — {e}")

    logger.info(f"Done: {done} replayed, {skipped} skipped")


if __name__ == "__main__":
    main()
