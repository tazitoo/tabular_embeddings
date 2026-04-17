#!/usr/bin/env python3
"""Cache baseline predictions for SAE test rows.

For each (model, dataset) in the SAE round10 test NPZ, fit the model on the
training split and save its predictions on the test rows — indexed the same
way as `load_test_embeddings(model)[dataset]`. These feed the contrastive
labeling pipeline so agents can see whether a feature fires on
confident-correct, confident-wrong, or uncertain predictions.

Output:
    output/baseline_predictions/{model}/{dataset}.npz
      pred_probs       (n, n_classes) float32 | (n,) float32 for regression
      pred_class       (n,) int64             | (n,) float32 for regression
      y_true           (n,) float32
      row_indices      (n,) int64
      model_key        str
      task_type        str ("classification" | "regression")
      extraction_layer int

Usage:
    python -m scripts.intervention.cache_baseline_predictions --model mitra
    python -m scripts.intervention.cache_baseline_predictions --model mitra \\
        --datasets SDSS17
    python -m scripts.intervention.cache_baseline_predictions --model tabicl_v2 \\
        --resume
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    build_tail,
    get_extraction_layer_taskaware,
    load_dataset_context,
    load_test_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "output" / "baseline_predictions"

# hyperfast excluded: representation mismatch (see hyperfast_architecture_findings.md)
# tabula8b excluded: out of main sweep (see ablation_model_scope.md)
SUPPORTED_MODELS = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "carte"]


def _normalize_preds(preds: np.ndarray, task: str, y_true: np.ndarray):
    """Return (pred_probs, pred_class) with consistent shapes.

    Classification:
        pred_probs: (n, n_classes) float32
        pred_class: (n,) int64
    Regression:
        pred_probs: (n,) float32
        pred_class: (n,) float32 (= pred_probs)
    """
    preds = np.asarray(preds)
    if task == "regression":
        vals = preds.astype(np.float32).ravel()
        return vals, vals.copy()

    # Classification: ensure 2D probability matrix
    if preds.ndim == 1:
        # Some tails return P(y=1) for binary — widen to 2 columns
        p1 = preds.astype(np.float32)
        pred_probs = np.column_stack([1.0 - p1, p1]).astype(np.float32)
    else:
        pred_probs = preds.astype(np.float32)

    pred_class = pred_probs.argmax(axis=1).astype(np.int64)
    return pred_probs, pred_class


def run_dataset(
    model_key: str,
    dataset: str,
    splits: dict,
    device: str,
) -> dict:
    """Build the tail, capture baseline predictions, return cache payload."""
    X_train, y_train, X_query, y_query, row_indices, task = load_dataset_context(
        model_key, dataset, splits,
    )

    # Mitra cross_entropy expects int64 labels (mirrors perrow_importance.py)
    if y_train.dtype == np.int32:
        y_train = y_train.astype(np.int64)

    cat_indices = None
    if model_key in ("hyperfast", "tabpfn"):
        from data.preprocessing import CACHE_DIR, load_preprocessed
        try:
            pre = load_preprocessed(model_key, dataset, CACHE_DIR)
            cat_indices = pre.cat_indices if pre.cat_indices else None
        except Exception:
            pass

    target_name = splits.get(dataset, {}).get("target", "target")
    extraction_layer = get_extraction_layer_taskaware(model_key, dataset=dataset)

    t0 = time.time()
    tail = build_tail(
        model_key, X_train, y_train, X_query,
        extraction_layer, task, device,
        cat_indices=cat_indices, target_name=target_name,
    )
    logger.info(
        f"  Context: {X_train.shape}, Query: {len(X_query)}, "
        f"Task: {task}, Layer: {extraction_layer}, Built in {time.time() - t0:.1f}s"
    )

    pred_probs, pred_class = _normalize_preds(tail.baseline_preds, task, y_query)

    return {
        "pred_probs": pred_probs,
        "pred_class": pred_class,
        "y_true": np.asarray(y_query, dtype=np.float32),
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "model_key": np.array(model_key),
        "task_type": np.array(task),
        "extraction_layer": np.array(extraction_layer, dtype=np.int64),
    }


def _valid_existing(path: Path, expected_n: int) -> bool:
    """Resume check: file exists and has matching row count."""
    try:
        d = np.load(path, allow_pickle=True)
        return len(d["row_indices"]) == expected_n
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Cache baseline predictions on SAE test rows.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    per_ds = load_test_embeddings(args.model)
    datasets = sorted(per_ds.keys())
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]

    out_dir = (args.output_dir if args.output_dir else OUTPUT_DIR) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baseline predictions: {args.model}")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Output:   {out_dir}")
    logger.info(f"  Device:   {args.device}")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, ds in enumerate(datasets):
        out_path = out_dir / f"{ds}.npz"
        expected_n = len(per_ds[ds])

        if args.resume and out_path.exists() and _valid_existing(out_path, expected_n):
            logger.info(f"[{i+1}/{len(datasets)}] {ds}: SKIP (exists, {expected_n} rows)")
            n_skip += 1
            continue

        logger.info(f"\n[{i+1}/{len(datasets)}] {ds}")
        try:
            result = run_dataset(args.model, ds, splits, args.device)
            np.savez_compressed(str(out_path), **result)
            logger.info(f"  -> {out_path.name}: {len(result['row_indices'])} rows saved")
            n_ok += 1
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    logger.info(f"\nDone: {n_ok} ok, {n_skip} skipped, {n_fail} failed "
                f"({len(datasets)} total)")


if __name__ == "__main__":
    main()
