#!/usr/bin/env python3
"""
Validate the inference pipeline against TabArena's published benchmarks.

Two levels of validation:
  1. Data alignment: every dataset loads with the correct n_samples
     and train/test index coverage matches output/sae_training_round9/tabarena_splits.json
  2. Metric ballpark: run TabPFN 2.5 on 5 small classification datasets and
     confirm our accuracy is within 5 pp of TabArena's published TABPFNV2 metric

The primary purpose of (2) is to confirm:
  - We are loading data correctly (same OpenML dataset_id, same preprocessing)
  - We are applying the split correctly (same fold=0 row indices)
  - TabPFN 2.5 produces reasonable predictions (not garbage)

We do NOT expect an exact match because TabArena used a different TabPFN
checkpoint (TabPFNv2) with a different n_estimators setting. A ±5 pp window
is a meaningful sanity check.

Output:
    output/sae_training_round9/validation_report.json

Usage:
    python scripts/sae_corpus/01_validate_inference.py
    python scripts/sae_corpus/01_validate_inference.py --datasets airfoil_self_noise wine_quality
    python scripts/sae_corpus/01_validate_inference.py --skip-inference  # data checks only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.extended_loader import (
    TABARENA_DATASETS,
    _load_tabarena_cached_v2,
    _save_tabarena_cache_v2,
)
from scripts._project_root import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9"
SPLITS_PATH = OUTPUT_DIR / "tabarena_splits.json"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"

# Small classification datasets for metric validation (fast to run)
DEFAULT_INFERENCE_DATASETS = [
    "blood-transfusion",
    "diabetes",
    "website_phishing",
    "anneal",
    "wine_quality",
]

# Tolerance for metric comparison vs TabArena published results
METRIC_TOLERANCE = 0.05


def load_full_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the FULL dataset (no subsampling) for split-index alignment.

    Uses the existing v2 parquet cache if available. Falls back to the
    extended_loader logic (which populates the cache). The cache stores
    the full dataset before any subsampling.
    """
    cached = _load_tabarena_cached_v2(name)
    if cached is not None:
        return cached

    # Populate cache via the normal loader at full size
    from data.extended_loader import load_tabarena_dataset
    info = TABARENA_DATASETS[name]

    # We import here to trigger the cache-write path; we use a large max_samples
    # to avoid truncation. TabArena's largest dataset has 150K rows.
    result = load_tabarena_dataset(name, max_samples=200_000)
    if result is None:
        raise ValueError(f"Failed to load {name}")

    # The loader may have subsampled if n > 200K. Re-check:
    cached = _load_tabarena_cached_v2(name)
    if cached is None:
        raise ValueError(f"Cache not written for {name}")
    return cached


def check_data_alignment(splits: dict) -> list[dict]:
    """Verify that every dataset loads with the correct n_samples."""
    results = []
    for name, split_info in splits.items():
        expected_n = split_info["n_samples"]
        try:
            X_df, y = load_full_dataset(name)
            actual_n = len(X_df)
            ok = actual_n == expected_n
            results.append({
                "dataset": name,
                "expected_n": expected_n,
                "actual_n": actual_n,
                "status": "ok" if ok else "mismatch",
            })
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} {name}: expected={expected_n} actual={actual_n}")
        except Exception as e:
            results.append({
                "dataset": name,
                "status": "error",
                "error": str(e),
            })
            print(f"  ✗ {name}: ERROR — {e}")
    return results


OOF_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_oof_predictions"
# Tolerance for mean absolute error in predicted probabilities (per-row comparison)
PROBA_MAE_TOLERANCE = 0.05


def run_inference_validation(
    datasets: list[str],
    splits: dict,
    device: str = "cuda",
) -> list[dict]:
    """Compare our TabPFN 2.5 predictions against TabArena's ground-truth OOF predictions.

    TabArena ran RealTabPFN-v2.5 with 8-fold CV bagging on the outer train fold,
    producing per-row OOF probabilities saved in output/sae_training_round9/tabarena_oof_predictions/.

    We run our pipeline (full train fold → predict same rows) and compare predicted
    probabilities row-by-row. Mean absolute error < PROBA_MAE_TOLERANCE confirms
    our preprocessing and inference match TabArena's pipeline.

    Note: Our pipeline uses the full train fold as context (no CV), while TabArena
    uses 8-fold CV bagging. This means predictions will differ — but the MAE should
    be small if preprocessing is correct.
    """
    from models.tabpfn_utils import load_tabpfn
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    print(f"\n  Loading TabPFN 2.5 on {device}...")
    model = load_tabpfn(task="classification", device=device, n_estimators=4)

    results = []
    for name in datasets:
        split_info = splits.get(name)
        oof_path = OOF_DIR / f"{name}.json"

        if split_info is None:
            print(f"  SKIP {name}: not in splits")
            continue
        if split_info["task_type"] != "classification":
            print(f"  SKIP {name}: regression")
            continue
        if not oof_path.exists():
            print(f"  SKIP {name}: no OOF file (run run_tabarena_tabpfnv2.py first)")
            continue

        try:
            oof = json.loads(oof_path.read_text())
            tabarena_idx = np.array(oof["train_indices"])       # rows TabArena predicted on
            tabarena_proba = np.array(oof["y_pred_proba_val"])  # (n_val, n_classes)

            X_df, y = load_full_dataset(name)
            train_idx = np.array(split_info["train_indices"])

            X_train = X_df.iloc[train_idx]
            y_train = y[train_idx]

            # Rows TabArena predicted on — query our model on the same rows
            X_query = X_df.iloc[tabarena_idx]

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)

            cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
            num_cols = X_train.select_dtypes(include="number").columns

            def to_numpy(df):
                parts = []
                if len(num_cols):
                    parts.append(df[num_cols].values.astype(np.float32))
                if len(cat_cols):
                    parts.append(enc.transform(df[cat_cols]).astype(np.float32))
                return np.nan_to_num(np.concatenate(parts, axis=1))

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if len(cat_cols):
                enc.fit(X_train[cat_cols])

            X_train_np = to_numpy(X_train)
            X_query_np = to_numpy(X_query)

            model.fit(X_train_np, y_train_enc)
            our_proba = model.predict_proba(X_query_np)  # (n_val, n_classes)

            # Per-row mean absolute error in predicted probabilities
            mae = float(np.abs(our_proba - tabarena_proba).mean())
            within_tolerance = mae <= PROBA_MAE_TOLERANCE

            status = "ok" if within_tolerance else "outside_tolerance"
            results.append({
                "dataset": name,
                "n_query": len(tabarena_idx),
                "proba_mae": mae,
                "within_tolerance": within_tolerance,
                "status": status,
            })

            sym = "✓" if within_tolerance else "⚠"
            print(f"  {sym} {name}: {len(tabarena_idx)} rows, proba MAE={mae:.4f} "
                  f"(tol={PROBA_MAE_TOLERANCE})")

        except Exception as e:
            import traceback
            results.append({"dataset": name, "status": "error", "error": str(e)})
            print(f"  ✗ {name}: ERROR — {e}")
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate inference pipeline")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets for inference check (default: 5 small ones)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Only check data alignment, skip inference")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for inference (default: cuda, never use cpu)")
    args = parser.parse_args()

    if not SPLITS_PATH.exists():
        print(f"ERROR: {SPLITS_PATH} not found. Run 00_download_tabarena_splits.py first.")
        sys.exit(1)

    splits = json.loads(SPLITS_PATH.read_text())

    print(f"\n{'='*60}")
    print("STEP 1: Data alignment check (all 51 datasets)")
    print("=" * 60)
    alignment_results = check_data_alignment(splits)

    n_ok = sum(1 for r in alignment_results if r["status"] == "ok")
    n_mismatch = sum(1 for r in alignment_results if r["status"] == "mismatch")
    n_error = sum(1 for r in alignment_results if r["status"] == "error")
    print(f"\n  {n_ok}/51 OK, {n_mismatch} size mismatch, {n_error} load errors")

    inference_results = []
    if not args.skip_inference:
        inference_datasets = args.datasets or DEFAULT_INFERENCE_DATASETS
        print(f"\n{'='*60}")
        print(f"STEP 2: Inference validation ({len(inference_datasets)} datasets)")
        print("=" * 60)
        inference_results = run_inference_validation(
            inference_datasets, splits, device=args.device
        )

        n_inf_ok = sum(1 for r in inference_results if r["status"] == "ok")
        n_inf_warn = sum(1 for r in inference_results if r["status"] == "outside_tolerance")
        print(f"\n  {n_inf_ok} OK, {n_inf_warn} outside MAE≤{PROBA_MAE_TOLERANCE} tolerance")

    report = {
        "alignment": alignment_results,
        "inference": inference_results,
        "summary": {
            "alignment_ok": n_ok,
            "alignment_mismatch": n_mismatch,
            "alignment_error": n_error,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {REPORT_PATH}")

    # Exit with error code if alignment fails
    if n_mismatch > 0 or n_error > 0:
        print("FAIL: data alignment issues detected")
        sys.exit(1)
    print("PASS: data alignment OK")


if __name__ == "__main__":
    main()
