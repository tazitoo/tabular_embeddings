#!/usr/bin/env python3
"""
Compute per-row surprise (self-information) for TabArena datasets.

For each row, computes mean self-information across all features:
    surprise(x_i=v) = -log2(P(x_i=v))

High surprise = this combination of feature values is rare in the dataset.
Low surprise = common/typical feature values.

This is target-UNAWARE — purely about how typical the row's values are
in the marginal distributions, regardless of class membership.

Usage:
    python -m scripts.matching.03c_compute_surprise_cache
    python -m scripts.matching.03c_compute_surprise_cache --datasets credit-g diabetes
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT
from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset

OUTPUT_DIR = PROJECT_ROOT / "output" / "surprise_cache"


def compute_row_surprise(
    X_df: pd.DataFrame,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute per-row mean self-information across all features.

    Categorical features use their raw values. Numeric features are
    discretized into equal-width bins.

    Args:
        X_df: Feature DataFrame (may contain mixed types).
        n_bins: Number of bins for numeric discretization.

    Returns:
        (n_rows,) array of mean surprise in bits per feature.
    """
    n_rows, n_cols = X_df.shape
    surprise = np.zeros(n_rows, dtype=np.float64)

    for col in X_df.columns:
        dtype = X_df[col].dtype
        if dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            try:
                vals = pd.cut(
                    X_df[col].astype(float), bins=n_bins, duplicates="drop",
                ).astype(str)
            except Exception:
                vals = X_df[col].astype(str)
        else:
            vals = X_df[col].astype(str)

        p = vals.value_counts(normalize=True).to_dict()
        for i in range(n_rows):
            prob = p.get(vals.iloc[i], 1e-10)
            surprise[i] += -np.log2(max(prob, 1e-10))

    surprise /= max(n_cols, 1)
    return surprise


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-row surprise cache for TabArena datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Specific datasets (default: all TabArena)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000,
        help="Max samples per dataset",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for numeric discretization",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets or sorted(TABARENA_DATASETS.keys())
    print(f"Computing row-level surprise for {len(dataset_names)} datasets")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    summary = {}
    failed = []

    for i, ds_name in enumerate(dataset_names):
        info = TABARENA_DATASETS.get(ds_name)
        if info is None:
            failed.append(ds_name)
            continue

        print(f"  [{i+1}/{len(dataset_names)}] {ds_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = load_tabarena_dataset(ds_name, max_samples=args.max_samples)
            if result is None:
                print("load failed")
                failed.append(ds_name)
                continue

            X, y, _ = result
            row_surprise = compute_row_surprise(X, n_bins=args.n_bins)

            stats = {
                "mean": float(np.mean(row_surprise)),
                "std": float(np.std(row_surprise)),
                "min": float(np.min(row_surprise)),
                "max": float(np.max(row_surprise)),
                "p10": float(np.percentile(row_surprise, 10)),
                "p25": float(np.percentile(row_surprise, 25)),
                "p50": float(np.percentile(row_surprise, 50)),
                "p75": float(np.percentile(row_surprise, 75)),
                "p90": float(np.percentile(row_surprise, 90)),
                "n_rows": len(row_surprise),
                "n_features": X.shape[1],
            }

            np.savez_compressed(
                OUTPUT_DIR / f"{ds_name}.npz",
                row_surprise=row_surprise.astype(np.float32),
                row_indices=np.arange(len(row_surprise), dtype=np.int32),
            )

            summary[ds_name] = stats
            elapsed = time.time() - t0
            print(
                f"{len(row_surprise)} rows, "
                f"surprise=[{stats['p10']:.3f}, {stats['p90']:.3f}] "
                f"in {elapsed:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ds_name)

    with open(OUTPUT_DIR / "surprise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(summary)} datasets, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
