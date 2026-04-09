#!/usr/bin/env python3
"""
Compute per-row pointwise mutual information (PMI) for TabArena datasets.

For each row, computes mean PMI across all features:
    PMI(x_i=v, y=c) = log2(P(x_i=v, y=c) / (P(x_i=v) * P(y=c)))

High PMI = feature values strongly predict this target class.
Low/negative PMI = feature values are surprising for this target.

Saves per-dataset NPZ files with:
    row_pmi: (n_samples,) mean PMI per row (bits per feature)
    row_indices: (n_samples,) dataset row indices
    dataset_stats: dict with mean, std, p10, p25, p50, p75, p90

Usage:
    python -m scripts.matching.03b_compute_pmi_cache
    python -m scripts.matching.03b_compute_pmi_cache --datasets credit-g diabetes
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset

OUTPUT_DIR = PROJECT_ROOT / "output" / "pmi_cache"


def compute_row_pmi(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute per-row mean PMI across all features.

    Categorical features use their raw values. Numeric features are
    discretized into equal-width bins.

    Args:
        X_df: Feature DataFrame (may contain mixed types).
        y: Target array.
        n_bins: Number of bins for numeric discretization.

    Returns:
        (n_rows,) array of mean PMI in bits per feature.
    """
    n_rows, n_cols = X_df.shape

    # Discretize: categoricals as-is, numerics into bins
    X_disc = pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        dtype = X_df[col].dtype
        if dtype == object or str(dtype) in ("category", "str"):
            X_disc[col] = X_df[col].astype(str)
        else:
            try:
                X_disc[col] = pd.cut(
                    X_df[col].astype(float), bins=n_bins, duplicates="drop",
                ).astype(str)
            except Exception:
                X_disc[col] = X_df[col].astype(str)

    y_str = y.astype(str)
    py = pd.Series(y_str).value_counts(normalize=True).to_dict()

    row_pmis = np.zeros(n_rows, dtype=np.float64)

    for col in X_disc.columns:
        px = X_disc[col].value_counts(normalize=True).to_dict()
        joint = pd.crosstab(X_disc[col], y_str, normalize=True)

        for i in range(n_rows):
            v = X_disc[col].iloc[i]
            c = y_str[i]
            p_xy = (
                joint.loc[v, c]
                if v in joint.index and c in joint.columns
                else 1e-10
            )
            p_x = px.get(v, 1e-10)
            p_y = py.get(c, 1e-10)
            row_pmis[i] += np.log2(max(p_xy, 1e-10) / (p_x * p_y + 1e-10))

    row_pmis /= max(n_cols, 1)
    return row_pmis


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-row PMI cache for TabArena datasets",
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
    print(f"Computing row-level PMI for {len(dataset_names)} datasets")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Bins: {args.n_bins}, Max samples: {args.max_samples}")
    print()

    summary = {}
    failed = []

    for i, ds_name in enumerate(dataset_names):
        info = TABARENA_DATASETS.get(ds_name)
        if info is None:
            print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: not in TABARENA_DATASETS")
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
            y_arr = np.asarray(y)

            row_pmis = compute_row_pmi(X, y_arr, n_bins=args.n_bins)

            stats = {
                "mean": float(np.mean(row_pmis)),
                "std": float(np.std(row_pmis)),
                "min": float(np.min(row_pmis)),
                "max": float(np.max(row_pmis)),
                "p10": float(np.percentile(row_pmis, 10)),
                "p25": float(np.percentile(row_pmis, 25)),
                "p50": float(np.percentile(row_pmis, 50)),
                "p75": float(np.percentile(row_pmis, 75)),
                "p90": float(np.percentile(row_pmis, 90)),
                "n_rows": len(row_pmis),
                "n_features": X.shape[1],
                "task": info["task"],
            }

            out_path = OUTPUT_DIR / f"{ds_name}.npz"
            np.savez_compressed(
                out_path,
                row_pmi=row_pmis.astype(np.float32),
                row_indices=np.arange(len(row_pmis), dtype=np.int32),
            )

            summary[ds_name] = stats
            elapsed = time.time() - t0
            print(
                f"{len(row_pmis)} rows, "
                f"PMI=[{stats['p10']:.3f}, {stats['p90']:.3f}] "
                f"in {elapsed:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ds_name)

    # Write summary JSON
    summary_path = OUTPUT_DIR / "pmi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(summary)} datasets, {len(failed)} failed")
    print(f"Summary: {summary_path}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
