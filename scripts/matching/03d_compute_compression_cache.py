#!/usr/bin/env python3
"""
Compute per-row LOO compression novelty for TabArena datasets.

For each row, measures how many compressed bytes it contributes to the
dataset: compress the full dataset, then compress without this row.
The difference = this row's compression contribution.

High contribution = the row is novel/incompressible relative to the rest.
Low contribution = the row is redundant/typical, already "predicted" by
other rows in the dataset.

This captures inter-row redundancy that marginal surprise misses.

Uses the SAE corpus subsample (typically 500 rows per dataset) rather
than the full dataset, matching the population the concepts were learned from.

Usage:
    python -m scripts.matching.03d_compute_compression_cache
    python -m scripts.matching.03d_compute_compression_cache --datasets credit-g diabetes
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT
from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset

OUTPUT_DIR = PROJECT_ROOT / "output" / "compression_cache"

# Match the SAE corpus subsample size
SAE_CORPUS_SAMPLES = 500


def compute_loo_compression(X_df: pd.DataFrame) -> np.ndarray:
    """Compute LOO compression contribution per row.

    For each row i, contribution[i] = len(gzip(full)) - len(gzip(without_i)).
    Positive = row adds to compressed size (novel).
    Near-zero = row is redundant.

    Args:
        X_df: Feature DataFrame.

    Returns:
        (n_rows,) array of compression contribution in bytes.
    """
    n = len(X_df)
    full_bytes = X_df.to_csv(index=False).encode("utf-8")
    full_compressed = len(gzip.compress(full_bytes))

    contributions = np.zeros(n, dtype=np.float32)
    for i in range(n):
        without_i = X_df.drop(index=i)
        loo_bytes = without_i.to_csv(index=False).encode("utf-8")
        loo_compressed = len(gzip.compress(loo_bytes))
        contributions[i] = full_compressed - loo_compressed

    return contributions


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-row LOO compression cache for TabArena datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Specific datasets (default: all TabArena)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=SAE_CORPUS_SAMPLES,
        help=f"Max samples per dataset (default: {SAE_CORPUS_SAMPLES}, matching SAE corpus)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets or sorted(TABARENA_DATASETS.keys())
    print(f"Computing LOO compression for {len(dataset_names)} datasets")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max samples: {args.max_samples}")
    print()

    summary = {}
    failed = []

    for i, ds_name in enumerate(dataset_names):
        info = TABARENA_DATASETS.get(ds_name)
        if info is None:
            failed.append(ds_name)
            continue

        out_path = OUTPUT_DIR / f"{ds_name}.npz"
        if out_path.exists():
            print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: cached, skipping")
            d = np.load(out_path, allow_pickle=True)
            contribs = d["compression_contribution"]
            summary[ds_name] = {
                "mean": float(np.mean(contribs)),
                "std": float(np.std(contribs)),
                "p10": float(np.percentile(contribs, 10)),
                "p90": float(np.percentile(contribs, 90)),
                "n_rows": len(contribs),
            }
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
            contribs = compute_loo_compression(X)

            stats = {
                "mean": float(np.mean(contribs)),
                "std": float(np.std(contribs)),
                "min": float(np.min(contribs)),
                "max": float(np.max(contribs)),
                "p10": float(np.percentile(contribs, 10)),
                "p25": float(np.percentile(contribs, 25)),
                "p50": float(np.percentile(contribs, 50)),
                "p75": float(np.percentile(contribs, 75)),
                "p90": float(np.percentile(contribs, 90)),
                "n_rows": len(contribs),
                "n_features": X.shape[1],
            }

            np.savez_compressed(
                out_path,
                compression_contribution=contribs,
                row_indices=np.arange(len(contribs), dtype=np.int32),
            )

            summary[ds_name] = stats
            elapsed = time.time() - t0
            print(
                f"{len(contribs)} rows, "
                f"contribution=[{stats['p10']:.0f}, {stats['p90']:.0f}] bytes "
                f"in {elapsed:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ds_name)

    with open(OUTPUT_DIR / "compression_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(summary)} datasets, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
