#!/usr/bin/env python3
"""
Extract PyMFE dataset-level meta-features for TabArena datasets and cache to JSON.

PyMFE provides ~80-140 dataset-level meta-features covering statistical, complexity,
information-theoretic, landmarking, and model-based properties.

Usage:
    python scripts/compute_pymfe_cache.py --output output/pymfe_tabarena_cache.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset


def extract_pymfe_features(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    max_samples_complexity: int = 1000,
    cat_cols=None,
    fast: bool = False,
) -> dict:
    """
    Extract PyMFE meta-features for one dataset.

    Args:
        X: Feature matrix (n_samples, n_features). May contain mixed types
           (object dtype for categoricals) if cat_cols is provided.
        y: Target vector.
        task: 'classification' or 'regression'.
        max_samples_complexity: Subsample limit for complexity features (O(n²) cost).
        cat_cols: List of integer column indices that are categorical, or None.
        fast: If True, skip complexity/landmarking/model-based groups.

    Returns:
        {feature_name: float_value} with NaN/inf replaced by 0.0.
    """
    from pymfe.mfe import MFE

    # Gate groups by task type and speed preference
    groups = ['general', 'statistical', 'info-theory']
    if not fast and task == 'classification':
        groups += ['landmarking', 'model-based']

    all_features = {}

    for group in groups:
        try:
            mfe = MFE(groups=[group], suppress_warnings=True)
            mfe.fit(X, y, cat_cols=cat_cols)
            names, values = mfe.extract()
            for n, v in zip(names, values):
                if isinstance(v, (int, float, np.integer, np.floating)):
                    all_features[n] = float(v) if np.isfinite(v) else 0.0
                elif isinstance(v, np.ndarray):
                    val = float(np.nanmean(v))
                    all_features[n] = val if np.isfinite(val) else 0.0
        except Exception as e:
            print(f"      Warning: group '{group}' failed: {e}")
            continue

    if fast:
        return all_features

    # Complexity features separately: subsample (O(n²) graph computations)
    # and enforce a timeout since graph metrics can hang on some datasets
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("Complexity features timed out")

    try:
        X_c, y_c = X, y
        if len(X) > max_samples_complexity:
            np.random.seed(42)
            idx = np.random.choice(len(X), max_samples_complexity, replace=False)
            X_c, y_c = X[idx], y[idx]

        # 120s timeout for complexity group
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(120)
        try:
            mfe = MFE(groups=['complexity'], suppress_warnings=True)
            # Complexity features need numeric data — use cat_cols so PyMFE
            # encodes categoricals internally via transform_cat='gray'
            cat_cols_c = cat_cols  # same column indices apply to subsampled X
            mfe.fit(X_c, y_c, cat_cols=cat_cols_c)
            names, values = mfe.extract()
            for n, v in zip(names, values):
                if isinstance(v, (int, float, np.integer, np.floating)):
                    all_features[n] = float(v) if np.isfinite(v) else 0.0
                elif isinstance(v, np.ndarray):
                    val = float(np.nanmean(v))
                    all_features[n] = val if np.isfinite(val) else 0.0
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (TimeoutError, Exception) as e:
        print(f"      Warning: group 'complexity' failed: {e}")

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract PyMFE meta-features for TabArena datasets"
    )
    parser.add_argument(
        "--output", type=str, default="output/pymfe_tabarena_cache.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--datasets", nargs='+', default=None,
        help="Specific datasets to extract (default: all TabArena)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000,
        help="Max samples per dataset for PyMFE extraction",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip complexity, landmarking, model-based groups (much faster)",
    )
    args = parser.parse_args()

    dataset_names = args.datasets or list(TABARENA_DATASETS.keys())
    print(f"Extracting PyMFE features for {len(dataset_names)} datasets")

    cache = {}
    failed = []

    for i, ds_name in enumerate(dataset_names):
        info = TABARENA_DATASETS.get(ds_name)
        if info is None:
            print(f"  [{i+1}/{len(dataset_names)}] {ds_name}: not in TABARENA_DATASETS, skipping")
            failed.append(ds_name)
            continue

        print(f"  [{i+1}/{len(dataset_names)}] {ds_name} ({info['task']})...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = load_tabarena_dataset(ds_name, max_samples=args.max_samples)
            if result is None:
                print("load failed")
                failed.append(ds_name)
                continue

            X, y, _ = result

            # Identify categorical columns from raw dtypes before conversion
            cat_cols = None
            if hasattr(X, 'dtypes'):
                cat_cols = [
                    i for i, dt in enumerate(X.dtypes)
                    if dt == object or str(dt) == 'category' or str(dt) == 'str'
                ]
                if not cat_cols:
                    cat_cols = None
                X_arr = X.values
            else:
                X_arr = np.asarray(X)

            y = np.asarray(y)

            features = extract_pymfe_features(
                X_arr, y, info['task'], cat_cols=cat_cols,
                fast=args.fast,
            )
            cache[ds_name] = features
            elapsed = time.time() - t0
            print(f"{len(features)} features in {elapsed:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ds_name)

    # Report
    if cache:
        # Find common features across all extracted datasets
        all_feature_sets = [set(v.keys()) for v in cache.values()]
        common = sorted(set.intersection(*all_feature_sets))
        all_union = sorted(set.union(*all_feature_sets))
        print(f"\nExtracted: {len(cache)}/{len(dataset_names)} datasets")
        print(f"Common features: {len(common)} (union: {len(all_union)})")
    else:
        print("\nNo datasets extracted successfully!")
        common = []

    if failed:
        print(f"Failed: {failed}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
