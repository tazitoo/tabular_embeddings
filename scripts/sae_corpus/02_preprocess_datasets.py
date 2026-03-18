#!/usr/bin/env python3
"""Preprocess all TabArena datasets for each model and cache to disk.

Runs once. Skips datasets that are already cached (use --force to overwrite).
Output: output/sae_training_round9/preprocessed/{model}/{dataset}.npz

Usage:
    python scripts/sae_corpus/02_preprocess_datasets.py
    python scripts/sae_corpus/02_preprocess_datasets.py --models tabpfn tabdpt
    python scripts/sae_corpus/02_preprocess_datasets.py --datasets blood-transfusion-service-center diabetes
    python scripts/sae_corpus/02_preprocess_datasets.py --force
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.extended_loader import _load_tabarena_cached_v2
from data.preprocessing import (
    AUTOGLUON_MODELS,
    CACHE_DIR,
    is_cached,
    preprocess_for_model,
    save_preprocessed,
)
from scripts._project_root import PROJECT_ROOT

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
SUPPORTED_MODELS = list(AUTOGLUON_MODELS) + ["hyperfast"]


def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache TabArena datasets")
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS,
                        help=f"Models to preprocess (supported: {SUPPORTED_MODELS})")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets (default: all 51)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cache files")
    args = parser.parse_args()

    # Validate model names before touching any data
    unsupported = [m for m in args.models if m.lower() not in SUPPORTED_MODELS]
    if unsupported:
        print(f"ERROR: unsupported models: {unsupported}. Supported: {SUPPORTED_MODELS}")
        sys.exit(1)

    splits = json.loads(SPLITS_PATH.read_text())
    dataset_names = args.datasets or list(splits.keys())

    done = skipped = errors = 0

    for model_name in args.models:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        for ds_name in dataset_names:
            if ds_name not in splits:
                print(f"  SKIP {ds_name}: not in splits")
                continue

            if not args.force and is_cached(model_name, ds_name, CACHE_DIR):
                print(f"  skip {ds_name}: cached")
                skipped += 1
                continue

            split_info = splits[ds_name]
            task_type = split_info["task_type"]

            cached = _load_tabarena_cached_v2(ds_name)
            if cached is None:
                print(f"  ERROR {ds_name}: no v2 cache — run 01_validate_inference.py first")
                errors += 1
                continue

            X_df, y = cached
            train_idx = np.array(split_info["train_indices"])
            test_idx = np.array(split_info["test_indices"])

            X_train = X_df.iloc[train_idx].reset_index(drop=True)
            X_test = X_df.iloc[test_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_test = y[test_idx]

            try:
                data = preprocess_for_model(
                    model_name, ds_name, X_train, y_train, X_test, y_test, task_type
                )
                save_preprocessed(data, CACHE_DIR)
                nan_count = int(np.isnan(data.X_train).sum())
                cat_preview = data.cat_indices[:3]
                suffix = "..." if len(data.cat_indices) > 3 else ""
                print(f"  ✓ {ds_name}: shape={data.X_train.shape} "
                      f"nan={nan_count} cats={cat_preview}{suffix}")
                done += 1
            except Exception as e:
                import traceback
                print(f"  ERROR {ds_name}: {e}")
                traceback.print_exc()
                errors += 1

    print(f"\n{'='*60}")
    print(f"Done: {done}  Skipped: {skipped}  Errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
