#!/usr/bin/env python3
"""
Download and cache TabArena splits from OpenML.

Fetches the canonical OpenML 3-fold outer CV split indices for all 51 TabArena
datasets (OpenML suite "tabarena-v0.1") and caches them locally.

We use fold=0, repeat=0 as our canonical split. The outer test fold is a sacred
holdout — never used for SAE training, context selection, or normalization.

Output:
    output/sae_training_round9/tabarena_splits.json

    {
        "<dataset_name>": {
            "task_id": int,
            "dataset_id": int,
            "task_type": "classification" | "regression",
            "target": str,
            "n_samples": int,
            "train_indices": [int, ...],   # outer fold 0, repeat 0
            "test_indices": [int, ...],    # sacred holdout — do not use
        },
        ...
    }

Usage:
    python scripts/sae_corpus/00_download_tabarena_splits.py
    python scripts/sae_corpus/00_download_tabarena_splits.py --force  # re-download
"""
import argparse
import json
import sys
from pathlib import Path

import openml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.extended_loader import TABARENA_DATASETS
from scripts._project_root import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9"
SPLITS_PATH = OUTPUT_DIR / "tabarena_splits.json"
SUITE_NAME = "tabarena-v0.1"
FOLD = 0
REPEAT = 0


def build_dataset_id_to_name() -> dict[int, str]:
    return {v["openml_id"]: k for k, v in TABARENA_DATASETS.items()}


def download_splits(force: bool = False) -> dict:
    """Download split indices for all 51 TabArena datasets.

    Returns the splits dict (also saved to SPLITS_PATH).
    """
    if SPLITS_PATH.exists() and not force:
        print(f"Splits already cached at {SPLITS_PATH}")
        print("Use --force to re-download.")
        return json.loads(SPLITS_PATH.read_text())

    print(f"Fetching suite '{SUITE_NAME}' from OpenML...")
    suite = openml.study.get_suite(SUITE_NAME)
    task_ids = list(suite.tasks)
    print(f"  {len(task_ids)} tasks found")

    ds_id_to_name = build_dataset_id_to_name()

    splits = {}
    errors = []

    for i, tid in enumerate(task_ids):
        print(f"  [{i+1}/{len(task_ids)}] task_id={tid}...", end=" ", flush=True)

        task = openml.tasks.get_task(tid, download_splits=True)
        ds_name = ds_id_to_name.get(task.dataset_id)

        if ds_name is None:
            print(f"WARNING: dataset_id={task.dataset_id} not in TABARENA_DATASETS")
            errors.append(tid)
            continue

        train_idx, test_idx = task.get_train_test_split_indices(
            fold=FOLD, repeat=REPEAT
        )

        # Resolve task type from our catalog (more reliable than OpenML task type string)
        task_type = TABARENA_DATASETS[ds_name]["task"]

        splits[ds_name] = {
            "task_id": tid,
            "dataset_id": task.dataset_id,
            "task_type": task_type,
            "target": task.target_name,
            "n_samples": len(train_idx) + len(test_idx),
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }

        print(
            f"n={splits[ds_name]['n_samples']} "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"({ds_name})"
        )

    if errors:
        print(f"\nWARNING: {len(errors)} tasks had no matching dataset: {errors}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    print(f"\nSaved {len(splits)} splits to {SPLITS_PATH}")
    return splits


def print_summary(splits: dict) -> None:
    n_cls = sum(1 for v in splits.values() if v["task_type"] == "classification")
    n_reg = sum(1 for v in splits.values() if v["task_type"] == "regression")
    sizes = [v["n_samples"] for v in splits.values()]
    train_sizes = [len(v["train_indices"]) for v in splits.values()]

    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Datasets:       {len(splits)} ({n_cls} classification, {n_reg} regression)")
    print(f"  Fold/repeat:    {FOLD}/{REPEAT} (canonical)")
    print(f"  N_samples:      min={min(sizes)}, median={sorted(sizes)[len(sizes)//2]}, max={max(sizes)}")
    print(f"  Train fold:     min={min(train_sizes)}, median={sorted(train_sizes)[len(train_sizes)//2]}, max={max(train_sizes)}")
    print(f"  Test fold:      sacred holdout (never used for training)")


def main():
    parser = argparse.ArgumentParser(description="Download TabArena OpenML splits")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    splits = download_splits(force=args.force)
    print_summary(splits)


if __name__ == "__main__":
    main()
