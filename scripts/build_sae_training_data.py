#!/usr/bin/env python3
"""
Build SAE training data from layerwise extractions.

Reads from extract_all_layers.py output and config/optimal_extraction_layers.json
to produce a single pooled file per model at the optimal layer.

Output structure:
    output/sae_training/{model}_layer{N}_sae_training.npz

Each file contains:
    embeddings: (n_total, hidden_dim) pooled across train datasets
    optimal_layer: int
    layer_name: string (e.g. "layer_17")
    source_datasets: list of dataset names included
    samples_per_dataset: array of (dataset_name, count) pairs
    split: "train" or "all"

Usage:
    # Build training data for TabPFN at optimal layer
    python scripts/build_sae_training_data.py --model tabpfn

    # Build for all models
    python scripts/build_sae_training_data.py --model all

    # Build including all datasets (not just train split)
    python scripts/build_sae_training_data.py --model tabpfn --split all

    # Custom max samples per dataset
    python scripts/build_sae_training_data.py --model tabpfn --max-per-dataset 300
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_optimal_layers, get_optimal_layer
from scripts.extract_layer_embeddings import sort_layer_names


LAYERWISE_DIR = PROJECT_ROOT / "output" / "embeddings" / "tabarena_layerwise"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training"


def get_available_datasets(model: str) -> List[str]:
    """Discover datasets with layerwise extractions for a model."""
    model_dir = LAYERWISE_DIR / model
    if not model_dir.exists():
        return []
    return sorted(
        f.stem.replace("tabarena_", "")
        for f in model_dir.glob("tabarena_*.npz")
    )


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic 70/30 train/test split matching sae_tabarena_sweep.py logic."""
    train, test = [], []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:
            train.append(ds)
        else:
            test.append(ds)
    return train, test


def load_layer_from_npz(
    model: str,
    dataset: str,
    layer_index: int,
) -> np.ndarray:
    """Load a specific layer's embeddings from a layerwise .npz file.

    Args:
        model: Model name
        dataset: Dataset name
        layer_index: Layer index to extract

    Returns:
        (n_samples, hidden_dim) array
    """
    npz_path = LAYERWISE_DIR / model / f"tabarena_{dataset}.npz"
    data = np.load(npz_path, allow_pickle=True)

    layer_names = list(data["layer_names"])
    sorted_names = sort_layer_names(layer_names)

    if layer_index < 0 or layer_index >= len(sorted_names):
        raise ValueError(
            f"Layer index {layer_index} out of range for {model}/{dataset}. "
            f"Available: {len(sorted_names)} layers"
        )

    layer_key = sorted_names[layer_index]
    return data[layer_key].astype(np.float32)


def build_training_data(
    model: str,
    split: str = "train",
    max_per_dataset: int = 500,
) -> Dict:
    """Build pooled SAE training data for a model.

    Args:
        model: Model name (e.g. 'tabpfn')
        split: "train" for train-only, "all" for all datasets
        max_per_dataset: Cap samples per dataset

    Returns:
        Dict with embeddings, metadata, and file path
    """
    optimal_layer = get_optimal_layer(model)
    config = load_optimal_layers()[model]

    available = get_available_datasets(model)
    if not available:
        raise ValueError(
            f"No layerwise extractions found for {model} at {LAYERWISE_DIR / model}"
        )

    if split == "train":
        datasets, _ = get_train_test_split(available)
    else:
        datasets = available

    print(f"  Available datasets: {len(available)}")
    print(f"  Using ({split}): {len(datasets)}")
    print(f"  Optimal layer: {optimal_layer}")

    all_embeddings = []
    samples_per_dataset = {}

    for ds in datasets:
        try:
            emb = load_layer_from_npz(model, ds, optimal_layer)
        except Exception as e:
            print(f"    Warning: {ds} - {e}")
            continue

        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]

        all_embeddings.append(emb)
        samples_per_dataset[ds] = len(emb)

    if not all_embeddings:
        raise ValueError(f"No embeddings loaded for {model}")

    pooled = np.concatenate(all_embeddings, axis=0)
    layer_name = f"layer_{optimal_layer}"

    # Build output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split}" if split != "train" else ""
    output_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_sae_training{suffix}.npz"

    # Save
    np.savez_compressed(
        str(output_path),
        embeddings=pooled,
        optimal_layer=np.array(optimal_layer),
        layer_name=np.array(layer_name),
        source_datasets=np.array(list(samples_per_dataset.keys())),
        samples_per_dataset=np.array(
            [(k, v) for k, v in samples_per_dataset.items()],
            dtype=[("dataset", "U100"), ("count", "i4")],
        ),
        split=np.array(split),
        config=np.array(json.dumps(config)),
    )

    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Datasets included: {len(samples_per_dataset)}")
    print(f"  Saved: {output_path}")

    return {
        "model": model,
        "optimal_layer": optimal_layer,
        "layer_name": layer_name,
        "shape": pooled.shape,
        "n_datasets": len(samples_per_dataset),
        "split": split,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SAE training data from layerwise extractions"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all' for all available models")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "all"],
                        help="Dataset split to use (default: train)")
    parser.add_argument("--max-per-dataset", type=int, default=500,
                        help="Max samples per dataset (default: 500)")
    args = parser.parse_args()

    config = load_optimal_layers()

    if args.model == "all":
        models = [m for m in config.keys() if get_available_datasets(m)]
    else:
        models = [args.model]

    if not models:
        print("No models with layerwise extractions found.")
        print(f"Run extract_all_layers.py first, output to {LAYERWISE_DIR}")
        return

    results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Building SAE training data: {model}")
        print("=" * 60)

        try:
            result = build_training_data(
                model,
                split=args.split,
                max_per_dataset=args.max_per_dataset,
            )
            results.append(result)

            # Also build "all" variant if we're doing train
            if args.split == "train":
                print(f"\n  Building 'all' variant...")
                result_all = build_training_data(
                    model,
                    split="all",
                    max_per_dataset=args.max_per_dataset,
                )
                results.append(result_all)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'Layer':>6} {'Shape':>20} {'Datasets':>10} {'Split':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['model']:<12} {r['optimal_layer']:>6} "
            f"{str(r['shape']):>20} {r['n_datasets']:>10} {r['split']:>8}"
        )


if __name__ == "__main__":
    main()
