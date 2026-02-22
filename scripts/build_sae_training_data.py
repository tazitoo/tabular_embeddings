#!/usr/bin/env python3
"""
Build SAE training data from layerwise extractions.

Reads from extract_all_layers.py output and config/optimal_extraction_layers.json
to produce pooled train/test files per model at the optimal layer.

Row-level 70/30 split: every dataset contributes rows to both train and test,
eliminating domain bias from dataset-level holdouts.

Output structure:
    output/sae_training_round5/{model}_layer{N}_sae_training.npz  (70% of rows)
    output/sae_training_round5/{model}_layer{N}_sae_test.npz      (30% of rows)

Each file contains:
    embeddings: (n_total, hidden_dim) pooled across datasets
    optimal_layer: int
    layer_name: string (e.g. "layer_17")
    source_datasets: list of dataset names included
    samples_per_dataset: array of (dataset_name, count) pairs
    split: "train" or "test"

Usage:
    # Build train+test data for TabPFN at optimal layer
    python scripts/build_sae_training_data.py --model tabpfn

    # Build for all models
    python scripts/build_sae_training_data.py --model all

    # Custom max samples per dataset
    python scripts/build_sae_training_data.py --model tabpfn --max-per-dataset 300
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_optimal_layers, get_optimal_layer
from scripts.extract_layer_embeddings import sort_layer_names


LAYERWISE_DIR = PROJECT_ROOT / "output" / "embeddings" / "tabarena_layerwise_round5"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round5"

SPLIT_SEED = 42
TRAIN_FRACTION = 0.7


def get_available_datasets(model: str) -> List[str]:
    """Discover datasets with layerwise extractions for a model."""
    model_dir = LAYERWISE_DIR / model
    if not model_dir.exists():
        return []
    return sorted(
        f.stem.replace("tabarena_", "")
        for f in model_dir.glob("tabarena_*.npz")
    )


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


def split_rows(
    emb: np.ndarray,
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split embedding rows into train/test with deterministic shuffle.

    Caps total rows at max_per_dataset first, then splits 70/30.

    Args:
        emb: (n_samples, hidden_dim) array
        max_per_dataset: Cap total samples before splitting

    Returns:
        (train_emb, test_emb) tuple
    """
    rng = np.random.RandomState(SPLIT_SEED)

    # Cap total samples
    if len(emb) > max_per_dataset:
        idx = rng.choice(len(emb), max_per_dataset, replace=False)
        emb = emb[idx]
    else:
        # Still shuffle for deterministic split
        idx = rng.permutation(len(emb))
        emb = emb[idx]

    n_train = int(len(emb) * TRAIN_FRACTION)
    return emb[:n_train], emb[n_train:]


def build_training_data(
    model: str,
    max_per_dataset: int = 500,
) -> Dict:
    """Build pooled SAE train+test data for a model.

    Every dataset contributes rows to both train and test files.
    Row-level 70/30 split eliminates domain bias.

    Args:
        model: Model name (e.g. 'tabpfn')
        max_per_dataset: Cap samples per dataset (before splitting)

    Returns:
        Dict with shapes, metadata, and file paths
    """
    optimal_layer = get_optimal_layer(model)
    config = load_optimal_layers()[model]

    datasets = get_available_datasets(model)
    if not datasets:
        raise ValueError(
            f"No layerwise extractions found for {model} at {LAYERWISE_DIR / model}"
        )

    print(f"  Available datasets: {len(datasets)}")
    print(f"  Optimal layer: {optimal_layer}")
    print(f"  Split: {TRAIN_FRACTION:.0%} train / {1-TRAIN_FRACTION:.0%} test (row-level)")

    train_embeddings = []
    test_embeddings = []
    train_samples = {}
    test_samples = {}

    for ds in datasets:
        try:
            emb = load_layer_from_npz(model, ds, optimal_layer)
        except Exception as e:
            print(f"    Warning: {ds} - {e}")
            continue

        train_emb, test_emb = split_rows(emb, max_per_dataset)

        train_embeddings.append(train_emb)
        test_embeddings.append(test_emb)
        train_samples[ds] = len(train_emb)
        test_samples[ds] = len(test_emb)

    if not train_embeddings:
        raise ValueError(f"No embeddings loaded for {model}")

    train_pooled = np.concatenate(train_embeddings, axis=0)
    test_pooled = np.concatenate(test_embeddings, axis=0)
    layer_name = f"layer_{optimal_layer}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_sae_training.npz"
    test_path = OUTPUT_DIR / f"{model}_layer{optimal_layer}_sae_test.npz"

    def _save(path, embeddings, samples_dict, split_name):
        np.savez_compressed(
            str(path),
            embeddings=embeddings,
            optimal_layer=np.array(optimal_layer),
            layer_name=np.array(layer_name),
            source_datasets=np.array(list(samples_dict.keys())),
            samples_per_dataset=np.array(
                [(k, v) for k, v in samples_dict.items()],
                dtype=[("dataset", "U100"), ("count", "i4")],
            ),
            split=np.array(split_name),
            config=np.array(json.dumps(config)),
        )

    _save(train_path, train_pooled, train_samples, "train")
    _save(test_path, test_pooled, test_samples, "test")

    print(f"  Train: {train_pooled.shape} from {len(train_samples)} datasets → {train_path.name}")
    print(f"  Test:  {test_pooled.shape} from {len(test_samples)} datasets → {test_path.name}")

    return {
        "model": model,
        "optimal_layer": optimal_layer,
        "layer_name": layer_name,
        "train_shape": train_pooled.shape,
        "test_shape": test_pooled.shape,
        "n_datasets": len(train_samples),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SAE training data from layerwise extractions"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all' for all available models")
    parser.add_argument("--max-per-dataset", type=int, default=500,
                        help="Max samples per dataset before split (default: 500)")
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
                max_per_dataset=args.max_per_dataset,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'Layer':>6} {'Train':>20} {'Test':>20} {'Datasets':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model']:<12} {r['optimal_layer']:>6} "
            f"{str(r['train_shape']):>20} {str(r['test_shape']):>20} "
            f"{r['n_datasets']:>10}"
        )


if __name__ == "__main__":
    main()
