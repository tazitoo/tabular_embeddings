#!/usr/bin/env python3
"""
Extract embeddings at a specific intermediate transformer layer across TabArena datasets.

Uses the layer extraction functions from layerwise_cka_analysis.py to hook into
model internals and save the activations at the chosen layer.

Output format matches extract_embeddings.py so the SAE sweep can consume them
directly via the {model}_layer{N} directory convention.

Usage:
    # Extract TabPFN layer 16 for all available datasets
    python scripts/extract_layer_embeddings.py --model tabpfn --layer 16 --device cuda

    # Extract for a single dataset (smoke test)
    python scripts/extract_layer_embeddings.py --model tabpfn --layer 16 \
        --device cuda --datasets adult

    # TabICL layer 5, Mitra layer 60
    python scripts/extract_layer_embeddings.py --model tabicl --layer 5 --device cuda
    python scripts/extract_layer_embeddings.py --model mitra --layer 60 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "4_results"))

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset
from layerwise_cka_analysis import (
    extract_tabpfn_all_layers,
    extract_mitra_all_layers,
    extract_tabicl_all_layers,
    extract_hyperfast_all_layers,
    extract_tabdpt_all_layers,
    extract_carte_all_layers,
    sort_layer_names,
)

EXTRACT_FN = {
    "tabpfn": extract_tabpfn_all_layers,
    "mitra": extract_mitra_all_layers,
    "tabicl": extract_tabicl_all_layers,
    "hyperfast": extract_hyperfast_all_layers,
    "tabdpt": extract_tabdpt_all_layers,
    "carte": extract_carte_all_layers,
}


def get_tabarena_dataset_names() -> list[str]:
    """Get all TabArena dataset names from the catalog."""
    return sorted(TABARENA_DATASETS.keys())


def load_context_query(
    dataset_name: str,
    max_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a TabArena dataset and split into context/query halves.

    Returns (X_context, y_context, X_query).
    """
    result = load_tabarena_dataset(dataset_name, max_samples=max_samples * 2)
    if result is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    X, y, _ = result
    split = len(X) // 2
    return X[:split], y[:split], X[split:]


def extract_single_layer(
    model: str,
    layer: int,
    dataset_name: str,
    device: str = "cuda",
    n_samples: int = 1000,
) -> np.ndarray:
    """Extract embeddings at a specific layer for one dataset.

    Returns the (n_query, hidden_dim) embedding array.
    """
    extract_fn = EXTRACT_FN[model]

    X_context, y_context, X_query = load_context_query(dataset_name, max_samples=n_samples)

    layer_embeddings = extract_fn(X_context, y_context, X_query, device=device)

    layer_key = f"layer_{layer}"
    if layer_key not in layer_embeddings:
        available = sort_layer_names(list(layer_embeddings.keys()))
        raise ValueError(
            f"Layer {layer} not found for {model}. "
            f"Available: {available}"
        )

    return layer_embeddings[layer_key]


def main():
    parser = argparse.ArgumentParser(
        description="Extract intermediate-layer embeddings across TabArena"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(EXTRACT_FN.keys()),
                        help="Model to extract from")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index to extract (e.g. 16 for TabPFN)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Max samples per dataset (split into context/query)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific dataset names (default: all TabArena)")
    args = parser.parse_args()

    datasets = args.datasets or get_tabarena_dataset_names()
    output_dir = (
        PROJECT_ROOT / "output" / "embeddings" / "tabarena"
        / f"{args.model}_layer{args.layer}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.model} layer {args.layer} embeddings")
    print(f"  Output: {output_dir}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Device: {args.device}")
    print()

    success, skipped, failed = 0, 0, 0

    for i, ds in enumerate(datasets):
        output_path = output_dir / f"tabarena_{ds}.npz"
        if output_path.exists():
            print(f"[{i+1}/{len(datasets)}] {ds}: exists, skipping")
            skipped += 1
            continue

        t0 = time.time()
        try:
            emb = extract_single_layer(
                args.model, args.layer, ds,
                device=args.device, n_samples=args.n_samples,
            )
            np.savez_compressed(
                str(output_path),
                embeddings=emb,
                extraction_point=np.array(f"layer_{args.layer}"),
                n_samples=np.array(emb.shape[0]),
                embedding_dim=np.array(emb.shape[1]),
                layer_names=np.array([f"layer_{args.layer}"], dtype=str),
            )
            dt = time.time() - t0
            print(f"[{i+1}/{len(datasets)}] {ds}: {emb.shape} ({dt:.1f}s)")
            success += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"[{i+1}/{len(datasets)}] {ds}: FAILED ({dt:.1f}s) - {e}")
            failed += 1

    print(f"\nDone: {success} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
