#!/usr/bin/env python3
"""
Extract all block-level layer embeddings for TabArena datasets.

Saves every layer's embeddings to disk so downstream scripts (build_sae_training_data.py)
can pick the optimal layer without re-running extraction. Each dataset produces one .npz
file containing all layers.

Output structure:
    output/embeddings/tabarena_layerwise_round5/{model}/tabarena_{dataset}.npz

Each .npz contains:
    layer_0, layer_1, ..., layer_N: (n_query, hidden_dim) arrays
    layer_names: sorted list of layer name strings
    n_samples: number of query samples
    embedding_dim: hidden dimension
    context_size: ICL context size used
    query_size: query set size used

Usage:
    # Extract all layers for TabPFN on all TabArena datasets
    python scripts/extract_all_layers.py --model tabpfn --device cuda

    # Smoke test on a single dataset
    python scripts/extract_all_layers.py --model tabpfn --device cuda --datasets adult

    # Extract specific models
    python scripts/extract_all_layers.py --model tabicl --device cuda
    python scripts/extract_all_layers.py --model tabula8b --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "4_results"))

from data.extended_loader import TABARENA_DATASETS
from scripts.extract_layer_embeddings import (
    EXTRACT_FN,
    load_context_query,
    get_dataset_task,
    sort_layer_names,
)


def get_tabarena_dataset_names() -> list[str]:
    """Get all TabArena dataset names from the catalog."""
    return sorted(TABARENA_DATASETS.keys())


def extract_all_layers_for_dataset(
    model: str,
    dataset_name: str,
    device: str = "cuda",
    context_size: int = 600,
    query_size: int = 500,
) -> dict[str, np.ndarray]:
    """Extract embeddings at all block-level layers for one dataset.

    Returns dict mapping layer names to (n_query, hidden_dim) arrays.
    """
    import inspect

    extract_fn = EXTRACT_FN[model]

    X_context, y_context, X_query = load_context_query(
        dataset_name, context_size=context_size, query_size=query_size,
    )

    task = get_dataset_task(dataset_name)
    sig = inspect.signature(extract_fn)
    kwargs = dict(device=device)
    if "task" in sig.parameters:
        kwargs["task"] = task

    layer_embeddings = extract_fn(X_context, y_context, X_query, **kwargs)
    return layer_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract all block-level layer embeddings across TabArena"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(EXTRACT_FN.keys()),
                        help="Model to extract from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--context-size", type=int, default=600,
                        help="Number of ICL context examples (default: 600)")
    parser.add_argument("--query-size", type=int, default=500,
                        help="Number of query rows to embed (default: 500)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific dataset names (default: all TabArena)")
    args = parser.parse_args()

    all_datasets = args.datasets or get_tabarena_dataset_names()
    output_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena_layerwise_round5" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out regression datasets for classification-only models
    import inspect
    extract_fn = EXTRACT_FN[args.model]
    cls_only = "task" not in inspect.signature(extract_fn).parameters
    if cls_only:
        datasets = [ds for ds in all_datasets if get_dataset_task(ds) == "classification"]
        n_skipped_task = len(all_datasets) - len(datasets)
    else:
        datasets = all_datasets
        n_skipped_task = 0

    print(f"Extracting ALL layers for {args.model}")
    print(f"  Output: {output_dir}")
    print(f"  Context: {args.context_size}, Query: {args.query_size}")
    print(f"  Datasets: {len(datasets)}" + (f" ({n_skipped_task} regression skipped)" if n_skipped_task else ""))
    print(f"  Device: {args.device}")
    print()

    success, skipped = 0, 0
    failures = []

    for i, ds in enumerate(datasets):
        output_path = output_dir / f"tabarena_{ds}.npz"
        if output_path.exists():
            print(f"[{i+1}/{len(datasets)}] {ds}: exists, skipping")
            skipped += 1
            continue

        t0 = time.time()
        try:
            layer_embeddings = extract_all_layers_for_dataset(
                args.model, ds,
                device=args.device,
                context_size=args.context_size,
                query_size=args.query_size,
            )

            # Sort layer names for consistent ordering
            sorted_names = sort_layer_names(list(layer_embeddings.keys()))

            # Build save dict with each layer as a separate array
            save_dict = {}
            for name in sorted_names:
                save_dict[name] = layer_embeddings[name]

            # Get shape from first layer
            first_emb = layer_embeddings[sorted_names[0]]

            save_dict["layer_names"] = np.array(sorted_names, dtype=str)
            save_dict["n_samples"] = np.array(first_emb.shape[0])
            save_dict["embedding_dim"] = np.array(first_emb.shape[1])
            save_dict["context_size"] = np.array(args.context_size)
            save_dict["query_size"] = np.array(args.query_size)

            np.savez_compressed(str(output_path), **save_dict)

            dt = time.time() - t0
            print(
                f"[{i+1}/{len(datasets)}] {ds}: "
                f"{len(sorted_names)} layers, "
                f"{first_emb.shape} per layer ({dt:.1f}s)"
            )
            success += 1

        except Exception as e:
            import traceback
            dt = time.time() - t0
            print(f"[{i+1}/{len(datasets)}] {ds}: FAILED ({dt:.1f}s)")
            traceback.print_exc()
            failures.append(ds)

    print(f"\nDone: {success} extracted, {skipped} skipped, {len(failures)} failed")
    print(f"Output: {output_dir}")

    if failures:
        print(f"\nFailed datasets: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
