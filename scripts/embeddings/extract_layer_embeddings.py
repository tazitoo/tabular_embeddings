#!/usr/bin/env python3
"""
Extract embeddings at a specific intermediate transformer layer across TabArena datasets.

Uses the layer extraction functions from layerwise_cka_analysis.py to hook into
model internals and save the activations at the chosen layer.

Output format matches extract_embeddings.py so the SAE sweep can consume them
directly via the {model}_layer{N} directory convention.

Usage:
    # Extract TabPFN layer 16 at default context size (600)
    python scripts/extract_layer_embeddings.py --model tabpfn --layer 16 --device cuda

    # Extract with custom context/query sizes
    python scripts/extract_layer_embeddings.py --model tabpfn --layer 16 \
        --context-size 200 --query-size 100 --device cuda

    # Extract for a single dataset (smoke test)
    python scripts/extract_layer_embeddings.py --model tabpfn --layer 16 \
        --device cuda --datasets adult

    # TabICL layer 5, Mitra layer 60
    python scripts/extract_layer_embeddings.py --model tabicl --layer 5 --device cuda
    python scripts/extract_layer_embeddings.py --model mitra --layer 60 --device cuda
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts._project_root import PROJECT_ROOT

from data.extended_loader import TABARENA_DATASETS, DatasetMetadata, load_tabarena_dataset
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures" / "4_results"))
from layerwise_cka_analysis import (
    extract_tabpfn_all_layers,
    extract_mitra_all_layers,
    extract_tabicl_all_layers,
    extract_tabicl_v2_all_layers,
    extract_hyperfast_all_layers,
    extract_tabdpt_all_layers,
    extract_carte_all_layers,
    extract_tabula8b_all_layers,
    sort_layer_names,
)

# Fixed random seed for reproducible splits
SPLIT_SEED = 42

EXTRACT_FN = {
    "tabpfn": extract_tabpfn_all_layers,
    "mitra": extract_mitra_all_layers,
    "tabicl": extract_tabicl_all_layers,
    "tabicl_v2": extract_tabicl_v2_all_layers,
    "hyperfast": extract_hyperfast_all_layers,
    "tabdpt": extract_tabdpt_all_layers,
    "carte": extract_carte_all_layers,
    "tabula8b": extract_tabula8b_all_layers,
}


def get_tabarena_dataset_names() -> list[str]:
    """Get all TabArena dataset names from the catalog."""
    return sorted(TABARENA_DATASETS.keys())


def load_context_query(
    dataset_name: str,
    context_size: int = 600,
    query_size: int = 500,
) -> tuple:
    """Load a TabArena dataset and split into context/query sets.

    Uses stratified sampling for classification datasets to ensure all classes
    are represented in both context and query sets, even for highly imbalanced
    datasets (e.g. QSAR-TID-11 with 0.02% minority class).

    Returns (X_context, y_context, X_query, metadata) where X_context and
    X_query are DataFrames with proper dtypes (categoricals as object dtype).
    """
    result = load_tabarena_dataset(dataset_name, max_samples=context_size + query_size)
    if result is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    X, y, meta = result
    n = len(X)
    if n < context_size + query_size:
        context_size = int(n * 0.7)
        query_size = n - context_size

    task = get_dataset_task(dataset_name)
    if task == "classification":
        # Ensure contiguous 0-indexed labels (e.g. QSAR-TID-11 has labels [4..11])
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Stratified split ensures all classes appear in both context and query
        query_frac = query_size / (context_size + query_size)
        try:
            X_ctx, X_q, y_ctx, _ = train_test_split(
                X, y, test_size=query_frac, random_state=SPLIT_SEED, stratify=y,
            )
        except ValueError:
            # Fallback for classes with too few samples to stratify
            X_ctx, X_q, y_ctx, _ = train_test_split(
                X, y, test_size=query_frac, random_state=SPLIT_SEED,
            )
        X_ctx = X_ctx.iloc[:context_size] if isinstance(X_ctx, pd.DataFrame) else X_ctx[:context_size]
        X_q = X_q.iloc[:query_size] if isinstance(X_q, pd.DataFrame) else X_q[:query_size]
        return X_ctx, y_ctx[:context_size], X_q, meta
    else:
        X_ctx = X.iloc[:context_size] if isinstance(X, pd.DataFrame) else X[:context_size]
        X_q = X.iloc[context_size:context_size + query_size] if isinstance(X, pd.DataFrame) else X[context_size:context_size + query_size]
        return X_ctx, y[:context_size], X_q, meta


def get_dataset_task(dataset_name: str) -> str:
    """Look up task type ('classification' or 'regression') from the catalog."""
    info = TABARENA_DATASETS.get(dataset_name, {})
    return info.get("task", "classification")


def extract_single_layer(
    model: str,
    layer: int,
    dataset_name: str,
    device: str = "cuda",
    context_size: int = 600,
    query_size: int = 500,
) -> np.ndarray:
    """Extract embeddings at a specific layer for one dataset.

    Returns the (n_query, hidden_dim) embedding array.
    """
    extract_fn = EXTRACT_FN[model]

    X_context, y_context, X_query, meta = load_context_query(
        dataset_name, context_size=context_size, query_size=query_size,
    )

    task = get_dataset_task(dataset_name)
    # Pass task kwarg for models that support it (e.g. TabPFN)
    sig = inspect.signature(extract_fn)
    kwargs = dict(device=device)
    if "task" in sig.parameters:
        kwargs["task"] = task
    # Pass cat_feature_indices for models that support it
    if "cat_feature_indices" in sig.parameters:
        kwargs["cat_feature_indices"] = meta.cat_feature_indices

    layer_embeddings = extract_fn(X_context, y_context, X_query, **kwargs)

    # Use index-based lookup into sorted layer list (matches depth analysis indexing).
    # Exact name match (layer_14) only works when sorted order == layer numbering,
    # which breaks for models with extra layers (TabICL's row_output, Mitra's final_norm).
    available = sort_layer_names(list(layer_embeddings.keys()))
    if 0 <= layer < len(available):
        layer_key = available[layer]
    else:
        raise ValueError(
            f"Layer index {layer} out of range for {model}. "
            f"Available ({len(available)}): {available}"
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
    parser.add_argument("--context-size", type=int, default=600,
                        help="Number of ICL context examples (default: 600)")
    parser.add_argument("--query-size", type=int, default=500,
                        help="Number of query rows to embed (default: 500)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific dataset names (default: all TabArena)")
    args = parser.parse_args()

    datasets = args.datasets or get_tabarena_dataset_names()
    output_dir = (
        PROJECT_ROOT / "output" / "embeddings" / "tabarena"
        / f"{args.model}_layer{args.layer}_ctx{args.context_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.model} layer {args.layer} embeddings")
    print(f"  Output: {output_dir}")
    print(f"  Context size: {args.context_size}, Query size: {args.query_size}")
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
                device=args.device,
                context_size=args.context_size,
                query_size=args.query_size,
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
