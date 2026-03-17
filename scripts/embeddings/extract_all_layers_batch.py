#!/usr/bin/env python3
"""
Batch layerwise embedding extraction across all TabArena datasets.

Extracts embeddings at ALL transformer layers for a given model across
all 51 TabArena datasets, using the corrected preprocessing pipeline
(DataFrames with proper dtypes, categorical columns preserved).

Saves per-dataset CKA matrices and raw layer embeddings for downstream
optimal-layer analysis.

Usage:
    # Single model on a GPU worker
    python scripts/embeddings/extract_all_layers_batch.py --model tabpfn --device cuda

    # Skip datasets that already have results
    python scripts/embeddings/extract_all_layers_batch.py --model tabpfn --device cuda

    # Specific datasets only
    python scripts/embeddings/extract_all_layers_batch.py --model tabpfn --datasets adult wine_quality
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts._project_root import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset
from analysis.similarity import centered_kernel_alignment

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

SPLIT_SEED = 42


def compute_cka_matrix(layer_embeddings: dict) -> tuple:
    """Compute pairwise CKA between all layers."""
    names = sort_layer_names(list(layer_embeddings.keys()))
    n = len(names)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cka = centered_kernel_alignment(
                layer_embeddings[names[i]], layer_embeddings[names[j]]
            )
            matrix[i, j] = cka
            matrix[j, i] = cka
    return matrix, names


def load_and_split(
    dataset_name: str,
    context_size: int = 600,
    query_size: int = 500,
):
    """Load dataset with proper preprocessing and split into context/query."""
    result = load_tabarena_dataset(
        dataset_name, max_samples=context_size + query_size
    )
    if result is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    X_df, y, meta = result
    n = len(X_df)
    if n < context_size + query_size:
        context_size = int(n * 0.7)
        query_size = n - context_size

    task = TABARENA_DATASETS[dataset_name]["task"]

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        query_frac = query_size / (context_size + query_size)
        try:
            X_ctx, X_q, y_ctx, _ = train_test_split(
                X_df, y, test_size=query_frac,
                random_state=SPLIT_SEED, stratify=y,
            )
        except ValueError:
            X_ctx, X_q, y_ctx, _ = train_test_split(
                X_df, y, test_size=query_frac, random_state=SPLIT_SEED,
            )
        X_ctx = X_ctx.iloc[:context_size].reset_index(drop=True)
        y_ctx = y_ctx[:context_size]
        X_q = X_q.iloc[:query_size].reset_index(drop=True)
    else:
        X_ctx = X_df.iloc[:context_size].reset_index(drop=True)
        y_ctx = y[:context_size]
        X_q = X_df.iloc[context_size:context_size + query_size].reset_index(drop=True)

    return X_ctx, y_ctx, X_q, task, meta


def main():
    parser = argparse.ArgumentParser(
        description="Batch layerwise extraction across TabArena"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(EXTRACT_FN.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-size", type=int, default=600)
    parser.add_argument("--query-size", type=int, default=500)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all TabArena)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    datasets = args.datasets or sorted(TABARENA_DATASETS.keys())
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / "layerwise_cka_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"Layerwise extraction: {args.model} × {len(datasets)} datasets")
    print(f"  Output: {output_dir}")
    print(f"  Context: {args.context_size}, Query: {args.query_size}")
    print(f"  Device: {args.device}")
    print(f"{'=' * 70}\n")

    import inspect
    extract_fn = EXTRACT_FN[args.model]
    sig = inspect.signature(extract_fn)

    success, skipped, failed = 0, 0, 0

    for i, ds_name in enumerate(datasets):
        out_path = output_dir / f"layerwise_cka_{args.model}_{ds_name}.npz"
        if out_path.exists():
            print(f"[{i+1}/{len(datasets)}] {ds_name}: exists, skipping")
            skipped += 1
            continue

        t0 = time.time()
        try:
            X_ctx, y_ctx, X_q, task, meta = load_and_split(
                ds_name, args.context_size, args.query_size,
            )

            # Build kwargs for the extract function
            kwargs = dict(device=args.device)
            if "task" in sig.parameters:
                kwargs["task"] = task
            if "cat_feature_indices" in sig.parameters:
                kwargs["cat_feature_indices"] = meta.cat_feature_indices

            # Extract all layers
            layer_embeddings = extract_fn(X_ctx, y_ctx, X_q, **kwargs)

            if not layer_embeddings:
                raise ValueError("No embeddings extracted")

            # Compute CKA matrix
            cka_matrix, layer_names = compute_cka_matrix(layer_embeddings)

            # Save CKA matrix + metadata
            np.savez_compressed(
                str(out_path),
                cka_matrix=cka_matrix,
                layer_names=np.array(layer_names),
                dataset=np.array(ds_name),
                model=np.array(args.model),
                task=np.array(task),
                n_cat_features=np.array(len(meta.cat_feature_indices)),
                n_features=np.array(meta.n_features),
            )

            dt = time.time() - t0
            # Print CKA of first vs last layer as a quick sanity check
            cka_first_last = cka_matrix[0, -1]
            print(f"[{i+1}/{len(datasets)}] {ds_name}: "
                  f"{len(layer_names)} layers, "
                  f"CKA(L0,L{len(layer_names)-1})={cka_first_last:.3f} "
                  f"({dt:.1f}s)")
            success += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"[{i+1}/{len(datasets)}] {ds_name}: FAILED ({dt:.1f}s)")
            traceback.print_exc()
            failed += 1

        # Clear GPU cache between datasets
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nDone: {success} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
