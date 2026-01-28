#!/usr/bin/env python3
"""
Compare embedding geometry across tabular foundation models.

Inspired by "Harnessing the Universal Geometry of Embeddings" (Jha et al., NeurIPS 2025),
this script extracts embeddings from multiple tabular FMs and analyzes their geometric
similarity to test the Platonic Representation Hypothesis for tabular data.

Usage:
    # Compare on synthetic data (quick test)
    python compare_embeddings.py --synthetic --n-samples 500

    # Compare on OpenML dataset
    python compare_embeddings.py --dataset adult

    # Compare on benchmark suite
    python compare_embeddings.py --suite quick

    # Compare specific models
    python compare_embeddings.py --dataset iris --models tabpfn hyperfast

    # Distributed across GPU workers (multi-dataset)
    python compare_embeddings.py --suite tabarena --models tabpfn hyperfast tabicl --distributed
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import get_extractor, MODEL_REGISTRY
from models.base import EmbeddingResult
from analysis.similarity import (
    compute_pairwise_similarity,
    compute_cka_matrix,
    intrinsic_dimensionality,
)
from data.loader import (
    load_dataset,
    load_benchmark_suite,
    generate_synthetic_classification,
    generate_synthetic_regression,
    list_datasets,
)


def extract_all_embeddings(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    models: List[str],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from all specified models.

    Args:
        X_context: Training context features
        y_context: Training context labels
        X_query: Query samples to get embeddings for
        models: List of model names to use
        device: Torch device

    Returns:
        Dict mapping model_name -> embeddings
    """
    embeddings = {}

    for model_name in models:
        print(f"  Extracting from {model_name}...", end=" ", flush=True)

        try:
            extractor = get_extractor(model_name, device=device)
            extractor.load_model()
            result = extractor.extract_embeddings(X_context, y_context, X_query)

            # Use the primary embedding
            embeddings[model_name] = result.embeddings
            print(f"shape={result.embeddings.shape}")

            # Also store layer embeddings if available
            for layer_name, layer_emb in result.layer_embeddings.items():
                key = f"{model_name}_{layer_name}"
                if key not in embeddings:  # Don't overwrite primary
                    embeddings[key] = layer_emb

        except ValueError as e:
            print(f"Unknown model: {e}")
        except Exception as e:
            print(f"Failed: {e}")

    return embeddings


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    models: List[str],
    context_size: int = 600,
    query_size: int = 100,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Run embedding comparison on a dataset.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        models: List of models to compare
        context_size: Number of context samples
        query_size: Number of query samples
        device: Torch device
        verbose: Print progress

    Returns:
        Dict with comparison results
    """
    # Split into context and query
    n_total = len(X)
    if n_total < context_size + query_size:
        context_size = int(n_total * 0.7)
        query_size = n_total - context_size

    # Use last samples as query
    X_context = X[:context_size]
    y_context = y[:context_size]
    X_query = X[context_size:context_size + query_size]

    if verbose:
        print(f"\nData split: {context_size} context, {query_size} query")
        print(f"Features: {X.shape[1]}")

    # Extract embeddings
    if verbose:
        print(f"\nExtracting embeddings from {len(models)} models...")

    embeddings = extract_all_embeddings(
        X_context, y_context, X_query,
        models=models,
        device=device,
    )

    if len(embeddings) < 2:
        print("Need at least 2 models for comparison")
        return {}

    # Compute similarity metrics
    if verbose:
        print(f"\nComputing similarity metrics...")

    pairwise_results = compute_pairwise_similarity(embeddings)
    cka_matrix, model_names = compute_cka_matrix(embeddings)

    # Intrinsic dimensionality
    intrinsic_dims = {
        name: intrinsic_dimensionality(emb)
        for name, emb in embeddings.items()
    }

    # Compile results
    results = {
        "embeddings": embeddings,
        "pairwise_similarity": pairwise_results,
        "cka_matrix": cka_matrix,
        "model_names": model_names,
        "intrinsic_dims": intrinsic_dims,
        "context_size": context_size,
        "query_size": query_size,
        "n_features": X.shape[1],
    }

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SIMILARITY RESULTS")
        print("=" * 70)

        for (m1, m2), sim_result in pairwise_results.items():
            print(sim_result.summary())

        print("\nCKA Matrix:")
        print("-" * 70)
        # Header
        print(f"{'':20s}", end="")
        for name in model_names:
            print(f"{name[:10]:>12s}", end="")
        print()
        # Rows
        for i, name in enumerate(model_names):
            print(f"{name[:20]:20s}", end="")
            for j in range(len(model_names)):
                print(f"{cka_matrix[i, j]:12.3f}", end="")
            print()

        print("\nIntrinsic Dimensionality:")
        print("-" * 70)
        for name, dim in intrinsic_dims.items():
            actual_dim = embeddings[name].shape[1]
            print(f"  {name[:30]:30s}: {dim:4d} (of {actual_dim})")

    return results


def run_multi_dataset_comparison(
    datasets: List[Tuple[np.ndarray, np.ndarray, Dict]],
    models: List[str],
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run comparison across multiple datasets.

    Args:
        datasets: List of (X, y, metadata) tuples
        models: Models to compare
        device: Torch device

    Returns:
        DataFrame with aggregated results
    """
    rows = []

    for X, y, meta in datasets:
        dataset_name = meta.get("name", "unknown")
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        results = run_comparison(X, y, models, device=device)

        if not results:
            continue

        # Extract pairwise CKA scores
        for (m1, m2), sim_result in results["pairwise_similarity"].items():
            rows.append({
                "dataset": dataset_name,
                "model_a": m1,
                "model_b": m2,
                "cka_score": sim_result.cka_score,
                "mean_cosine_sim": sim_result.mean_cosine_sim,
                "procrustes_distance": sim_result.procrustes_distance,
                "dim_a": sim_result.embedding_dim_a,
                "dim_b": sim_result.embedding_dim_b,
            })

    return pd.DataFrame(rows)


def run_multi_dataset_comparison_distributed(
    datasets: List[Tuple[np.ndarray, np.ndarray, Dict]],
    models: List[str],
    context_size: int = 600,
    query_size: int = 100,
) -> pd.DataFrame:
    """
    Run comparison across multiple datasets using distributed GPU workers.

    Distributes embedding extraction across workers, then runs similarity
    analysis locally on the orchestrator.

    Args:
        datasets: List of (X, y, metadata) tuples
        models: Models to compare
        context_size: Context samples per dataset
        query_size: Query samples per dataset

    Returns:
        DataFrame with aggregated results
    """
    from distributed import run_on_workers, extract_embeddings_task

    # Build task list: (model, dataset) pairs
    tasks = []
    for X, y, meta in datasets:
        dataset_name = meta.get("name", "unknown")
        n_total = len(X)
        ctx = context_size if n_total >= context_size + query_size else int(n_total * 0.7)
        qry = min(query_size, n_total - ctx)

        X_ctx = X[:ctx]
        y_ctx = y[:ctx]
        X_q = X[ctx:ctx + qry]

        for model_name in models:
            tasks.append({
                "model_name": model_name,
                "X_context": X_ctx,
                "y_context": y_ctx,
                "X_query": X_q,
                "dataset_name": dataset_name,
            })

    print(f"\nDistributed: {len(tasks)} tasks ({len(datasets)} datasets x {len(models)} models)")

    # Distribute extraction across workers
    results = run_on_workers(extract_embeddings_task, tasks)

    # Group results by dataset
    dataset_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for r in results:
        if r is None or r.get("status") != "ok" or r["embeddings"] is None:
            if r is not None:
                print(f"  Skipping {r.get('model')} @ {r.get('dataset')}: {r.get('status')}")
            continue
        ds = r["dataset"]
        if ds not in dataset_embeddings:
            dataset_embeddings[ds] = {}
        dataset_embeddings[ds][r["model"]] = r["embeddings"]
        # Also store layer embeddings
        for layer_name, layer_emb in r.get("layer_embeddings", {}).items():
            key = f"{r['model']}_{layer_name}"
            if key not in dataset_embeddings[ds]:
                dataset_embeddings[ds][key] = layer_emb

    # Run similarity analysis locally
    rows = []
    for dataset_name, embeddings in dataset_embeddings.items():
        if len(embeddings) < 2:
            print(f"  {dataset_name}: <2 models succeeded, skipping similarity")
            continue

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        pairwise_results = compute_pairwise_similarity(embeddings)
        for (m1, m2), sim_result in pairwise_results.items():
            print(sim_result.summary())
            rows.append({
                "dataset": dataset_name,
                "model_a": m1,
                "model_b": m2,
                "cka_score": sim_result.cka_score,
                "mean_cosine_sim": sim_result.mean_cosine_sim,
                "procrustes_distance": sim_result.procrustes_distance,
                "dim_a": sim_result.embedding_dim_a,
                "dim_b": sim_result.embedding_dim_b,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding geometry across tabular foundation models"
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--dataset", type=str, help="Dataset name (e.g., adult, iris)")
    data_group.add_argument("--suite", type=str,
                           choices=["tabzilla", "cc18", "quick", "regression", "tabarena", "relbench"],
                           help="Run on entire benchmark suite")
    data_group.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    # Data params
    parser.add_argument("--n-samples", type=int, default=1000, help="Samples for synthetic")
    parser.add_argument("--n-features", type=int, default=20, help="Features for synthetic")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per dataset")
    parser.add_argument("--max-datasets", type=int, default=None, help="Max datasets from suite")
    parser.add_argument("--context-size", type=int, default=600, help="Context samples")
    parser.add_argument("--query-size", type=int, default=100, help="Query samples")

    # Model params
    available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["tabpfn"],
                        help=f"Models to compare ({available_models})")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])

    # Distributed
    parser.add_argument("--distributed", action="store_true",
                        help="Distribute extraction across GPU workers (multi-dataset only)")

    # Output
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")

    args = parser.parse_args()

    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:\n")
        for source, names in list_datasets("all").items():
            print(f"{source} ({len(names)} datasets):")
            for name in names[:10]:
                print(f"  - {name}")
            if len(names) > 10:
                print(f"  ... and {len(names) - 10} more")
            print()
        return

    # Load data
    datasets = []

    if args.synthetic:
        print("Generating synthetic data...")
        X, y, meta = generate_synthetic_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
        )
        datasets.append((X, y, meta))

    elif args.dataset:
        print(f"Loading {args.dataset}...")
        result = load_dataset(args.dataset, max_samples=args.max_samples)
        if result:
            datasets.append(result)
        else:
            print(f"Failed to load {args.dataset}")
            sys.exit(1)

    elif args.suite:
        print(f"Loading {args.suite} benchmark suite...")
        datasets = load_benchmark_suite(
            args.suite,
            max_samples=args.max_samples,
            max_datasets=args.max_datasets,
        )
        datasets = [(X, y, meta) for X, y, meta in datasets]

    else:
        # Default: synthetic data
        print("No data source specified, using synthetic data...")
        X, y, meta = generate_synthetic_classification(n_samples=500, n_features=15)
        datasets.append((X, y, meta))

    # Run comparison
    if len(datasets) == 1:
        X, y, meta = datasets[0]
        results = run_comparison(
            X, y,
            models=args.models,
            context_size=args.context_size,
            query_size=args.query_size,
            device=args.device,
        )
    else:
        if args.distributed:
            results_df = run_multi_dataset_comparison_distributed(
                datasets,
                models=args.models,
                context_size=args.context_size,
                query_size=args.query_size,
            )
        else:
            results_df = run_multi_dataset_comparison(
                datasets,
                models=args.models,
                device=args.device,
            )

        # Summary across datasets
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS")
        print("=" * 70)

        if len(results_df) > 0:
            summary = results_df.groupby(["model_a", "model_b"]).agg({
                "cka_score": ["mean", "std"],
                "mean_cosine_sim": ["mean", "std"],
                "procrustes_distance": ["mean", "std"],
            }).round(4)
            print(summary)

            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
