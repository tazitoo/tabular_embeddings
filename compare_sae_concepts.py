#!/usr/bin/env python3
"""
Compare learned concepts across tabular foundation models using Sparse Autoencoders.

Hypothesis: A more universal/capable tabular FM will learn a richer, more complete
dictionary of concepts that transfer across domains.

This script:
1. Extracts embeddings from multiple tabular FMs
2. Trains SAEs on each model's embeddings
3. Compares dictionary richness and structure
4. Tests if dictionary similarity correlates with prediction similarity

Usage:
    # Quick test with synthetic data
    python compare_sae_concepts.py --synthetic --n-samples 1000

    # Compare on OpenML dataset
    python compare_sae_concepts.py --dataset adult

    # Compare with specific dictionary size
    python compare_sae_concepts.py --dataset iris --dict-expansion 8

    # Distributed embedding extraction across GPU workers
    python compare_sae_concepts.py --dataset adult --models tabpfn hyperfast tabicl --distributed
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models import get_extractor, MODEL_REGISTRY
from models.base import EmbeddingResult
from analysis.sparse_autoencoder import (
    SAEConfig, SAEResult,
    train_sae,
    compare_dictionaries,
    measure_dictionary_richness,
    analyze_feature_geometry,
)
from analysis.similarity import centered_kernel_alignment
from data.loader import (
    load_dataset,
    generate_synthetic_classification,
)


def extract_embeddings_for_sae(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    models: List[str],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from models for SAE training.

    For SAE, we want embeddings for ALL samples (context + query)
    since we're training on the representation space, not doing prediction.
    """
    embeddings = {}

    # Combine context and query for SAE training
    X_all = np.vstack([X_context, X_query])
    y_all = np.concatenate([y_context, np.zeros(len(X_query), dtype=int)])

    for model_name in models:
        print(f"  Extracting from {model_name}...", end=" ", flush=True)

        try:
            extractor = get_extractor(model_name, device=device)
            extractor.load_model()

            # For SAE, we use a sliding window approach to get embeddings for all samples
            # Use first portion as context, rest as query, then slide
            emb_list = []
            window_size = min(500, len(X_all) // 3)
            step = window_size // 2

            for start in range(0, len(X_all) - window_size, step):
                ctx_end = start + window_size
                query_end = min(ctx_end + step, len(X_all))

                result = extractor.extract_embeddings(
                    X_all[start:ctx_end],
                    y_all[start:ctx_end],
                    X_all[ctx_end:query_end],
                )
                emb_list.append(result.embeddings)

            if emb_list:
                # Concatenate all embeddings
                all_emb = np.vstack(emb_list)
                embeddings[model_name] = all_emb
                print(f"shape={all_emb.shape}")
            else:
                print("no embeddings extracted")

        except ValueError as e:
            print(f"Unknown model: {e}")
        except Exception as e:
            print(f"Failed: {e}")

    return embeddings


def extract_embeddings_for_sae_distributed(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    models: List[str],
) -> Dict[str, np.ndarray]:
    """
    Extract sliding-window embeddings using distributed GPU workers.

    Same logic as extract_embeddings_for_sae but distributed across workers.
    Each (model) becomes a task.
    """
    from cluster import run_on_workers, extract_sae_embeddings_task

    X_all = np.vstack([X_context, X_query])
    y_all = np.concatenate([y_context, np.zeros(len(X_query), dtype=int)])

    tasks = []
    for model_name in models:
        tasks.append({
            "model_name": model_name,
            "X_all": X_all,
            "y_all": y_all,
            "dataset_name": "sae_extraction",
        })

    print(f"\nDistributed SAE extraction: {len(tasks)} models across GPU workers")
    results = run_on_workers(extract_sae_embeddings_task, tasks)

    embeddings = {}
    for r in results:
        if r is None or r.get("status") != "ok" or r["embeddings"] is None:
            if r is not None:
                print(f"  Skipping {r.get('model')}: {r.get('status')}")
            continue
        embeddings[r["model"]] = r["embeddings"]
        print(f"  {r['model']}: shape={r['embeddings'].shape} (worker: {r.get('worker', '?')})")

    return embeddings


def run_sae_comparison(
    X: np.ndarray,
    y: np.ndarray,
    models: List[str],
    dict_expansion: int = 4,
    sparsity_penalty: float = 1e-3,
    n_epochs: int = 100,
    device: str = "cpu",
    verbose: bool = True,
    distributed: bool = False,
) -> Dict:
    """
    Run SAE-based concept comparison across models.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        models: List of models to compare
        dict_expansion: Dictionary size = embedding_dim * expansion
        sparsity_penalty: L1 penalty for sparsity
        n_epochs: SAE training epochs
        device: Torch device
        verbose: Print progress

    Returns:
        Dict with comparison results
    """
    # Split data
    n_total = len(X)
    context_size = int(n_total * 0.7)
    X_context = X[:context_size]
    y_context = y[:context_size]
    X_query = X[context_size:]

    if verbose:
        print(f"\nData: {n_total} samples, {X.shape[1]} features")
        print(f"Split: {context_size} context, {len(X_query)} query")

    # Extract embeddings
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 1: Extract Embeddings")
        print(f"{'='*60}")

    if distributed:
        embeddings = extract_embeddings_for_sae_distributed(
            X_context, y_context, X_query,
            models=models,
        )
    else:
        embeddings = extract_embeddings_for_sae(
            X_context, y_context, X_query,
            models=models,
            device=device,
        )

    if len(embeddings) == 0:
        print("No embeddings extracted!")
        return {}

    # Train SAEs on each model's embeddings
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 2: Train Sparse Autoencoders")
        print(f"{'='*60}")

    sae_results = {}
    for model_name, emb in embeddings.items():
        if verbose:
            print(f"\n{model_name}:")

        embedding_dim = emb.shape[1]
        hidden_dim = embedding_dim * dict_expansion

        config = SAEConfig(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            sparsity_penalty=sparsity_penalty,
            sparsity_type="l1",
            n_epochs=n_epochs,
            learning_rate=1e-3,
        )

        if verbose:
            print(f"  Config: {embedding_dim}D -> {hidden_dim}D dictionary")

        _, result = train_sae(emb, config, device=device, verbose=verbose)
        sae_results[model_name] = result

        if verbose:
            print(f"  Alive features: {result.alive_features}/{hidden_dim}")
            print(f"  Mean active/sample: {result.mean_active_features:.1f}")

    # Analyze richness
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 3: Measure Dictionary Richness")
        print(f"{'='*60}")

    richness_results = {}
    for model_name, result in sae_results.items():
        richness = measure_dictionary_richness(result)
        richness_results[model_name] = richness

        if verbose:
            print(f"\n{model_name}:")
            print(f"  Richness score: {richness['richness_score']:.4f}")
            print(f"  Alive ratio: {richness['alive_ratio']:.2%}")
            print(f"  Diversity: {richness['dictionary_diversity']:.4f}")
            print(f"  Effective dims: {richness['effective_dimensions']:.1f}")

    # Analyze geometry
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 4: Analyze Feature Geometry")
        print(f"{'='*60}")

    geometry_results = {}
    for model_name, result in sae_results.items():
        geometry = analyze_feature_geometry(result.dictionary, result.feature_activations)
        geometry_results[model_name] = geometry

        if verbose:
            print(f"\n{model_name}:")
            print(f"  Power law alpha: {geometry['power_law_alpha']:.3f}")
            print(f"  Clustering: {geometry['mean_clustering']:.4f}")
            print(f"  Co-activation: {geometry['mean_coactivation']:.4f}")

    # Compare dictionaries across models
    if verbose and len(sae_results) >= 2:
        print(f"\n{'='*60}")
        print("STEP 5: Cross-Model Dictionary Comparison")
        print(f"{'='*60}")

    dict_comparisons = {}
    model_names = list(sae_results.keys())
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            comparison = compare_dictionaries(
                sae_results[model_a].dictionary,
                sae_results[model_b].dictionary,
            )
            dict_comparisons[(model_a, model_b)] = comparison

            if verbose:
                print(f"\n{model_a} vs {model_b}:")
                print(f"  Bidirectional matches: {comparison['bidirectional_matches']}")
                print(f"  Bidirectional rate: {comparison['bidirectional_rate']:.2%}")
                print(f"  Coverage A@0.7: {comparison['coverage_a_at_0.7']:.2%}")
                print(f"  Coverage B@0.7: {comparison['coverage_b_at_0.7']:.2%}")

    # Compare with embedding CKA
    if verbose and len(embeddings) >= 2:
        print(f"\n{'='*60}")
        print("STEP 6: Embedding Space vs Dictionary Similarity")
        print(f"{'='*60}")

    embedding_cka = {}
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            # Align sample counts
            min_samples = min(len(embeddings[model_a]), len(embeddings[model_b]))
            cka = centered_kernel_alignment(
                embeddings[model_a][:min_samples],
                embeddings[model_b][:min_samples],
            )
            embedding_cka[(model_a, model_b)] = cka

            if verbose and (model_a, model_b) in dict_comparisons:
                dict_sim = dict_comparisons[(model_a, model_b)]['bidirectional_rate']
                print(f"\n{model_a} vs {model_b}:")
                print(f"  Embedding CKA: {cka:.4f}")
                print(f"  Dictionary bidirectional: {dict_sim:.4f}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY: Dictionary Richness Ranking")
        print(f"{'='*60}")

        ranked = sorted(richness_results.items(), key=lambda x: x[1]['richness_score'], reverse=True)
        for rank, (model_name, richness) in enumerate(ranked, 1):
            print(f"  {rank}. {model_name}: {richness['richness_score']:.4f}")

    return {
        "embeddings": embeddings,
        "sae_results": sae_results,
        "richness": richness_results,
        "geometry": geometry_results,
        "dict_comparisons": dict_comparisons,
        "embedding_cka": embedding_cka,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare learned concepts across tabular FMs using SAEs"
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--dataset", type=str, help="Dataset name (e.g., adult, iris)")
    data_group.add_argument("--synthetic", action="store_true", help="Synthetic data")

    # Data params
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples from dataset")

    # SAE params
    parser.add_argument("--dict-expansion", type=int, default=4,
                        help="Dictionary size = embedding_dim * expansion")
    parser.add_argument("--sparsity", type=float, default=1e-3,
                        help="L1 sparsity penalty")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="SAE training epochs")

    # Model params
    available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["tabpfn"],
                        help=f"Models to compare ({available_models})")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])

    # Distributed
    parser.add_argument("--distributed", action="store_true",
                        help="Distribute embedding extraction across GPU workers")

    args = parser.parse_args()

    # Load data
    if args.synthetic:
        print("Generating synthetic data...")
        X, y, meta = generate_synthetic_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
        )
        print(f"  {meta}")

    elif args.dataset:
        print(f"Loading {args.dataset}...")
        result = load_dataset(args.dataset, max_samples=args.max_samples)
        if result:
            X, y, meta = result
            print(f"  {meta}")
        else:
            print(f"Failed to load {args.dataset}")
            sys.exit(1)

    else:
        print("Using default synthetic data...")
        X, y, meta = generate_synthetic_classification(n_samples=500, n_features=15)

    # Run comparison
    results = run_sae_comparison(
        X, y,
        models=args.models,
        dict_expansion=args.dict_expansion,
        sparsity_penalty=args.sparsity,
        n_epochs=args.n_epochs,
        device=args.device,
        distributed=args.distributed,
    )


if __name__ == "__main__":
    main()
