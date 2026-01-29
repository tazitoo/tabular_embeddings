#!/usr/bin/env python3
"""
Extract embeddings from a single model across all datasets in a benchmark suite.

Designed to run on GPU workers via SSH. Saves one .npz file per dataset
under output_dir/model_name/.

Usage:
    # On a GPU worker
    python extract_embeddings.py --model tabpfn --suite tabarena \
        --output-dir /data/embeddings/tabarena --device cuda

    # Quick smoke test
    python extract_embeddings.py --model tabpfn --suite quick \
        --output-dir /tmp/test_emb --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import get_extractor, MODEL_REGISTRY
from data.loader import load_benchmark_suite


def save_embeddings(result, output_path: Path):
    """Save EmbeddingResult to .npz file."""
    save_dict = {
        "embeddings": result.embeddings,
        "extraction_point": np.array(result.extraction_point),
        "n_samples": np.array(result.n_samples),
        "embedding_dim": np.array(result.embedding_dim),
    }

    # Save layer embeddings
    layer_names = list(result.layer_embeddings.keys())
    save_dict["layer_names"] = np.array(layer_names, dtype=str)
    for name, emb in result.layer_embeddings.items():
        save_dict[f"layer_{name}"] = emb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a single model across benchmark datasets"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help="Model to extract embeddings from")
    parser.add_argument("--suite", type=str, default="tabarena",
                        choices=["tabzilla", "cc18", "quick", "regression", "tabarena", "relbench"],
                        help="Benchmark suite")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory (e.g. /data/embeddings/tabarena)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--context-size", type=int, default=600)
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per dataset")
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="Max datasets to process")
    args = parser.parse_args()

    # Load benchmark suite
    print(f"Loading {args.suite} benchmark suite...")
    datasets = load_benchmark_suite(
        args.suite,
        max_samples=args.max_samples,
        max_datasets=args.max_datasets,
    )
    print(f"Loaded {len(datasets)} datasets\n")

    # Load model once
    print(f"Loading {args.model}...")
    extractor = get_extractor(args.model, device=args.device)
    extractor.load_model()
    print(f"Model loaded\n")

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    failed_names = []

    for i, (X, y, meta) in enumerate(datasets):
        dataset_name = meta.get("name") if isinstance(meta, dict) else getattr(meta, "name", "unknown")
        print(f"[{i+1}/{len(datasets)}] {dataset_name}...", end=" ", flush=True)

        try:
            # Split into context and query (same logic as run_comparison)
            n_total = len(X)
            ctx = args.context_size
            qry = args.query_size
            if n_total < ctx + qry:
                ctx = int(n_total * 0.7)
                qry = n_total - ctx

            X_context = X[:ctx]
            y_context = y[:ctx]
            X_query = X[ctx:ctx + qry]

            result = extractor.extract_embeddings(X_context, y_context, X_query)

            output_path = output_dir / f"{dataset_name}.npz"
            save_embeddings(result, output_path)

            n_query = result.embeddings.shape[0]
            dim = result.embeddings.shape[1] if result.embeddings.ndim > 1 else 1
            print(f"shape=({n_query}, {dim})")
            succeeded += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            failed_names.append(dataset_name)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done: {succeeded} succeeded, {failed} failed")
    print(f"Output: {output_dir}")
    if failed_names:
        print(f"Failed datasets: {', '.join(failed_names)}")


if __name__ == "__main__":
    main()
