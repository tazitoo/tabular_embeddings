#!/usr/bin/env python3
"""Compute layerwise CKA matrices from pre-extracted all-layer embeddings.

Reads the output of 04_extract_all_layers.py and computes pairwise CKA
between all layers for each dataset.  Outputs are compatible with
analyze_layerwise_cka.py for optimal layer selection.

Output:
    output/layerwise_cka_v2/layerwise_cka_{model}_{dataset}.npz
        cka_matrix:   (n_layers, n_layers) float64
        layer_names:  list of layer name strings

Usage:
    python scripts/sae_corpus/07_compute_layerwise_cka.py --model tabpfn
    python scripts/sae_corpus/07_compute_layerwise_cka.py --model all
    python scripts/sae_corpus/07_compute_layerwise_cka.py --model tabula8b --max-rows 2000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.similarity import centered_kernel_alignment
from models.layer_extraction import sort_layer_names
from scripts._project_root import PROJECT_ROOT

EMBEDDINGS_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "output" / "layerwise_cka_v2"

MODELS = ["tabpfn", "mitra", "tabdpt", "carte", "tabicl", "tabicl_v2", "hyperfast", "tabula8b"]


def compute_cka_matrix(
    layer_embeddings: dict[str, np.ndarray],
    max_rows: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise CKA between all layers.

    Args:
        layer_embeddings: {layer_name: (n_samples, dim)} arrays.
        max_rows: Subsample rows for speed (CKA is O(n^2) in samples).

    Returns:
        (cka_matrix, sorted_layer_names)
    """
    names = sort_layer_names(list(layer_embeddings.keys()))
    n = len(names)

    # Optionally subsample rows (same indices for all layers)
    if max_rows and len(next(iter(layer_embeddings.values()))) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(next(iter(layer_embeddings.values()))), max_rows, replace=False)
        layer_embeddings = {k: v[idx] for k, v in layer_embeddings.items()}

    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cka = centered_kernel_alignment(
                layer_embeddings[names[i]], layer_embeddings[names[j]]
            )
            matrix[i, j] = cka
            matrix[j, i] = cka
    return matrix, names


def process_model(model: str, max_rows: int | None, force: bool) -> dict:
    """Compute CKA matrices for all datasets of a model."""
    emb_dir = EMBEDDINGS_DIR / model
    if not emb_dir.exists():
        print(f"  No embeddings for {model}")
        return {"success": 0, "skipped": 0, "errors": 0}

    npz_files = sorted(emb_dir.glob("*.npz"))
    print(f"  {len(npz_files)} datasets")

    success = skipped = errors = 0

    for fi, npz_path in enumerate(npz_files):
        dataset = npz_path.stem
        out_path = OUTPUT_DIR / f"layerwise_cka_{model}_{dataset}.npz"

        if out_path.exists() and not force:
            skipped += 1
            continue

        t0 = time.time()
        try:
            data = np.load(npz_path, allow_pickle=True)
            layer_names = sort_layer_names(list(data["layer_names"]))

            # Load layer embeddings
            layer_embs = {}
            for lname in layer_names:
                if lname in data:
                    layer_embs[lname] = data[lname].astype(np.float32)

            if len(layer_embs) < 2:
                print(f"  [{fi+1}/{len(npz_files)}] {dataset}: SKIP (< 2 layers)")
                skipped += 1
                continue

            n_rows = len(next(iter(layer_embs.values())))
            cka_matrix, sorted_names = compute_cka_matrix(layer_embs, max_rows=max_rows)

            np.savez_compressed(
                str(out_path),
                cka_matrix=cka_matrix,
                layer_names=np.array(sorted_names),
            )

            dt = time.time() - t0
            print(f"  [{fi+1}/{len(npz_files)}] {dataset}: "
                  f"{len(sorted_names)} layers, {n_rows} rows ({dt:.1f}s)")
            success += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"  [{fi+1}/{len(npz_files)}] {dataset}: FAILED ({dt:.1f}s) — {e}")
            errors += 1

    return {"success": success, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Compute layerwise CKA from pre-extracted embeddings"
    )
    parser.add_argument("--model", required=True,
                        help="Model name or 'all'")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Subsample rows for CKA speed (default: 2000)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing CKA matrices")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = [m for m in MODELS if (EMBEDDINGS_DIR / m).exists()]
    else:
        models = [args.model]

    print(f"Computing layerwise CKA")
    print(f"  Input:    {EMBEDDINGS_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Max rows: {args.max_rows}")
    print(f"  Models:   {models}")
    print()

    for model in models:
        print(f"{'=' * 50}")
        print(f"  {model}")
        print("=" * 50)
        result = process_model(model, args.max_rows, args.force)
        print(f"  → {result['success']} computed, "
              f"{result['skipped']} skipped, {result['errors']} failed\n")


if __name__ == "__main__":
    main()
