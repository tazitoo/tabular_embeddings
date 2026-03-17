#!/usr/bin/env python3
"""
Energy distance between SAE training embeddings and full dataset embeddings.

For each model × dataset:
  1. Load the 350-row SAE training subset (already normalized)
  2. Extract embeddings from the full dataset
  3. Normalize with the same per-dataset stats
  4. Compute energy distance between the two sets

Energy distance = 0 iff distributions are identical. Higher values indicate
the training subset misses structure present in the full embedding distribution.

Output:
  output/sae_energy_distance/{model}_energy_results.json

Usage:
  python scripts/embeddings/embedding_energy_distance.py --model tabpfn --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from scripts._project_root import PROJECT_ROOT
from scripts.embeddings.kl_divergence_subsample import (
    load_full_context_query,
    extract_embeddings,
    load_sae_training_subset,
    load_norm_stats,
)
from scripts.embeddings.extract_layer_embeddings import (
    EXTRACT_FN,
    get_dataset_task,
)
from config import get_optimal_layer
from data.extended_loader import TABARENA_DATASETS

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_energy_distance"


def energy_distance(X: np.ndarray, Y: np.ndarray, max_samples: int = 2000) -> float:
    """Compute energy distance between two multivariate samples.

    Subsamples both sets to max_samples for computational tractability.
    """
    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    dXY = cdist(X, Y, metric="euclidean").mean()
    dXX = cdist(X, X, metric="euclidean").mean()
    dYY = cdist(Y, Y, metric="euclidean").mean()

    return float(2 * dXY - dXX - dYY)


def analyze_model(model: str, device: str = "cuda") -> dict:
    """Compute energy distance for all datasets of one model."""
    optimal_layer = get_optimal_layer(model)
    datasets = sorted(TABARENA_DATASETS.keys())

    results = {
        "model": model,
        "optimal_layer": optimal_layer,
        "datasets": {},
        "summary": {},
    }

    all_ed = []

    for ds_name in datasets:
        task = get_dataset_task(ds_name)

        if model in ("hyperfast", "tabicl") and task == "regression":
            print(f"  {ds_name} ({task})... SKIP (cls-only model)")
            continue

        print(f"  {ds_name} ({task})...", end=" ", flush=True)

        # Load norm stats
        mean, std = load_norm_stats(model, optimal_layer, ds_name)
        if mean is None:
            print("SKIP (no norm stats)")
            continue

        # Load SAE training subset (already normalized)
        train_subset = load_sae_training_subset(model, ds_name, optimal_layer)
        if train_subset is None:
            print("SKIP (no training data)")
            continue

        # Load full dataset
        try:
            X_ctx, y_ctx, X_query, y_query = load_full_context_query(ds_name)
        except (ValueError, Exception) as e:
            print(f"SKIP (load: {e})")
            continue

        n_full = len(X_query)
        n_train = len(train_subset)

        if n_full <= n_train * 1.5:
            print(f"SKIP (only {n_full} query rows)")
            continue

        # Extract and normalize full embeddings
        try:
            full_emb = extract_embeddings(
                model, optimal_layer, X_ctx, y_ctx, X_query,
                dataset_task=task, device=device,
            )
        except Exception as e:
            print(f"SKIP (extract: {e})")
            continue

        mean_np = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
        std_np = std.cpu().numpy() if isinstance(std, torch.Tensor) else std
        full_emb_norm = (full_emb - mean_np) / std_np

        # Energy distance
        ed = energy_distance(full_emb_norm, train_subset)
        all_ed.append(ed)

        results["datasets"][ds_name] = {
            "n_full": n_full,
            "n_train": n_train,
            "energy_distance": round(ed, 6),
            "task": task,
        }

        print(f"n={n_full}, ED={ed:.4f}")

    if all_ed:
        results["summary"] = {
            "n_datasets": len(all_ed),
            "mean_ed": round(float(np.mean(all_ed)), 6),
            "median_ed": round(float(np.median(all_ed)), 6),
            "max_ed": round(float(np.max(all_ed)), 6),
            "p95_ed": round(float(np.percentile(all_ed, 95)), 6),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Energy distance: SAE training embeddings vs full dataset"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = sorted(k for k in EXTRACT_FN.keys() if k != "tabula8b")
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Embedding energy distance: {model}")
        print(f"{'='*60}")

        results = analyze_model(model, device=args.device)

        out_path = OUTPUT_DIR / f"{model}_energy_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved: {out_path}")
        if results["summary"]:
            s = results["summary"]
            print(f"  Mean ED: {s['mean_ed']:.4f}")
            print(f"  Median ED: {s['median_ed']:.4f}")
            print(f"  Max ED: {s['max_ed']:.4f}")


if __name__ == "__main__":
    main()
