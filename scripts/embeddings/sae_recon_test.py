#!/usr/bin/env python3
"""
SAE reconstruction test: does the SAE generalize beyond its training data?

For each model × dataset:
  1. Extract embeddings from the FULL dataset (up to 10K query rows)
  2. Normalize with the SAE's per-dataset stats
  3. Run through SAE encode→decode
  4. Compare MSE on full data vs MSE on the 350-row training subset

If MSE_full ≈ MSE_train, the 350-row subsample was sufficient.
If MSE_full >> MSE_train, the SAE overfit to the subsample.

Output:
  output/sae_recon_test/{model}_recon_results.json

Usage:
  python scripts/embeddings/sae_recon_test.py --model tabpfn --device cuda
  python scripts/embeddings/sae_recon_test.py --model all --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
from scripts.intervention.intervene_sae import load_sae
from config import get_optimal_layer
from data.extended_loader import TABARENA_DATASETS

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_recon_test"


def recon_mse(sae, x_norm: np.ndarray, device: str = "cuda", batch_size: int = 2048) -> float:
    """Compute mean reconstruction MSE for normalized embeddings."""
    sae.eval()
    total_mse = 0.0
    n = len(x_norm)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.tensor(x_norm[start:end], dtype=torch.float32, device=device)
            h = sae.encode(x)
            x_hat = sae.decode(h)
            mse = ((x - x_hat) ** 2).mean(dim=1)  # per-row MSE
            total_mse += mse.sum().item()

    return total_mse / n


def analyze_model(model: str, device: str = "cuda") -> dict:
    """Run reconstruction test for all datasets of one model."""
    optimal_layer = get_optimal_layer(model)
    sae, config = load_sae(model, device=device)
    datasets = sorted(TABARENA_DATASETS.keys())

    results = {
        "model": model,
        "optimal_layer": optimal_layer,
        "datasets": {},
        "summary": {},
    }

    mse_ratios = []

    for ds_name in datasets:
        task = get_dataset_task(ds_name)

        if model in ("hyperfast", "tabicl") and task == "regression":
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

        # MSE on training subset
        mse_train = recon_mse(sae, train_subset, device=device)

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

        # Normalize with SAE training stats
        mean_np = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
        std_np = std.cpu().numpy() if isinstance(std, torch.Tensor) else std
        full_emb_norm = (full_emb - mean_np) / std_np

        # MSE on full dataset
        mse_full = recon_mse(sae, full_emb_norm, device=device)

        ratio = mse_full / mse_train if mse_train > 0 else float('inf')
        mse_ratios.append(ratio)

        results["datasets"][ds_name] = {
            "n_full": n_full,
            "n_train": n_train,
            "mse_train": round(mse_train, 6),
            "mse_full": round(mse_full, 6),
            "ratio": round(ratio, 4),
            "task": task,
        }

        print(f"n={n_full}, MSE train={mse_train:.4f} full={mse_full:.4f} ratio={ratio:.2f}")

    if mse_ratios:
        results["summary"] = {
            "n_datasets": len(mse_ratios),
            "mean_ratio": round(float(np.mean(mse_ratios)), 4),
            "median_ratio": round(float(np.median(mse_ratios)), 4),
            "max_ratio": round(float(np.max(mse_ratios)), 4),
            "p95_ratio": round(float(np.percentile(mse_ratios, 95)), 4),
            "datasets_ratio_gt_2": [
                ds for ds, r in results["datasets"].items()
                if r["ratio"] > 2.0
            ],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAE reconstruction test: training subset vs full dataset"
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
        print(f"SAE reconstruction test: {model}")
        print(f"{'='*60}")

        results = analyze_model(model, device=args.device)

        out_path = OUTPUT_DIR / f"{model}_recon_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved: {out_path}")
        if results["summary"]:
            s = results["summary"]
            print(f"  Mean MSE ratio (full/train): {s['mean_ratio']:.2f}")
            print(f"  Median: {s['median_ratio']:.2f}")
            print(f"  Max: {s['max_ratio']:.2f}")
            print(f"  Datasets with ratio > 2x: {s['datasets_ratio_gt_2']}")


if __name__ == "__main__":
    main()
