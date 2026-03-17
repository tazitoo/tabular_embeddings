#!/usr/bin/env python3
"""
SAE reconstruction: training stats vs full-dataset stats.

For each model × dataset:
  1. Extract full-dataset embeddings (raw)
  2. Normalize two ways:
     a) With 350-row training stats (current pipeline)
     b) With full-dataset stats (computed from all rows)
  3. Run both through the SAE
  4. Compare reconstruction MSE

If MSE improves with full stats, the problem is normalization, not the SAE.
If MSE stays bad, the SAE dictionary is undertrained.

Output:
  output/sae_recon_normtest/{model}_normtest_results.json

Usage:
  python scripts/embeddings/sae_recon_normtest.py --model tabpfn --device cuda
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
    load_norm_stats,
)
from scripts.intervention.intervene_sae import load_sae
from scripts.embeddings.extract_layer_embeddings import (
    EXTRACT_FN,
    get_dataset_task,
)
from config import get_optimal_layer
from data.extended_loader import TABARENA_DATASETS

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_recon_normtest"
MIN_STD = 1e-8


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
            mse = ((x - x_hat) ** 2).mean(dim=1)
            total_mse += mse.sum().item()
    return total_mse / n


def analyze_model(model: str, device: str = "cuda") -> dict:
    """Compare reconstruction with training stats vs full-dataset stats."""
    optimal_layer = get_optimal_layer(model)
    sae, config = load_sae(model, device=device)
    datasets = sorted(TABARENA_DATASETS.keys())

    results = {
        "model": model,
        "optimal_layer": optimal_layer,
        "datasets": {},
        "summary": {},
    }

    improvements = []

    for ds_name in datasets:
        task = get_dataset_task(ds_name)

        if model in ("hyperfast", "tabicl") and task == "regression":
            continue

        print(f"  {ds_name} ({task})...", end=" ", flush=True)

        # Load 350-row training stats
        train_mean, train_std = load_norm_stats(model, optimal_layer, ds_name)
        if train_mean is None:
            print("SKIP (no norm stats)")
            continue

        train_mean_np = train_mean.cpu().numpy() if isinstance(train_mean, torch.Tensor) else train_mean
        train_std_np = train_std.cpu().numpy() if isinstance(train_std, torch.Tensor) else train_std

        # Load full dataset
        try:
            X_ctx, y_ctx, X_query, y_query = load_full_context_query(ds_name)
        except (ValueError, Exception) as e:
            print(f"SKIP (load: {e})")
            continue

        n_full = len(X_query)
        if n_full < 600:
            print(f"SKIP (only {n_full} rows)")
            continue

        # Extract raw embeddings
        try:
            raw_emb = extract_embeddings(
                model, optimal_layer, X_ctx, y_ctx, X_query,
                dataset_task=task, device=device,
            )
        except Exception as e:
            print(f"SKIP (extract: {e})")
            continue

        # Compute full-dataset stats
        full_mean = raw_emb.mean(axis=0)
        full_std = raw_emb.std(axis=0)
        full_std[full_std < MIN_STD] = 1.0

        # Normalize two ways
        emb_train_stats = (raw_emb - train_mean_np) / train_std_np
        emb_full_stats = (raw_emb - full_mean) / full_std

        # Reconstruction with each normalization
        mse_train_stats = recon_mse(sae, emb_train_stats, device=device)
        mse_full_stats = recon_mse(sae, emb_full_stats, device=device)

        # How different are the stats?
        mean_diff = float(np.abs(train_mean_np - full_mean).mean())
        std_ratio = float((train_std_np / full_std).mean())

        improvement = mse_train_stats / mse_full_stats if mse_full_stats > 0 else float('inf')
        improvements.append(improvement)

        results["datasets"][ds_name] = {
            "n_full": n_full,
            "mse_train_stats": round(mse_train_stats, 6),
            "mse_full_stats": round(mse_full_stats, 6),
            "improvement": round(improvement, 4),
            "mean_diff": round(mean_diff, 6),
            "std_ratio": round(std_ratio, 4),
            "task": task,
        }

        print(f"n={n_full}, MSE train_stats={mse_train_stats:.4f} full_stats={mse_full_stats:.4f} improvement={improvement:.2f}x")

    if improvements:
        results["summary"] = {
            "n_datasets": len(improvements),
            "mean_improvement": round(float(np.mean(improvements)), 4),
            "median_improvement": round(float(np.median(improvements)), 4),
            "max_improvement": round(float(np.max(improvements)), 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAE reconstruction: training stats vs full-dataset stats"
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
        print(f"Normalization test: {model}")
        print(f"{'='*60}")

        results = analyze_model(model, device=args.device)

        out_path = OUTPUT_DIR / f"{model}_normtest_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved: {out_path}")
        if results["summary"]:
            s = results["summary"]
            print(f"  Mean improvement (train_stats MSE / full_stats MSE): {s['mean_improvement']:.2f}x")
            print(f"  Median: {s['median_improvement']:.2f}x")


if __name__ == "__main__":
    main()
