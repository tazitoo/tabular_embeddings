#!/usr/bin/env python3
"""Evaluate fixed vs per-dataset layer SAEs on two test datasets.

Compares the two SAE variants on:
  - airfoil_self_noise (CKA critical layer L6, regression)
  - polish_companies_bankruptcy (CKA critical layer L23, classification)

For each (SAE variant, dataset), computes:
  1. Reconstruction quality (MSE on held-out embeddings)
  2. Feature activation statistics (alive, L0, max activation)
  3. Single-feature ablation importance (which features matter most)
  4. Top-10 most important features and their overlap between variants

The fixed SAE always uses L18 embeddings (its training layer).
The per-dataset SAE uses the dataset's critical layer.

Output:
    output/sae_training_round10/layer_comparison_eval.json
    stdout: comparison tables

Usage:
    python scripts/sae_corpus/09_evaluate_layer_comparison.py --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder
from models.layer_extraction import sort_layer_names
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"
EMBEDDINGS_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"

# Test datasets: chosen for maximum contrast with L18
EVAL_DATASETS = {
    "airfoil_self_noise": {"critical_layer": 6, "task": "regression"},
    "polish_companies_bankruptcy": {"critical_layer": 23, "task": "classification"},
}

VARIANTS = {
    "fixed": {
        "sae_path": OUTPUT_DIR / "tabpfn_layer18_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_layer18_norm_stats.npz",
        "layer_mode": "fixed",
        "global_layer": 18,
    },
    "per_dataset": {
        "sae_path": OUTPUT_DIR / "tabpfn_perds_sae.pt",
        "stats_path": OUTPUT_DIR / "tabpfn_perds_norm_stats.npz",
        "layer_mode": "per_dataset",
    },
}


def load_sae(path: Path, device: str) -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    config = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(config)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def load_norm_stats(stats_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load per-dataset norm stats. Returns (mean, std, layer)."""
    stats = np.load(str(stats_path), allow_pickle=True)
    datasets = list(stats["datasets"])
    if dataset not in datasets:
        return None, None, None
    idx = datasets.index(dataset)
    layer = int(stats["layers"][idx]) if "layers" in stats else None
    return stats["means"][idx], stats["stds"][idx], layer


def load_embeddings(model: str, dataset: str, layer: int) -> np.ndarray | None:
    """Load embeddings for a dataset at a specific layer."""
    npz_path = EMBEDDINGS_DIR / model / f"{dataset}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    layer_names = sort_layer_names(list(data["layer_names"]))
    if layer >= len(layer_names):
        return None
    return data[layer_names[layer]].astype(np.float32)


def evaluate_sae_on_dataset(
    sae: SparseAutoencoder,
    embeddings: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> dict:
    """Evaluate SAE reconstruction and feature importance on embeddings."""
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32, device=device)
        t_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        t_std = torch.tensor(std, dtype=torch.float32, device=device)

        # Normalize
        x_norm = (x - t_mean) / t_std

        # Encode
        h = sae.encode(x_norm)

        # Reconstruction
        recon = sae.decode(h)
        recon_mse = ((x_norm - recon) ** 2).mean().item()
        per_row_mse = ((x_norm - recon) ** 2).mean(dim=1).cpu().numpy()

        # Feature statistics
        h_np = h.cpu().numpy()
        active_mask = h_np > 0
        alive_features = active_mask.any(axis=0).sum()
        l0_per_row = active_mask.sum(axis=1).mean()
        mean_activation = h_np[active_mask].mean() if active_mask.any() else 0.0

        # Single-feature ablation importance
        # For each alive feature, zero it out and measure reconstruction change
        alive_idx = np.where(active_mask.any(axis=0))[0]
        feature_importance = np.zeros(sae.config.hidden_dim)

        for fi in alive_idx:
            h_ablated = h.clone()
            h_ablated[:, fi] = 0.0
            recon_ablated = sae.decode(h_ablated)
            ablated_mse = ((x_norm - recon_ablated) ** 2).mean().item()
            feature_importance[fi] = ablated_mse - recon_mse  # positive = feature helps

        # Top features
        top_k = 20
        top_idx = np.argsort(feature_importance)[::-1][:top_k]
        top_importance = feature_importance[top_idx]

        # Feature firing rates
        firing_rates = active_mask.mean(axis=0)

    return {
        "recon_mse": float(recon_mse),
        "per_row_mse_mean": float(per_row_mse.mean()),
        "per_row_mse_std": float(per_row_mse.std()),
        "alive_features": int(alive_features),
        "l0_per_row": float(l0_per_row),
        "mean_activation": float(mean_activation),
        "top_features": top_idx.tolist(),
        "top_importance": top_importance.tolist(),
        "feature_importance": feature_importance.tolist(),
        "firing_rates": firing_rates.tolist(),
        "n_rows": len(embeddings),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed vs per-dataset layer SAEs"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = {}

    for ds_name, ds_info in EVAL_DATASETS.items():
        critical_layer = ds_info["critical_layer"]
        task = ds_info["task"]
        print(f"\n{'=' * 60}")
        print(f"  {ds_name} (critical layer L{critical_layer}, {task})")
        print("=" * 60)

        ds_results = {}

        for var_name, var_info in VARIANTS.items():
            print(f"\n  --- {var_name} ---")

            # Determine which layer to use
            if var_name == "fixed":
                layer = var_info["global_layer"]
            else:
                # Per-dataset: use the critical layer
                layer = critical_layer

            # Load embeddings at the appropriate layer
            emb = load_embeddings("tabpfn", ds_name, layer)
            if emb is None:
                print(f"  SKIP: no embeddings at layer {layer}")
                ds_results[var_name] = {"error": f"no embeddings at L{layer}"}
                continue

            # Load norm stats
            mean, std, stats_layer = load_norm_stats(var_info["stats_path"], ds_name)
            if mean is None:
                print(f"  SKIP: no norm stats for {ds_name}")
                ds_results[var_name] = {"error": "no norm stats"}
                continue

            # Load SAE
            sae = load_sae(var_info["sae_path"], args.device)

            print(f"  Layer: L{layer}  |  Embeddings: {emb.shape}  |  "
                  f"Stats layer: L{stats_layer}")

            t0 = time.time()
            eval_result = evaluate_sae_on_dataset(sae, emb, mean, std, args.device)
            dt = time.time() - t0

            eval_result["layer_used"] = layer
            eval_result["task"] = task
            ds_results[var_name] = eval_result

            print(f"  Recon MSE:      {eval_result['recon_mse']:.4f}")
            print(f"  Alive features: {eval_result['alive_features']}")
            print(f"  L0/row:         {eval_result['l0_per_row']:.1f}")
            print(f"  Mean activation:{eval_result['mean_activation']:.4f}")
            print(f"  Top-5 features: {eval_result['top_features'][:5]}")
            print(f"  Top-5 Δ-MSE:    {['%.4f' % x for x in eval_result['top_importance'][:5]]}")
            print(f"  ({dt:.1f}s)")

        # Compare top features between variants
        if "fixed" in ds_results and "per_dataset" in ds_results:
            if "error" not in ds_results["fixed"] and "error" not in ds_results["per_dataset"]:
                top_fixed = set(ds_results["fixed"]["top_features"][:10])
                top_perds = set(ds_results["per_dataset"]["top_features"][:10])
                overlap = top_fixed & top_perds
                print(f"\n  Top-10 feature overlap: {len(overlap)}/10")
                if overlap:
                    print(f"  Shared features: {sorted(overlap)}")

        results[ds_name] = ds_results

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<35} {'Variant':<15} {'Layer':>5} {'Recon':>8} {'Alive':>6} {'Top-1 Δ':>8}")
    print("-" * 77)
    for ds_name in EVAL_DATASETS:
        for var_name in ["fixed", "per_dataset"]:
            r = results.get(ds_name, {}).get(var_name, {})
            if "error" in r:
                print(f"{ds_name:<35} {var_name:<15} {'ERR':>5} {r['error']}")
            elif r:
                top1 = r["top_importance"][0] if r["top_importance"] else 0.0
                print(f"{ds_name:<35} {var_name:<15} "
                      f"L{r['layer_used']:>3} {r['recon_mse']:>8.4f} "
                      f"{r['alive_features']:>6} {top1:>8.4f}")

    # Save
    # Strip large arrays for JSON
    for ds_name in results:
        for var_name in results[ds_name]:
            r = results[ds_name][var_name]
            if isinstance(r, dict):
                r.pop("feature_importance", None)
                r.pop("firing_rates", None)

    out_path = OUTPUT_DIR / "layer_comparison_eval.json"
    json.dump(results, open(str(out_path), "w"), indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
