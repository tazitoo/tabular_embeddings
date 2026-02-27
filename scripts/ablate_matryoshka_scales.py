#!/usr/bin/env python3
"""
Ablate Matryoshka scale bands and measure RMSE contribution.

For each scale band (S1, S2, S3, S4, S5+), zero out those features
and measure how much RMSE increases. This tells us which bands actually
contribute to reconstruction vs. which are harmful.

If higher scale bands (S4, S5) contribute little or increase RMSE,
that explains the decreasing R² pattern in the domain reconstruction plots.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import sae_sweep_dir
from scripts.sae_tabarena_sweep import get_tabarena_splits, pool_embeddings
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint


def compute_rmse(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute RMSE between original and reconstruction."""
    return np.sqrt(np.mean((original - reconstruction) ** 2))


def ablate_scale_band(
    model,
    activations: torch.Tensor,
    scale_start: int,
    scale_end: int,
) -> np.ndarray:
    """
    Zero out a scale band and reconstruct.

    Args:
        model: SAE model
        activations: (n_samples, hidden_dim) activation tensor
        scale_start: Start index of band to ablate
        scale_end: End index of band to ablate

    Returns:
        Reconstructed embeddings with ablated band
    """
    # Clone activations to avoid modifying original
    ablated = activations.clone()

    # Zero out the scale band
    ablated[:, scale_start:scale_end] = 0.0

    # Reconstruct
    with torch.no_grad():
        reconstruction = model.decode(ablated).cpu().numpy()

    return reconstruction


def analyze_scale_contributions(
    model,
    config,
    embeddings: np.ndarray,
    train_std: np.ndarray,
    train_mean: np.ndarray,
) -> Dict[str, Dict]:
    """
    Ablate each Matryoshka scale band and measure RMSE.

    Returns dict with:
        - baseline_rmse: RMSE with all features
        - scale_band_rmse: RMSE when each band is ablated
        - scale_band_contribution: How much each band reduces RMSE
    """
    # Normalize embeddings (same as training)
    embeddings_norm = embeddings / train_std - train_mean

    # Get activations
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings_norm, dtype=torch.float32)
        activations = model.encode(embeddings_tensor)

        # Baseline: full reconstruction
        baseline_recon = model.decode(activations).cpu().numpy()
        baseline_rmse = compute_rmse(embeddings_norm, baseline_recon)

    # Define scale bands
    mat_dims = config.matryoshka_dims or [32, 64, 128, 256]
    mat_dims = [d for d in mat_dims if d <= config.hidden_dim]

    # Add full dim if not already there
    if mat_dims[-1] < config.hidden_dim:
        mat_dims.append(config.hidden_dim)

    # Build band boundaries
    boundaries = [0] + mat_dims
    bands = []
    for i in range(len(boundaries) - 1):
        bands.append((boundaries[i], boundaries[i+1]))

    band_labels = []
    for i, (start, end) in enumerate(bands):
        band_labels.append(f"S{i+1} [{start},{end})")

    # Ablate each band
    results = {
        'baseline_rmse': baseline_rmse,
        'bands': {},
    }

    print(f"\n{'Band':<15} {'RMSE (ablated)':<18} {'Δ RMSE':<12} {'Contribution':<15}")
    print("-" * 70)
    print(f"{'Baseline':<15} {baseline_rmse:<18.6f} {'':<12} {'':<15}")

    for label, (start, end) in zip(band_labels, bands):
        # Ablate this band
        ablated_recon = ablate_scale_band(model, activations, start, end)
        ablated_rmse = compute_rmse(embeddings_norm, ablated_recon)

        # Delta: positive means band was helping, negative means band was hurting
        delta_rmse = ablated_rmse - baseline_rmse

        # Contribution: fraction of baseline RMSE reduced by this band
        contribution = -delta_rmse / baseline_rmse if baseline_rmse > 0 else 0.0

        results['bands'][label] = {
            'start': start,
            'end': end,
            'ablated_rmse': ablated_rmse,
            'delta_rmse': delta_rmse,
            'contribution': contribution,
        }

        sign = '+' if delta_rmse >= 0 else ''
        contrib_pct = 100 * contribution
        contrib_sign = '' if contribution >= 0 else ''

        print(f"{label:<15} {ablated_rmse:<18.6f} {sign}{delta_rmse:<11.6f} "
              f"{contrib_sign}{contrib_pct:>6.2f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablate Matryoshka scale bands")
    parser.add_argument("--model", default="tabicl_layer10", help="Model SAE directory name")
    parser.add_argument("--emb-dir", default=None, help="Embeddings directory (if different from model)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Matryoshka Scale Band Ablation: {args.model}")
    print("=" * 70)

    # Load SAE
    sae_path = sae_sweep_dir() / args.model / "sae_matryoshka_archetypal_validated.pt"
    if not sae_path.exists():
        print(f"Error: SAE not found: {sae_path}")
        return 1

    model, config, metrics = load_sae_checkpoint(sae_path)
    print(f"\n=== SAE Config ===")
    print(f"  Input: {config.input_dim}, Hidden: {config.hidden_dim}, TopK: {config.topk}")
    print(f"  Matryoshka dims: {config.matryoshka_dims}")
    print(f"  Ghost grads: {config.use_ghost_grads}")

    # Map SAE dir to embeddings dir
    emb_model = args.emb_dir or args.model.split('_layer')[0]

    # Load embeddings
    train_datasets, test_datasets = get_tabarena_splits(emb_model)
    print(f"\n=== Loading test embeddings ===")
    print(f"  Embeddings dir: {emb_model}")
    print(f"  Test datasets: {len(test_datasets)}")

    pooled, _ = pool_embeddings(emb_model, test_datasets, max_per_dataset=500, normalize=False)
    print(f"  Pooled: {pooled.shape[0]} samples × {pooled.shape[1]} dims")

    # Compute train stats for normalization
    train_pooled, _ = pool_embeddings(emb_model, train_datasets, max_per_dataset=500, normalize=False)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)

    # Analyze scale contributions
    print(f"\n=== Scale Band Ablation Analysis ===")
    print("Positive Δ RMSE = band helps (ablating it increases error)")
    print("Negative Δ RMSE = band hurts (ablating it decreases error)")

    results = analyze_scale_contributions(
        model, config, pooled, train_std, train_mean
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    bands_helping = []
    bands_hurting = []

    for label, band_results in results['bands'].items():
        if band_results['delta_rmse'] > 0:
            bands_helping.append((label, band_results['contribution']))
        else:
            bands_hurting.append((label, band_results['contribution']))

    if bands_helping:
        print("\nBands contributing to reconstruction:")
        for label, contrib in sorted(bands_helping, key=lambda x: -x[1]):
            print(f"  {label}: {100*contrib:>6.2f}%")

    if bands_hurting:
        print("\nBands HURTING reconstruction:")
        for label, contrib in sorted(bands_hurting, key=lambda x: x[1]):
            print(f"  {label}: {100*contrib:>6.2f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
