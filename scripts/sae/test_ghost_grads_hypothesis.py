#!/usr/bin/env python3
"""
Test hypothesis: Ghost grads degrade Matryoshka-Archetypal SAE reconstruction.

Retrain TabICL SAE with identical hyperparameters but ghost_grads=False,
then compare reconstruction R² across scales to the original.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import sae_sweep_dir
from scripts.sae_tabarena_sweep import (
    get_tabarena_splits,
    pool_embeddings,
    compute_stability,
)
from analysis.sparse_autoencoder import SAEConfig, train_sae


def main():
    print("=" * 70)
    print("Testing Ghost Grads Hypothesis: TabICL SAE Retraining")
    print("=" * 70)

    # Load original config
    original_path = sae_sweep_dir() / "tabicl_layer10/sae_matryoshka_archetypal_validated.pt"
    ckpt = torch.load(original_path, map_location='cpu')
    old_config = ckpt['config']

    print("\n=== Original Config (WITH ghost grads) ===")
    print(f"  Input: {old_config['input_dim']}, Hidden: {old_config['hidden_dim']}, TopK: {old_config['topk']}")
    print(f"  Ghost grads: {old_config['use_ghost_grads']}")
    print(f"  Validation score: {ckpt['validation']['composite_score']:.3f}")
    print(f"  Validation R²: {ckpt['validation']['r2']:.3f}")
    print(f"  Validation stability: {ckpt['validation']['stability']:.3f}")

    # Get train/test split
    model_name = "tabicl_layer10"
    train_datasets, test_datasets = get_tabarena_splits(model_name)
    print(f"\n=== Dataset split ===")
    print(f"  Train: {len(train_datasets)} datasets")
    print(f"  Test: {len(test_datasets)} datasets")

    # Pool training embeddings
    print(f"\n=== Loading training embeddings ===")
    pooled, dataset_counts = pool_embeddings(
        model_name, train_datasets, max_per_dataset=500, normalize=True
    )
    print(f"  Pooled: {pooled.shape[0]} samples × {pooled.shape[1]} dims")

    # New config: identical hyperparams but ghost_grads=False
    new_config = SAEConfig(
        input_dim=old_config['input_dim'],
        hidden_dim=old_config['hidden_dim'],
        sparsity_type='matryoshka_archetypal',
        topk=old_config['topk'],
        matryoshka_dims=old_config['matryoshka_dims'],
        archetypal_n_archetypes=old_config['archetypal_n_archetypes'],
        archetypal_simplex_temp=old_config['archetypal_simplex_temp'],
        lr=old_config['learning_rate'],
        batch_size=old_config['batch_size'],
        n_epochs=old_config['n_epochs'],
        use_ghost_grads=False,  # ← THE ONLY CHANGE
        dead_freq_threshold=old_config.get('dead_freq_threshold', 0.001),
    )

    print("\n=== New Config (WITHOUT ghost grads) ===")
    print(f"  Input: {new_config.input_dim}, Hidden: {new_config.hidden_dim}, TopK: {new_config.topk}")
    print(f"  Ghost grads: {new_config.use_ghost_grads}")
    print(f"  All other hyperparams identical to original")

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Training on {device} ===")

    model, result = train_sae(pooled, new_config, device=device, verbose=True)

    print(f"\n=== Training Results ===")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  R²: {result.r_squared:.3f}")
    print(f"  Alive features: {(result.activations.max(axis=0) > 1e-3).sum()}/{new_config.hidden_dim}")

    # Compute stability
    print(f"\n=== Computing stability (2 independent runs) ===")
    stability = compute_stability(pooled, new_config, n_runs=2, device=device)
    print(f"  Stability: {stability:.3f}")

    # Composite score (like in the sweep)
    composite_score = 0.4 * result.r_squared + 0.6 * stability
    print(f"  Composite score: {composite_score:.3f}")

    # Save with special name
    output_dir = sae_sweep_dir() / "tabicl_layer10"
    output_path = output_dir / "sae_matryoshka_archetypal_NO_GHOST_GRADS.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in vars(new_config).items()},
        'validation': {
            'composite_score': composite_score,
            'r2': result.r_squared,
            'stability': stability,
            'n_alive': (result.activations.max(axis=0) > 1e-3).sum(),
        },
    }, output_path)
    print(f"\nSaved: {output_path}")

    # Compare to original
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Original (ghost=True)':<25} {'New (ghost=False)':<25}")
    print("-" * 70)
    print(f"{'Composite score':<20} {ckpt['validation']['composite_score']:>24.3f} {composite_score:>24.3f}")
    print(f"{'R²':<20} {ckpt['validation']['r2']:>24.3f} {result.r_squared:>24.3f}")
    print(f"{'Stability':<20} {ckpt['validation']['stability']:>24.3f} {stability:>24.3f}")

    delta_r2 = result.r_squared - ckpt['validation']['r2']
    delta_stability = stability - ckpt['validation']['stability']
    print(f"\n{'R² delta':<20} {delta_r2:>24.3f} {'(+better)' if delta_r2 > 0 else '(worse)'}")
    print(f"{'Stability delta':<20} {delta_stability:>24.3f} {'(+better)' if delta_stability > 0 else '(worse)'}")

    if delta_r2 > 0.05 or delta_stability > 0.1:
        print("\n✓ HYPOTHESIS CONFIRMED: Disabling ghost grads improves reconstruction")
    elif delta_r2 < -0.05:
        print("\n✗ HYPOTHESIS REJECTED: Ghost grads do not harm reconstruction")
    else:
        print("\n~ INCONCLUSIVE: Difference is small")


if __name__ == '__main__':
    main()
