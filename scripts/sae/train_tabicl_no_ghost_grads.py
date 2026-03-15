#!/usr/bin/env python3
"""
Train TabICL SAE with best hyperparams but ghost_grads=False.

Loads the exact hyperparameters from the original validated checkpoint,
trains a new SAE with ghost_grads=False, and saves for comparison.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT

from analysis.sparse_autoencoder import SAEConfig, train_sae, measure_dictionary_richness
from scripts.sae.compare_sae_cross_model import sae_sweep_dir
from scripts.sae.sae_tabarena_sweep import (
    get_tabarena_splits,
    pool_embeddings,
    compute_stability,
)


def main():
    print("=" * 70)
    print("Training TabICL SAE without Ghost Grads")
    print("=" * 70)

    # Load original config to get hyperparameters
    original_path = sae_sweep_dir() / "tabicl_layer10/sae_matryoshka_archetypal_validated.pt"
    if not original_path.exists():
        print(f"Error: Original checkpoint not found: {original_path}")
        return 1

    ckpt = torch.load(original_path, map_location='cpu')
    old_config = ckpt['config']

    print("\n=== Original Config (WITH ghost grads) ===")
    print(f"  Input: {old_config['input_dim']}, Hidden: {old_config['hidden_dim']}, TopK: {old_config['topk']}")
    print(f"  Ghost grads: {old_config.get('use_ghost_grads', 'N/A')}")

    # Handle both checkpoint formats
    metrics = ckpt.get('validation') or ckpt.get('metrics', {})
    print(f"  Validation score: {metrics.get('composite_score', 'N/A')}")
    print(f"  Validation R²: {metrics.get('r2', 'N/A')}")
    print(f"  Validation stability: {metrics.get('stability', 'N/A')}")

    # Get train split and pool embeddings
    model_name = "tabicl_layer10"
    train_datasets, test_datasets = get_tabarena_splits(model_name)
    print(f"\n=== Loading embeddings ===")
    print(f"  Train datasets: {len(train_datasets)}")
    print(f"  Test datasets: {len(test_datasets)}")

    pooled, dataset_counts = pool_embeddings(
        model_name, train_datasets, max_per_dataset=500, normalize=True
    )
    print(f"  Pooled: {pooled.shape[0]} samples × {pooled.shape[1]} dims")

    # Build new config with ghost_grads=False
    new_config = SAEConfig(
        input_dim=old_config['input_dim'],
        hidden_dim=old_config['hidden_dim'],
        sparsity_type='matryoshka_archetypal',
        topk=old_config['topk'],
        matryoshka_dims=old_config['matryoshka_dims'],
        archetypal_n_archetypes=old_config['archetypal_n_archetypes'],
        archetypal_simplex_temp=old_config['archetypal_simplex_temp'],
        archetypal_relaxation=old_config.get('archetypal_relaxation', 1.0),
        archetypal_use_centroids=old_config.get('archetypal_use_centroids', True),
        learning_rate=old_config['learning_rate'],
        batch_size=old_config['batch_size'],
        n_epochs=old_config['n_epochs'],
        use_ghost_grads=False,  # ← THE FIX
        dead_freq_threshold=old_config.get('dead_freq_threshold', 0.001),
        use_aux_loss=old_config.get('use_aux_loss', True),
        aux_topk=old_config.get('aux_topk', 512),
    )

    print("\n=== New Config (WITHOUT ghost grads) ===")
    print(f"  Input: {new_config.input_dim}, Hidden: {new_config.hidden_dim}, TopK: {new_config.topk}")
    print(f"  Ghost grads: {new_config.use_ghost_grads}")
    print(f"  All other hyperparams identical")

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Training on {device} ===")

    model, result = train_sae(pooled, new_config, device=device, verbose=True)

    print(f"\n=== Training Results ===")
    print(f"  Final loss: {result.total_loss:.4f}")

    # Measure dictionary richness for R²
    richness = measure_dictionary_richness(result, input_features=pooled, sae_model=model)
    r2 = richness.get("explained_variance", 0.0)
    n_alive = richness.get("alive_features", 0)

    print(f"  R²: {r2:.3f}")
    print(f"  Alive features: {n_alive}/{new_config.hidden_dim}")

    # Compute stability
    print(f"\n=== Computing stability (2 independent runs) ===")
    stability = compute_stability(pooled, new_config, n_runs=2, device=device)
    print(f"  Stability: {stability:.3f}")

    # Composite score
    composite_score = 0.4 * r2 + 0.6 * stability
    print(f"  Composite score: {composite_score:.3f}")

    # Save
    output_path = sae_sweep_dir() / "tabicl_layer10/sae_matryoshka_archetypal_NO_GHOST_GRADS.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in vars(new_config).items()},
        'validation': {
            'composite_score': composite_score,
            'r2': r2,
            'stability': stability,
            'n_alive': int(n_alive),
        },
    }, output_path)
    print(f"\n=== Saved: {output_path.name} ===")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Original (ghost=True)':<25} {'New (ghost=False)':<25} {'Delta':<15}")
    print("-" * 70)

    old_score = metrics.get('composite_score', 0.0)
    old_r2 = metrics.get('r2', 0.0)
    old_stab = metrics.get('stability', 0.0)

    print(f"{'Composite score':<20} {old_score:>24.3f} {composite_score:>24.3f} {composite_score - old_score:>+14.3f}")
    print(f"{'R²':<20} {old_r2:>24.3f} {r2:>24.3f} {r2 - old_r2:>+14.3f}")
    print(f"{'Stability':<20} {old_stab:>24.3f} {stability:>24.3f} {stability - old_stab:>+14.3f}")

    delta_r2 = r2 - old_r2
    delta_stability = stability - old_stab

    print()
    if delta_r2 > 0.05 or delta_stability > 0.1:
        print("✓ HYPOTHESIS CONFIRMED: Disabling ghost grads improves reconstruction")
    elif delta_r2 < -0.05:
        print("✗ HYPOTHESIS REJECTED: Ghost grads do not harm reconstruction")
    else:
        print("~ INCONCLUSIVE: Difference is small (within ±0.05 R²)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
