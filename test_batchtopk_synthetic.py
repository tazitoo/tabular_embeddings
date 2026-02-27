#!/usr/bin/env python3
"""
Test BatchTopK with synthetic embeddings (mimics TabPFN 192-dim output).
Verifies the core BatchTopK mechanism without requiring TabPFN installation.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SAEConfig, train_sae, measure_dictionary_richness


def generate_synthetic_embeddings(n_samples=8000, dim=192, complexity_variance=True):
    """
    Generate synthetic embeddings that mimic TabPFN behavior.

    Args:
        complexity_variance: If True, samples have varying complexity (different L2 norms)
    """
    np.random.seed(42)

    if complexity_variance:
        # Generate samples with varying complexity
        # Simple samples: low L2 norm (few active dimensions)
        # Complex samples: high L2 norm (many active dimensions)
        complexities = np.random.beta(2, 5, size=n_samples)  # Skewed toward simple
        embeddings = []

        for complexity in complexities:
            # Number of active dimensions scales with complexity
            n_active = int(dim * (0.1 + 0.9 * complexity))
            emb = np.zeros(dim)
            active_idx = np.random.choice(dim, n_active, replace=False)
            emb[active_idx] = np.random.randn(n_active) * complexity
            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        # Uniform complexity
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)

    return embeddings


def test_batchtopk_synthetic():
    """Test BatchTopK on synthetic embeddings."""
    print("=" * 80)
    print("Testing BatchTopK SAE on Synthetic Embeddings")
    print("=" * 80)

    # Generate embeddings
    print("\nGenerating synthetic embeddings (mimics TabPFN 192-dim)...")
    train_embeddings = generate_synthetic_embeddings(n_samples=8000, dim=192, complexity_variance=True)
    test_embeddings = generate_synthetic_embeddings(n_samples=2000, dim=192, complexity_variance=True)
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")

    # Train BatchTopK SAE
    print(f"\n{'='*80}")
    print("Training BatchTopK SAE")
    print(f"{'='*80}")

    config = SAEConfig(
        input_dim=192,
        hidden_dim=192 * 8,  # 1536 features
        sparsity_type="batchtopk",
        topk=32,  # Target average L0
        learning_rate=1e-3,
        batch_size=256,
        n_epochs=50,  # Fewer epochs for synthetic data
        use_ghost_grads=True,
    )

    print(f"Config:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Target avg L0: {config.topk}")

    # Train on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model, result = train_sae(train_embeddings, config, device=device, verbose=True)

    # Results
    print(f"\n{'='*80}")
    print("Training Results")
    print(f"{'='*80}")
    print(f"Reconstruction loss: {result.reconstruction_loss:.6f}")
    print(f"Alive features: {result.alive_features}/{config.hidden_dim} ({100*result.alive_features/config.hidden_dim:.1f}%)")
    print(f"Mean active per sample: {result.mean_active_features:.1f}")

    # Richness
    richness = measure_dictionary_richness(result, input_features=train_embeddings, sae_model=model)
    print(f"\nRichness Metrics:")
    print(f"  R²: {richness.get('explained_variance', 0):.4f}")
    print(f"  L0 sparsity: {richness['l0_sparsity']:.1f}")
    print(f"  Dictionary diversity: {richness['dictionary_diversity']:.4f}")

    # Inference threshold
    print(f"\nInference Threshold:")
    print(f"  Value: {model.inference_threshold.item():.6f}")
    print(f"  Updates: {model.batchtopk_n_updates.item()}")

    # Test adaptive sparsity
    print(f"\n{'='*80}")
    print("Testing Adaptive Sparsity (Test Set)")
    print(f"{'='*80}")

    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_embeddings, dtype=torch.float32, device=device)
        x = x - x.mean(dim=0, keepdim=True)  # Center
        h = model.encode(x)
        active_per_sample = (h > 0).sum(dim=1).cpu().float().numpy()

    print(f"\nActive features per sample (n={len(test_embeddings)}):")
    print(f"  Mean: {active_per_sample.mean():.1f}")
    print(f"  Std: {active_per_sample.std():.1f}")
    print(f"  Min: {active_per_sample.min():.0f}")
    print(f"  Max: {active_per_sample.max():.0f}")
    print(f"  Median: {np.median(active_per_sample):.0f}")
    print(f"  25th percentile: {np.percentile(active_per_sample, 25):.0f}")
    print(f"  75th percentile: {np.percentile(active_per_sample, 75):.0f}")

    # Plot
    output_dir = PROJECT_ROOT / "output" / "batchtopk_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(active_per_sample, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(active_per_sample.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {active_per_sample.mean():.1f}')
    ax.axvline(config.topk, color='green', linestyle='--',
               linewidth=2, label=f'Target: {config.topk}')
    ax.set_xlabel('Active Features per Sample', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('BatchTopK: Distribution of Active Features (Synthetic Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig_path = output_dir / "batchtopk_synthetic_distribution.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close(fig)

    # Verify adaptive behavior
    print(f"\n{'='*80}")
    print("Adaptive Sparsity Analysis")
    print(f"{'='*80}")

    if active_per_sample.std() > 5.0:
        print(f"✓ Strong adaptive sparsity (std={active_per_sample.std():.1f})")
        verdict = "PASS"
    elif active_per_sample.std() > 2.0:
        print(f"✓ Moderate adaptive sparsity (std={active_per_sample.std():.1f})")
        verdict = "PASS"
    else:
        print(f"⚠ Weak adaptive sparsity (std={active_per_sample.std():.1f})")
        verdict = "WARN"

    ratio = active_per_sample.mean() / config.topk
    print(f"\nMean vs Target ratio: {ratio:.2f}x")
    if 0.8 <= ratio <= 1.2:
        print(f"✓ Mean close to target")
    else:
        print(f"⚠ Mean differs from target")

    print(f"\n{'='*80}")
    print(f"BatchTopK Test: {verdict}")
    print(f"{'='*80}")

    return verdict == "PASS"


if __name__ == "__main__":
    success = test_batchtopk_synthetic()
    sys.exit(0 if success else 1)
