#!/usr/bin/env python3
"""
Test BatchTopK SAE on adult dataset with real TabPFN embeddings.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SAEConfig, train_sae, measure_dictionary_richness
from data.loader import load_dataset


def test_batchtopk_on_adult():
    """Test BatchTopK on adult dataset."""
    print("=" * 80)
    print("Testing BatchTopK SAE on Adult Dataset")
    print("=" * 80)

    # Load adult dataset
    print("\nLoading adult dataset...")
    from sklearn.model_selection import train_test_split

    data = load_dataset("adult")
    if data is None:
        print("Error: Could not load adult dataset")
        return

    X, y, metadata = data
    print(f"  Total samples: {X.shape}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Extract TabPFN embeddings
    print("\nExtracting TabPFN embeddings...")
    from models.tabpfn_embeddings import TabPFNEmbeddingExtractor
    extractor = TabPFNEmbeddingExtractor()

    # Extract embeddings using train as context, all data as query
    # TabPFN requires context (X_context, y_context) and query (X_query)
    extractor.load_model(task="classification")
    result = extractor.extract_embeddings(
        X_context=X_train, y_context=y_train,
        X_query=X  # Query on all data
    )
    embeddings = result.embeddings
    print(f"  Embeddings shape: {embeddings.shape}")

    # Split embeddings based on original train/test split
    # (We queried on full X, so first 8000 are train, last 2000 are test)
    train_embeddings = embeddings[:len(X_train)]
    test_embeddings = embeddings[len(X_train):]

    # Train BatchTopK SAE
    print(f"\n{'='*80}")
    print("Training BatchTopK SAE")
    print(f"{'='*80}")

    config = SAEConfig(
        input_dim=embeddings.shape[1],
        hidden_dim=embeddings.shape[1] * 8,  # 8x expansion
        sparsity_type="batchtopk",
        topk=32,  # Target average L0
        learning_rate=1e-3,
        batch_size=256,
        n_epochs=100,
        use_ghost_grads=True,
    )

    print(f"Config:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Target avg L0: {config.topk}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.n_epochs}")

    # Train
    model, result = train_sae(train_embeddings, config, device="cuda", verbose=True)

    # Evaluate
    print(f"\n{'='*80}")
    print("Training Results")
    print(f"{'='*80}")
    print(f"Reconstruction loss: {result.reconstruction_loss:.6f}")
    print(f"Alive features: {result.alive_features}/{config.hidden_dim}")
    print(f"Dead features: {result.dead_features}")
    print(f"Mean active per sample: {result.mean_active_features:.1f}")

    # Richness metrics
    richness = measure_dictionary_richness(result, input_features=train_embeddings, sae_model=model)
    print(f"\nRichness Metrics:")
    print(f"  R²: {richness.get('explained_variance', 0):.4f}")
    print(f"  L0 sparsity: {richness['l0_sparsity']:.1f}")
    print(f"  Dictionary diversity: {richness['dictionary_diversity']:.4f}")

    # Check learned threshold
    print(f"\nInference Threshold:")
    print(f"  Value: {model.inference_threshold.item():.6f}")
    print(f"  Updates: {model.batchtopk_n_updates.item()}")

    # Test adaptive sparsity on test set
    print(f"\n{'='*80}")
    print("Testing Adaptive Sparsity (Test Set)")
    print(f"{'='*80}")

    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_embeddings, dtype=torch.float32, device="cuda")
        # Center (same as training)
        x = x - x.mean(dim=0, keepdim=True)
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

    # Plot distribution
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
    ax.set_title('BatchTopK: Distribution of Active Features (Adult Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig_path = output_dir / "batchtopk_adult_distribution.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot: {fig_path}")
    plt.close(fig)

    # Verify adaptive behavior
    print(f"\n{'='*80}")
    print("Adaptive Sparsity Analysis")
    print(f"{'='*80}")

    # Check if sparsity varies meaningfully
    if active_per_sample.std() > 5.0:
        print(f"✓ Strong adaptive sparsity (std={active_per_sample.std():.1f})")
    elif active_per_sample.std() > 2.0:
        print(f"✓ Moderate adaptive sparsity (std={active_per_sample.std():.1f})")
    else:
        print(f"⚠ Weak adaptive sparsity (std={active_per_sample.std():.1f})")

    # Check if mean is close to target
    ratio = active_per_sample.mean() / config.topk
    print(f"\nMean vs Target ratio: {ratio:.2f}x")
    if 0.8 <= ratio <= 1.2:
        print(f"✓ Mean close to target ({active_per_sample.mean():.1f} vs {config.topk})")
    else:
        print(f"⚠ Mean differs from target ({active_per_sample.mean():.1f} vs {config.topk})")

    print(f"\n{'='*80}")
    print("BatchTopK Adult Test Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_batchtopk_on_adult()
