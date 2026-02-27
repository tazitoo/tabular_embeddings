#!/usr/bin/env python3
"""
Quick test to verify BatchTopK implementation works correctly.
"""

import numpy as np
import torch
from analysis.sparse_autoencoder import SAEConfig, train_sae

def test_batchtopk():
    """Test that BatchTopK training runs and learns threshold."""
    print("=" * 60)
    print("Testing BatchTopK SAE")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    embeddings = np.random.randn(500, 64).astype(np.float32)

    # BatchTopK config
    config = SAEConfig(
        input_dim=64,
        hidden_dim=256,
        sparsity_type="batchtopk",
        topk=32,  # Average 32 features per sample
        learning_rate=1e-3,
        batch_size=128,
        n_epochs=20,
        use_ghost_grads=True,
    )

    print(f"\nTraining BatchTopK SAE...")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Target avg L0: {config.topk}")
    print(f"  Batch size: {config.batch_size}")

    # Train
    model, result = train_sae(embeddings, config, device="cpu", verbose=True)

    # Check results
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Reconstruction loss: {result.reconstruction_loss:.6f}")
    print(f"Alive features: {result.alive_features}/{config.hidden_dim}")
    print(f"Mean active per sample: {result.mean_active_features:.1f}")

    # Check inference threshold was learned
    print(f"\nInference threshold: {model.inference_threshold.item():.6f}")
    print(f"Threshold updates: {model.batchtopk_n_updates.item()}")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings[:10], dtype=torch.float32)
        h = model.encode(x)
        active_per_sample = (h > 0).sum(dim=1).float()
        print(f"\nInference mode (10 samples):")
        print(f"  Active features per sample: {active_per_sample.tolist()}")
        print(f"  Mean: {active_per_sample.mean():.1f}")
        print(f"  Range: {active_per_sample.min():.0f}-{active_per_sample.max():.0f}")

    # Verify adaptive sparsity
    if active_per_sample.std() > 1.0:
        print(f"\n✓ Adaptive sparsity confirmed (std={active_per_sample.std():.1f})")
    else:
        print(f"\n✗ Warning: Low sparsity variation (std={active_per_sample.std():.1f})")

    print(f"\n{'='*60}")
    print("BatchTopK test complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_batchtopk()
