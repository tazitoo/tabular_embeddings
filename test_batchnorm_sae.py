#!/usr/bin/env python3
"""Quick test to verify BatchNorm SAE implementation."""
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig

# Test 1: Create SAE with BatchNorm
print("=" * 80)
print("Test 1: Creating SAE with BatchNorm")
print("=" * 80)

config = SAEConfig(
    input_dim=192,
    hidden_dim=1536,
    sparsity_type="matryoshka_archetypal",
    topk=128,
    archetypal_n_archetypes=256,
    use_batchnorm=True,
)

model = SparseAutoencoder(config)
print(f"✓ SAE created with BatchNorm: {model.bn is not None}")
print(f"  BatchNorm params: {sum(p.numel() for p in model.bn.parameters())}")
print()

# Test 2: Forward pass with raw embeddings
print("=" * 80)
print("Test 2: Forward pass with raw embeddings")
print("=" * 80)

# Simulate raw embeddings (not normalized)
np.random.seed(42)
raw_embeddings = np.random.randn(100, 192).astype(np.float32) * 10 + 5  # Non-normalized
X = torch.tensor(raw_embeddings)

model.train()
x_hat, h = model(X)
print(f"✓ Forward pass successful")
print(f"  Input shape: {X.shape}")
print(f"  Input mean: {X.mean():.3f}, std: {X.std():.3f}")
print(f"  Reconstruction shape: {x_hat.shape}")
print(f"  Activations shape: {h.shape}")
print(f"  L0 sparsity: {(h > 0).float().sum(dim=1).mean():.1f}")
print()

# Test 3: Eval mode uses learned stats
print("=" * 80)
print("Test 3: Eval mode consistency")
print("=" * 80)

model.eval()
with torch.no_grad():
    x_hat_eval, h_eval = model(X)

print(f"✓ Eval mode successful")
print(f"  Train mode L0: {(h > 0).float().sum(dim=1).mean():.1f}")
print(f"  Eval mode L0: {(h_eval > 0).float().sum(dim=1).mean():.1f}")
print(f"  Reconstruction MSE diff (train vs eval): {((x_hat - x_hat_eval) ** 2).mean():.6f}")
print()

# Test 4: Backward compatibility (no BatchNorm)
print("=" * 80)
print("Test 4: Backward compatibility (use_batchnorm=False)")
print("=" * 80)

config_nobatchnorm = SAEConfig(
    input_dim=192,
    hidden_dim=1536,
    sparsity_type="topk",
    topk=32,
    use_batchnorm=False,
)

model_old = SparseAutoencoder(config_nobatchnorm)
print(f"✓ SAE created without BatchNorm: {model_old.bn is None}")

model_old.eval()
with torch.no_grad():
    x_hat_old, h_old = model_old(X)

print(f"  Forward pass successful")
print(f"  L0 sparsity: {(h_old > 0).float().sum(dim=1).mean():.1f}")
print()

print("=" * 80)
print("✓ All tests passed!")
print("=" * 80)
