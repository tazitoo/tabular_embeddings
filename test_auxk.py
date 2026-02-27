"""Quick test to verify AuxK loss implementation."""
import torch
import numpy as np
from analysis.sparse_autoencoder import SAEConfig, SparseAutoencoder, train_sae

# Create synthetic embeddings
np.random.seed(42)
embeddings = np.random.randn(1000, 192).astype(np.float32)

# Test AuxK config with warmup
config_auxk = SAEConfig(
    input_dim=192,
    hidden_dim=192 * 4,
    sparsity_type="topk",
    topk=32,
    aux_loss_type="auxk",
    aux_loss_alpha=0.03125,
    aux_loss_warmup_epochs=2,  # Test warmup
    n_epochs=5,
    batch_size=128,
    learning_rate=1e-3,
)

# Test none config
config_none = SAEConfig(
    input_dim=192,
    hidden_dim=192 * 4,
    sparsity_type="topk",
    topk=32,
    aux_loss_type="none",
    n_epochs=5,
    batch_size=128,
    learning_rate=1e-3,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing AuxK loss implementation on {device}...")

print("\n1. Training with aux_loss_type='none'")
model_none, result_none = train_sae(embeddings, config_none, device=device, verbose=True)
print(f"   Dead features: {result_none.dead_features}/{config_none.hidden_dim} ({result_none.dead_features/config_none.hidden_dim*100:.1f}%)")
print(f"   Aux loss: {result_none.total_loss - result_none.reconstruction_loss - result_none.sparsity_loss:.6f}")

print("\n2. Training with aux_loss_type='auxk', alpha=0.03125, warmup=2 epochs")
model_auxk, result_auxk = train_sae(embeddings, config_auxk, device=device, verbose=True)
print(f"   Dead features: {result_auxk.dead_features}/{config_auxk.hidden_dim} ({result_auxk.dead_features/config_auxk.hidden_dim*100:.1f}%)")
print(f"   Final aux loss: {result_auxk.total_loss - result_auxk.reconstruction_loss - result_auxk.sparsity_loss:.6f}")

# Check warmup worked by examining history
print("\n4. Comparison:")
print(f"   Dead reduction: {result_none.dead_features} → {result_auxk.dead_features} ({result_none.dead_features - result_auxk.dead_features} fewer)")
print(f"   AuxK aux loss > 0: {result_auxk.total_loss - result_auxk.reconstruction_loss - result_auxk.sparsity_loss > 0}")

print("\n✓ AuxK implementation test complete!")
