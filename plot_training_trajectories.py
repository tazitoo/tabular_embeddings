"""Re-run best trials and plot training trajectories."""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.sparse_autoencoder import SAEConfig, train_sae
from scripts.sae_tabarena_sweep import pool_embeddings, get_tabarena_splits

# Load TabPFN embeddings using the same method as the sweep
print("Loading TabPFN embeddings...")
train_datasets, test_datasets = get_tabarena_splits("tabpfn")
embeddings, counts = pool_embeddings("tabpfn", train_datasets, max_per_dataset=500, normalize=True)
print(f"Loaded: {embeddings.shape}")
print(f"Datasets: {len(counts)}")

# Best baseline config (Trial #138)
print("\n=== Running Best Baseline (Trial #138) ===")
baseline_config = SAEConfig(
    input_dim=192,
    hidden_dim=192 * 8,
    sparsity_type="matryoshka_batchtopk_archetypal",
    sparsity_penalty=0.0036,
    learning_rate=0.0099,
    topk=16,
    archetypal_n_archetypes=512,
    archetypal_simplex_temp=0.431,
    archetypal_relaxation=0.241,
    n_epochs=100,
    batch_size=128,
    aux_loss_type="none",
    use_lr_schedule=True,
)

torch.manual_seed(42)
np.random.seed(42)
baseline_model, baseline_result = train_sae(embeddings, baseline_config, device="cuda", verbose=True)
print(f"  Final loss: {baseline_result.total_loss:.6f}, alive: {baseline_result.alive_features}")

# Best AuxK config (Trial #137)
print("\n=== Running Best AuxK (Trial #137) ===")
auxk_config = SAEConfig(
    input_dim=192,
    hidden_dim=192 * 8,
    sparsity_type="matryoshka_batchtopk_archetypal",
    sparsity_penalty=0.0057,
    learning_rate=0.0084,
    topk=16,
    archetypal_n_archetypes=512,
    archetypal_simplex_temp=0.360,
    archetypal_relaxation=0.336,
    n_epochs=100,
    batch_size=128,
    aux_loss_type="auxk",
    aux_loss_alpha=0.0018,
    aux_loss_warmup_epochs=10,
    use_lr_schedule=True,
)

torch.manual_seed(42)
np.random.seed(42)
auxk_model, auxk_result = train_sae(embeddings, auxk_config, device="cuda", verbose=True)
print(f"  Final loss: {auxk_result.total_loss:.6f}, alive: {auxk_result.alive_features}")

# Plot training trajectories
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training Trajectories: Baseline vs AuxK (with LR Schedule)', fontsize=16, y=0.995)

epochs = np.arange(100)

# Plot 1: Total Loss
ax = axes[0, 0]
ax.plot(epochs, baseline_result.training_history["total_loss"], label='Baseline (no aux)', linewidth=2)
ax.plot(epochs, auxk_result.training_history["total_loss"], label='AuxK (α=0.0018)', linewidth=2)
ax.axvline(10, color='red', linestyle='--', alpha=0.3, label='Aux warmup ends')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.set_title('Total Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reconstruction Loss
ax = axes[0, 1]
ax.plot(epochs, baseline_result.training_history["recon_loss"], label='Baseline', linewidth=2)
ax.plot(epochs, auxk_result.training_history["recon_loss"], label='AuxK', linewidth=2)
ax.axvline(10, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Loss')
ax.set_title('Reconstruction Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Sparsity Loss
ax = axes[0, 2]
ax.plot(epochs, baseline_result.training_history["sparsity_loss"], label='Baseline', linewidth=2)
ax.plot(epochs, auxk_result.training_history["sparsity_loss"], label='AuxK', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Sparsity Loss')
ax.set_title('Sparsity Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Auxiliary Loss
ax = axes[1, 0]
ax.plot(epochs, baseline_result.training_history["aux_loss"], label='Baseline (always 0)', linewidth=2)
ax.plot(epochs, auxk_result.training_history["aux_loss"], label='AuxK', linewidth=2)
ax.axvline(10, color='red', linestyle='--', alpha=0.3, label='Warmup ends')
ax.set_xlabel('Epoch')
ax.set_ylabel('Auxiliary Loss')
ax.set_title('Auxiliary Loss (AuxK)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Learning Rate Schedule
ax = axes[1, 1]
ax.plot(epochs, baseline_result.training_history["lr"], label='Baseline', linewidth=2)
ax.plot(epochs, auxk_result.training_history["lr"], label='AuxK', linewidth=2)
ax.axvline(5, color='orange', linestyle='--', alpha=0.3, label='Warmup ends (5%)')
ax.axvline(80, color='purple', linestyle='--', alpha=0.3, label='Decay starts (80%)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule (3-phase)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Loss Components Stacked
ax = axes[1, 2]
ax.plot(epochs, baseline_result.training_history["total_loss"], label='Baseline Total', linewidth=2, color='tab:blue')
auxk_total = np.array(auxk_result.training_history["recon_loss"]) + \
             np.array(auxk_result.training_history["sparsity_loss"]) + \
             np.array(auxk_result.training_history["aux_loss"])
ax.plot(epochs, auxk_total, label='AuxK Total', linewidth=2, color='tab:orange')
ax.fill_between(epochs, 0, auxk_result.training_history["recon_loss"], 
                 alpha=0.3, label='Recon', color='tab:red')
ax.fill_between(epochs, auxk_result.training_history["recon_loss"], 
                 auxk_total, alpha=0.3, label='Sparse+Aux', color='tab:green')
ax.axvline(10, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss Components (AuxK)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/training_trajectories.pdf', dpi=300, bbox_inches='tight')
plt.savefig('output/training_trajectories.png', dpi=150, bbox_inches='tight')
print(f"\nSaved plots to output/training_trajectories.[pdf,png]")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nBaseline (Trial #138):")
print(f"  Final loss: {baseline_result.total_loss:.6f}")
print(f"  Recon: {baseline_result.reconstruction_loss:.6f}")
print(f"  Sparsity: {baseline_result.sparsity_loss:.6f}")
print(f"  Aux: {baseline_result.aux_loss:.6f}")
print(f"  Alive features: {baseline_result.alive_features}/1536")

print(f"\nAuxK (Trial #137):")
print(f"  Final loss: {auxk_result.total_loss:.6f}")
print(f"  Recon: {auxk_result.reconstruction_loss:.6f}")
print(f"  Sparsity: {auxk_result.sparsity_loss:.6f}")
print(f"  Aux: {auxk_result.aux_loss:.6f}")
print(f"  Alive features: {auxk_result.alive_features}/1536")

print(f"\nDifference:")
print(f"  Total loss: {auxk_result.total_loss / baseline_result.total_loss:.2f}x")
print(f"  Alive features: {auxk_result.alive_features - baseline_result.alive_features:+d}")
