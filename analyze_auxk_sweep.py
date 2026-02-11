"""Analyze AuxK sweep results and create response surface plots."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load trial data
with open('/tmp/auxk_trials.json') as f:
    trials = json.load(f)

# Filter to trials with complete data (skip failed trials)
complete_trials = [t for t in trials if t.get('total_loss') is not None and t.get('total_loss', 0) > 0 and 'aux_loss_type' in t]

print(f"Loaded {len(complete_trials)} complete trials out of {len(trials)} total")

# Extract data
aux_types = [t['aux_loss_type'] for t in complete_trials]
alphas = [t.get('aux_loss_alpha', 0) for t in complete_trials]
total_losses = [t['total_loss'] for t in complete_trials]
recon_losses = [t.get('reconstruction_loss', np.nan) for t in complete_trials]
sparsity_losses = [t.get('sparsity_loss', np.nan) for t in complete_trials]
dead_fracs = [t.get('alive_features', 0) / 1536 for t in complete_trials]  # Convert to alive fraction

# Separate by aux_loss_type
none_mask = [t['aux_loss_type'] == 'none' for t in complete_trials]
auxk_mask = [t['aux_loss_type'] == 'auxk' for t in complete_trials]

none_trials = [t for t, m in zip(complete_trials, none_mask) if m]
auxk_trials = [t for t, m in zip(complete_trials, auxk_mask) if m]

print(f"\nBreakdown:")
print(f"  aux_loss_type='none': {len(none_trials)}")
print(f"  aux_loss_type='auxk': {len(auxk_trials)}")

# Create figure with subplots - 3 loss components + total + alive features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('AuxK Sweep: Loss Components vs Alpha', fontsize=16, y=0.995)

# Plot 1: Total Loss vs Alpha
ax = axes[0, 0]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    losses_auxk = [t['total_loss'] for t in auxk_trials]
    ax.scatter(alphas_auxk, losses_auxk, alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    losses_none = [t['total_loss'] for t in none_trials]
    ax.scatter([0]*len(losses_none), losses_none, alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('Total Loss')
ax.set_xscale('log')
ax.set_title('Total Loss (Objective)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reconstruction Loss vs Alpha
ax = axes[0, 1]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    recon_auxk = [t.get('reconstruction_loss', np.nan) for t in auxk_trials]
    valid = ~np.isnan(recon_auxk)
    if np.any(valid):
        ax.scatter(np.array(alphas_auxk)[valid], np.array(recon_auxk)[valid],
                  alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    recon_none = [t.get('reconstruction_loss', np.nan) for t in none_trials]
    valid = ~np.isnan(recon_none)
    if np.any(valid):
        ax.scatter([0.001]*np.sum(valid), np.array(recon_none)[valid],
                  alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('Reconstruction Loss (MSE)')
ax.set_xscale('log')
ax.set_title('Reconstruction Quality')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Sparsity Loss vs Alpha
ax = axes[0, 2]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    sparse_auxk = [t.get('sparsity_loss', np.nan) for t in auxk_trials]
    valid = ~np.isnan(sparse_auxk)
    if np.any(valid):
        ax.scatter(np.array(alphas_auxk)[valid], np.array(sparse_auxk)[valid],
                  alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    sparse_none = [t.get('sparsity_loss', np.nan) for t in none_trials]
    valid = ~np.isnan(sparse_none)
    if np.any(valid):
        ax.scatter([0.001]*np.sum(valid), np.array(sparse_none)[valid],
                  alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('Sparsity Loss')
ax.set_xscale('log')
ax.set_title('Sparsity Penalty')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Auxiliary Loss vs Alpha
ax = axes[1, 0]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    aux_auxk = [t.get('aux_loss', np.nan) for t in auxk_trials]
    valid = ~np.isnan(aux_auxk)
    if np.any(valid):
        ax.scatter(np.array(alphas_auxk)[valid], np.array(aux_auxk)[valid],
                  alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    # Aux loss should be 0 for none trials
    aux_none = [t.get('aux_loss', 0) for t in none_trials]
    ax.scatter([0.001]*len(aux_none), aux_none, alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('Auxiliary Loss')
ax.set_xscale('log')
ax.set_title('Dead Neuron Reconstruction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Alive Features vs Alpha
ax = axes[1, 1]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    alive_auxk = [t.get('alive_features', 0) for t in auxk_trials]
    ax.scatter(alphas_auxk, alive_auxk, alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    alive_none = [t.get('alive_features', 0) for t in none_trials]
    ax.scatter([0.001]*len(alive_none), alive_none, alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.axhline(y=1536*0.17, color='red', linestyle='--', alpha=0.5, label='Baseline (17% alive = 260)')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('Alive Features (out of 1536)')
ax.set_xscale('log')
ax.set_title('Feature Utilization')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: L0 Sparsity vs Alpha
ax = axes[1, 2]
if auxk_trials:
    alphas_auxk = [t['aux_loss_alpha'] for t in auxk_trials]
    l0_auxk = [t.get('l0_sparsity', np.nan) for t in auxk_trials]
    valid = ~np.isnan(l0_auxk)
    if np.any(valid):
        ax.scatter(np.array(alphas_auxk)[valid], np.array(l0_auxk)[valid],
                  alpha=0.6, s=80, label='AuxK', color='tab:blue')
if none_trials:
    l0_none = [t.get('l0_sparsity', np.nan) for t in none_trials]
    valid = ~np.isnan(l0_none)
    if np.any(valid):
        ax.scatter([0.001]*np.sum(valid), np.array(l0_none)[valid],
                  alpha=0.6, s=80, label='None', color='tab:orange', marker='s')
ax.set_xlabel('aux_loss_alpha (log scale)')
ax.set_ylabel('L0 Sparsity')
ax.set_xscale('log')
ax.set_title('Active Features per Sample')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/auxk_response_surface.pdf', dpi=300, bbox_inches='tight')
plt.savefig('output/auxk_response_surface.png', dpi=150, bbox_inches='tight')
print(f"\nSaved plots to output/auxk_response_surface.[pdf,png]")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\naux_loss_type='none' (baseline):")
if none_trials:
    print(f"  N trials: {len(none_trials)}")
    print(f"  Total loss: {np.mean([t['total_loss'] for t in none_trials]):.4f} ± {np.std([t['total_loss'] for t in none_trials]):.4f}")
    print(f"  Reconstruction loss: {np.mean([t.get('reconstruction_loss', np.nan) for t in none_trials]):.4f} ± {np.std([t.get('reconstruction_loss', np.nan) for t in none_trials]):.4f}")
    print(f"  Sparsity loss: {np.mean([t.get('sparsity_loss', np.nan) for t in none_trials]):.4f} ± {np.std([t.get('sparsity_loss', np.nan) for t in none_trials]):.4f}")
    print(f"  Auxiliary loss: {np.mean([t.get('aux_loss', 0) for t in none_trials]):.4f} (should be 0)")
    print(f"  Alive features: {np.mean([t.get('alive_features', 0) for t in none_trials]):.1f} ± {np.std([t.get('alive_features', 0) for t in none_trials]):.1f}")

print("\naux_loss_type='auxk':")
if auxk_trials:
    print(f"  N trials: {len(auxk_trials)}")
    alphas = [t['aux_loss_alpha'] for t in auxk_trials]
    warmups = [t.get('aux_warmup', 3) for t in auxk_trials]
    print(f"  Alpha range: {min(alphas):.4f} - {max(alphas):.4f}")
    print(f"  Warmup range: {min(warmups)} - {max(warmups)} epochs")
    print(f"  Total loss: {np.mean([t['total_loss'] for t in auxk_trials]):.4f} ± {np.std([t['total_loss'] for t in auxk_trials]):.4f}")
    recon_losses = [t.get('reconstruction_loss', np.nan) for t in auxk_trials if not np.isnan(t.get('reconstruction_loss', np.nan))]
    if recon_losses:
        print(f"  Reconstruction loss: {np.mean(recon_losses):.4f} ± {np.std(recon_losses):.4f}")
    sparse_losses = [t.get('sparsity_loss', np.nan) for t in auxk_trials if not np.isnan(t.get('sparsity_loss', np.nan))]
    if sparse_losses:
        print(f"  Sparsity loss: {np.mean(sparse_losses):.4f} ± {np.std(sparse_losses):.4f}")
    aux_losses = [t.get('aux_loss', np.nan) for t in auxk_trials if not np.isnan(t.get('aux_loss', np.nan))]
    if aux_losses:
        print(f"  Auxiliary loss: {np.mean(aux_losses):.4f} ± {np.std(aux_losses):.4f}")
    print(f"  Alive features: {np.mean([t.get('alive_features', 0) for t in auxk_trials]):.1f} ± {np.std([t.get('alive_features', 0) for t in auxk_trials]):.1f}")

    # Find best by different metrics
    print("\n  Best trials by metric:")
    best_alive_trial = max(auxk_trials, key=lambda t: t.get('alive_features', 0))
    print(f"    Most alive ({best_alive_trial.get('alive_features', 0)}): α={best_alive_trial['aux_loss_alpha']:.4f}, warmup={best_alive_trial.get('aux_warmup', 3)}, loss={best_alive_trial['total_loss']:.4f}")

    best_loss_trial = min(auxk_trials, key=lambda t: t['total_loss'])
    print(f"    Lowest loss ({best_loss_trial['total_loss']:.4f}): α={best_loss_trial['aux_loss_alpha']:.4f}, warmup={best_loss_trial.get('aux_warmup', 3)}, alive={best_loss_trial.get('alive_features', 0)}")

print("\n" + "="*60)
