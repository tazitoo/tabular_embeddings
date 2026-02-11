"""Create 2D response surface plots for AuxK sweep."""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# Load trial data
with open('/tmp/auxk_trials.json') as f:
    trials = json.load(f)

# Filter to complete AuxK trials only
auxk_trials = [t for t in trials
               if t.get('total_loss', 0) > 0
               and t.get('aux_loss_type') == 'auxk'
               and t.get('r2', -999) > -100]  # Filter extreme outliers

print(f"Loaded {len(auxk_trials)} AuxK trials with valid metrics")

if len(auxk_trials) < 3:
    print("Not enough trials for surface plots")
    exit(1)

# Extract hyperparameters and metrics
hp_names = ['aux_loss_alpha', 'aux_warmup', 'learning_rate', 'sparsity_penalty', 'archetypal_temp', 'topk']
metrics = ['reconstruction_loss', 'sparsity_loss', 'aux_loss', 'total_loss', 'alive_features', 'l0_sparsity']

# Prepare data
data = {}
for hp in hp_names:
    data[hp] = np.array([t.get(hp, np.nan) for t in auxk_trials])

for metric in metrics:
    data[metric] = np.array([t.get(metric, np.nan) for t in auxk_trials])

# Define HP pairs to plot - focus on aux loss hyperparameters
hp_pairs = [
    ('aux_loss_alpha', 'aux_warmup'),
    ('aux_loss_alpha', 'learning_rate'),
    ('aux_loss_alpha', 'archetypal_temp'),
    ('aux_warmup', 'learning_rate'),
]

# Metrics to plot - all three loss components plus total and utilization
plot_metrics = [
    ('reconstruction_loss', 'Reconstruction Loss (MSE)', 'viridis'),
    ('sparsity_loss', 'Sparsity Loss', 'plasma'),
    ('aux_loss', 'Auxiliary Loss', 'cividis'),
    ('total_loss', 'Total Loss', 'inferno'),
    ('alive_features', 'Alive Features', 'RdYlGn'),
]

# Create figure
fig, axes = plt.subplots(len(plot_metrics), len(hp_pairs),
                        figsize=(5*len(hp_pairs), 4*len(plot_metrics)))

for col, (hp1, hp2) in enumerate(hp_pairs):
    for row, (metric, metric_label, cmap) in enumerate(plot_metrics):
        ax = axes[row, col] if len(plot_metrics) > 1 and len(hp_pairs) > 1 else axes

        # Get valid data points
        x = data[hp1]
        y = data[hp2]
        z = data[metric]

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(f'{metric_label}\n{hp1} vs {hp2}')
            continue

        # Create scatter plot with color
        scatter = ax.scatter(x, y, c=z, s=100, alpha=0.7, cmap=cmap,
                           edgecolors='black', linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(metric_label, rotation=270, labelpad=20)

        # Add value labels on points
        for xi, yi, zi in zip(x, y, z):
            if metric == 'alive_features':
                ax.annotate(f'{zi:.0f}', (xi, yi), fontsize=7,
                          ha='center', va='bottom', alpha=0.7)
            elif metric == 'r2':
                ax.annotate(f'{zi:.2f}', (xi, yi), fontsize=7,
                          ha='center', va='bottom', alpha=0.7)

        # Set labels and scale
        ax.set_xlabel(hp1, fontsize=10)
        ax.set_ylabel(hp2, fontsize=10)
        ax.set_xscale('log' if hp1 in ['aux_loss_alpha', 'learning_rate', 'sparsity_penalty'] else 'linear')
        ax.set_yscale('log' if hp2 in ['aux_loss_alpha', 'learning_rate', 'sparsity_penalty'] else 'linear')

        if row == 0:
            ax.set_title(f'{hp1} vs {hp2}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.text(-0.3, 0.5, metric_label, transform=ax.transAxes,
                   rotation=90, va='center', ha='center', fontsize=11, fontweight='bold')

        ax.grid(True, alpha=0.3)

plt.suptitle('AuxK Response Surfaces: Loss Components over Hyperparameter Pairs',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plt.savefig('output/auxk_response_surfaces_2d.pdf', dpi=300, bbox_inches='tight')
plt.savefig('output/auxk_response_surfaces_2d.png', dpi=150, bbox_inches='tight')
print(f"\nSaved 2D response surface plots to output/auxk_response_surfaces_2d.[pdf,png]")

# Print correlation matrix
print("\n" + "="*70)
print("HYPERPARAMETER CORRELATIONS WITH METRICS")
print("="*70)

for metric, metric_label, _ in plot_metrics:
    print(f"\n{metric_label}:")
    z = data[metric]
    mask = ~np.isnan(z)

    for hp in hp_names:
        x = data[hp]
        valid = mask & ~np.isnan(x)
        if np.sum(valid) > 2:
            corr = np.corrcoef(x[valid], z[valid])[0, 1]
            print(f"  {hp:20s}: {corr:+.3f}")

print("\n" + "="*70)
