#!/usr/bin/env python3
"""
Generate SAEBench-style Pareto frontier plots for SAE architectures.

Plots L0 vs R² and L0 vs Stability trade-offs across all models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SWEEP_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep"
OUTPUT_DIR = Path(__file__).parent  # Save outputs in scripts/pareto/

# Model display names
MODEL_NAMES = {
    'tabpfn': 'TabPFN',
    'tabicl_layer10': 'TabICL',
    'mitra_layer12': 'Mitra',
    'tabdpt_layer14': 'TabDPT',
    'carte_layer1': 'CARTE',
    'hyperfast_layer2': 'HyperFast',
    'tabula8b_layer21': 'Tabula-8B',
}

# Architecture display names
ARCH_NAMES = {
    'l1': 'L1',
    'topk': 'TopK',
    'matryoshka': 'Matryoshka',
    'archetypal': 'Archetypal',
    'matryoshka_archetypal': 'Mat-Arch',
    'matryoshka_batchtopk_archetypal': 'Mat-BatchTopK-Arch',
}


def load_sweep_results(model_name):
    """Load best configs for a model."""
    config_file = SWEEP_DIR / model_name / "best_configs.json"
    if not config_file.exists():
        return None
    with open(config_file) as f:
        return json.load(f)


def compute_pareto_frontier(points):
    """
    Compute Pareto frontier where higher is better for both dimensions.

    Args:
        points: List of (x, y) tuples

    Returns:
        Boolean array indicating Pareto-optimal points
    """
    points = np.array(points)
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Point j dominates point i if it's better in both dimensions
                if points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1]:
                    if points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1]:
                        is_pareto[i] = False
                        break

    return is_pareto


def plot_pareto_frontier_r2():
    """Generate Pareto frontier plot: L0 vs R²."""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Collect all models with results
    models = []
    for model_key in MODEL_NAMES.keys():
        results = load_sweep_results(model_key)
        if results:
            models.append((model_key, results))

    for idx, (model_key, results) in enumerate(models):
        if idx >= 8:  # Max 8 subplots
            break

        ax = axes[idx]
        model_name = MODEL_NAMES[model_key]

        # Collect points for each architecture
        points_by_arch = {}
        for arch_key, arch_data in results.items():
            if arch_key not in ARCH_NAMES:
                continue

            metrics = arch_data['metrics']
            l0 = metrics.get('l0_sparsity', 0.0)
            r2 = metrics.get('r2', 0.0)

            points_by_arch[arch_key] = (l0, r2)

        if not points_by_arch:
            continue

        # Convert to arrays for Pareto computation
        archs = list(points_by_arch.keys())
        points = np.array([points_by_arch[a] for a in archs])

        # Compute Pareto frontier (minimizing L0, maximizing R²)
        # Transform L0 to negative so both dimensions are "higher is better"
        transformed_points = np.column_stack((-points[:, 0], points[:, 1]))
        is_pareto = compute_pareto_frontier(transformed_points)

        # Plot all points
        colors = {'l1': 'red', 'topk': 'blue', 'matryoshka': 'green',
                  'archetypal': 'purple', 'matryoshka_archetypal': 'orange',
                  'matryoshka_batchtopk_archetypal': 'brown'}

        for i, arch in enumerate(archs):
            l0, r2 = points[i]
            color = colors.get(arch, 'gray')
            marker = 'o' if is_pareto[i] else 's'
            size = 100 if is_pareto[i] else 50
            label = ARCH_NAMES.get(arch, arch)
            ax.scatter(l0, r2, c=color, marker=marker, s=size, label=label, alpha=0.7, edgecolors='black', linewidths=0.5)

        # Draw Pareto frontier line
        pareto_points = points[is_pareto]
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k--', alpha=0.3, linewidth=1.5)

        # SAEBench optimal L0 range
        ax.axvspan(50, 150, alpha=0.1, color='green', label='SAEBench optimal')

        ax.set_xlabel('L0 Sparsity', fontsize=11)
        ax.set_ylabel('R² (Reconstruction)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models), 8):
        axes[idx].axis('off')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sae_pareto_r2.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated Pareto frontier (R²): {output_file}")
    plt.close()


def plot_pareto_frontier_stability():
    """Generate Pareto frontier plot: L0 vs Stability."""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Collect all models with results
    models = []
    for model_key in MODEL_NAMES.keys():
        results = load_sweep_results(model_key)
        if results:
            models.append((model_key, results))

    for idx, (model_key, results) in enumerate(models):
        if idx >= 8:
            break

        ax = axes[idx]
        model_name = MODEL_NAMES[model_key]

        # Collect points
        points_by_arch = {}
        for arch_key, arch_data in results.items():
            if arch_key not in ARCH_NAMES:
                continue

            metrics = arch_data['metrics']
            l0 = metrics.get('l0_sparsity', 0.0)
            stability = metrics.get('stability', 0.0)

            points_by_arch[arch_key] = (l0, stability)

        if not points_by_arch:
            continue

        archs = list(points_by_arch.keys())
        points = np.array([points_by_arch[a] for a in archs])

        # Compute Pareto frontier (minimizing L0, maximizing stability)
        transformed_points = np.column_stack((-points[:, 0], points[:, 1]))
        is_pareto = compute_pareto_frontier(transformed_points)

        # Plot
        colors = {'l1': 'red', 'topk': 'blue', 'matryoshka': 'green',
                  'archetypal': 'purple', 'matryoshka_archetypal': 'orange',
                  'matryoshka_batchtopk_archetypal': 'brown'}

        for i, arch in enumerate(archs):
            l0, stability = points[i]
            color = colors.get(arch, 'gray')
            marker = 'o' if is_pareto[i] else 's'
            size = 100 if is_pareto[i] else 50
            label = ARCH_NAMES.get(arch, arch)
            ax.scatter(l0, stability, c=color, marker=marker, s=size, label=label, alpha=0.7, edgecolors='black', linewidths=0.5)

        # Draw Pareto frontier
        pareto_points = points[is_pareto]
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k--', alpha=0.3, linewidth=1.5)

        # SAEBench optimal L0 range
        ax.axvspan(50, 150, alpha=0.1, color='green', label='SAEBench optimal')

        ax.set_xlabel('L0 Sparsity', fontsize=11)
        ax.set_ylabel('Stability ($s_n^{dec}$)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models), 8):
        axes[idx].axis('off')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sae_pareto_stability.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated Pareto frontier (Stability): {output_file}")
    plt.close()


def main():
    """Generate Pareto frontier plots."""
    print("Generating SAE Pareto frontier plots...")
    print()

    plot_pareto_frontier_r2()
    plot_pareto_frontier_stability()

    print()
    print(f"✓ Plots saved to: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  - sae_pareto_r2.pdf")
    print("  - sae_pareto_stability.pdf")


if __name__ == "__main__":
    main()
