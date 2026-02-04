#!/usr/bin/env python3
"""
Generate appendix figure for layer-wise CKA analysis.

Creates a publication-ready figure showing:
- Panel A: Representative heatmap (adult dataset)
- Panel B: CKA vs layer distance across all datasets
- Panel C: CKA from layer 0 to each layer (representation drift)

Usage:
    python scripts/4_results/plot_layerwise_appendix.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_cka_data(dataset_name: str) -> np.ndarray:
    """Load CKA matrix from .npz file."""
    path = OUTPUT_DIR / f"layerwise_cka_tabpfn_{dataset_name}.npz"
    data = np.load(path)
    return data['cka_matrix']


def compute_distance_cka(matrix: np.ndarray) -> tuple:
    """Compute average CKA by layer distance."""
    n = matrix.shape[0]
    max_dist = n - 1
    avg_cka = []
    std_cka = []

    for d in range(1, max_dist + 1):
        vals = []
        for i in range(n - d):
            vals.append(matrix[i, i + d])
        avg_cka.append(np.mean(vals))
        std_cka.append(np.std(vals))

    return np.array(avg_cka), np.array(std_cka)


def plot_appendix_figure():
    """Create the appendix figure."""
    # Load data
    datasets = {
        'synthetic': 'Synthetic',
        'adult': 'Adult',
        'SpeedDating': 'SpeedDating',
    }

    matrices = {}
    for key in datasets:
        try:
            matrices[key] = load_cka_data(key)
        except FileNotFoundError:
            print(f"Warning: {key} data not found, skipping")

    if not matrices:
        print("No data found!")
        return

    # Create figure with 3 panels
    fig = plt.figure(figsize=(12, 4))

    # Panel A: Heatmap (adult dataset)
    ax1 = fig.add_subplot(131)
    matrix = matrices.get('adult', list(matrices.values())[0])
    n_layers = matrix.shape[0]

    # Use every 4th label to avoid crowding
    tick_positions = list(range(0, n_layers, 4))
    tick_labels = [f'L{i}' for i in tick_positions]

    im = ax1.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Layer')
    ax1.set_title('(A) Layer-wise CKA Matrix\n(Adult dataset)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('CKA')

    # Panel B: CKA vs layer distance
    ax2 = fig.add_subplot(132)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for (key, label), color in zip(datasets.items(), colors):
        if key not in matrices:
            continue
        avg_cka, std_cka = compute_distance_cka(matrices[key])
        distances = np.arange(1, len(avg_cka) + 1)
        ax2.plot(distances, avg_cka, 'o-', label=label, color=color,
                 markersize=4, linewidth=1.5)
        ax2.fill_between(distances, avg_cka - std_cka, avg_cka + std_cka,
                         alpha=0.15, color=color)

    # Mark 2/3 depth
    two_thirds = int(n_layers * 2 / 3)
    ax2.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.5,
                label=f'2/3 depth ({two_thirds} layers)')

    ax2.set_xlabel('Layer Distance')
    ax2.set_ylabel('Average CKA')
    ax2.set_title('(B) CKA vs Layer Distance')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, n_layers)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Panel C: CKA from L0 to each layer (representation drift from input)
    ax3 = fig.add_subplot(133)

    for (key, label), color in zip(datasets.items(), colors):
        if key not in matrices:
            continue
        matrix = matrices[key]
        l0_cka = matrix[0, :]  # First row = CKA of L0 with all other layers
        layers = np.arange(n_layers)
        ax3.plot(layers, l0_cka, 'o-', label=label, color=color,
                 markersize=4, linewidth=1.5)

    # Mark 2/3 depth
    ax3.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.5,
                label=f'2/3 depth (L{two_thirds})')

    ax3.set_xlabel('Layer')
    ax3.set_ylabel('CKA with Layer 0')
    ax3.set_title('(C) Representation Drift from Input')
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, n_layers - 1)
    ax3.legend(loc='lower left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "layerwise_cka_appendix"
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def plot_compact_figure():
    """Create a more compact 2-panel version."""
    # Load data
    datasets = {
        'synthetic': 'Synthetic',
        'adult': 'Adult',
        'SpeedDating': 'SpeedDating',
    }

    matrices = {}
    for key in datasets:
        try:
            matrices[key] = load_cka_data(key)
        except FileNotFoundError:
            pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: CKA from L0 to each layer
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    n_layers = 24

    for (key, label), color in zip(datasets.items(), colors):
        if key not in matrices:
            continue
        matrix = matrices[key]
        n_layers = matrix.shape[0]
        l0_cka = matrix[0, :]
        layers = np.arange(n_layers)
        ax1.plot(layers, l0_cka, 'o-', label=label, color=color,
                 markersize=5, linewidth=2)

    two_thirds = int(n_layers * 2 / 3)
    ax1.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.7,
                linewidth=2, label=f'2/3 depth (L{two_thirds})')

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('CKA Similarity with Layer 0', fontsize=12)
    ax1.set_title('(A) Representation Drift Through Network', fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, n_layers - 1)
    ax1.legend(loc='lower left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # Panel B: CKA vs layer distance
    for (key, label), color in zip(datasets.items(), colors):
        if key not in matrices:
            continue
        avg_cka, _ = compute_distance_cka(matrices[key])
        distances = np.arange(1, len(avg_cka) + 1)
        ax2.plot(distances, avg_cka, 'o-', label=label, color=color,
                 markersize=5, linewidth=2)

    ax2.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.7,
                linewidth=2, label=f'2/3 depth')

    ax2.set_xlabel('Layer Distance', fontsize=12)
    ax2.set_ylabel('Average CKA Similarity', fontsize=12)
    ax2.set_title('(B) CKA Decay with Layer Separation', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, n_layers)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "layerwise_cka_appendix_compact"
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    plot_appendix_figure()
    plot_compact_figure()
