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

from scripts._project_root import PROJECT_ROOT
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


def plot_single_model_appendix(model_name: str, matrix: np.ndarray, output_path: Path):
    """
    Generate 3-panel appendix figure for a single model.

    Panel A: Layer-wise CKA heatmap
    Panel B: CKA vs layer distance
    Panel C: Representation drift from first layer
    """
    n_layers = matrix.shape[0]

    fig = plt.figure(figsize=(14, 4.5))

    # Panel A: Heatmap
    ax1 = fig.add_subplot(131)

    # Determine tick spacing based on number of layers
    if n_layers <= 6:
        tick_step = 1
    elif n_layers <= 15:
        tick_step = 2
    else:
        tick_step = 4

    tick_positions = list(range(0, n_layers, tick_step))
    tick_labels = [f'L{i}' for i in tick_positions]

    im = ax1.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=9)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=9)
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Layer', fontsize=11)
    ax1.set_title(f'(A) {model_name} Layer-wise CKA', fontsize=12)

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('CKA', fontsize=10)

    # Panel B: CKA vs layer distance
    ax2 = fig.add_subplot(132)

    avg_cka, std_cka = compute_distance_cka(matrix)
    distances = np.arange(1, len(avg_cka) + 1)

    ax2.plot(distances, avg_cka, 'o-', color='#1f77b4', markersize=6, linewidth=2)
    ax2.fill_between(distances, avg_cka - std_cka, avg_cka + std_cka,
                     alpha=0.2, color='#1f77b4')

    # Mark 2/3 depth
    two_thirds = int(n_layers * 2 / 3)
    if two_thirds > 0 and two_thirds < n_layers:
        ax2.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.7,
                    linewidth=2, label=f'2/3 depth ({two_thirds} layers)')
        ax2.legend(loc='lower left', fontsize=9)

    ax2.set_xlabel('Layer Distance', fontsize=11)
    ax2.set_ylabel('Average CKA', fontsize=11)
    ax2.set_title('(B) CKA Decay with Distance', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, n_layers)
    ax2.grid(True, alpha=0.3)

    # Panel C: Representation drift from first layer
    ax3 = fig.add_subplot(133)

    l0_cka = matrix[0, :]
    layers = np.arange(n_layers)

    ax3.plot(layers, l0_cka, 'o-', color='#2ca02c', markersize=6, linewidth=2)

    # Mark 2/3 depth
    if two_thirds > 0 and two_thirds < n_layers:
        ax3.axvline(x=two_thirds, color='red', linestyle='--', alpha=0.7,
                    linewidth=2, label=f'2/3 depth (L{two_thirds})')
        ax3.legend(loc='lower left', fontsize=9)

    # Annotate first and last CKA values
    ax3.annotate(f'{l0_cka[0]:.2f}', (0, l0_cka[0]), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)
    ax3.annotate(f'{l0_cka[-1]:.2f}', (n_layers-1, l0_cka[-1]), textcoords="offset points",
                 xytext=(-20, 5), fontsize=9)

    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('CKA with Layer 0', fontsize=11)
    ax3.set_title('(C) Representation Drift from Input', fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(-0.5, n_layers - 0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def plot_model_evidence(model_key: str, display_name: str, batch_data: dict,
                        cka_matrix: np.ndarray = None, optimal_layer: int = None,
                        n_layers_config: int = None, output_path: Path = None):
    """
    Generate publication-quality 3-panel evidence figure for a single model.

    Uses ALL datasets from batch analysis (8-15 per model) to robustly justify
    the extraction layer choice.

    Panel A: Representative CKA heatmap (if available)
    Panel B: CKA drift from L0 for ALL datasets (individual + mean + critical layer)
    Panel C: Distribution of critical depths across datasets
    """
    import json

    datasets = list(batch_data.keys())
    n_datasets = len(datasets)

    # Get number of layers from first dataset
    first = batch_data[datasets[0]]
    n_layers = first['n_layers']

    # Collect profiles and critical depths
    profiles = []
    critical_depths = []
    critical_layers = []
    for ds_name, r in batch_data.items():
        profiles.append(np.array(r['l0_cka_profile']))
        critical_depths.append(r['critical_depth_frac'])
        critical_layers.append(r['critical_layer'])

    mean_profile = np.mean(profiles, axis=0)
    mean_critical = np.mean(critical_depths)
    std_critical = np.std(critical_depths)

    # Figure layout
    if cka_matrix is not None:
        fig = plt.figure(figsize=(15, 4.5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    else:
        fig = plt.figure(figsize=(11, 4.5))
        ax2 = fig.add_subplot(121)
        ax3 = fig.add_subplot(122)
        ax1 = None

    # --- Panel A: Heatmap (if available) ---
    if ax1 is not None and cka_matrix is not None:
        n_hm = cka_matrix.shape[0]
        if n_hm <= 6:
            tick_step = 1
        elif n_hm <= 15:
            tick_step = 2
        else:
            tick_step = 4

        tick_positions = list(range(0, n_hm, tick_step))
        tick_labels = [f'L{i}' for i in tick_positions]

        im = ax1.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, fontsize=8)
        ax1.set_yticks(tick_positions)
        ax1.set_yticklabels(tick_labels, fontsize=8)
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_ylabel('Layer', fontsize=11)
        ax1.set_title(f'(A) Layer-wise CKA', fontsize=12)

        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('CKA', fontsize=10)

        # Mark optimal layer
        if optimal_layer is not None and optimal_layer < n_hm:
            ax1.axhline(y=optimal_layer, color='lime', linestyle='--', linewidth=1.5, alpha=0.8)
            ax1.axvline(x=optimal_layer, color='lime', linestyle='--', linewidth=1.5, alpha=0.8)

    # --- Panel B: CKA drift from L0 (ALL datasets) ---
    layers = np.arange(n_layers)
    colors_palette = plt.cm.tab10(np.linspace(0, 1, min(n_datasets, 10)))

    # Individual dataset profiles (thin, colored)
    for i, (ds_name, profile) in enumerate(zip(datasets, profiles)):
        color = colors_palette[i % len(colors_palette)]
        ax2.plot(layers, profile, color=color, alpha=0.3, linewidth=1)

    # Mean profile (bold black)
    ax2.plot(layers, mean_profile, color='black', linewidth=3, label=f'Mean (n={n_datasets})',
             zorder=10)

    # CKA = 0.5 threshold
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.text(0.5, 0.52, 'CKA = 0.5', fontsize=8, color='gray', alpha=0.7)

    # Mark critical layer (mean)
    mean_crit_layer = np.mean(critical_layers)
    ax2.axvline(x=mean_crit_layer, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Critical (L{mean_crit_layer:.0f})')

    # Mark optimal extraction layer
    if optimal_layer is not None:
        ax2.axvline(x=optimal_layer, color='#2ca02c', linestyle='-', linewidth=2.5, alpha=0.9,
                    label=f'Extract (L{optimal_layer})')
        # Annotate CKA at extraction point
        cka_at_optimal = mean_profile[optimal_layer] if optimal_layer < len(mean_profile) else None
        if cka_at_optimal is not None:
            ax2.plot(optimal_layer, cka_at_optimal, 'o', color='#2ca02c', markersize=10,
                     zorder=11, markeredgecolor='black', markeredgewidth=1.5)
            # Position annotation to avoid overlap with data
            x_offset = max(n_layers * 0.08, 1.5)
            y_offset = 0.1 if cka_at_optimal < 0.85 else -0.12
            ax2.annotate(f'CKA={cka_at_optimal:.2f}',
                         xy=(optimal_layer, cka_at_optimal),
                         xytext=(optimal_layer + x_offset, cka_at_optimal + y_offset),
                         fontsize=10, fontweight='bold', color='#2ca02c',
                         arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))

    panel_b_label = '(B)' if ax1 is not None else '(A)'
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('CKA with Layer 0', fontsize=11)
    ax2.set_title(f'{panel_b_label} Representation Drift ({n_datasets} datasets)', fontsize=12)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlim(-0.5, n_layers - 0.5)
    ax2.legend(loc='lower left', fontsize=8, ncol=1 if n_datasets <= 8 else 2)
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Critical depth distribution ---
    panel_c_label = '(C)' if ax1 is not None else '(B)'

    # Strip plot of individual critical depths
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, n_datasets)
    ax3.scatter(jitter, critical_layers, c='steelblue', s=50, alpha=0.6, edgecolors='black',
                linewidths=0.5, zorder=5)

    # Mean + std
    ax3.errorbar(0, np.mean(critical_layers), yerr=np.std(critical_layers),
                 fmt='D', color='red', markersize=10, capsize=8, capthick=2,
                 linewidth=2, zorder=10, label=f'Mean: L{np.mean(critical_layers):.1f} '
                 f'({mean_critical:.0%} depth)')

    # Optimal extraction layer
    if optimal_layer is not None:
        ax3.axhline(y=optimal_layer, color='#2ca02c', linestyle='-', linewidth=2.5, alpha=0.9,
                    label=f'Extraction: L{optimal_layer}', zorder=8)

    # Add dataset labels for small n
    if n_datasets <= 15:
        for i, (ds_name, cl) in enumerate(zip(datasets, critical_layers)):
            short_name = ds_name[:15] + '...' if len(ds_name) > 15 else ds_name
            ax3.annotate(short_name, (jitter[i], cl),
                         xytext=(0.25, cl), fontsize=7, alpha=0.6,
                         arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))

    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, n_layers - 0.5)
    ax3.set_ylabel('Critical Layer', fontsize=11)
    ax3.set_title(f'{panel_c_label} Critical Depth Distribution', fontsize=12)
    ax3.set_xticks([])
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add depth fraction on right y-axis
    ax3_right = ax3.twinx()
    ax3_right.set_ylim(-0.5 / n_layers, (n_layers - 0.5) / n_layers)
    ax3_right.set_ylabel('Depth Fraction', fontsize=10)

    fig.suptitle(f'{display_name} — Layer Selection Evidence', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path.with_suffix('.png')} ({n_datasets} datasets)")
    plt.close()


def plot_all_model_appendix_figures():
    """Generate individual appendix figures for all models using batch analysis data."""
    import json
    import sys
    
    # Load optimal layer config
    try:
        from config import load_optimal_layers
        optimal_config = load_optimal_layers()
    except (ImportError, FileNotFoundError):
        optimal_config = {}

    # Model configurations
    models = {
        'tabpfn': {'name': 'TabPFN', 'heatmap_key': 'tabpfn_adult'},
        'tabicl': {'name': 'TabICL', 'heatmap_key': 'tabicl_adult'},
        'mitra': {'name': 'Mitra', 'heatmap_key': 'mitra_adult'},
        'tabdpt': {'name': 'TabDPT', 'heatmap_key': 'tabdpt_adult'},
        'carte': {'name': 'CARTE', 'heatmap_key': 'carte_SpeedDating'},
        'hyperfast': {'name': 'HyperFast', 'heatmap_key': 'hyperfast_adult'},
        'tabula8b': {'name': 'Tabula-8B', 'heatmap_key': None},
    }

    for model_key, config in models.items():
        # Load batch analysis data
        json_path = OUTPUT_DIR / f"layerwise_depth_analysis_{model_key}.json"
        if not json_path.exists():
            print(f"Skipping {config['name']}: batch analysis not found")
            continue

        with open(json_path) as f:
            batch_data = json.load(f)

        # Load CKA matrix for heatmap (if available)
        cka_matrix = None
        if config['heatmap_key']:
            npz_path = OUTPUT_DIR / f"layerwise_cka_{config['heatmap_key']}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                cka_matrix = data['cka_matrix']

        # Get optimal layer from config
        optimal_layer = None
        if model_key in optimal_config:
            optimal_layer = optimal_config[model_key]['optimal_layer']

        output_path = OUTPUT_DIR / f"layerwise_cka_appendix_{model_key}"
        plot_model_evidence(
            model_key=model_key,
            display_name=config['name'],
            batch_data=batch_data,
            cka_matrix=cka_matrix,
            optimal_layer=optimal_layer,
            output_path=output_path,
        )


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


def plot_model_comparison():
    """Compare layer-wise CKA across different models."""
    # Load data for multiple models
    model_data = {}

    # TabPFN
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_tabpfn_adult.npz")
        model_data['TabPFN'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    # TabICL
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_tabicl_adult.npz")
        model_data['TabICL'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    # Mitra
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_mitra_adult.npz")
        model_data['Mitra'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    # HyperFast
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_hyperfast_adult.npz")
        model_data['HyperFast'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    # TabDPT
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_tabdpt_adult.npz")
        model_data['TabDPT'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    # CARTE (uses SpeedDating since adult has preprocessing issues)
    try:
        data = np.load(OUTPUT_DIR / "layerwise_cka_carte_SpeedDating.npz")
        model_data['CARTE'] = data['cka_matrix']
    except FileNotFoundError:
        pass

    if len(model_data) < 2:
        print("Need at least 2 models for comparison")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = {'TabPFN': '#1f77b4', 'TabICL': '#ff7f0e', 'Mitra': '#2ca02c', 'HyperFast': '#d62728',
               'TabDPT': '#9467bd', 'CARTE': '#8c564b'}
    markers = {'TabPFN': 'o', 'TabICL': 's', 'Mitra': '^', 'HyperFast': 'D',
               'TabDPT': 'v', 'CARTE': 'P'}

    # Panel A: Representation drift (CKA with layer 0)
    ax = axes[0]
    for model_name, matrix in model_data.items():
        n_layers = matrix.shape[0]
        l0_cka = matrix[0, :]
        # Normalize to relative depth (0 to 1)
        rel_depth = np.linspace(0, 1, n_layers)
        marker = markers.get(model_name, 'o')
        ax.plot(rel_depth, l0_cka, marker=marker, linestyle='-',
                label=f'{model_name} ({n_layers}L)',
                color=colors.get(model_name, 'gray'), markersize=5, linewidth=2)

    ax.axvline(x=2/3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='2/3 depth')
    ax.set_xlabel('Relative Depth (0=input, 1=output)', fontsize=11)
    ax.set_ylabel('CKA with First Layer', fontsize=11)
    ax.set_title('(A) Representation Drift', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: CKA decay with relative distance
    ax = axes[1]
    for model_name, matrix in model_data.items():
        n_layers = matrix.shape[0]
        avg_cka, _ = compute_distance_cka(matrix)
        # Normalize distance to relative (0 to 1)
        rel_dist = np.linspace(1/n_layers, 1, len(avg_cka))
        marker = markers.get(model_name, 'o')
        ax.plot(rel_dist, avg_cka, marker=marker, linestyle='-',
                label=f'{model_name}',
                color=colors.get(model_name, 'gray'), markersize=5, linewidth=2)

    ax.axvline(x=2/3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='2/3 depth')
    ax.set_xlabel('Relative Layer Distance', fontsize=11)
    ax.set_ylabel('Average CKA', fontsize=11)
    ax.set_title('(B) CKA Decay with Distance', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: First-to-last CKA comparison (bar chart)
    ax = axes[2]
    model_names = list(model_data.keys())
    first_last_cka = [model_data[m][0, -1] for m in model_names]
    two_thirds_cka = [model_data[m][0, int(model_data[m].shape[0] * 2/3)] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, first_last_cka, width, label='First vs Last', color='#d62728')
    bars2 = ax.bar(x + width/2, two_thirds_cka, width, label='First vs 2/3', color='#2ca02c')

    ax.set_ylabel('CKA Similarity', fontsize=11)
    ax.set_title('(C) Layer Similarity Summary', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "layerwise_cka_model_comparison"
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def plot_combined_all_models():
    """
    Create the 'one ring' figure: all 6 models in a single comprehensive visualization.

    Panel A: All CKA drift profiles overlaid (normalized depth)
    Panel B: Optimal depth summary with error bars
    """
    import json

    # Model configurations
    models = {
        'tabpfn': {'name': 'TabPFN', 'color': '#1f77b4', 'marker': 'o'},
        'tabicl': {'name': 'TabICL', 'color': '#ff7f0e', 'marker': 's'},
        'mitra': {'name': 'Mitra', 'color': '#2ca02c', 'marker': '^'},
        'hyperfast': {'name': 'HyperFast', 'color': '#d62728', 'marker': 'D'},
        'tabdpt': {'name': 'TabDPT', 'color': '#9467bd', 'marker': 'v'},
        'carte': {'name': 'CARTE', 'color': '#8c564b', 'marker': 'P'},
        'tabula8b': {'name': 'Tabula-8B', 'color': '#e377c2', 'marker': 'X'},
    }

    # Load aggregated results for each model
    all_results = {}
    for model_key, config in models.items():
        json_path = OUTPUT_DIR / f"layerwise_depth_analysis_{model_key}.json"
        if json_path.exists():
            with open(json_path) as f:
                all_results[model_key] = json.load(f)

    if not all_results:
        print("No results found. Run batch analysis first.")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: All CKA drift profiles overlaid
    ax = axes[0]

    for model_key, config in models.items():
        if model_key not in all_results:
            continue

        results = all_results[model_key]
        color = config['color']
        name = config['name']

        # Plot each dataset's profile with low alpha
        for dataset, r in results.items():
            profile = r['l0_cka_profile']
            n_layers = len(profile)
            x_norm = np.arange(n_layers) / (n_layers - 1) if n_layers > 1 else [0]
            ax.plot(x_norm, profile, color=color, alpha=0.15, linewidth=1)

        # Compute and plot mean profile
        # Interpolate all profiles to common x-axis
        x_common = np.linspace(0, 1, 50)
        interpolated = []
        for dataset, r in results.items():
            profile = np.array(r['l0_cka_profile'])
            n_layers = len(profile)
            x_orig = np.arange(n_layers) / (n_layers - 1) if n_layers > 1 else [0]
            interp = np.interp(x_common, x_orig, profile)
            interpolated.append(interp)

        if interpolated:
            mean_profile = np.mean(interpolated, axis=0)
            ax.plot(x_common, mean_profile, color=color, linewidth=2.5,
                    label=f'{name} (n={len(results)})', marker=config['marker'],
                    markevery=5, markersize=6)

    ax.axvline(x=2/3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='2/3 depth')
    ax.axvline(x=0.75, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='3/4 depth')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Normalized Depth (0=input, 1=output)', fontsize=12)
    ax.set_ylabel('CKA with Layer 0', fontsize=12)
    ax.set_title('(A) Representation Drift: All Models Across TabArena', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Optimal depth summary
    ax = axes[1]

    model_names = []
    depths_mean = []
    depths_std = []
    n_datasets = []
    colors_list = []

    for model_key, config in models.items():
        if model_key not in all_results:
            continue
        results = all_results[model_key]
        depths = [r['critical_depth_frac'] for r in results.values()]

        model_names.append(config['name'])
        depths_mean.append(np.mean(depths))
        depths_std.append(np.std(depths))
        n_datasets.append(len(depths))
        colors_list.append(config['color'])

    x = np.arange(len(model_names))
    bars = ax.bar(x, depths_mean, yerr=depths_std, capsize=5,
                  color=colors_list, edgecolor='black', linewidth=1, alpha=0.8)

    # Add reference lines
    ax.axhline(y=2/3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='2/3 depth (0.67)')
    ax.axhline(y=0.75, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='3/4 depth (0.75)')

    # Add value labels
    for i, (bar, mean, std, n) in enumerate(zip(bars, depths_mean, depths_std, n_datasets)):
        ax.annotate(f'{mean:.0%}\n(n={n})',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Critical Depth (fraction)', fontsize=12)
    ax.set_title('(B) Optimal Layer Depth by Model\n(where CKA with L0 < 0.5)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "layerwise_depth_all_models_combined"
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    plot_appendix_figure()
    plot_compact_figure()
    plot_model_comparison()
    plot_all_model_appendix_figures()
    plot_combined_all_models()
