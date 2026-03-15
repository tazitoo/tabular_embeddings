#!/usr/bin/env python3
"""
SAEBench-style Pareto frontier plots for SAE architecture comparison.

Generates multi-panel plots showing L0 sparsity vs R² trade-offs across
different SAE architectures, highlighting Pareto-optimal configurations.

Usage:
    # Plot all models
    python scripts/plot_sae_pareto_frontier.py

    # Plot specific models
    python scripts/plot_sae_pareto_frontier.py --models tabpfn mitra_layer12

    # Include test set metrics
    python scripts/plot_sae_pareto_frontier.py --include-test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SAEConfig
from scripts.compare_sae_cross_model import sae_sweep_dir


# Architecture display names and colors
ARCH_STYLES = {
    'l1': {'name': 'L1', 'color': '#e74c3c', 'marker': 'o', 'alpha': 0.6},
    'topk': {'name': 'TopK', 'color': '#3498db', 'marker': 's', 'alpha': 0.6},
    'matryoshka': {'name': 'Matryoshka', 'color': '#f39c12', 'marker': '^', 'alpha': 0.6},
    'archetypal': {'name': 'Archetypal', 'color': '#9b59b6', 'marker': 'D', 'alpha': 0.6},
    'matryoshka_archetypal': {'name': 'Mat-Arch', 'color': '#2ecc71', 'marker': '*', 'alpha': 0.8, 'size': 150},
    'batchtopk': {'name': 'BatchTopK', 'color': '#1abc9c', 'marker': 'v', 'alpha': 0.6},
    'batchtopk_archetypal': {'name': 'BatchTopK-Arch', 'color': '#16a085', 'marker': 'p', 'alpha': 0.6},
    'matryoshka_batchtopk_archetypal': {'name': 'Mat-BatchTopK-Arch', 'color': '#27ae60', 'marker': 'h', 'alpha': 0.7},
}


def load_sweep_results(model_name: str, base_dir: Path = None) -> Dict[str, List[Dict]]:
    """
    Load sweep results for a model.

    Returns:
        Dictionary mapping architecture type to list of trial results.
        Each trial has: {l0, r2, stability, params, ...}
    """
    if base_dir is None:
        base_dir = sae_sweep_dir()

    model_dir = base_dir / model_name
    if not model_dir.exists():
        print(f"Warning: No sweep results found for {model_name} at {model_dir}")
        return {}

    results = {}

    # Look for result files from the sweep
    for json_file in model_dir.glob("*_results.json"):
        arch_type = json_file.stem.replace('_results', '')

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract trials
            trials = []
            if 'trials' in data:
                for trial in data['trials']:
                    if trial.get('state') == 'COMPLETE' and 'value' in trial:
                        trials.append({
                            'l0': trial.get('l0_sparsity', trial.get('l0', 0)),
                            'r2': trial.get('r2', 0),
                            'stability': trial.get('stability', 0),
                            'loss': trial.get('value', float('inf')),
                            'params': trial.get('params', {}),
                        })

            if trials:
                results[arch_type] = trials
                print(f"  {model_name}/{arch_type}: {len(trials)} trials")

        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")

    # Also try loading from validated checkpoints
    for checkpoint_file in model_dir.glob("sae_*_validated.pt"):
        arch_type = checkpoint_file.stem.replace('sae_', '').replace('_validated', '')

        if arch_type in results:
            continue  # Already have full trial data

        try:
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            metrics = ckpt.get('metrics', {})

            if metrics:
                trial = {
                    'l0': metrics.get('l0_sparsity', metrics.get('l0', 0)),
                    'r2': metrics.get('r2', 0),
                    'stability': metrics.get('stability', 0),
                    'loss': metrics.get('loss', 0),
                    'params': ckpt.get('params', {}),
                }
                results[arch_type] = [trial]
                print(f"  {model_name}/{arch_type}: 1 checkpoint (best config)")

        except Exception as e:
            print(f"  Warning: Failed to load {checkpoint_file}: {e}")

    return results


def compute_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier for 2D points (x, y) where larger is better for both.

    Returns:
        Boolean mask indicating which points are on the frontier.
    """
    is_pareto = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if is_pareto[i]:
            # Point is dominated if another point is better in both dimensions
            is_pareto[is_pareto] = np.any(points[is_pareto] > point, axis=1)
            is_pareto[i] = True  # Keep current point

    return is_pareto


def plot_pareto_frontier(
    results: Dict[str, Dict[str, List[Dict]]],
    output_path: Path,
    title: str = "SAE Architecture Pareto Frontiers",
    include_test: bool = False,
):
    """
    Generate SAEBench-style Pareto frontier plot.

    Args:
        results: {model_name: {arch_type: [trials]}}
        output_path: Where to save plot
        title: Plot title
        include_test: Whether to include test set metrics
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, arch_results) in enumerate(sorted(results.items())):
        ax = axes[idx]

        # Collect all points for Pareto frontier computation
        all_points = []
        all_archs = []

        for arch_type, trials in arch_results.items():
            if arch_type not in ARCH_STYLES:
                continue

            style = ARCH_STYLES[arch_type]

            # Extract metrics
            l0_vals = [t['l0'] for t in trials]
            r2_vals = [t['r2'] for t in trials]

            if not l0_vals:
                continue

            # Plot points
            scatter_size = style.get('size', 80)
            ax.scatter(
                l0_vals, r2_vals,
                label=style['name'],
                color=style['color'],
                marker=style['marker'],
                s=scatter_size,
                alpha=style['alpha'],
                edgecolors='black',
                linewidths=0.5,
            )

            # Collect for Pareto computation
            for l0, r2 in zip(l0_vals, r2_vals):
                all_points.append([l0, r2])
                all_archs.append(arch_type)

        # Compute and plot Pareto frontier
        if all_points:
            points = np.array(all_points)

            # For Pareto frontier, we want to MAXIMIZE R² and MINIMIZE L0
            # So we negate L0 for the Pareto computation
            pareto_points = np.column_stack([-points[:, 0], points[:, 1]])
            pareto_mask = compute_pareto_frontier(pareto_points)

            # Sort Pareto points by L0 for connecting line
            pareto_idx = np.where(pareto_mask)[0]
            pareto_l0 = points[pareto_idx, 0]
            pareto_r2 = points[pareto_idx, 1]
            sort_idx = np.argsort(pareto_l0)

            # Plot Pareto frontier line
            ax.plot(
                pareto_l0[sort_idx], pareto_r2[sort_idx],
                'k--', alpha=0.3, linewidth=1, zorder=0,
                label='Pareto frontier'
            )

            # Highlight Pareto-optimal points
            ax.scatter(
                pareto_l0, pareto_r2,
                facecolors='none',
                edgecolors='black',
                s=200,
                linewidths=2,
                zorder=10,
            )

        # Styling
        ax.set_xlabel('L0 Sparsity (lower is better)', fontsize=11)
        ax.set_ylabel('R² (higher is better)', fontsize=11)
        ax.set_title(model_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)

        # Add optimal L0 region from SAEBench (50-150)
        ax.axvspan(50, 150, alpha=0.1, color='green', zorder=0)
        ax.text(
            100, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            'SAEBench\nOptimal\nL0: 50-150',
            ha='center', va='bottom', fontsize=8, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto frontier plot: {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved PNG: {png_path}")

    plt.close()


def plot_stability_frontier(
    results: Dict[str, Dict[str, List[Dict]]],
    output_path: Path,
    title: str = "SAE Stability vs Sparsity Trade-off",
):
    """
    Plot L0 vs Stability (s_n^dec) - our unique contribution.

    SAEBench doesn't measure stability, but we do!
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, arch_results) in enumerate(sorted(results.items())):
        ax = axes[idx]

        for arch_type, trials in arch_results.items():
            if arch_type not in ARCH_STYLES:
                continue

            style = ARCH_STYLES[arch_type]

            # Extract metrics
            l0_vals = [t['l0'] for t in trials]
            stability_vals = [t.get('stability', 0) for t in trials]

            if not l0_vals or not any(stability_vals):
                continue

            # Plot points
            scatter_size = style.get('size', 80)
            ax.scatter(
                l0_vals, stability_vals,
                label=style['name'],
                color=style['color'],
                marker=style['marker'],
                s=scatter_size,
                alpha=style['alpha'],
                edgecolors='black',
                linewidths=0.5,
            )

        # Styling
        ax.set_xlabel('L0 Sparsity', fontsize=11)
        ax.set_ylabel('Stability (s_n^dec)', fontsize=11)
        ax.set_title(model_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved stability plot: {output_path}")

    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_pareto_summary(results: Dict[str, Dict[str, List[Dict]]]):
    """Print summary of Pareto-optimal configurations."""
    print("\n" + "="*80)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("="*80)

    for model_name, arch_results in sorted(results.items()):
        print(f"\n{model_name.upper()}")
        print("-" * 80)

        # Collect all points
        all_points = []
        all_data = []

        for arch_type, trials in arch_results.items():
            if arch_type not in ARCH_STYLES:
                continue

            for trial in trials:
                all_points.append([trial['l0'], trial['r2']])
                all_data.append({
                    'arch': arch_type,
                    'l0': trial['l0'],
                    'r2': trial['r2'],
                    'stability': trial.get('stability', 0),
                    'params': trial.get('params', {}),
                })

        if not all_points:
            print("  No data available")
            continue

        # Compute Pareto frontier (minimize L0, maximize R²)
        points = np.array(all_points)
        pareto_points = np.column_stack([-points[:, 0], points[:, 1]])
        pareto_mask = compute_pareto_frontier(pareto_points)

        # Print Pareto-optimal configs
        pareto_configs = [all_data[i] for i in range(len(all_data)) if pareto_mask[i]]
        pareto_configs.sort(key=lambda x: x['l0'])

        for cfg in pareto_configs:
            arch_name = ARCH_STYLES[cfg['arch']]['name']
            print(f"  {arch_name:20s} L0={cfg['l0']:>6.1f}  R²={cfg['r2']:.4f}  "
                  f"Stability={cfg['stability']:.4f}")

            # Print key hyperparameters
            params = cfg['params']
            if 'expansion' in params:
                print(f"    → expansion={params['expansion']}, topk={params.get('topk', '-')}, "
                      f"lr={params.get('learning_rate', 0):.2e}")


def main():
    parser = argparse.ArgumentParser(description='Generate SAEBench-style Pareto frontier plots')
    parser.add_argument('--models', nargs='+',
                       help='Models to plot (default: all available)')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'output' / 'figures',
                       help='Output directory for plots')
    parser.add_argument('--include-test', action='store_true',
                       help='Include test set metrics')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available models
    sweep_dir = sae_sweep_dir()
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    available_models = [d.name for d in sweep_dir.iterdir() if d.is_dir()]

    if args.models:
        models_to_plot = [m for m in args.models if m in available_models]
        if not models_to_plot:
            print(f"Error: None of the specified models found: {args.models}")
            print(f"Available models: {available_models}")
            return
    else:
        models_to_plot = available_models

    print(f"Loading sweep results for {len(models_to_plot)} models...")

    # Load results
    all_results = {}
    for model_name in sorted(models_to_plot):
        print(f"\n{model_name}:")
        results = load_sweep_results(model_name)
        if results:
            all_results[model_name] = results

    if not all_results:
        print("\nNo sweep results found!")
        return

    print(f"\nLoaded results for {len(all_results)} models")

    # Generate plots
    print("\nGenerating Pareto frontier plots...")

    # Main Pareto plot: L0 vs R²
    plot_pareto_frontier(
        all_results,
        args.output_dir / 'sae_pareto_frontier.pdf',
        title='SAE Architecture Pareto Frontiers (L0 vs R²)',
    )

    # Stability plot: L0 vs Stability
    plot_stability_frontier(
        all_results,
        args.output_dir / 'sae_stability_frontier.pdf',
        title='SAE Stability Trade-off (L0 vs s_n^dec)',
    )

    # Print summary
    print_pareto_summary(all_results)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Plots saved to: {args.output_dir}")
    print("\nKey findings:")
    print("  - Check if Matryoshka-Archetypal advances the Pareto frontier")
    print("  - Verify optimal L0 range (SAEBench suggests 50-150)")
    print("  - Compare stability across architectures")


if __name__ == '__main__':
    main()
