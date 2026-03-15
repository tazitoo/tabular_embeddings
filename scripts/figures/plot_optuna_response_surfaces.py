#!/usr/bin/env python3
"""
Plot response surface from Optuna hyperparameter sweep.

Shows how R², stability, and composite score vary across the
hyperparameter space to diagnose if S5 harm is due to poor tuning
or a fundamental architecture issue.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import sae_sweep_dir


def plot_response_surfaces(study_path: Path, study_name: str, output_dir: Path):
    """Generate response surface plots from Optuna study."""

    # Load study
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{study_path}",
    )

    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    print(f"Loaded {len(trials)} completed trials")

    if len(trials) == 0:
        print("No completed trials found")
        return

    # Extract data
    params = {}
    metrics = {
        'composite_score': [],
        'r2': [],
        'stability': [],
    }

    for trial in trials:
        for key in trial.params:
            if key not in params:
                params[key] = []
            params[key].append(trial.params[key])

        metrics['composite_score'].append(trial.value)
        for key in ['r2', 'stability']:
            if key in trial.user_attrs:
                metrics[key].append(trial.user_attrs[key])

    # Convert to arrays
    for key in params:
        params[key] = np.array(params[key])
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    print(f"\nHyperparameters explored:")
    for key, values in params.items():
        if len(np.unique(values)) > 1:
            print(f"  {key}: {values.min():.4g} to {values.max():.4g} ({len(np.unique(values))} unique)")
        else:
            print(f"  {key}: {values[0]:.4g} (constant)")

    print(f"\nMetrics:")
    for key, values in metrics.items():
        if len(values) > 0:
            print(f"  {key}: {values.min():.3f} to {values.max():.3f}")

    # Find most important hyperparameters (those with variance)
    important_params = []
    for key, values in params.items():
        if len(np.unique(values)) > 5:  # At least 5 unique values
            important_params.append(key)

    print(f"\nImportant hyperparameters (varied): {important_params}")

    # Plot 1: Pairwise response surfaces for top metrics
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot composite score vs key hyperparameters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_idx = 0
    for param in important_params[:6]:
        ax = axes[plot_idx]

        # Scatter plot with color = composite score
        scatter = ax.scatter(
            params[param],
            metrics['r2'],
            c=metrics['composite_score'],
            s=50,
            alpha=0.6,
            cmap='viridis',
        )

        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.set_title(f'R² vs {param}', fontsize=11)
        ax.grid(alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label='Composite Score')

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, 6):
        axes[idx].axis('off')

    fig.tight_layout()
    fig_path = output_dir / "response_surface_r2_vs_params.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close(fig)

    # Plot 2: Composite score vs R² and Stability
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax.scatter(
        metrics['r2'],
        metrics['stability'],
        c=metrics['composite_score'],
        s=100,
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5,
    )

    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('Stability', fontsize=12)
    ax.set_title('Hyperparameter Search: R² vs Stability', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Mark best trial
    best_idx = np.argmax(metrics['composite_score'])
    ax.scatter(
        metrics['r2'][best_idx],
        metrics['stability'][best_idx],
        s=300,
        marker='*',
        c='red',
        edgecolors='darkred',
        linewidth=2,
        label='Best trial',
        zorder=10,
    )

    ax.legend(fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Score (0.4×R² + 0.6×Stability)', fontsize=10)

    fig.tight_layout()
    fig_path = output_dir / "response_surface_r2_vs_stability.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)

    # Plot 3: Distribution of key hyperparameters
    if len(important_params) > 0:
        n_params = min(len(important_params), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, param in enumerate(important_params[:6]):
            ax = axes[idx]

            # Histogram with color by composite score
            bins = min(20, len(np.unique(params[param])))

            # Sort trials by composite score
            sorted_idx = np.argsort(metrics['composite_score'])

            # Plot bottom 50% and top 50%
            mid = len(sorted_idx) // 2
            bottom_half = sorted_idx[:mid]
            top_half = sorted_idx[mid:]

            ax.hist(params[param][bottom_half], bins=bins, alpha=0.5,
                   label='Bottom 50%', color='red', edgecolor='black')
            ax.hist(params[param][top_half], bins=bins, alpha=0.5,
                   label='Top 50%', color='green', edgecolor='black')

            ax.set_xlabel(param, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Distribution: {param}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')

        # Hide unused
        for idx in range(n_params, 6):
            axes[idx].axis('off')

        fig.tight_layout()
        fig_path = output_dir / "hyperparameter_distributions.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close(fig)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_trial = study.best_trial
    print(f"\nBest trial (#{best_trial.number}):")
    print(f"  Composite score: {best_trial.value:.4f}")
    print(f"  R²: {best_trial.user_attrs.get('r2', 'N/A'):.4f}")
    print(f"  Stability: {best_trial.user_attrs.get('stability', 'N/A'):.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Check if there's a Pareto frontier
    pareto_trials = []
    for trial in trials:
        r2 = trial.user_attrs.get('r2', 0)
        stab = trial.user_attrs.get('stability', 0)

        # Check if any other trial dominates this one
        dominated = False
        for other in trials:
            other_r2 = other.user_attrs.get('r2', 0)
            other_stab = other.user_attrs.get('stability', 0)

            if other_r2 >= r2 and other_stab >= stab and (other_r2 > r2 or other_stab > stab):
                dominated = True
                break

        if not dominated:
            pareto_trials.append(trial)

    print(f"\nPareto frontier: {len(pareto_trials)} trials")
    print("(Trials not dominated in both R² and stability)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tabicl_layer10", help="Model SAE directory")
    args = parser.parse_args()

    study_path = sae_sweep_dir() / args.model / f"{args.model}_matryoshka_archetypal.db"

    if not study_path.exists():
        print(f"Error: Study not found: {study_path}")
        return 1

    study_name = f"{args.model}_matryoshka_archetypal"
    output_dir = PROJECT_ROOT / "output" / "sae_response_surfaces" / args.model

    print("=" * 60)
    print(f"Optuna Response Surface Analysis: {args.model}")
    print("=" * 60)

    plot_response_surfaces(study_path, study_name, output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
