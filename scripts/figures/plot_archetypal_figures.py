#!/usr/bin/env python3
"""
Generate Archetypal SAE paper-style figures (Figures 3 & 4).

Figure 3: Stability vs L2 Loss for different TFMs (panels) and SAE types (colors)
Figure 4: R² and Stability vs Sparsity for different δ values (single TFM)

Usage:
    python scripts/plot_archetypal_figures.py --figure 3
    python scripts/plot_archetypal_figures.py --figure 4 --model tabpfn
    python scripts/plot_archetypal_figures.py --all
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
    compare_dictionaries,
)


@dataclass
class SAERunResult:
    """Result from a single SAE training run."""
    sae_type: str
    l2_loss: float  # Reconstruction MSE
    r2_score: float  # Explained variance
    stability: float  # Cross-run dictionary similarity
    sparsity: float  # Fraction of zero activations (or L0 as %)
    l0_sparsity: float  # Mean active features per sample
    config: dict


def load_embeddings(model_name: str, dataset: str = "credit-g") -> Optional[np.ndarray]:
    """Load cached embeddings for a model."""
    # Try different path patterns
    paths = [
        PROJECT_ROOT / f"output/embeddings/tabarena/{model_name}/tabarena_{dataset}.npz",
        PROJECT_ROOT / f"output/embeddings/{model_name}_{dataset}.npz",
        PROJECT_ROOT / f"output/{model_name}_embeddings_{dataset}.npy",
    ]

    for path in paths:
        if path.exists():
            if path.suffix == ".npz":
                data = np.load(path, allow_pickle=True)
                embeddings = data['embeddings'].astype(np.float32)
            else:
                embeddings = np.load(path).astype(np.float32)

            # Normalize to unit variance per dimension for numerical stability
            std = embeddings.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0  # Avoid division by zero
            embeddings = embeddings / std

            return embeddings

    return None


def compute_stability(embeddings: np.ndarray, config: SAEConfig, n_runs: int = 3) -> Tuple[float, List[np.ndarray]]:
    """
    Train SAE multiple times and measure dictionary stability.

    Returns:
        (mean_stability, list_of_dictionaries)
    """
    dictionaries = []
    seeds = [123, 456, 789][:n_runs]

    for seed in seeds:
        torch.manual_seed(seed)
        model, result = train_sae(embeddings, config, verbose=False)
        dictionaries.append(result.dictionary)

    # Pairwise stability
    stabilities = []
    for i in range(len(dictionaries)):
        for j in range(i + 1, len(dictionaries)):
            comp = compare_dictionaries(dictionaries[i], dictionaries[j])
            stabilities.append(comp['mean_best_match_a'])

    return float(np.mean(stabilities)), dictionaries


def run_sae_experiment(
    embeddings: np.ndarray,
    sae_type: str,
    expansion: int = 4,
    sparsity_penalty: float = 1e-3,
    topk: int = 32,
    archetypal_n_archetypes: int = None,
    archetypal_relaxation: float = 0.0,
    n_epochs: int = 100,
    n_stability_runs: int = 2,
) -> SAERunResult:
    """Run a single SAE experiment and return metrics."""
    input_dim = embeddings.shape[1]
    hidden_dim = input_dim * expansion

    if archetypal_n_archetypes is None:
        archetypal_n_archetypes = min(1000, len(embeddings))

    config = SAEConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_penalty=sparsity_penalty,
        sparsity_type=sae_type if sae_type != "vanilla" else "l1",
        topk=topk,
        archetypal_n_archetypes=archetypal_n_archetypes,
        archetypal_simplex_temp=0.1,
        archetypal_relaxation=archetypal_relaxation,
        archetypal_use_centroids=True,
        use_aux_loss=(sae_type not in ["archetypal"]),
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=1e-2 if sae_type == "archetypal" else 1e-3,
    )

    # Get stability and one trained model
    stability, dictionaries = compute_stability(embeddings, config, n_runs=n_stability_runs)

    # Train once more for metrics
    torch.manual_seed(42)
    model, result = train_sae(embeddings, config, verbose=False)
    richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)

    return SAERunResult(
        sae_type=sae_type,
        l2_loss=result.reconstruction_loss,
        r2_score=richness.get('explained_variance', 0.0),
        stability=stability,
        sparsity=richness['sparsity'],
        l0_sparsity=richness['l0_sparsity'],
        config={
            'expansion': expansion,
            'sparsity_penalty': sparsity_penalty,
            'topk': topk,
            'archetypal_relaxation': archetypal_relaxation,
        }
    )


def generate_figure3(
    models: List[str] = None,
    dataset: str = "credit-g",
    output_dir: Path = None,
    n_runs_per_type: int = 3,
) -> plt.Figure:
    """
    Generate Figure 3: Stability vs L2 Loss scatter plots for different TFMs.

    Each panel is a different TFM, points colored by SAE type.
    """
    if models is None:
        models = ["tabpfn", "tabicl", "mitra", "hyperfast"]

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "archetypal_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAE types to test (matching paper's categories)
    sae_configs = [
        # Vanilla SAE (L1) - multiple sparsity levels
        {"type": "vanilla", "label": "Vanilla SAE", "color": "#6baed6", "configs": [
            {"sparsity_penalty": 1e-4}, {"sparsity_penalty": 5e-4},
            {"sparsity_penalty": 1e-3}, {"sparsity_penalty": 5e-3},
        ]},
        # TopK SAE - multiple k values
        {"type": "topk", "label": "Top-K SAE", "color": "#9e9ac8", "configs": [
            {"topk": 16}, {"topk": 32}, {"topk": 64}, {"topk": 128},
        ]},
        # Archetypal SAE - multiple centroid counts
        {"type": "archetypal", "label": "Archetypal SAE", "color": "#9467bd", "configs": [
            {"archetypal_n_archetypes": 50}, {"archetypal_n_archetypes": 100},
            {"archetypal_n_archetypes": 200}, {"archetypal_n_archetypes": 500},
        ]},
    ]

    # Collect results for each model
    all_results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        print('='*60)

        embeddings = load_embeddings(model_name, dataset)
        if embeddings is None:
            print(f"  No embeddings found for {model_name}, skipping")
            continue

        print(f"  Loaded embeddings: {embeddings.shape}")
        model_results = []

        for sae_group in sae_configs:
            sae_type = sae_group["type"]
            print(f"  Testing {sae_group['label']}...")

            for config in sae_group["configs"]:
                try:
                    result = run_sae_experiment(
                        embeddings,
                        sae_type=sae_type,
                        n_stability_runs=2,
                        n_epochs=100,
                        **config
                    )
                    result.sae_type = sae_group["label"]
                    model_results.append(result)
                    print(f"    {config}: L2={result.l2_loss:.4f}, Stability={result.stability:.4f}")
                except Exception as e:
                    print(f"    {config}: FAILED - {e}")

        all_results[model_name] = model_results

    # Create figure
    n_models = len([m for m in models if m in all_results])
    if n_models == 0:
        print("No models with data found!")
        return None

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    axes = axes.flatten()

    # Color map for SAE types
    color_map = {cfg["label"]: cfg["color"] for cfg in sae_configs}

    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]

        for sae_label in color_map.keys():
            type_results = [r for r in results if r.sae_type == sae_label]
            if not type_results:
                continue

            l2_losses = [r.l2_loss for r in type_results]
            stabilities = [r.stability for r in type_results]

            ax.scatter(l2_losses, stabilities, c=color_map[sae_label],
                      label=sae_label, s=50, alpha=0.8, edgecolors='white', linewidth=0.5)

        ax.set_xlabel("L2 Loss", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Stability", fontsize=11)
        ax.set_title(model_name.upper(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xscale('log')  # Log scale for L2 loss (varies widely)
        ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=len(sae_configs), frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save
    fig.savefig(output_dir / "figure3_stability_vs_l2.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "figure3_stability_vs_l2.pdf", bbox_inches='tight')
    print(f"\nSaved Figure 3 to {output_dir}")

    return fig


def generate_figure4(
    model_name: str = "tabpfn",
    dataset: str = "credit-g",
    output_dir: Path = None,
) -> plt.Figure:
    """
    Generate Figure 4: R² and Stability vs Sparsity for different δ values.

    Two panels: R² Score (left) and Stability (right) vs Sparsity percentage.
    Lines for different relaxation parameters δ.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "archetypal_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(model_name, dataset)
    if embeddings is None:
        print(f"No embeddings found for {model_name}")
        return None

    print(f"Loaded {model_name} embeddings: {embeddings.shape}")

    # Relaxation values to test
    delta_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    delta_colors = {
        0.0: "#1f77b4",    # Blue
        0.1: "#2ca02c",    # Green
        0.5: "#ff7f0e",    # Orange
        1.0: "#d62728",    # Red
        2.0: "#9467bd",    # Purple
    }

    # Sparsity levels via different L0 targets (topk values as proxy)
    # Higher topk = less sparse = lower sparsity %
    sparsity_configs = [
        {"topk": 16, "label": "99.0"},   # Very sparse
        {"topk": 32, "label": "97.5"},
        {"topk": 64, "label": "95.0"},
        {"topk": 128, "label": "92.5"},
        {"topk": 256, "label": "90.0"},
    ]

    # Collect results
    results_by_delta = {delta: {"sparsity": [], "r2": [], "stability": []}
                        for delta in delta_values}

    # Also get baseline (TopK SAE without archetypal constraint)
    baseline_results = {"sparsity": [], "r2": [], "stability": []}

    input_dim = embeddings.shape[1]
    hidden_dim = input_dim * 4

    print("\nRunning experiments...")

    # Baseline: TopK SAE
    print("  Baseline (TopK SAE)...")
    for sp_cfg in sparsity_configs:
        config = SAEConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_penalty=1e-3,
            sparsity_type="topk",
            topk=sp_cfg["topk"],
            use_aux_loss=True,
            n_epochs=100,
            batch_size=64,
        )

        stability, _ = compute_stability(embeddings, config, n_runs=2)
        torch.manual_seed(42)
        model, result = train_sae(embeddings, config, verbose=False)
        richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)

        # Compute actual sparsity percentage (100 - L0/hidden_dim * 100)
        actual_sparsity = 100 * (1 - richness['l0_sparsity'] / hidden_dim)

        baseline_results["sparsity"].append(actual_sparsity)
        baseline_results["r2"].append(richness.get('explained_variance', 0))
        baseline_results["stability"].append(stability)
        print(f"    topk={sp_cfg['topk']}: sparsity={actual_sparsity:.1f}%, R²={richness.get('explained_variance', 0):.4f}, stability={stability:.4f}")

    # Archetypal with different δ values
    # Use TopK for sparsity control (same as baseline) + archetypal dictionary constraint
    for delta in delta_values:
        print(f"  Archetypal δ={delta}...")

        for sp_cfg in sparsity_configs:
            # Use same topk values to match baseline sparsity levels
            config = SAEConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                sparsity_penalty=1e-3,
                sparsity_type="archetypal",
                topk=sp_cfg["topk"],  # Control sparsity via topk
                archetypal_n_archetypes=min(500, len(embeddings)),
                archetypal_simplex_temp=0.1,
                archetypal_relaxation=delta,
                archetypal_use_centroids=True,
                use_aux_loss=False,
                n_epochs=100,
                batch_size=64,
                learning_rate=1e-2,
            )

            try:
                stability, _ = compute_stability(embeddings, config, n_runs=2)
                torch.manual_seed(42)
                model, result = train_sae(embeddings, config, verbose=False)
                richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)

                actual_sparsity = 100 * (1 - richness['l0_sparsity'] / hidden_dim)

                results_by_delta[delta]["sparsity"].append(actual_sparsity)
                results_by_delta[delta]["r2"].append(richness.get('explained_variance', 0))
                results_by_delta[delta]["stability"].append(stability)
                print(f"    topk={sp_cfg['topk']}: sparsity={actual_sparsity:.1f}%, R²={richness.get('explained_variance', 0):.4f}, stability={stability:.4f}")
            except Exception as e:
                print(f"    topk={sp_cfg['topk']}: FAILED - {e}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot baseline
    ax1.plot(baseline_results["sparsity"], baseline_results["r2"],
             'ko-', label="Baseline", markersize=6, linewidth=2)
    ax2.plot(baseline_results["sparsity"], baseline_results["stability"],
             'ko-', label="Baseline", markersize=6, linewidth=2)

    # Plot each δ value
    for delta in delta_values:
        data = results_by_delta[delta]
        if not data["sparsity"]:
            continue

        # Sort by sparsity for clean lines
        sorted_idx = np.argsort(data["sparsity"])
        sparsity = [data["sparsity"][i] for i in sorted_idx]
        r2 = [data["r2"][i] for i in sorted_idx]
        stability = [data["stability"][i] for i in sorted_idx]

        label = f"δ={delta}"
        color = delta_colors[delta]

        ax1.plot(sparsity, r2, 'o-', color=color, label=label, markersize=5, linewidth=1.5)
        ax2.plot(sparsity, stability, 'o-', color=color, label=label, markersize=5, linewidth=1.5)

    # Format axes
    ax1.set_xlabel("Sparsity (%)", fontsize=11)
    ax1.set_ylabel("R² Score", fontsize=11)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher sparsity on left like paper

    ax2.set_xlabel("Sparsity (%)", fontsize=11)
    ax2.set_ylabel("Stability (Hungarian)", fontsize=11)
    ax2.set_ylim(0.3, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.suptitle(f"Impact of Relaxation Parameter (δ) - {model_name.upper()}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save
    fig.savefig(output_dir / f"figure4_relaxation_{model_name}.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f"figure4_relaxation_{model_name}.pdf", bbox_inches='tight')
    print(f"\nSaved Figure 4 to {output_dir}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate Archetypal SAE figures")
    parser.add_argument("--figure", type=int, choices=[3, 4], help="Which figure to generate")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--model", type=str, default="tabpfn", help="Model for Figure 4")
    parser.add_argument("--dataset", type=str, default="credit-g", help="Dataset to use")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["tabpfn", "tabicl", "mitra", "hyperfast"],
                       help="Models for Figure 3")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "output" / "archetypal_figures"

    if args.all or args.figure == 3:
        print("\n" + "="*60)
        print("GENERATING FIGURE 3: Stability vs L2 Loss")
        print("="*60)
        generate_figure3(models=args.models, dataset=args.dataset, output_dir=output_dir)

    if args.all or args.figure == 4:
        print("\n" + "="*60)
        print("GENERATING FIGURE 4: R²/Stability vs Sparsity")
        print("="*60)
        generate_figure4(model_name=args.model, dataset=args.dataset, output_dir=output_dir)

    if not args.all and args.figure is None:
        parser.print_help()


if __name__ == "__main__":
    main()
