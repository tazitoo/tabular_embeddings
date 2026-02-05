#!/usr/bin/env python3
"""
SAE Pareto Analysis: L0 vs Reconstruction Trade-off

Implements the methodology from "How to find the right satisficing satisficing L0 for your SAE"
(arXiv:2508.16560) to identify the "correct" L0 sparsity level.

Key metric: s_n^dec (Nth decoder projection score)
- When L0 is correct, most latents have near-zero projection on arbitrary inputs
- Minimize s_n^dec across L0 values to find optimal sparsity

Usage:
    python scripts/sae_pareto_analysis.py --study-name sae_tabpfn_adult_v3 --model tabpfn
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def compute_decoder_projection_score(
    sae_model,
    embeddings: np.ndarray,
    n_fraction: float = 0.5,
    n_samples: int = 1000,
) -> float:
    """
    Compute s_n^dec (Nth decoder projection score) from arXiv:2508.16560.

    When L0 is correct, most latents should have near-zero projection on
    arbitrary training inputs.

    Args:
        sae_model: Trained SparseAutoencoder
        embeddings: (n_samples, input_dim) training data
        n_fraction: Fraction of latents to use for score (default 0.5 = median)
        n_samples: Number of samples to use for computation

    Returns:
        s_n^dec: The nth highest decoder projection (lower is better)
    """
    import torch

    device = next(sae_model.parameters()).device
    sae_model.eval()

    # Subsample if needed
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]

    # Center data (same as training)
    x_mean = embeddings.mean(axis=0, keepdims=True)
    x_centered = embeddings - x_mean

    with torch.no_grad():
        x = torch.tensor(x_centered, dtype=torch.float32, device=device)

        # Get decoder weights and bias
        if sae_model.W_dec is not None:
            W_dec = sae_model.W_dec  # (input_dim, hidden_dim)
        else:
            # Tied weights: decoder = encoder.T
            W_dec = sae_model.W_enc.T  # (input_dim, hidden_dim)

        b_dec = sae_model.b_dec  # (input_dim,)

        # Compute decoder projections: Z = (x - b_dec) @ W_dec
        # Shape: (n_samples, hidden_dim)
        Z = (x - b_dec) @ W_dec
        Z = Z.cpu().numpy()

    # Flatten and sort in descending order
    Z_flat = Z.flatten()
    Z_sorted = np.sort(Z_flat)[::-1]  # Descending

    # Select nth highest projection
    n = int(n_fraction * len(Z_sorted))
    s_n_dec = Z_sorted[n]

    return float(s_n_dec)


def compute_pareto_frontier(
    l0_values: np.ndarray,
    r2_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Pareto frontier for L0 vs R² (we want LOW L0 and HIGH R²).

    Returns:
        pareto_l0, pareto_r2, pareto_idx: Pareto-optimal points and their indices
    """
    n = len(l0_values)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if: j has lower L0 AND higher R²
                if l0_values[j] <= l0_values[i] and r2_values[j] >= r2_values[i]:
                    if l0_values[j] < l0_values[i] or r2_values[j] > r2_values[i]:
                        is_pareto[i] = False
                        break

    pareto_idx = np.where(is_pareto)[0]
    # Sort by L0 for nice plotting
    sort_idx = np.argsort(l0_values[pareto_idx])
    pareto_idx = pareto_idx[sort_idx]

    return l0_values[pareto_idx], r2_values[pareto_idx], pareto_idx


def sweep_l0_for_optimal(
    embeddings: np.ndarray,
    l0_candidates: List[int] = None,
    expansion_factor: int = 8,
    device: str = "cuda",
) -> Dict:
    """
    Sweep L0 values to find optimal using s_n^dec metric.

    Following arXiv:2508.16560: train SAEs at different L0 targets,
    compute s_n^dec for each, find minimum.
    """
    from analysis.sparse_autoencoder import SAEConfig, train_sae

    if l0_candidates is None:
        # Sweep from very sparse to dense
        hidden_dim = embeddings.shape[1] * expansion_factor
        l0_candidates = [16, 32, 64, 128, 256, 512, 768, 1024]
        l0_candidates = [l for l in l0_candidates if l < hidden_dim]

    results = []

    for target_l0 in l0_candidates:
        print(f"\nTraining SAE with target L0 = {target_l0}")

        # Use TopK to enforce exact L0
        config = SAEConfig(
            input_dim=embeddings.shape[1],
            hidden_dim=embeddings.shape[1] * expansion_factor,
            sparsity_type="topk",
            topk=target_l0,
            n_epochs=100,
            learning_rate=1e-3,
            batch_size=128,
        )

        model, result = train_sae(embeddings, config, device=device, verbose=False)

        # Compute metrics
        s_n_dec = compute_decoder_projection_score(model, embeddings)

        # Compute R²
        import torch
        x_mean = embeddings.mean(axis=0, keepdims=True)
        x_centered = embeddings - x_mean
        with torch.no_grad():
            x = torch.tensor(x_centered, dtype=torch.float32, device=device)
            x_hat, _ = model(x)
            x_hat = x_hat.cpu().numpy()
        ss_res = np.sum((x_centered - x_hat) ** 2)
        ss_tot = np.sum(x_centered ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        # Actual L0
        with torch.no_grad():
            x = torch.tensor(x_centered, dtype=torch.float32, device=device)
            h = model.encode(x)
            actual_l0 = (h != 0).float().sum(dim=1).mean().item()

        results.append({
            "target_l0": int(target_l0),
            "actual_l0": float(actual_l0),
            "s_n_dec": float(s_n_dec),
            "r2": float(r2),
            "recon_loss": float(result.reconstruction_loss),
        })

        print(f"  Actual L0: {actual_l0:.1f}, s_n^dec: {s_n_dec:.4f}, R²: {r2:.4f}")

    # Find optimal L0 (minimum s_n_dec)
    s_n_values = [r["s_n_dec"] for r in results]
    optimal_idx = np.argmin(s_n_values)
    optimal_l0 = results[optimal_idx]["target_l0"]

    print(f"\n=== Optimal L0 (min s_n^dec): {optimal_l0} ===")

    return {
        "results": results,
        "optimal_l0": int(optimal_l0),
        "optimal_idx": int(optimal_idx),
    }


def plot_l0_sweep_results(
    sweep_results: Dict,
    output_dir: Path,
    model_name: str = "tabpfn",
):
    """Plot s_n^dec and R² vs L0 from sweep."""
    results = sweep_results["results"]
    optimal_l0 = sweep_results["optimal_l0"]

    l0_values = [r["actual_l0"] for r in results]
    s_n_values = [r["s_n_dec"] for r in results]
    r2_values = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: s_n^dec vs L0
    ax = axes[0]
    ax.plot(l0_values, s_n_values, 'o-', markersize=8, linewidth=2)
    ax.axvline(optimal_l0, color='red', linestyle='--', label=f'Optimal L0={optimal_l0}')
    ax.set_xlabel('L0 Sparsity', fontsize=12)
    ax.set_ylabel('s_n^dec (lower is better)', fontsize=12)
    ax.set_title(f'(A) Decoder Projection Score vs L0 - {model_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: R² vs L0
    ax = axes[1]
    ax.plot(l0_values, r2_values, 'o-', markersize=8, linewidth=2, color='C1')
    ax.axvline(optimal_l0, color='red', linestyle='--', label=f'Optimal L0={optimal_l0}')
    ax.set_xlabel('L0 Sparsity', fontsize=12)
    ax.set_ylabel('Explained Variance (R²)', fontsize=12)
    ax.set_title(f'(B) Reconstruction Quality vs L0 - {model_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.0)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"l0_sweep_sndec_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"l0_sweep_sndec_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved L0 sweep plot to {output_dir}")


def plot_pareto_from_study(
    study_path: Path,
    study_name: str,
    output_dir: Path,
    model_name: str = "tabpfn",
):
    """
    Create Pareto plot from existing Optuna study.
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("optuna required")

    storage = f"sqlite:///{study_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Extract L0 and R² from trials
    l0_values = []
    r2_values = []
    richness_values = []
    sae_types = []

    for trial in study.trials:
        if trial.value is not None and trial.user_attrs:
            l0 = trial.user_attrs.get("l0_sparsity", None)
            r2 = trial.user_attrs.get("explained_variance", None)
            if l0 is not None and r2 is not None:
                l0_values.append(l0)
                r2_values.append(r2)
                richness_values.append(trial.value)
                sae_types.append(trial.params.get("sae_type", "unknown"))

    l0_values = np.array(l0_values)
    r2_values = np.array(r2_values)
    richness_values = np.array(richness_values)
    sae_types = np.array(sae_types)

    # Compute Pareto frontier
    pareto_l0, pareto_r2, pareto_idx = compute_pareto_frontier(l0_values, r2_values)

    # Find best richness point
    best_idx = np.argmax(richness_values)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: L0 vs R² with Pareto frontier
    ax = axes[0]

    # Color by SAE type (architecture variant)
    colors = {"l1": "C0", "topk": "C1", "matryoshka": "C2", "archetypal": "C3", "matryoshka_archetypal": "C4"}
    labels = {
        "l1": "L1-penalty SAE",
        "topk": "TopK SAE",
        "matryoshka": "Matryoshka SAE",
        "archetypal": "Archetypal SAE",
        "matryoshka_archetypal": "Matryoshka-Archetypal SAE"
    }
    for sae_type in np.unique(sae_types):
        mask = sae_types == sae_type
        ax.scatter(l0_values[mask], r2_values[mask],
                   c=colors.get(sae_type, "gray"), label=labels.get(sae_type, sae_type),
                   alpha=0.6, s=50)

    # Plot Pareto frontier
    ax.plot(pareto_l0, pareto_r2, 'k-', linewidth=2, label='Pareto frontier', zorder=10)
    ax.scatter(pareto_l0, pareto_r2, c='red', s=100, marker='*', zorder=11, label='Pareto optimal')

    # Mark best richness
    ax.scatter([l0_values[best_idx]], [r2_values[best_idx]],
               c='gold', s=200, marker='p', edgecolors='black', linewidth=2,
               zorder=12, label=f'Best richness (L0={l0_values[best_idx]:.0f})')

    ax.set_xlabel('L0 Sparsity (mean active features)', fontsize=12)
    ax.set_ylabel('Explained Variance (R²)', fontsize=12)
    ax.set_title(f'(A) L0 vs Reconstruction Trade-off - {model_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: L0 vs Richness
    ax = axes[1]
    for sae_type in np.unique(sae_types):
        mask = sae_types == sae_type
        ax.scatter(l0_values[mask], richness_values[mask],
                   c=colors.get(sae_type, "gray"), label=labels.get(sae_type, sae_type),
                   alpha=0.6, s=50)

    ax.scatter([l0_values[best_idx]], [richness_values[best_idx]],
               c='gold', s=200, marker='p', edgecolors='black', linewidth=2,
               zorder=12, label=f'Best (L0={l0_values[best_idx]:.0f}, R={richness_values[best_idx]:.3f})')

    ax.set_xlabel('L0 Sparsity (mean active features)', fontsize=12)
    ax.set_ylabel('Richness Score', fontsize=12)
    ax.set_title(f'(B) L0 vs Richness Trade-off - {model_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"pareto_l0_r2_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"pareto_l0_r2_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved Pareto plot to {output_dir}")

    # Print summary
    print(f"\n=== Pareto Summary for {model_name} ===")
    print(f"Total trials: {len(l0_values)}")
    print(f"Pareto optimal points: {len(pareto_idx)}")
    print(f"\nPareto frontier (L0 → R²):")
    for l0, r2 in zip(pareto_l0, pareto_r2):
        print(f"  L0={l0:6.1f}  R²={r2:.4f}")
    print(f"\nBest richness config:")
    print(f"  L0={l0_values[best_idx]:.1f}, R²={r2_values[best_idx]:.4f}, Richness={richness_values[best_idx]:.4f}")

    return {
        "pareto_l0": pareto_l0.tolist(),
        "pareto_r2": pareto_r2.tolist(),
        "best_l0": float(l0_values[best_idx]),
        "best_r2": float(r2_values[best_idx]),
        "best_richness": float(richness_values[best_idx]),
    }


def main():
    parser = argparse.ArgumentParser(description="SAE Pareto Analysis")
    parser.add_argument("--study-name", type=str, default="sae_tabpfn_adult_v3")
    parser.add_argument("--model", type=str, default="tabpfn")
    parser.add_argument("--sweep-optimal", action="store_true",
                        help="Run L0 sweep to find optimal via s_n^dec")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "output" / "sae_sweep" / args.model
    study_path = model_dir / f"{args.study_name}.db"

    if not study_path.exists():
        print(f"Study not found: {study_path}")
        return

    # Plot Pareto from existing study
    result = plot_pareto_from_study(
        study_path=study_path,
        study_name=args.study_name,
        output_dir=model_dir / "plots",
        model_name=args.model,
    )

    # Optionally sweep for optimal L0 using s_n^dec
    if args.sweep_optimal:
        embeddings_path = PROJECT_ROOT / "output" / "tabpfn_embeddings_adult_L17.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            sweep_result = sweep_l0_for_optimal(embeddings, device=args.device)

            # Save results
            with open(model_dir / "optimal_l0_sweep.json", "w") as f:
                json.dump(sweep_result, f, indent=2)

            # Plot sweep results
            plot_l0_sweep_results(sweep_result, model_dir / "plots", args.model)


if __name__ == "__main__":
    main()
