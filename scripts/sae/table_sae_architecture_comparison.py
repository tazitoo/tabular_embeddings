#!/usr/bin/env python3
"""
Generate SAE Architecture Comparison Table (Table 8 equivalent).

Loads the best trial for each SAE architecture type and reports:
- Winning hyperparameters
- Composite score
- RMSE (reconstruction error)
- Stability
- L0 (measured sparsity)
- Dead % (percentage of features that never activate)

Also prints the hyperparameter search space for the methods section.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT

from scripts.sae.analyze_sae_concepts_deep import load_sae_checkpoint
from scripts.sae.compare_sae_cross_model import sae_sweep_dir
from scripts.sae.sae_tabarena_sweep import get_tabarena_splits, pool_embeddings


def compute_sae_metrics(model, embeddings: np.ndarray) -> Dict:
    """
    Compute all SAE quality metrics.

    Returns:
        rmse: Root mean squared reconstruction error
        l0: Average number of active features per sample
        dead_pct: Percentage of features that never activate
    """
    # Normalize embeddings (same as training)
    emb_std = embeddings.std(axis=0, keepdims=True)
    emb_std[emb_std < 1e-8] = 1.0
    emb_norm = embeddings / emb_std
    emb_norm = emb_norm - emb_norm.mean(axis=0, keepdims=True)

    # Get activations and reconstruction
    with torch.no_grad():
        emb_tensor = torch.tensor(emb_norm, dtype=torch.float32)
        activations = model.encode(emb_tensor)
        reconstruction = model.decode(activations).numpy()

    # RMSE
    rmse = np.sqrt(np.mean((emb_norm - reconstruction) ** 2))

    # L0: average number of active (non-zero) features per sample
    l0 = (activations > 0).sum(dim=1).float().mean().item()

    # Dead %: percentage of features that never activate
    alive_mask = (activations > 1e-6).any(dim=0).cpu().numpy()
    dead_pct = 100 * (1 - alive_mask.mean())

    return {
        'rmse': rmse,
        'l0': l0,
        'dead_pct': dead_pct,
    }


def format_hps(params: Dict, architecture: str) -> str:
    """Format hyperparameters for table display."""
    if architecture == "matryoshka_archetypal":
        return (f"exp={params.get('expansion', 'N/A')}, "
                f"k={params.get('topk', 'N/A')}, "
                f"n_arch={params.get('archetypal_n', 'N/A')}, "
                f"temp={params.get('archetypal_temp', 0):.2f}")
    elif architecture in ["topk", "l1"]:
        return f"k={params.get('topk', 'N/A')}, exp={params.get('expansion', 'N/A')}"
    elif architecture == "archetypal":
        return f"n_arch={params.get('archetypal_n', 'N/A')}, temp={params.get('archetypal_temp', 0):.2f}"
    elif architecture == "matryoshka":
        return f"exp={params.get('expansion', 'N/A')}, k={params.get('topk', 'N/A')}"
    else:
        return "N/A"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tabpfn",
                       help="Base model to analyze (e.g., tabpfn, tabicl_layer10)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"SAE Architecture Comparison: {args.model}")
    print("=" * 80)

    # Define architecture types to compare
    # For TabPFN, these should exist from the original sweep
    architectures = [
        "l1",
        "topk",
        "matryoshka",
        "archetypal",
        "matryoshka_archetypal",
    ]

    # Load test embeddings for metrics computation
    emb_model = args.model.split('_layer')[0]
    train_datasets, test_datasets = get_tabarena_splits(emb_model)
    test_embeddings, _ = pool_embeddings(emb_model, test_datasets,
                                         max_per_dataset=500, normalize=False)
    print(f"\nTest embeddings: {test_embeddings.shape[0]} samples × {test_embeddings.shape[1]} dims")

    # Print hyperparameter search space
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH SPACE")
    print("=" * 80)
    print("""
For all architectures:
  - expansion: {4, 8} (hidden_dim = expansion × input_dim)
  - learning_rate: log-uniform [1e-4, 1e-2]
  - sparsity_penalty: log-uniform [1e-4, 1e-2]
  - batch_size: {128, 256}
  - n_epochs: {50, 100}

TopK-specific:
  - topk: {16, 32, 64, 128}

Archetypal-specific:
  - n_archetypes: {256, 500, 1000}
  - simplex_temp: log-uniform [0.05, 0.5]
  - relaxation: uniform [0.1, 2.0]

Matryoshka-Archetypal:
  - All TopK + Archetypal parameters
  - matryoshka_dims: [32, 64, 128, 256]
""")

    # Collect results
    results = []

    base_dir = sae_sweep_dir() / args.model

    # Try to find checkpoints for each architecture
    # Naming convention: sae_{architecture}_validated.pt
    for arch in architectures:
        checkpoint_name = f"sae_{arch}_validated.pt"
        checkpoint_path = base_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"\nWarning: No checkpoint found for {arch}: {checkpoint_path}")
            continue

        print(f"\n--- {arch} ---")

        # Load SAE
        model, config, _ = load_sae_checkpoint(checkpoint_path)

        # Load metrics directly from checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        metrics = ckpt.get('metrics', ckpt.get('validation', {}))

        # Compute metrics
        sae_metrics = compute_sae_metrics(model, test_embeddings)

        # Get stored metrics
        r2 = metrics.get('r2', 0.0)
        stability = metrics.get('stability', 0.0)
        stored_l0 = metrics.get('l0_sparsity', sae_metrics['l0'])
        alive_features = metrics.get('alive_features', 0)

        # Compute composite score
        composite_score = 0.4 * r2 + 0.6 * stability

        # Compute dead% from alive_features if available
        hidden_dim = config.hidden_dim
        if alive_features > 0:
            dead_pct_stored = 100 * (1 - alive_features / hidden_dim)
        else:
            dead_pct_stored = sae_metrics['dead_pct']

        # Format hyperparameters
        # Extract from config
        params = {
            'expansion': config.hidden_dim // config.input_dim,
            'topk': config.topk,
            'archetypal_n': config.archetypal_n_archetypes if hasattr(config, 'archetypal_n_archetypes') else None,
            'archetypal_temp': config.archetypal_simplex_temp if hasattr(config, 'archetypal_simplex_temp') else None,
        }
        hp_str = format_hps(params, arch)

        result = {
            'type': arch.replace('_', '-').title(),
            'hps': hp_str,
            'score': composite_score,
            'rmse': sae_metrics['rmse'],
            'stability': stability,
            'l0': stored_l0,  # Use stored L0 from training
            'dead_pct': dead_pct_stored,  # Use stored alive_features
        }

        results.append(result)

        print(f"  HPs: {hp_str}")
        print(f"  Score: {composite_score:.3f}, RMSE: {sae_metrics['rmse']:.4f}, "
              f"Stability: {stability:.3f}, L0: {sae_metrics['l0']:.1f}, Dead: {sae_metrics['dead_pct']:.1f}%")

    # Generate table
    print("\n" + "=" * 80)
    print("TABLE: SAE ARCHITECTURE COMPARISON")
    print("=" * 80)
    print()

    # Header
    print(f"{'Type':<25} {'HPs':<35} {'Score':>7} {'RMSE':>7} {'Stab':>6} {'L0':>5} {'Dead%':>6}")
    print("-" * 105)

    # Rows
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{r['type']:<25} {r['hps']:<35} {r['score']:>7.3f} {r['rmse']:>7.4f} "
              f"{r['stability']:>6.3f} {r['l0']:>5.1f} {r['dead_pct']:>6.1f}")

    print()
    print("Score = 0.4 × (1 - RMSE) + 0.6 × Stability")
    print("RMSE = Root mean squared error on test embeddings")
    print("Stab = s_n^dec decoder stability (Chanin & Garriga-Alonso 2025)")
    print("L0 = Average number of active features per sample")
    print("Dead% = Percentage of features that never activate")

    # LaTeX version
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(r"\begin{table}[t]")
    print(r"\caption{SAE Architecture Comparison on " + args.model.upper().replace('_', r'\_') + r" Embeddings}")
    print(r"\label{tab:sae_comparison}")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{l l r r r r r}")
    print(r"\toprule")
    print(r"Type & Hyperparameters & Score & RMSE & Stab & L0 & Dead\% \\")
    print(r"\midrule")

    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        # Escape underscores in type name
        type_name = r['type'].replace('-', r'\mbox{-}')
        hp_str = r['hps'].replace('_', r'\_')
        print(f"{type_name} & {hp_str} & {r['score']:.3f} & {r['rmse']:.4f} & "
              f"{r['stability']:.3f} & {int(r['l0'])} & {r['dead_pct']:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
