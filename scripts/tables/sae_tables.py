#!/usr/bin/env python3
"""
Generate LaTeX tables for SAE sweep results.

Creates appendix tables with:
1. Best hyperparameters per architecture per model
2. Metrics (R², L0, stability, alive features) per architecture per model
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SWEEP_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep"
OUTPUT_DIR = Path(__file__).parent  # Save outputs in scripts/tables/

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


def generate_hyperparameter_table():
    """Generate LaTeX table of best hyperparameters per architecture."""

    # Collect all models with results
    models = []
    for model_key in MODEL_NAMES.keys():
        results = load_sweep_results(model_key)
        if results:
            models.append((model_key, results))

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Architecture & Expansion & LR & Sparsity & TopK \\")
    lines.append(r"\midrule")

    for model_key, results in models:
        model_name = MODEL_NAMES[model_key]
        for arch_key in ['l1', 'topk', 'matryoshka', 'archetypal', 'matryoshka_archetypal']:
            if arch_key not in results:
                continue

            arch_name = ARCH_NAMES[arch_key]
            params = results[arch_key]['params']

            expansion = params.get('expansion', '-')
            lr = params.get('learning_rate', 0.0)
            sparsity = params.get('sparsity_penalty', 0.0)
            topk = params.get('topk', '-')

            # Format values
            lr_str = f"{lr:.2e}" if lr > 0 else "-"
            sparsity_str = f"{sparsity:.2e}" if sparsity > 0 else "-"

            lines.append(f"{model_name} & {arch_name} & {expansion} & {lr_str} & {sparsity_str} & {topk} \\\\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"  # Replace last midrule with bottomrule
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Best hyperparameters for each SAE architecture across models. "
                 r"LR = learning rate, Sparsity = sparsity penalty.}")
    lines.append(r"\label{tab:sae_hyperparameters}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / "sae_hyperparameters.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated hyperparameters table: {output_file}")


def generate_metrics_table():
    """Generate LaTeX table of metrics per architecture."""

    # Collect all models with results
    models = []
    for model_key in MODEL_NAMES.keys():
        results = load_sweep_results(model_key)
        if results:
            models.append((model_key, results))

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Architecture & R² & L0 & Alive & Stability \\")
    lines.append(r"\midrule")

    for model_key, results in models:
        model_name = MODEL_NAMES[model_key]
        for arch_key in ['l1', 'topk', 'matryoshka', 'archetypal', 'matryoshka_archetypal']:
            if arch_key not in results:
                continue

            arch_name = ARCH_NAMES[arch_key]
            metrics = results[arch_key]['metrics']

            r2 = metrics.get('r2', 0.0)
            l0 = metrics.get('l0_sparsity', 0.0)
            alive = metrics.get('alive_features', 0)
            stability = metrics.get('stability', 0.0)

            lines.append(f"{model_name} & {arch_name} & {r2:.4f} & {l0:.1f} & {alive} & {stability:.3f} \\\\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{SAE metrics for each architecture across models. "
                 r"R² = reconstruction quality, L0 = average sparsity, "
                 r"Alive = number of active features, Stability = $s_n^{dec}$ metric.}")
    lines.append(r"\label{tab:sae_metrics}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / "sae_metrics.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated metrics table: {output_file}")


def main():
    """Generate all SAE tables."""
    print("Generating SAE LaTeX tables...")
    print()

    generate_hyperparameter_table()
    generate_metrics_table()

    print()
    print(f"✓ Tables saved to: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  - sae_hyperparameters.tex")
    print("  - sae_metrics.tex")


if __name__ == "__main__":
    main()
