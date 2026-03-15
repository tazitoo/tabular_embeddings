#!/usr/bin/env python3
"""
Generate LaTeX tables for SAE sweep results.

Creates appendix tables with:
1. Best hyperparameters per architecture per model
2. Metrics (R², L0, stability, alive features) per architecture per model
3. Pairwise shared concept counts between models
"""

import json
import sys
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

import torch

from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    SAE_FILENAME,
    sae_sweep_dir,
)

SWEEP_DIR = sae_sweep_dir()
OUTPUT_DIR = Path(__file__).parent  # Save outputs in scripts/tables/


def load_checkpoint_config(model_dir_name):
    """Load config from SAE checkpoint for a model."""
    ckpt_path = SWEEP_DIR / model_dir_name / SAE_FILENAME
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    metrics = ckpt.get("metrics", {})
    return {"config": config, "metrics": metrics}


def generate_hyperparameter_table():
    """Generate LaTeX table of SAE hyperparameters and metrics per model."""

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrr rrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & $d$ & $d_{\text{SAE}}$ & TopK & LR & $\lambda$ & $K$ & $\tau$"
        r" & Recon & \%Alive & $s_n^{\text{dec}}$ \\"
    )
    lines.append(r"\midrule")

    for display_name, sweep_dir, _ in DEFAULT_MODELS:
        result = load_checkpoint_config(sweep_dir)
        if not result:
            continue
        c = result["config"]
        m = result["metrics"]

        input_dim = c.get("input_dim", "?")
        hidden_dim = c.get("hidden_dim", "?")
        topk = c.get("topk", "-")
        lr = c.get("learning_rate", 0.0)
        sparsity = c.get("sparsity_penalty", 0.0)
        n_arch = c.get("archetypal_n_archetypes", "-")
        temp = c.get("archetypal_simplex_temp", "-")

        recon = m.get("reconstruction_loss", 0.0)
        alive = m.get("alive_features", 0)
        pct_alive = 100 * alive / hidden_dim if isinstance(hidden_dim, int) else 0
        stability = m.get("stability", 0.0)

        lr_str = f"{lr:.1e}"
        sp_str = f"{sparsity:.1e}" if sparsity > 0 else "---"
        temp_str = f"{temp:.2f}" if isinstance(temp, float) else str(temp)
        stab_str = f"{stability:.3f}" if stability > 0 else "---"

        lines.append(
            f"{display_name} & {input_dim} & {hidden_dim} & {topk} & "
            f"{lr_str} & {sp_str} & {n_arch} & {temp_str} & "
            f"{recon:.4f} & {pct_alive:.1f} & {stab_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{SAE configuration and quality metrics per model "
        r"(Matryoshka-Archetypal architecture). "
        r"$d$ = embedding dim, $d_{\text{SAE}}$ = dictionary size, "
        r"$\lambda$ = sparsity penalty, $K$ = archetypes, $\tau$ = simplex temperature, "
        r"Recon = MSE reconstruction loss, "
        r"\%Alive = percentage of active dictionary elements, "
        r"$s_n^{\text{dec}}$ = decoder stability across seeds.}"
    )
    lines.append(r"\label{tab:sae_hyperparameters}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / "sae_hyperparameters.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated hyperparameters table: {output_file}")


def generate_shared_concepts_table():
    """Generate LaTeX table of pairwise shared concept counts between models."""
    hierarchy_path = PROJECT_ROOT / "output" / "concept_hierarchy_full.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    shared = hierarchy["model_comparison"]["shared"]

    # Ordered to match paper's model presentation
    display_order = [
        "TabPFN", "Mitra", "TabDPT", "TabICL", "CARTE",
        "HyperFast", "Tabula-8B", "TabICL-v2",
    ]

    # Build symmetric matrix
    n = len(display_order)
    matrix = [[0] * n for _ in range(n)]
    for key, val in shared.items():
        a, b = key.split("__")
        count = val if isinstance(val, int) else val.get("n_groups", len(val))
        if a in display_order and b in display_order:
            i, j = display_order.index(a), display_order.index(b)
            matrix[i][j] = count
            matrix[j][i] = count

    # Build LaTeX
    col_spec = "l" + "r" * n
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join([""] + display_order) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(display_order):
        cells = [model]
        for j in range(n):
            if i == j:
                cells.append("---")
            else:
                cells.append(str(matrix[i][j]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Pairwise shared concept counts between models. "
        r"Each cell shows the number of concept groups containing features "
        r"from both models, after Leiden community detection splits the "
        r"mega-group into coherent sub-communities.}"
    )
    lines.append(r"\label{tab:shared_concepts}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / "shared_concepts.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated shared concepts table: {output_file}")


def main():
    """Generate all SAE tables."""
    print("Generating SAE LaTeX tables...")
    print()

    generate_hyperparameter_table()
    generate_shared_concepts_table()

    print()
    print(f"Tables saved to: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  - sae_hyperparameters.tex")
    print("  - shared_concepts.tex")


if __name__ == "__main__":
    main()
