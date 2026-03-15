#!/usr/bin/env python3
"""
Generate CKA heatmap figure for the paper.

Usage:
    python scripts/4_results/plot_cka_heatmap.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
from scripts._project_root import PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_CSV = OUTPUT_DIR / "geometric_sweep_tabarena_7model_summary.csv"

# Model ordering (cluster transformers together)
MODEL_ORDER = ['mitra', 'tabdpt', 'tabpfn', 'tabicl', 'carte', 'hyperfast', 'tabula']
MODEL_LABELS = ['Mitra', 'TabDPT', 'TabPFN', 'TabICL', 'CARTE', 'HyperFast', 'Tabula-8B']


def load_cka_matrix(csv_path: Path) -> np.ndarray:
    """Load summary CSV and build symmetric CKA matrix."""
    df = pd.read_csv(csv_path)
    n = len(MODEL_ORDER)
    matrix = np.eye(n)

    for _, row in df.iterrows():
        try:
            i = MODEL_ORDER.index(row['model_a'])
            j = MODEL_ORDER.index(row['model_b'])
            matrix[i, j] = row['cka_mean']
            matrix[j, i] = row['cka_mean']
        except ValueError:
            continue  # Skip models not in our order

    return matrix


def plot_heatmap(matrix: np.ndarray, output_path: Path):
    """Create and save the CKA heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        xticklabels=MODEL_LABELS,
        yticklabels=MODEL_LABELS,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'CKA Similarity', 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )

    ax.set_title('Pairwise CKA Similarity\n(TabArena, 51 datasets)', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    # Save both PDF and PNG
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def main():
    matrix = load_cka_matrix(SUMMARY_CSV)
    output_path = OUTPUT_DIR / "cka_heatmap_7model"
    plot_heatmap(matrix, output_path)


if __name__ == "__main__":
    main()
