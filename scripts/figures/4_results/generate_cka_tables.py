#!/usr/bin/env python3
"""
Generate LaTeX tables for CKA results.

Produces three table formats:
- cka_table_matrix.tex: Lower-triangle matrix (compact, recommended)
- cka_table_pairwise.tex: Full pairwise list sorted by CKA
- cka_table_clusters.tex: Grouped by architecture cluster

Usage:
    python scripts/4_results/generate_cka_tables.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
from scripts._project_root import PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_CSV = OUTPUT_DIR / "geometric_sweep_tabarena_7model_summary.csv"

# Model configuration
MODEL_ORDER = ['mitra', 'tabdpt', 'tabpfn', 'tabicl', 'carte', 'hyperfast', 'tabula']
MODEL_LABELS = ['Mitra', 'TabDPT', 'TabPFN', 'TabICL', 'CARTE', 'HyperFast', 'Tabula-8B']
NAME_MAP = dict(zip(MODEL_ORDER, MODEL_LABELS))


def generate_matrix_table(df: pd.DataFrame, output_path: Path):
    """Generate lower-triangle matrix table (compact format)."""
    n = len(MODEL_ORDER)
    matrix = np.eye(n)

    for _, row in df.iterrows():
        try:
            i = MODEL_ORDER.index(row['model_a'])
            j = MODEL_ORDER.index(row['model_b'])
            matrix[i, j] = row['cka_mean']
            matrix[j, i] = row['cka_mean']
        except ValueError:
            continue

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CKA similarity matrix across 7 tabular foundation models (TabArena, 51 datasets).}",
        r"\label{tab:cka_matrix}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r" & " + " & ".join([f"\\rotatebox{{45}}{{{m}}}" for m in MODEL_LABELS]) + r" \\",
        r"\midrule",
    ]

    for i, name in enumerate(MODEL_LABELS):
        row_vals = []
        for j in range(n):
            if i == j:
                row_vals.append("1.00")
            elif i > j:
                row_vals.append(f"{matrix[i,j]:.2f}")
            else:
                row_vals.append("--")
        lines.append(f"{name} & " + " & ".join(row_vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text('\n'.join(lines))
    print(f"Saved: {output_path}")


def generate_pairwise_table(df: pd.DataFrame, output_path: Path):
    """Generate full pairwise table sorted by CKA."""
    df_sorted = df.sort_values('cka_mean', ascending=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pairwise CKA similarity between tabular foundation models (TabArena, 51 datasets).}",
        r"\label{tab:cka_pairwise}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model A & Model B & CKA & Cosine & Procrustes \\",
        r"\midrule",
    ]

    for _, row in df_sorted.iterrows():
        ma = NAME_MAP.get(row['model_a'], row['model_a'])
        mb = NAME_MAP.get(row['model_b'], row['model_b'])
        cka = f"{row['cka_mean']:.2f} \\tiny{{$\\pm${row['cka_std']:.2f}}}"
        cos = f"{row['cosine_mean']:.2f}"
        proc = f"{row['procrustes_mean']:.2f}"
        lines.append(f"{ma} & {mb} & {cka} & {cos} & {proc} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text('\n'.join(lines))
    print(f"Saved: {output_path}")


def generate_cluster_table(df: pd.DataFrame, output_path: Path):
    """Generate table grouped by architecture cluster."""

    def get_cka(a, b):
        row = df[((df['model_a'] == a) & (df['model_b'] == b)) |
                 ((df['model_a'] == b) & (df['model_b'] == a))]
        if len(row) == 0:
            return None, None
        return row.iloc[0]['cka_mean'], row.iloc[0]['cka_std']

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CKA similarity reveals three geometric clusters among tabular foundation models.}",
        r"\label{tab:cka_clusters}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Comparison & CKA \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Transformer ICL cluster (high similarity)}} \\",
    ]

    # Transformer pairs
    transformer_pairs = [
        ('mitra', 'tabdpt'), ('mitra', 'tabpfn'), ('tabdpt', 'tabpfn'),
        ('mitra', 'tabicl'), ('tabdpt', 'tabicl'), ('tabicl', 'tabpfn')
    ]
    for a, b in transformer_pairs:
        mean, std = get_cka(a, b)
        if mean:
            lines.append(f"{NAME_MAP[a]}--{NAME_MAP[b]} & {mean:.2f} $\\pm$ {std:.2f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{CARTE (GNN) vs Transformers}} \\")
    for b in ['mitra', 'tabdpt', 'tabpfn', 'tabicl']:
        mean, std = get_cka('carte', b)
        if mean:
            lines.append(f"CARTE--{NAME_MAP[b]} & {mean:.2f} $\\pm$ {std:.2f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{HyperFast (hypernetwork) vs others}} \\")
    for b in ['mitra', 'tabdpt', 'tabpfn', 'tabicl', 'carte']:
        mean, std = get_cka('hyperfast', b)
        if mean:
            lines.append(f"HyperFast--{NAME_MAP[b]} & {mean:.2f} $\\pm$ {std:.2f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Tabula-8B (LLM) vs all others}} \\")
    for b in ['mitra', 'tabdpt', 'tabpfn', 'tabicl', 'carte', 'hyperfast']:
        mean, std = get_cka('tabula', b)
        if mean:
            lines.append(f"Tabula-8B--{NAME_MAP[b]} & {mean:.2f} $\\pm$ {std:.2f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text('\n'.join(lines))
    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(SUMMARY_CSV)

    generate_matrix_table(df, OUTPUT_DIR / "cka_table_matrix.tex")
    generate_pairwise_table(df, OUTPUT_DIR / "cka_table_pairwise.tex")
    generate_cluster_table(df, OUTPUT_DIR / "cka_table_clusters.tex")


if __name__ == "__main__":
    main()
