#!/usr/bin/env python3
"""
Table 2: Pairwise Procrustes residuals between tabular foundation models.

Computes normalized Procrustes distance for all model pairs on pooled
TabArena embeddings and outputs a lower-triangle LaTeX table.

Usage:
    python scripts/table2/table2.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from analysis.similarity import procrustes_align
from scripts.compare_sae_cross_model import find_common_datasets

# ── Model configuration ─────────────────────────────────────────────────────
MODEL_CONFIGS = [
    ("TabPFN", "tabpfn_layer16_ctx600"),
    ("CARTE", "carte_layer1_ctx600"),
    ("TabICL", "tabicl_layer10_ctx600"),
    ("TabDPT", "tabdpt_layer14_ctx600"),
    ("Mitra", "mitra_layer12_ctx600"),
    ("HyperFast", "hyperfast_layer2_ctx600"),
    ("Tabula-8B", "tabula8b_layer21_ctx600"),
]

EMB_BASE = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
MAX_PER_DATASET = 500
SEED = 42

OUTPUT_TEX = Path(__file__).parent / "procrustes_table.tex"


def pool_embeddings(
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = MAX_PER_DATASET,
) -> np.ndarray:
    """Load and pool embeddings for the given datasets, subsampling with a fixed seed."""
    all_embs = []
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(SEED)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        all_embs.append(emb)
    return np.concatenate(all_embs)


def compute_procrustes_matrix(
    embeddings: Dict[str, np.ndarray],
    model_names: List[str],
) -> np.ndarray:
    """Compute pairwise Procrustes disparity d^2 in [0, 1].

    procrustes_align returns ||A@R - B||_F where A, B are centered and
    unit-Frobenius-norm. Since ||A||_F = ||B||_F = 1:

        d_F^2 = ||A@R - B||_F^2 = 2 - 2*trace(R^T A^T B)

    The Procrustes disparity is d^2 = d_F^2 / 2 = 1 - trace(R^T A^T B),
    ranging from 0 (identical up to rotation) to 1 (orthogonal).
    """
    n = len(model_names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance, _, _ = procrustes_align(
                embeddings[model_names[i]], embeddings[model_names[j]]
            )
            # d_F in [0, sqrt(2)] -> d^2 = d_F^2 / 2 in [0, 1]
            disparity = distance ** 2 / 2.0

            matrix[i, j] = disparity
            matrix[j, i] = disparity

    return matrix


def generate_latex_table(
    matrix: np.ndarray,
    model_labels: List[str],
    n_datasets: int,
    n_samples: int,
) -> str:
    """Generate a lower-triangle LaTeX table of Procrustes distances."""
    n = len(model_labels)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Procrustes disparity ($d^2 = 1 - \mathrm{tr}(R^\top A^\top B)$, "
        r"$\in [0,1]$) after optimal orthogonal alignment. "
        f"{n_datasets} TabArena datasets, {{\\num{{{n_samples}}}}} pooled samples.}}",
        r"\label{tab:procrustes_matrix}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r" & " + " & ".join(
            [rf"\rotatebox{{45}}{{{m}}}" for m in model_labels]
        ) + r" \\",
        r"\midrule",
    ]

    for i, name in enumerate(model_labels):
        row_vals = []
        for j in range(n):
            if i == j:
                row_vals.append("0")
            elif i > j:
                row_vals.append(f"{matrix[i, j]:.4f}")
            else:
                row_vals.append("--")
        lines.append(f"{name} & " + " & ".join(row_vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    # ── Resolve embedding directories ────────────────────────────────────
    emb_dirs: Dict[str, Path] = {}
    model_names: List[str] = []
    for display_name, dir_name in MODEL_CONFIGS:
        emb_dir = EMB_BASE / dir_name
        if not emb_dir.exists():
            print(f"Warning: missing {emb_dir}, skipping {display_name}")
            continue
        emb_dirs[display_name] = emb_dir
        model_names.append(display_name)

    if len(model_names) < 2:
        print("Error: need at least 2 models")
        sys.exit(1)

    # ── Find common datasets ─────────────────────────────────────────────
    print("Finding common datasets...")
    common_datasets = find_common_datasets(emb_dirs)
    print(f"\nUsing {len(common_datasets)} common datasets\n")

    # ── Pool embeddings ──────────────────────────────────────────────────
    embeddings: Dict[str, np.ndarray] = {}
    for name in model_names:
        emb = pool_embeddings(emb_dirs[name], common_datasets)
        print(f"  {name:8s}: {emb.shape[0]:>6,} samples, dim={emb.shape[1]}")
        embeddings[name] = emb

    n_samples = embeddings[model_names[0]].shape[0]

    # ── Compute Procrustes matrix ────────────────────────────────────────
    print("\nComputing pairwise Procrustes distances...")
    matrix = compute_procrustes_matrix(embeddings, model_names)

    # ── Print summary to stdout ──────────────────────────────────────────
    n = len(model_names)
    print(f"\nProcrustes Disparity d² ∈ [0,1] (lower triangle):")
    header = f"{'':10s}" + "".join(f"{m:>10s}" for m in model_names)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{model_names[i]:10s}"
        for j in range(n):
            if i == j:
                row += f"{'0':>10s}"
            elif i > j:
                row += f"{matrix[i, j]:>10.4f}"
            else:
                row += f"{'--':>10s}"
        print(row)

    # ── Generate LaTeX ───────────────────────────────────────────────────
    tex = generate_latex_table(matrix, model_names, len(common_datasets), n_samples)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(tex)
    print(f"\nLaTeX table saved to {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
