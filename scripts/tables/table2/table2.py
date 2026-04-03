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
from scripts.sae.compare_sae_cross_model import find_common_datasets

# ── Model configuration ─────────────────────────────────────────────────────
# (display_name, dir_name, extraction_layer_key)
MODEL_CONFIGS = [
    ("TabPFN", "tabpfn", "layer_18"),
    ("Mitra", "mitra", "final_norm"),
    ("TabICL", "tabicl", "layer_9"),
    ("TabICL-v2", "tabicl_v2", "layer_11"),
    ("TabDPT", "tabdpt", "layer_13"),
    ("CARTE", "carte", "layer_2"),
]

EMB_BASE = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"
MAX_PER_DATASET = 500
SEED = 42

OUTPUT_TEX = Path(__file__).parent / "procrustes_table.tex"


def pool_embeddings(
    emb_dir: Path,
    datasets: List[str],
    layer_key: str,
    max_per_dataset: int = MAX_PER_DATASET,
) -> np.ndarray:
    """Load and pool embeddings for the given datasets at a specific layer."""
    all_embs = []
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            path = emb_dir / f"{ds}.npz"
        data = np.load(path, allow_pickle=True)
        emb = data[layer_key].astype(np.float32)
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
    layer_keys: Dict[str, str] = {}
    model_names: List[str] = []
    for display_name, dir_name, layer_key in MODEL_CONFIGS:
        emb_dir = EMB_BASE / dir_name
        if not emb_dir.exists():
            print(f"Warning: missing {emb_dir}, skipping {display_name}")
            continue
        emb_dirs[display_name] = emb_dir
        layer_keys[display_name] = layer_key
        model_names.append(display_name)

    if len(model_names) < 2:
        print("Error: need at least 2 models")
        sys.exit(1)

    # ── Find common datasets ─────────────────────────────────────────────
    print("Finding common datasets...")
    dataset_sets = []
    for name, emb_dir in emb_dirs.items():
        datasets = set(
            f.stem.replace("tabarena_", "")
            for f in emb_dir.glob("*.npz")
            if f.stem != "layer_names"
        )
        print(f"  {name}: {len(datasets)} datasets")
        dataset_sets.append(datasets)
    common_datasets = sorted(set.intersection(*dataset_sets))
    print(f"\nUsing {len(common_datasets)} common datasets\n")

    # ── Pool embeddings ──────────────────────────────────────────────────
    embeddings: Dict[str, np.ndarray] = {}
    for name in model_names:
        emb = pool_embeddings(emb_dirs[name], common_datasets, layer_keys[name])
        print(f"  {name:>10s}: {emb.shape[0]:>6,} samples, dim={emb.shape[1]}")
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
