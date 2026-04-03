#!/usr/bin/env python3
"""
Table 1: Pairwise CKA similarity between tabular foundation models.

Computes per-dataset linear CKA for all model pairs on common TabArena
datasets and outputs a lower-triangle LaTeX table with mean ± std.

Usage:
    python scripts/table1/table1.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from analysis.similarity import centered_kernel_alignment
from scripts.sae.compare_sae_cross_model import find_common_datasets

# ── Model configuration ─────────────────────────────────────────────────────
# (display_name, dir_name, extraction_layer_key)
MODEL_CONFIGS = [
    ("TabPFN", "tabpfn", "layer_18"),
    ("Mitra", "mitra", "layer_11"),
    ("TabICL", "tabicl", "layer_9"),
    ("TabICL-v2", "tabicl_v2", "layer_11"),
    ("TabDPT", "tabdpt", "layer_13"),
    ("CARTE", "carte", "layer_2"),
]

EMB_BASE = PROJECT_ROOT / "output" / "sae_training_round9" / "embeddings"
MAX_PER_DATASET = 500
SEED = 42

OUTPUT_TEX = Path(__file__).parent / "cka_table.tex"


def load_embeddings(
    emb_dir: Path,
    dataset: str,
    layer_key: str,
    max_per_dataset: int = MAX_PER_DATASET,
) -> np.ndarray:
    """Load embeddings for a single dataset at a specific layer."""
    # Try both naming conventions
    path = emb_dir / f"tabarena_{dataset}.npz"
    if not path.exists():
        path = emb_dir / f"{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    emb = data[layer_key].astype(np.float32)
    if len(emb) > max_per_dataset:
        np.random.seed(SEED)
        idx = np.random.choice(len(emb), max_per_dataset, replace=False)
        emb = emb[idx]
    return emb


def compute_cka_matrix(
    emb_dirs: Dict[str, Path],
    layer_keys: Dict[str, str],
    model_names: List[str],
    datasets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise CKA averaged across datasets.

    Returns:
        mean_matrix: (n_models, n_models) mean CKA across datasets
        std_matrix:  (n_models, n_models) std of CKA across datasets
    """
    n = len(model_names)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Pre-collect per-pair CKA values across datasets
    pair_scores: Dict[Tuple[int, int], List[float]] = {p: [] for p in pairs}

    for ds_idx, ds in enumerate(datasets):
        if (ds_idx + 1) % 10 == 0 or ds_idx == 0:
            print(f"  Dataset {ds_idx + 1}/{len(datasets)}: {ds}")

        # Load all model embeddings for this dataset
        embs = {}
        for name in model_names:
            embs[name] = load_embeddings(emb_dirs[name], ds, layer_keys[name])

        # Align sample counts (should already match from same dataset)
        n_samples = min(e.shape[0] for e in embs.values())
        for name in model_names:
            embs[name] = embs[name][:n_samples]

        # CKA for each pair
        for i, j in pairs:
            cka = centered_kernel_alignment(embs[model_names[i]], embs[model_names[j]])
            if not np.isnan(cka):
                pair_scores[(i, j)].append(cka)

    # Build matrices
    mean_matrix = np.eye(n)
    std_matrix = np.zeros((n, n))

    for i, j in pairs:
        scores = pair_scores[(i, j)]
        if scores:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
        else:
            mean_val, std_val = 0.0, 0.0
        mean_matrix[i, j] = mean_matrix[j, i] = mean_val
        std_matrix[i, j] = std_matrix[j, i] = std_val

    return mean_matrix, std_matrix


def generate_latex_table(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    model_labels: List[str],
    n_datasets: int,
) -> str:
    """Generate a lower-triangle LaTeX table of CKA similarities."""
    n = len(model_labels)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Linear CKA similarity ($\in [0,1]$) between model "
        r"representations, averaged over "
        f"{n_datasets} TabArena datasets. Higher values indicate more "
        r"similar representational geometry.}",
        r"\label{tab:cka_matrix}",
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
                row_vals.append("1")
            elif i > j:
                mean_val = mean_matrix[i, j]
                std_val = std_matrix[i, j]
                row_vals.append(f"{mean_val:.2f}\\scriptsize{{$\\pm${std_val:.2f}}}")
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
    # find_common_datasets expects tabarena_*.npz; handle both conventions
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

    # ── Compute CKA matrix ───────────────────────────────────────────────
    print("Computing pairwise CKA (per-dataset, then averaged)...")
    mean_matrix, std_matrix = compute_cka_matrix(
        emb_dirs, layer_keys, model_names, common_datasets
    )

    # ── Print summary to stdout ──────────────────────────────────────────
    n = len(model_names)
    print(f"\nCKA Similarity (mean ± std, lower triangle):")
    header = f"{'':10s}" + "".join(f"{m:>12s}" for m in model_names)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{model_names[i]:10s}"
        for j in range(n):
            if i == j:
                row += f"{'1.00':>12s}"
            elif i > j:
                row += f"{mean_matrix[i, j]:>6.2f}±{std_matrix[i, j]:.2f}"
            else:
                row += f"{'--':>12s}"
        print(row)

    # ── Generate LaTeX ───────────────────────────────────────────────────
    tex = generate_latex_table(
        mean_matrix, std_matrix, model_names, len(common_datasets)
    )
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(tex)
    print(f"\nLaTeX table saved to {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
