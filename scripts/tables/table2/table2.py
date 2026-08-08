#!/usr/bin/env python3
"""
Table 2: Pairwise Procrustes residuals between tabular foundation models.

Computes normalized Procrustes distance for all model pairs from the round10
SAE training corpus (one pooled NPZ per model at its optimal extraction layer).
TabICL and HyperFast are classification-only; for the 8-model matrix the rows
are restricted to the 38 classification datasets that all eight models share.

Usage:
    python -m scripts.tables.table2.table2
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from analysis.similarity import procrustes_align

# (display_name, file_basename)
MODEL_CONFIGS = [
    ("TabPFN",    "tabpfn"),
    ("Mitra",     "mitra"),
    ("TabICL",    "tabicl"),
    ("TabICL-v2", "tabicl_v2"),
    ("TabDPT",    "tabdpt"),
    ("CARTE",     "carte"),
    ("HyperFast", "hyperfast"),
    ("Tabula-8B", "tabula8b"),
]

ROUND10_DIR = PROJECT_ROOT / "output" / "sae_training_round10"
OUTPUT_TEX = Path(__file__).parent / "procrustes_table.tex"


def load_round10(name: str) -> Dict:
    """Load a round10 SAE-test NPZ and return the embeddings keyed by
    (dataset, row_index) for cross-model row matching.
    """
    path = ROUND10_DIR / f"{name}_taskaware_sae_test.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    data = np.load(path, allow_pickle=True)

    # Build per-row (dataset, row_idx) keys by replaying samples_per_dataset.
    spd = data["samples_per_dataset"]
    row_indices = np.asarray(data["row_indices"])
    keys = []
    cursor = 0
    for entry in spd:
        ds = str(entry["dataset"])
        count = int(entry["count"])
        for k in range(count):
            keys.append((ds, int(row_indices[cursor + k])))
        cursor += count
    assert cursor == len(row_indices)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    assert len(keys) == embeddings.shape[0]
    return {"keys": keys, "embeddings": embeddings,
            "datasets": set(str(s) for s in data["source_datasets"])}


def aligned_rows(model_data: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, int]]]:
    """Restrict each model's embeddings to the rows present in *every* model.

    Row identity is the (dataset, row_index) tuple. The five all-task models
    cover 51 datasets; TabICL/HyperFast cover 38; Tabula-8B covers 51. The
    intersection is the 38 classification datasets, with each model
    contributing the rows it has on those datasets.
    """
    common_keys = None
    for name, d in model_data.items():
        keys = set(d["keys"])
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys_sorted = sorted(common_keys)

    aligned = {}
    for name, d in model_data.items():
        idx_map = {k: i for i, k in enumerate(d["keys"])}
        idx = np.fromiter((idx_map[k] for k in common_keys_sorted), dtype=int)
        aligned[name] = d["embeddings"][idx]
    return aligned, common_keys_sorted


def compute_procrustes_matrix(
    embeddings: Dict[str, np.ndarray],
    model_names: List[str],
) -> np.ndarray:
    n = len(model_names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance, _, _ = procrustes_align(
                embeddings[model_names[i]], embeddings[model_names[j]]
            )
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
    n = len(model_labels)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Procrustes disparity ($d^2 = 1 - \mathrm{{tr}}(R^\top A^\top B)$, "
        rf"$\in [0,1]$) after optimal orthogonal alignment. "
        rf"{n_datasets} TabArena classification datasets, {n_samples} pooled samples "
        r"(restricted to datasets shared by all 8 models).}",
        r"\label{tab:procrustes_matrix}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r" & " + " & ".join(rf"\rotatebox{{45}}{{{m}}}" for m in model_labels) + r" \\",
        r"\midrule",
    ]
    for i, name in enumerate(model_labels):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("0")
            elif i > j:
                cells.append(f"{matrix[i, j]:.4f}")
            else:
                cells.append("--")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main():
    print("Loading round10 SAE-test NPZs…")
    data = {}
    for display, fname in MODEL_CONFIGS:
        try:
            data[display] = load_round10(fname)
            n = data[display]["embeddings"].shape[0]
            d = data[display]["embeddings"].shape[1]
            print(f"  {display:<10s}: rows={n:>5d} dim={d:>5d} datasets={len(data[display]['datasets'])}")
        except FileNotFoundError as e:
            print(f"  {display:<10s}: MISSING ({e})")

    if len(data) < 2:
        sys.exit(1)

    aligned, keys = aligned_rows(data)
    common_datasets = sorted({k[0] for k in keys})
    n_datasets = len(common_datasets)
    n_samples = len(keys)
    print(f"\nAligned: {n_samples} rows across {n_datasets} datasets common to all {len(data)} models")

    print("\nComputing Procrustes…")
    names = list(aligned.keys())
    matrix = compute_procrustes_matrix(aligned, names)

    n = len(names)
    print(f"\nProcrustes Disparity d² ∈ [0,1] (lower triangle):")
    header = f"{'':10s}" + "".join(f"{m:>10s}" for m in names)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{names[i]:10s}"
        for j in range(n):
            if i == j:
                row += f"{'0':>10s}"
            elif i > j:
                row += f"{matrix[i, j]:>10.4f}"
            else:
                row += f"{'--':>10s}"
        print(row)

    tex = generate_latex_table(matrix, names, n_datasets, n_samples)
    OUTPUT_TEX.write_text(tex)
    print(f"\nSaved {OUTPUT_TEX}")
    paper_path = Path.home() / "src" / "tabular_embedding_paper" / "tables" / "procrustes_table.tex"
    if paper_path.parent.exists():
        paper_path.write_text(tex)
        print(f"Also wrote {paper_path}")


if __name__ == "__main__":
    main()
