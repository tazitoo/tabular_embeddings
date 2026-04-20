#!/usr/bin/env python3
"""
Rank-ordered LOO importance decay curves, split by matched vs unmatched concepts.

For each model M (6 main models, excluding HyperFast and Tabula-8B):
    1. Load every row of `output/perrow_importance/{model}/{dataset}.npz`
       restricted to TabArena classification datasets (38 of 51),
       so the y-axis is a consistent prob-space |Delta pred|.
    2. Partition alive features into matched vs unmatched using
       `output/sae_feature_matching_mnn_floor_p90.json`:
       a feature is *matched* if it has >=1 MNN match with any other
       main model (model-level union); otherwise *unmatched*.
    3. Sort |drops| descending within each row, accumulate mean |drop|
       by rank position across all (row, dataset) pairs.

Plot: 2 panels (matched left, unmatched right), shared log y-axis,
x = raw rank 1..max_alive, one line per model.

Usage:
    python -m scripts.figures.4_results.plot_importance_decay
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.extended_loader import TABARENA_DATASETS
from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_figure_path

IMPORTANCE_DIR = PROJECT_ROOT / "output" / "perrow_importance"
MATCHING_FILE = PROJECT_ROOT / "output" / "sae_feature_matching_mnn_floor_p90.json"
OUT_DIR = PROJECT_ROOT / "output" / "figures"

MODELS = ["carte", "mitra", "tabdpt", "tabicl", "tabicl_v2", "tabpfn"]
DISPLAY = {
    "carte": "CARTE",
    "mitra": "Mitra",
    "tabdpt": "TabDPT",
    "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2",
    "tabpfn": "TabPFN",
}
EXCLUDED_FROM_MAIN = {"HyperFast", "Tabula-8B"}

# Okabe-Ito colour-blind-safe palette. Mitra gets the darkest blue because
# it is the dominant line in both panels.
MODEL_COLORS = {
    "mitra":     "#0072B2",  # blue (dominant)
    "tabpfn":    "#D55E00",  # vermillion
    "carte":     "#CC79A7",  # reddish purple
    "tabdpt":    "#009E73",  # bluish green
    "tabicl":    "#E69F00",  # orange
    "tabicl_v2": "#56B4E9",  # sky blue
}

# Secondary differentiator so the plot reads in B&W / for colour-blind users.
# Solid reserved for Mitra (the dominant series).
MODEL_LINESTYLE = {
    "mitra":     "-",                              # solid (dominant)
    "tabpfn":    "--",                             # dashed
    "carte":     "-.",                             # dashdot
    "tabdpt":    ":",                              # dotted
    "tabicl":    (0, (3, 1, 1, 1)),                # short dash + short dot
    "tabicl_v2": (0, (5, 1, 1, 1, 1, 1)),          # long dash + double dot
}

CLS_DATASETS = {
    name for name, meta in TABARENA_DATASETS.items()
    if meta.get("task") == "classification"
}


def build_matched_sets(matching_path: Path) -> dict[str, set[int]]:
    """Union of matched feature indices per model across all main-model pairs."""
    data = json.loads(matching_path.read_text())
    display_to_code = {v: k for k, v in DISPLAY.items()}
    matched: dict[str, set[int]] = {m: set() for m in MODELS}
    for pair_key, pair_data in data["pairs"].items():
        a, b = pair_key.split("__")
        if a in EXCLUDED_FROM_MAIN or b in EXCLUDED_FROM_MAIN:
            continue
        a_code = display_to_code.get(a)
        b_code = display_to_code.get(b)
        if a_code is None or b_code is None:
            continue
        for m in pair_data.get("matches", []):
            matched[a_code].add(int(m["idx_a"]))
            matched[b_code].add(int(m["idx_b"]))
    return matched


def accumulate_rank_curve(
    model: str,
    matched_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        matched_mean, unmatched_mean — mean |Delta pred| at each rank position.
        Length of each array is the max alive-subset width encountered
        across (row, dataset) for that partition.
    """
    sums_m: dict[int, float] = defaultdict(float)
    counts_m: dict[int, int] = defaultdict(int)
    sums_u: dict[int, float] = defaultdict(float)
    counts_u: dict[int, int] = defaultdict(int)

    model_dir = IMPORTANCE_DIR / model
    for npz_path in sorted(model_dir.glob("*.npz")):
        if npz_path.stem not in CLS_DATASETS:
            continue
        data = np.load(npz_path)
        drops = np.abs(data["row_feature_drops"]).astype(np.float64)
        feat_idx = data["feature_indices"]

        is_matched = np.array([int(f) in matched_set for f in feat_idx], dtype=bool)
        matched_drops = drops[:, is_matched]
        unmatched_drops = drops[:, ~is_matched]

        if matched_drops.shape[1] > 0:
            sorted_m = -np.sort(-matched_drops, axis=1)
            # column-wise sums/counts
            col_sum = sorted_m.sum(axis=0)
            col_cnt = np.full(sorted_m.shape[1], sorted_m.shape[0], dtype=np.int64)
            for k in range(sorted_m.shape[1]):
                sums_m[k] += col_sum[k]
                counts_m[k] += col_cnt[k]

        if unmatched_drops.shape[1] > 0:
            sorted_u = -np.sort(-unmatched_drops, axis=1)
            col_sum = sorted_u.sum(axis=0)
            col_cnt = np.full(sorted_u.shape[1], sorted_u.shape[0], dtype=np.int64)
            for k in range(sorted_u.shape[1]):
                sums_u[k] += col_sum[k]
                counts_u[k] += col_cnt[k]

    def to_array(sums: dict, counts: dict) -> np.ndarray:
        if not counts:
            return np.array([])
        max_k = max(counts) + 1
        out = np.full(max_k, np.nan)
        for k in range(max_k):
            if counts[k] > 0:
                out[k] = sums[k] / counts[k]
        return out

    return to_array(sums_m, counts_m), to_array(sums_u, counts_u)


def plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    order: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.4), sharey=True)

    for model in order:
        matched_curve, unmatched_curve = curves[model]
        color = MODEL_COLORS[model]
        ls = MODEL_LINESTYLE[model]
        label = DISPLAY[model]

        if matched_curve.size:
            x = np.arange(1, matched_curve.size + 1)
            axes[0].plot(x, matched_curve, color=color, linestyle=ls,
                         label=label, lw=1.6)
        if unmatched_curve.size:
            x = np.arange(1, unmatched_curve.size + 1)
            axes[1].plot(x, unmatched_curve, color=color, linestyle=ls,
                         label=label, lw=1.6)

    for ax, title in zip(axes, ["Matched concepts", "Unmatched concepts"]):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e-4, 1.5e0)
        ax.set_xlabel("Rank descending")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("Mean $|\\Delta\\text{pred}|$ (LOO)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=1, frameon=True)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUT_DIR / "importance_decay.pdf"
    paper_path = paper_figure_path("4_results", "importance_decay.pdf")
    for path in (local_path, paper_path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")


def main() -> None:
    matched_sets = build_matched_sets(MATCHING_FILE)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        m, u = accumulate_rank_curve(model, matched_sets[model])
        print(
            f"{DISPLAY[model]:10s}  matched: "
            f"{len(matched_sets[model])} feats, curve len {m.size}  "
            f"|  unmatched curve len {u.size}"
        )
        curves[model] = (m, u)

    # Order by top-1 matched |Delta pred| descending so the legend
    # reads in the same order as the curves stack at rank 1.
    order = sorted(
        MODELS,
        key=lambda m: curves[m][0][0] if curves[m][0].size else -1.0,
        reverse=True,
    )
    print("Plot order (matched top-1 desc):",
          ", ".join(f"{DISPLAY[m]} ({curves[m][0][0]:.3f})" for m in order))
    plot(curves, order)


if __name__ == "__main__":
    main()
