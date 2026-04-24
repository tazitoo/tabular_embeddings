#!/usr/bin/env python3
"""Zoomed single-row L-shape scatter.

Same row used in row_intervention_figure (carte_vs_mitra, credit-g,
row 325), rendered in the visual style of the binary-classification
panel in plot_intervention_example_3panel but with a tight axis window
around the one L-shape and per-step diamonds with f_N annotations.

Reads the precomputed JSON produced by
scripts/paper/sec4/compute_row_intervention_data.py. No GPU needed.

Usage:
    python scripts/paper/sec4/row_intervention_scatter.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_figure_path

DATA_FILE = Path(__file__).parent / "row_intervention_data.json"
ABLATION_FILE = (
    PROJECT_ROOT / "output" / "ablation_figure_data"
    / "carte_vs_mitra" / "credit-g.npz"
)
TRANSFER_FILE = (
    PROJECT_ROOT / "output" / "transfer_figure_data"
    / "carte_vs_mitra" / "credit-g.npz"
)
OUTPUT_DIR = Path(__file__).parent
OUTPUT_NAME = "row_intervention_scatter.pdf"

# Same palette as plot_intervention_example_3panel.
BASE_COLOR = "#aaaaaa"
INTERV_COLOR = "black"
LINE_COLOR = "#666666"

DISPLAY_NAMES = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}


def _scalar(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(true class) for classification; 1-D scalar pass-through otherwise."""
    if preds.ndim == 2 and preds.shape[1] > 1:
        return preds[np.arange(len(y)), y]
    return preds.ravel()


def main():
    d = json.loads(DATA_FILE.read_text())

    p_strong = d["p_strong"]
    p_weak = d["p_weak"]
    ab_feats = d["ablation"]["features"]
    ab_preds = d["ablation"]["step_preds"]
    tr_feats = d["transfer"]["features"]
    tr_preds = d["transfer"]["step_preds"]
    strong = DISPLAY_NAMES.get(d["strong_model"], d["strong_model"])
    weak = DISPLAY_NAMES.get(d["weak_model"], d["weak_model"])

    # Context: load every row in this (pair, dataset) npz so we can
    # show baseline (P_strong, P_weak) points faintly behind the focal L.
    da = np.load(ABLATION_FILE, allow_pickle=True)
    y_q = da["y_query"].astype(int)
    ctx_p_strong = _scalar(da["preds_strong"], y_q)
    ctx_p_weak = _scalar(da["preds_weak"], y_q)

    # Axis window: tight around focal L (baseline + ablation + transfer).
    xs_all = [p_strong] + list(ab_preds)
    ys_all = [p_weak] + list(tr_preds)
    span = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all))
    pad = max(0.015, span * 0.15)
    x_lo = min(xs_all) - pad
    x_hi = max(xs_all) + pad
    y_lo = min(ys_all) - pad
    y_hi = max(ys_all) + pad
    lo = min(x_lo, y_lo)
    hi = max(x_hi, y_hi)

    fig, ax = plt.subplots(figsize=(3.8, 3.6), constrained_layout=True)

    # y = x diagonal.
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4, zorder=1)

    # Background baselines for every row, same style as the focal
    # non-intervened dot below (only those falling inside the zoom
    # window will render).
    ax.plot(ctx_p_strong, ctx_p_weak, "o", color=BASE_COLOR,
            markeredgecolor="none", markersize=3.5, linestyle="none",
            zorder=2)

    # Focal-row baseline marker at the L corner — same grey as the
    # background scatter (it has not been intervened on). Drawn at
    # markersize=3.5 with no edge so its visual diameter matches the
    # black intervention dots (markersize=2.5 + default 1pt edge ≈ 3.5).
    ax.plot(p_strong, p_weak, "o", color=BASE_COLOR,
            markeredgecolor="none", markersize=3.5, zorder=4)

    # Ablation walk: x decreases, y stays at p_weak.
    prev_x = p_strong
    for feat, pred in zip(ab_feats, ab_preds):
        ax.annotate(
            "", xy=(pred, p_weak), xytext=(prev_x, p_weak),
            arrowprops=dict(arrowstyle="-|>", color=LINE_COLOR, lw=1.0,
                            shrinkA=4, shrinkB=3),
        )
        ax.plot(pred, p_weak, "o", color=INTERV_COLOR, markersize=2.5,
                zorder=4)
        mid_x = (prev_x + pred) / 2
        ax.annotate(
            f"$f_{{{feat}}}$", (mid_x, p_weak),
            xytext=(0, -4), textcoords="offset points",
            fontsize=9, color="black", ha="center", va="top",
        )
        prev_x = pred

    # Transfer walk: y increases, x stays at p_strong. Only the first
    # step gets an arrow — later steps' Δ is too small to draw a
    # legible arrow (dots collide, arrowhead gets clipped). Their dots
    # + labels still render to show the picked features.
    prev_y = p_weak
    n_tr = len(tr_feats)
    for i, (feat, pred) in enumerate(zip(tr_feats, tr_preds)):
        if i == 0:
            ax.annotate(
                "", xy=(p_strong, pred), xytext=(p_strong, prev_y),
                arrowprops=dict(arrowstyle="-|>", color=LINE_COLOR, lw=1.0,
                                shrinkA=4, shrinkB=3),
            )
        ax.plot(p_strong, pred, "o", color=INTERV_COLOR, markersize=2.5,
                zorder=4)
        mid_y = (prev_y + pred) / 2
        # Last transfer step (f_36) is raised above its diamond to avoid
        # overlap with the preceding f_86 label when successive steps
        # produce very small Δ (arrows crowd together near the top).
        if i == n_tr - 1 and n_tr >= 3:
            ax.annotate(
                f"$f_{{{feat}}}$", (p_strong, pred),
                xytext=(7, 4), textcoords="offset points",
                fontsize=9, color="black", ha="left", va="center",
            )
        else:
            ax.annotate(
                f"$f_{{{feat}}}$", (p_strong, mid_y),
                xytext=(7, 0), textcoords="offset points",
                fontsize=9, color="black", ha="left", va="center",
            )
        prev_y = pred

    # Axis labels: x=strong (Mitra), y=weak (CARTE).
    ax.set_xlabel(f"{strong} prediction", fontsize=9)
    ax.set_ylabel(f"{weak} prediction", fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    # Keep y ticks on the same grid as x ticks (square axes, shared range).
    xticks = [t for t in ax.get_xticks() if lo <= t <= hi]
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(length=2, pad=1)
    ax.grid(True, which="major", ls=":", alpha=0.3)
    ax.text(
        0.03, 0.97, "(a)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold",
    )

    local_path = OUTPUT_DIR / OUTPUT_NAME
    paper_path = paper_figure_path("4_results", OUTPUT_NAME)
    for path in (local_path, paper_path):
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
