#!/usr/bin/env python3
"""Two-by-two importance decay grid.

Top row: LOO importance decay, matched (left) and unmatched (right),
same data as plot_importance_decay.py.

Bottom row: placeholders for the ablation-replay and transfer-replay
step-Δpred curves, which will be populated once
    - scripts/intervention/replay_ablation_step_preds.py has been run
      across all 15 pairs and
    - the transfer re-sweep finishes with the new step_preds save
      (scripts/intervention/transfer_sweep_v2.py).

Empty frames are drawn deliberately so the missing panels appear in the
rendered figure as a visible to-do reminder.

Usage:
    python -m scripts.figures.4_results.plot_importance_decay_grid
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import importlib

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_figure_path

# Python identifiers can't start with a digit, so `4_results` can't appear
# in a regular import statement — use importlib to pull the helpers from
# the sibling plot_importance_decay module.
_decay = importlib.import_module(
    "scripts.figures.4_results.plot_importance_decay"
)
MODELS = _decay.MODELS
DISPLAY = _decay.DISPLAY
MODEL_COLORS = _decay.MODEL_COLORS
MODEL_LINESTYLE = _decay.MODEL_LINESTYLE
MATCHING_FILE = _decay.MATCHING_FILE
build_matched_sets = _decay.build_matched_sets
accumulate_rank_curve = _decay.accumulate_rank_curve

OUT_DIR = PROJECT_ROOT / "output" / "figures"


def _style_axes(ax, title: str) -> None:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5e0)
    ax.set_xlabel("Rank descending")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def _draw_placeholder(ax, title: str, message: str) -> None:
    _style_axes(ax, title)
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9, color="#888888", style="italic",
    )
    # Log-log axes need a positive dummy point to render tick grid.
    ax.plot([1], [1e-2], alpha=0)


def main() -> None:
    matched_sets = build_matched_sets(MATCHING_FILE)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        m, u = accumulate_rank_curve(model, matched_sets[model])
        curves[model] = (m, u)
        print(
            f"{DISPLAY[model]:10s}  matched curve len {m.size}  "
            f"unmatched curve len {u.size}"
        )

    order = sorted(
        MODELS,
        key=lambda m: curves[m][0][0] if curves[m][0].size else -1.0,
        reverse=True,
    )
    print("Plot order:", ", ".join(DISPLAY[m] for m in order))

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharey=True)
    top_left, top_right = axes[0]
    bot_left, bot_right = axes[1]

    for model in order:
        m_curve, u_curve = curves[model]
        color = MODEL_COLORS[model]
        ls = MODEL_LINESTYLE[model]
        label = DISPLAY[model]
        if m_curve.size:
            x = np.arange(1, m_curve.size + 1)
            top_left.plot(x, m_curve, color=color, linestyle=ls,
                          lw=1.6, label=label)
        if u_curve.size:
            x = np.arange(1, u_curve.size + 1)
            top_right.plot(x, u_curve, color=color, linestyle=ls,
                           lw=1.6, label=label)

    _style_axes(top_left, "(a) LOO decay — matched concepts")
    _style_axes(top_right, "(b) LOO decay — unmatched concepts")
    _draw_placeholder(
        bot_left,
        "(c) Ablation step $|\\Delta\\text{pred}|$",
        "pending: replay_ablation_step_preds\nacross 15 pairs",
    )
    _draw_placeholder(
        bot_right,
        "(d) Transfer step $|\\Delta\\text{pred}|$",
        "pending: transfer re-sweep\nwith step_preds save",
    )

    axes[0, 0].set_ylabel("Mean $|\\Delta\\text{pred}|$ (LOO)")
    axes[1, 0].set_ylabel("Mean $|\\Delta\\text{pred}|$ (step)")
    top_left.legend(fontsize=8, loc="upper right", ncol=1, frameon=True)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUT_DIR / "importance_decay_grid.pdf"
    paper_path = paper_figure_path("4_results", "importance_decay_grid.pdf")
    for path in (local_path, paper_path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
