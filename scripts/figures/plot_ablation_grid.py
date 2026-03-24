#!/usr/bin/env python3
"""Plot all 28 pairwise ablation comparisons for one dataset on a single page.

Reads ablation_sweep NPZ files for all model pairs and tiles them into a
7×4 grid on an 8.5×11 page. Each panel shows the strong model's predictions
(x-axis) vs the weak model's (y-axis), with ablation arrows showing how
removing concepts degrades the strong model toward the weak.

Usage:
    python -m scripts.figures.plot_ablation_grid --dataset credit-g
    python -m scripts.figures.plot_ablation_grid --dataset credit-g --sweep-dir output/ablation_sweep
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.concept_performance_diagnostic import DISPLAY_NAMES

MODELS = ["tabpfn", "mitra", "tabicl", "tabicl_v2", "tabdpt", "hyperfast", "carte", "tabula8b"]
SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

NCOLS = 4
NROWS = 7


def _draw_panel(ax, npz_path: Path):
    """Draw one ablation scatter panel on the given axes."""
    data = np.load(npz_path, allow_pickle=True)

    # Handle degenerate results (n_strong_wins=0, no predictions saved)
    if "preds_strong" not in data:
        strong_model = str(data["strong_model"]) if "strong_model" in data else "?"
        weak_model = str(data["weak_model"]) if "weak_model" in data else "?"
        disp_s = DISPLAY_NAMES.get(strong_model, strong_model)
        disp_w = DISPLAY_NAMES.get(weak_model, weak_model)
        ax.text(0.5, 0.5, "tied", ha="center", va="center",
                fontsize=6, color="#999999", transform=ax.transAxes)
        ax.set_title(f"{disp_s} vs {disp_w}", fontsize=6, pad=2, color="#999999")
        ax.tick_params(labelsize=4, length=2, pad=1)
        return

    preds_s = data["preds_strong"]
    preds_w = data["preds_weak"]
    preds_i = data["preds_intervened"]
    y = data["y_query"].astype(int)
    optimal_k = data["optimal_k"]
    strong_wins = data["strong_wins"]

    strong_model = str(data["strong_model"]) if "strong_model" in data else None
    weak_model = str(data["weak_model"]) if "weak_model" in data else None

    # Fall back to parsing dir name if not in NPZ
    if not strong_model:
        pair_name = npz_path.parent.name
        parts = pair_name.split("_vs_")
        strong_model = parts[0]
        weak_model = parts[1] if len(parts) == 2 else "?"

    disp_s = DISPLAY_NAMES.get(strong_model, strong_model)
    disp_w = DISPLAY_NAMES.get(weak_model, weak_model)

    # Scalar scores for axes
    n_classes = preds_s.shape[1] if preds_s.ndim == 2 else 1
    if n_classes > 1:
        p_s = preds_s[np.arange(len(y)), y]
        p_w = preds_w[np.arange(len(y)), y]
        p_i = preds_i[np.arange(len(y)), y]
    else:
        p_s = preds_s.ravel()
        p_w = preds_w.ravel()
        p_i = preds_i.ravel()

    n_strong_wins = int(strong_wins.sum())
    modified = optimal_k > 0
    n_intervened = int((strong_wins & modified).sum())
    valid_k = optimal_k[strong_wins & modified]

    # --- Draw layers ---
    # All rows: gray with transparency
    ax.scatter(p_s, p_w, c="#999999", s=4, alpha=0.3, edgecolors="none", zorder=2)

    # Intervened positions: small black dots
    sw_mod = strong_wins & modified
    if sw_mod.any():
        ax.scatter(p_i[sw_mod], p_w[sw_mod], c="black", s=1, alpha=0.7,
                   edgecolors="none", marker=".", zorder=4)

    # y=x line
    lo, hi = 0, 1
    if n_classes == 1:
        all_vals = np.concatenate([p_s, p_w, p_i])
        lo = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
        hi = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Labels
    k_str = f"k\u0303={np.median(valid_k):.0f}" if len(valid_k) else "—"
    ax.set_title(f"{disp_s} \u2192 {disp_w}  ({n_intervened}/{n_strong_wins})  {k_str}",
                 fontsize=6, pad=2)
    ax.tick_params(labelsize=4, length=2, pad=1)


def _draw_empty_panel(ax, model_a: str, model_b: str):
    """Draw placeholder for missing data."""
    disp_a = DISPLAY_NAMES.get(model_a, model_a)
    disp_b = DISPLAY_NAMES.get(model_b, model_b)
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            fontsize=6, color="#999999", transform=ax.transAxes)
    ax.set_title(f"{disp_a} vs {disp_b}", fontsize=6, pad=2, color="#999999")
    ax.tick_params(labelsize=4, length=2, pad=1)


def main():
    parser = argparse.ArgumentParser(
        description="Plot all 28 pairwise ablation comparisons for one dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to include (default: all 8)")
    args = parser.parse_args()

    pairs = list(combinations(args.models, 2))
    n_pairs = len(pairs)
    ncols = NCOLS
    nrows = (n_pairs + ncols - 1) // ncols

    # 8.5 x 11 inches (letter)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 11))
    axes = axes.flatten()

    found = 0
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx]
        # Try sorted pair name (new format)
        pair_sorted = f"{min(a,b)}_vs_{max(a,b)}"
        npz_path = args.sweep_dir / pair_sorted / f"{args.dataset}.npz"

        # Also try both orderings (old format)
        if not npz_path.exists():
            npz_path = args.sweep_dir / f"{a}_vs_{b}" / f"{args.dataset}.npz"
        if not npz_path.exists():
            npz_path = args.sweep_dir / f"{b}_vs_{a}" / f"{args.dataset}.npz"

        if npz_path.exists():
            _draw_panel(ax, npz_path)
            found += 1
        else:
            _draw_empty_panel(ax, a, b)

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{args.dataset} — pairwise ablation ({found}/{n_pairs} pairs)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=0.8, w_pad=0.5)

    output_path = args.output or OUTPUT_DIR / f"ablation_grid_{args.dataset}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} ({found}/{n_pairs} pairs)")


if __name__ == "__main__":
    main()
