#!/usr/bin/env python3
"""Combined ablation + transfer example: binary cls, multi-class cls, regression.

Each row (where the strong model beat the weak model AND both interventions
modified the prediction) is drawn as an L-shape:
    - grey circle at (strong P(correct), weak P(correct)) — baseline
    - leftward arrow to the ablated point (strong's P(correct) after
      ablating unmatched strong concepts)
    - upward arrow to the transferred point (weak's P(correct) after
      injecting mapped strong concepts)

Both arrow endpoints sit closer to the y=x diagonal when the intervention
succeeds, so successful rows form a right-angled dog-leg pointing toward
the diagonal from below.

Default pair carte_vs_mitra: Mitra consistently strong across all three
task types, deterministic with seed=13.

Default datasets are all Mitra-strong (so the x-axis label is consistent
across panels):
    credit-g                                  (binary, n=59)
    students_dropout_and_academic_success     (3-class, n=118)
    houses                                    (regression, n=130)

Usage:
    python -m scripts.figures.plot_intervention_example_3panel
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

DISPLAY_NAMES = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

DEFAULT_PAIR = "carte_vs_mitra"
DEFAULT_DATASETS = [
    "credit-g",                              # binary (59, Mitra strong)
    "students_dropout_and_academic_success",  # 3-class (118, Mitra strong)
    "houses",                                # regression (130, Mitra strong)
]

# Same visual language as plot_ablation_grid._draw_panel: grey baseline
# points and small black post-intervention points. Direction (left for
# ablation, up for transfer) is read off the line segments, not colour.
BASE_COLOR = "#aaaaaa"
INTERV_COLOR = "black"
LINE_COLOR = "#666666"


def _scalar(preds, y):
    if preds.ndim == 2 and preds.shape[1] > 1:
        return preds[np.arange(len(y)), y]
    return preds.ravel()


def draw_intervention_cell(ax, ablation_path: Path, transfer_path: Path,
                           marker_size: float = 6.0, base_size: float = 10.0):
    """Draw the combined-intervention L-shape scatter onto `ax`.

    No title / tag / axis labels — the caller is responsible for those,
    so this function can be reused by both the 3-panel example script and
    the per-dataset grid script.

    Returns (n_drawn, n_strong_wins, strong_model, weak_model, is_regression).
    """
    d_a = np.load(ablation_path, allow_pickle=True)
    d_t = np.load(transfer_path, allow_pickle=True)

    y = d_a["y_query"].astype(int)
    p_s = _scalar(d_a["preds_strong"], y)
    p_w = _scalar(d_a["preds_weak"], y)
    # Both files use 'preds_intervened' — in ablation_sweep it's the strong
    # model after ablating its concepts (x-coordinate changes); in
    # transfer_sweep it's the weak model after injecting mapped strong
    # concepts (y-coordinate changes).
    p_abl = _scalar(d_a["preds_intervened"], y)
    p_trf = _scalar(d_t["preds_intervened"], y)

    strong_wins = d_a["strong_wins"]
    ok_a = d_a["optimal_k"]
    ok_t = d_t["optimal_k"]
    mask = strong_wins & (ok_a > 0) & (ok_t > 0)

    is_regression = d_a["preds_strong"].ndim == 1 or (
        d_a["preds_strong"].ndim == 2 and d_a["preds_strong"].shape[1] == 1
    )
    if is_regression:
        all_vals = np.concatenate([p_s, p_w, p_abl, p_trf])
        lo = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
        hi = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
    else:
        lo, hi = 0, 1

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4, zorder=1)

    if not is_regression:
        event_rate = y.mean()
        ax.axhline(event_rate, color="#dddddd", lw=0.5, ls=":", zorder=1)
        ax.axvline(event_rate, color="#dddddd", lw=0.5, ls=":", zorder=1)

    ax.scatter(p_s, p_w, c=BASE_COLOR, s=base_size, alpha=0.35,
               edgecolors="none", zorder=2)

    for i in np.flatnonzero(mask):
        ax.plot([p_s[i], p_abl[i]], [p_w[i], p_w[i]],
                color=LINE_COLOR, lw=0.5, alpha=0.4, zorder=3)
        ax.plot([p_s[i], p_s[i]], [p_w[i], p_trf[i]],
                color=LINE_COLOR, lw=0.5, alpha=0.4, zorder=3)

    ax.scatter(p_abl[mask], p_w[mask], c=INTERV_COLOR, s=marker_size,
               alpha=0.7, edgecolors="none", zorder=4)
    ax.scatter(p_s[mask], p_trf[mask], c=INTERV_COLOR, s=marker_size,
               alpha=0.7, edgecolors="none", zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    return (int(mask.sum()), int(strong_wins.sum()),
            str(d_a["strong_model"]), str(d_a["weak_model"]), is_regression)


def draw_panel(ax, ablation_path: Path, transfer_path: Path, tag: str, title: str):
    n_drawn, n_wins, *_ = draw_intervention_cell(
        ax, ablation_path, transfer_path,
        marker_size=6.0, base_size=10.0,
    )
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.text(
        0.03, 0.97, tag,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold",
    )
    return n_drawn, n_wins


def main():
    parser = argparse.ArgumentParser(
        description="Combined ablation+transfer example across task types")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--datasets", nargs=3, default=None)
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--transfer-dir", type=Path, default=TRANSFER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    datasets = args.datasets or DEFAULT_DATASETS

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    panel_tags = ["(a)", "(b)", "(c)"]

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ap = args.ablation_dir / args.pair / f"{ds}.npz"
        tp = args.transfer_dir / args.pair / f"{ds}.npz"
        if not ap.exists() or not tp.exists():
            ax.text(0.5, 0.5, f"no data\n{ds}", ha="center", va="center",
                    fontsize=8, color="#999999", transform=ax.transAxes)
            ax.set_title(ds, fontsize=9)
            continue

        n_arrows, n_wins = draw_panel(ax, ap, tp, panel_tags[idx], ds)

        d = np.load(ap, allow_pickle=True)
        disp_s = DISPLAY_NAMES.get(str(d["strong_model"]), str(d["strong_model"]))
        disp_w = DISPLAY_NAMES.get(str(d["weak_model"]), str(d["weak_model"]))
        ax.set_xlabel(disp_s, fontsize=9)
        if idx == 0:
            ax.set_ylabel(disp_w, fontsize=9)

        print(f"{ds}: {n_arrows}/{n_wins} rows drawn")

    # Legend inside panel (a), lower-right
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_COLOR,
               markersize=6, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTERV_COLOR,
               markersize=5, label="post-intervention"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7,
                   frameon=True, framealpha=0.9)

    fig.tight_layout()

    output = args.output or OUTPUT_DIR / "intervention_example_3panel.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output} and .png")


if __name__ == "__main__":
    main()
