#!/usr/bin/env python3
"""Transfer panel: gap_closed toward strong vs top-m, as the MAP varies.

Sibling of plot_pipeline_knob_ablation (the ablation panel); the two stack as
LaTeX subfigures. Where the ablation panel shows what closes the strong->weak
gap, this shows whether the learned cross-model MAP earns its keep: it plots the
no-search none arm (importance, dataset) under each map -- learned ridge (M1),
shuffled-landmark ridge (M2), random orthogonal (M4) -- plus the random-
dictionary ridge (T5) and the per-row ridge arm (T2), against top-m on a log-m,
symlog-gc axis. The on-disk greedy (T1) is the horizontal reference with its
median k marked. If ridge ~ shuffled ~ random, the gap-closing is the search,
not the map (the negative-R^2 / near-null correspondence finding).

Reads the per-(tag,pair,dataset) npz from transfer_decomposition.py. Mac-only.
Usage:
    python -m scripts.figures.plot_transfer_decomposition
    python -m scripts.figures.plot_transfer_decomposition --yscale symlog
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts._project_root import PROJECT_ROOT
from scripts.round_paths import FORWARD_ABLATION_DIR, FORWARD_DELTAS_DIR

TD_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "transfer_decomposition"
OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "transfer_decomposition.pdf"
M_GRID = (1, 2, 5, 10, 20, 50)
M_LABELS = [str(m) for m in M_GRID] + ["|U|"]

# (label, tag, arm, color, linestyle, marker)
CURVES = [
    ("M1 ridge (imp, dataset)",     "ridge",       "imp_ds",  "#0072B2", "-",  "s"),
    ("M2 shuffled (imp, dataset)",  "shuffled",    "imp_ds",  "#0072B2", "--", "s"),
    ("M4 random-orth (imp, dataset)","random_orth","imp_ds",  "#009E73", ":",  "^"),
    ("T5 ridge, random dict",       "ridge_random","imp_ds",  "#999999", "--", "o"),
    ("T2 ridge (imp, row)",         "ridge",       "imp_row", "#D55E00", "-",  "o"),
]
GREEDY_COLOR = "#000000"


def _pool(tag, arm):
    root = TD_DIR / tag
    if not root.exists():
        return None, None
    cols = {lab: [] for lab in M_LABELS}
    U = []
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if f"{arm}__1" not in z:
                continue
            for lab in M_LABELS:
                cols[lab].append(z[f"{arm}__{lab}"])
            U.append(np.full(len(z[f"{arm}__1"]), len(z["uni_pool_ds"])))
    if not U:
        return None, None
    return {lab: np.concatenate(v) for lab, v in cols.items()}, np.concatenate(U)


def _greedy_disk():
    gc, k = [], []
    if not FORWARD_DELTAS_DIR.exists():
        return None, None
    for pdir in sorted(d for d in FORWARD_DELTAS_DIR.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "optimal_k" not in z:
                continue
            kk = z["optimal_k"].astype(int); act = kk > 0
            if act.sum():
                gc.append(z["gap_closed"][act]); k.append(kk[act])
    if not gc:
        return None, None
    return np.concatenate(gc), np.concatenate(k)


def main():
    p = argparse.ArgumentParser(description="Transfer decomposition panel")
    p.add_argument("--yscale", choices=["linear", "symlog"], default="symlog")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    # median+IQR gc on top, frac toward W (gc<0) strip below (shared log-x)
    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(4.2, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[3.4, 1.0], hspace=0.08),
        constrained_layout=True)
    meanU = None
    n_ds = 0
    for label, tag, arm, color, ls, mk in CURVES:
        cols, U = _pool(tag, arm)
        if cols is None:
            continue
        meanU = float(U.mean())
        n_ds = max(n_ds, len(U))
        xs = np.array(list(M_GRID) + [meanU])
        med = np.array([np.median(cols[lab]) for lab in M_LABELS])
        p25 = np.array([np.percentile(cols[lab], 25) for lab in M_LABELS])
        p75 = np.array([np.percentile(cols[lab], 75) for lab in M_LABELS])
        fneg = np.array([float((cols[lab] < 0).mean()) for lab in M_LABELS])
        # Per-row transfer gc is heavy-tailed; a solid IQR block muddies. Trace it
        # instead with a very light fill + thin p25/p75 edge lines, and only on the
        # two solid primary arms (M1 ridge, T2) to keep the panel clean.
        if ls == "-":
            ax.fill_between(xs, p25, p75, color=color, alpha=0.06, linewidth=0)
            ax.plot(xs, p25, color=color, linewidth=0.7, alpha=0.4)
            ax.plot(xs, p75, color=color, linewidth=0.7, alpha=0.4)
        ax.plot(xs, med, color=color, linestyle=ls, marker=mk, markersize=4,
                linewidth=1.5, label=label)
        axs.plot(xs, fneg, color=color, linestyle=ls, marker=mk, markersize=3,
                 linewidth=1.2)
    if meanU is None:
        print(f"No transfer npz under {TD_DIR} yet.")
        return

    gc, k = _greedy_disk()
    if gc is not None:
        ax.axhline(np.median(gc), color=GREEDY_COLOR, linewidth=1.3,
                   label="T1 greedy ridge (on disk)")
        ax.axvline(np.median(k), color=GREEDY_COLOR, linewidth=0.9,
                   linestyle=":", alpha=0.5)   # full-height median-k guide

    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=(0, (1, 3)), alpha=0.4)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=(0, (1, 3)), alpha=0.3)
    ax.set_xscale("log")
    # symlog so the p25/p75 edges (~[-6,+2.6]) fit while linscale gives the [-1,1]
    # band (where the medians and greedy ref live) room to read.
    from matplotlib.ticker import SymmetricalLogLocator
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.1)
    ax.set_ylim(-2.5, 2.5)
    ax.yaxis.set_minor_locator(SymmetricalLogLocator(
        base=10.0, linthresh=1.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
    ax.tick_params(axis="y", which="minor", length=2.5)
    ax.set_ylabel("gap closed", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.015, 0.97, "b) transfer", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, fontweight="bold")
    # legend above the panel in a few rows (title/counts/caveats go in the caption:
    # map varies, dataset-sign none arms, mitra-recipient datasets excluded)
    ax.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, columnspacing=1.1, handletextpad=0.4, framealpha=0.9)

    axs.set_ylim(0, 0.5)
    axs.set_ylabel("frac. wrong\ndirection", fontsize=8)
    axs.set_xticks(list(M_GRID) + [meanU])
    axs.set_xticklabels([str(m) for m in M_GRID] + [f"|U|\n≈{meanU:.0f}"], fontsize=8)
    axs.set_xlabel("concepts injected  (top-m, no search)", fontsize=9)
    axs.tick_params(labelsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}  ({n_ds} datasets)")
    if gc is not None:
        print(f"T1 greedy ridge: median gc={np.median(gc):.3f}  median k={np.median(k):.0f}")


if __name__ == "__main__":
    main()
