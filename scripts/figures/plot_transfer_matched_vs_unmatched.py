#!/usr/bin/env python3
"""Transfer panel (matched vs unmatched): gap_closed toward strong vs top-m, for
strong concepts the weak model HAS a dictionary twin for vs ones it does NOT.

Sibling of plot_transfer_decomposition (the map-variation panel); same skeleton,
different question. Three no-search none arms (importance, dataset ordering):

  unmatched, via map    strong concepts with NO weak SAE twin, mapped in (tag
                        `ridge`)       -- the headline arm (SOLID, dominant).
  matched, via map      the MATCHED strong concepts, same ridge map/scale (tag
                        `ridge_matched`).
  matched, W's own atom amplify W's OWN usage of each matched concept by its own
                        activation, no map (tag `matched_wdec`) -- "turn up a
                        concept W already has", the underweighting operation.

If unmatched >> matched-map >~ matched-wdec, the gap is closed by concepts the
weak model lacks a readout for, NOT by re-weighting shared ones (underweighting
refuted). The on-disk greedy (per-row search) is the horizontal reference.

Reads the per-(tag,pair,dataset) gc npz from transfer_decomposition.py; reuses
the map-panel's pooling and greedy reference. Mac-only. Usage:
    python -m scripts.figures.plot_transfer_matched_vs_unmatched
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scripts._project_root import PROJECT_ROOT
from scripts.figures.plot_transfer_decomposition import (
    _pool, _greedy_disk, M_GRID, M_LABELS, GREEDY_COLOR,
)

OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "transfer_matched_vs_unmatched.pdf"

# (label, tag, arm, color, linestyle, marker) -- Okabe-Ito; SOLID reserved for the
# dominant (unmatched) series, matched arms dashed/dotted with distinct markers.
CURVES = [
    ("unmatched, via map",        "ridge",         "imp_ds", "#0072B2", "-",  "s"),
    ("matched, via map",          "ridge_matched", "imp_ds", "#D55E00", "--", "o"),
    ("matched, W's own decoder",  "matched_wdec",  "imp_ds", "#009E73", ":",  "^"),
]


def main():
    p = argparse.ArgumentParser(description="Transfer matched-vs-unmatched panel")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(4.2, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[3.4, 1.0], hspace=0.08),
        constrained_layout=True)
    # the "|U|" (all concepts) endpoint always sits to the RIGHT of the largest grid
    # point -- a fixed display slot, so the degradation reads left->right even when the
    # mean pool |U| is smaller than 50 (m=50 is then capped at |U|, same measurement).
    X_END = M_GRID[-1] * 1.7
    meanU, n_ds = None, 0
    for label, tag, arm, color, ls, mk in CURVES:
        med_d, frac_d, U = _pool(tag, arm)
        if med_d is None:
            print(f"  (no npz for tag {tag} yet -- skipping)")
            continue
        meanU = float(U.mean()); n_ds = max(n_ds, len(U))
        xs = np.array(list(M_GRID) + [X_END])
        med = np.array([np.median(med_d[lab]) for lab in M_LABELS])
        p25 = np.array([np.percentile(med_d[lab], 25) for lab in M_LABELS])
        p75 = np.array([np.percentile(med_d[lab], 75) for lab in M_LABELS])
        fneg = np.array([float(frac_d[lab].mean()) for lab in M_LABELS])
        if ls == "-":                       # IQR band on the dominant (unmatched) arm
            ax.fill_between(xs, p25, p75, color=color, alpha=0.10, linewidth=0)
            ax.plot(xs, p25, color=color, linewidth=0.7, alpha=0.4)
            ax.plot(xs, p75, color=color, linewidth=0.7, alpha=0.4)
        ax.plot(xs, med, color=color, linestyle=ls, marker=mk, markersize=4,
                linewidth=1.5, label=label)
        axs.plot(xs, fneg, color=color, linestyle=ls, marker=mk, markersize=3,
                 linewidth=1.2)
    if meanU is None:
        print("No matched/unmatched transfer npz on disk yet."); return

    gc, k = _greedy_disk()
    if gc is not None:
        ax.axhline(np.median(gc), color=GREEDY_COLOR, linewidth=1.3,
                   label="greedy (per-row search)")
        ax.axvline(np.median(k), color=GREEDY_COLOR, linewidth=0.9,
                   linestyle=":", alpha=0.5)

    ax.set_xscale("log")
    ax.set_ylim(-1.5, 1.5); ax.set_yticks([-1, 0, 1])
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="y", which="minor", length=2.5)
    ax.set_ylabel("gap closed", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.015, 0.97, "matched vs unmatched", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, columnspacing=1.1, handletextpad=0.4, framealpha=0.9)

    axs.set_ylim(0, 0.5)
    axs.set_ylabel("frac. wrong\ndirection", fontsize=8)
    axs.set_xticks(list(M_GRID) + [X_END])
    axs.set_xticklabels([str(m) for m in M_GRID] + [f"|U|\n≈{meanU:.0f}"], fontsize=8)
    axs.set_xlabel("concepts injected  (top-m, no search)", fontsize=9)
    axs.tick_params(labelsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}  ({n_ds} datasets)")


if __name__ == "__main__":
    main()
