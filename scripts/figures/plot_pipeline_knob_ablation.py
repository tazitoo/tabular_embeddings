#!/usr/bin/env python3
"""Ablation panel: gap_closed vs top-m for the pipeline-knob decomposition.

One panel of the two-panel pipeline figure (the transfer panel is a sibling,
combined as LaTeX subfigures). It shows what actually closes the strong->weak
gap by varying one pipeline knob at a time, holding model/pair/pool fixed.

x-axis  : m, the number of concepts ablated (log scale), 1..50 then the per-
          dataset filtered-pool endpoint |U| (plotted at the row-weighted mean
          |U|). y-axis: UNCLIPPED gap_closed (raw moved/gap, pooled row-weighted
          over every active row across datasets/pairs). Unclipped is the fair
          readout for the no-search "none" arms: they have no goal, so the
          paper's min(1,max(0,.)) clamp only hides the overshoot that ranking-
          by-damage produces at small m (see pipeline_ablation.py).

Curves (the three no-search "none" arms, trained SAE solid / random SAE dashed):
    importance, row,     none   per-row importance ranking, top-m
    importance, dataset, none   one shared importance-ranked set, top-m
    uniform,    dataset, none   one shared random set, top-m (seed-averaged)
Reference lines (searched arms, no m-axis -- greedy picks its own k per row):
    paper greedy (importance, row, greedy)   horizontal at its reported gc,
                                             median k marked on the x-axis
    uniform, row, greedy                     the uninformed search
    random-SAE greedy                        the random control's greedy

Reads the per-(pair,dataset) npz from pipeline_ablation.py (trained + --random).
Mac-only, no compute. Usage:
    python -m scripts.figures.plot_pipeline_knob_ablation
    python -m scripts.figures.plot_pipeline_knob_ablation --yscale symlog
    python -m scripts.figures.plot_pipeline_knob_ablation --pairs tabicl_vs_tabpfn
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts._project_root import PROJECT_ROOT
from scripts.round_paths import FORWARD_ABLATION_DIR, FORWARD_ABLATION_RANDOM_DIR

TRAINED_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "pipeline_ablation"
RANDOM_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "pipeline_ablation_random"
OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "pipeline_knob_ablation.pdf"

M_GRID = (1, 2, 5, 10, 20, 50)          # the fixed top-m points
M_LABELS = [str(m) for m in M_GRID] + ["|U|"]

# Okabe-Ito (colorblind-safe), one hue per none arm; trained solid, random dashed.
ARMS = [
    ("importance, row, none",     "imp_row", "#D55E00", "o"),   # vermillion
    ("importance, dataset, none", "imp_ds",  "#0072B2", "s"),   # blue
    ("uniform, dataset, none",    "uni_ds",  "#009E73", "^"),   # bluish green
]
PAPER_GREEDY_COLOR = "#000000"          # black  -- the paper's reported result
UNI_GREEDY_COLOR = "#CC79A7"            # reddish purple -- uninformed search
RANDOM_GREEDY_COLOR = "#999999"         # grey   -- random-SAE greedy control


def _load(pdir):
    """npz for one pair dir that actually carry the none-arm curves."""
    out = []
    for f in sorted(pdir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "imp_row__1" in z:
            out.append(z)
    return out


def _collect(root, pairs):
    """Per-DATASET aggregation: for every arm/label, each dataset contributes its
    own median gc and its own frac(rows<0). Each dataset is one data point (equal
    weight), so the curve/IQR aren't dominated by large datasets or per-row tails.

    Returns dict: key -> {label -> array of per-dataset medians}, a matching
    per-dataset frac(<0), per-dataset |U|, and per-dataset greedy medians."""
    pair_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if pairs:
        pair_dirs = [d for d in pair_dirs if d.name in pairs]
    med = {key: {lab: [] for lab in M_LABELS} for _, key, _, _ in ARMS}
    frac = {key: {lab: [] for lab in M_LABELS} for _, key, _, _ in ARMS}
    U, uni_gc_med, uni_k_med = [], [], []
    for pdir in pair_dirs:
        for z in _load(pdir):
            for _, key, _, _ in ARMS:
                for lab in M_LABELS:
                    a = z[f"{key}__{lab}"]
                    med[key][lab].append(float(np.median(a)))
                    frac[key][lab].append(float((a < 0).mean()))
            U.append(len(z["uni_pool_ds"]))
            if "uni_row_gc" in z:
                uni_gc_med.append(float(np.median(z["uni_row_gc"])))
                uni_k_med.append(float(np.median(z["uni_row_k"])))
    if not U:
        return None
    arr = lambda d: {k: {lab: np.asarray(v) for lab, v in dd.items()} for k, dd in d.items()}
    return dict(
        med=arr(med), frac=arr(frac), n_ds=len(U), U=np.asarray(U),
        uni_gc_med=np.asarray(uni_gc_med) if uni_gc_med else None,
        uni_k_med=np.asarray(uni_k_med) if uni_k_med else None,
    )


def _greedy_disk(root, pairs):
    """Per-dataset medians of the on-disk greedy paper arm (clamped gc, optimal_k)."""
    pair_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if pairs:
        pair_dirs = [d for d in pair_dirs if d.name in pairs]
    gc, k = [], []
    for pdir in pair_dirs:
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "optimal_k" not in z:
                continue
            kk = z["optimal_k"].astype(int)
            act = kk > 0
            if act.sum() == 0:
                continue
            gc.append(float(np.median(z["gap_closed"][act])))
            k.append(float(np.median(kk[act])))
    if not gc:
        return None, None
    return np.asarray(gc), np.asarray(k)


def _stats(data, key):
    """Per-m x positions + per-DATASET MEDIAN / p25 / p75 / frac(gc<0).

    Each dataset contributes its own median gc; the curve is the median across
    the ~696 dataset medians and the band their 25/75 (equal dataset weight, not
    row-weighted). The strip is the dataset-mean frac of rows moved the WRONG way."""
    med_d = data["med"][key]        # {lab: per-dataset medians}
    frac_d = data["frac"][key]      # {lab: per-dataset frac(<0)}
    meanU = float(data["U"].mean())
    xs = np.array(list(M_GRID) + [meanU])
    med = np.array([np.median(med_d[lab]) for lab in M_LABELS])
    p25 = np.array([np.percentile(med_d[lab], 25) for lab in M_LABELS])
    p75 = np.array([np.percentile(med_d[lab], 75) for lab in M_LABELS])
    fneg = np.array([float(frac_d[lab].mean()) for lab in M_LABELS])
    return xs, med, p25, p75, fneg, meanU


def main():
    p = argparse.ArgumentParser(description="Pipeline-knob ablation curves (gc vs m)")
    # symlog(linthresh=1) is the sensible default: the none arms overshoot the
    # weak loss by 10-1000x at small m (unclipped), which crushes the greedy
    # reference (~1.0) and the <1 onset on a linear axis. linthresh=1 keeps the
    # [0,1] region -- where the honest "how many concepts" signal lives -- linear.
    p.add_argument("--yscale", choices=["linear", "symlog", "log"], default="symlog")
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    trained = _collect(TRAINED_DIR, args.pairs)
    random = _collect(RANDOM_DIR, args.pairs)
    if trained is None:
        print(f"No trained npz under {TRAINED_DIR} (sweep not done yet).")
        return
    pg_gc, pg_k = _greedy_disk(FORWARD_ABLATION_DIR, args.pairs)
    rg_gc, rg_k = _greedy_disk(FORWARD_ABLATION_RANDOM_DIR, args.pairs)

    # two stacked axes: median+IQR gc on top, frac(gc<0) strip below (shared log-x)
    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(4.2, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[3.4, 1.0], hspace=0.08),
        constrained_layout=True)

    # ── none curves: MEDIAN, trained (solid, IQR band) + random (dashed) ──
    for name, key, color, marker in ARMS:
        xs, med, p25, p75, fneg, meanU = _stats(trained, key)
        ax.fill_between(xs, p25, p75, color=color, alpha=0.13, linewidth=0)
        ax.plot(xs, med, color=color, marker=marker, markersize=4, linewidth=1.6,
                label=name)
        axs.plot(xs, fneg, color=color, marker=marker, markersize=3, linewidth=1.3)
        if random is not None:
            xr, medr, _, _, fnegr, _ = _stats(random, key)
            ax.plot(xr, medr, color=color, marker=marker, markersize=3, linewidth=1.1,
                    linestyle="--", alpha=0.85)
            axs.plot(xr, fnegr, color=color, marker=marker, markersize=2.5,
                     linewidth=1.0, linestyle="--", alpha=0.85)

    # ── greedy reference lines (median gc) + median-k ticks on the x-axis ──
    kmarks = []
    if pg_gc is not None:
        ax.axhline(np.median(pg_gc), color=PAPER_GREEDY_COLOR, linewidth=1.3,
                   label="paper greedy (imp,row)")
        kmarks.append((np.median(pg_k), PAPER_GREEDY_COLOR))
    if trained["uni_gc_med"] is not None:
        ax.axhline(np.median(trained["uni_gc_med"]), color=UNI_GREEDY_COLOR,
                   linewidth=1.3, linestyle="-.", label="uniform, row, greedy")
        kmarks.append((np.median(trained["uni_k_med"]), UNI_GREEDY_COLOR))
    if rg_gc is not None:
        ax.axhline(np.median(rg_gc), color=RANDOM_GREEDY_COLOR, linewidth=1.3,
                   linestyle=":", label="random-SAE greedy")
    # median-k of each greedy arm, full-height so you can read where the search
    # lands against the top-m none curves (importance k=2 vs uniform k=6)
    for kval, color in kmarks:
        ax.axvline(kval, color=color, linewidth=0.9, linestyle=":", alpha=0.5)

    ax.set_xscale("log")
    if args.yscale == "symlog":
        # keep the [0,1] linear band readable (greedy refs, dataset onset, random
        # curves live there) but crop the near-empty negative tail; minor subticks
        # mark 2..9 in each log decade.
        from matplotlib.ticker import SymmetricalLogLocator
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
        ax.set_ylim(-0.3, 120)
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(
            base=10.0, linthresh=1.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        ax.tick_params(axis="y", which="minor", length=2.5)
    elif args.yscale == "log":
        ax.set_yscale("log")
    ax.set_ylabel("gap closed", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.015, 0.97, "a) ablation", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, fontweight="bold")
    # legend above the panel in a few rows (title/counts live in the caption)
    ax.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, columnspacing=1.1, handletextpad=0.4, framealpha=0.9)

    # ── frac(gc<0) strip ──
    meanU = float(trained["U"].mean())
    axs.set_ylim(0, 0.5)
    axs.set_ylabel("frac. wrong\ndirection", fontsize=8)
    axs.set_xticks(list(M_GRID) + [meanU])
    axs.set_xticklabels([str(m) for m in M_GRID] + [f"|U|\n≈{meanU:.0f}"], fontsize=8)
    axs.set_xlabel("concepts ablated  (top-m, no search)", fontsize=9)
    axs.tick_params(labelsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}")
    print(f"trained datasets={trained['n_ds']}  "
          f"random datasets={random['n_ds'] if random else 0}")
    if pg_gc is not None:
        print(f"paper greedy: median gc={np.median(pg_gc):.3f}  median k={np.median(pg_k):.0f}")


if __name__ == "__main__":
    main()
