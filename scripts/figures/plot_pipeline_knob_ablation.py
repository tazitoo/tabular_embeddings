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
    """Row-pool every arm/label across all datasets/pairs under root.

    Returns dict: key -> {label -> pooled per-row vector}, plus pooled greedy
    arrays (uni_row_*) and the per-row |U| (each active row tagged with its
    dataset's filtered-pool size)."""
    pair_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if pairs:
        pair_dirs = [d for d in pair_dirs if d.name in pairs]
    pooled = {key: {lab: [] for lab in M_LABELS} for _, key, _, _ in ARMS}
    uni_row_gc, uni_row_k, per_row_U = [], [], []
    n_ds = 0
    for pdir in pair_dirs:
        for z in _load(pdir):
            n_ds += 1
            for _, key, _, _ in ARMS:
                for lab in M_LABELS:
                    pooled[key][lab].append(z[f"{key}__{lab}"])
            nrows = len(z["imp_row__1"])
            per_row_U.append(np.full(nrows, len(z["uni_pool_ds"])))
            if "uni_row_gc" in z:
                uni_row_gc.append(z["uni_row_gc"])
                uni_row_k.append(z["uni_row_k"])
    if n_ds == 0:
        return None
    pooled = {k: {lab: np.concatenate(v) for lab, v in d.items()} for k, d in pooled.items()}
    return dict(
        pooled=pooled, n_ds=n_ds,
        per_row_U=np.concatenate(per_row_U),
        uni_row_gc=np.concatenate(uni_row_gc) if uni_row_gc else None,
        uni_row_k=np.concatenate(uni_row_k) if uni_row_k else None,
    )


def _greedy_disk(root, pairs):
    """Pool the on-disk greedy paper arm (clamped gap_closed, optimal_k)."""
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
            gc.append(z["gap_closed"][act])
            k.append(kk[act])
    if not gc:
        return None, None
    return np.concatenate(gc), np.concatenate(k)


def _curve_xy(data, key):
    """x positions (1..50, then mean |U|) and pooled-mean unclipped gc per m."""
    pooled = data["pooled"][key]
    meanU = float(data["per_row_U"].mean())
    xs = list(M_GRID) + [meanU]
    ys = [float(pooled[lab].mean()) for lab in M_LABELS]
    return np.array(xs), np.array(ys), meanU


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

    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)

    # ── none curves: trained (solid) + random (dashed), same hue per arm ──
    for name, key, color, marker in ARMS:
        xs, ys, meanU = _curve_xy(trained, key)
        ax.plot(xs, ys, color=color, marker=marker, markersize=4, linewidth=1.6,
                label=name)
        if random is not None:
            xr, yr, _ = _curve_xy(random, key)
            ax.plot(xr, yr, color=color, marker=marker, markersize=3, linewidth=1.2,
                    linestyle="--", alpha=0.8)

    # ── greedy reference lines (no m-axis) + median-k ticks on the x-axis ──
    kmarks = []
    if pg_gc is not None:
        ax.axhline(pg_gc.mean(), color=PAPER_GREEDY_COLOR, linewidth=1.3,
                   label="paper greedy (imp,row)")
        kmarks.append((np.median(pg_k), PAPER_GREEDY_COLOR, "paper k"))
    if trained["uni_row_gc"] is not None:
        ax.axhline(trained["uni_row_gc"].mean(), color=UNI_GREEDY_COLOR,
                   linewidth=1.3, linestyle="-.", label="uniform, row, greedy")
        kmarks.append((np.median(trained["uni_row_k"]), UNI_GREEDY_COLOR, "uni k"))
    if rg_gc is not None:
        ax.axhline(rg_gc.mean(), color=RANDOM_GREEDY_COLOR, linewidth=1.3,
                   linestyle=":", label="random-SAE greedy")

    # median-k markers: short vertical ticks at the bottom of the axis
    ymin = 0 if args.yscale == "linear" else None
    for kval, color, _ in kmarks:
        ax.axvline(kval, color=color, linewidth=0.9, linestyle=":", alpha=0.6, ymax=0.06)

    ax.set_xscale("log")
    if args.yscale != "linear":
        ax.set_yscale(args.yscale, **({"linthresh": 1.0} if args.yscale == "symlog" else {}))
        if args.yscale == "symlog":
            ax.set_ylim(bottom=-1.0)   # pooled means are >=0; show only a sliver below 0
    meanU = float(trained["per_row_U"].mean())
    ax.set_xticks(list(M_GRID) + [meanU])
    ax.set_xticklabels([str(m) for m in M_GRID] + [f"|U|\n≈{meanU:.0f}"], fontsize=8)
    ax.set_xlabel("concepts ablated  (top-m, no search)", fontsize=9)
    ax.set_ylabel("gap closed (unclipped)", fontsize=9)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=(0, (1, 3)), alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9)
    ax.set_title(f"ablation  (trained solid / random dashed)\n"
                 f"{trained['n_ds']} datasets"
                 + (f", random {random['n_ds']}" if random else ""),
                 fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}")
    print(f"trained datasets={trained['n_ds']}  "
          f"random datasets={random['n_ds'] if random else 0}")
    if pg_gc is not None:
        print(f"paper greedy: gc={pg_gc.mean():.3f}  median k={np.median(pg_k):.0f}")


if __name__ == "__main__":
    main()
