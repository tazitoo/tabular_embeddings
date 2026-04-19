#!/usr/bin/env python3
"""Three-panel distribution comparison: ablation vs transfer, trained vs random.

Per (pair, dataset), aggregates:
    a) mean gap closed on strong-wins rows
    b) median optimal_k on strong-wins rows where optimal_k > 0
    c) acceptance rate = fraction of strong-wins rows with optimal_k > 0
       (transfer also stores an explicit `acceptance_rate` field; we use
       the derived fraction so ablation and transfer are directly comparable)

Each panel overlays 4 step histograms:
    ablation (trained) / ablation (random)
    transfer (trained) / transfer (random)

Trained curves are solid; random curves dashed. Ablation is vermillion,
transfer is blue (Okabe-Ito, colour-blind safe + linestyle as backup).

Usage:
    python -m scripts.figures.plot_intervention_distributions
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_figure_path

OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
# Four distinct Okabe-Ito colours (no hatching). Randoms are drawn first
# with full opacity so the trained versions blend on top with alpha and
# the random shapes stay legible underneath.
SOURCES = [
    # (label, sweep_dir, face_color, alpha)
    ("ablation (random)",  PROJECT_ROOT / "output" / "ablation_sweep_random",               "#E69F00", 0.55),  # orange
    ("transfer (random)",  PROJECT_ROOT / "output" / "transfer_random",                     "#56B4E9", 0.55),  # sky blue
    ("ablation (trained)", PROJECT_ROOT / "output" / "ablation_sweep_tols",                 "#D55E00", 0.60),  # vermillion
    ("transfer (trained)", PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols", "#0072B2", 0.60),  # blue
]

EXCLUDE_SUBSTRINGS = ("hyperfast", "tabula8b")

Record = namedtuple("Record", "pair dataset mean_gc median_k acceptance")


def load_sweep(sweep_dir: Path) -> list[Record]:
    out: list[Record] = []
    if not sweep_dir.exists():
        print(f"WARN: {sweep_dir} missing — skipping")
        return out
    for pair_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        pair = pair_dir.name
        if any(bad in pair for bad in EXCLUDE_SUBSTRINGS):
            continue
        for npz_path in sorted(pair_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            if "strong_wins" not in d:
                continue
            sw = d["strong_wins"]
            if int(sw.sum()) == 0:
                continue
            gc = d["gap_closed"][sw]
            gc = gc[~np.isnan(gc)]
            mean_gc = float(gc.mean()) if len(gc) else np.nan
            ok = d["optimal_k"][sw]
            mask = ok > 0
            median_k = float(np.median(ok[mask])) if mask.any() else np.nan
            acceptance = float(mask.sum() / len(ok)) if len(ok) else np.nan
            out.append(Record(pair, npz_path.stem, mean_gc, median_k, acceptance))
    return out


def _finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def main() -> None:
    loaded: dict[str, list[Record]] = {}
    for label, sweep_dir, *_ in SOURCES:
        recs = load_sweep(sweep_dir)
        loaded[label] = recs
        print(f"{label:22s} n=({len(recs)})  dir={sweep_dir.name}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # Panel (a) is truncated to [0, 1] — the random baselines have a small
    # negative tail (random interventions that widen the gap) that we drop
    # for visual clarity. The bin grid has half-a-bar padding on each side.
    gc_bins = np.linspace(0.0, 1.0, 33)
    half_bar = (gc_bins[1] - gc_bins[0]) / 2

    panel_cfg = [
        (axes[0], "mean_gc",    "(a) Gap closed",
         "Mean gap closed", gc_bins, (-half_bar, 1.0 + half_bar), (0, 1)),
        (axes[1], "median_k",   "(b) Concepts used",
         "Median $k$", np.arange(0.5, 22.5, 1), None, None),
        (axes[2], "acceptance", "(c) Acceptance",
         "Fraction of strong-wins rows modified",
         np.linspace(0, 1.0, 33), None, None),
    ]

    for ax, field, title, xlabel, bins, xlim, clip_range in panel_cfg:
        # Stacked histograms: pass all 4 datasets together so matplotlib
        # stacks them at each bin.
        all_vals = []
        labels = []
        colors = []
        for label, _sweep_dir, face, _alpha in SOURCES:
            recs = loaded[label]
            vals = _finite([getattr(r, field) for r in recs])
            if clip_range is not None:
                vals = vals[(vals >= clip_range[0]) & (vals <= clip_range[1])]
            all_vals.append(vals)
            labels.append(label)
            colors.append(face)
        ax.hist(all_vals, bins=bins, histtype="barstacked",
                color=colors, label=labels, edgecolor="white", lw=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, which="major", ls=":", alpha=0.4)

    axes[0].set_ylabel("(pair, dataset) count", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUTPUT_DIR / "intervention_distributions.pdf"
    paper_path = paper_figure_path("4_results", "intervention_distributions.pdf")
    for path in (local_path, paper_path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
