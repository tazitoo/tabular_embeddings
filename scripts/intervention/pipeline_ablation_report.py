#!/usr/bin/env python3
"""Aggregate the pipeline-ablation none-arms across datasets and pairs.

Reads the per-(pair,dataset) npz written by pipeline_ablation.py and pools the
per-row UNCLIPPED gap_closed across datasets (row-weighted -- every active row
counts once), per pair and overall. Unclipped is the fair readout for the "none"
arms: they have no goal and no search, so the paper's min(1,max(0,.)) clamp only
hides the overshoot that ranking-by-damage produces (see pipeline_ablation.py).

Three arms, top-m sweep with endpoint |U| (per-dataset filtered pool size):
    importance, row,     none   per-row importance ranking, top-m
    importance, dataset, none   one shared importance-ranked set, top-m
    uniform,    dataset, none   one shared random set, top-m (seed-averaged)

Mac-only, no compute. Usage:
    python -m scripts.intervention.pipeline_ablation_report
    python -m scripts.intervention.pipeline_ablation_report --pairs tabicl_vs_tabpfn
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.round_paths import FORWARD_ABLATION_DIR

PIPELINE_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "pipeline_ablation"
M_LABELS = ["1", "2", "5", "10", "20", "50", "|U|"]
ARMS = [("importance, row,     none", "imp_row"),
        ("importance, dataset, none", "imp_ds"),
        ("uniform,    dataset, none", "uni_ds")]
GC_TOLERANCE = 0.99


def _pool(npzs):
    """Row-pool each arm/label across datasets -> {arm: {label: vector}}."""
    pooled = {key: {lab: [] for lab in M_LABELS} for _, key in ARMS}
    gc_disk, k_disk = [], []
    for z in npzs:
        for _, key in ARMS:
            for lab in M_LABELS:
                pooled[key][lab].append(z[f"{key}__{lab}"])
        gc_disk.append(z["gc_disk"])
        k_disk.append(z["k_disk"])
    pooled = {k: {lab: np.concatenate(v) for lab, v in d.items()} for k, d in pooled.items()}
    return pooled, np.concatenate(gc_disk), np.concatenate(k_disk)


def _print_block(title, pooled, gc_disk, k_disk, n_ds):
    n = len(gc_disk)
    hdr = "  ".join(f"{lab:>5}" for lab in M_LABELS)
    print(f"\n=== {title}   datasets={n_ds}  active rows={n} ===")
    # Per-row unclipped gc is heavy-tailed (a few rows blow up to 1e3+), so the
    # MEDIAN is the honest central tendency; the mean is outlier-dominated.
    print(f"UNCLIPPED gap_closed -- MEDIAN (raw moved/gap; >1 = overshoot)")
    print(f"{'arm':28} {hdr}")
    print("-" * (28 + len(hdr) + 2))
    for name, key in ARMS:
        cells = "  ".join(f"{np.median(pooled[key][lab]):>5.2f}" for lab in M_LABELS)
        print(f"{name:28} {cells}")
    print(f"clamped gap_closed (paper convention)")
    for name, key in ARMS:
        cells = "  ".join(f"{np.clip(pooled[key][lab],0,1).mean():>5.2f}" for lab in M_LABELS)
        print(f"{name:28} {cells}")
    print(f"frac rows reaching gc>={GC_TOLERANCE}")
    for name, key in ARMS:
        cells = "  ".join(f"{(pooled[key][lab]>=GC_TOLERANCE).mean():>5.2f}" for lab in M_LABELS)
        print(f"{name:28} {cells}")
    print(f"PAPER (importance,row,greedy): clamped gc={gc_disk.mean():.2f}  "
          f"mean k={k_disk.mean():.2f}  median k={np.median(k_disk):.0f}")


def main():
    p = argparse.ArgumentParser(description="Aggregate pipeline-ablation none arms")
    p.add_argument("--dir", type=Path, default=PIPELINE_DIR)
    p.add_argument("--pairs", nargs="+", default=None)
    args = p.parse_args()

    pair_dirs = sorted(d for d in args.dir.iterdir() if d.is_dir())
    if args.pairs:
        pair_dirs = [d for d in pair_dirs if d.name in args.pairs]

    all_npzs = []
    for pdir in pair_dirs:
        npzs = [np.load(f, allow_pickle=True) for f in sorted(pdir.glob("*.npz"))]
        npzs = [z for z in npzs if "imp_row__1" in z]
        if not npzs:
            continue
        pooled, gc_disk, k_disk = _pool(npzs)
        _print_block(pdir.name, pooled, gc_disk, k_disk, len(npzs))
        all_npzs.extend(npzs)

    if len(pair_dirs) > 1 and all_npzs:
        pooled, gc_disk, k_disk = _pool(all_npzs)
        _print_block("ALL PAIRS POOLED", pooled, gc_disk, k_disk, len(all_npzs))


if __name__ == "__main__":
    main()
