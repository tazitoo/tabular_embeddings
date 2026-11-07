#!/usr/bin/env python3
"""Do the active rows share the same missing concepts, or does each need its own?

Pure on-disk analysis of the paper's per-row greedy ablation (forward_ablation).
The greedy recorded, for every active row (strong wins per-row), which unmatched
concepts it accepted to close that row's strong->weak gap (`selected_features`,
`optimal_k`). This script measures how much those per-row accepted sets AGREE
across rows -- the direct test of the circularity objection:

  - Significant overlap  -> a dominant shared concept set carries the gap for most
    rows, and the per-row freedom is just greedy fine-tuning the last sliver to the
    0.99 early-stop (avoid overshoot). The "per-row" selection is cosmetic.
  - Little overlap        -> the gap is genuinely per-row / heterogeneous; the
    circularity concern stands and there is no single named set.

No model forwards, no SAE encode -- only the recorded selections. Firing (the
sufficient candidate pool) down-selects to these accepted (necessary) sets, so the
accepted sets are the sharpest signal already on disk.

Metrics per dataset (over active rows = optimal_k > 0):
    n_active, set size mean+/-std, mean per-row gap_closed (sanity, ~1.0)
    pool_used     distinct concepts used across all active rows
    top_cov       coverage of the single most-used concept (frac of rows whose set contains it)
    core@.5       concepts used by >=50% of active rows, and the mean coverage within it
    jaccard       mean pairwise Jaccard of the per-row accepted sets (set agreement)
    subset_core   frac of active rows whose entire accepted set is within core@.5
    n_distinct_sets / top exact-set groups  (how many rows share the identical set)

Usage (defaults = canonical path, all pairs):
    python -m scripts.intervention.perrow_selection_overlap
    python -m scripts.intervention.perrow_selection_overlap --pairs tabicl_vs_tabpfn
"""
import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.round_paths import FORWARD_ABLATION_DIR

CORE_THRESH = 0.5


def _accepted_sets(z):
    """Per active row (optimal_k>0), the accepted concept set as a frozenset."""
    k = z["optimal_k"].astype(int)
    sel = z["selected_features"]
    sets = []
    for r in range(len(k)):
        if k[r] <= 0:
            continue
        feats = [int(f) for f in sel[r][:k[r]] if int(f) >= 0]
        if feats:
            sets.append(frozenset(feats))
    return sets


def _mean_jaccard(sets):
    if len(sets) < 2:
        return float("nan")
    js = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
    return float(np.mean(js)) if js else float("nan")


def analyze(pair_dir):
    rows = []
    for npz_path in sorted(pair_dir.glob("*.npz")):
        z = np.load(npz_path, allow_pickle=True)
        if "selected_features" not in z or "optimal_k" not in z:
            continue
        sets = _accepted_sets(z)
        n = len(sets)
        if n == 0:
            continue
        sizes = np.array([len(s) for s in sets])
        concept_count = Counter(c for s in sets for c in s)
        coverage = {c: cnt / n for c, cnt in concept_count.items()}
        top_c, top_cov = max(coverage.items(), key=lambda kv: kv[1])
        core = {c for c, v in coverage.items() if v >= CORE_THRESH}
        core_cov = float(np.mean([coverage[c] for c in core])) if core else float("nan")
        subset_core = float(np.mean([1.0 if s <= core else 0.0 for s in sets]))
        set_groups = Counter(sets)
        ok = z["optimal_k"].astype(int) > 0
        gc = z["gap_closed"][ok] if "gap_closed" in z else np.array([])
        rows.append(dict(
            dataset=npz_path.stem, n_active=n,
            size_mean=float(sizes.mean()), size_std=float(sizes.std()),
            gc_mean=float(np.mean(gc)) if len(gc) else float("nan"),
            pool_used=len(concept_count),
            top_concept=int(top_c), top_cov=top_cov,
            core_size=len(core), core_cov=core_cov,
            jaccard=_mean_jaccard(sets), subset_core=subset_core,
            n_distinct_sets=len(set_groups),
            top_groups=set_groups.most_common(3),
        ))
    return rows


def _f(x):
    return f"{x:.2f}" if isinstance(x, float) and np.isfinite(x) else "  .  "


def main():
    p = argparse.ArgumentParser(description="Per-row accepted-set overlap across active rows")
    p.add_argument("--ablation-dir", type=Path, default=FORWARD_ABLATION_DIR)
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--show-groups", action="store_true",
                   help="print the most common exact accepted sets per dataset")
    args = p.parse_args()

    pair_dirs = sorted(d for d in args.ablation_dir.iterdir() if d.is_dir())
    if args.pairs:
        pair_dirs = [d for d in pair_dirs if d.name in args.pairs]

    all_rows = []
    for pdir in pair_dirs:
        rows = analyze(pdir)
        if not rows:
            continue
        print(f"\n=== {pdir.name} ===")
        hdr = (f"{'dataset':34} {'act':>4} {'size':>9} {'gc':>5} {'pool':>5} "
               f"{'topcov':>6} {'core':>5} {'corecov':>7} {'jac':>5} {'subset':>6} {'dsets':>5}")
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['dataset']:34.34} {r['n_active']:>4} "
                  f"{r['size_mean']:.1f}±{r['size_std']:.1f}  {_f(r['gc_mean']):>5} "
                  f"{r['pool_used']:>5} {_f(r['top_cov']):>6} {r['core_size']:>5} "
                  f"{_f(r['core_cov']):>7} {_f(r['jaccard']):>5} {_f(r['subset_core']):>6} "
                  f"{r['n_distinct_sets']:>5}")
            r["pair"] = pdir.name
            if args.show_groups:
                for s, c in r["top_groups"]:
                    print(f"      {c:>3} rows  <- {sorted(s)}")
        all_rows.extend(rows)

    if all_rows:
        jac = np.array([r["jaccard"] for r in all_rows if np.isfinite(r["jaccard"])])
        tc = np.array([r["top_cov"] for r in all_rows])
        sc = np.array([r["subset_core"] for r in all_rows])
        print("\n" + "=" * 60)
        print(f"POOLED over {len(all_rows)} datasets:")
        print(f"  mean pairwise Jaccard   : {jac.mean():.2f}  [{jac.min():.2f}, {jac.max():.2f}]")
        print(f"  top-concept coverage    : {tc.mean():.2f}  [{tc.min():.2f}, {tc.max():.2f}]")
        print(f"  set-within-core fraction: {sc.mean():.2f}  [{sc.min():.2f}, {sc.max():.2f}]")

    if args.csv and all_rows:
        import csv
        flat = [{k: (json.dumps([[sorted(s), c] for s, c in v]) if k == "top_groups" else v)
                 for k, v in r.items()} for r in all_rows]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
