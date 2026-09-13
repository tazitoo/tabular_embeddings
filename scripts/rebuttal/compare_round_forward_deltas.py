#!/usr/bin/env python3
"""Which (strong, weak, dataset) forward-delta cells of one round reproduce another's?

The patching population is defined on the forward deltas, and a concept's v31 patch
readout is a function of the deployed_delta of its (donor, recipient, dataset) cells.
If a cell's round-11 npz carries the same deployed_delta, feature_acceptance and
preds_intervened as round 10's, the round-10 patch results for that cell are the
round-11 results too, and only cells that changed need re-patching.

Per pair: datasets present in both rounds, how many are bit-identical on the readout
arrays, how many differ (with the max |delta| difference), and which datasets are in
one round only. Exit code is 0 either way; this is a report.

Usage:
    python -m scripts.rebuttal.compare_round_forward_deltas
    python -m scripts.rebuttal.compare_round_forward_deltas --a output/rebuttal/forward_deltas \
        --b output/round11/forward_deltas --keys deployed_delta feature_acceptance
"""
import argparse
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.round_paths import FORWARD_DELTAS_DIR

READOUT_KEYS = ("deployed_delta", "feature_acceptance", "preds_intervened", "gap_closed",
                "strong_wins", "row_indices")


def compare_cell(fa: Path, fb: Path, keys) -> tuple[bool, float, list[str]]:
    """(identical, max |deployed_delta| difference, keys that differ)."""
    a = np.load(fa, allow_pickle=True)
    b = np.load(fb, allow_pickle=True)
    differing = []
    for k in keys:
        if k not in a.files or k not in b.files:
            differing.append(f"{k}:missing")
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.array_equal(x, y):
            differing.append(k)
    max_dd = float("nan")
    if "deployed_delta" in a.files and "deployed_delta" in b.files:
        x, y = np.asarray(a["deployed_delta"], dtype=np.float64), np.asarray(b["deployed_delta"], dtype=np.float64)
        max_dd = float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf")
    return not differing, max_dd, differing


def compare_pair(dir_a: Path, dir_b: Path, keys):
    names_a = {p.stem for p in dir_a.glob("*.npz")}
    names_b = {p.stem for p in dir_b.glob("*.npz")}
    both = sorted(names_a & names_b)
    identical, differ = [], []
    for ds in both:
        same, max_dd, diff_keys = compare_cell(dir_a / f"{ds}.npz", dir_b / f"{ds}.npz", keys)
        (identical if same else differ).append((ds, max_dd, diff_keys))
    return identical, differ, sorted(names_a - names_b), sorted(names_b - names_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas",
                    help="reference round (default: the NeurIPS forward deltas)")
    ap.add_argument("--b", type=Path, default=FORWARD_DELTAS_DIR,
                    help="round under test (default: the current round)")
    ap.add_argument("--keys", nargs="+", default=list(READOUT_KEYS))
    ap.add_argument("--show-differing", type=int, default=5,
                    help="list up to this many differing datasets per pair")
    args = ap.parse_args()

    pairs = sorted({p.name for p in args.a.iterdir() if p.is_dir()} &
                   {p.name for p in args.b.iterdir() if p.is_dir()})
    print(f"A = {args.a}\nB = {args.b}\nkeys = {args.keys}\n")
    print(f"{'pair':24s} {'both':>4s} {'ident':>5s} {'diff':>4s} {'A-only':>6s} {'B-only':>6s}  max|d delta| over differing")
    tot_both = tot_ident = 0
    for pair in pairs:
        identical, differ, a_only, b_only = compare_pair(args.a / pair, args.b / pair, args.keys)
        n_both = len(identical) + len(differ)
        tot_both += n_both
        tot_ident += len(identical)
        max_dd = max((d[1] for d in differ), default=0.0)
        print(f"{pair:24s} {n_both:4d} {len(identical):5d} {len(differ):4d} {len(a_only):6d} {len(b_only):6d}  {max_dd:.3g}")
        for ds, dd, keys in differ[: args.show_differing]:
            print(f"    {ds}: max|d delta|={dd:.3g} differing={keys}")
        if a_only:
            print(f"    A-only: {a_only}")
        if b_only:
            print(f"    B-only: {b_only}")
    print(f"\n{tot_ident}/{tot_both} shared cells bit-identical on the readout arrays")


if __name__ == "__main__":
    main()
