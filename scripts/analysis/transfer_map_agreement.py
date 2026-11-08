#!/usr/bin/env python3
"""Do the transfer maps close the SAME rows, or different rows that average out?

The transfer panel shows ridge ~= shuffled ~= procrustes in median gap_closed.
Equal medians are consistent with two very different stories:
  (a) the maps move the same rows by the same amount  -> correspondence truly
      doesn't matter; the gc agreement is mechanistic.
  (b) each map closes a different subset of rows  -> they're doing different
      things that merely coincide in the aggregate.

This distinguishes them with the per-row gc already saved for each map
(imp_ds__{m}). For a (pair, dataset) every map scores the SAME S-wins rows in the
SAME order, so the per-row vectors are directly comparable. We report, pooled
over the datasets common to the compared maps, at m=1 and m=5:

  spearman   rank correlation of per-row gc between two maps (1 = same ordering)
  sign-agree fraction of rows where both maps have the same sign of gc
  jaccard    overlap of the row sets each map moves TOWARD strong (gc > 0)

High spearman + high jaccard  -> story (a): same rows, same amounts.
Low spearman / low jaccard with equal medians -> story (b): different rows.

Mac-only, no compute. Usage:
    python -m scripts.analysis.transfer_map_agreement
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.round_paths import FORWARD_ABLATION_DIR

TD_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "transfer_decomposition"
MAPS = ["ridge", "shuffled", "procrustes"]
MS = ["1", "5"]
ARM = "imp_ds"


def _load_pair_dataset(tag, pair, dataset, m):
    f = TD_DIR / tag / pair / f"{dataset}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return z[f"{ARM}__{m}"] if f"{ARM}__{m}" in z else None


def _common_keys():
    """(pair, dataset) keys present for all three maps."""
    sets = []
    for tag in MAPS:
        root = TD_DIR / tag
        keys = set()
        if root.exists():
            for pdir in root.iterdir():
                if pdir.is_dir():
                    for f in pdir.glob("*.npz"):
                        keys.add((pdir.name, f.stem))
        sets.append(keys)
    return sorted(set.intersection(*sets))


def main():
    argparse.ArgumentParser(description="Per-row gc agreement across transfer maps").parse_args()
    keys = _common_keys()
    print(f"datasets common to {MAPS}: {len(keys)}")

    # pool per-row gc per map at each m across the common (pair,dataset) keys
    pooled = {tag: {m: [] for m in MS} for tag in MAPS}
    for pair, ds in keys:
        for m in MS:
            arrs = {tag: _load_pair_dataset(tag, pair, ds, m) for tag in MAPS}
            if any(a is None for a in arrs.values()):
                continue
            n = min(len(a) for a in arrs.values())
            for tag in MAPS:
                pooled[tag][m].append(arrs[tag][:n])
    pooled = {tag: {m: np.concatenate(v) if v else np.array([]) for m, v in d.items()}
              for tag, d in pooled.items()}
    n_rows = len(pooled[MAPS[0]][MS[0]])
    print(f"pooled S-wins rows: {n_rows}\n")

    for m in MS:
        print(f"=== m = {m} ===")
        print(f"{'map A':11} {'map B':11} {'spearman':>9} {'sign-agree':>11} "
              f"{'jaccard(gc>0)':>14}  medians(A,B)")
        for a, b in itertools.combinations(MAPS, 2):
            ga, gb = pooled[a][m], pooled[b][m]
            rho = spearmanr(ga, gb).correlation
            sign_agree = float(((ga > 0) == (gb > 0)).mean())
            pa, pb = ga > 0, gb > 0
            jac = float((pa & pb).sum() / max((pa | pb).sum(), 1))
            print(f"{a:11} {b:11} {rho:>9.3f} {sign_agree:>11.3f} {jac:>14.3f}"
                  f"  ({np.median(ga):+.2f}, {np.median(gb):+.2f})")
        # fraction of rows every map moves toward strong, and none / all agreement
        pos = {a: pooled[a][m] > 0 for a in MAPS}
        all_pos = np.logical_and.reduce(list(pos.values()))
        any_pos = np.logical_or.reduce(list(pos.values()))
        print(f"  rows moved toward S: " + "  ".join(f"{a}={pos[a].mean():.2f}" for a in MAPS)
              + f"   all-3={all_pos.mean():.2f}  any={any_pos.mean():.2f}")
        print()


if __name__ == "__main__":
    main()
