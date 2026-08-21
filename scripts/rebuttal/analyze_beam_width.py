#!/usr/bin/env python3
"""App F.3: is the beam wide enough, and do deep roots win on merit or by tie?

Two readouts from the recorded beam_branches (per-branch root, menu rank, final
objective score; winner = beam_root):

  1. WINNER ROOT-RANK HISTOGRAM. The width stopping rule from the widening
     design: if the menu ranking carried signal, win share should collapse with
     rank; where it collapses is how wide the beam needs to be. (v28 @ beam 6:
     27/20/15/14/12/12% -- it does NOT collapse; rank 6 still wins 12%.)

  2. DEEP-WIN MARGINS. A flat histogram has a benign reading: near-tied
     branches, deep roots winning coin flips. Discriminator: for rows won by a
     root of rank > --shallow, the winner's final score over the best shallow
     branch's. Ratios are >= 1 by construction (the winner is the max); THIN
     ratios mean ties, FAT ratios mean merit. Scores compare within a row only
     (row constants cancel), and they are in the sweep's own objective units --
     re-run this per round rather than carrying conclusions across objective
     changes (v28's spend objective != v29's delta objective).

Usage:
    python -m scripts.rebuttal.analyze_beam_width --inputs "output/rebuttal/patchv29clf_*.json"
"""
import argparse
import glob
import json
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--shallow", type=int, default=3,
                    help="ranks <= this are 'shallow'; margins are computed for "
                         "winners rooted deeper")
    ap.add_argument("--tie-band", type=float, default=1.10,
                    help="winner/shallow ratios below this count as coin flips")
    args = ap.parse_args()

    paths = sorted(p for pat in args.inputs for p in glob.glob(pat))
    print(f"{len(paths)} files: {[p.rsplit('/', 1)[-1] for p in paths]}")

    wr = Counter()
    margins = []
    n_rows = n_patched = 0
    for p in paths:
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    n_rows += 1
                    if r.get("best"):
                        n_patched += 1
                    brs = r.get("beam_branches") or []
                    win = next((b for b in brs if b["root"] == r.get("beam_root")), None)
                    if win is None:
                        continue
                    wr[win["root_rank"]] += 1
                    if win["root_rank"] > args.shallow and win.get("score") is not None:
                        shallow = [b["score"] for b in brs
                                   if b["root_rank"] <= args.shallow
                                   and b.get("score") is not None]
                        if shallow:
                            margins.append(win["score"] / max(max(shallow), 1e-12))

    print(f"rows {n_rows}, patched {n_patched}, rows with a winning branch {sum(wr.values())}")
    tot = max(sum(wr.values()), 1)
    print("\nwinner root-rank histogram:")
    for k in sorted(wr):
        print(f"  rank {k}: {wr[k]:5d}  ({wr[k] / tot:5.1%})")

    if margins:
        m = np.array(margins)
        print(f"\ndeep wins (root rank > {args.shallow}): {len(m)} rows with a "
              f"scoreable shallow branch")
        print(f"  winner/best-shallow ratio: p25 {np.percentile(m, 25):.2f}  "
              f"med {np.median(m):.2f}  p75 {np.percentile(m, 75):.2f}  "
              f"p90 {np.percentile(m, 90):.2f}")
        print(f"  coin-flip zone (< {args.tie_band:.2f}x): {(m < args.tie_band).mean():.0%}")
        print(f"  >= 2x better: {(m >= 2).mean():.0%}")


if __name__ == "__main__":
    main()
