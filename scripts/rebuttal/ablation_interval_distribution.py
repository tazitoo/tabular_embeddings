#!/usr/bin/env python3
"""Where does the ablation interval sit relative to toward_ablation's resolution floor?

The interval |p_ablated - p_transfer| is toward_ablation's denominator before the floor:
how much removing this one concept moves the recipient's true-class probability. The
floor (min_gap = 0.01) caps the credit a row below it can claim. What the floor MEANS
depends on where the interval mass sits:

  - mass well below 0.01: concepts genuinely have tiny individual effects there, and the
    cap is doing honest work
  - mass bunched just under 0.01: the floor is clipping a marginal population and the
    recipient term's influence is sensitive to the exact constant

v17 could not answer this -- it recorded only the post-gate nan. v18+ records the
interval per searched row (patch_search row key `ablation_interval`).

Reported per recipient (the floor's own granularity), on the rows with a recipient
readout: quantiles of |interval|, the fraction below the floor, and the split of that
fraction into "an order below" (< 0.001) vs "marginal" (0.001-0.01).

Usage:
    python -m scripts.rebuttal.ablation_interval_distribution \
        --inputs output/rebuttal/patchv19clf_*.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

FLOOR = 0.01          # min_gap, the borrowed resolution floor
DECADE_BELOW = 0.001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="patch sweep shards; shell-expanded glob")
    args = ap.parse_args()
    paths = args.inputs or sorted(glob.glob(str(
        PROJECT_ROOT / "output" / "rebuttal" / "patchv19clf_*.json")))

    by_recipient = defaultdict(list)
    n_rows = n_without = 0
    for p in paths:
        for concept in json.load(open(p)):
            for cell in concept.get("datasets") or []:
                for row in cell.get("rows") or []:
                    n_rows += 1
                    iv = row.get("ablation_interval")
                    if iv is None or not np.isfinite(iv):
                        n_without += 1      # no recipient readout (carte), or pre-v18 file
                        continue
                    by_recipient[cell["recipient"]].append(abs(float(iv)))

    print(f"{len(paths)} shards, {n_rows} searched rows, "
          f"{n_without} without a finite interval (no readout, or pre-v18 schema)\n")
    print(f"{'recipient':<12} {'rows':>6} {'p10':>8} {'p50':>8} {'p90':>8}"
          f" {'<floor':>8} {'<1e-3':>8} {'1e-3..floor':>12}")
    allv = []
    for recipient, vals in sorted(by_recipient.items()):
        v = np.asarray(vals)
        allv.append(v)
        below = v < FLOOR
        deep = v < DECADE_BELOW
        print(f"{recipient:<12} {len(v):>6} {np.percentile(v, 10):>8.4f} "
              f"{np.median(v):>8.4f} {np.percentile(v, 90):>8.4f} "
              f"{below.mean():>8.1%} {deep.mean():>8.1%} "
              f"{(below & ~deep).mean():>12.1%}")
    if allv:
        v = np.concatenate(allv)
        below, deep = v < FLOOR, v < DECADE_BELOW
        print(f"{'ALL':<12} {len(v):>6} {np.percentile(v, 10):>8.4f} "
              f"{np.median(v):>8.4f} {np.percentile(v, 90):>8.4f} "
              f"{below.mean():>8.1%} {deep.mean():>8.1%} "
              f"{(below & ~deep).mean():>12.1%}")


if __name__ == "__main__":
    main()
