#!/usr/bin/env python3
"""Where a patch_queue run stands: coverage by donor and recent throughput.

Reads the same pool/done definition the dispatcher uses -- per-concept files in
the run directory plus any whole-arm files -- so the numbers here and the
dispatcher's "N queued" agree.

Usage:
    python -m scripts.rebuttal.patch_queue_status --run v30q \
        --pool-from "output/rebuttal/patchv28clf_*.json" \
        --done-from "output/rebuttal/patchv30clf_*.json"
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

from scripts._project_root import PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-from", nargs="+", required=True)
    ap.add_argument("--done-from", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=["beam6"],
                    help="substrings marking control arms that are not pool work")
    args = ap.parse_args()

    pool, seen = [], set()
    for pat in args.pool_from:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                k = (c["donor"], int(c["feat"]))
                if k not in seen:
                    seen.add(k)
                    pool.append(k)

    outdir = PROJECT_ROOT / "output" / "rebuttal" / args.run
    done, mtimes = set(), []
    for f in outdir.glob("*_f*.json"):
        donor, feat = f.stem.rsplit("_f", 1)
        done.add((donor, int(feat)))
        mtimes.append(f.stat().st_mtime)
    for pat in args.done_from:
        for p in sorted(glob.glob(pat)):
            if any(x in p for x in args.exclude):
                continue
            for c in json.load(open(p)):
                done.add((c["donor"], int(c["feat"])))

    P = Counter(d for d, _ in pool)
    D = Counter(d for d, f in pool if (d, f) in done)
    print(f"  {'donor':12s} {'pool':>5s} {'done':>5s} {'left':>5s}")
    for d in sorted(P):
        print(f"  {d:12s} {P[d]:5d} {D[d]:5d} {P[d] - D[d]:5d}")
    n_done = sum(1 for k in pool if k in done)
    print(f"  {'TOTAL':12s} {len(pool):5d} {n_done:5d} {len(pool) - n_done:5d}"
          f"   = {n_done / max(len(pool), 1):.1%} complete")

    now = time.time()
    ages = sorted((now - m) / 3600 for m in mtimes)
    print(f"\n  throughput ({len(mtimes)} per-concept files in {args.run})")
    for w in (1, 2, 3, 6, 12, 24):
        n = sum(1 for a in ages if a <= w)
        print(f"    last {w:2d}h: {n:3d} concepts  ({n / w:.1f}/h)")
    recent = sum(1 for a in ages if a <= 3) / 3
    if recent:
        print(f"\n  at the last-3h rate ({recent:.1f}/h), {len(pool) - n_done} left "
              f"=> ~{(len(pool) - n_done) / recent:.1f}h")


if __name__ == "__main__":
    main()
