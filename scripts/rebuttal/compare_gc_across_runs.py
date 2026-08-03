#!/usr/bin/env python3
"""Diff gc AGGREGATES between two functional_decomposition output dirs.

Cross-host prediction differences are tiny (1e-6..8e-3 on baseline preds), but gc
divides by a per-row gap that is small on near-tie rows, so prediction-level
smallness does NOT imply aggregate-level smallness. This compares the quantities
actually reported -- per-recipient gc_on / gc_off / gc_full and the derived
rel_on / rel_off -- over the cells the two dirs share.

Usage:
    python -m scripts.rebuttal.compare_gc_across_runs A_dir B_dir [--label-a X --label-b Y]
"""
import argparse
import glob
import json
import os

import numpy as np

from scripts._project_root import PROJECT_ROOT


def load(d):
    out = {}
    for f in glob.glob(str(PROJECT_ROOT / "output" / "rebuttal" / d / "*.json")):
        pair = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            out[(pair, r["dataset"])] = r
    return out


def agg(recs, keys):
    """Row-pooled per-recipient aggregates over the given cells."""
    b = {}
    for k in keys:
        r = recs[k]
        d = b.setdefault(r["recipient"], {"on": [], "off": [], "full": []})
        d["on"].extend(r["gc_on_manifold_rows"])
        d["off"].extend(r["gc_off_manifold_rows"])
        d["full"].extend(r["gc_full_rows"])
    out = {}
    for rec, d in b.items():
        on, off, full = (np.asarray(d[x], float) for x in ("on", "off", "full"))
        gf = full.mean()
        out[rec] = dict(n=len(on), gc_on=on.mean(), gc_off=off.mean(), gc_full=gf,
                        rel_on=on.mean() / gf if gf > 1e-9 else np.nan,
                        rel_off=off.mean() / gf if gf > 1e-9 else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()
    la = args.label_a or args.dir_a
    lb = args.label_b or args.dir_b

    A, B = load(args.dir_a), load(args.dir_b)
    common = sorted(set(A) & set(B))
    print(f"{la}: {len(A)} cells   {lb}: {len(B)} cells   shared: {len(common)}")
    if len(A) != len(common) or len(B) != len(common):
        print(f"  NOTE: comparing the {len(common)} shared cells only")

    aa, bb = agg(A, common), agg(B, common)
    print(f"\n{'recipient':<11}{'n_rows':>7}"
          f"{'gc_on A':>10}{'gc_on B':>10}{'d':>9}"
          f"{'rel_off A':>11}{'rel_off B':>11}{'d':>9}")
    for rec in sorted(aa):
        a, b = aa[rec], bb[rec]
        print(f"  {rec:<9}{a['n']:>7}"
              f"{a['gc_on']:>10.4f}{b['gc_on']:>10.4f}{b['gc_on']-a['gc_on']:>+9.4f}"
              f"{a['rel_off']:>11.4f}{b['rel_off']:>11.4f}{b['rel_off']-a['rel_off']:>+9.4f}")

    # pooled
    for nm, S in (("POOLED", None),):
        on_a = np.concatenate([np.asarray(A[k]["gc_on_manifold_rows"], float) for k in common])
        on_b = np.concatenate([np.asarray(B[k]["gc_on_manifold_rows"], float) for k in common])
        off_a = np.concatenate([np.asarray(A[k]["gc_off_manifold_rows"], float) for k in common])
        off_b = np.concatenate([np.asarray(B[k]["gc_off_manifold_rows"], float) for k in common])
        fu_a = np.concatenate([np.asarray(A[k]["gc_full_rows"], float) for k in common])
        fu_b = np.concatenate([np.asarray(B[k]["gc_full_rows"], float) for k in common])
        print(f"  {nm:<9}{len(on_a):>7}{on_a.mean():>10.4f}{on_b.mean():>10.4f}"
              f"{on_b.mean()-on_a.mean():>+9.4f}"
              f"{off_a.mean()/fu_a.mean():>11.4f}{off_b.mean()/fu_b.mean():>11.4f}"
              f"{off_b.mean()/fu_b.mean()-off_a.mean()/fu_a.mean():>+9.4f}")

    # per-cell, the sharpest view: how far does an individual gc_full move?
    d = np.array([abs(A[k]["gc_full"] - B[k]["gc_full"]) for k in common])
    ident = int((d == 0).sum())
    print(f"\nper-cell |d gc_full| over {len(common)} cells: "
          f"mean={d.mean():.4f}  median={np.median(d):.4f}  max={d.max():.4f}")
    print(f"  bit-identical cells: {ident}/{len(common)} ({100*ident/len(common):.0f}%)")
    for rec in sorted(aa):
        dr = np.array([abs(A[k]["gc_full"] - B[k]["gc_full"])
                       for k in common if A[k]["recipient"] == rec])
        ir = int((dr == 0).sum())
        print(f"    {rec:<11} n={len(dr):>3}  mean={dr.mean():.4f}  max={dr.max():.4f}"
              f"  identical={ir}/{len(dr)}")


if __name__ == "__main__":
    main()
