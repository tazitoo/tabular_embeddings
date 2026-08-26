#!/usr/bin/env python3
"""Is the centrality/blast tension intrinsic, or an artifact of one round's config?

Rounds differ in many ways (beam width, blast form, repair rules), so no two are a
clean control for each other. But every round back to v20 records blast_raw -- the
real bystander displacement, in the intervention's own units, independent of whichever
blast form that round's objective traded -- and centrality_ratio. Reading both on the
SAME rows across rounds shows whether centrality-on rounds always sit at higher
displacement, which a single A/B cannot establish.

Rows are keyed (feat, dataset, row) and only rows present in every listed round are
compared, so the population is fixed across the columns.

Usage:
    python -m scripts.rebuttal.centrality_blast_tradeoff --feats 38 129 \
        --rounds v27:output/rebuttal/patchv27clf_tabdpt.json \
                 v28:output/rebuttal/patchv28clf_tabdpt.json
"""
import argparse
import glob
import json

import numpy as np


def load(pattern, feats):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for c in json.load(open(p)):
            if feats and int(c["feat"]) not in feats:
                continue
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    if not b:
                        continue
                    out[(int(c["feat"]), ds["dataset"], r["row"])] = {
                        "cen": b.get("centrality_ratio"),
                        "raw": b.get("blast_raw"),
                        "supp": b.get("suppression_frac"),
                        "cols": len(b.get("columns") or []),
                    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="*", type=int, default=[])
    ap.add_argument("--rounds", nargs="+", required=True,
                    help="label:glob, in the order to display")
    args = ap.parse_args()

    feats = set(args.feats)
    rounds = []
    for spec in args.rounds:
        lab, _, pat = spec.partition(":")
        d = load(pat, feats)
        if d:
            rounds.append((lab, d))
        else:
            print(f"  {lab}: no rows (skipped)")

    shared = set.intersection(*(set(d) for _, d in rounds)) if rounds else set()
    print(f"\n{len(shared)} rows present in all {len(rounds)} rounds "
          f"(of {max(len(d) for _, d in rounds)} max)")
    ks = sorted(shared)
    if not ks:
        return

    def med(d, f):
        v = np.array([d[k][f] for k in ks if d[k][f] is not None], float)
        return np.nanmedian(v) if len(v) else float("nan")

    print(f"\n  {'round':10s} {'centrality':>11s} {'blast_raw':>11s} "
          f"{'suppression':>12s} {'cols':>6s}")
    for lab, d in rounds:
        print(f"  {lab:10s} {med(d, 'cen'):11.3f} {med(d, 'raw'):11.5f} "
              f"{med(d, 'supp'):12.3f} {med(d, 'cols'):6.1f}")

    # the pairing that matters: within a row, do the two move together across rounds?
    print("\n  per-row correlation of centrality and blast_raw across rounds:")
    cs, bs = [], []
    for k in ks:
        c = [d[k]["cen"] for _, d in rounds if d[k]["cen"] is not None]
        b = [d[k]["raw"] for _, d in rounds if d[k]["raw"] is not None]
        if len(c) == len(rounds) and len(b) == len(rounds) and np.std(c) > 0:
            cs.append(c)
            bs.append(b)
    if cs:
        r = [np.corrcoef(c, b)[0, 1] for c, b in zip(cs, bs)
             if np.std(b) > 0 and np.isfinite(np.std(b))]
        r = [x for x in r if np.isfinite(x)]
        if r:
            print(f"    n={len(r)} rows   med r={np.median(r):+.2f}   "
                  f"positive on {np.mean(np.array(r) > 0):.0%} of rows")
            print("    (positive = a round that made the row more typical also "
                  "displaced more bystanders)")


if __name__ == "__main__":
    main()
