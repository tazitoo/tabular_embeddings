#!/usr/bin/env python3
"""Landing rates for two sweeps, stratified by attribution purity.

A landing (toward_ablation in [0.8, 1.2)) says the recipient moved to where
ablating the concept would have put it -- but the raw rate credits movement the
patch's collateral produced. attribution_purity is the share of the measured
effect attributable to c, so an unstratified landing rate mixes two populations
that mean different things. Comparing rounds on the raw rate therefore compares
their collateral as much as their search.

Reports the raw rate, the pure-landing rate, and pure landings as a count, which
is the number that survives the objective changing between rounds.

Usage:
    python -m scripts.rebuttal.compare_landings_by_purity \
        --a "output/rebuttal/patchv28clf_*.json" --label-a v28 \
        --b "output/rebuttal/v30q/*.json" --label-b v30 --pure 0.8
"""
import argparse
import glob
import json

import numpy as np

LAND_LO, LAND_HI = 0.8, 1.2


def load(patterns, exclude):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if any(x in p for x in exclude):
                continue
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        b, ro = r.get("best"), r.get("readout") or {}
                        if not b:
                            continue
                        tw = b.get("toward_ablation")
                        if tw is None or not np.isfinite(tw):
                            continue
                        rows.append((float(tw), ro.get("attribution_purity")))
    return rows


def report(label, rows, pure_at):
    tw = np.array([t for t, _ in rows], float)
    pur = np.array([np.nan if p is None else float(p) for _, p in rows], float)
    lands = (tw >= LAND_LO) & (tw < LAND_HI)
    has_p = np.isfinite(pur)
    pure = has_p & (pur >= pure_at)
    print(f"\n  {label}: {len(rows)} rows with a measurable reversal")
    print(f"    raw landings            {lands.sum():5d}  ({lands.mean():5.1%})")
    print(f"    purity recorded         {has_p.sum():5d}  "
          f"(med {np.nanmedian(pur):.2f})")
    if lands.sum():
        print(f"    purity ON landing rows  med "
              f"{np.nanmedian(pur[lands & has_p]):.2f}   "
              f"share >= {pure_at}: {(pur[lands & has_p] >= pure_at).mean():.1%}")
    print(f"    PURE landings           {(lands & pure).sum():5d}  "
          f"({(lands & pure).sum() / max(len(rows), 1):5.1%} of rows)")
    return (lands & pure).sum(), lands.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--pure", type=float, default=0.8)
    ap.add_argument("--exclude", nargs="*", default=["beam6", "_static"])
    args = ap.parse_args()

    a, b = load(args.a, args.exclude), load(args.b, args.exclude)
    pa, ra = report(args.label_a, a, args.pure)
    pb, rb = report(args.label_b, b, args.pure)
    print(f"\n  raw landings  {args.label_a} {ra} -> {args.label_b} {rb}"
          f"   ({rb - ra:+d})")
    print(f"  PURE landings {args.label_a} {pa} -> {args.label_b} {pb}"
          f"   ({pb - pa:+d})")


if __name__ == "__main__":
    main()
