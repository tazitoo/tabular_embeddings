#!/usr/bin/env python3
"""Compare two patching rounds over EVERY patched row, not just the landings.

A landing (toward_ablation in [0.8, 1.2)) is one band of one metric, and the
crossing guard rejects any candidate moving further than max(|interval|,
MIN_GAP), so committed patches cannot exceed 1.0 -- the band is [0.8, 1.0] in
practice and landings are the best-case tail. Judging a round by its landing
rate reports the rows we like and ignores the majority.

The two attributions measure different things and both are reported:

  toward_ablation  prediction space, ALREADY net of the estimated bystander
                   contribution (movement = observed - est_bystander), except
                   where attribution_fallback fired and the raw observed
                   movement was used -- those rows are inflated, so their rate
                   is reported.
  attribution_purity  representation space: the share of the patch's embedding
                   displacement coming from c rather than from every other
                   concept it moved. Selectivity, not prediction accuracy.

Usage:
    python -m scripts.rebuttal.compare_landings_by_purity \
        --a "output/rebuttal/patchv28clf_*.json" --label-a v28 \
        --b "output/rebuttal/v30q/*.json" --label-b v30
"""
import argparse
import glob
import json

import numpy as np

LAND_LO, LAND_HI = 0.8, 1.2
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load(patterns, exclude):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if any(x in p for x in exclude):
                continue
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        b = r.get("best")
                        if not b:
                            continue
                        ro = r.get("readout") or {}
                        rows.append({
                            "toward": b.get("toward_ablation"),
                            "purity": ro.get("attribution_purity"),
                            "supp": b.get("suppression_frac"),
                            "fallback": bool(b.get("attribution_fallback")),
                            "n_cols": len(b.get("columns") or []),
                        })
    return rows


def arr(rows, key):
    v = np.array([np.nan if r[key] is None else float(r[key]) for r in rows], float)
    return v[np.isfinite(v)]


def report(label, rows, pure_at):
    print(f"\n  === {label}: {len(rows)} patched rows")
    tw, pu, sp = arr(rows, "toward"), arr(rows, "purity"), arr(rows, "supp")
    fb = np.mean([r["fallback"] for r in rows]) if rows else float("nan")
    for name, v in (("toward_ablation", tw), ("purity", pu), ("suppression", sp)):
        if len(v):
            qs = "  ".join(f"{q:5.3f}" for q in np.quantile(v, QS))
            print(f"    {name:16s} n={len(v):5d}  p10/25/50/75/90: {qs}")
    print(f"    attribution_fallback (raw movement used): {fb:.1%}")
    print(f"    toward >= 0.2: {(tw >= 0.2).mean():5.1%}   "
          f">= 0.5: {(tw >= 0.5).mean():5.1%}   "
          f">= 0.8 (LANDS): {((tw >= LAND_LO) & (tw < LAND_HI)).mean():5.1%}")
    if len(pu):
        print(f"    purity >= {pure_at}: {(pu >= pure_at).mean():5.1%}   "
              f"(selectivity of the patch, all rows)")
    return tw, pu


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
    twa, pua = report(args.label_a, a, args.pure)
    twb, pub = report(args.label_b, b, args.pure)

    # the joint view: a row is only evidence about c if the prediction moved AND
    # the displacement that moved it was c's
    print(f"\n  === joint (both conditions), share of all patched rows")
    for lo in (0.2, 0.5, 0.8):
        for name, tw, pu, n in ((args.label_a, twa, pua, len(a)),
                                (args.label_b, twb, pub, len(b))):
            if len(tw) == len(pu):
                j = ((tw >= lo) & (pu >= args.pure)).sum()
            else:                       # purity missing on some rows: report on the
                j = float("nan")        # intersection only, never a padded estimate
            print(f"    toward >= {lo} AND purity >= {args.pure}: "
                  f"{name} {j if j == j else 'n/a':>6}"
                  f"{'' if j != j else f'  ({j / max(n, 1):5.1%})'}")


if __name__ == "__main__":
    main()
