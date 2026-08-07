#!/usr/bin/env python3
"""Compare two App F.3 patch sweeps on the population, not row by row.

Written for one question: does a change to the search (here, the minimal-edit tie-break)
right-size the patch without giving up suppression? Checking that concept by concept
would not give coverage over 335 concepts, so everything below is a distribution over the
rows both sweeps searched.

Comparison is LIKE-FOR-LIKE by construction: rows are keyed on
(donor, feat, recipient, dataset, row) and only keys present in BOTH sweeps are compared,
so a difference cannot come from one arm having searched a different population. The
counts of dropped-on-either-side rows are printed rather than silently excluded.

The metrics, and why each is here:

  n_cols_changed   the thing the tie-break is supposed to move.
  drop_frac        the target. Must NOT degrade -- suppression already saturates at 1.000
                   with one column, so a smaller patch should cost nothing here.
  blast            disturbance to the other accepted concepts at the row. Selectivity is
                   what makes the recipient effect attributable to c.
  edit_distance    size of the input edit in each column's own IQR units.
  recon_excess     how far outside the row's own reconstruction error the patch lands.
  overshoot        reversal > 1.2: the patch moved the recipient PAST its original
                   prediction. A patch that overshoots is not a measurement of c.
                   SIGNED, not |reversal|: a negative reversal is the patch pushing the
                   recipient further AWAY from the original, which is a different defect
                   and is counted separately as `reversed_wrong_way`.

Overshoot is reported against `reversal` -- the objective's own recipient term --
NOT against capture_of_ceiling. The ceiling is the effect of ABLATING the concept
outright, which is an artificial intervention we constructed; scoring a real input patch
against it makes the yardstick fake. The two disagree on both level and trend over patch
size (capture: 27/51/45% for 1/2/3 columns, non-monotone; reversal: 12/21/24%, monotone),
so which one is quoted changes the conclusion, and only reversal corresponds to something
the search optimises.

Usage:
    python -m scripts.rebuttal.compare_patch_sweeps \
        --a "output/rebuttal/patchv3_*.json" --b "output/rebuttal/patchv4_*.json" \
        --label-a v3-no-tiebreak --label-b v4-tiebreak
"""
import argparse
import glob
import json
from collections import Counter

import numpy as np

from scripts._project_root import PROJECT_ROOT

OVERSHOOT = 1.2


def load(patterns):
    """(donor, feat, recipient, dataset, row) -> the row's chosen patch and readout."""
    rows = {}
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        b = r.get("best")
                        if not b:
                            continue
                        key = (c["donor"], int(c["feat"]), ds["recipient"], ds["dataset"],
                               int(r["row"]))
                        rows[key] = {
                            "n_cols": int(r.get("n_cols_changed") or len(b["columns"])),
                            "drop_frac": b.get("drop_frac"),
                            "blast": b.get("blast"),
                            "edit": b.get("edit_distance"),
                            "recon_excess": b.get("recon_excess"),
                            "reversal": b.get("reversal"),
                            "score": b.get("score"),
                        }
    return rows


def fin(xs):
    a = np.array([x for x in xs if x is not None], dtype=float)
    return a[np.isfinite(a)]


def med(xs):
    a = fin(xs)
    return float(np.median(a)) if a.size else float("nan")


def overshoot_rate(rows):
    """Signed: past the original prediction. See the module docstring on why not capture."""
    rev = fin([r["reversal"] for r in rows])
    return (float(np.mean(rev > OVERSHOOT)) if rev.size else float("nan"), rev.size)


def wrong_way_rate(rows):
    """reversal < 0: the patch pushed the recipient AWAY from its original prediction."""
    rev = fin([r["reversal"] for r in rows])
    return (float(np.mean(rev < 0.0)) if rev.size else float("nan"), rev.size)


def by_ncols(rows, label):
    print(f"\n  {label}: chosen patch size")
    n = len(rows)
    cnt = Counter(r["n_cols"] for r in rows)
    print(f"    {'cols':>4s} {'rows':>6s} {'share':>7s} {'drop':>7s} {'blast':>7s} "
          f"{'edit':>7s} {'over':>6s}")
    for k in sorted(cnt):
        sub = [r for r in rows if r["n_cols"] == k]
        o, _ = overshoot_rate(sub)
        print(f"    {k:4d} {cnt[k]:6d} {cnt[k]/n:6.1%} {med([r['drop_frac'] for r in sub]):7.3f} "
              f"{med([r['blast'] for r in sub]):7.3f} {med([r['edit'] for r in sub]):7.2f} "
              f"{o:5.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="glob(s) for sweep A")
    ap.add_argument("--b", nargs="+", required=True, help="glob(s) for sweep B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" /
                                         "patch_sweep_comparison.json"))
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    shared = sorted(set(A) & set(B))
    print(f"{args.label_a}: {len(A)} rows with a chosen patch")
    print(f"{args.label_b}: {len(B)} rows with a chosen patch")
    print(f"shared: {len(shared)}   only-{args.label_a}: {len(set(A)-set(B))}   "
          f"only-{args.label_b}: {len(set(B)-set(A))}")
    if not shared:
        print("\nNo shared rows -- the two sweeps searched different populations.")
        return
    a = [A[k] for k in shared]
    b = [B[k] for k in shared]

    oa, na = overshoot_rate(a)
    ob, nb = overshoot_rate(b)
    print(f"\n  {'metric':<16s} {args.label_a:>14s} {args.label_b:>14s} {'change':>10s}")
    for name, key in [("n_cols (med)", "n_cols"), ("drop_frac (med)", "drop_frac"),
                      ("blast (med)", "blast"), ("edit_dist (med)", "edit"),
                      ("recon_excess", "recon_excess"), ("score (med)", "score")]:
        va, vb = med([r[key] for r in a]), med([r[key] for r in b])
        print(f"  {name:<16s} {va:14.3f} {vb:14.3f} {vb - va:+10.3f}")
    print(f"  {'overshoot >1.2':<16s} {oa:13.1%} {ob:13.1%} {ob - oa:+9.1%}"
          f"   (n={na}, {nb})")
    wa, _ = wrong_way_rate(a); wb, _ = wrong_way_rate(b)
    print(f"  {'wrong-way <0':<16s} {wa:13.1%} {wb:13.1%} {wb - wa:+9.1%}")

    # Did suppression survive? Per-row, not just at the median: a median that holds can
    # still hide rows where the smaller patch stopped suppressing.
    da = np.array([r["drop_frac"] if r["drop_frac"] is not None else np.nan for r in a])
    db = np.array([r["drop_frac"] if r["drop_frac"] is not None else np.nan for r in b])
    ok = np.isfinite(da) & np.isfinite(db)
    worse = int(np.sum(db[ok] < da[ok] - 1e-9))
    print(f"\n  rows where suppression got WORSE: {worse}/{int(ok.sum())} "
          f"({worse/max(int(ok.sum()),1):.1%})")
    ca = np.array([r["n_cols"] for r in a]); cb = np.array([r["n_cols"] for r in b])
    print(f"  rows where the patch got SMALLER:  {int(np.sum(cb < ca))}/{len(shared)}  "
          f"LARGER: {int(np.sum(cb > ca))}  same: {int(np.sum(cb == ca))}")

    by_ncols(a, args.label_a)
    by_ncols(b, args.label_b)

    json.dump({"label_a": args.label_a, "label_b": args.label_b,
               "n_a": len(A), "n_b": len(B), "n_shared": len(shared),
               "overshoot_a": oa, "overshoot_b": ob,
               "wrong_way_a": wa, "wrong_way_b": wb,
               "median": {k: [med([r[k] for r in a]), med([r[k] for r in b])]
                          for k in ("n_cols", "drop_frac", "blast", "edit",
                                    "recon_excess", "score")},
               "rows_suppression_worse": worse,
               "rows_smaller": int(np.sum(cb < ca)),
               "rows_larger": int(np.sum(cb > ca))},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
