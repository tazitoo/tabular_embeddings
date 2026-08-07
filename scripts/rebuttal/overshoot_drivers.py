#!/usr/bin/env python3
"""What drives patch overshoot -- the search, or the recipient?

Overshoot is reversal > 1.2: the patch moved the recipient PAST its original prediction
rather than back to it. A patch that overshoots is not a clean measurement of the concept,
so it matters which knob can reduce it.

The working assumption was that overshoot is a patch-SIZE phenomenon -- bigger edits move
more, so right-sizing the patch would bring it down. That is testable, and it is the
reason this exists: the minimal-edit tie-break cut median patch size from 3 columns to 2
and median edit distance from 3.00 to 2.02, and moved overshoot by 1.3 points. If size
were the driver, that should have moved much more.

So every patch-side quantity is binned into quartiles and the overshoot rate reported per
quartile. A driver shows up as a monotone trend across quartiles; a non-driver is flat.
The recipient and donor are tabulated separately, because those are properties of the
model being patched INTO, which no amount of search can change.

Usage:
    python -m scripts.rebuttal.overshoot_drivers --inputs "output/rebuttal/patchv4_*.json"
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

OVERSHOOT = 1.2

# (key, where it lives, what it would mean if it were the driver)
DRIVERS = [
    ("blast", "best", "disturbance to the other accepted concepts"),
    ("drop_frac", "best", "how completely the concept was suppressed"),
    ("recon_excess", "best", "how far out of the SAE's representable region the patch is"),
    ("edit_distance", "best", "size of the input edit, in column IQR units"),
    ("selectivity_ratio", "best", "target movement relative to collateral movement"),
    ("n_cols_changed", "row", "number of columns edited"),
    ("n_other_concepts", "row", "how crowded the row is"),
    ("acceptance_rank", "row", "when the concept entered the greedy"),
    ("activation", "row", "how strongly the concept fires at this row"),
]


def collect(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        b = r.get("best")
                        if not b or b.get("reversal") is None:
                            continue
                        if not np.isfinite(b["reversal"]):
                            continue
                        rec = {"reversal": float(b["reversal"]),
                               "donor": c["donor"], "recipient": ds["recipient"],
                               "dataset": ds["dataset"]}
                        for k, where, _ in DRIVERS:
                            v = (b if where == "best" else r).get(k)
                            rec[k] = float(v) if v is not None else np.nan
                        rows.append(rec)
    return rows


def quartile_table(rows, over, key):
    v = np.array([r[key] for r in rows], dtype=float)
    m = np.isfinite(v)
    if m.sum() < 50 or np.unique(v[m]).size < 4:
        return None
    q = np.quantile(v[m], [0.0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        sel = m.copy()
        hi = (v[m] <= q[i + 1]) if i == 3 else (v[m] < q[i + 1])
        sel[m] = (v[m] >= q[i]) & hi
        out.append((float(over[sel].mean()) if sel.sum() else np.nan, int(sel.sum())))
    return q, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=[str(PROJECT_ROOT / "output" / "rebuttal" / "patchv4_*.json")])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" /
                                         "overshoot_drivers.json"))
    args = ap.parse_args()

    rows = collect(args.inputs)
    over = np.array([r["reversal"] > OVERSHOOT for r in rows])
    print(f"{len(rows)} rows with a finite reversal;  overshoot (reversal > {OVERSHOOT}) "
          f"{over.mean():.1%}\n")

    print("  Patch-side drivers -- overshoot rate by quartile of each.")
    print("  A driver trends across quartiles; a non-driver is flat.\n")
    print(f"    {'quantity':<18s} {'Q1':>7s} {'Q2':>7s} {'Q3':>7s} {'Q4':>7s}   {'spread':>7s}")
    results = {}
    for key, _, _ in DRIVERS:
        t = quartile_table(rows, over, key)
        if t is None:
            print(f"    {key:<18s}   (too few distinct values to bin)")
            continue
        q, cells = t
        rates = [c[0] for c in cells]
        fin = [r for r in rates if np.isfinite(r)]
        spread = (max(fin) - min(fin)) if fin else np.nan
        print(f"    {key:<18s} " + " ".join(f"{r:6.1%}" if np.isfinite(r) else "     -"
                                            for r in rates) + f"   {spread:6.1%}")
        results[key] = {"quantiles": [float(x) for x in q], "rates": rates,
                        "n": [c[1] for c in cells]}

    for axis in ("recipient", "donor"):
        print(f"\n  By {axis} -- a property of the model, not of the search:")
        agg = defaultdict(list)
        for r, o in zip(rows, over):
            agg[r[axis]].append(o)
        results[axis] = {}
        for k in sorted(agg, key=lambda k: -float(np.mean(agg[k]))):
            print(f"    {k:<12s} {float(np.mean(agg[k])):6.1%}   (n={len(agg[k])})")
            results[axis][k] = {"rate": float(np.mean(agg[k])), "n": len(agg[k])}

    json.dump({"n_rows": len(rows), "overshoot_rate": float(over.mean()),
               "threshold": OVERSHOOT, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
