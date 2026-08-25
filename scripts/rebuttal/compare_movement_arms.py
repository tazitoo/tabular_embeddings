#!/usr/bin/env python3
"""Does a movement term change WHICH columns the repair phase picks?

Three arms on the same host, same concept, same rows:
  A  exponents 1,0,1,0            -- movement out of the objective (v30)
  B  exponents 1,1,1,0            -- movement live pre-saturation, FROZEN in repair
  C  exponents 1,1,1,0 + measured -- live measured movement in repair too

B vs A isolates the freeze: repair sees a constant, so any difference there
arrives from pre-saturation commits moving the base. C vs B is the question --
whether steering repair by movement that is genuinely c's changes the columns.

Compares the repair columns per row, not just the final metrics: two arms can
land on similar suppression while repairing with entirely different columns.

Usage:
    python -m scripts.rebuttal.compare_movement_arms --stem output/rebuttal/mv_tabicl_142
"""
import argparse
import json

import numpy as np


def load(path):
    rows = {}
    for c in json.load(open(path)):
        for ds in c.get("datasets") or []:
            for r in ds.get("rows") or []:
                if not r.get("best"):
                    continue
                traj = r.get("trajectory") or []
                rows[(ds["dataset"], r["row"])] = {
                    "repair_cols": [t["column"] for t in traj if t.get("repair")],
                    "repair_names": [t.get("column_name") for t in traj if t.get("repair")],
                    "all_cols": list(r["best"].get("columns") or []),
                    "supp": r["best"].get("suppression_frac"),
                    "blast": r["best"].get("blast_term"),
                    "toward": r["best"].get("toward_ablation"),
                    "n_repair": r.get("n_repair_steps") or 0,
                }
    return rows


def summarize(tag, rows):
    rep = [v for v in rows.values() if v["n_repair"]]
    print(f"  {tag}: {len(rows)} rows, {len(rep)} entered repair, "
          f"{sum(v['n_repair'] for v in rows.values())} repair commits")


def compare(la, a, lb, b):
    shared = sorted(set(a) & set(b))
    both_rep = [k for k in shared if a[k]["n_repair"] or b[k]["n_repair"]]
    same_cols = [k for k in both_rep if a[k]["repair_cols"] == b[k]["repair_cols"]]
    same_set = [k for k in both_rep
                if set(a[k]["repair_cols"]) == set(b[k]["repair_cols"])]
    print(f"\n  {la} vs {lb}: {len(shared)} shared rows, {len(both_rep)} with repair")
    if both_rep:
        print(f"    identical repair column SEQUENCE: {len(same_cols)}/{len(both_rep)} "
              f"= {len(same_cols) / len(both_rep):.0%}")
        print(f"    identical repair column SET:      {len(same_set)}/{len(both_rep)} "
              f"= {len(same_set) / len(both_rep):.0%}")
    for f in ("supp", "blast", "toward"):
        va = np.array([a[k][f] for k in shared], float)
        vb = np.array([b[k][f] for k in shared], float)
        m = np.isfinite(va) & np.isfinite(vb)
        if m.sum():
            print(f"    {f:7s} med {np.median(va[m]):8.4f} -> {np.median(vb[m]):8.4f}"
                  f"   changed on {np.mean(np.abs(va[m] - vb[m]) > 1e-9):.0%} of rows")
    diff = [k for k in both_rep if a[k]["repair_cols"] != b[k]["repair_cols"]]
    for k in diff[:5]:
        print(f"      {k[0][:26]:26s} row {k[1]:<4d} "
              f"{la}: {a[k]['repair_names']} -> {lb}: {b[k]['repair_names']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="path stem; _A/_B/_C.json are read")
    args = ap.parse_args()
    arms = {}
    for tag in ("A", "B", "C"):
        try:
            arms[tag] = load(f"{args.stem}_{tag}.json")
        except FileNotFoundError:
            print(f"  {tag}: missing")
    print(f"=== {args.stem}")
    for tag, rows in arms.items():
        summarize(tag, rows)
    if "A" in arms and "B" in arms:
        compare("A", arms["A"], "B", arms["B"])
    if "B" in arms and "C" in arms:
        compare("B", arms["B"], "C", arms["C"])
    if "A" in arms and "C" in arms:
        compare("A", arms["A"], "C", arms["C"])


if __name__ == "__main__":
    main()
