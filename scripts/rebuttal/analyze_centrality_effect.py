#!/usr/bin/env python3
"""What did replacing recon_excess with the centrality ratio change, v18 -> v19?

Like-for-like on (donor, feat, recipient, dataset, row) keys present in both sweeps,
the same discipline as compare_patch_sweeps. v18 still writes the old key names
(reversal, drop_frac, recon_rel); v19 writes the new ones -- both read here.

Four questions, one table each:

  1. TRADE. v19's suppression is lower on some rows -- what did those rows buy?
     Split the shared rows by whether suppression fell, and compare what each group
     gained in centrality_ratio and paid in blast.
  2. POSITION. Where do chosen patches END in the dataset's reconstruction-loss
     distribution? v19 records centrality directly; the question is whether the tails
     emptied relative to v18 (whose endpoint centrality is recoverable only as a
     recon-loss comparison against the row's start -- so this table reports v19's own
     start->end movement instead, which v18 could not even measure).
  3. TAILS. Rows STARTING atypical (centrality_start < 0.2) -- does the search pull
     them toward the density, and what does that cost in suppression?
  4. FLOOR CREDIT. On rows whose ablation interval is below min_gap (the marginal
     population the old gate deleted), what toward_ablation credit do chosen patches
     now earn? Should be small and finite -- not 1.0 (the gate's unearned credit) and
     not absent (the gate's deletion).

Usage:
    python -m scripts.rebuttal.analyze_centrality_effect
"""
import glob
import json
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"


def load(pattern):
    rows = {}
    for p in sorted(glob.glob(str(OUT_DIR / pattern))):
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    if not b:
                        continue
                    key = (c["donor"], int(c["feat"]), ds["recipient"], ds["dataset"],
                           int(r["row"]))
                    rows[key] = {
                        "supp": b.get("suppression_frac", b.get("drop_frac")),
                        "toward": b.get("toward_ablation", b.get("reversal")),
                        "blast": b.get("blast"),
                        "cen_ratio": b.get("centrality_ratio"),
                        "cen_end": b.get("centrality"),
                        "cen_start": r.get("centrality_start"),
                        "interval": r.get("ablation_interval"),
                        "n_cols": len(b.get("columns") or []),
                        # bystanders' estimated prediction spend for the CHOSEN patch,
                        # in prediction units: sum of moved_frac x loo over live concepts
                        "bystander_spend": sum(
                            c.get("disturbed") or 0.0
                            for c in (r.get("collateral") or [])
                            if not c.get("inactive")),
                    }
    return rows


def fin(xs):
    return np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)


def q(v, label):
    v = fin(v)
    if not len(v):
        return f"  {label:<34} n=0"
    return (f"  {label:<34} n={len(v):<5} p25 {np.percentile(v, 25):7.3f}  "
            f"med {np.median(v):7.3f}  p75 {np.percentile(v, 75):7.3f}")


def main():
    a = load("patchv18clf_*.json")
    b = load("patchv19clf_*.json")
    shared = sorted(set(a) & set(b))
    print(f"v18 {len(a)} rows, v19 {len(b)}, shared {len(shared)}\n")

    # 1. TRADE
    fell, held = [], []
    for k in shared:
        (fell if (b[k]["supp"] or 0) < (a[k]["supp"] or 0) - 1e-9 else held).append(k)
    print(f"1. TRADE -- suppression fell on {len(fell)} rows, held/rose on {len(held)}")
    for name, grp in (("suppression fell", fell), ("held/rose", held)):
        print(f"  [{name}]")
        print(q([b[k]["supp"] - a[k]["supp"] for k in grp
                 if b[k]["supp"] is not None and a[k]["supp"] is not None],
                "d suppression (v19 - v18)"))
        print(q([b[k]["cen_ratio"] for k in grp], "v19 centrality_ratio"))
        print(q([b[k]["blast"] - a[k]["blast"] for k in grp
                 if b[k]["blast"] is not None and a[k]["blast"] is not None],
                "d blast (v19 - v18)"))

    # 2/3. POSITION and TAILS, v19's own start -> end movement
    starts = fin([b[k]["cen_start"] for k in shared])
    ends = fin([b[k]["cen_end"] for k in shared])
    print(f"\n2. POSITION -- v19 centrality, start vs chosen end over {len(ends)} rows")
    print(q(starts, "centrality at start"))
    print(q(ends, "centrality at chosen patch"))
    more_central = [k for k in shared
                    if b[k]["cen_end"] is not None and b[k]["cen_start"] is not None
                    and b[k]["cen_end"] > b[k]["cen_start"]]
    print(f"  patch ends MORE central than the row started: "
          f"{len(more_central)}/{len(shared)} = {len(more_central) / len(shared):.1%}")

    tails = [k for k in shared
             if b[k]["cen_start"] is not None and b[k]["cen_start"] < 0.2]
    print(f"\n3. TAILS -- rows starting atypical (centrality_start < 0.2): {len(tails)}")
    print(q([b[k]["cen_ratio"] for k in tails], "centrality_ratio there"))
    print(q([b[k]["supp"] for k in tails], "v19 suppression there"))
    print(q([b[k]["supp"] for k in shared], "v19 suppression overall"))

    # 4. FLOOR CREDIT -- and whether it is ATTRIBUTABLE to c at all.
    #
    # toward_ablation's numerator is the TOTAL recipient movement (every accepted concept
    # rescaled by its measured shift), while the denominator is c's own interval. On a
    # sub-floor row c's own ceiling is < min_gap by definition, so bystander shifts of
    # comparable prediction-spend can account for the movement -- blast PRICES that
    # disturbance but does not remove it from the numerator. So the credit is evidence
    # about c only where the bystanders' spend is small next to the measured movement.
    # Strata: spend < 25% of the movement (attributable), 25-100% (mixed), >= 100%
    # (bystanders alone could account for all of it).
    marginal = [k for k in shared
                if b[k]["interval"] is not None and np.isfinite(b[k]["interval"])
                and abs(b[k]["interval"]) < 0.01]
    tw = fin([b[k]["toward"] for k in marginal])
    print(f"\n4. FLOOR CREDIT -- rows with |ablation interval| < min_gap: {len(marginal)}")
    print(q(tw, "toward_ablation earned there"))
    print(f"  at the gate's old unearned value (~1.0): "
          f"{(np.abs(tw - 1.0) < 0.05).mean() if len(tw) else 0:.1%}"
          f"   finite (measured, not deleted): {len(tw)}/{len(marginal)}")
    strata = {"attributable (spend < 25% of movement)": [],
              "mixed (25-100%)": [],
              "bystanders could account for it (>= 100%)": []}
    for k in marginal:
        t = b[k]["toward"]
        if t is None or not np.isfinite(t):
            continue
        movement = abs(t) * 0.01              # the floored denominator, by construction
        spend = b[k]["bystander_spend"]
        if movement <= 0:
            continue
        r = spend / movement
        name = ("attributable (spend < 25% of movement)" if r < 0.25 else
                "mixed (25-100%)" if r < 1.0 else
                "bystanders could account for it (>= 100%)")
        strata[name].append(t)
    print("  attribution of that credit, by bystander prediction-spend vs movement:")
    for name, vals in strata.items():
        v = fin(vals)
        share = len(v) / max(len(tw), 1)
        med = f"{np.median(v):.3f}" if len(v) else "-"
        print(f"    {name:<44} {len(v):>5} ({share:5.1%})   toward med {med}")


if __name__ == "__main__":
    main()
