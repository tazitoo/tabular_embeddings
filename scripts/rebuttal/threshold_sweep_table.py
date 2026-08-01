#!/usr/bin/env python3
"""REBUTTAL: threshold-sensitivity table for the on/off-manifold decomposition.

For each variance threshold t, the on-manifold subspace E keeps the top-k_e
eigenvectors reaching t of the recipient's activation variance. We report, per
recipient and overall, the RELATIVE gap-closure of each component:

    rel_on  = mean(gc_on_manifold_rows)  / mean(gc_full_rows)
    rel_off = mean(gc_off_manifold_rows) / mean(gc_full_rows)

pooled at the ROW level over the strong-wins-with-delta population (same pooling
as aggregate_functional_clean.py). The point is sensitivity: if rel_off is flat
across t, the split is not a 90% artifact; if it swings, we know exactly where.

gc_full is threshold-independent (the full delta is unchanged), so it is NOT
reported as a headline here -- only the relative split is, per the agreed framing
(the deployed-delta subset's gc is higher than the paper's all-strong-wins gc and
must not be placed next to it).

Dirs are matched by threshold; a missing dir is skipped so the table can be built
incrementally as sweep waves land. Trained arm by default; --arm random reads the
functional_decomposition_random* dirs.

Usage:
    python -m scripts.rebuttal.threshold_sweep_table
    python -m scripts.rebuttal.threshold_sweep_table --arm random
"""
import argparse
import glob
import json
import os

import numpy as np

from scripts._project_root import PROJECT_ROOT

REC_ORDER = ["carte", "mitra", "tabpfn", "tabdpt", "tabicl_v2", "tabicl"]
# (threshold, dir-suffix). 0.90 lives in the un-suffixed canonical dir.
THRESHOLDS = [(0.80, "_t80"), (0.90, ""), (0.95, "_t95"), (0.99, "_t99")]


def _base(arm):
    return "functional_decomposition" + ("_random" if arm == "random" else "")


def pool_recipient(dirpath):
    """Return {recipient: (rel_on, rel_off, e_on, n_rows, n_ds)}.

    gc is row-pooled; on_manifold_energy (e_on) is a per-dataset fraction, so it
    is dataset-averaged (matching aggregate_functional_clean). e_off = 1 - e_on.
    """
    on, off, full, en, ds = ({} for _ in range(5))
    for f in glob.glob(f"{dirpath}/*.json"):
        for r in json.load(open(f)):
            rec = r["recipient"]
            on.setdefault(rec, []).extend(r["gc_on_manifold_rows"])
            off.setdefault(rec, []).extend(r["gc_off_manifold_rows"])
            full.setdefault(rec, []).extend(r["gc_full_rows"])
            en.setdefault(rec, []).append(float(r["on_manifold_energy"]))
            ds.setdefault(rec, set()).add((os.path.basename(f), r["dataset"]))
    out = {}
    for rec in on:
        mf = float(np.mean(full[rec])) if full[rec] else float("nan")
        out[rec] = (
            float(np.mean(on[rec])) / mf if mf else float("nan"),
            float(np.mean(off[rec])) / mf if mf else float("nan"),
            float(np.mean(en[rec])),
            len(on[rec]), len(ds[rec]),
        )
    allon = [v for L in on.values() for v in L]
    alloff = [v for L in off.values() for v in L]
    allfull = [v for L in full.values() for v in L]
    allen = [v for L in en.values() for v in L]
    mf = float(np.mean(allfull)) if allfull else float("nan")
    out["ALL"] = (
        float(np.mean(allon)) / mf if mf else float("nan"),
        float(np.mean(alloff)) / mf if mf else float("nan"),
        float(np.mean(allen)),
        len(allon), sum(len(s) for s in ds.values()),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["trained", "random"], default="trained")
    args = ap.parse_args()
    base = PROJECT_ROOT / "output" / "rebuttal"

    # gather per-threshold pooled results (skip missing dirs)
    cols = []
    for t, suf in THRESHOLDS:
        d = base / f"{_base(args.arm)}{suf}"
        if d.is_dir() and glob.glob(f"{d}/*.json"):
            cols.append((t, pool_recipient(str(d))))
    if not cols:
        print(f"No {args.arm}-arm decomposition dirs found yet.")
        return

    ts = [t for t, _ in cols]
    print(f"\nThreshold sweep -- {args.arm} arm -- on-manifold ENERGY and relative gap-closure")
    print("Each cell: E = on-manifold energy fraction (dataset-avg) | on/off = rel_on / rel_off")
    print("  rel_on = gc_on/gc_full, rel_off = gc_off/gc_full (row-pooled, strong-wins-with-delta).")
    print("  E is where the delta's mass sits; on/off is what each component DOES -- the gap")
    print("  between them is the point (energy != function). e_off = 1 - E.\n")
    hdr = f"{'recipient':<11}" + "".join(f"   {int(t*100):>3}%: E  on/off " for t in ts)
    print(hdr)
    print("-" * len(hdr))
    for rec in REC_ORDER + ["ALL"]:
        row = f"{rec:<11}"
        for _, res in cols:
            if rec in res:
                ron, roff, eon, n, nd = res[rec]
                row += f"  {eon:.2f} {ron:.2f}/{roff:.2f} "
            else:
                row += f"  {'--':>14} "
        print(row)
    # spread of rel_off across thresholds per recipient (the sensitivity number)
    print("\nrel_off spread across thresholds (max - min), the sensitivity signal:")
    for rec in REC_ORDER + ["ALL"]:
        vals = [res[rec][1] for _, res in cols if rec in res]
        if len(vals) >= 2:
            print(f"  {rec:<11} rel_off in [{min(vals):.2f}, {max(vals):.2f}]  spread {max(vals)-min(vals):.2f}")


if __name__ == "__main__":
    main()
