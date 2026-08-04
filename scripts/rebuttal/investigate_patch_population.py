#!/usr/bin/env python3
"""App F.3: WHERE do we patch -- donor-side rows or recipient-side rows?

The suppression pipeline (optimize_activation_suppression.py) patches DONOR rows and
measures the DONOR's SAE activation dropping. Its binding constraint is that the donor
must have both firing and non-firing rows for the concept -- a firing-density artifact
that has nothing to do with the concept's importance to the recipient, and which
disqualifies a chunk of the locked 335 outright.

The alternative is to patch the SAME physical rows but select and measure them on the
RECIPIENT side: the rows where the concept's virtual atom was actually accepted into the
recipient (which is the population the off-manifold contribution is computed over).

This script quantifies both populations for the locked 335 so the choice is made on
numbers, not intuition:
  - donor-side: does a dataset exist with enough firing AND non-firing rows
    (the dataset-quality cache's own feasibility flags, as used by
    build_contrastive_examples.py)
  - recipient-side: the (recipient, dataset) cells where the concept was accepted, and
    how many rows each carries, read from selected_features in the forward-delta files

Usage:
    python -m scripts.rebuttal.investigate_patch_population
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT


def scan_acceptance(fwd_dir):
    """(donor, fid) -> {(recipient, dataset): n_accepted_rows} from per-row selections."""
    acc = defaultdict(lambda: defaultdict(int))
    n_files = 0
    for f in sorted(glob.glob(os.path.join(fwd_dir, "*", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files:
            continue
        donor = str(z["strong_model"]); recipient = str(z["weak_model"])
        dataset = os.path.basename(f)[:-4]
        sel = z["selected_features"]
        if sel.size == 0:
            continue
        n_files += 1
        # sel is (n_query, max_k), padded with -1. A row contributes one accepted
        # instance per distinct feature id present in that row's slots.
        for r in range(sel.shape[0]):
            fids = sel[r]
            for fid in np.unique(fids[fids >= 0]):
                acc[(donor, int(fid))][(recipient, dataset)] += 1
    return acc, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burndown", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"))
    ap.add_argument("--fwd", default=str(PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"))
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "output" / "rebuttal" / "patch_population_comparison.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.burndown)))
    acc, n_files = scan_acceptance(args.fwd)
    print(f"scanned {n_files} forward-delta files\n")

    out_rows = []
    for r in rows:
        key = (r["donor"], int(r["feat_id"]))
        cells = acc.get(key, {})
        per_cell = sorted(cells.values(), reverse=True)
        donor_ok = r["note"] == ""
        out_rows.append({
            "donor": r["donor"], "feat_id": r["feat_id"],
            "off_mass_share": r["off_mass_share"],
            "acceptance": r["acceptance"],
            "universality": r["universality"],
            "donor_side_workable": int(donor_ok),
            "donor_side_note": r["note"],
            "recip_cells": len(cells),
            "recip_rows_total": sum(cells.values()),
            "recip_rows_max_cell": per_cell[0] if per_cell else 0,
            "recip_cells_ge3": sum(1 for v in per_cell if v >= 3),
            "recip_cells_ge6": sum(1 for v in per_cell if v >= 6),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    n = len(out_rows)
    dn = [r for r in out_rows if not r["donor_side_workable"]]
    dy = [r for r in out_rows if r["donor_side_workable"]]
    share = lambda rs: sum(float(r["off_mass_share"]) for r in rs)

    print(f"{n} concepts -> {args.out}\n")
    print(f"donor-side workable:     {len(dy):3d}  ({share(dy):.1%} of total off-mass)")
    print(f"donor-side UNWORKABLE:   {len(dn):3d}  ({share(dn):.1%} of total off-mass)\n")

    tot = np.array([r["recip_rows_total"] for r in out_rows])
    cells = np.array([r["recip_cells"] for r in out_rows])
    print("recipient-side population over ALL 335:")
    print(f"  accepted-row instances per concept: "
          f"min {tot.min()}, p25 {np.percentile(tot,25):.0f}, median {np.median(tot):.0f}, "
          f"p75 {np.percentile(tot,75):.0f}, max {tot.max()}")
    print(f"  (recipient,dataset) cells per concept: "
          f"min {cells.min()}, median {np.median(cells):.0f}, max {cells.max()}")
    print(f"  concepts with zero accepted rows found: {(tot==0).sum()}")

    print("\nDoes flipping to the recipient side rescue the donor-unworkable ones?")
    for name, grp in (("donor-unworkable", dn), ("donor-workable", dy)):
        if not grp:
            continue
        t = np.array([r["recip_rows_total"] for r in grp])
        c3 = np.array([r["recip_cells_ge3"] for r in grp])
        c6 = np.array([r["recip_cells_ge6"] for r in grp])
        print(f"  {name:18s} n={len(grp):3d}  median accepted rows={np.median(t):5.0f}  "
              f"has a >=3-row cell: {(c3>0).sum():3d}  has a >=6-row cell: {(c6>0).sum():3d}")


if __name__ == "__main__":
    main()
