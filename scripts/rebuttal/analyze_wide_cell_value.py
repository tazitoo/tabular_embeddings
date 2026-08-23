#!/usr/bin/env python3
"""App F.3: do the wide-dataset grind cells earn their compute?

Three questions (user, 2026-08-21), each with its denominator on the line:

  1. YIELD: how do wide-cell rows perform vs narrow-cell rows in a completed
     sweep -- patched rate, landing rate, and the cost proxies the rows record
     (n_probes, n_menu_columns)?
  2. COVERAGE: which concepts' PICKS include wide cells, and of those, which
     have enough narrow alternative datasets that the wide cells could be
     swapped out without losing the concept? A concept whose only deployments
     are wide datasets cannot be swapped -- dropping wide cells loses it.
  3. The swap rule this implies, if the numbers support one: prefer narrow
     datasets at pick time (a FILTER, like the carte and env rules -- a sort
     preference dies in the dataset dedup), falling back to wide cells only
     for concepts that have nothing else. Coverage preserved by construction,
     wide cells kept where they are load-bearing, dropped where they are not.

Alternatives are computed from forward_deltas in ONE pass (inverting
cells_for_concept's per-concept loop; 335 concepts x 687 npz would be hours).

Usage:
    python -m scripts.rebuttal.analyze_wide_cell_value \
        --inputs "output/rebuttal/patchv28clf_*.json" --wide 500
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.rebuttal.patch_search import dataset_width

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"


def all_candidate_datasets():
    """(donor, feat) -> {dataset: n_accepted_rows}, one pass over forward_deltas."""
    out = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files:
            continue
        donor = str(z["strong_model"])
        dataset = os.path.basename(f)[:-4]
        sel = np.asarray(z["selected_features"])
        for r in range(sel.shape[0]):
            for feat in sel[r][sel[r] >= 0]:
                out[(donor, int(feat))][dataset] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--wide", type=int, default=500,
                    help="a dataset is 'wide' when the donor's preprocessed table "
                         "exceeds this many columns (matches --wide-last)")
    args = ap.parse_args()

    paths = sorted(p for pat in args.inputs for p in glob.glob(pat))
    print(f"{len(paths)} files: {[p.rsplit('/', 1)[-1] for p in paths]}")

    # ---- 1. yield: wide vs narrow rows in the completed sweep ------------------
    groups = {True: [], False: []}
    picks = defaultdict(set)          # (donor, feat) -> picked datasets
    for p in paths:
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                if "rows" not in ds:
                    continue
                w = dataset_width(c["donor"], ds["dataset"])
                picks[(c["donor"], c["feat"])].add((ds["dataset"], w > args.wide))
                for r in ds.get("rows") or []:
                    groups[w > args.wide].append(r)

    print(f"\n=== yield: rows in wide (> {args.wide} col) vs narrow cells")
    for wide, rows in ((False, groups[False]), (True, groups[True])):
        if not rows:
            continue
        patched = [r for r in rows if r.get("best")]
        # LANDS uses the coverage report's own definition: toward_ablation in
        # [0.8, 1.2) -- NOT gap_opened, which is the evidential metric on a
        # different scale
        rev = [r["best"].get("toward_ablation") for r in patched
               if r["best"].get("toward_ablation") is not None
               and np.isfinite(r["best"].get("toward_ablation"))]
        lands = [v for v in rev if 0.8 <= v < 1.2]
        gapped = rev
        probes = [r.get("n_probes") for r in rows if r.get("n_probes")]
        menus = [r.get("n_menu_columns") for r in rows if r.get("n_menu_columns")]
        print(f"  {'wide' if wide else 'narrow':6s}: rows {len(rows):5d}  "
              f"patched {len(patched) / len(rows):5.1%}  "
              f"lands {len(lands) / max(len(gapped), 1):5.1%} (of {len(gapped)} gapped)  "
              f"probes med {np.median(probes):6.0f}  menu med {np.median(menus):5.0f}")

    # ---- 2. coverage: can the wide picks be swapped? ---------------------------
    cands = all_candidate_datasets()
    wide_picked = {k: v for k, v in picks.items() if any(w for _, w in v)}
    swap_full = swap_part = stuck = 0
    for (donor, feat), pk in sorted(wide_picked.items()):
        n_picked = len(pk)
        narrow_alts = [d for d, n in cands.get((donor, feat), {}).items()
                       if dataset_width(donor, d) <= args.wide]
        if len(narrow_alts) >= n_picked:
            swap_full += 1
        elif narrow_alts:
            swap_part += 1
        else:
            stuck += 1
    print(f"\n=== coverage: {len(wide_picked)} concepts have >=1 wide cell in their picks")
    print(f"  fully swappable (enough narrow alternative datasets): {swap_full}")
    print(f"  partially swappable (some narrow alternatives):       {swap_part}")
    print(f"  wide-only (NO narrow deployment anywhere -- dropping "
          f"wide loses the concept): {stuck}")


if __name__ == "__main__":
    main()
