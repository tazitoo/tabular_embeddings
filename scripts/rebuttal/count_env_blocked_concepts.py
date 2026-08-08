#!/usr/bin/env python3
"""How many of the 335 concepts can never get a recipient readout, and why?

Three exclusions remove a cell's READOUT while leaving its donor-side patch intact. A
concept is only truly blocked when EVERY cell it has is excluded, so the question is not
"how many cells are lost" but "how many concepts have nothing left".

  cross-version   tabicl <-> tabicl_v2, either direction. tabicl v1 and v2 cannot coexist
                  in one conda env, so no interpreter can hold both the donor forward and
                  the recipient tail. This is STRUCTURAL: unlike an env-scheduling gap it
                  cannot be swept by re-running under a different interpreter.
  carte           refit tail, so a rebuild is a different model and the cached delta
                  lands in a different embedding space.
  both            a concept whose cells are some of each.

The donor-side claim survives all of them -- whether an input edit suppresses the concept
never involves the recipient -- so a blocked concept is reported as patch-without-readout,
not as "no patch found".

Usage:
    python -m scripts.rebuttal.count_env_blocked_concepts
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.rebuttal.patch_search import READOUT_EXCLUDED, required_env

FWD = PROJECT_ROOT / "output" / "rebuttal" / "forward_deltas"
BURNDOWN = PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burndown", default=str(BURNDOWN))
    args = ap.parse_args()

    want = {(r["donor"], int(r["feat_id"])) for r in csv.DictReader(open(args.burndown))}
    print(f"concepts in the locked cell: {len(want)}\n")

    # concept -> set of (recipient, dataset) it has a deployed cell with
    recips = defaultdict(set)
    cells = defaultdict(set)
    rows = defaultdict(dict)
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files or z["selected_features"].size == 0:
            continue
        donor, recipient = str(z["strong_model"]), str(z["weak_model"])
        sel = z["selected_features"]
        ds_name = os.path.basename(f)[:-4]
        # ROW counts, not presence: a concept can keep a cell under a restriction and
        # still be too thin to patch there. Rows are what the search actually consumes.
        per_fid = defaultdict(int)
        for r in range(sel.shape[0]):
            for fid in {int(x) for x in sel[r] if x >= 0}:
                per_fid[fid] += 1
        for fid, nrows in per_fid.items():
            if (donor, fid) in want:
                recips[(donor, fid)].add(recipient)
                cells[(donor, fid)].add((recipient, ds_name))
                rows[(donor, fid)][(recipient, ds_name)] = nrows

    no_cells = [c for c in want if c not in recips]
    counts = defaultdict(list)
    for c, rs in recips.items():
        donor = c[0]
        ok = [r for r in rs
              if r not in READOUT_EXCLUDED and required_env(donor, r) is not None]
        if ok:
            counts["has_readout"].append(c)
            continue
        xver = {r for r in rs if required_env(donor, r) is None}
        cart = {r for r in rs if r in READOUT_EXCLUDED}
        if xver and not cart:
            counts["blocked_cross_version"].append(c)
        elif cart and not xver:
            counts["blocked_carte"].append(c)
        else:
            counts["blocked_both"].append(c)

    n = len(want)
    print(f"  {'has a usable readout cell':<34s} {len(counts['has_readout']):4d} "
          f"({len(counts['has_readout'])/n:.1%})")
    for k, label in [("blocked_cross_version", "blocked: ONLY tabicl<->tabicl_v2"),
                     ("blocked_carte", "blocked: ONLY carte"),
                     ("blocked_both", "blocked: only carte + cross-version")]:
        v = counts[k]
        print(f"  {label:<34s} {len(v):4d} ({len(v)/n:.1%})")
        for c in sorted(v)[:8]:
            print(f"       {c[0]} f{c[1]}   recipients={sorted(recips[c])}")
        if len(v) > 8:
            print(f"       ... and {len(v)-8} more")
    print(f"  {'no deployed cell at all':<34s} {len(no_cells):4d} ({len(no_cells)/n:.1%})")

    blocked = sum(len(counts[k]) for k in
                  ("blocked_cross_version", "blocked_carte", "blocked_both"))
    print(f"\n  donor-side patchable: {n - len(no_cells)} of {n} -- the input edit and its "
          f"suppression never involve the recipient.")
    print(f"  readout-blocked:      {blocked}. Report as patch-without-readout, not as "
          f"'no qualifying patch found'.")
    classification_only(want, cells, rows)


def classification_only(want, cells, rows):
    """What a classification-only restriction would cost.

    Asked because a concept that only ever appears in regression cells cannot be
    generalised across task types from this sweep -- but the cost has to be counted
    before the restriction is worth making.
    """
    import json
    from scripts.intervention.intervene_lib import SPLITS_PATH
    splits = json.loads(SPLITS_PATH.read_text())
    task_of = {d: splits[d].get("task_type", "?") for d in splits}

    n = len(want)
    all_cells = sum(len(v) for v in cells.values())
    clf_cells, reg_cells = 0, 0
    only_reg, mixed, only_clf = [], [], []
    ds_seen, ds_reg = set(), set()
    for c, cs in cells.items():
        t = {task_of.get(d, "?") for _, d in cs}
        for _, d in cs:
            ds_seen.add(d)
            if task_of.get(d) != "classification":
                ds_reg.add(d)
        k = sum(1 for _, d in cs if task_of.get(d) == "classification")
        clf_cells += k
        reg_cells += len(cs) - k
        if k == 0:
            only_reg.append(c)
        elif k < len(cs):
            mixed.append(c)
        else:
            only_clf.append(c)

    print("\n\nIF RESTRICTED TO CLASSIFICATION DATASETS")
    print(f"  datasets in play: {len(ds_seen)}, of which non-classification: {len(ds_reg)}")
    print(f"  cells: {all_cells} total -> {clf_cells} kept, {reg_cells} dropped "
          f"({reg_cells/max(all_cells,1):.1%})")
    print(f"\n  {'concepts with ONLY classification cells':<44s} {len(only_clf):4d} "
          f"({len(only_clf)/n:.1%})  unaffected")
    print(f"  {'concepts with a mix (lose cells, keep some)':<44s} {len(mixed):4d} "
          f"({len(mixed)/n:.1%})  narrower, still patchable")
    print(f"  {'concepts with ONLY regression cells':<44s} {len(only_reg):4d} "
          f"({len(only_reg)/n:.1%})  LOST entirely")
    for c in sorted(only_reg)[:10]:
        print(f"       {c[0]} f{c[1]}   {sorted(cells[c])[:3]}")
    if len(only_reg) > 10:
        print(f"       ... and {len(only_reg)-10} more")

    # Row coverage. Cells surviving a restriction says nothing about whether enough ROWS
    # survive with them -- a concept can keep three cells of two rows each.
    tot_all, tot_clf = [], []
    for c in want:
        rr = rows.get(c, {})
        tot_all.append(sum(rr.values()))
        tot_clf.append(sum(n for (rec, d), n in rr.items()
                           if task_of.get(d) == "classification"))
    a, b = np.array(tot_all), np.array(tot_clf)
    print(f"\n  accepted ROWS per concept   {'all':>10s} {'clf-only':>10s}")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:<3d}                     {np.percentile(a,q):10.0f} "
              f"{np.percentile(b,q):10.0f}")
    print(f"    total                    {a.sum():10d} {b.sum():10d}  "
          f"({b.sum()/max(a.sum(),1):.1%} kept)")
    print(f"\n  concepts below a row floor  {'all':>10s} {'clf-only':>10s}")
    for thr in (10, 30, 60, 100):
        print(f"    < {thr:<4d} rows              {int((a<thr).sum()):10d} "
              f"{int((b<thr).sum()):10d}")
    print(f"\n  the sweep draws up to n_datasets(3) x n_rows(10) = 30 rows per concept,")
    print(f"  so a concept under ~30 rows is already sampling everything it has.")
    return only_reg


if __name__ == "__main__":
    main()
