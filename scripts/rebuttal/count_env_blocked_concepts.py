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

    # concept -> set of recipients it has a deployed cell with
    recips = defaultdict(set)
    for f in sorted(glob.glob(str(FWD / "*" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "selected_features" not in z.files or z["selected_features"].size == 0:
            continue
        donor, recipient = str(z["strong_model"]), str(z["weak_model"])
        sel = z["selected_features"]
        present = {int(x) for x in np.unique(sel) if x >= 0}
        for fid in present:
            if (donor, fid) in want:
                recips[(donor, fid)].add(recipient)

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


if __name__ == "__main__":
    main()
