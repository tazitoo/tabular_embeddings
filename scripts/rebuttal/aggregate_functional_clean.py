#!/usr/bin/env python3
"""REBUTTAL / main-text table: aggregate the functional decomposition of the
transfer delta (on-manifold vs off-manifold gap-closure) over the FULL strong-wins
population, per recipient model and overall, for both arms.

Population = every deployed-delta row in functional_decomposition/<pair>.json. The
forward transfer only runs on strong-wins (lower-triangle) rows, so these rows ARE
the strong-wins population — all task types, no sub-filtering. This is the same
population the paper's headline gap-closure (gc) is reported on, which is the point:
gc_full here must reconcile with the paper's gc (~0.90 trained / ~0.52 random), and
the on/off-manifold pieces are reported RELATIVE to it so a component can never be
mistaken for exceeding the whole (the R1-vs-R2 naming pitfall ofnL would catch).

Row-level gc (on/off/full) is pooled over rows (row-weighted). on_manifold_energy is
a per-dataset fraction, so it is dataset-averaged. Relative contributions:
    rel_on  = gc_on_manifold  / gc_full     (fraction of achieved gap-closure the
    rel_off = gc_off_manifold / gc_full      on-/off-manifold component alone gives)
They need not sum to 1 (the components are injected separately; the split is not
additive in log-loss), but each is < 1 by construction of "component <= full".

No baseline-swap / recipient-flip filtering: the user wants the whole strong-wins
set. We still REPORT the count of datasets whose recipient differs between the two
arms (near-tie baseline drift, mostly regression) as a caveat, not a filter.

Usage:
    python -m scripts.rebuttal.aggregate_functional_clean
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

TRAINED = PROJECT_ROOT / "output/rebuttal/functional_decomposition"
RANDOM = PROJECT_ROOT / "output/rebuttal/functional_decomposition_random"


def load(dirpath):
    out = {}
    for f in glob.glob(f"{dirpath}/*.json"):
        pair = os.path.basename(f)[:-5]
        for rec in json.load(open(f)):
            out[(pair, rec["dataset"])] = rec
    return out


def _blank():
    return {"on": [], "off": [], "full": [], "energy": [], "ds": set()}


def _accumulate(store, rec):
    """Pool ALL rows of a dataset record into its recipient bucket."""
    b = store[rec["recipient"]]
    b["on"].extend(rec["gc_on_manifold_rows"])
    b["off"].extend(rec["gc_off_manifold_rows"])
    b["full"].extend(rec["gc_full_rows"])
    b["energy"].append(float(rec["on_manifold_energy"]))
    b["ds"].add(rec["dataset"])


def _row(b):
    if not b["on"]:
        return None
    on, off, full = (np.asarray(b[k], float) for k in ("on", "off", "full"))
    gc_full = float(full.mean())
    return dict(
        n_ds=len(b["ds"]), n_rows=len(on),
        gc_on=float(on.mean()), gc_on_sd=float(on.std()),
        gc_off=float(off.mean()), gc_off_sd=float(off.std()),
        gc_full=gc_full,
        rel_on=float(on.mean() / gc_full) if gc_full > 1e-9 else float("nan"),
        rel_off=float(off.mean() / gc_full) if gc_full > 1e-9 else float("nan"),
        energy=float(np.nanmean(b["energy"])),
    )


def _arm_table(arm, per):
    rows = {r: _row(per[r]) for r in per}
    rows = {r: v for r, v in rows.items() if v}
    if not rows:
        print(f"\n[{arm}] no data yet")
        return None
    allb = _blank()
    for r in per:
        for k in ("on", "off", "full", "energy"):
            allb[k].extend(per[r][k])
        allb["ds"].update(per[r]["ds"])
    ov = _row(allb)
    print(f"\n=== FUNCTIONAL DECOMPOSITION [{arm}] — full strong-wins population "
          f"(all task types, no sub-filtering) ===")
    print(f"  {'recipient':11}{'N_ds':>5}{'N_rows':>7}"
          f"{'gc_on':>13}{'gc_off':>13}{'gc_full':>9}{'rel_on':>8}{'rel_off':>8}{'energy':>8}")
    for r in sorted(rows):
        x = rows[r]
        print(f"  {r:11}{x['n_ds']:>5}{x['n_rows']:>7}"
              f"{x['gc_on']:>7.3f}±{x['gc_on_sd']:<5.2f}"
              f"{x['gc_off']:>7.3f}±{x['gc_off_sd']:<5.2f}"
              f"{x['gc_full']:>9.3f}{x['rel_on']:>8.2f}{x['rel_off']:>8.2f}{x['energy']:>8.2f}")
    print(f"  {'OVERALL':11}{ov['n_ds']:>5}{ov['n_rows']:>7}"
          f"{ov['gc_on']:>7.3f}±{ov['gc_on_sd']:<5.2f}"
          f"{ov['gc_off']:>7.3f}±{ov['gc_off_sd']:<5.2f}"
          f"{ov['gc_full']:>9.3f}{ov['rel_on']:>8.2f}{ov['rel_off']:>8.2f}{ov['energy']:>8.2f}")
    return {**rows, "OVERALL": ov}


def main():
    tr = load(str(TRAINED))
    rn = load(str(RANDOM)) if RANDOM.exists() else {}

    per_tr = defaultdict(_blank)
    for rec in tr.values():
        _accumulate(per_tr, rec)
    per_rn = defaultdict(_blank)
    for rec in rn.values():
        _accumulate(per_rn, rec)

    out = {"trained": _arm_table("trained", per_tr)}
    if rn:
        out["random"] = _arm_table("random", per_rn)

    # caveat, not a filter: datasets whose recipient differs between arms
    flips = sum(1 for k in tr if k in rn and tr[k]["recipient"] != rn[k]["recipient"])
    print(f"\n  trained pairs={len({p for p,_ in tr})}/15, "
          f"random pairs={len({p for p,_ in rn})}/15; "
          f"recipient-differs-between-arms datasets (near-tie drift, reported not dropped)={flips}")

    dst = PROJECT_ROOT / "output/rebuttal/functional_clean_table.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
