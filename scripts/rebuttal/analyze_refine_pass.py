#!/usr/bin/env python3
"""What the --refine-pass coordinate descent bought, and whether it stayed honest.

Two readings:

  WITHIN-RUN: the last trajectory entry is the row as it stood before
    refinement, so final-minus-tail is the delta without a second run. Also
    self-checks: n_refine_steps == 0 must land bit-identical to that tail, and
    no row may break the Pareto invariants (suppression held, blast not raised).

  A/B vs a --no-refine-pass control: the honest measure, since refinement can
    change which branch wins. Same-host only -- cross-host float drift flips
    near-tied ranks.

Usage:
    python -m scripts.rebuttal.analyze_refine_pass \
        --inputs "output/rebuttal/patchv31smoke_refine.json" \
        --control "output/rebuttal/patchv31smoke_norefine.json"
"""
import argparse
import glob
import json

import numpy as np

TOL = 1e-9


def load(patterns):
    """(donor, feat, dataset, row) -> row record, for rows with a chosen patch."""
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        if r.get("best"):
                            out[(c["donor"], c["feat"], ds["dataset"], r["row"])] = r
    return out


def traded(entry, form):
    """The blast the objective actually trades, for a candidate or a commit.

    The 'blast' field keeps its historical spend-form meaning, so grading on it
    would judge the search by a quantity it was not optimizing. Trajectory
    entries predate the blast_term name and carry blast_delta, the same number
    under --blast-form delta.
    """
    if entry.get("blast_term") is not None:
        return entry["blast_term"]
    if form == "delta" and entry.get("blast_delta") is not None:
        return entry["blast_delta"]
    return entry.get("blast")


def q(vals, label):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return f"  {label:34s} (none)"
    return (f"  {label:34s} n={len(v):4d}  med {np.median(v):9.4f}  "
            f"mean {v.mean():9.4f}  min {v.min():9.4f}  max {v.max():9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[])
    args = ap.parse_args()

    rows = load(args.inputs)
    print(f"refine arm: {len(rows)} rows with a chosen patch")

    steps = [r.get("n_refine_steps") or 0 for r in rows.values()]
    fired = [k for k, r in rows.items() if (r.get("n_refine_steps") or 0) > 0]
    print(f"  rows where refinement fired: {len(fired)}/{len(rows)} "
          f"= {len(fired) / max(len(rows), 1):.1%}   "
          f"steps: total {sum(steps)}, max {max(steps) if steps else 0}")

    # ---- within-run delta vs the trajectory tail (pre-refine state) -----------
    d_supp, d_blast, d_score, ratios, d_raw = [], [], [], [], []
    viol_supp, viol_blast, viol_score, viol_noop = [], [], [], []
    for k, r in rows.items():
        traj = r.get("trajectory") or []
        b = r["best"]
        if not traj:
            continue
        pre = traj[-1]
        form = r.get("blast_form")
        ds_ = b["suppression_frac"] - pre["suppression_frac"]
        db_ = traded(b, form) - traded(pre, form)
        dsc = b["score"] - pre["score"]
        n = r.get("n_refine_steps") or 0
        if n == 0:
            # untouched rows must land exactly on their own trajectory tail;
            # any drift means the pass mutated state it should not have
            if abs(ds_) > TOL or abs(db_) > TOL:
                viol_noop.append((k, ds_, db_))
            continue
        d_supp.append(ds_)
        d_blast.append(db_)
        d_score.append(dsc)
        if b.get("blast_raw") is not None and pre.get("blast_raw") is not None:
            d_raw.append(b["blast_raw"] - pre["blast_raw"])
        if pre["score"] > 0 and np.isfinite(b["score"]):
            ratios.append(b["score"] / pre["score"])
        if ds_ < -TOL:
            viol_supp.append((k, ds_))
        if db_ > TOL:
            viol_blast.append((k, db_))
        if dsc <= 0:
            viol_score.append((k, dsc))

    print(f"\n=== within-run: refined rows vs their own pre-refine state "
          f"(n={len(d_supp)})")
    print(q(d_supp, "d suppression_frac"))
    print(q(d_blast, "d blast (traded term)"))
    print(q(d_raw, "d blast_raw (real displacement)"))
    print(q(d_score, "d score"))
    print(q(ratios, "score ratio (after/before)"))

    print("\n=== Pareto invariants (each must be empty)")
    print(f"  suppression DEGRADED:        {len(viol_supp)}  {viol_supp[:3]}")
    print(f"  blast INCREASED:             {len(viol_blast)}  {viol_blast[:3]}")
    print(f"  accepted without score gain: {len(viol_score)}  {viol_score[:3]}")
    print(f"  n_refine_steps==0 but state moved: {len(viol_noop)}  {viol_noop[:3]}")

    # partial-suppression rows are the population the pass was built for
    part = [(k, r) for k, r in rows.items() if r["best"]["suppression_frac"] < 0.999]
    part_fired = [k for k, r in part if (r.get("n_refine_steps") or 0) > 0]
    print(f"\n=== target population: {len(part)} rows still below full suppression, "
          f"{len(part_fired)} refined ({len(part_fired) / max(len(part), 1):.1%})")
    print(q([r["best"]["suppression_frac"] for _, r in part], "their final suppression"))

    if not args.control:
        return

    # ---- A/B vs the no-refine control on the shared population ----------------
    ctrl = load(args.control)
    shared = sorted(set(rows) & set(ctrl))
    print(f"\n=== A/B vs control: {len(rows)} refine, {len(ctrl)} control, "
          f"{len(shared)} shared rows")
    def pick(rec, field):
        return (traded(rec["best"], rec.get("blast_form")) if field == "blast"
                else rec["best"][field])

    for field, label in (("suppression_frac", "suppression"),
                         ("blast", "blast (traded)"), ("blast_raw", "blast_raw"),
                         ("score", "score"), ("centrality_ratio", "centrality_ratio")):
        a = np.asarray([pick(rows[k], field) for k in shared], float)
        c = np.asarray([pick(ctrl[k], field) for k in shared], float)
        m = np.isfinite(a) & np.isfinite(c)
        print(f"  {label:16s} control med {np.median(c[m]):9.4f} -> "
              f"refine med {np.median(a[m]):9.4f}   "
              f"better {np.mean(a[m] > c[m]):5.1%}  worse {np.mean(a[m] < c[m]):5.1%}")
    same = sum(1 for k in shared
               if rows[k]["best"]["columns"] == ctrl[k]["best"]["columns"]
               and rows[k]["best"]["values"] == ctrl[k]["best"]["values"])
    print(f"  identical chosen patch: {same}/{len(shared)} = "
          f"{same / max(len(shared), 1):.1%}")


if __name__ == "__main__":
    main()
