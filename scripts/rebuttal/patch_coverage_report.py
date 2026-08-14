#!/usr/bin/env python3
"""App F.3 report over one sweep: coverage, failures, and statistics over the patches.

The rebuttal commits to "reporting coverage, the concepts for which no qualifying patch
is found, and summary statistics over the patches obtained, with representative examples
shown in full". This produces those numbers from a sweep's output, for ONE task type --
classification and regression are swept separately, so pooling them here would undo the
split.

Every section reports its denominator. A patch statistic conditioned on "rows that got a
patch" is not a coverage statistic, and quoting one for the other is how a sweep that
searched 60% of its population reads as though it searched all of it.

Distinctions that matter and are kept apart:

  no cell            the concept has no deployed cell of this task type at all
  no patch           cells were searched, no candidate qualified
  patch, no readout  a donor-side patch exists but the recipient effect is not
                     measurable (carte's refit tail). This is NOT "no patch found" --
                     the input edit and its suppression never involve the recipient
  patch + readout    the full claim

Usage:
    python -m scripts.rebuttal.patch_coverage_report --inputs "output/rebuttal/patchv8clf_*.json"
"""
import argparse
import csv
import glob
import json
from collections import Counter, defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

BURNDOWN = PROJECT_ROOT / "output" / "rebuttal" / "patching_burndown.csv"


def pct(n, d):
    return f"{n:5d} ({n/d:6.1%})" if d else f"{n:5d}      -"


def fin(xs):
    a = np.array([x for x in xs if x is not None], dtype=float)
    return a[np.isfinite(a)]


def dist(name, xs, fmt="{:8.3f}"):
    a = fin(xs)
    if not a.size:
        print(f"    {name:<22s} (none)")
        return
    q = np.percentile(a, [10, 25, 50, 75, 90])
    print(f"    {name:<22s} n={a.size:5d}  " +
          "  ".join(fmt.format(v) for v in q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=[str(PROJECT_ROOT / "output" / "rebuttal" / "patchv8clf_*.json")])
    ap.add_argument("--label", default="v8-clf")
    ap.add_argument("--burndown", default=str(BURNDOWN))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" /
                                         "patch_coverage_report.json"))
    args = ap.parse_args()

    want = {(r["donor"], int(r["feat_id"])) for r in csv.DictReader(open(args.burndown))}
    entries, rows = [], []
    for pat in args.inputs:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                entries.append(c)
                for ds in c.get("datasets") or []:
                    for r in ds.get("rows") or []:
                        rows.append({**r, "_donor": c["donor"], "_feat": c["feat"],
                                     "_recipient": ds["recipient"], "_dataset": ds["dataset"],
                                     "_readout_usable": ds.get("readout_usable", True)})

    seen = {(e["donor"], int(e["feat"])) for e in entries}
    with_patch, patch_no_readout, searched_no_patch, no_cell = set(), set(), set(), set()
    for e in entries:
        key = (e["donor"], int(e["feat"]))
        ds_list = e.get("datasets") or []
        if not ds_list:
            no_cell.add(key); continue
        got = [r for ds in ds_list for r in (ds.get("rows") or []) if r.get("best")]
        if not got:
            searched_no_patch.add(key); continue
        if any(ds.get("readout_usable") for ds in ds_list):
            with_patch.add(key)
        else:
            patch_no_readout.add(key)

    n = len(want)
    print(f"=== {args.label}: App F.3 coverage over {n} concepts in the locked cell\n")
    print(f"  attempted in this sweep      {pct(len(seen), n)}")
    print(f"  not attempted                {pct(n - len(seen), n)}")
    print(f"\n  patch + recipient readout    {pct(len(with_patch), n)}")
    print(f"  patch, readout unavailable   {pct(len(patch_no_readout), n)}   (carte refit tail)")
    print(f"  searched, NO qualifying patch{pct(len(searched_no_patch), n)}")
    print(f"  no deployed cell of this task{pct(len(no_cell), n)}")
    if searched_no_patch:
        print("\n  concepts with no qualifying patch:")
        for k in sorted(searched_no_patch)[:15]:
            print(f"      {k[0]} f{k[1]}")
        if len(searched_no_patch) > 15:
            print(f"      ... and {len(searched_no_patch)-15} more")

    got = [r for r in rows if r.get("best")]
    print(f"\n=== rows: {len(rows)} searched, {len(got)} with a chosen patch "
          f"({len(got)/max(len(rows),1):.1%})")
    print(f"\n  stop reason over searched rows:")
    for k, v in Counter(r.get("stop_reason") for r in rows).most_common():
        print(f"    {str(k):<28s} {pct(v, len(rows))}")

    print(f"\n=== statistics over the {len(got)} patches obtained"
          f"        p10      p25      p50      p75      p90")
    dist("suppression_frac", [r.get("suppression_frac", r.get("drop_frac")) for r in got])
    dist("columns changed", [r.get("n_cols_changed") for r in got], "{:8.0f}")
    dist("edit distance", [(r["best"] or {}).get("edit_distance") for r in got])
    dist("blast (other concepts)", [(r["best"] or {}).get("blast") for r in got])
    dist("recon term", [(r["best"] or {}).get("centrality_ratio",
                                              (r["best"] or {}).get("recon_excess"))
                        for r in got])
    dist("concepts at row", [r.get("n_concepts_at_row") for r in got], "{:8.0f}")

    rev = fin([(r["best"] or {}).get("toward_ablation",
                                     (r["best"] or {}).get("reversal")) for r in got])
    print(f"\n  recipient effect over {rev.size} rows with a measurable reversal:")
    for lo, hi, lbl in [(-np.inf, 0, "wrong way      (<0)"),
                        (0, 0.8, "undershoot   (0-0.8)"),
                        (0.8, 1.2, "LANDS      (0.8-1.2)"),
                        (1.2, 2, "overshoot    (1.2-2)"),
                        (2, np.inf, "far over       (>2)")]:
        m = (rev >= lo) & (rev < hi)
        print(f"    {lbl:<24s} {pct(int(m.sum()), rev.size)}")

    print("\n=== by donor")
    print(f"    {'donor':<11s} {'rows':>6s} {'drop p50':>9s} {'blast p50':>10s} "
          f"{'cols p50':>9s} {'lands':>7s}")
    for d in sorted({r["_donor"] for r in got}):
        sub = [r for r in got if r["_donor"] == d]
        rv = fin([(r["best"] or {}).get("toward_ablation", (r["best"] or {}).get("reversal")) for r in sub])
        lands = float(np.mean((rv >= 0.8) & (rv < 1.2))) if rv.size else float("nan")
        print(f"    {d:<11s} {len(sub):6d} "
              f"{np.median(fin([r.get('suppression_frac', r.get('drop_frac')) for r in sub])):9.3f} "
              f"{np.median(fin([(r['best'] or {}).get('blast') for r in sub])):10.3f} "
              f"{np.median(fin([r.get('n_cols_changed') for r in sub])):9.0f} {lands:7.1%}")

    print("\n=== by recipient")
    print(f"    {'recipient':<11s} {'rows':>6s} {'drop p50':>9s} {'blast p50':>10s} {'lands':>7s}")
    for rec in sorted({r["_recipient"] for r in got}):
        sub = [r for r in got if r["_recipient"] == rec]
        rv = fin([(r["best"] or {}).get("toward_ablation", (r["best"] or {}).get("reversal")) for r in sub])
        lands = float(np.mean((rv >= 0.8) & (rv < 1.2))) if rv.size else float("nan")
        print(f"    {rec:<11s} {len(sub):6d} "
              f"{np.median(fin([r.get('suppression_frac', r.get('drop_frac')) for r in sub])):9.3f} "
              f"{np.median(fin([(r['best'] or {}).get('blast') for r in sub])):10.3f} {lands:7.1%}")

    json.dump({"label": args.label, "n_concepts": n, "attempted": len(seen),
               "patch_with_readout": len(with_patch),
               "patch_no_readout": len(patch_no_readout),
               "searched_no_patch": sorted(f"{a} f{b}" for a, b in searched_no_patch),
               "no_cell": sorted(f"{a} f{b}" for a, b in no_cell),
               "rows_searched": len(rows), "rows_with_patch": len(got)},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
