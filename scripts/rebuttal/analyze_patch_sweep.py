#!/usr/bin/env python3
"""Audit and summarise the App F.3 patch sweep.

This is written as an AUDIT first and a summary second: the sweep is a baseline whose
job is to expose what needs work, so every quantity that could be silently wrong is
checked and counted rather than averaged away. Known-suspect quantities at the time of
writing:

  - rows where the concept is INACTIVE under re-extraction (a_start ~ 0). Accepted in
    the corpus but re-extracting gives ~0, so drop_frac is nan and any relative metric
    divides by ~0. Seen at 8/20 rows for mitra.
  - purity > 1, which is possible (other concepts' shifts can partially cancel c's) but
    is not a "share" and must not be averaged as one.
  - placebo rows inheriting the same near-zero denominator, which produced a reported
    target movement of 198,990,863%.
  - readout failures when the chosen cell's RECIPIENT needs a different conda env than
    the donor (tabicl_v2 under tfm).
  - this sweep ran with the min_rows=8 cell filter, which raised the median searched k
    from 3 to 21, so its coverage is a LOWER BOUND, not the headline.

Usage:
    python -m scripts.rebuttal.analyze_patch_sweep --inputs output/rebuttal/patch_sweep_*.json
"""
import argparse
import glob
import json
from collections import Counter, defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT

FINITE = lambda xs: np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)


def load(paths):
    """Merge per-host outputs, de-duplicating concepts (shards overlapped)."""
    best = {}
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception as exc:
            print(f"  skip {p}: {type(exc).__name__}"); continue
        for e in data:
            key = (e.get("donor"), e.get("feat"))
            prev = best.get(key)
            # prefer the entry that actually searched rows
            if prev is None or len(e.get("rows") or []) > len(prev.get("rows") or []):
                best[key] = e
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--recon-tol", type=float, default=1.25,
                    help="a patch is in-sample if recon <= this multiple of the row's own baseline")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal" / "patch_sweep_summary.json"))
    args = ap.parse_args()

    paths = args.inputs or sorted(glob.glob(str(
        PROJECT_ROOT / "output" / "rebuttal" / "patch_sweep_*.json")))
    entries = load(paths)
    print(f"merged {len(paths)} files -> {len(entries)} unique concepts\n")

    status = Counter(e.get("status", "searched") for e in entries)
    print("CONCEPT STATUS")
    for k, v in status.most_common():
        print(f"  {k:28s} {v:4d}")

    rows, anomalies = [], Counter()
    readout_err = Counter()
    recips = Counter()
    for e in entries:
        if not e.get("rows"):
            continue
        recips[e.get("recipient")] += 1
        for r in e["rows"]:
            n_concepts = (r.get("n_other_concepts") or 0) + 1
            a0 = r.get("a_start")
            fs = r.get("final_shift") or {}
            ro = r.get("readout") or {}
            rec = {"donor": e["donor"], "feat": e["feat"], "recipient": e.get("recipient"),
                   "dataset": e.get("dataset"), "row": r.get("row"),
                   "n_concepts": n_concepts,
                   "a_start": a0, "drop_frac": r.get("drop_frac"),
                   "stop": r.get("stop_reason"),
                   "target_rel": fs.get("target_rel"), "other_p90": fs.get("other_rel_p90"),
                   "sel_ratio": fs.get("selectivity_ratio"),
                   "recon0": r.get("recon_rel_start"),
                   "purity": ro.get("attribution_purity"),
                   "capture_of_ceiling": ro.get("capture_of_ceiling"),
                   "ceiling_effect": ro.get("ceiling_effect"),
                   "gc0": ro.get("gc_deployed"), "gc1": ro.get("gc_counterfactual"),
                   "delta_moved": ro.get("delta_rel_change")}
            # --- audit -------------------------------------------------------
            if a0 is None or not np.isfinite(a0) or abs(a0 or 0) < 1e-6:
                anomalies["row: concept INACTIVE on re-extraction (a_start~0)"] += 1
                rec["invalid"] = True
            if rec["drop_frac"] is None or not np.isfinite(rec["drop_frac"]):
                anomalies["row: drop_frac nan/None"] += 1
                rec["invalid"] = True
            for nm in ("target_rel", "sel_ratio"):
                v = rec[nm]
                if v is not None and np.isfinite(v) and v > 100:
                    anomalies[f"row: implausible {nm} (>100x)"] += 1
                    rec["invalid"] = True
            if rec["purity"] is not None and np.isfinite(rec["purity"]) and rec["purity"] > 1:
                anomalies["row: purity > 1 (others partially cancel c)"] += 1
            if "error" in ro:
                readout_err[ro["error"].split("(")[0][:60]] += 1
            elif not ro:
                anomalies["row: no readout produced"] += 1
            rows.append(rec)

    print(f"\nROWS: {len(rows)}   valid: {sum(1 for r in rows if not r.get('invalid'))}")
    if anomalies:
        print("\nANOMALIES (each is a defect to fix, not a result)")
        for k, v in anomalies.most_common():
            print(f"  {v:5d}  {k}")
    if readout_err:
        print("\nREADOUT ERRORS")
        for k, v in readout_err.most_common():
            print(f"  {v:5d}  {k}")

    print("\nSTOP REASONS")
    for k, v in Counter(r["stop"] for r in rows).most_common():
        print(f"  {k:28s} {v:5d}")

    good = [r for r in rows if not r.get("invalid")]
    if good:
        print("\nDISTRIBUTIONS over valid rows")
        for nm in ("n_concepts", "drop_frac", "sel_ratio", "other_p90", "purity",
                   "delta_moved", "capture_of_ceiling", "ceiling_effect"):
            v = FINITE([r[nm] for r in good])
            if len(v):
                print(f"  {nm:12s} n={len(v):4d}  p25={np.percentile(v,25):8.3f} "
                      f"median={np.median(v):8.3f}  p75={np.percentile(v,75):8.3f}")

        # purity vs k -- the relationship the whole design turns on
        print("\nPURITY vs n_concepts (does attribution survive as co-deployment grows?)")
        for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]:
            sub = FINITE([r["purity"] for r in good
                          if r["n_concepts"] and lo <= r["n_concepts"] <= hi
                          and r["purity"] is not None])
            band = f"n_concepts {lo}-{hi if hi < 10**9 else '+'}"
            if len(sub):
                print(f"  {band:10s} n={len(sub):4d}  median purity={np.median(sub):.3f}  "
                      f"frac>0.5: {np.mean(sub > 0.5):.0%}")

        # coverage: suppressed AND selective AND in-sample
        cov = [r for r in good
               if (r["drop_frac"] or 0) >= 0.5 and (r["sel_ratio"] or 0) >= 2.0
               and r["recon0"] is not None]
        by_concept = defaultdict(list)
        for r in good:
            by_concept[(r["donor"], r["feat"])].append(r)
        qual = {c for c, rs in by_concept.items()
                if any((r["drop_frac"] or 0) >= 0.5 and (r["sel_ratio"] or 0) >= 2.0 for r in rs)}
        print(f"\nCOVERAGE (drop>=50% AND selectivity-ratio>=2)")
        print(f"  qualifying rows:     {len(cov)} / {len(good)}")
        print(f"  qualifying concepts: {len(qual)} / {len(by_concept)}")
        print("  NOTE: lower bound -- this sweep ran with min_rows=8, median searched k~21")

        print("\nBEST EXAMPLES (by selectivity ratio)")
        for r in sorted([r for r in good if r["sel_ratio"] is not None],
                        key=lambda r: -r["sel_ratio"])[:8]:
            print(f"  {r['donor']:9s} f{r['feat']:<4d} -> {str(r['recipient']):9s} "
                  f"{str(r['dataset'])[:22]:22s} row {r['row']:4d} nc={r['n_concepts']:3d} "
                  f"drop={r['drop_frac']:.0%} sel={r['sel_ratio']:7.2f} "
                  f"purity={r['purity'] if r['purity'] is not None else float('nan'):.3f}")

    print("\nCHOSEN RECIPIENTS (env compatibility depends on this)")
    for k, v in recips.most_common():
        print(f"  {str(k):12s} {v:4d}")

    json.dump({"n_concepts": len(entries), "n_rows": len(rows),
               "anomalies": dict(anomalies), "readout_errors": dict(readout_err),
               "status": dict(status), "rows": rows}, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
