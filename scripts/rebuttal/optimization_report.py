#!/usr/bin/env python3
"""How the patch search's optimization is actually behaving, from its own records.

Four views, none of which a table of outcome medians provides:

  1. COMPOSITION  ln score = ln supp + 0.5 ln toward + ln cen_ratio - ln(blast + EPS),
                  per chosen patch. Which term carries the score's level, and which its
                  VARIANCE -- the variance carrier is what the argmax is sensitive to,
                  and an exponent sweep that ignores it tunes the wrong knob.
  2. STEP GAINS   the greedy path (trajectory records every committed column with all
                  terms). For each step: what the added column bought, term by term, in
                  log units -- and what it paid. An optimizer that keeps buying with
                  one term and paying with another is telling us the trade it sees.
  3. TRADE        Spearman correlations between terms at the chosen points -- the shape
                  of the frontier the selection lives on.
  4. GUARDS       how often each guard binds: sub-floor intervals (toward's resolution
                  floor), attribution fallback, negative-toward choices, stop reasons.

Log contributions use the CURRENT EXPONENTS; negative/zero terms are counted and set
aside per view rather than silently dropped.

Usage:
    python -m scripts.rebuttal.optimization_report --inputs output/rebuttal/patchv20clf_*.json
"""
import argparse
import glob
import json
from collections import Counter, defaultdict

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.rebuttal.patch_search import EPS, EXPONENTS, MIN_GAP

OUT_DIR = PROJECT_ROOT / "output" / "rebuttal"

TERMS = (("suppression", lambda t: t.get("suppression_frac")),
         ("toward", lambda t: t.get("toward_ablation")),
         ("centrality", lambda t: t.get("centrality_ratio")),
         ("blast", lambda t: t.get("blast")))


def contributions(t):
    """Signed log contribution of each term to ln(score), or None if not decomposable
    (negative or missing toward, missing term)."""
    s, tw = t.get("suppression_frac"), t.get("toward_ablation")
    cr, bl = t.get("centrality_ratio"), t.get("blast")
    vals = (s, tw, cr, bl)
    if any(v is None or not np.isfinite(v) for v in vals) or s <= 0 or tw <= 0 or cr <= 0:
        return None
    return {"suppression": EXPONENTS["suppression"] * np.log(s),
            "toward": EXPONENTS["toward_ablation"] * np.log(tw),
            "centrality": EXPONENTS["centrality"] * np.log(cr),
            "blast": -EXPONENTS["blast"] * np.log(max(bl, 0.0) + EPS)}


def q(v, ps=(10, 25, 50, 75, 90)):
    return "  ".join(f"{np.percentile(v, p):7.2f}" for p in ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None)
    args = ap.parse_args()
    paths = args.inputs or sorted(glob.glob(str(OUT_DIR / "patchv20clf_*.json")))

    chosen, trajs, stops, rowmeta = [], [], Counter(), []
    for p in paths:
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    b = r.get("best")
                    stops[r.get("stop_reason", "?")] += 1
                    if b:
                        chosen.append(b)
                        rowmeta.append({"interval": r.get("ablation_interval"),
                                        "fallback": b.get("attribution_fallback")})
                    tr = r.get("trajectory") or []
                    if len(tr) >= 1:
                        trajs.append(tr)
    print(f"{len(paths)} shards; {len(chosen)} chosen patches, {len(trajs)} trajectories\n")

    # 1. COMPOSITION
    cons = [c_ for c_ in (contributions(b) for b in chosen) if c_ is not None]
    excluded = len(chosen) - len(cons)
    print(f"1. COMPOSITION of ln(score) at the chosen patch  ({len(cons)} rows; "
          f"{excluded} not log-decomposable: negative toward or missing term)")
    print(f"   {'term':<12} {'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}"
          f"  {'var':>7}  {'var share':>9}")
    arr = {name: np.array([c_[name] for c_ in cons]) for name, _ in
           (("suppression", 0), ("toward", 0), ("centrality", 0), ("blast", 0))}
    total = sum(arr.values())
    var_tot = total.var()
    for name, v in arr.items():
        # variance SHARE including covariance: cov(term, total)/var(total) sums to 1
        share = np.cov(v, total)[0, 1] / var_tot
        print(f"   {name:<12} {q(v)}  {v.var():7.2f}  {share:9.1%}")
    print(f"   {'ln score':<12} {q(total)}  {var_tot:7.2f}")

    # 2. STEP GAINS
    print("\n2. WHAT EACH ADDED COLUMN BUYS (log units, consecutive trajectory steps)")
    gains = defaultdict(list)
    n_steps = []
    for tr in trajs:
        n_steps.append(len(tr))
        for a, b in zip(tr, tr[1:]):
            ca, cb = contributions(a), contributions(b)
            if ca is None or cb is None:
                continue
            for name in ca:
                gains[name].append(cb[name] - ca[name])
            gains["ln score"].append(sum(cb.values()) - sum(ca.values()))
    print(f"   steps per row: med {np.median(n_steps):.0f}, p90 {np.percentile(n_steps, 90):.0f}"
          f"; {len(gains['ln score'])} decomposable steps")
    print(f"   {'term':<12} {'med gain':>9} {'p25':>7} {'p75':>7}  {'% steps buying':>14} {'% paying':>9}")
    for name in ("suppression", "toward", "centrality", "blast", "ln score"):
        v = np.array(gains[name])
        if not len(v):
            continue
        print(f"   {name:<12} {np.median(v):9.3f} {np.percentile(v, 25):7.3f} "
              f"{np.percentile(v, 75):7.3f}  {(v > 1e-9).mean():14.1%} {(v < -1e-9).mean():9.1%}")

    # 3. TRADE SURFACE
    print("\n3. TERM CORRELATIONS AT THE CHOSEN POINTS (Spearman)")
    names = [n for n, _ in TERMS]
    vals = {}
    for n, get in TERMS:
        vals[n] = np.array([get(b) if get(b) is not None else np.nan for b in chosen])
    print("   " + " ".join(f"{n:>12}" for n in [""] + names))
    for ni in names:
        row = [f"{ni:>12}"]
        for nj in names:
            m = np.isfinite(vals[ni]) & np.isfinite(vals[nj])
            ri = np.argsort(np.argsort(vals[ni][m])); rj = np.argsort(np.argsort(vals[nj][m]))
            row.append(f"{np.corrcoef(ri, rj)[0, 1]:12.2f}")
        print("   " + " ".join(row))

    # 4. GUARDS
    print("\n4. GUARD AND CONSTRAINT ACTIVITY")
    iv = np.array([m["interval"] for m in rowmeta if m["interval"] is not None
                   and np.isfinite(m["interval"])])
    fb = [m["fallback"] for m in rowmeta if m["fallback"] is not None]
    neg = sum(1 for b in chosen
              if b.get("toward_ablation") is not None
              and np.isfinite(b["toward_ablation"]) and b["toward_ablation"] < 0)
    print(f"   toward's resolution floor binds (|interval| < {MIN_GAP}): "
          f"{(np.abs(iv) < MIN_GAP).mean():.1%} of {len(iv)} rows")
    print(f"   attribution fallback (correction out-of-model): "
          f"{sum(bool(x) for x in fb)}/{len(fb)} = {np.mean([bool(x) for x in fb]):.1%}")
    print(f"   chosen patch moves recipient the WRONG way (toward < 0): "
          f"{neg}/{len(chosen)} = {neg / max(len(chosen), 1):.1%}")
    print("   stop reasons: " + ", ".join(f"{k} {v}" for k, v in stops.most_common()))


if __name__ == "__main__":
    main()
