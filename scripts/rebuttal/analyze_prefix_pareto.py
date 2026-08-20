#!/usr/bin/env python3
"""App F.3: the interpretability operating point, read from recorded prefix curves.

The tension (user, 2026-08-19): the "perfect patch" may be many columns at full
suppression and zero blast; the interpretable patch is fewer columns and less
performant. The search is NOT truncated for this -- max_steps died for silently
cutting 27.3% of v10 patches mid-improvement -- so the judgment is post-hoc:
every prefix of a committed path is itself a patch, the trajectory records the
full metric state per commit (score, suppression, blast forms, recon,
cumulative edit distance), and losing beam branches carry per-step metrics too,
because a losing branch can own the best SHORT patch while losing at full
length.

This report answers, per sweep:

  1. Marginal value by step: what does column k buy, in score and in its
     components? If the distributions show a consistent elbow, patch@elbow
     becomes a REPORTED second operating point beside patch@full. If they are
     mushy, that is the finding -- "I know it when I see it" with data on why.
  2. patch@k vs patch@full: the metric cost of stopping at each k, using the
     cross-branch best prefix (not just the winner's), since stopping the beam
     at k is not the same as truncating the winner.
  3. The elbow rule candidate: smallest k whose marginal score gain falls below
     --elbow-tol of the running score. A parameter, printed with every number
     it produces, never a silent default baked into results.

Usage:
    python -m scripts.rebuttal.analyze_prefix_pareto --inputs "output/rebuttal/patchv29clf_*.json"
"""
import argparse
import glob
import json

import numpy as np

from scripts._project_root import PROJECT_ROOT


def branch_prefix_curves(row):
    """Every branch's prefix curve for one row: list of per-branch lists of
    per-step metric dicts. Falls back to the winner's trajectory for pre-4bd3af8
    outputs that lack per-branch steps."""
    brs = row.get("beam_branches") or []
    curves = [b["steps"] for b in brs if b.get("steps")]
    if not curves and row.get("trajectory"):
        curves = [row["trajectory"]]
    return curves


def best_prefix_at_k(curves, k):
    """The best (by score) k-column prefix across ALL branches; None if no
    branch reaches k commits."""
    cands = [c[k - 1] for c in curves if len(c) >= k]
    return max(cands, key=lambda t: t["score"]) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="sweep files or globs; every file is reported, none skipped")
    ap.add_argument("--elbow-tol", type=float, default=0.10,
                    help="elbow candidate = smallest k whose marginal score gain "
                         "is below this fraction of the running score")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "rebuttal"
                                         / "prefix_pareto.json"))
    args = ap.parse_args()

    paths = sorted(p for pat in args.inputs for p in glob.glob(pat))
    print(f"{len(paths)} files: {[p.rsplit('/', 1)[-1] for p in paths]}")

    rows_out = []
    for p in paths:
        for c in json.load(open(p)):
            for ds in c.get("datasets") or []:
                for r in ds.get("rows") or []:
                    if not r.get("best"):
                        continue
                    curves = branch_prefix_curves(r)
                    if not curves:
                        continue
                    kmax = max(len(cv) for cv in curves)
                    prefix = [best_prefix_at_k(curves, k) for k in range(1, kmax + 1)]
                    prefix = [t for t in prefix if t is not None]
                    if not prefix:
                        continue
                    scores = [t["score"] for t in prefix]
                    # marginal gain of step k relative to the best (k-1)-prefix
                    marg = [float("nan")] + [
                        (scores[i] - scores[i - 1]) / max(abs(scores[i - 1]), 1e-12)
                        for i in range(1, len(scores))]
                    elbow = next((i + 1 for i in range(1, len(scores))
                                  if marg[i] < args.elbow_tol), len(scores))
                    full = prefix[-1]
                    at_e = prefix[elbow - 1]
                    rows_out.append({
                        "donor": c["donor"], "feat": c["feat"],
                        "dataset": ds["dataset"], "row": r["row"],
                        "k_full": len(prefix), "k_elbow": elbow,
                        "marginal_gains": [round(m, 4) if np.isfinite(m) else None
                                           for m in marg],
                        "full": {m: full.get(m) for m in
                                 ("score", "suppression_frac", "blast", "blast_delta",
                                  "blast_raw", "recon_loss", "edit_distance")},
                        "elbow": {m: at_e.get(m) for m in
                                  ("score", "suppression_frac", "blast", "blast_delta",
                                   "blast_raw", "recon_loss", "edit_distance")}})

    n = len(rows_out)
    print(f"\n{n} patched rows with prefix curves   (elbow-tol = {args.elbow_tol})")
    kf = np.array([r["k_full"] for r in rows_out])
    ke = np.array([r["k_elbow"] for r in rows_out])
    print(f"k_full   p25/p50/p75: {np.percentile(kf, 25):.0f}/{np.median(kf):.0f}/"
          f"{np.percentile(kf, 75):.0f}   k_elbow: {np.percentile(ke, 25):.0f}/"
          f"{np.median(ke):.0f}/{np.percentile(ke, 75):.0f}")
    print(f"rows where elbow == full (nothing to trim): {(ke == kf).sum()} "
          f"({(ke == kf).mean():.0%})")

    print("\nmarginal score gain by step (over rows reaching that step):")
    for k in range(2, int(kf.max()) + 1):
        g = [r["marginal_gains"][k - 1] for r in rows_out
             if r["k_full"] >= k and r["marginal_gains"][k - 1] is not None]
        if len(g) < 10:
            print(f"  step {k}: n={len(g)} (too few, not summarized)")
            continue
        print(f"  step {k}: n={len(g):4d}  med {np.median(g):+7.1%}  "
              f"p25 {np.percentile(g, 25):+7.1%}  p75 {np.percentile(g, 75):+7.1%}")

    trimmed = [r for r in rows_out if r["k_elbow"] < r["k_full"]]
    if trimmed:
        print(f"\npatch@elbow vs patch@full on the {len(trimmed)} trimmable rows "
              f"(what stopping early gives back):")
        for m in ("score", "suppression_frac", "blast_delta", "blast_raw",
                  "edit_distance", "recon_loss"):
            fv = np.array([t["full"][m] for t in trimmed
                           if t["full"].get(m) is not None], dtype=float)
            ev = np.array([t["elbow"][m] for t in trimmed
                           if t["elbow"].get(m) is not None], dtype=float)
            if len(fv) and len(ev):
                print(f"  {m:17s} full med {np.nanmedian(fv):8.3f}  "
                      f"elbow med {np.nanmedian(ev):8.3f}")

    json.dump({"elbow_tol": args.elbow_tol, "rows": rows_out},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
