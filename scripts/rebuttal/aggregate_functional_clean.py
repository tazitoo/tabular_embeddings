#!/usr/bin/env python3
"""REBUTTAL: aggregate functional_decomposition gc_active/gc_tail (ofnL / R2)
over a CLEAN comparison set — classification only, matched recipient across
trained/random, and dropping baseline-swap rows.

Baselines drifted between perrow_importance (trained) and
perrow_importance_random (random), so we compare only where the two arms agree:
  - drop datasets whose recipient (weaker model) flips between arms;
  - drop rows whose baseline predicted class differs between arms.
Regression is excluded upstream (functional_decomposition is classification-only).

Reads functional_decomposition/<pair>.json (per-row gc arrays + row_idx) for the
trained arm and, if present, functional_decomposition_random/<pair>.json for the
random arm; reports pooled-row means for each.

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
IMP = PROJECT_ROOT / "output/perrow_importance"
IMP_R = PROJECT_ROOT / "output/perrow_importance_random"


def clean_mask(recipient, dataset):
    """True where the recipient's baseline predicted class AGREES across arms."""
    fa = IMP / recipient / f"{dataset}.npz"
    fb = IMP_R / recipient / f"{dataset}.npz"
    if not (fa.exists() and fb.exists()):
        return None
    pa = np.asarray(np.load(fa, allow_pickle=True)["baseline_preds"], float)
    pb = np.asarray(np.load(fb, allow_pickle=True)["baseline_preds"], float)
    if pa.ndim != 2 or pa.shape != pb.shape:
        return None
    return np.argmax(pa, 1) == np.argmax(pb, 1)


def load(dirpath):
    out = {}
    for f in glob.glob(f"{dirpath}/*.json"):
        pair = os.path.basename(f)[:-5]
        for rec in json.load(open(f)):
            out[(pair, rec["dataset"])] = rec
    return out


def main():
    tr = load(str(TRAINED))
    rn = load(str(RANDOM)) if RANDOM.exists() else {}
    pooled = {"trained": defaultdict(list), "random": defaultdict(list)}
    by_recip = defaultdict(lambda: {"trained": defaultdict(list), "random": defaultdict(list)})
    n_ds = n_flip = n_nomask = 0

    for (pair, ds), rec_t in tr.items():
        rec_r = rn.get((pair, ds))
        if rec_r is not None and rec_r["recipient"] != rec_t["recipient"]:
            n_flip += 1
            continue
        m = clean_mask(rec_t["recipient"], ds)
        if m is None:
            n_nomask += 1
            continue
        n_ds += 1
        rec = rec_t["recipient"]
        for arm, r in (("trained", rec_t), ("random", rec_r)):
            if r is None:
                continue
            ri = np.asarray(r["row_idx"], int)
            keep = ri < len(m)
            ri = ri[keep]
            km = m[ri]
            for key in ("active", "tail", "full"):
                vals = np.asarray(r[f"gc_{key}_rows"], float)[keep][km]
                pooled[arm][key].extend(vals.tolist())
                by_recip[rec][arm][key].extend(vals.tolist())

    print(f"\nCLEAN functional decomposition (cls-only, matched recipient, no baseline-swap rows)")
    print(f"  datasets used={n_ds}  recipient-flip dropped={n_flip}  no-mask={n_nomask}")
    print(f"\n  {'arm':9}{'n_rows':>8}{'gc_active':>11}{'gc_tail':>10}{'gc_full':>10}")
    for arm in ("trained", "random"):
        a = pooled[arm]["active"]
        if not a:
            print(f"  {arm:9}{'(no data)':>8}")
            continue
        print(f"  {arm:9}{len(a):>8}{np.mean(a):>11.3f}"
              f"{np.mean(pooled[arm]['tail']):>10.3f}{np.mean(pooled[arm]['full']):>10.3f}")

    print(f"\n  by recipient (trained | random gc_active / gc_tail):")
    for rec in sorted(by_recip):
        t, r = by_recip[rec]["trained"], by_recip[rec]["random"]
        ta = f"{np.mean(t['active']):.2f}/{np.mean(t['tail']):.2f}" if t["active"] else "--"
        ra = f"{np.mean(r['active']):.2f}/{np.mean(r['tail']):.2f}" if r["active"] else "--"
        print(f"    {rec:11} trained {ta:12} random {ra:12} (n_tr={len(t['active'])})")

    out = PROJECT_ROOT / "output/rebuttal/functional_clean_summary.json"
    out.write_text(json.dumps({
        arm: {k: (float(np.mean(pooled[arm][k])) if pooled[arm][k] else None)
              for k in ("active", "tail", "full")} | {"n_rows": len(pooled[arm]["active"])}
        for arm in ("trained", "random")}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
