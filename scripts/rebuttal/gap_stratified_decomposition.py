#!/usr/bin/env python3
"""REBUTTAL: is the on/off-manifold split an artifact of gc-normalization?

gc = (loss_weak - loss_int)/(loss_weak - loss_strong) normalizes by the per-row gap,
so a near-tie row (strong barely beats weak) can post gc~1 on a meaningless nat of
improvement, and mean-of-gc over-weights those trivial rows. This re-does the on/off
split two gc-robust ways, classification rows only (logloss gap in nats):

  1. LOSS-WEIGHTED rel: instead of mean(gc_on)/mean(gc_full), aggregate absolute
     reductions -- rel_on = sum(gc_on * gap) / sum(gc_full * gap). Since gc_c*gap =
     (loss_weak - loss_c), this is (nats removed by the component)/(nats removed by
     the full delta), pooled -- automatically down-weighting near-tie rows.
  2. GAP-STRATIFIED: band rows by |gap| (quartiles) and report the split per band,
     so we can see whether the off-manifold share rides on trivial-gap rows.

gap = logloss_weak - logloss_strong, reconstructed from the forward_deltas npz
(preds_weak/preds_strong/y) joined to the decomposition rows via row_idx.

Usage:
    python -m scripts.rebuttal.gap_stratified_decomposition            # 99% both arms
    python -m scripts.rebuttal.gap_stratified_decomposition --thr 90
"""
import argparse
import glob
import json
import os

import numpy as np

from scripts._project_root import PROJECT_ROOT

EPS = 1e-7


def _cls_gap(pw, ps, y):
    """logloss_weak - logloss_strong per row (nats), matching _gc's clipping."""
    idx = np.arange(len(y))
    ol = -np.log(np.clip(pw[idx, y], EPS, 1 - EPS))
    tl = -np.log(np.clip(ps[idx, y], EPS, 1 - EPS))
    return ol - tl


def collect(arm, thr):
    suf = {80: "_t80", 90: "", 95: "_t95", 99: "_t99"}[thr]
    dec = PROJECT_ROOT / "output" / "rebuttal" / (
        f"functional_decomposition_random{suf}" if arm == "random"
        else f"functional_decomposition{suf}")
    fwd = PROJECT_ROOT / "output" / "rebuttal" / (
        "forward_deltas_random" if arm == "random" else "forward_deltas")
    on, off, full, gap = [], [], [], []
    n_reg = 0
    for jf in glob.glob(str(dec / "*.json")):
        pair = os.path.basename(jf)[:-5]
        for rec in json.load(open(jf)):
            npz = fwd / pair / f"{rec['dataset']}.npz"
            if not npz.exists():
                continue
            z = np.load(npz, allow_pickle=True)
            pw = np.asarray(z["preds_weak"], dtype=np.float64)
            ps = np.asarray(z["preds_strong"], dtype=np.float64)
            if pw.ndim != 2:                      # regression: different loss metric
                n_reg += len(rec["row_idx"])
                continue
            y = np.asarray(z["y_query"]).astype(int)
            r = np.asarray(rec["row_idx"], dtype=int)
            g = _cls_gap(pw[r], ps[r], y[r])
            on.extend(rec["gc_on_manifold_rows"])
            off.extend(rec["gc_off_manifold_rows"])
            full.extend(rec["gc_full_rows"])
            gap.extend(g.tolist())
    return (np.array(on), np.array(off), np.array(full), np.array(gap), n_reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=99, choices=[80, 90, 95, 99])
    args = ap.parse_args()

    for arm in ["trained", "random"]:
        on, off, full, gap, n_reg = collect(arm, args.thr)
        n = len(on)
        # mean-of-gc (the current numbers) vs loss-weighted
        rel_on_m, rel_off_m = on.mean() / full.mean(), off.mean() / full.mean()
        wf = (full * gap).sum()
        rel_on_w, rel_off_w = (on * gap).sum() / wf, (off * gap).sum() / wf
        print(f"\n=== {arm.upper()} @ {args.thr}%  (classification rows n={n}; "
              f"{n_reg} regression rows excluded) ===")
        print(f"  mean-of-gc     : rel_on={rel_on_m:.2f}  rel_off={rel_off_m:.2f}")
        print(f"  LOSS-WEIGHTED  : rel_on={rel_on_w:.2f}  rel_off={rel_off_w:.2f}"
              f"   <- gc-robust")
        # gap-stratified (quartiles)
        q = np.quantile(gap, [0, .25, .5, .75, 1.0])
        print(f"  gap-stratified (logloss_w - logloss_s, nats):")
        print(f"    {'band':<18}{'n':>6}{'med-gap':>9}{'rel_on':>8}{'rel_off':>9}")
        for i in range(4):
            lo, hi = q[i], q[i + 1]
            m = (gap >= lo) & (gap <= hi if i == 3 else gap < hi)
            if not m.sum():
                continue
            f_ = full[m].mean()
            print(f"    [{lo:.3f},{hi:.3f}){'':<3}{m.sum():>6}{np.median(gap[m]):>9.3f}"
                  f"{on[m].mean()/f_:>8.2f}{off[m].mean()/f_:>9.2f}")


if __name__ == "__main__":
    main()
