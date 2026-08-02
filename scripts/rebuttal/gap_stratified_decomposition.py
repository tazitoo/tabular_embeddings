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
    python -m scripts.rebuttal.gap_stratified_decomposition --by recipient
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
    on, off, full, gap, recip, pair_lbl = [], [], [], [], [], []
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
            recip.extend([rec["recipient"]] * len(r))
            pair_lbl.extend([f"{rec['donor']}->{rec['recipient']}"] * len(r))
    return (np.array(on), np.array(off), np.array(full), np.array(gap),
            np.array(recip), np.array(pair_lbl), n_reg)


def _quartile_rows(on, off, full, gap):
    """Per-quartile (rel_on, rel_off) rows, quartiles cut within the given subset."""
    q = np.quantile(gap, [0, .25, .5, .75, 1.0])
    out = []
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        m = (gap >= lo) & (gap <= hi if i == 3 else gap < hi)
        if not m.sum():
            continue
        f_ = full[m].mean()
        out.append((lo, hi, int(m.sum()), np.median(gap[m]),
                    on[m].mean() / f_, off[m].mean() / f_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=99, choices=[80, 90, 95, 99])
    ap.add_argument("--by", choices=["recipient", "pair"],
                    help="also break the quartile trend down per group "
                         "(robustness: is the rise broad-based or a few pairs?)")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="skip groups smaller than this in the --by breakdown")
    args = ap.parse_args()

    per_group = {}
    for arm in ["trained", "random"]:
        on, off, full, gap, recip, pair_lbl, n_reg = collect(arm, args.thr)
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
        print(f"  gap-stratified (logloss_w - logloss_s, nats):")
        print(f"    {'band':<18}{'n':>6}{'med-gap':>9}{'rel_on':>8}{'rel_off':>9}")
        for lo, hi, m_n, med, r_on, r_off in _quartile_rows(on, off, full, gap):
            print(f"    [{lo:.3f},{hi:.3f}){'':<3}{m_n:>6}{med:>9.3f}"
                  f"{r_on:>8.2f}{r_off:>9.2f}")

        if args.by:
            labels = recip if args.by == "recipient" else pair_lbl
            per_group[arm] = {}
            for g in sorted(set(labels)):
                m = labels == g
                if m.sum() < args.min_rows:
                    continue
                rows = _quartile_rows(on[m], off[m], full[m], gap[m])
                if len(rows) == 4:
                    per_group[arm][g] = ([r[5] for r in rows], int(m.sum()))

    if args.by:
        print(f"\n=== PER-{args.by.upper()} rel_off by gap quartile "
              f"(quartiles cut within group; groups <{args.min_rows} rows skipped) ===")
        groups = sorted(set(per_group["trained"]) | set(per_group["random"]))
        print(f"  {'group':<26}{'arm':<9}{'n':>6}   Q1   Q2   Q3   Q4   Q4-Q1")
        n_rise = {"trained": 0, "random": 0}
        for g in groups:
            for arm in ["trained", "random"]:
                if g not in per_group[arm]:
                    print(f"  {g:<26}{arm:<9}{'--':>6}   (below min-rows)")
                    continue
                qs, gn = per_group[arm][g]
                n_rise[arm] += qs[3] > qs[0]
                print(f"  {g:<26}{arm:<9}{gn:>6}" +
                      "".join(f"{v:>5.2f}" for v in qs) + f"{qs[3]-qs[0]:>8.2f}")
        # The gate: if the pooled Q1->Q4 rise were a real trained-vs-random signal it
        # should show up broadly here (trained rises, random does not). It does not --
        # see camera_ready_todo.md SS C.
        for arm in ["trained", "random"]:
            print(f"  {arm}: Q4 > Q1 in {n_rise[arm]}/{len(per_group[arm])} groups")


if __name__ == "__main__":
    main()
