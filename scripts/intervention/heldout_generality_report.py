#!/usr/bin/env python3
"""Join cross-fitted held-out generality outputs against the paper's per-row
circular gap-closed and report the contrast.

The GPU workers that run `heldout_concept_generality.py` do not carry the
forward-ablation cache, so they emit the honest out-of-fold gc keyed by global
row id (`honest_row_ids`, `honest_gc_all`) and leave the circular join blank.
This script runs where the cache lives (the Mac): for each pair/dataset it pairs
the honest OOF gc with the reported per-row `gap_closed` row-for-row on the active
rows, and reports how much of the circular closure survives cross-fitting.

Columns:
    strong>weak, n_active, n_paired
    honest      pooled honest OOF gc over all scored active rows
    honest_pr   honest OOF gc restricted to the paired rows
    circular    per-row circular gc on the same paired rows (the paper's number)
    retention   honest_pr / circular   (fraction of closure that survives)
    drop        circular - honest_pr   (mean per-row loss going honest)
    paper_gc    the dataset's reported mean_gap_closed (reference; strong-wins rows)

Usage (defaults = canonical path):
    python -m scripts.intervention.heldout_generality_report
    python -m scripts.intervention.heldout_generality_report --pairs tabicl_vs_tabpfn
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.round_paths import RESULTS_DIR, FORWARD_ABLATION_DIR

GENERALITY_DIR = RESULTS_DIR / "heldout_generality"


def _pair_rows(gen_npz, abl_npz):
    """Pair honest OOF gc against circular per-row gc by global row id.
    Returns (honest_paired, circ_paired) as float arrays over shared rows."""
    gids = gen_npz["honest_row_ids"].astype(np.int64)
    honest = gen_npz["honest_gc_all"].astype(np.float32)
    cpos = {int(ri): i for i, ri in enumerate(abl_npz["row_indices"])}
    cgc = abl_npz["gap_closed"]
    h, c = [], []
    for gid, hv in zip(gids, honest):
        j = cpos.get(int(gid))
        if j is not None and np.isfinite(cgc[j]):
            h.append(float(hv)); c.append(float(cgc[j]))
    return np.asarray(h, dtype=np.float32), np.asarray(c, dtype=np.float32)


def report(generality_dir, ablation_dir, pairs):
    rows = []
    pair_dirs = sorted(p for p in generality_dir.iterdir() if p.is_dir())
    if pairs:
        pair_dirs = [p for p in pair_dirs if p.name in pairs]
    for pdir in pair_dirs:
        for npz_path in sorted(pdir.glob("*.npz")):
            g = np.load(npz_path, allow_pickle=True)
            if str(g["status"]) != "ok":
                continue
            dataset = npz_path.stem
            abl = ablation_dir / pdir.name / f"{dataset}.npz"
            honest_all = float(g["honest_oof_gc"])
            n_active = int(g["n_active"])
            fold_std = float(g["oof_gc_fold_std"])
            if not abl.exists():
                rows.append(dict(pair=pdir.name, dataset=dataset,
                                 sw=f"{g['strong_model']}>{g['weak_model']}",
                                 n_active=n_active, n_paired=0, honest=honest_all,
                                 honest_pr=np.nan, circular=np.nan, retention=np.nan,
                                 drop=np.nan, paper_gc=np.nan, fold_std=fold_std))
                continue
            a = np.load(abl, allow_pickle=True)
            h, c = _pair_rows(g, a)
            paper_gc = float(a["mean_gap_closed"]) if "mean_gap_closed" in a else np.nan
            honest_pr = float(h.mean()) if len(h) else np.nan
            circular = float(c.mean()) if len(c) else np.nan
            retention = honest_pr / circular if circular and circular > 1e-8 else np.nan
            rows.append(dict(pair=pdir.name, dataset=dataset,
                             sw=f"{g['strong_model']}>{g['weak_model']}",
                             n_active=n_active, n_paired=int(len(h)), honest=honest_all,
                             honest_pr=honest_pr, circular=circular, retention=retention,
                             drop=(circular - honest_pr) if len(h) else np.nan,
                             paper_gc=paper_gc, fold_std=fold_std))
    return rows


def main():
    p = argparse.ArgumentParser(description="Report held-out generality vs circular gc")
    p.add_argument("--generality-dir", type=Path, default=GENERALITY_DIR)
    p.add_argument("--ablation-dir", type=Path, default=FORWARD_ABLATION_DIR)
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    rows = report(args.generality_dir, args.ablation_dir, args.pairs)
    if not rows:
        print("no ok results found under", args.generality_dir); return

    hdr = (f"{'pair':22} {'dataset':34} {'S>W':16} "
           f"{'act':>4} {'pr':>4} {'honest':>7} {'circ':>7} {'ret':>5} {'drop':>6} {'paper':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        def f(x): return f"{x:.2f}" if isinstance(x, float) and np.isfinite(x) else "  . "
        print(f"{r['pair']:22} {r['dataset']:34.34} {r['sw']:16} "
              f"{r['n_active']:>4} {r['n_paired']:>4} {f(r['honest']):>7} {f(r['circular']):>7} "
              f"{f(r['retention']):>5} {f(r['drop']):>6} {f(r['paper_gc']):>6}")

    paired = [r for r in rows if r["n_paired"] > 0]
    if paired:
        # Row-weighted pooled contrast across all paired datasets.
        tot = sum(r["n_paired"] for r in paired)
        pooled_h = sum(r["honest_pr"] * r["n_paired"] for r in paired) / tot
        pooled_c = sum(r["circular"] * r["n_paired"] for r in paired) / tot
        rets = np.array([r["retention"] for r in paired if np.isfinite(r["retention"])])
        print("-" * len(hdr))
        print(f"POOLED ({len(paired)} datasets, {tot} rows): "
              f"honest={pooled_h:.3f} circular={pooled_c:.3f} "
              f"retention={pooled_h / pooled_c:.1%} | "
              f"per-dataset retention median={np.median(rets):.1%} "
              f"[{rets.min():.0%}, {rets.max():.0%}]")

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
