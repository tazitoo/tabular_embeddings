#!/usr/bin/env python3
"""Join cross-fitted held-out generality outputs against the paper's per-row
self-selected gap-closed and report the contrast.

The GPU workers that run `heldout_concept_generality.py` do not carry the
forward-ablation cache, so they emit the out-of-fold set gc keyed by global row id
(`oof_row_ids`, `oof_gc_all`) and leave the per-row join blank. This script runs
where the cache lives (the Mac): for each pair/dataset it pairs the OOF set gc
with the reported per-row `gap_closed` row-for-row on the active rows, and also
characterises the selected set -- its size, the fold-to-fold overlap (Jaccard),
and closure as a function of how many unmatched concepts are applied.

Per-dataset columns:
    strong>weak, n_active, n_paired
    size        mean +/- std of the frozen-set size across folds
    jac         mean pairwise Jaccard of the per-fold selected sets (set stability)
    oof         out-of-fold set gc on the paired rows
    perrow      per-row self-selected gc on the same rows (the paper's number)
    drop        perrow - oof
    gc@k        pooled held-out gc at prefix size k = 1,2,3,5 (closure vs set size)

Usage (defaults = canonical path):
    python -m scripts.intervention.heldout_generality_report
    python -m scripts.intervention.heldout_generality_report --pairs tabicl_vs_tabpfn
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.round_paths import RESULTS_DIR, FORWARD_ABLATION_DIR

GENERALITY_DIR = RESULTS_DIR / "heldout_generality"
K_SHOW = (1, 2, 3, 5)


def _pair_rows(gen_npz, abl_npz):
    """Pair OOF set gc against per-row self-selected gc by global row id."""
    gids = gen_npz["oof_row_ids"].astype(np.int64)
    oof = gen_npz["oof_gc_all"].astype(np.float32)
    cpos = {int(ri): i for i, ri in enumerate(abl_npz["row_indices"])}
    cgc = abl_npz["gap_closed"]
    o, p = [], []
    for gid, ov in zip(gids, oof):
        j = cpos.get(int(gid))
        if j is not None and np.isfinite(cgc[j]):
            o.append(float(ov)); p.append(float(cgc[j]))
    return np.asarray(o, dtype=np.float32), np.asarray(p, dtype=np.float32)


def _mean_jaccard(fold_feats):
    """Mean pairwise Jaccard over the per-fold selected sets (1.0 = identical sets,
    0.0 = disjoint). Measures how stable the selected concepts are across folds."""
    sets = [set(f) for f in fold_feats if f]
    if len(sets) < 2:
        return float("nan")
    js = []
    for a, b in combinations(sets, 2):
        u = a | b
        js.append(len(a & b) / len(u) if u else 1.0)
    return float(np.mean(js)) if js else float("nan")


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
            sizes = json.loads(str(g["fold_sizes"]))
            fold_feats = json.loads(str(g["fold_feats"]))
            gc_by_k = json.loads(str(g["gc_by_k"]))      # {k: [mean, n]}
            rec = dict(
                pair=pdir.name, dataset=dataset,
                sw=f"{g['strong_model']}>{g['weak_model']}",
                n_active=int(g["n_active"]),
                size_mean=float(np.mean(sizes)) if sizes else np.nan,
                size_std=float(np.std(sizes)) if sizes else np.nan,
                jaccard=_mean_jaccard(fold_feats),
                n_distinct=len({fi for f in fold_feats for fi in f}),
                gc_at_k={int(k): v[0] for k, v in gc_by_k.items()},
            )
            abl = ablation_dir / pdir.name / f"{dataset}.npz"
            if abl.exists():
                a = np.load(abl, allow_pickle=True)
                o, p = _pair_rows(g, a)
                rec["n_paired"] = int(len(o))
                rec["oof"] = float(o.mean()) if len(o) else np.nan
                rec["perrow"] = float(p.mean()) if len(p) else np.nan
                rec["drop"] = (rec["perrow"] - rec["oof"]) if len(o) else np.nan
            else:
                rec.update(n_paired=0, oof=float(g["oof_set_gc"]),
                           perrow=np.nan, drop=np.nan)
            rows.append(rec)
    return rows


def _f(x):
    return f"{x:.2f}" if isinstance(x, float) and np.isfinite(x) else "  . "


def main():
    p = argparse.ArgumentParser(description="Report held-out generality vs per-row self-selected gc")
    p.add_argument("--generality-dir", type=Path, default=GENERALITY_DIR)
    p.add_argument("--ablation-dir", type=Path, default=FORWARD_ABLATION_DIR)
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    rows = report(args.generality_dir, args.ablation_dir, args.pairs)
    if not rows:
        print("no ok results found under", args.generality_dir); return

    kcols = " ".join(f"gc@{k:>2}" for k in K_SHOW)
    hdr = (f"{'dataset':34} {'S>W':16} {'act':>4} {'pr':>4} "
           f"{'size':>9} {'jac':>5} {'oof':>5} {'perrow':>6} {'drop':>5} {kcols}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        size_s = (f"{r['size_mean']:.1f}±{r['size_std']:.1f}"
                  if np.isfinite(r['size_mean']) else "   .  ")
        kvals = " ".join(f"{_f(r['gc_at_k'].get(k, float('nan'))):>5}" for k in K_SHOW)
        print(f"{r['dataset']:34.34} {r['sw']:16} {r['n_active']:>4} {r['n_paired']:>4} "
              f"{size_s:>9} {_f(r['jaccard']):>5} {_f(r['oof']):>5} {_f(r['perrow']):>6} "
              f"{_f(r['drop']):>5} {kvals}")

    paired = [r for r in rows if r["n_paired"] > 0]
    if paired:
        tot = sum(r["n_paired"] for r in paired)
        pooled_o = sum(r["oof"] * r["n_paired"] for r in paired) / tot
        pooled_p = sum(r["perrow"] * r["n_paired"] for r in paired) / tot
        jac = np.array([r["jaccard"] for r in paired if np.isfinite(r["jaccard"])])
        szm = np.array([r["size_mean"] for r in paired if np.isfinite(r["size_mean"])])
        print("-" * len(hdr))
        print(f"POOLED ({len(paired)} datasets, {tot} rows): "
              f"oof={pooled_o:.3f} perrow={pooled_p:.3f} drop={pooled_p - pooled_o:.3f} | "
              f"set size mean={szm.mean():.1f} | mean pairwise Jaccard={jac.mean():.2f} "
              f"[{jac.min():.2f}, {jac.max():.2f}]")

    if args.csv:
        import csv
        flat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
                for r in rows]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
