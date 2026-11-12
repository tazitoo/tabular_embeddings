#!/usr/bin/env python3
"""LaTeX companion table for the matched-vs-unmatched transfer panel: the numbers
behind plot_transfer_matched_vs_unmatched, so reviewers can cite them directly.

Per no-search none arm (importance, dataset ordering) x menu m in {1, 5, |U|}:

  median gc (IQR)   : per-DATASET median gap closed, median over datasets, with
                      the [p25, p75] interquartile range over datasets. Matches
                      the curve + IQR band in the figure.
  pooled-row gc     : median gap closed over ALL rows pooled across datasets
                      (no per-dataset weighting) -- the typical row, not the
                      typical dataset.
  wrong-dir. frac   : fraction of rows with gc < 0 (injection moved the weak model
                      AWAY from strong), pooled over all rows.

Arms: unmatched via ridge map (the headline), matched via the same map, matched
via W's own decoder (the underweighting operation). The on-disk greedy (per-row
search) is the reference row: it has no top-m menu, so it reports its gc and the
median accepted-concept count K instead.

Reads the same per-(tag,pair,dataset) npz as the figure (transfer_decomposition.py
dumps them). Mitra-recipient datasets are not dumped for the none arms (OOM), so
they are absent here by construction. Mac-only CPU. Usage:
    python -m scripts.tables.transfer_gc_table
"""
import argparse
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.round_paths import FORWARD_DELTAS_DIR
from scripts.figures.plot_transfer_decomposition import TD_DIR

OUT_TEX = PROJECT_ROOT / "scripts" / "tables" / "transfer_gc_table.tex"
PAPER_TABLES = Path.home() / "src" / "tabular_embedding_paper" / "tables"

# (display, tag, arm-key). arm-key "imp_ds" = importance order, per-dataset sign.
ARMS = [
    ("unmatched (map)",      "ridge",         "imp_ds"),
    ("matched (map)",        "ridge_matched", "imp_ds"),
    ("matched (W decoder)",  "matched_wdec",  "imp_ds"),
]
M_SHOW = ["1", "5", "|U|"]


def _collect(tag, arm):
    """Return per-dataset median gc, pooled row gc, and |U|, per menu label.
    {lab: ndarray of per-dataset medians}, {lab: ndarray of ALL pooled rows}, |U| array."""
    root = TD_DIR / tag
    if not root.exists():
        return None, None, None
    per_ds = {lab: [] for lab in M_SHOW}
    pooled = {lab: [] for lab in M_SHOW}
    U = []
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if f"{arm}__1" not in z:
                continue
            for lab in M_SHOW:
                a = np.asarray(z[f"{arm}__{lab}"], dtype=np.float64)
                per_ds[lab].append(float(np.median(a)))
                pooled[lab].append(a)
            U.append(len(z["uni_pool_ds"]))
    if not U:
        return None, None, None
    return ({lab: np.asarray(v) for lab, v in per_ds.items()},
            {lab: np.concatenate(v) for lab, v in pooled.items()},
            np.asarray(U))


def _greedy():
    """Per-dataset median gc, pooled active-row gc, median K (on-disk greedy search)."""
    if not FORWARD_DELTAS_DIR.exists():
        return None
    per_ds, pooled, ks = [], [], []
    for pdir in sorted(d for d in FORWARD_DELTAS_DIR.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "optimal_k" not in z:
                continue
            k = z["optimal_k"].astype(int)
            act = k > 0
            if act.sum():
                g = np.asarray(z["gap_closed"], dtype=np.float64)[act]
                per_ds.append(float(np.median(g)))
                pooled.append(g)
                ks.append(float(np.median(k[act])))
    if not per_ds:
        return None
    return (np.asarray(per_ds), np.concatenate(pooled), np.asarray(ks))


def _iqr(a):
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def _sg(x):
    """Signed 2-dp, but never a negative zero (rounding artifact)."""
    s = f"{x:+.2f}"
    return "+0.00" if s == "-0.00" else s


def main():
    ap = argparse.ArgumentParser(description="Transfer gc companion LaTeX table")
    ap.add_argument("--out", type=Path, default=OUT_TEX)
    args = ap.parse_args()

    rows = []            # (disp, m-label, med, p25, p75, pooled_med, wrongfrac, K)
    meanU = None
    for disp, tag, arm in ARMS:
        per_ds, pooled, U = _collect(tag, arm)
        if per_ds is None:
            print(f"  (no npz for tag {tag} -- skipping)")
            continue
        meanU = float(U.mean())
        for lab in M_SHOW:
            med, p25, p75 = _iqr(per_ds[lab])
            prow = float(np.median(pooled[lab]))
            wf = float((pooled[lab] < 0).mean())
            rows.append((disp, lab, med, p25, p75, prow, wf, None))

    g = _greedy()
    greedy_row = None
    if g is not None:
        per_ds, pooled, ks = g
        greedy_row = ("greedy (search)", "search", float(np.median(per_ds)),
                      None, None, float(np.median(pooled)),
                      float((pooled < 0).mean()), float(np.median(ks)))

    def mlabel(lab):
        return f"$|U|\\approx{meanU:.0f}$" if lab == "|U|" else lab

    # ---- stdout pre-read --------------------------------------------------------
    print("\n=== TRANSFER gc COMPANION (pre-read) ===")
    print(f"{'arm':20} {'m':>8} {'med gc':>8} {'IQR':>15} {'pooled':>8} {'wrong%':>8} {'K':>5}")
    for disp, lab, med, p25, p75, prow, wf, K in rows:
        iqr = f"[{p25:+.2f},{p75:+.2f}]"
        print(f"{disp:20} {lab if lab!='|U|' else f'|U|~{meanU:.0f}':>8} {med:>8.2f} {iqr:>15} "
              f"{prow:>8.2f} {wf*100:>7.0f}% {'':>5}")
    if greedy_row:
        _, _, med, _, _, prow, wf, K = greedy_row
        print(f"{'greedy (search)':20} {'search':>8} {med:>8.2f} {'':>15} "
              f"{prow:>8.2f} {wf*100:>7.0f}% {int(K):>5}")

    # ---- LaTeX ------------------------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l l r r r r}", r"\toprule",
         r"Arm & $m$ & median gc (IQR) & pooled-row gc & wrong-dir.\ frac & $K$ \\",
         r"\midrule"]
    last = None
    for disp, lab, med, p25, p75, prow, wf, K in rows:
        name = disp if disp != last else ""
        last = disp
        if name and name != rows[0][0]:
            L.append(r"\addlinespace[2pt]")
        iqr = f"[{_sg(p25)},\\,{_sg(p75)}]"
        L.append(f"{name} & {mlabel(lab)} & {_sg(med)}\\ {iqr} & {_sg(prow)} & "
                 f"{wf:.2f} & \\\\")
    if greedy_row:
        _, _, med, _, _, prow, wf, K = greedy_row
        L.append(r"\midrule")
        L.append(f"greedy (search) & search & {_sg(med)} & {_sg(prow)} & {wf:.2f} & {int(K)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{%",
          r"  Gap closed by each no-search transfer arm versus menu size $m$ (top-$m$ "
          r"  donor concepts by importance, per-dataset sign). \emph{median gc (IQR)}: "
          r"  per-dataset median, median and interquartile range over datasets. "
          r"  \emph{pooled-row gc}: median over all rows pooled. \emph{wrong-dir.\ frac}: "
          r"  fraction of rows moved away from the strong model ($gc<0$). The greedy "
          r"  per-row search has no fixed menu, so it reports its gc and median accepted "
          r"  count $K$. Mitra-recipient datasets are excluded from the none arms.",
          r"}", r"\label{tab:transfer_gc}", r"\end{table}"]
    tex = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"\nWrote {args.out}")
    if PAPER_TABLES.exists():
        (PAPER_TABLES / args.out.name).write_text(tex)
        print(f"Copied to {PAPER_TABLES / args.out.name}")


if __name__ == "__main__":
    main()
