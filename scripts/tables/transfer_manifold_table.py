#!/usr/bin/env python3
"""LaTeX table: where do transferred concept directions land relative to the weak
model's manifold -- per transfer arm, two lenses.

  dir off-manifold : per-DIRECTION energy outside the recipient's active 90%
                     subspace, top-20 by donor importance (arm_direction_manifold).
                     High = the individual injected direction is novel to W.
  deployed-Δ novel : 1 - aligned, where aligned is the share of the SUMMED
                     deployed-delta's energy inside the active subspace
                     (subspace_analysis.analyze_unit, run on the dumped deltas).
                     High = the actual edit the model sees is novel.
  Δ dims           : effective (90%-energy) rank of the deployed delta -- how many
                     recipient directions the summed injection spans.

Rows: the greedy (per-row search, on-disk deltas) and the no-search none-curve arms
(unmatched / matched-map / matched-wdec), trained and random dictionary. The greedy
has no per-direction entry (its menu is per-row), shown as ---.

Pre-read to stdout (overall + by recipient) and LaTeX to scripts/tables/ (+ paper
repo if present). Mac-only CPU. Usage:
    python -m scripts.tables.transfer_manifold_table
    python -m scripts.tables.transfer_manifold_table --pairs carte_vs_tabpfn  # fast test
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.analysis.arm_deployed_delta_subspace import ARMS, _units
from scripts.analysis.arm_direction_manifold import collect as collect_dir

OUT_TEX = PROJECT_ROOT / "scripts" / "tables" / "transfer_manifold_table.tex"
PAPER_TABLES = Path.home() / "src" / "tabular_embedding_paper" / "tables"
RECIPS = ["carte", "tabdpt", "tabicl", "tabicl_v2", "tabpfn", "mitra"]

# row spec: (display, deployed-delta ARMS label, (dir-arm, random) or None)
ROWS = [
    ("greedy (search)",      "greedy trained",       None),
    ("greedy, random dict",  "greedy random",        None),
    ("unmatched",            "unmatched (ridge)",    ("unmatched", False)),
    ("unmatched, random",    "unmatched random",     ("unmatched", True)),
    ("matched (map)",        "matched_map (ridge)",  ("matched_map", False)),
    ("matched (W decoder)",  "matched_wdec",         ("matched_wdec", False)),
]
DD_DIR = {label: d for label, d in ARMS}


def _med(xs):
    a = np.asarray([x for x in xs if x == x], dtype=np.float64)
    return (float(np.median(a)), len(a)) if len(a) else (float("nan"), 0)


def _fmt(x):
    return "---" if (x != x) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Transfer manifold LaTeX table")
    ap.add_argument("--pairs", nargs="+", default=None, help="subset for fast testing")
    ap.add_argument("--var-threshold", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=OUT_TEX)
    args = ap.parse_args()

    # ---- per-direction off-manifold (top-20) : trained + random -----------------
    print("computing per-direction off-manifold (trained)...")
    _, dir_top_t, dir_recip_t, _ = collect_dir(False, args.var_threshold, args.pairs)
    print("computing per-direction off-manifold (random)...")
    _, dir_top_r, dir_recip_r, _ = collect_dir(True, args.var_threshold, args.pairs)
    dir_top = {False: dir_top_t, True: dir_top_r}
    dir_recip = {False: dir_recip_t, True: dir_recip_r}

    # ---- deployed-delta subspace split per arm ----------------------------------
    emb_cache, norm_cache = {}, {}
    dd = {}
    for label, d in ARMS:
        if not Path(d).exists():
            dd[label] = []
            continue
        us = _units(d, emb_cache, norm_cache)
        if args.pairs:
            us = [u for u in us if f"{min(u['donor'],u['recipient'])}_vs_{max(u['donor'],u['recipient'])}" in args.pairs]
        dd[label] = us

    # ---- assemble rows ----------------------------------------------------------
    table = []
    for disp, ddlabel, perdir in ROWS:
        us = dd.get(ddlabel, [])
        aligned, n = _med([u["aligned_fraction"] for u in us])
        angle, _ = _med([u["median_principal_angle_deg"] for u in us])
        kd, _ = _med([u["delta_sig_dim_90"] for u in us])
        novel = (1 - aligned) if aligned == aligned else float("nan")
        if perdir is not None:
            arm, rnd = perdir
            diroff, _ = _med(dir_top[rnd][arm])
        else:
            diroff = float("nan")
        # per-recipient deployed-Δ novel
        by = defaultdict(list)
        for u in us:
            by[u["recipient"]].append(u["aligned_fraction"])
        recip_novel = {r: (1 - _med(by[r])[0]) if by.get(r) else float("nan") for r in RECIPS}
        table.append(dict(disp=disp, n=n, diroff=diroff, aligned=aligned, novel=novel,
                          kd=kd, angle=angle, recip_novel=recip_novel))

    # ---- stdout pre-read --------------------------------------------------------
    print("\n=== TRANSFER MANIFOLD (pre-read) ===")
    print(f"{'arm':22} {'n':>5} {'dir_off':>8} {'dd_align':>9} {'dd_novel':>9} {'Δdim':>5} {'angle':>6}")
    for r in table:
        print(f"{r['disp']:22} {r['n']:>5} {_fmt(r['diroff']):>8} {_fmt(r['aligned']):>9} "
              f"{_fmt(r['novel']):>9} {('' if r['kd']!=r['kd'] else str(int(r['kd']))):>5} "
              f"{_fmt(r['angle']):>6}")
    print(f"\ndeployed-Δ novel (1-aligned) by recipient")
    print(f"{'arm':22} " + "  ".join(f"{r:>9}" for r in RECIPS))
    for r in table:
        print(f"{r['disp']:22} " + "  ".join(f"{_fmt(r['recip_novel'][x]):>9}" for x in RECIPS))

    # ---- LaTeX ------------------------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"Transfer arm & $n$ & dir.\ off-man. & dep.\ aligned & dep.\ novel & $\Delta$ dim \\",
         r"\midrule"]
    for r in table:
        kd = "" if r["kd"] != r["kd"] else str(int(r["kd"]))
        L.append(f"{r['disp']} & {r['n']} & {_fmt(r['diroff'])} & {_fmt(r['aligned'])} "
                 f"& {_fmt(r['novel'])} & {kd} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{%",
          r"  Where transferred concept directions land relative to the weak (recipient) "
          r"  model's active manifold. \emph{dir.\ off-man.}: per-direction energy outside "
          r"  the recipient's top-90\% PCA subspace, median over the top-20 injected "
          r"  concepts. \emph{dep.\ aligned}/\emph{novel}: share of the summed deployed "
          r"  delta's energy inside/outside that subspace. $\Delta$ dim: effective "
          r"  (90\%-energy) rank of the deployed delta. The greedy searches a per-row "
          r"  menu, so it has no single per-direction entry (---). Mitra-recipient "
          r"  datasets are excluded from the none-curve arms.",
          r"}", r"\label{tab:transfer_manifold}", r"\end{table}"]
    tex = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"\nWrote {args.out}")
    if PAPER_TABLES.exists():
        (PAPER_TABLES / args.out.name).write_text(tex)
        print(f"Copied to {PAPER_TABLES / args.out.name}")


if __name__ == "__main__":
    main()
