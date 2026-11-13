#!/usr/bin/env python3
"""LaTeX table: decisive on/off-manifold injection test on the GREEDY accepted
deltas -- the per-recipient companion to tab:offmanifold (which reports only the
COMPOSITION, i.e. how much of each greedy delta is off-manifold). Here we inject
the on-manifold-only and off-manifold-only components (each norm-matched to the
full delta) and report which closes the gap.

Produced by `transfer_decomposition.py --greedy-split` (greedy_split dir). Per
recipient and overall: per-dataset median gap closed for full / on-only / off-only
(clamped to [0,1] -- the greedy delta overshoots, so clamping reads as "fraction of
the gap closed"), plus the on-manifold energy share (composition, same subspace).
The full column should track the paper's greedy gc (~0.9-1.0); on vs off is the
mechanism answer. Mac-only CPU. Usage:
    python -m scripts.tables.transfer_greedy_split_table
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.figures.plot_transfer_decomposition import TD_DIR

GS_DIR = TD_DIR.parent / "transfer_decomposition" / "greedy_split"
OUT_TEX = PROJECT_ROOT / "scripts" / "tables" / "transfer_greedy_split_table.tex"
PAPER_TABLES = Path.home() / "src" / "tabular_embedding_paper" / "tables"
RECIP_NAME = {"tabpfn": "TabPFN", "tabicl": "TabICL", "tabicl_v2": "TabICL-v2",
              "mitra": "Mitra", "tabdpt": "TabDPT", "carte": "CARTE"}
ORDER = ["tabpfn", "tabicl", "tabicl_v2", "mitra", "tabdpt", "carte"]


def _clip01(a):
    return np.clip(a, 0.0, 1.0)


def _collect(clamp=True):
    """Per-recipient lists of per-dataset medians for full/on/off + on-energy."""
    if not GS_DIR.exists():
        return None
    by = defaultdict(lambda: {k: [] for k in ("full", "on", "off", "energy_on")})
    allr = {k: [] for k in ("full", "on", "off", "energy_on")}
    n = 0
    for pdir in sorted(d for d in GS_DIR.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "full" not in z.files or len(z["full"]) == 0:
                continue
            n += 1
            recip = str(z["weak"])
            for k in ("full", "on", "off"):
                a = np.asarray(z[k], dtype=np.float64)
                a = a[np.isfinite(a)]
                if not len(a):
                    continue
                m = float(np.median(_clip01(a) if clamp else a))
                by[recip][k].append(m); allr[k].append(m)
            eo = np.asarray(z["energy_on"], dtype=np.float64); eo = eo[np.isfinite(eo)]
            if len(eo):
                by[recip]["energy_on"].append(float(np.median(eo)))
                allr["energy_on"].append(float(np.median(eo)))
    return by, allr, n


def _med(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(np.median(a)) if len(a) else float("nan")


def _f(x):
    return "---" if x != x else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Greedy on/off-manifold split table")
    ap.add_argument("--out", type=Path, default=OUT_TEX)
    ap.add_argument("--no-clamp", action="store_true", help="report unclipped gc medians")
    args = ap.parse_args()

    got = _collect(clamp=not args.no_clamp)
    if got is None:
        print(f"No greedy_split npz under {GS_DIR} yet.")
        return
    by, allr, n = got
    recips = [r for r in ORDER if r in by] + [r for r in by if r not in ORDER]

    # ---- stdout pre-read --------------------------------------------------------
    print(f"\n=== GREEDY on/off-manifold SPLIT (pre-read) ===  units={n}  "
          f"{'(clamped [0,1])' if not args.no_clamp else '(unclipped)'}")
    print(f"{'recipient':12} {'n':>4} {'full':>6} {'on':>6} {'off':>6} {'on-energy':>10}")
    for r in recips:
        d = by[r]
        print(f"{RECIP_NAME.get(r,r):12} {len(d['full']):>4} {_f(_med(d['full'])):>6} "
              f"{_f(_med(d['on'])):>6} {_f(_med(d['off'])):>6} {_f(_med(d['energy_on'])):>10}")
    print(f"{'OVERALL':12} {len(allr['full']):>4} {_f(_med(allr['full'])):>6} "
          f"{_f(_med(allr['on'])):>6} {_f(_med(allr['off'])):>6} {_f(_med(allr['energy_on'])):>10}")

    # ---- LaTeX ------------------------------------------------------------------
    # Composition (off-manifold energy) is NOT re-reported here -- it stays the single
    # number in tab:offmanifold (this table adds only the mechanism: on vs off gc).
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l r r r}", r"\toprule",
         r"Recipient & full $\Delta$ & on-manifold only & off-manifold only \\",
         r"\midrule"]
    for r in recips:
        d = by[r]
        L.append(f"{RECIP_NAME.get(r,r)} & {_f(_med(d['full']))} & {_f(_med(d['on']))} "
                 f"& {_f(_med(d['off']))} \\\\")
    L.append(r"\midrule")
    L.append(f"Overall & {_f(_med(allr['full']))} & {_f(_med(allr['on']))} "
             f"& {_f(_med(allr['off']))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{%",
          r"  Decisive on/off-manifold test on the greedy accepted transfer deltas "
          r"  (the same deltas whose composition is Table~\ref{tab:offmanifold}). Each "
          r"  per-row searched delta is split against the recipient's active subspace; "
          r"  the on-manifold-only and off-manifold-only components are each "
          r"  norm-matched to the full delta and injected. Columns are per-dataset "
          r"  median gap closed (clamped to $[0,1]$). Whether off-only closes the gap "
          r"  tests directly whether the off-manifold energy reported in "
          r"  Table~\ref{tab:offmanifold} is doing the work. Mitra-recipient datasets "
          r"  excluded (forward intractable on 24\,GB).",
          r"}", r"\label{tab:greedy_split}", r"\end{table}"]
    tex = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"\nWrote {args.out}  ({n} units)")
    if PAPER_TABLES.exists():
        (PAPER_TABLES / args.out.name).write_text(tex)
        print(f"Copied to {PAPER_TABLES / args.out.name}")


if __name__ == "__main__":
    main()
