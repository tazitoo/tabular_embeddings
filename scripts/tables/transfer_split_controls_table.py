#!/usr/bin/env python3
"""Unified on/off-manifold intervention CONTROLS table: where do the transfer deltas
lie on/off the recipient manifold, and which component closes the gap, across the
minimal control set that localizes the effect:

  trained unmatched, no search   (ridge_split,        menu m)
  random dictionary, no search   (ridge_random_split, menu m)   -- the floor control
  trained unmatched, greedy      (greedy_split,       per-row search)   -- headline

For each arm: per-dataset median gap closed (clamped to [0,1]) for the full delta,
its norm-matched on-manifold-only and off-manifold-only components, and the share of
injected energy that is on-manifold. The no-search arms are read at a menu matched to
the greedy's budget (default m=5 ~ median K). Comparing the rows shows whether "the
on-manifold component carries the gap-closing" is (a) specific to trained concepts
(trained vs random) and (b) a property of the search or of any injection (greedy vs
no-search). Mac-only CPU. Usage:
    python -m scripts.tables.transfer_split_controls_table
    python -m scripts.tables.transfer_split_controls_table --menu 5
"""
import argparse
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.figures.plot_transfer_decomposition import TD_DIR

BASE = TD_DIR.parent / "transfer_decomposition"
OUT_TEX = PROJECT_ROOT / "scripts" / "tables" / "transfer_split_controls_table.tex"
PAPER_TABLES = Path.home() / "src" / "tabular_embedding_paper" / "tables"


def _clip01(a):
    return np.clip(a, 0.0, 1.0)


def _read_nosearch(d, menu, clamp):
    """Per-dataset medians of full/on/off (clamped) + on-energy, at menu label."""
    if not d.exists():
        return None
    acc = {k: [] for k in ("full", "on", "off", "energy_on")}
    wrong = {k: [] for k in ("full", "on", "off")}
    n = 0
    for pdir in sorted(x for x in d.iterdir() if x.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if f"full__{menu}" not in z.files:
                continue
            n += 1
            for k in ("full", "on", "off"):
                a = np.asarray(z[f"{k}__{menu}"], dtype=np.float64); a = a[np.isfinite(a)]
                if len(a):
                    acc[k].append(float(np.median(_clip01(a) if clamp else a)))
                    wrong[k].append(float((a < 0).mean()))
            eo = np.asarray(z[f"energy_on__{menu}"], dtype=np.float64); eo = eo[np.isfinite(eo)]
            if len(eo):
                acc["energy_on"].append(float(np.median(eo)))
    return acc, wrong, n


def _read_greedy(d, clamp):
    if not d.exists():
        return None
    acc = {k: [] for k in ("full", "on", "off", "energy_on")}
    wrong = {k: [] for k in ("full", "on", "off")}
    n = 0
    for pdir in sorted(x for x in d.iterdir() if x.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "full" not in z.files or len(z["full"]) == 0:
                continue
            n += 1
            for k in ("full", "on", "off"):
                a = np.asarray(z[k], dtype=np.float64); a = a[np.isfinite(a)]
                if len(a):
                    acc[k].append(float(np.median(_clip01(a) if clamp else a)))
                    wrong[k].append(float((a < 0).mean()))
            eo = np.asarray(z["energy_on"], dtype=np.float64); eo = eo[np.isfinite(eo)]
            if len(eo):
                acc["energy_on"].append(float(np.median(eo)))
    return acc, wrong, n


def _m(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(np.median(a)) if len(a) else float("nan")


def _f(x):
    return "---" if x != x else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Unified on/off split controls table")
    ap.add_argument("--menu", default="5", help="no-search menu label (1/5/|U|)")
    ap.add_argument("--no-clamp", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_TEX)
    args = ap.parse_args()
    clamp = not args.no_clamp

    arms = [
        (f"trained unmatched, no search ($m{{=}}{args.menu}$)",
         _read_nosearch(BASE / "ridge_split", args.menu, clamp)),
        (f"random dict, no search ($m{{=}}{args.menu}$)",
         _read_nosearch(BASE / "ridge_random_split", args.menu, clamp)),
        ("trained unmatched, greedy search",
         _read_greedy(BASE / "greedy_split", clamp)),
    ]

    # ---- stdout pre-read --------------------------------------------------------
    print(f"\n=== ON/OFF SPLIT CONTROLS (pre-read) ===  {'clamped' if clamp else 'unclipped'}")
    print(f"{'arm':44} {'n':>4} {'full':>6} {'on':>6} {'off':>6} {'on-energy':>10}")
    for name, got in arms:
        if got is None:
            print(f"{name:44} {'--':>4}  (no data yet)"); continue
        acc, wrong, n = got
        print(f"{name:44} {n:>4} {_f(_m(acc['full'])):>6} {_f(_m(acc['on'])):>6} "
              f"{_f(_m(acc['off'])):>6} {_f(_m(acc['energy_on'])):>10}")

    # ---- LaTeX ------------------------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l r r r r}", r"\toprule",
         r"Transfer arm & full $\Delta$ & on-only & off-only & on-energy \\",
         r"\midrule"]
    for name, got in arms:
        if got is None:
            L.append(f"{name} & --- & --- & --- & --- \\\\"); continue
        acc, _, n = got
        L.append(f"{name} & {_f(_m(acc['full']))} & {_f(_m(acc['on']))} "
                 f"& {_f(_m(acc['off']))} & {_f(_m(acc['energy_on']))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{%",
          r"  On/off-manifold intervention controls. Each transfer arm's deployed "
          r"  delta is split against the recipient's active subspace; the on- and "
          r"  off-manifold components are norm-matched to the full delta and injected. "
          r"  Columns: per-dataset median gap closed (clamped to $[0,1]$) for the full "
          r"  delta, on-manifold-only, and off-manifold-only, and the on-manifold share "
          r"  of injected energy. No-search arms read at the menu matched to the "
          r"  greedy's budget. Comparing trained vs random isolates the dictionary's "
          r"  role; comparing no-search vs greedy isolates the search's. Mitra-recipient "
          r"  datasets excluded.",
          r"}", r"\label{tab:split_controls}", r"\end{table}"]
    tex = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"\nWrote {args.out}")
    if PAPER_TABLES.exists():
        (PAPER_TABLES / args.out.name).write_text(tex)
        print(f"Copied to {PAPER_TABLES / args.out.name}")


if __name__ == "__main__":
    main()
