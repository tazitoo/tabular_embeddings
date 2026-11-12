#!/usr/bin/env python3
"""LaTeX table for the decisive on/off-manifold injection test (A2 mechanism).

For the unmatched transfer arm, the summed deployed delta Delta is split against
the recipient's active subspace E (the 90%-energy PCA of W's test embeddings, the
SAME basis the A2 composition uses):
    Delta_on  = E E^T Delta ;  Delta_off = Delta - Delta_on
both NORM-MATCHED to ||Delta|| (and each other), then injected and forwarded. This
answers which component closes the gap -- magnitude held fixed, only direction
differs. Produced by `transfer_decomposition.py --project-split` (ridge_split dir).

Same layout as the A1 companion: per-dataset median gap closed at m in {1, 5, |U|}
with pooled-row wrong-direction fraction, three rows -- full injection (reference),
on-manifold only, off-manifold only -- plus the on-manifold ENERGY share (the
composition, from the same E) so the geometry and the mechanism sit together.

Mac-only CPU. Usage:
    python -m scripts.tables.transfer_split_table
"""
import argparse
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.figures.plot_transfer_decomposition import TD_DIR

SPLIT_DIR = TD_DIR.parent / "transfer_decomposition" / "ridge_split"
OUT_TEX = PROJECT_ROOT / "scripts" / "tables" / "transfer_split_table.tex"
PAPER_TABLES = Path.home() / "src" / "tabular_embedding_paper" / "tables"
M_SHOW = ["1", "5", "|U|"]
# (display, npz key prefix) -- full is the reference row.
ARMS = [("full injection", "full"), ("on-manifold only", "on"), ("off-manifold only", "off")]


def _collect():
    """Per-dataset median gc + pooled rows + on-energy, per arm x menu label."""
    if not SPLIT_DIR.exists():
        return None
    per_ds = {k: {l: [] for l in M_SHOW} for _, k in ARMS}
    pooled = {k: {l: [] for l in M_SHOW} for _, k in ARMS}
    e_on = {l: [] for l in M_SHOW}            # per-dataset median on-energy share
    U, n = [], 0
    for pdir in sorted(d for d in SPLIT_DIR.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "full__1" not in z:
                continue
            n += 1
            U.append(len(z["uni_pool_ds"]))
            for lab in M_SHOW:
                for _, k in ARMS:
                    a = np.asarray(z[f"{k}__{lab}"], dtype=np.float64)
                    if np.isfinite(a).any():
                        per_ds[k][lab].append(float(np.nanmedian(a)))
                        pooled[k][lab].append(a[np.isfinite(a)])
                eo = np.asarray(z[f"energy_on__{lab}"], dtype=np.float64)
                if np.isfinite(eo).any():
                    e_on[lab].append(float(np.nanmedian(eo)))
    if not n:
        return None
    return per_ds, pooled, e_on, np.asarray(U), n


def _med(xs):
    a = np.asarray([x for x in xs if x == x], dtype=np.float64)
    return float(np.median(a)) if len(a) else float("nan")


def _sg(x):
    s = f"{x:+.2f}"
    return "+0.00" if s == "-0.00" else s


def main():
    ap = argparse.ArgumentParser(description="On/off-manifold split LaTeX table")
    ap.add_argument("--out", type=Path, default=OUT_TEX)
    args = ap.parse_args()

    got = _collect()
    if got is None:
        print(f"No split npz under {SPLIT_DIR} yet.")
        return
    per_ds, pooled, e_on, U, n = got
    medU = float(np.median(U))      # |U| = full unmatched pool (median; mean is inflated)

    def cell(k, lab):
        med = _med(per_ds[k][lab])
        rows = np.concatenate(pooled[k][lab]) if pooled[k][lab] else np.array([])
        wf = float((rows < 0).mean()) if len(rows) else float("nan")
        return med, wf

    # ---- stdout pre-read --------------------------------------------------------
    print(f"\n=== ON/OFF-MANIFOLD SPLIT (pre-read) ===  units={n}")
    print(f"{'arm':20}" + "".join(f"{'m='+l:>18}" for l in M_SHOW))
    print(f"{'':20}" + "".join(f"{'gc (wrong%)':>18}" for _ in M_SHOW))
    for disp, k in ARMS:
        line = f"{disp:20}"
        for lab in M_SHOW:
            med, wf = cell(k, lab)
            line += f"{(_sg(med)+f' ({wf*100:.0f}%)'):>18}"
        print(line)
    print(f"{'on-energy share':20}" + "".join(f"{_med(e_on[l]):>18.2f}" for l in M_SHOW))
    print(f"(|U| = full unmatched pool, median {medU:.0f} concepts)")

    # ---- LaTeX ------------------------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{l r r r}", r"\toprule",
         r"Injected component & $m{=}1$ & $m{=}5$ & $m{=}|U|$ \\",
         r"\midrule"]
    for disp, k in ARMS:
        cells = []
        for lab in M_SHOW:
            med, wf = cell(k, lab)
            cells.append(f"{_sg(med)} ({wf:.2f})")
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"on-manifold energy share & "
             + " & ".join(f"{_med(e_on[l]):.2f}" for l in M_SHOW) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{%",
          r"  Decisive on/off-manifold injection test (unmatched transfer arm). The "
          r"  summed deployed delta is projected onto the recipient's active subspace "
          r"  (90\%-energy PCA of its embeddings) and its complement; both components "
          r"  are norm-matched to the full delta (and each other) before injection, so "
          r"  only direction differs. Cells are per-dataset median gap closed with the "
          r"  pooled-row wrong-direction fraction in parentheses. The last row is the "
          r"  share of injected energy that is on-manifold (the composition, same "
          r"  subspace). $|U|$ is the full unmatched pool (median " + f"{medU:.0f}" + r" "
          r"  concepts; the overshoot regime). Mitra-recipient datasets excluded.",
          r"}", r"\label{tab:transfer_split}", r"\end{table}"]
    tex = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"\nWrote {args.out}  ({n} units)")
    if PAPER_TABLES.exists():
        (PAPER_TABLES / args.out.name).write_text(tex)
        print(f"Copied to {PAPER_TABLES / args.out.name}")


if __name__ == "__main__":
    main()
