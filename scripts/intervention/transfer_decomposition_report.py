#!/usr/bin/env python3
"""Aggregate the transfer-panel decomposition across datasets and pairs.

Reads the per-(tag,pair,dataset) npz written by transfer_decomposition.py (tag =
map, plus "_random" for the random-SAE dictionary) and pools the UNCLIPPED
gap_closed toward strong, row-weighted over S-wins rows, per cell.

Selection cells (map = learned ridge):
    T2  importance, row,     none   ridge imp_row
    T3  importance, dataset, none   ridge imp_ds   (== M1)
    T4  uniform,    dataset, none   ridge uni_ds
    T5  importance, dataset, none   ridge imp_ds, RANDOM dictionary
Map cells (selection fixed = importance, dataset, none):
    M1  ridge        == T3
    M2  shuffled     ridge pipeline, landmark pairing permuted
    M4  random_orth  random semi-orthogonal floor
Greedy references (read from the on-disk transfer sweep, not recomputed):
    T1  importance, row, greedy, ridge, trained   (forward_deltas)
    T6  importance, row, greedy, ridge, random    (forward_deltas_random)

Mac-only, no compute. Usage:
    python -m scripts.intervention.transfer_decomposition_report
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.round_paths import (
    FORWARD_ABLATION_DIR, FORWARD_DELTAS_DIR, FORWARD_DELTAS_RANDOM_DIR,
)

TD_DIR = Path(str(FORWARD_ABLATION_DIR)).parent / "transfer_decomposition"
M_LABELS = ["1", "2", "5", "10", "20", "50", "|U|"]
GC_TOLERANCE = 0.99


def _pool_arm(tag, arm):
    """Row-pool one arm ('imp_row'|'imp_ds'|'uni_ds') across all pairs of a tag."""
    root = TD_DIR / tag
    if not root.exists():
        return None, 0, 0
    cols = {lab: [] for lab in M_LABELS}
    ds_med = {lab: [] for lab in M_LABELS}
    nflip, npool, nds = 0, 0, 0
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if f"{arm}__1" not in z:
                continue
            for lab in M_LABELS:
                arr = z[f"{arm}__{lab}"]
                cols[lab].append(arr)
                ds_med[lab].append(float(np.median(arr)))
            if "n_sign_flip" in z:
                nflip += int(z["n_sign_flip"]); npool += int(z["n_pool_signed"])
            nds += 1
    if nds == 0:
        return None, None, 0, 0
    return ({lab: np.concatenate(v) for lab, v in cols.items()},
            {lab: np.asarray(v) for lab, v in ds_med.items()}, nds, (nflip, npool))


def _greedy_disk(root):
    gc, k = [], []
    if not root.exists():
        return None, None
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "optimal_k" not in z:
                continue
            kk = z["optimal_k"].astype(int); act = kk > 0
            if act.sum() == 0:
                continue
            gc.append(z["gap_closed"][act]); k.append(kk[act])
    if not gc:
        return None, None
    return np.concatenate(gc), np.concatenate(k)


def _row(name, cols):
    # median -- per-row unclipped gc is heavy-tailed, the mean is outlier-dominated
    unc = "  ".join(f"{np.median(cols[lab]):>6.2f}" for lab in M_LABELS)
    return f"{name:34} {unc}"


def _fracW(name, cols):
    f = "  ".join(f"{(cols[lab] < 0).mean():>6.2f}" for lab in M_LABELS)
    return f"{name:34} {f}"


def _clamped(name, cols):
    cl = "  ".join(f"{np.clip(cols[lab],0,1).mean():>6.2f}" for lab in M_LABELS)
    return f"{name:34} {cl}"


def main():
    argparse.ArgumentParser(description="Aggregate transfer decomposition").parse_args()
    hdr = "  ".join(f"{lab:>6}" for lab in M_LABELS)

    cells = [
        ("T2  ridge  importance,row,none",     "ridge",       "imp_row"),
        ("T3=M1 ridge  importance,ds,none",    "ridge",       "imp_ds"),
        ("T4  ridge  uniform,ds,none",         "ridge",       "uni_ds"),
        ("T5  ridge  imp,ds,none  RANDOM-dict","ridge_random","imp_ds"),
        ("M2  shuffled  importance,ds,none",   "shuffled",    "imp_ds"),
        ("M4  random_orth  importance,ds,none","random_orth", "imp_ds"),
    ]
    pooled = []
    for name, tag, arm in cells:
        cols, ds_med, nds, flip = _pool_arm(tag, arm)
        pooled.append((name, cols, ds_med, nds, flip))

    print(f"\n=== TRANSFER DECOMPOSITION  (UNCLIPPED gc toward strong, MEDIAN) ===")
    print(f"MEDIAN -- pooled rows (row-weighted)")
    print(f"{'cell':34} {hdr}")
    print("-" * (34 + len(hdr) + 2))
    for name, cols, ds_med, nds, flip in pooled:
        if cols is None:
            print(f"{name:34} (not on disk yet)")
        else:
            print(_row(name, cols) + f"   [{nds} ds]")
    print(f"\nMEDIAN -- per-dataset (median of dataset medians, equal dataset weight)")
    for name, cols, ds_med, nds, flip in pooled:
        if ds_med is not None:
            print(f"{name:34} " + "  ".join(f"{np.median(ds_med[lab]):>6.2f}" for lab in M_LABELS))
    print(f"\nfrac of rows moved TOWARD W (gc<0)")
    for name, cols, ds_med, nds, flip in pooled:
        if cols is not None:
            print(_fracW(name, cols))
    print(f"\nclamped gap_closed (paper convention)")
    for name, cols, ds_med, nds, flip in pooled:
        if cols is not None:
            print(_clamped(name, cols))
    print(f"\nper-dataset sign flips (how often the natural map sign was wrong)")
    for name, cols, ds_med, nds, flip in pooled:
        if cols is not None and flip and flip[1]:
            print(f"{name:34} {flip[0]}/{flip[1]} = {flip[0]/flip[1]:.2f}")

    for label, root in [("T1 greedy ridge (forward_deltas)", FORWARD_DELTAS_DIR),
                        ("T6 greedy ridge RANDOM-dict", FORWARD_DELTAS_RANDOM_DIR)]:
        gc, k = _greedy_disk(root)
        if gc is not None:
            print(f"\n{label}: clamped gc={gc.mean():.3f}  mean k={k.mean():.2f}  "
                  f"median k={np.median(k):.0f}  (n={len(gc)})")


if __name__ == "__main__":
    main()
