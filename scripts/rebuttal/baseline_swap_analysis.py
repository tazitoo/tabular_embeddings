#!/usr/bin/env python3
"""REBUTTAL triage: how much does the baseline-prediction drift between
perrow_importance (trained arm) and perrow_importance_random (random arm)
actually change the DECISION, per model?

The raw probability maxdiff can be large without changing anything material.
The decision-level measure is the "swapset": rows whose predicted class flips
(argmax changes) between the two runs. For regression there is no class, so we
report rows whose prediction moves by more than 1% of that dataset's prediction
std as a material-change proxy. Same query rows in both arms (row_indices match),
so this is a row-for-row comparison.

Usage:
    python -m scripts.rebuttal.baseline_swap_analysis
"""
import glob
import os

import numpy as np

MODELS = ["carte", "mitra", "tabdpt", "tabpfn", "tabicl", "tabicl_v2"]


def main():
    print(f"{'model':11}{'cls_rows':>9}{'cls_swaps':>10}{'swap%':>8}   "
          f"{'reg_rows':>9}{'reg_material':>13}{'mat%':>7}   {'ds_w/swap':>10}")
    for m in MODELS:
        da, db = f"output/perrow_importance/{m}", f"output/perrow_importance_random/{m}"
        if not (os.path.isdir(da) and os.path.isdir(db)):
            print(f"  {m}: dir missing"); continue
        common = sorted(set(os.path.basename(p) for p in glob.glob(f"{da}/*.npz")) &
                        set(os.path.basename(p) for p in glob.glob(f"{db}/*.npz")))
        cls_rows = cls_sw = reg_rows = reg_mat = ds_sw = 0
        for ds in common:
            a = np.load(f"{da}/{ds}", allow_pickle=True)
            b = np.load(f"{db}/{ds}", allow_pickle=True)
            pa = np.asarray(a["baseline_preds"], float)
            pb = np.asarray(b["baseline_preds"], float)
            if pa.shape != pb.shape:
                continue
            if pa.ndim == 2 and pa.shape[1] > 1:                 # classification
                sw = np.argmax(pa, 1) != np.argmax(pb, 1)
                cls_rows += len(sw); cls_sw += int(sw.sum())
                if sw.any():
                    ds_sw += 1
            else:                                               # regression
                p, q = pa.reshape(-1), pb.reshape(-1)
                std = p.std() + 1e-9
                mat = np.abs(p - q) > 0.01 * std
                reg_rows += len(p); reg_mat += int(mat.sum())
        swpct = 100 * cls_sw / max(cls_rows, 1)
        matpct = 100 * reg_mat / max(reg_rows, 1)
        print(f"  {m:9}{cls_rows:>9}{cls_sw:>10}{swpct:>7.2f}%   "
              f"{reg_rows:>9}{reg_mat:>13}{matpct:>6.1f}%   {ds_sw:>10}")


if __name__ == "__main__":
    main()
