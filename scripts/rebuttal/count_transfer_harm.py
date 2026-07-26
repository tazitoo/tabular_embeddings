#!/usr/bin/env python3
"""REBUTTAL (dVDs): does concept transfer ever move the recipient to a WORSE
prediction? Counts, over every intervened row of the full-test transfer
(below-diagonal forward_deltas + above-diagonal symmetric_transfer), the rows
whose intervened loss on y_true exceeds the recipient's baseline loss.

The acceptance criterion only injects concepts that move the prediction toward
the donor and rejects overshoot, so harm should be rare; this quantifies it.

Loss: cross-entropy -log p[y] for classification (2-D preds), squared error for
regression. A row is "harmed" if intervened_loss > baseline_loss; "materially
harmed" if the increase exceeds 0.01. Only rows actually intervened
(optimal_k > 0) are counted.

Usage:
    python -m scripts.rebuttal.count_transfer_harm
"""
import glob
import numpy as np

from scripts._project_root import PROJECT_ROOT

EPS = 1e-7
DIRS = {
    "below-diag (forward_deltas)": PROJECT_ROOT / "output/rebuttal/forward_deltas",
    "above-diag (symmetric_transfer)": PROJECT_ROOT / "output/rebuttal/symmetric_transfer",
}


def _row_loss(preds, y):
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] > 1:      # classification
        return -np.log(np.clip(preds[np.arange(len(y)), y.astype(int)], EPS, 1 - EPS))
    p = preds.reshape(len(y))                        # regression
    return (p - y.astype(float)) ** 2


def main():
    tot_int = tot_harm = tot_harm_mat = 0
    dloss_harm = []
    dloss_all = []
    n_files = 0
    for label, d in DIRS.items():
        di = ti = th = tm = 0
        for f in sorted(glob.glob(str(d / "*/*.npz"))):
            npz = np.load(f, allow_pickle=True)
            if not {"preds_weak", "preds_intervened", "optimal_k", "y_query"} <= set(npz.files):
                continue
            k = np.asarray(npz["optimal_k"])
            rows = np.where(k > 0)[0]
            if len(rows) == 0:
                continue
            y = np.asarray(npz["y_query"])
            base = _row_loss(npz["preds_weak"], y)
            inter = _row_loss(npz["preds_intervened"], y)
            dl = inter[rows] - base[rows]            # >0 => worse (harm)
            di += 1
            ti += len(rows)
            harmed = dl > 1e-6
            th += int(harmed.sum())
            tm += int((dl > 0.01).sum())
            dloss_harm.extend(dl[harmed].tolist())
            dloss_all.extend(dl.tolist())
        n_files += di
        tot_int += ti; tot_harm += th; tot_harm_mat += tm
        print(f"  {label:34s} intervened_rows={ti:6d}  harmed={th:5d} "
              f"({100*th/max(ti,1):4.1f}%)  materially(>0.01)={tm:4d} ({100*tm/max(ti,1):4.1f}%)")

    print(f"\n  FULL TEST SET ({n_files} datasets)")
    print(f"    intervened rows            : {tot_int}")
    print(f"    moved to worse loss (any)  : {tot_harm}  ({100*tot_harm/max(tot_int,1):.2f}%)")
    print(f"    materially worse (>0.01)   : {tot_harm_mat}  ({100*tot_harm_mat/max(tot_int,1):.2f}%)")
    if dloss_all:
        dl = np.asarray(dloss_all)
        # scale-invariant summary (raw Δloss magnitude is dominated by unnormalized
        # regression targets, so report the sign distribution instead of the mean)
        print(f"    improved (Δloss<0)         : {100*np.mean(dl < -1e-6):.2f}% of intervened rows")
        print(f"    unchanged (|Δloss|<=1e-6)  : {100*np.mean(np.abs(dl) <= 1e-6):.2f}%")
    if dloss_harm:
        print(f"    median Δloss on harmed rows: {np.median(dloss_harm):+.4f}")


if __name__ == "__main__":
    main()
