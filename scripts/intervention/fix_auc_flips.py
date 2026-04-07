#!/usr/bin/env python3
"""Identify and optionally delete ablation/transfer entries where strong/weak
flips when using AUC instead of logloss.

Usage:
    # Dry run (show what would be deleted):
    python -m scripts.intervention.fix_auc_flips

    # Delete flipped entries from ablation:
    python -m scripts.intervention.fix_auc_flips --delete --sweep-dir output/ablation_sweep

    # Delete flipped entries from transfer:
    python -m scripts.intervention.fix_auc_flips --delete --sweep-dir output/transfer_sweep_v2

    # Both:
    python -m scripts.intervention.fix_auc_flips --delete --sweep-dir output/ablation_sweep output/transfer_sweep_v2
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts._project_root import PROJECT_ROOT

EXCLUDE = {"hyperfast", "tabula8b"}


def find_flips(sweep_dir: Path):
    """Find (pair_dir, dataset) entries where AUC flips strong/weak vs logloss."""
    flips = []
    total = 0

    for pair_dir in sorted(sweep_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("_vs_")
        if len(parts) != 2:
            continue
        if parts[0] in EXCLUDE or parts[1] in EXCLUDE:
            continue

        for npz_path in sorted(pair_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue

            if "preds_strong" not in data or "y_query" not in data:
                continue

            y = data["y_query"]
            if len(np.unique(y)) < 2:
                continue

            metric_name = str(data["metric_name"])
            if metric_name in ("neg_rmse", "degenerate"):
                continue

            preds_s = data["preds_strong"]
            preds_w = data["preds_weak"]

            try:
                if preds_s.ndim == 2 and preds_s.shape[1] == 2:
                    auc_s = roc_auc_score(y, preds_s[:, 1])
                    auc_w = roc_auc_score(y, preds_w[:, 1])
                elif preds_s.ndim == 2 and preds_s.shape[1] > 2:
                    auc_s = roc_auc_score(
                        y, preds_s, multi_class="ovr",
                        labels=np.arange(preds_s.shape[1]),
                    )
                    auc_w = roc_auc_score(
                        y, preds_w, multi_class="ovr",
                        labels=np.arange(preds_w.shape[1]),
                    )
                else:
                    continue
            except ValueError:
                continue

            total += 1
            if auc_s < auc_w:
                flips.append(npz_path)

    return flips, total


def main():
    parser = argparse.ArgumentParser(
        description="Find/delete ablation entries where AUC flips strong/weak",
    )
    parser.add_argument(
        "--sweep-dir", type=str, nargs="+",
        default=["output/ablation_sweep", "output/transfer_sweep_v2"],
        help="Sweep directories to check",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete flipped NPZ files (default: dry run)",
    )
    args = parser.parse_args()

    for sd in args.sweep_dir:
        sweep_dir = PROJECT_ROOT / sd
        if not sweep_dir.exists():
            print(f"SKIP {sd}: directory not found")
            continue

        flips, total = find_flips(sweep_dir)
        print(f"\n{sd}: {len(flips)}/{total} classification entries flip ({len(flips)/max(total,1)*100:.1f}%)")

        by_pair = defaultdict(list)
        for p in flips:
            by_pair[p.parent.name].append(p.stem)

        for pair, datasets in sorted(by_pair.items()):
            print(f"  {pair}: {len(datasets)}")
            for ds in datasets:
                print(f"    {ds}")

        if args.delete:
            for p in flips:
                p.unlink()
                print(f"  DELETED {p.relative_to(PROJECT_ROOT)}")
            print(f"Deleted {len(flips)} files from {sd}")
        else:
            print(f"Dry run — use --delete to remove {len(flips)} files")


if __name__ == "__main__":
    main()
