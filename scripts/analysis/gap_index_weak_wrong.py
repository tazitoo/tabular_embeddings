#!/usr/bin/env python3
"""Gap-index top features per weak model, all catalogued rows vs weak-wrong subset.

Mirrors scripts/concepts/weak_model_gap_index.py but (a) works off the
current ablation_sweep_tols data, (b) prints two terminal tables instead
of writing JSON/LaTeX, and (c) adds the weak-wrong filter (event-rate
threshold for binary, argmax for multiclass — same convention as the
paper scatter figures).

Classification only. Cell format mirrors the TeX gap_index:
    <Strong>  pct%  f<idx>
where pct = count / (rows weak lost to that strong model) × 100.

Usage:
    python -m scripts.analysis.gap_index_weak_wrong
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"

MODEL_NORM = {
    "carte": "CARTE", "mitra": "Mitra", "tabdpt": "TabDPT",
    "tabicl": "TabICL", "tabicl_v2": "TabICL-v2", "tabpfn": "TabPFN",
    "hyperfast": "HyperFast",
}
WEAK_ORDER = ["CARTE", "Mitra", "TabDPT", "TabICL", "TabICL-v2", "TabPFN"]
N_RANKS = 5


def _weak_wrong(preds_weak: np.ndarray, y: np.ndarray) -> np.ndarray:
    if preds_weak.shape[1] == 2:
        p1 = preds_weak[:, 1]
        thresh = float(y.mean())
        return ((p1 > thresh).astype(int) != y)
    return preds_weak.argmax(axis=1) != y


def _is_regression(preds_strong: np.ndarray) -> bool:
    return preds_strong.ndim == 1 or (preds_strong.ndim == 2 and preds_strong.shape[1] == 1)


def _new_bucket():
    return {
        "losses_by_strong": defaultdict(int),
        "total_losses": 0,
        "tuple_counts": defaultdict(int),
    }


def aggregate(task_types: dict[str, str]):
    """Return gaps_all, gaps_ww — keyed by weak model."""
    gaps_all: dict = defaultdict(_new_bucket)
    gaps_ww:  dict = defaultdict(_new_bucket)

    for pair_dir in sorted(p for p in ABLATION_DIR.iterdir() if p.is_dir()):
        for npz_path in sorted(pair_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            if "selected_features" not in d or _is_regression(d["preds_strong"]):
                continue
            ds_name = npz_path.stem
            if task_types.get(ds_name) != "classification":
                continue

            sm = MODEL_NORM.get(str(d["strong_model"]), str(d["strong_model"]))
            wm = MODEL_NORM.get(str(d["weak_model"]), str(d["weak_model"]))

            y = d["y_query"].astype(int)
            sw = d["strong_wins"].astype(bool)
            ok = d["optimal_k"]
            sel = d["selected_features"]
            preds_weak = np.asarray(d["preds_weak"])

            catalogued = sw & (ok > 0)
            ww_mask    = catalogued & _weak_wrong(preds_weak, y)

            for mask, bucket_map in ((catalogued, gaps_all), (ww_mask, gaps_ww)):
                bucket = bucket_map[wm]
                for r in np.flatnonzero(mask):
                    bucket["total_losses"] += 1
                    bucket["losses_by_strong"][sm] += 1
                    seen: set[int] = set()
                    k = int(ok[r])
                    for j in range(k):
                        fi = int(sel[r, j])
                        if fi < 0:
                            break
                        if fi in seen:
                            continue
                        seen.add(fi)
                        bucket["tuple_counts"][(sm, fi)] += 1

    return gaps_all, gaps_ww


def top_rows_for_weak(bucket: dict, n: int, *, sort_by: str) -> list[tuple[str, int, float, int]]:
    """Return [(strong, feat, pct_S_losses, count)] top-n sorted by sort_by."""
    rows = []
    for (sm, fi), count in bucket["tuple_counts"].items():
        sl = bucket["losses_by_strong"][sm]
        pct = (count / sl * 100) if sl else 0.0
        rows.append((sm, fi, pct, count))
    if sort_by == "count":
        rows.sort(key=lambda r: (-r[3], -r[2]))
    else:
        rows.sort(key=lambda r: (-r[2], -r[3]))
    return rows[:n]


def print_table(label: str, gaps: dict, *, sort_by: str) -> None:
    print(f"\n{'='*104}\n{label}  [sorted by {sort_by}]\n{'='*104}")
    col_w = 20
    header = f"{'Weak':<12}" + "".join(f"{'#'+str(i+1):<{col_w}}" for i in range(N_RANKS))
    print(header)
    print("-" * len(header))
    for weak in WEAK_ORDER:
        bucket = gaps.get(weak)
        if not bucket or bucket["total_losses"] == 0:
            print(f"{weak:<12}(no rows)")
            continue
        tot = bucket["total_losses"]
        line = f"{weak:<12}"
        cells = top_rows_for_weak(bucket, N_RANKS, sort_by=sort_by)
        for sm, fi, pct, count in cells:
            cell = f"{sm}^{int(round(pct))}_f{fi}(n={count})"
            line += f"{cell:<{col_w}}"
        print(line + f"   [n={tot}]")


def main() -> None:
    task_types = json.loads(SPLITS_PATH.read_text())
    task_types = {k: v["task_type"] for k, v in task_types.items()}

    gaps_all, gaps_ww = aggregate(task_types)

    # Match the paper's sort-by-count convention so columns line up with
    # Table~\ref{tab:gap_index_classification}.
    print_table("gap_index_classification  —  ALL catalogued strong_wins rows",
                gaps_all, sort_by="count")
    print_table("gap_index_classification  —  WEAK-WRONG subset "
                "(binary: P>event_rate; multi: argmax)",
                gaps_ww, sort_by="count")


if __name__ == "__main__":
    main()
