#!/usr/bin/env python3
"""Among catalogued strong_wins rows, how many are weak-wrong vs weak-right-but-less-confident?

Most paper analysis (ablation/transfer) restricts to rows where the strong
model beat the weak model on P(true class). That "strong wins" mask includes
two qualitatively different cases for classification tasks:

    (A) weak is "right" — weak would predict the true class at the paper's
        decision threshold; the gap is a confidence gap only.
    (B) weak is "wrong" — weak would predict a different class than the
        true label; strong is actually correct where weak fails.

Thresholding conventions:
    - Binary: the paper's scatter figures draw a reference line at the
      per-dataset event rate P(y=1), so we report two decision rules:
         * event-rate:   predict y=1 iff P(y=1) > event_rate
         * argmax@0.5:   predict y=1 iff P(y=1) > 0.5 (argmax on the
                         2-column probability matrix)
      These disagree on imbalanced datasets.
    - Multi-class: argmax over C classes (no single threshold).

Regression tasks are skipped (no notion of "wrong" vs "less confident").

Usage:
    python -m scripts.analysis.weak_wrong_vs_less_confident
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

SWEEP_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"


def classify_task(preds_strong: np.ndarray) -> str:
    if preds_strong.ndim == 1 or (preds_strong.ndim == 2 and preds_strong.shape[1] == 1):
        return "regression"
    if preds_strong.shape[1] == 2:
        return "binary"
    return "multiclass"


def _weak_wrong_binary(preds_weak: np.ndarray, y: np.ndarray, rule: str) -> np.ndarray:
    """Return a boolean mask of rows where weak would mis-classify under `rule`."""
    p1 = preds_weak[:, 1]
    if rule == "argmax":
        thresh = 0.5
    elif rule == "event_rate":
        thresh = float(y.mean())
    else:
        raise ValueError(rule)
    pred = (p1 > thresh).astype(int)
    return pred != y


def main() -> None:
    binary_totals = {
        "argmax":     {"catalogued": 0, "weak_wrong": 0, "all_rows": 0, "all_wrong": 0},
        "event_rate": {"catalogued": 0, "weak_wrong": 0, "all_rows": 0, "all_wrong": 0},
    }
    binary_datasets = 0
    mc_totals = {"catalogued": 0, "weak_wrong": 0, "all_rows": 0, "all_wrong": 0,
                 "datasets": 0}

    for pair_dir in sorted(p for p in SWEEP_DIR.iterdir() if p.is_dir()):
        for npz_path in sorted(pair_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            if "strong_wins" not in d:
                continue
            task = classify_task(d["preds_strong"])
            if task == "regression":
                continue

            y = d["y_query"].astype(int)
            preds_weak = np.asarray(d["preds_weak"])
            sw = d["strong_wins"].astype(bool)

            if task == "binary":
                binary_datasets += 1
                for rule, bucket in binary_totals.items():
                    wrong = _weak_wrong_binary(preds_weak, y, rule)
                    bucket["catalogued"] += int(sw.sum())
                    bucket["weak_wrong"] += int((sw & wrong).sum())
                    bucket["all_rows"]   += len(y)
                    bucket["all_wrong"]  += int(wrong.sum())
            else:  # multiclass
                wrong = preds_weak.argmax(axis=1) != y
                mc_totals["catalogued"] += int(sw.sum())
                mc_totals["weak_wrong"] += int((sw & wrong).sum())
                mc_totals["all_rows"]   += len(y)
                mc_totals["all_wrong"]  += int(wrong.sum())
                mc_totals["datasets"]   += 1

    print("BINARY — catalogued (strong_wins) rows")
    print(f"  (pair, dataset) cases: {binary_datasets}")
    print(f"  {'rule':<12}{'catalogued':>12}{'weak wrong':>12}{'% wrong':>10}")
    for rule in ("event_rate", "argmax"):
        t = binary_totals[rule]
        pct = 100.0 * t["weak_wrong"] / t["catalogued"] if t["catalogued"] else 0.0
        print(f"  {rule:<12}{t['catalogued']:>12}{t['weak_wrong']:>12}{pct:>9.1f}%")

    print()
    print("MULTI-CLASS — catalogued (strong_wins) rows (argmax over C classes)")
    print(f"  (pair, dataset) cases: {mc_totals['datasets']}")
    pct = 100.0 * mc_totals["weak_wrong"] / mc_totals["catalogued"] if mc_totals["catalogued"] else 0.0
    print(f"  catalogued={mc_totals['catalogued']}  weak_wrong={mc_totals['weak_wrong']}  %={pct:.1f}")

    print()
    print("For reference — all query rows (ignoring strong_wins):")
    print(f"BINARY:")
    for rule in ("event_rate", "argmax"):
        a = binary_totals[rule]
        pct = 100.0 * a["all_wrong"] / a["all_rows"] if a["all_rows"] else 0.0
        print(f"  {rule:<12}rows={a['all_rows']}  wrong={a['all_wrong']}  %={pct:.1f}")
    pct = 100.0 * mc_totals["all_wrong"] / mc_totals["all_rows"] if mc_totals["all_rows"] else 0.0
    print(f"MULTICLASS (argmax): rows={mc_totals['all_rows']}  wrong={mc_totals['all_wrong']}  %={pct:.1f}")


if __name__ == "__main__":
    main()
