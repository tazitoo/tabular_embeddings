#!/usr/bin/env python3
"""Do catalogued strong_wins rows behave differently in ablation/transfer when weak is actually wrong?

For each pair, splits strong_wins rows into:
    - all catalogued  (strong_wins & optimal_k > 0)
    - weak-wrong      (the subset where weak's predicted label != y_true,
                       using event-rate threshold for binary, argmax for
                       multi-class — matches the paper's scatter convention)

Reports, per (pair, direction):
    - n_rows, mean optimal_k
    - top-1 "lead" feature  — selected_features[:, 0], greedy first pick
    - top-1 "any-position"  — most frequent feature across selected_features[i, :optimal_k[i]]

Regression datasets are skipped (no notion of weak-wrong).

Usage:
    python -m scripts.analysis.weak_wrong_intervention_stats
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

ABLATION_DIR        = PROJECT_ROOT / "output" / "ablation_sweep_tols"
TRANSFER_DIR        = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"
ABLATION_RANDOM_DIR = PROJECT_ROOT / "output" / "ablation_sweep_random_tols"
TRANSFER_RANDOM_DIR = PROJECT_ROOT / "output" / "transfer_random"


def _is_regression(preds_strong: np.ndarray) -> bool:
    return preds_strong.ndim == 1 or (preds_strong.ndim == 2 and preds_strong.shape[1] == 1)


def _weak_wrong(preds_weak: np.ndarray, y: np.ndarray) -> np.ndarray:
    if preds_weak.shape[1] == 2:
        p1 = preds_weak[:, 1]
        thresh = float(y.mean())
        return ((p1 > thresh).astype(int) != y)
    return preds_weak.argmax(axis=1) != y


def _feature_counters(selected: np.ndarray, ok: np.ndarray, mask: np.ndarray):
    """Return (lead_counter, any_counter) over rows where mask is True."""
    lead, any_pos = Counter(), Counter()
    idx = np.flatnonzero(mask & (ok > 0))
    for i in idx:
        feats = selected[i, : int(ok[i])]
        if len(feats) == 0:
            continue
        lead[int(feats[0])] += 1
        for f in feats:
            any_pos[int(f)] += 1
    return lead, any_pos


def _fmt_top(counter: Counter) -> str:
    if not counter:
        return "—"
    f, n = counter.most_common(1)[0]
    return f"f{f} ({n})"


def summarise_direction(label: str, sweep_dir: Path, *, track_features: bool = True) -> None:
    rows = []
    agg_all = {"n": 0, "sum_k": 0, "sum_gc": 0.0, "n_gc": 0}
    agg_ww  = {"n": 0, "sum_k": 0, "sum_gc": 0.0, "n_gc": 0}

    if not sweep_dir.exists():
        print(f"\n[skip] {label}: {sweep_dir} missing")
        return

    for pair_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        pair = pair_dir.name
        lead_all, any_all = Counter(), Counter()
        lead_ww,  any_ww  = Counter(), Counter()
        n_all = n_ww = 0
        sum_k_all = sum_k_ww = 0
        sum_gc_all = sum_gc_ww = 0.0
        n_gc_all = n_gc_ww = 0

        for npz_path in sorted(pair_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            if "strong_wins" not in d or _is_regression(d["preds_strong"]):
                continue
            y  = d["y_query"].astype(int)
            sw = d["strong_wins"].astype(bool)
            ok = d["optimal_k"]
            gc = d["gap_closed"] if "gap_closed" in d else None
            sel = d["selected_features"] if "selected_features" in d else None

            catalogued = sw & (ok > 0)
            ww_mask    = catalogued & _weak_wrong(np.asarray(d["preds_weak"]), y)

            n_all += int(catalogued.sum())
            n_ww  += int(ww_mask.sum())
            sum_k_all += int(ok[catalogued].sum())
            sum_k_ww  += int(ok[ww_mask].sum())

            if gc is not None:
                g_all = gc[catalogued]; g_all = g_all[~np.isnan(g_all)]
                g_ww  = gc[ww_mask];    g_ww  = g_ww [~np.isnan(g_ww)]
                sum_gc_all += float(g_all.sum()); n_gc_all += len(g_all)
                sum_gc_ww  += float(g_ww.sum());  n_gc_ww  += len(g_ww)

            if track_features and sel is not None:
                l_a, a_a = _feature_counters(sel, ok, catalogued)
                l_w, a_w = _feature_counters(sel, ok, ww_mask)
                lead_all.update(l_a); any_all.update(a_a)
                lead_ww.update(l_w);  any_ww.update(a_w)

        if n_all == 0:
            continue
        mean_gc_all = sum_gc_all / n_gc_all if n_gc_all else float("nan")
        mean_gc_ww  = sum_gc_ww  / n_gc_ww  if n_gc_ww  else float("nan")
        rows.append((
            pair, n_all, sum_k_all / n_all, mean_gc_all,
            _fmt_top(lead_all), _fmt_top(any_all),
            n_ww, (sum_k_ww / n_ww) if n_ww else float("nan"), mean_gc_ww,
            _fmt_top(lead_ww), _fmt_top(any_ww),
        ))
        agg_all["n"] += n_all; agg_all["sum_k"] += sum_k_all
        agg_all["sum_gc"] += sum_gc_all; agg_all["n_gc"] += n_gc_all
        agg_ww["n"]  += n_ww;  agg_ww["sum_k"]  += sum_k_ww
        agg_ww["sum_gc"]  += sum_gc_ww;  agg_ww["n_gc"]  += n_gc_ww

    print(f"\n{'='*132}\n{label}  ({sweep_dir.name})\n{'='*132}")
    header = (f"{'pair':<22}│ {'n_all':>6} {'mean_k':>7} {'mean_gc':>8} {'lead_all':>12} {'any_all':>12}"
              f" │ {'n_ww':>6} {'mean_k':>7} {'mean_gc':>8} {'lead_ww':>12} {'any_ww':>12}")
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r[0]:<22}│ {r[1]:>6} {r[2]:>7.2f} {r[3]:>8.3f} {r[4]:>12} {r[5]:>12}"
              f" │ {r[6]:>6} {r[7]:>7.2f} {r[8]:>8.3f} {r[9]:>12} {r[10]:>12}")

    k_all  = agg_all["sum_k"]  / agg_all["n"]    if agg_all["n"]    else float("nan")
    k_ww   = agg_ww["sum_k"]   / agg_ww["n"]     if agg_ww["n"]     else float("nan")
    gc_all = agg_all["sum_gc"] / agg_all["n_gc"] if agg_all["n_gc"] else float("nan")
    gc_ww  = agg_ww["sum_gc"]  / agg_ww["n_gc"]  if agg_ww["n_gc"]  else float("nan")
    print("─" * len(header))
    print(f"{'ALL PAIRS (micro avg)':<22}│ {agg_all['n']:>6} {k_all:>7.2f} {gc_all:>8.3f}"
          f" {'':>12} {'':>12} │ {agg_ww['n']:>6} {k_ww:>7.2f} {gc_ww:>8.3f}")


def main() -> None:
    print("\n######## TRAINED SAEs ########")
    summarise_direction("ABLATION (trained)", ABLATION_DIR)
    summarise_direction("TRANSFER (trained)", TRANSFER_DIR)

    print("\n######## RANDOM SAEs ########")
    summarise_direction("ABLATION (random)", ABLATION_RANDOM_DIR)
    summarise_direction("TRANSFER (random)", TRANSFER_RANDOM_DIR)


if __name__ == "__main__":
    main()
