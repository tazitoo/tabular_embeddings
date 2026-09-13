#!/usr/bin/env python3
"""Do the forward-delta cells carry the same baselines as the round's prediction cache?

Every forward npz records preds_strong / preds_weak for its query rows; the cache
(cache_baseline_predictions.py) records pred_probs for the same rows from a standalone
fit on the pool's hardware class with the default CUDA allocator. A cell whose baselines
differ from the cache was produced under a different numeric path (other GPU class,
other allocator, other env), and its deltas are not comparable with the ablation and
importance stages that were checked against the cache.

Per pair: cells checked, cells whose strong AND weak baselines match the cache exactly,
and the max |p| difference over the rest.

Usage:
    python -m scripts.rebuttal.check_forward_baselines
    python -m scripts.rebuttal.check_forward_baselines --deltas output/round11/forward_deltas_random
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.round_paths import BASELINE_PREDICTIONS_DIR, FORWARD_DELTAS_DIR


def cached_probs(cache_dir: Path, model: str, dataset: str, rows: np.ndarray):
    f = cache_dir / model / f"{dataset}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    idx = {int(r): i for i, r in enumerate(z["row_indices"])}
    try:
        return np.asarray(z["pred_probs"])[[idx[int(r)] for r in rows]]
    except KeyError:
        return None


def check_cell(f: Path, cache_dir: Path):
    z = np.load(f, allow_pickle=True)
    rows = z["row_indices"]
    out = {}
    for role in ("strong", "weak"):
        model = str(z[f"{role}_model"])
        mine = np.asarray(z[f"preds_{role}"])
        ref = cached_probs(cache_dir, model, f.stem, rows)
        if ref is None or ref.shape != mine.shape:
            out[role] = (model, None)
            continue
        out[role] = (model, float(np.max(np.abs(mine - ref))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=Path, default=FORWARD_DELTAS_DIR)
    ap.add_argument("--cache", type=Path, default=BASELINE_PREDICTIONS_DIR)
    ap.add_argument("--show", type=int, default=3, help="differing cells to list per pair")
    args = ap.parse_args()

    print(f"deltas = {args.deltas}\ncache  = {args.cache}\n")
    print(f"{'pair':24s} {'cells':>5s} {'exact':>5s} {'uncached':>8s}  max|dp| strong / weak   (differing cells)")
    tot = exact_tot = 0
    for pair_dir in sorted(p for p in args.deltas.iterdir() if p.is_dir()):
        cells = sorted(pair_dir.glob("*.npz"))
        exact, uncached, differing = 0, 0, []
        max_s = max_w = 0.0
        for f in cells:
            r = check_cell(f, args.cache)
            ds, dw = r["strong"][1], r["weak"][1]
            if ds is None or dw is None:
                uncached += 1
                continue
            if ds == 0.0 and dw == 0.0:
                exact += 1
            else:
                differing.append((f.stem, r["strong"][0], ds, r["weak"][0], dw))
                max_s, max_w = max(max_s, ds), max(max_w, dw)
        tot += len(cells) - uncached
        exact_tot += exact
        print(f"{pair_dir.name:24s} {len(cells):5d} {exact:5d} {uncached:8d}  {max_s:.2e} / {max_w:.2e}")
        for ds, sm, dsv, wm, dwv in differing[: args.show]:
            print(f"    {ds}: {sm} {dsv:.2e}, {wm} {dwv:.2e}")
    print(f"\n{exact_tot}/{tot} cached cells carry bit-identical baselines")


if __name__ == "__main__":
    main()
