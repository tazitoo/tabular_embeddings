#!/usr/bin/env python3
"""REBUTTAL: full-test-set scalars combining forward + reverse interventions.

The paper reports interventions only on the below-diagonal rows (the row set
where the overall-stronger model wins per-row). The reverse sweep
(scripts/rebuttal/{ablation,transfer}_sweep_symmetric.py) fills in the
above-diagonal rows (where the overall-weaker model wins). Together they cover
the full test set.

This script pairs each forward NPZ (output/transfer_sweep_v2,
output/ablation_sweep_tols) with its reverse counterpart
(output/rebuttal/symmetric_{transfer,ablation}) and emits the scalars the
rebuttal needs, pooled at the ROW level across all pairs/datasets:

  - below-diagonal gap closed (the published number)
  - above-diagonal gap closed (new; the previously-untouched rows)
  - full-test-set gap closed (both triangles) -> answers dVDs' selection-bias
  - reverse TRANSFER gap closed = weak->strong -> answers SD9t
  - mean concepts/row (optimal_k), below vs above
  - coverage: #rows below / above / near-diagonal(untouched) / total, so the
    fraction of the test set now covered is explicit (no hidden truncation)

Row masks: `strong_wins` marks the intervened rows in each file. Forward's mask
(below-diagonal) and reverse's mask (above-diagonal) are disjoint by
construction; near-diagonal / tied rows are in neither (the method injects 0
concepts there -> selectivity).

Deterministic: no CLI args needed; defaults are the canonical paths.

Usage:
    python -m scripts.rebuttal.summarize_symmetric
    python -m scripts.rebuttal.summarize_symmetric --json out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

# kind -> (forward canonical dir, reverse rebuttal dir)
# (forward below-diagonal dir, reverse above-diagonal dir) per intervention kind.
DIRS_TRAINED = {
    "transfer": (
        PROJECT_ROOT / "output" / "transfer_sweep_v2",
        PROJECT_ROOT / "output" / "rebuttal" / "symmetric_transfer",
    ),
    "ablation": (
        PROJECT_ROOT / "output" / "ablation_sweep_tols",
        PROJECT_ROOT / "output" / "rebuttal" / "symmetric_ablation",
    ),
}
# Random-baseline control: same structure, random-SAE forward + reverse arms.
DIRS_RANDOM = {
    "transfer": (
        PROJECT_ROOT / "output" / "transfer_random",
        PROJECT_ROOT / "output" / "rebuttal" / "symmetric_transfer_random",
    ),
    "ablation": (
        PROJECT_ROOT / "output" / "ablation_sweep_random_tols",
        PROJECT_ROOT / "output" / "rebuttal" / "symmetric_ablation_random",
    ),
}
DIRS = DIRS_TRAINED  # set per --mode in main()


def _wins_stats(npz):
    """Row-level (gap_closed, optimal_k) on the intervened (strong_wins) rows.

    Returns (gc_rows, k_rows, n_query, n_wins). gc_rows drops NaNs (rows where
    gap_closed is undefined), mirroring the paper's np.nanmean convention.
    """
    gap_closed = np.asarray(npz["gap_closed"], dtype=np.float64)
    optimal_k = np.asarray(npz["optimal_k"], dtype=np.float64)
    wins = np.asarray(npz["strong_wins"], dtype=bool)
    n_query = int(npz["n_query"])
    gc = gap_closed[wins]
    k = optimal_k[wins]
    finite = ~np.isnan(gc)
    return gc[finite], k[finite], n_query, int(wins.sum())


def summarize_kind(kind: str) -> dict:
    fwd_dir, rev_dir = DIRS[kind]

    # Row-level pools
    below_gc, below_k = [], []      # forward / below-diagonal
    above_gc, above_k = [], []      # reverse / above-diagonal
    n_total = n_below = n_above = 0
    pairs_matched = set()
    n_ds_matched = 0
    n_ds_fwd_only = 0

    for fwd_pair_dir in sorted(fwd_dir.glob("*_vs_*")):
        pair = fwd_pair_dir.name
        rev_pair_dir = rev_dir / pair
        for fwd_file in sorted(fwd_pair_dir.glob("*.npz")):
            ds = fwd_file.name
            fwd = np.load(fwd_file, allow_pickle=True)
            # Files that skipped (degenerate / tied / no wins) lack per-row arrays
            if "gap_closed" not in fwd.files:
                continue
            gcb, kb, nq, nwb = _wins_stats(fwd)
            n_total += nq
            n_below += nwb
            below_gc.append(gcb); below_k.append(kb)

            rev_file = rev_pair_dir / ds
            if rev_file.exists():
                rev = np.load(rev_file, allow_pickle=True)
                if "gap_closed" in rev.files:
                    gca, ka, _, nwa = _wins_stats(rev)
                    n_above += nwa
                    above_gc.append(gca); above_k.append(ka)
                    pairs_matched.add(pair)
                    n_ds_matched += 1
            else:
                n_ds_fwd_only += 1

    below_gc = np.concatenate(below_gc) if below_gc else np.array([])
    above_gc = np.concatenate(above_gc) if above_gc else np.array([])
    below_k = np.concatenate(below_k) if below_k else np.array([])
    above_k = np.concatenate(above_k) if above_k else np.array([])
    full_gc = np.concatenate([below_gc, above_gc]) if (len(below_gc) or len(above_gc)) else np.array([])

    def m(a):
        return float(a.mean()) if len(a) else None

    return {
        "kind": kind,
        # gap closed, pooled over rows
        "below_diagonal_gap_closed": m(below_gc),   # published number
        "above_diagonal_gap_closed": m(above_gc),   # new rows
        "full_test_gap_closed": m(full_gc),         # both triangles
        # concepts per row
        "below_diagonal_mean_k": m(below_k),
        "above_diagonal_mean_k": m(above_k),
        # coverage (transparency: every count visible)
        "n_rows_total": n_total,
        "n_rows_below": n_below,
        "n_rows_above": n_above,
        "n_rows_near_diagonal_untouched": n_total - n_below - n_above,
        "frac_test_covered_published": (n_below / n_total) if n_total else None,
        "frac_test_covered_symmetric": ((n_below + n_above) / n_total) if n_total else None,
        # data completeness
        "n_pairs_matched": len(pairs_matched),
        "n_datasets_matched": n_ds_matched,
        "n_datasets_forward_only_missing_reverse": n_ds_fwd_only,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trained", "random"], default="trained",
                    help="trained (default) or the random-baseline control arms")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    global DIRS
    DIRS = DIRS_RANDOM if args.mode == "random" else DIRS_TRAINED
    if args.json is None:
        name = "symmetric_summary.json" if args.mode == "trained" else "symmetric_summary_random.json"
        args.json = PROJECT_ROOT / "output" / "rebuttal" / name

    out = {k: summarize_kind(k) for k in DIRS}

    # transfer above-diagonal == weak->strong (the SD9t answer); surface it plainly
    if out["transfer"]["above_diagonal_gap_closed"] is not None:
        out["weak_to_strong_transfer_gap_closed"] = out["transfer"]["above_diagonal_gap_closed"]

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))

    # Human-readable table
    print(f"\n{'='*70}\nSYMMETRIC FULL-TEST-SET SUMMARY\n{'='*70}")
    for kind in DIRS:
        s = out[kind]
        print(f"\n[{kind.upper()}]  (pairs matched: {s['n_pairs_matched']}/15, "
              f"datasets matched: {s['n_datasets_matched']}, "
              f"fwd-only missing reverse: {s['n_datasets_forward_only_missing_reverse']})")
        print(f"  gap closed  below-diag (published): {s['below_diagonal_gap_closed']}")
        print(f"  gap closed  above-diag (new)      : {s['above_diagonal_gap_closed']}")
        print(f"  gap closed  FULL TEST (both)      : {s['full_test_gap_closed']}")
        print(f"  mean concepts/row  below | above  : {s['below_diagonal_mean_k']} | {s['above_diagonal_mean_k']}")
        print(f"  coverage rows  below={s['n_rows_below']}  above={s['n_rows_above']}  "
              f"near-diag(untouched)={s['n_rows_near_diagonal_untouched']}  total={s['n_rows_total']}")
        print(f"  test covered:  published={s['frac_test_covered_published']:.3f}  "
              f"symmetric={s['frac_test_covered_symmetric']:.3f}"
              if s['frac_test_covered_published'] is not None else "")
    if "weak_to_strong_transfer_gap_closed" in out:
        print(f"\n>> weak->strong transfer gap closed (SD9t): "
              f"{out['weak_to_strong_transfer_gap_closed']}")
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
