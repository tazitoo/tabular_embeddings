#!/usr/bin/env python3
"""
Extend MNN matches with correlation-thresholded best-correlate matching.

For each unmatched feature, finds its best correlate in each partner model.
Accepts the match only if r exceeds the model's per-model random baseline
null distribution at a given percentile (default: p95).

Reads:
  - MNN matching results (JSON)
  - Random baseline results (JSON)
  - Cross-correlation matrices (NPZ, from --save-correlations)

Produces:
  - Extended matching JSON with MNN + correlation-thresholded matches
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from scripts.sae.analyze_sae_concepts_deep import NumpyEncoder, convert_keys_to_native


def compute_null_thresholds(
    baseline_path: Path, percentile: float = 95.0
) -> Dict[str, float]:
    """Compute per-model r threshold from random baseline MNN matches."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    thresholds = {}
    for pair_key, result in baseline["pairs"].items():
        model = pair_key.replace("__random", "")
        rs = np.array([m["r"] for m in result["matches"]])
        if len(rs) == 0:
            thresholds[model] = 0.0
        else:
            thresholds[model] = float(np.percentile(rs, percentile))
    return thresholds


def extend_pair(
    corr_matrix: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    mnn_matched_a: Set[int],
    mnn_matched_b: Set[int],
    threshold_a: float,
    threshold_b: float,
) -> list:
    """Find additional matches for unmatched features above threshold.

    For each unmatched feature in A, find its best correlate in B (if not
    already MNN-matched on the B side) and accept if r > threshold_a.
    Then do the same in reverse for unmatched B features.

    Returns list of (idx_a, idx_b, r, direction) dicts.
    """
    extended = []
    used_a = set(mnn_matched_a)
    used_b = set(mnn_matched_b)

    # Map from original feature index to correlation matrix position
    pos_a = {int(idx): i for i, idx in enumerate(indices_a)}
    pos_b = {int(idx): i for i, idx in enumerate(indices_b)}

    # A→B: unmatched A features find best B correlate
    candidates = []
    for orig_idx in indices_a:
        orig_idx = int(orig_idx)
        if orig_idx in used_a:
            continue
        i = pos_a[orig_idx]
        row = corr_matrix[i, :]
        best_j = int(row.argmax())
        best_r = float(row[best_j])
        best_b_idx = int(indices_b[best_j])
        if best_r > threshold_a:
            candidates.append((orig_idx, best_b_idx, best_r, "A→B"))

    # Sort by r descending, greedily assign (no duplicate B targets)
    candidates.sort(key=lambda x: x[2], reverse=True)
    for idx_a, idx_b, r, direction in candidates:
        if idx_b not in used_b:
            extended.append({"idx_a": idx_a, "idx_b": idx_b, "r": r, "direction": direction})
            used_a.add(idx_a)
            used_b.add(idx_b)

    # B→A: unmatched B features find best A correlate
    candidates = []
    for orig_idx in indices_b:
        orig_idx = int(orig_idx)
        if orig_idx in used_b:
            continue
        j = pos_b[orig_idx]
        col = corr_matrix[:, j]
        best_i = int(col.argmax())
        best_r = float(col[best_i])
        best_a_idx = int(indices_a[best_i])
        if best_r > threshold_b:
            candidates.append((best_a_idx, orig_idx, best_r, "B→A"))

    candidates.sort(key=lambda x: x[2], reverse=True)
    for idx_a, idx_b, r, direction in candidates:
        if idx_a not in used_a:
            extended.append({"idx_a": idx_a, "idx_b": idx_b, "r": r, "direction": direction})
            used_a.add(idx_a)
            used_b.add(idx_b)

    return extended


def main():
    parser = argparse.ArgumentParser(description="Extend MNN matches with thresholded correlations")
    parser.add_argument(
        "--mnn-path", type=str,
        default="output/sae_feature_matching_mnn_t0.001_n500.json",
        help="Path to MNN matching results",
    )
    parser.add_argument(
        "--baseline-path", type=str,
        default="output/sae_feature_matching_mnn_t0.001_n500_random_baseline.json",
        help="Path to random baseline results",
    )
    parser.add_argument(
        "--corr-dir", type=str,
        default="output/sae_cross_correlations",
        help="Directory with cross-correlation NPZ files",
    )
    parser.add_argument(
        "--percentile", type=float, default=95.0,
        help="Null distribution percentile for threshold (default: 95)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path",
    )
    args = parser.parse_args()

    mnn_path = PROJECT_ROOT / args.mnn_path
    baseline_path = PROJECT_ROOT / args.baseline_path
    corr_dir = PROJECT_ROOT / args.corr_dir

    if args.output is None:
        args.output = f"output/sae_feature_matching_extended_p{int(args.percentile)}.json"

    # Load MNN results
    with open(mnn_path) as f:
        mnn_data = json.load(f)

    # Compute per-model null thresholds
    thresholds = compute_null_thresholds(baseline_path, args.percentile)
    print(f"Per-model r thresholds (p{int(args.percentile)}):")
    for model, t in sorted(thresholds.items(), key=lambda x: x[1]):
        print(f"  {model:<12s} {t:.3f}")
    print()

    # Process each pair
    extended_pairs = {}
    summary = {}

    for pair_key, mnn_result in sorted(mnn_data["pairs"].items()):
        npz_path = corr_dir / f"{pair_key}.npz"
        if not npz_path.exists():
            print(f"  Skipping {pair_key}: no correlation matrix")
            continue

        name_a, name_b = pair_key.split("__")
        threshold_a = thresholds.get(name_a, 0.5)
        threshold_b = thresholds.get(name_b, 0.5)

        # Load correlation matrix
        npz = np.load(npz_path)
        corr_matrix = npz["corr_matrix"]
        indices_a = npz["indices_a"]
        indices_b = npz["indices_b"]

        # Get MNN-matched feature indices
        mnn_matched_a = {m["idx_a"] for m in mnn_result["matches"]}
        mnn_matched_b = {m["idx_b"] for m in mnn_result["matches"]}

        # Extend
        new_matches = extend_pair(
            corr_matrix, indices_a, indices_b,
            mnn_matched_a, mnn_matched_b,
            threshold_a, threshold_b,
        )

        n_mnn = len(mnn_result["matches"])
        n_ext = len(new_matches)
        n_ab = sum(1 for m in new_matches if m["direction"] == "A→B")
        n_ba = sum(1 for m in new_matches if m["direction"] == "B→A")
        ext_rs = [m["r"] for m in new_matches]
        mean_ext_r = float(np.mean(ext_rs)) if ext_rs else 0.0

        total = n_mnn + n_ext
        total_alive_a = mnn_result["n_alive_a"]
        total_alive_b = mnn_result["n_alive_b"]

        extended_pairs[pair_key] = {
            "mnn_matches": mnn_result["matches"],
            "extended_matches": new_matches,
            "n_mnn": n_mnn,
            "n_extended": n_ext,
            "n_total": total,
            "n_alive_a": total_alive_a,
            "n_alive_b": total_alive_b,
            "mean_mnn_r": mnn_result["mean_match_r"],
            "mean_ext_r": mean_ext_r,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "unmatched_a": [
                f for f in mnn_result.get("unmatched_a", [])
                if f not in {m["idx_a"] for m in new_matches}
            ],
            "unmatched_b": [
                f for f in mnn_result.get("unmatched_b", [])
                if f not in {m["idx_b"] for m in new_matches}
            ],
        }

        summary[pair_key] = {
            "n_mnn": n_mnn,
            "n_extended": n_ext,
            "n_total": total,
            "frac_a": round(total / max(total_alive_a, 1), 4),
            "frac_b": round(total / max(total_alive_b, 1), 4),
            "mean_mnn_r": round(mnn_result["mean_match_r"], 4),
            "mean_ext_r": round(mean_ext_r, 4),
        }

        print(
            f"{pair_key:<30s}  MNN={n_mnn:>4d}  +ext={n_ext:>4d} "
            f"(A→B={n_ab}, B→A={n_ba})  total={total:>4d}  "
            f"ext_r={mean_ext_r:.3f}  "
            f"thresh=({threshold_a:.3f}, {threshold_b:.3f})"
        )

    # Write output
    output = {
        "metadata": {
            "mnn_source": str(mnn_path),
            "baseline_source": str(baseline_path),
            "null_percentile": args.percentile,
            "thresholds": thresholds,
            **mnn_data["metadata"],
        },
        "pairs": extended_pairs,
        "summary": {"per_pair": summary},
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'Pair':<30s} {'MNN':>5s} {'+ Ext':>6s} {'Total':>6s} {'Frac A':>8s} {'Frac B':>8s} {'MNN r':>7s} {'Ext r':>7s}")
    print("-" * 80)
    for pk in sorted(summary.keys(), key=lambda k: -summary[k]["n_total"]):
        s = summary[pk]
        print(
            f"{pk:<30s} {s['n_mnn']:>5d} {s['n_extended']:>+6d} {s['n_total']:>6d} "
            f"{s['frac_a']:>8.3f} {s['frac_b']:>8.3f} {s['mean_mnn_r']:>7.3f} {s['mean_ext_r']:>7.3f}"
        )


if __name__ == "__main__":
    main()
