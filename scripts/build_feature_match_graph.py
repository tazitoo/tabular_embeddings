#!/usr/bin/env python3
"""
Build per-feature cross-model match graph.

For each alive feature in each model, finds its best correlate across all
partner models and classifies it as:
  - mnn: mutual nearest neighbor match AND above pair-specific threshold
  - threshold: best correlate exceeds pair-specific p90 random baseline
  - mnn_below_threshold: MNN match but below pair threshold
  - unmatched: no correlate above noise floor in any partner

Thresholds are per-pair and per-direction, derived from cross-model
trained-vs-random baselines (the correct null for cross-model matching).

Inputs:
  - MNN matching results (JSON)
  - Cross-model random baseline (JSON, from trained-A vs random-B)
  - Cross-correlation matrices (NPZ)

Output:
  - Per-feature match graph (JSON) with tiers, best correlates, and groups
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_sae_concepts_deep import NumpyEncoder, convert_keys_to_native


def load_pair_thresholds(
    baseline_path: Path, percentile_key: str = "p90"
) -> Dict[Tuple[str, str], float]:
    """Load per-pair per-direction thresholds from cross-model random baseline.

    Returns:
        Dict mapping (trained_model, random_model) → threshold.
        For feature in model A with correlate in model B,
        look up (A, B) to get the threshold.
    """
    with open(baseline_path) as f:
        data = json.load(f)

    thresholds = {}
    for pair_key, stats in data["pairs"].items():
        # Key format: "ModelA__trained_vs_ModelB__random"
        parts = pair_key.split("__trained_vs_")
        if len(parts) != 2:
            continue
        model_a = parts[0]
        model_b = parts[1].replace("__random", "")
        thresholds[(model_a, model_b)] = stats[percentile_key]
    return thresholds


def build_feature_graph(
    mnn_path: Path,
    baseline_path: Path,
    corr_dir: Path,
    percentile_key: str = "p90",
    floor: float = 0.0,
) -> dict:
    """Build the per-feature match graph with per-pair thresholds."""

    # Load MNN results
    with open(mnn_path) as f:
        mnn_data = json.load(f)

    # Load per-pair cross-model thresholds, apply floor
    pair_thresholds = load_pair_thresholds(baseline_path, percentile_key)
    if floor > 0:
        pair_thresholds = {k: max(v, floor) for k, v in pair_thresholds.items()}

    # ── Pass 1: Index alive features, best correlates, MNN matches ────

    model_alive: Dict[str, Set[int]] = defaultdict(set)
    mnn_matches: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    # best_correlates[feature_key][partner_model] = (feat_idx, r)
    best_correlates: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)

    for pair_key in sorted(mnn_data["pairs"].keys()):
        npz_path = corr_dir / f"{pair_key}.npz"
        if not npz_path.exists():
            continue

        name_a, name_b = pair_key.split("__")
        npz = np.load(npz_path)
        corr = npz["corr_matrix"]
        indices_a = npz["indices_a"]
        indices_b = npz["indices_b"]

        for idx in indices_a:
            model_alive[name_a].add(int(idx))
        for idx in indices_b:
            model_alive[name_b].add(int(idx))

        # A→B best correlates
        best_b_pos = corr.argmax(axis=1)
        for i, idx_a in enumerate(indices_a):
            j = best_b_pos[i]
            r = float(corr[i, j])
            idx_b = int(indices_b[j])
            key_a = f"{name_a}::{int(idx_a)}"
            if name_b not in best_correlates[key_a] or r > best_correlates[key_a][name_b][1]:
                best_correlates[key_a][name_b] = (idx_b, r)

        # B→A best correlates
        best_a_pos = corr.argmax(axis=0)
        for j, idx_b in enumerate(indices_b):
            i = best_a_pos[j]
            r = float(corr[i, j])
            idx_a = int(indices_a[i])
            key_b = f"{name_b}::{int(idx_b)}"
            if name_a not in best_correlates[key_b] or r > best_correlates[key_b][name_a][1]:
                best_correlates[key_b][name_a] = (idx_a, r)

    # Index MNN matches
    for pair_key, result in mnn_data["pairs"].items():
        name_a, name_b = pair_key.split("__")
        for m in result["matches"]:
            key_a = f"{name_a}::{m['idx_a']}"
            key_b = f"{name_b}::{m['idx_b']}"
            mnn_matches[key_a].append((key_b, m["r"]))
            mnn_matches[key_b].append((key_a, m["r"]))

    # ── Pass 2: Classify each feature ─────────────────────────────────

    features = {}
    tier_counts = defaultdict(int)
    model_tier_counts = defaultdict(lambda: defaultdict(int))

    for model, alive_set in sorted(model_alive.items()):
        for feat_idx in sorted(alive_set):
            key = f"{model}::{feat_idx}"

            # Find best correlate across all partners, checking per-pair threshold
            best_partner = None
            best_partner_feat = None
            best_r = 0.0
            above_any_threshold = False

            for partner_model, (partner_feat, r) in best_correlates.get(key, {}).items():
                # Per-pair threshold: what's the noise floor for model→partner?
                threshold = pair_thresholds.get((model, partner_model), 0.5)

                if r > best_r:
                    best_r = r
                    best_partner = partner_model
                    best_partner_feat = partner_feat

                if r >= threshold:
                    above_any_threshold = True

            # Look up the specific threshold for the best partner
            best_threshold = pair_thresholds.get(
                (model, best_partner), 0.5
            ) if best_partner else 0.5
            above_best_threshold = best_r >= best_threshold

            # Classify
            has_mnn = key in mnn_matches
            if has_mnn and above_any_threshold:
                tier = "mnn"
            elif has_mnn:
                tier = "mnn_below_threshold"
            elif above_any_threshold:
                tier = "threshold"
            else:
                tier = "unmatched"

            entry = {
                "model": model,
                "feat_idx": feat_idx,
                "tier": tier,
                "best_correlate": {
                    "model": best_partner,
                    "feat_idx": best_partner_feat,
                    "r": round(best_r, 4),
                    "threshold": round(best_threshold, 4),
                } if best_partner else None,
            }

            # For threshold-matched features, record which partner(s) exceeded
            if above_any_threshold:
                above_partners = []
                for pm, (pf, r) in best_correlates.get(key, {}).items():
                    t = pair_thresholds.get((model, pm), 0.5)
                    if r >= t:
                        above_partners.append({
                            "model": pm,
                            "feat_idx": pf,
                            "r": round(r, 4),
                            "threshold": round(t, 4),
                        })
                above_partners.sort(key=lambda x: -x["r"])
                entry["above_threshold_partners"] = above_partners

            if has_mnn:
                entry["mnn_partners"] = [
                    {"feature": partner, "r": round(r, 4)}
                    for partner, r in sorted(mnn_matches[key], key=lambda x: -x[1])
                ]

            features[key] = entry
            tier_counts[tier] += 1
            model_tier_counts[model][tier] += 1

    # Build threshold summary for output
    threshold_summary = {}
    for (a, b), t in sorted(pair_thresholds.items()):
        threshold_summary[f"{a}_vs_{b}"] = round(t, 4)

    return {
        "features": features,
        "pair_thresholds": threshold_summary,
        "percentile_key": percentile_key,
        "floor": floor,
        "tier_counts": dict(tier_counts),
        "model_tier_counts": {m: dict(c) for m, c in model_tier_counts.items()},
        "model_alive_counts": {m: len(s) for m, s in model_alive.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build per-feature cross-model match graph"
    )
    parser.add_argument(
        "--mnn-path", type=str,
        default="output/sae_feature_matching_mnn_t0.001_n500.json",
    )
    parser.add_argument(
        "--baseline-path", type=str,
        default="output/sae_cross_model_random_baseline.json",
    )
    parser.add_argument(
        "--corr-dir", type=str,
        default="output/sae_cross_correlations",
    )
    parser.add_argument(
        "--percentile", type=str, default="p90",
        choices=["p90", "p95"],
        help="Percentile key from baseline JSON (default: p90)",
    )
    parser.add_argument(
        "--floor", type=float, default=0.0,
        help="Minimum correlation threshold floor (default: 0.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        floor_str = f"_floor{args.floor:.2f}" if args.floor > 0 else ""
        args.output = f"output/sae_feature_match_graph_{args.percentile}{floor_str}.json"

    result = build_feature_graph(
        mnn_path=PROJECT_ROOT / args.mnn_path,
        baseline_path=PROJECT_ROOT / args.baseline_path,
        corr_dir=PROJECT_ROOT / args.corr_dir,
        percentile_key=args.percentile,
        floor=args.floor,
    )

    # Print summary
    floor_str = f", floor={args.floor}" if args.floor > 0 else ""
    print(f"Threshold: {args.percentile} from cross-model random baseline{floor_str}")

    print(f"\nGlobal tier counts:")
    for tier, count in sorted(result["tier_counts"].items()):
        print(f"  {tier:<25s} {count:>6d}")
    total = sum(result["tier_counts"].values())
    print(f"  {'TOTAL':<25s} {total:>6d}")

    print(f"\nPer-model breakdown:")
    print(f"{'Model':<12s} {'Alive':>6s} {'MNN':>6s} {'MNN<t':>6s} {'Thresh':>7s} {'Unmatch':>8s} {'Match%':>7s}")
    print("-" * 60)
    for model in sorted(result["model_alive_counts"].keys()):
        alive = result["model_alive_counts"][model]
        tc = result["model_tier_counts"].get(model, {})
        mnn = tc.get("mnn", 0)
        mnn_below = tc.get("mnn_below_threshold", 0)
        thresh = tc.get("threshold", 0)
        unmatched = tc.get("unmatched", 0)
        matched = mnn + thresh
        pct = matched / alive * 100 if alive else 0
        print(f"{model:<12s} {alive:>6d} {mnn:>6d} {mnn_below:>6d} {thresh:>7d} {unmatched:>8d} {pct:>6.1f}%")

    # Save
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(result), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
