#!/usr/bin/env python3
"""
Stacked bar chart: per-feature explained variance for one reference model.

For each reference model feature, classifies it by cross-model matching status:
  - Universal: matched in 3+ other models with shared probes
  - Bilateral: matched in 1-2 other models with shared probes
  - Match only: matched but no shared probes (novel correspondence)
  - Unmatched explained: not matched, but R² > threshold from probes
  - Unexplained: not matched, low R²

Usage:
    python scripts/figure_feature_matching.py \
        --annotated output/annotated_feature_matches.json \
        --concepts output/concept_regression_with_pymfe.json \
        --ref-model TabPFN
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_splitting_lookup(splitting: dict) -> dict:
    """
    Build lookup: (pair_key, target_idx_b) → group result dict.

    Also build per-feature lookup: (pair_key, idx_a) → group result.
    """
    pair_lookup = {}  # (pair_key, idx_a) → group
    for pair_key, pair_data in splitting.get("pairs", {}).items():
        for group in pair_data.get("groups", []):
            for member in group.get("members", []):
                pair_lookup[(pair_key, member["idx_a"])] = group
    return pair_lookup


def classify_features(
    annotated: dict,
    concepts: dict,
    ref_model: str,
    r2_threshold: float = 0.1,
    min_r: float = 0.0,
    splitting: dict = None,
) -> dict:
    """
    Classify each reference model feature by matching + explanation status.

    Args:
        min_r: Minimum |r| for a match to count. Matches below this threshold
            are treated as unmatched (filters noise from many-to-one argmax).
        splitting: Optional concept splitting results JSON. When provided:
            - "split" group members: use sqrt(group_r2) as effective |r|
            - "single_match" groups: only best_individual_idx counts
            - "noise" groups: all members treated as unmatched

    Returns dict mapping feature_idx → {
        n_matched, n_shared, best_r, r2, category,
        matched_models, shared_models
    }
    """
    per_feature = concepts["models"][ref_model]["per_feature"]
    split_lookup = _build_splitting_lookup(splitting) if splitting else {}

    # Collect match info per feature across all pairs
    feat_info = {}
    for idx_str, feat_data in per_feature.items():
        idx = int(idx_str)
        feat_info[idx] = {
            "r2": feat_data["r2"],
            "matched_models": [],  # models where this feature is matched
            "shared_models": [],   # models where match has shared probes
            "best_r": 0.0,
        }

    # Walk all pairs involving ref_model
    for pair_key, pair_data in annotated["pairs"].items():
        model_a, model_b = pair_data["model_a"], pair_data["model_b"]
        if ref_model not in (model_a, model_b):
            continue
        other_model = model_b if model_a == ref_model else model_a
        is_a = model_a == ref_model

        for m in pair_data["matches"]:
            idx = m["idx_a"] if is_a else m["idx_b"]
            if idx not in feat_info:
                continue

            # Determine effective |r| using splitting results
            effective_r = m["r"]
            skip = False

            if split_lookup:
                # Look up using the A-side index and pair key for this match
                idx_a_for_lookup = m["idx_a"]
                group = split_lookup.get((pair_key, idx_a_for_lookup))
                if group is not None:
                    cls = group["classification"]
                    if cls == "split":
                        # Use sqrt(group_r2) as effective correlation
                        effective_r = group["group_r2"] ** 0.5
                    elif cls == "single_match":
                        # Only the best individual counts
                        if idx_a_for_lookup != group["best_individual_idx"]:
                            skip = True
                    elif cls == "noise":
                        skip = True

            if skip or effective_r < min_r:
                continue

            feat_info[idx]["matched_models"].append(other_model)
            feat_info[idx]["best_r"] = max(feat_info[idx]["best_r"], effective_r)
            if m.get("n_shared", 0) > 0:
                feat_info[idx]["shared_models"].append(other_model)

    # Classify — use n_matched (MNN match count) for tiers,
    # n_shared (probes agree) tracked separately
    for idx, info in feat_info.items():
        n_matched = len(set(info["matched_models"]))
        n_shared = len(set(info["shared_models"]))
        info["n_matched"] = n_matched
        info["n_shared"] = n_shared

        if n_matched >= 4:
            info["category"] = "matched_4plus"
        elif n_matched >= 2:
            info["category"] = "matched_2_3"
        elif n_matched >= 1:
            info["category"] = "matched_1"
        elif info["r2"] >= r2_threshold:
            info["category"] = "unmatched_explained"
        else:
            info["category"] = "unexplained"

    return feat_info


def make_stacked_bar(feat_info: dict, ref_model: str, output_path: Path, min_r: float = 0.0):
    """
    Two-panel figure:
      Left: stacked bar showing feature count per category
      Right: stacked bar showing fraction of total R² per category
    """
    categories = [
        ("matched_4plus", "Matched 4+ models", "#1a9850"),
        ("matched_2_3", "Matched 2-3 models", "#91cf60"),
        ("matched_1", "Matched 1 model", "#d9ef8b"),
        ("unmatched_explained", "Unmatched, probe-explained", "#9b59b6"),
        ("unexplained", "Unexplained", "#e74c3c"),
    ]

    # Count features and sum R² per category
    counts = defaultdict(int)
    r2_sums = defaultdict(float)
    for idx, info in feat_info.items():
        cat = info["category"]
        counts[cat] += 1
        r2_sums[cat] += info["r2"]

    total_features = sum(counts.values())
    total_r2 = sum(r2_sums.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: feature counts ──
    bottom = 0
    for cat_key, cat_label, color in categories:
        val = counts[cat_key]
        pct = 100 * val / total_features if total_features else 0
        ax1.bar(0, val, bottom=bottom, color=color, label=f"{cat_label} ({val})",
                width=0.5, edgecolor="white", linewidth=0.5)
        if pct > 3:
            ax1.text(0, bottom + val / 2, f"{val}\n({pct:.0f}%)",
                     ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += val

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([0])
    ax1.set_xticklabels([ref_model])
    ax1.set_ylabel("Number of alive SAE features")
    threshold_note = f", min |r|={min_r:.2f}" if min_r > 0 else ""
    ax1.set_title(f"{ref_model}: Feature classification\n({total_features} alive features{threshold_note})")

    # ── Right panel: per-model match breakdown ──
    # For each other model, show how many ref features are matched
    model_matches = defaultdict(lambda: {"shared": 0, "no_shared": 0, "unmatched": 0})
    other_models = set()

    for idx, info in feat_info.items():
        for model in info["shared_models"]:
            other_models.add(model)
        for model in info["matched_models"]:
            other_models.add(model)

    other_models = sorted(other_models)

    for model in other_models:
        for idx, info in feat_info.items():
            if model in info["shared_models"]:
                model_matches[model]["shared"] += 1
            elif model in info["matched_models"]:
                model_matches[model]["no_shared"] += 1
            else:
                model_matches[model]["unmatched"] += 1

    x = np.arange(len(other_models))
    width = 0.6

    shared_vals = [model_matches[m]["shared"] for m in other_models]
    no_shared_vals = [model_matches[m]["no_shared"] for m in other_models]
    unmatched_vals = [model_matches[m]["unmatched"] for m in other_models]

    ax2.bar(x, shared_vals, width, label="Matched + shared probes",
            color="#2ecc71", edgecolor="white", linewidth=0.5)
    ax2.bar(x, no_shared_vals, width, bottom=shared_vals,
            label="Matched, no shared probes", color="#f39c12",
            edgecolor="white", linewidth=0.5)
    ax2.bar(x, unmatched_vals, width,
            bottom=[s + n for s, n in zip(shared_vals, no_shared_vals)],
            label="Unmatched", color="#ecf0f1",
            edgecolor="white", linewidth=0.5)

    # Add match count labels on shared segment
    for i, (s, n) in enumerate(zip(shared_vals, no_shared_vals)):
        total_matched = s + n
        pct = 100 * total_matched / total_features
        ax2.text(i, total_matched + 5, f"{total_matched}\n({pct:.0f}%)",
                 ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(other_models, rotation=30, ha="right")
    ax2.set_ylabel(f"Number of {ref_model} features")
    ax2.set_title(f"{ref_model} features matched to each model")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, total_features * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar: feature matching explained variance"
    )
    parser.add_argument(
        "--annotated",
        type=str,
        default="output/annotated_feature_matches.json",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="output/concept_regression_with_pymfe.json",
    )
    parser.add_argument(
        "--ref-model",
        type=str,
        default="TabPFN",
        help="Reference model (default: TabPFN)",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.1,
        help="R² threshold for 'explained' (default: 0.1)",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=0.10,
        help="Minimum |r| for a match to count (default: 0.10)",
    )
    parser.add_argument(
        "--splitting",
        type=str,
        default=None,
        help="Concept splitting results JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: auto from ref model)",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.annotated) as f:
        annotated = json.load(f)
    with open(PROJECT_ROOT / args.concepts) as f:
        concepts = json.load(f)

    splitting = None
    if args.splitting:
        with open(PROJECT_ROOT / args.splitting) as f:
            splitting = json.load(f)

    if args.output is None:
        args.output = f"output/paper_figures/feature_matching_{args.ref_model.lower()}.png"

    feat_info = classify_features(
        annotated, concepts, args.ref_model, args.r2_threshold, args.min_r,
        splitting=splitting,
    )

    # Print summary
    from collections import Counter
    cats = Counter(info["category"] for info in feat_info.values())
    print(f"\n{args.ref_model} feature classification ({len(feat_info)} features):")
    for cat, count in cats.most_common():
        pct = 100 * count / len(feat_info)
        print(f"  {cat:<25} {count:>4} ({pct:.1f}%)")

    # Mean R² per category
    from collections import defaultdict
    r2_by_cat = defaultdict(list)
    for info in feat_info.values():
        r2_by_cat[info["category"]].append(info["r2"])
    # Also print shared-probe stats per tier
    shared_by_cat = defaultdict(list)
    for info in feat_info.values():
        shared_by_cat[info["category"]].append(info.get("n_shared", 0))

    print(f"\nMean R² and probe overlap by category:")
    for cat in ["matched_4plus", "matched_2_3", "matched_1", "unmatched_explained", "unexplained"]:
        r2_vals = r2_by_cat.get(cat, [])
        sh_vals = shared_by_cat.get(cat, [])
        if r2_vals:
            frac_shared = sum(1 for s in sh_vals if s > 0) / len(sh_vals) if sh_vals else 0
            print(f"  {cat:<25} R²={np.mean(r2_vals):.3f}  "
                  f"probe overlap: {frac_shared:.0%} (n={len(r2_vals)})")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_stacked_bar(feat_info, args.ref_model, out_path, args.min_r)


if __name__ == "__main__":
    main()
