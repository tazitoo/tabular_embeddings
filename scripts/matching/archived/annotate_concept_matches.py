#!/usr/bin/env python3
"""
Annotate MNN feature matches with concept regression probes.

For each matched pair (model_A:feat_i ↔ model_B:feat_k), looks up both
features' top concept probes and computes overlap. Identifies universal
concepts (matched across 3+ model pairs with shared probes) and produces
a summary table.

Usage:
    python scripts/annotate_feature_matches.py \
        --matching output/sae_feature_matching_mnn_t0.001_n500.json \
        --concepts output/sae_concept_analysis_round8.json \
        --output output/annotated_feature_matches.json \
        --top-k 5
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from scripts.sae.analyze_sae_concepts_deep import NumpyEncoder, convert_keys_to_native
from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND


def load_concept_probes(
    concepts: dict, top_k: int = 5
) -> Dict[str, Dict[int, dict]]:
    """
    Build lookup: model_name → {feature_idx: {r2, probes: [name, ...]}}.

    Args:
        concepts: Loaded sae_concept_analysis JSON
        top_k: Number of top probes to keep per feature

    Returns:
        Nested dict for fast lookup by model and feature index.
    """
    lookup = {}
    for model_name, model_data in concepts["models"].items():
        per_feature = model_data.get("per_feature", {})
        features = {}
        for feat_idx_str, feat_data in per_feature.items():
            feat_idx = int(feat_idx_str)
            probes = feat_data.get("top_probes", [])[:top_k]
            features[feat_idx] = {
                "r2": feat_data["r2"],
                "probes": [(p[0], float(p[1])) for p in probes],
                "probe_names": {p[0] for p in probes},
            }
        lookup[model_name] = features
    return lookup


def annotate_match(
    idx_a: int,
    idx_b: int,
    r: float,
    probes_a: Optional[dict],
    probes_b: Optional[dict],
) -> dict:
    """
    Annotate a single matched pair with probe overlap.

    Returns dict with match info, per-feature probes, and overlap stats.
    """
    result = {"idx_a": idx_a, "idx_b": idx_b, "r": r}

    if probes_a is None or probes_b is None:
        result["r2_a"] = probes_a["r2"] if probes_a else None
        result["r2_b"] = probes_b["r2"] if probes_b else None
        result["shared_probes"] = []
        result["n_shared"] = 0
        return result

    result["r2_a"] = probes_a["r2"]
    result["r2_b"] = probes_b["r2"]
    result["probes_a"] = [p[0] for p in probes_a["probes"]]
    result["probes_b"] = [p[0] for p in probes_b["probes"]]

    shared = probes_a["probe_names"] & probes_b["probe_names"]
    result["shared_probes"] = sorted(shared)
    result["n_shared"] = len(shared)

    # Sign agreement: do shared probes have same-sign coefficients?
    if shared:
        coeff_a = {p[0]: p[1] for p in probes_a["probes"]}
        coeff_b = {p[0]: p[1] for p in probes_b["probes"]}
        same_sign = sum(
            1 for p in shared
            if (coeff_a[p] > 0) == (coeff_b[p] > 0)
        )
        result["sign_agreement"] = same_sign
    else:
        result["sign_agreement"] = 0

    return result


def annotate_all_pairs(
    matching: dict, probe_lookup: Dict[str, Dict[int, dict]]
) -> dict:
    """Annotate all matched pairs across all model combinations."""
    annotated_pairs = {}

    for pair_key, pair_data in matching["pairs"].items():
        model_a, model_b = pair_key.split("__")
        lookup_a = probe_lookup.get(model_a, {})
        lookup_b = probe_lookup.get(model_b, {})

        annotated_matches = []
        for m in pair_data["matches"]:
            probes_a = lookup_a.get(m["idx_a"])
            probes_b = lookup_b.get(m["idx_b"])
            ann = annotate_match(m["idx_a"], m["idx_b"], m["r"], probes_a, probes_b)
            annotated_matches.append(ann)

        # Sort by n_shared desc, then r desc
        annotated_matches.sort(key=lambda x: (-x["n_shared"], -x["r"]))

        n_with_shared = sum(1 for m in annotated_matches if m["n_shared"] > 0)
        mean_shared = (
            np.mean([m["n_shared"] for m in annotated_matches])
            if annotated_matches else 0
        )

        annotated_pairs[pair_key] = {
            "model_a": model_a,
            "model_b": model_b,
            "n_matched": len(annotated_matches),
            "n_with_shared_probes": n_with_shared,
            "mean_shared_probes": round(float(mean_shared), 2),
            "matches": annotated_matches,
        }

    return annotated_pairs


def find_universal_concepts(
    annotated_pairs: dict, min_pairs: int = 3, min_shared: int = 1
) -> List[dict]:
    """
    Find probes that appear as shared in matches across multiple model pairs.

    A "universal concept" is a probe name that appears in shared_probes of
    at least `min_pairs` different model pairs.

    Returns list of {probe, n_pairs, pairs, example_matches}.
    """
    # probe_name → list of (pair_key, match) where it appears shared
    probe_appearances = defaultdict(list)

    for pair_key, pair_data in annotated_pairs.items():
        for m in pair_data["matches"]:
            for probe in m.get("shared_probes", []):
                probe_appearances[probe].append((pair_key, m))

    # Deduplicate: count unique pairs per probe
    universal = []
    for probe, appearances in probe_appearances.items():
        unique_pairs = set(pk for pk, _ in appearances)
        if len(unique_pairs) >= min_pairs:
            # Pick best example match per pair (highest r)
            best_per_pair = {}
            for pk, m in appearances:
                if pk not in best_per_pair or m["r"] > best_per_pair[pk]["r"]:
                    best_per_pair[pk] = m
            universal.append({
                "probe": probe,
                "n_pairs": len(unique_pairs),
                "n_matches": len(appearances),
                "pairs": sorted(unique_pairs),
                "best_examples": [
                    {
                        "pair": pk,
                        "idx_a": m["idx_a"],
                        "idx_b": m["idx_b"],
                        "r": m["r"],
                    }
                    for pk, m in sorted(
                        best_per_pair.items(), key=lambda x: -x[1]["r"]
                    )[:3]
                ],
            })

    universal.sort(key=lambda x: (-x["n_pairs"], -x["n_matches"]))
    return universal


def compute_summary_stats(annotated_pairs: dict) -> dict:
    """Compute summary statistics across all pairs."""
    rows = []
    for pair_key, pair_data in annotated_pairs.items():
        matches = pair_data["matches"]
        if not matches:
            continue
        r_values = [m["r"] for m in matches]
        shared_counts = [m["n_shared"] for m in matches]
        r2_a = [m["r2_a"] for m in matches if m.get("r2_a") is not None]
        r2_b = [m["r2_b"] for m in matches if m.get("r2_b") is not None]
        rows.append({
            "pair": pair_key,
            "n_matched": len(matches),
            "mean_r": round(float(np.mean(r_values)), 4),
            "n_with_shared": sum(1 for s in shared_counts if s > 0),
            "frac_with_shared": round(
                sum(1 for s in shared_counts if s > 0) / len(matches), 3
            ),
            "mean_shared": round(float(np.mean(shared_counts)), 2),
            "mean_r2_a": round(float(np.mean(r2_a)), 3) if r2_a else None,
            "mean_r2_b": round(float(np.mean(r2_b)), 3) if r2_b else None,
        })
    rows.sort(key=lambda x: -x["frac_with_shared"])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Annotate MNN feature matches with concept probes"
    )
    parser.add_argument(
        "--matching",
        type=str,
        default="output/sae_feature_matching_mnn_t0.001_n500.json",
        help="MNN matching JSON",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=f"output/sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json",
        help="SAE concept analysis JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/annotated_feature_matches.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top probes per feature to compare (default: 5)",
    )
    parser.add_argument(
        "--min-pairs-universal",
        type=int,
        default=3,
        help="Min model pairs for a probe to be 'universal' (default: 3)",
    )
    args = parser.parse_args()

    # Load data
    with open(PROJECT_ROOT / args.matching) as f:
        matching = json.load(f)
    with open(PROJECT_ROOT / args.concepts) as f:
        concepts = json.load(f)

    print(f"Matching: {len(matching['pairs'])} pairs, "
          f"method={matching['metadata']['method']}")
    print(f"Concepts: {len(concepts['models'])} models, "
          f"{concepts['metadata']['n_total_probes']} probes")

    # Build probe lookup
    probe_lookup = load_concept_probes(concepts, top_k=args.top_k)
    for model, feats in sorted(probe_lookup.items()):
        print(f"  {model}: {len(feats)} features with probes")

    # Annotate
    annotated_pairs = annotate_all_pairs(matching, probe_lookup)

    # Universal concepts
    universal = find_universal_concepts(
        annotated_pairs, min_pairs=args.min_pairs_universal
    )

    # Summary
    summary = compute_summary_stats(annotated_pairs)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Pair':<30} {'N':>5} {'Mean r':>8} {'Shared':>7} "
          f"{'Frac':>6} {'MeanSh':>7} {'R²_A':>6} {'R²_B':>6}")
    print("-" * 80)
    for row in summary:
        print(
            f"{row['pair']:<30} {row['n_matched']:>5} "
            f"{row['mean_r']:>8.3f} {row['n_with_shared']:>7} "
            f"{row['frac_with_shared']:>6.2f} {row['mean_shared']:>7.2f} "
            f"{row['mean_r2_a'] or 0:>6.3f} {row['mean_r2_b'] or 0:>6.3f}"
        )

    # Print universal concepts
    print(f"\n{'='*80}")
    print(f"Universal concepts (shared in {args.min_pairs_universal}+ model pairs):")
    print("-" * 80)
    if universal:
        for uc in universal:
            pairs_short = ", ".join(
                p.replace("__", "↔") for p in uc["pairs"][:5]
            )
            print(f"  {uc['probe']:<30} {uc['n_pairs']:>2} pairs, "
                  f"{uc['n_matches']:>3} matches  [{pairs_short}]")
    else:
        print("  (none found)")

    # Print top annotated matches
    print(f"\n{'='*80}")
    print("Top matches with shared probes (highest r):")
    print("-" * 80)
    all_annotated = []
    for pair_key, pair_data in annotated_pairs.items():
        for m in pair_data["matches"]:
            if m["n_shared"] > 0:
                all_annotated.append((pair_key, m))
    all_annotated.sort(key=lambda x: -x[1]["r"])
    for pair_key, m in all_annotated[:20]:
        shared = ", ".join(m["shared_probes"])
        sign = f"{m['sign_agreement']}/{m['n_shared']}" if m["n_shared"] else ""
        print(
            f"  {pair_key:<28} "
            f"{m['idx_a']:>5}↔{m['idx_b']:<5} "
            f"r={m['r']:.3f}  "
            f"R²={m['r2_a']:.2f}/{m['r2_b']:.2f}  "
            f"shared({sign}): {shared}"
        )

    # Save
    output = {
        "metadata": {
            "matching_file": args.matching,
            "concepts_file": args.concepts,
            "top_k": args.top_k,
            "min_pairs_universal": args.min_pairs_universal,
        },
        "pairs": annotated_pairs,
        "universal_concepts": universal,
        "summary": summary,
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(output), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
