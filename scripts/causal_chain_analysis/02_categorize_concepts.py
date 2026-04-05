#!/usr/bin/env python3
"""Stage 2: Categorize concepts into interpretable groups.

Groups the top-N concepts from Stage 1 into categories by keyword
analysis of their labels:
- Magnitude patterns (z-scores, log-magnitude, absolute values)
- Distribution shape (variance, range, skew, spread)
- Outlier detection (centroid distance, isolation, extreme values)
- Categorical patterns (entropy, rarity, modal dominance)
- Sparsity patterns (zero fraction, binary features, missing values)
- Boundary/density (decision boundary, local density, neighbors)
- PCA alignment (principal component projections)

Usage:
    python -m scripts.causal_chain_analysis.02_categorize_concepts
    python -m scripts.causal_chain_analysis.02_categorize_concepts --pair tabicl_vs_tabicl_v2
"""
import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"

# Category definitions: (category_name, keywords_to_match)
CATEGORIES = [
    ("Outlier detection", [
        "outlier", "extreme", "far from", "distant", "isolated",
        "centroid distance", "atypical", "peripheral",
    ]),
    ("Magnitude patterns", [
        "magnitude", "z-score", "elevated", "large absolute",
        "high log-magnitude", "low magnitude", "near zero",
        "strongly elevated", "suppressed",
    ]),
    ("Distribution shape", [
        "variance", "range", "spread", "heterogene", "homogene",
        "narrow", "wide", "compressed", "dispersed",
        "log-magnitude std", "within-row",
    ]),
    ("Categorical patterns", [
        "categorical", "entropy", "rarity", "modal", "rare categor",
        "common categor", "one-hot", "cardinality", "distinct values",
    ]),
    ("Sparsity patterns", [
        "sparse", "zero", "binary", "missing", "NaN", "empty",
        "all-zero", "near-zero", "degenerate",
    ]),
    ("Boundary/density", [
        "boundary", "density", "neighbor", "local", "nearest",
        "decision", "separab", "cluster",
    ]),
    ("PCA alignment", [
        "pca", "principal component", "pc1", "pc2",
    ]),
    ("Minimum/baseline", [
        "minimum possible", "baseline", "default", "modal frac",
        "typical", "unremarkable", "generic",
    ]),
]


def categorize_label(label: str) -> str:
    """Assign a category to a concept label based on keyword matching."""
    label_lower = label.lower()
    scores = {}
    for cat_name, keywords in CATEGORIES:
        score = sum(1 for kw in keywords if kw.lower() in label_lower)
        if score > 0:
            scores[cat_name] = score

    if scores:
        return max(scores, key=scores.get)
    return "Other"


def categorize_pair(pair_name):
    """Categorize concepts for one pair."""
    ranking_path = EVIDENCE_DIR / pair_name / "concept_ranking.json"
    if not ranking_path.exists():
        return None

    ranking = json.loads(ranking_path.read_text())
    results = {}

    for model_key, model_data in ranking["models"].items():
        features = model_data["top_features"]
        if not features:
            continue

        # Categorize each feature
        categorized = []
        category_counts = Counter()
        category_rows = Counter()

        for feat in features:
            cat = categorize_label(feat["label"])
            categorized.append({
                "feature_idx": feat["feature_idx"],
                "category": cat,
                "label": feat["label"],
                "n_rows": feat["n_rows"],
                "n_datasets": feat["n_datasets"],
                "band": feat["band"],
                "is_unmatched": feat["is_unmatched"],
            })
            category_counts[cat] += 1
            category_rows[cat] += feat["n_rows"]

        # Sort categories by total row selections
        sorted_cats = sorted(category_rows.items(), key=lambda x: -x[1])

        results[model_key] = {
            "display_name": model_data["display_name"],
            "categories": {
                cat: {
                    "n_features": category_counts[cat],
                    "total_rows": rows,
                    "fraction_of_selections": rows / max(
                        sum(category_rows.values()), 1),
                    "features": [c for c in categorized if c["category"] == cat],
                }
                for cat, rows in sorted_cats
            },
            "categorized_features": categorized,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Categorize concepts into interpretable groups")
    parser.add_argument("--pair", default=None)
    args = parser.parse_args()

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted(d.name for d in EVIDENCE_DIR.iterdir()
                        if d.is_dir() and (d / "concept_ranking.json").exists())

    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair}")

        result = categorize_pair(pair)
        if result is None:
            continue

        out_path = EVIDENCE_DIR / pair / "concept_categories.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        for model_key, data in result.items():
            disp = data["display_name"]
            logger.info(f"\n  {disp} (when strong):")
            for cat, info in data["categories"].items():
                n = info["n_features"]
                rows = info["total_rows"]
                frac = info["fraction_of_selections"]
                logger.info(f"    {cat}: {n} features, {rows} rows ({frac:.0%})")

        logger.info(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
