#!/usr/bin/env python3
"""Stage 7: Cross-pair synthesis.

Aggregates concept rankings across all 15 pairs to identify:
- Universal differentiators: concepts in top-10 for many pairs
- Pair-specific concepts: concepts important for only one pair
- Model fingerprint: which concept categories define each model
- Transfer success by category: do certain concept types transfer better?

Usage:
    python -m scripts.causal_chain_analysis.07_cross_pair_synthesis
"""
import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts._project_root import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"
TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_sweep_v2"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Cross-pair synthesis")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    pairs = sorted(d.name for d in EVIDENCE_DIR.iterdir()
                    if d.is_dir() and (d / "concept_ranking.json").exists())

    # Aggregate: (model, feature) -> list of pairs where it appears in top-N
    feature_pairs = defaultdict(list)
    # Per-model: category counts across all pairs
    model_categories = defaultdict(Counter)
    # Per-model: total row selections
    model_total_rows = defaultdict(int)

    for pair in pairs:
        ranking = json.loads(
            (EVIDENCE_DIR / pair / "concept_ranking.json").read_text())
        cat_path = EVIDENCE_DIR / pair / "concept_categories.json"
        categories = json.loads(cat_path.read_text()) if cat_path.exists() else {}

        for model_key, model_data in ranking["models"].items():
            top_feats = model_data["top_features"][:args.top_n]
            for feat in top_feats:
                key = (model_key, feat["feature_idx"])
                feature_pairs[key].append(pair)

            # Category aggregation
            if model_key in categories:
                for cat, info in categories[model_key]["categories"].items():
                    model_categories[model_key][cat] += info["total_rows"]
                    model_total_rows[model_key] += info["total_rows"]

    # Universal differentiators: features in top-N for many pairs
    logger.info("=" * 60)
    logger.info("UNIVERSAL DIFFERENTIATORS")
    logger.info("(concepts in top-10 across many pairs)")
    logger.info("")

    multi_pair = [(key, pairs_list) for key, pairs_list in feature_pairs.items()
                   if len(pairs_list) >= 3]
    multi_pair.sort(key=lambda x: -len(x[1]))

    # Load labels for display
    labels_path = EVIDENCE_DIR / pairs[0] / "concept_ranking.json"
    all_rankings = {}
    for pair in pairs:
        r = json.loads((EVIDENCE_DIR / pair / "concept_ranking.json").read_text())
        all_rankings[pair] = r

    from scripts.intervention.intervene_lib import MODEL_KEY_TO_LABEL_KEY

    for (model_key, fi), pair_list in multi_pair[:20]:
        disp = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
        # Get label from first pair's ranking
        label = ""
        for pair in pair_list:
            for feat in all_rankings[pair]["models"].get(model_key, {}).get(
                    "top_features", []):
                if feat["feature_idx"] == fi:
                    label = feat["label"][:50]
                    break
            if label:
                break
        logger.info(f"  {disp} f{fi} ({len(pair_list)} pairs): {label}")
        logger.info(f"    pairs: {', '.join(pair_list)}")

    # Model fingerprint: category distribution per model
    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL FINGERPRINTS")
    logger.info("(concept categories when model is strong)")
    logger.info("")

    for model_key in sorted(model_categories.keys()):
        disp = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
        total = model_total_rows[model_key]
        if total == 0:
            continue
        logger.info(f"  {disp}:")
        cats = model_categories[model_key]
        for cat, rows in cats.most_common():
            frac = rows / total
            logger.info(f"    {cat}: {frac:.0%} ({rows} rows)")

    # Pair-specific concepts: only appear for one pair
    pair_specific = [(key, pairs_list) for key, pairs_list in feature_pairs.items()
                      if len(pairs_list) == 1]
    n_pair_specific = len(pair_specific)
    n_total = len(feature_pairs)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PAIR-SPECIFIC: {n_pair_specific}/{n_total} concepts "
                f"({n_pair_specific/max(n_total,1):.0%}) appear in only 1 pair")

    # Transfer success by category (if transfer results exist)
    transfer_by_category = defaultdict(lambda: {"tried": 0, "accepted": 0})
    for pair in pairs:
        cat_path = EVIDENCE_DIR / pair / "concept_categories.json"
        if not cat_path.exists():
            continue
        categories = json.loads(cat_path.read_text())

        # Check for transfer feature_acceptance
        transfer_pair_dir = TRANSFER_DIR / pair
        if not transfer_pair_dir.exists():
            continue

        for model_key, cat_data in categories.items():
            feat_to_cat = {f["feature_idx"]: f["category"]
                           for f in cat_data.get("categorized_features", [])}

            for f in transfer_pair_dir.glob("*.npz"):
                d = np.load(f, allow_pickle=True)
                if str(d.get("strong_model", "")) != model_key:
                    continue
                if "feature_acceptance" not in d.files:
                    continue
                fa = d["feature_acceptance"]
                if fa.size == 0:
                    continue
                for row in fa:
                    fi, tried, accepted = int(row[0]), int(row[1]), int(row[2])
                    cat = feat_to_cat.get(fi, "Other")
                    transfer_by_category[cat]["tried"] += tried
                    transfer_by_category[cat]["accepted"] += accepted

    if transfer_by_category:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRANSFER SUCCESS BY CONCEPT CATEGORY")
        logger.info("")
        logger.info(f"{'Category':<25} {'Tried':>8} {'Accepted':>10} {'Rate':>8}")
        logger.info("-" * 55)
        for cat in sorted(transfer_by_category.keys(),
                           key=lambda c: -transfer_by_category[c]["accepted"]):
            info = transfer_by_category[cat]
            rate = info["accepted"] / max(info["tried"], 1)
            logger.info(f"{cat:<25} {info['tried']:>8} {info['accepted']:>10} "
                        f"{rate:>7.0%}")

    # Save synthesis
    output = {
        "universal_differentiators": [
            {"model": key[0], "feature_idx": key[1],
             "n_pairs": len(plist), "pairs": plist}
            for (key, plist) in multi_pair[:50]
        ],
        "model_fingerprints": {
            model: {cat: rows / max(model_total_rows[model], 1)
                    for cat, rows in model_categories[model].most_common()}
            for model in sorted(model_categories.keys())
        },
        "pair_specific_fraction": n_pair_specific / max(n_total, 1),
        "transfer_by_category": dict(transfer_by_category),
    }

    out_path = EVIDENCE_DIR / "cross_pair_synthesis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
