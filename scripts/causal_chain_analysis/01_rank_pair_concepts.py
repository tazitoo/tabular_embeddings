#!/usr/bin/env python3
"""Stage 1: Rank concepts that differentiate each model pair.

For each of the 15 model pairs, aggregates ablation-selected features
across all datasets and rows. Produces a ranked list of concepts that
explain the strong model's advantage, annotated with:
- Matryoshka band
- Concept group and label
- Top datasets where the concept fires
- Matched/unmatched status

Usage:
    python -m scripts.causal_chain_analysis.01_rank_pair_concepts
    python -m scripts.causal_chain_analysis.01_rank_pair_concepts --pair mitra_vs_tabpfn
"""
import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.ablation_sweep import get_unmatched_features
from scripts.intervention.intervene_lib import MODEL_KEY_TO_LABEL_KEY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
CONCEPT_LABELS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels_round10.json"
CONCEPT_ANALYSIS_PATH = PROJECT_ROOT / "output" / "sae_concept_analysis_round10.json"
SAE_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10"
OUTPUT_DIR = PROJECT_ROOT / "output" / "concept_evidence"


def load_band_boundaries(model_key):
    """Load Matryoshka band boundaries for a model."""
    from analysis.sparse_autoencoder import SAEConfig
    ckpt_path = SAE_DIR / model_key / "sae_matryoshka_archetypal_validated.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    return config["matryoshka_dims"] if isinstance(config, dict) else config.matryoshka_dims


def get_band(fi, bands):
    for bi, b in enumerate(bands):
        if fi < b:
            return bi
    return len(bands) - 1


def rank_pair(pair_name, labels_data, concept_analysis):
    """Rank concepts for one model pair."""
    pair_dir = ABLATION_DIR / pair_name
    if not pair_dir.exists():
        logger.warning(f"No ablation results for {pair_name}")
        return None

    lookup = labels_data["feature_lookup"]
    groups = labels_data["concept_groups"]

    # Parse pair name
    parts = pair_name.split("_vs_")
    model_a, model_b = parts[0], parts[1]

    # Get unmatched features for both directions
    unmatched_a = set(get_unmatched_features(model_a, model_b))
    unmatched_b = set(get_unmatched_features(model_b, model_a))

    # Band boundaries
    bands_a = load_band_boundaries(model_a)
    bands_b = load_band_boundaries(model_b)

    # Aggregate across all datasets
    # Key: (strong_model, feature_idx) -> stats
    feature_stats = defaultdict(lambda: {
        "n_rows": 0, "datasets": Counter(), "importance_sum": 0.0,
    })

    n_datasets = 0
    for f in sorted(pair_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "strong_wins" not in d.files or "selected_features" not in d.files:
            continue
        strong = str(d["strong_model"])
        sw = d["strong_wins"]
        sel = d["selected_features"]
        if sel.size == 0 or sw.sum() == 0:
            continue

        n_datasets += 1
        ds = f.stem

        for r in range(len(sw)):
            if not sw[r]:
                continue
            if sel.ndim == 2:
                for fi in sel[r]:
                    if fi >= 0:
                        key = (strong, int(fi))
                        feature_stats[key]["n_rows"] += 1
                        feature_stats[key]["datasets"][ds] += 1

    # Build ranked list per strong model
    results = {"pair": pair_name, "n_datasets": n_datasets, "models": {}}

    for model_key in [model_a, model_b]:
        model_disp = MODEL_KEY_TO_LABEL_KEY.get(model_key, model_key)
        unmatched = unmatched_a if model_key == model_a else unmatched_b
        bands = bands_a if model_key == model_a else bands_b

        # Filter to this model's features when it's strong
        model_features = []
        for (strong, fi), stats in feature_stats.items():
            if strong != model_key:
                continue

            is_unmatched = fi in unmatched
            band = get_band(fi, bands) if bands else -1

            # Get concept label
            entry = lookup.get(model_disp, {}).get(str(fi))
            group_id = None
            label = "(no label)"
            if entry:
                gid = entry.get("group_id")
                if gid is not None and str(gid) in groups:
                    group_id = gid
                    lbl = groups[str(gid)].get("label", "unlabeled")
                    if lbl != "unlabeled":
                        label = lbl

            # Get top datasets from concept analysis
            per_feature = concept_analysis.get("models", {}).get(
                model_disp, {}).get("per_feature", {})
            feat_data = per_feature.get(str(fi), {})
            top_datasets = [ds for ds, _ in feat_data.get("top_datasets", [])]

            model_features.append({
                "feature_idx": fi,
                "n_rows": stats["n_rows"],
                "n_datasets": len(stats["datasets"]),
                "top_ablation_datasets": stats["datasets"].most_common(5),
                "top_firing_datasets": top_datasets[:5],
                "is_unmatched": is_unmatched,
                "band": band,
                "group_id": group_id,
                "label": label,
            })

        # Sort by n_rows descending
        model_features.sort(key=lambda x: -x["n_rows"])

        # Summary stats
        n_strong_datasets = sum(1 for (s, _) in feature_stats if s == model_key)
        total_selections = sum(
            stats["n_rows"] for (s, _), stats in feature_stats.items()
            if s == model_key
        )

        # Band distribution
        band_dist = Counter()
        for feat in model_features:
            band_dist[feat["band"]] += feat["n_rows"]

        results["models"][model_key] = {
            "display_name": model_disp,
            "n_features_selected": len(model_features),
            "total_selections": total_selections,
            "band_distribution": dict(band_dist),
            "top_features": model_features[:50],  # Top 50
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Rank concepts that differentiate model pairs")
    parser.add_argument("--pair", default=None,
                        help="Single pair to process (default: all)")
    args = parser.parse_args()

    # Load shared data
    labels_data = json.loads(CONCEPT_LABELS_PATH.read_text())
    concept_analysis = json.loads(CONCEPT_ANALYSIS_PATH.read_text())

    # Find all pairs
    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted(d.name for d in ABLATION_DIR.iterdir() if d.is_dir())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair}")
        result = rank_pair(pair, labels_data, concept_analysis)
        if result is None:
            continue

        # Save
        pair_dir = OUTPUT_DIR / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        out_path = pair_dir / "concept_ranking.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Print summary
        for model_key, model_data in result["models"].items():
            disp = model_data["display_name"]
            n = model_data["n_features_selected"]
            total = model_data["total_selections"]
            logger.info(f"\n  {disp} (when strong): {n} features, "
                        f"{total} total row-selections")
            logger.info(f"  Band distribution: {model_data['band_distribution']}")
            logger.info(f"  Top 5 features:")
            for feat in model_data["top_features"][:5]:
                fi = feat["feature_idx"]
                nr = feat["n_rows"]
                nd = feat["n_datasets"]
                unm = "UNMATCHED" if feat["is_unmatched"] else "matched"
                lbl = feat["label"][:50]
                logger.info(f"    f{fi} (rows={nr}, ds={nd}, {unm}, "
                            f"band={feat['band']}): {lbl}")

        logger.info(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
