#!/usr/bin/env python3
"""Build a traversable concept hierarchy from cross-model SAE feature matching.

Combines Matryoshka scale bands, PyMFE super-categories, cross-model concept groups,
and individual features into a single JSON file for ablation experiments.

Hierarchy levels:
    L0: Root
     └─ L1: Matryoshka Scale Band (S1..S5)
         └─ L2: PyMFE Super-Category (General, Statistical, Info-Theory, etc.)
             └─ L3: Concept Group (cross-model) or Unmatched Feature
                 └─ L4: Individual Feature (model, feat_idx)

Usage:
    python scripts/concepts/build_concept_hierarchy.py
    python scripts/concepts/build_concept_hierarchy.py --verbose
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from scripts._project_root import PROJECT_ROOT

from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND

# Default paths
DEFAULT_LABELS_PATH = PROJECT_ROOT / "output" / f"cross_model_concept_labels_round{DEFAULT_SAE_ROUND}.json"
DEFAULT_CONCEPTS_PATH = PROJECT_ROOT / "output" / f"sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json"
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "config" / "pymfe_taxonomy.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "concept_hierarchy_full.json"

# Band names
BAND_NAMES = ["S1", "S2", "S3", "S4", "S5"]

# Category for features with no meaningful probes
UNEXPLAINED_CATEGORY = "Unexplained"

# Row-level probe categories (used when probes aren't in PyMFE taxonomy)
ROW_LEVEL_CATEGORIES = {
    "Numeric-Distribution": [
        "numeric_mean_zscore", "numeric_std", "numeric_range", "numeric_iqr",
        "numeric_kurtosis", "numeric_skewness", "numeric_max_zscore",
        "numeric_min_zscore", "log_magnitude_mean", "log_magnitude_std",
    ],
    "Sparsity-Structure": [
        "frac_zeros", "frac_integers", "frac_round_tens", "frac_very_large",
        "frac_very_small", "frac_negative", "n_distinct_values",
        "frac_negative_outliers", "frac_positive_outliers",
    ],
    "Categorical": [
        "categorical_entropy", "categorical_modal_frac", "categorical_rarity",
        "n_rare_categories",
    ],
    "Geometry-Neighborhood": [
        "centroid_distance", "nearest_neighbor_dist", "local_density",
        "local_intrinsic_dim", "local_clustering", "hub_score",
        "pca_pc1", "pca_pc2",
    ],
    "Classification-Boundary": [
        "borderline", "knn_class_ratio", "linear_boundary_dist",
        "fisher_ratio", "target_is_minority",
    ],
    "Information-Content": [
        "row_entropy", "row_surprise", "row_uniformity", "mi_contribution",
    ],
}

# Fallback for probes not in any category
UNCATEGORIZED = "Uncategorized"


def load_sae_configs_from_concepts(concepts_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract SAE configs from concept analysis JSON (no .pt checkpoints needed).

    Parses band_summary labels like 'S1 [0:48]' to recover matryoshka_dims.

    Returns:
        Dict mapping display_name -> {input_dim, hidden_dim, matryoshka_dims, topk}
    """
    with open(concepts_path) as f:
        concepts = json.load(f)

    # TopK values per model (from round 10 sweep selection)
    TOPK = {
        "TabPFN": 256, "TabICL": 32, "TabICL-v2": 32, "Mitra": 32,
        "CARTE": 16, "HyperFast": 16, "TabDPT": 16, "Tabula-8B": 64,
    }

    configs = {}
    for model_name, model_data in concepts["models"].items():
        band_summary = model_data.get("band_summary", {})
        dims = []
        for band_label in sorted(band_summary.keys()):
            match = re.search(r"\[(\d+):(\d+)\]", band_label)
            if match:
                dims.append(int(match.group(2)))
        if not dims:
            logger.warning("No band boundaries for %s, skipping", model_name)
            continue
        hidden_dim = dims[-1]
        # Input dim = hidden_dim / expansion (all round 10 are 4x except Tabula-8B 1x)
        expansion = 1 if model_name == "Tabula-8B" else 4
        configs[model_name] = {
            "input_dim": hidden_dim // expansion,
            "hidden_dim": hidden_dim,
            "matryoshka_dims": dims,
            "topk": TOPK.get(model_name),
        }
    return configs


def build_probe_to_category(taxonomy: dict) -> Dict[str, str]:
    """Build mapping from probe name to category.

    Covers both PyMFE dataset-level features and row-level probes.
    Probes not in any category are mapped to UNCATEGORIZED.
    """
    probe_to_cat = {}
    # PyMFE dataset-level features
    for cat_name, cat_info in taxonomy["categories"].items():
        for feat in cat_info["features"]:
            probe_to_cat[feat] = cat_name
    # Row-level probes
    for cat_name, probes in ROW_LEVEL_CATEGORIES.items():
        for probe in probes:
            probe_to_cat[probe] = cat_name
    return probe_to_cat


def assign_band(feat_idx: int, matryoshka_dims: List[int]) -> str:
    """Map feature index to Matryoshka scale band (S1-S5).

    Band boundaries are defined by matryoshka_dims:
        S1: [0, dims[0])          -- most global/universal
        S2: [dims[0], dims[1])
        S3: [dims[1], dims[2])
        S4: [dims[2], dims[3])
        S5: [dims[3], dims[4])    -- most specific
    """
    for i, boundary in enumerate(matryoshka_dims):
        if feat_idx < boundary:
            return BAND_NAMES[i]
    # Should not happen if feat_idx < hidden_dim
    return BAND_NAMES[-1]


def assign_category(
    top_probes: List[list],
    probe_to_cat: Dict[str, str],
    r2: float = 0.0,
    r2_threshold: float = 0.01,
) -> str:
    """Assign a feature to a PyMFE super-category via its dominant top probe.

    Args:
        top_probes: List of [probe_name, coeff_or_count, ...] from concept labels
        probe_to_cat: Mapping from probe name to category
        r2: R-squared of the concept regression
        r2_threshold: Below this, the feature is unexplained

    Returns:
        Category name (e.g., "Statistical", "Complexity", "Row-Level", "Unexplained")
    """
    if r2 < r2_threshold or not top_probes:
        return UNEXPLAINED_CATEGORY

    # Count category votes weighted by absolute coefficient
    category_votes: Dict[str, float] = defaultdict(float)
    for probe_info in top_probes:
        probe_name = probe_info[0]
        # Coefficient is last element (index 1 for unmatched, index 2 for groups)
        coeff = abs(probe_info[-1]) if len(probe_info) > 1 else 0.0
        cat = probe_to_cat.get(probe_name, UNCATEGORIZED)
        category_votes[cat] += coeff

    if not category_votes:
        return UNEXPLAINED_CATEGORY

    return max(category_votes, key=category_votes.get)


def _normalize_model_name(name: str) -> str:
    """Map display name to SAE checkpoint key (lowercase)."""
    mapping = {
        "TabPFN": "tabpfn",
        "CARTE": "carte",
        "TabICL": "tabicl",
        "TabICL-v2": "tabicl_v2",
        "TabDPT": "tabdpt",
        "Mitra": "mitra",
        "HyperFast": "hyperfast",
        "Tabula-8B": "tabula8b",
    }
    return mapping.get(name, name.lower().replace("-", ""))


def _deduplicate_groups(labels: dict) -> Tuple[dict, Set[Tuple[str, int]]]:
    """Deduplicate round 10 labels: mega-group 0 was split via Leiden into
    sub-groups, but the labeling pipeline duplicated those sub-groups.

    Returns:
        (filtered_groups, orphaned_features) where filtered_groups excludes
        the mega-group and its duplicates, and orphaned_features is a set of
        (model_name, feat_idx) tuples from group 0 not covered by sub-groups.
    """
    concept_groups = labels.get("concept_groups", {})

    # Detect mega-group (split_into > 0)
    mega_id = None
    n_splits = 0
    for gid, g in concept_groups.items():
        split_into = g.get("split_into", 0)
        if isinstance(split_into, int) and split_into > 0:
            mega_id = gid
            n_splits = split_into
            break

    if mega_id is None:
        # No mega-group deduplication needed
        return concept_groups, set()

    mega_group = concept_groups[mega_id]
    mega_members = set(tuple(x) for x in mega_group["members"])

    # Find the first contiguous run of Leiden sub-groups (members overlap mega_members)
    first_split_start = int(mega_id) + 1
    first_copy_ids = set()
    covered = set()
    for gid_int in range(first_split_start, first_split_start + n_splits * 3):
        gid = str(gid_int)
        if gid not in concept_groups:
            continue
        members = set(tuple(x) for x in concept_groups[gid]["members"])
        if members & mega_members:
            first_copy_ids.add(gid)
            covered |= (members & mega_members)
        if len(first_copy_ids) >= n_splits:
            break

    # Second copy starts right after the first
    second_copy_start = max(int(x) for x in first_copy_ids) + 1 if first_copy_ids else len(concept_groups)
    second_copy_ids = set()
    for gid_int in range(second_copy_start, second_copy_start + n_splits * 3):
        gid = str(gid_int)
        if gid not in concept_groups:
            continue
        members = set(tuple(x) for x in concept_groups[gid]["members"])
        if members & mega_members:
            second_copy_ids.add(gid)
        if len(second_copy_ids) >= n_splits:
            break

    # Prefer whichever copy has more labels; fall back to first copy
    def _n_labeled(ids):
        return sum(
            1 for gid in ids
            if concept_groups[gid].get("label", "unlabeled") != "unlabeled"
        )

    keep_ids = first_copy_ids
    drop_ids = second_copy_ids
    if second_copy_ids and _n_labeled(second_copy_ids) > _n_labeled(first_copy_ids):
        keep_ids, drop_ids = drop_ids, first_copy_ids

    # Orphaned = mega-group members not in any Leiden sub-group
    orphaned = mega_members - covered

    # Build filtered groups: skip mega-group and the duplicate copy
    skip = {mega_id} | drop_ids
    filtered = {gid: g for gid, g in concept_groups.items() if gid not in skip}

    logger.info(
        "Dedup: mega-group %s (%d members) -> %d sub-groups kept, "
        "%d duplicates dropped, %d orphaned features",
        mega_id, len(mega_members), len(keep_ids), len(drop_ids), len(orphaned),
    )
    return filtered, orphaned


def build_hierarchy(
    labels: dict,
    taxonomy: dict,
    sae_configs: Dict[str, dict],
    concepts_data: Optional[dict] = None,
) -> dict:
    """Build the full concept hierarchy from cross-model labels.

    Args:
        labels: Loaded cross_model_concept_labels_round10.json
        taxonomy: Loaded pymfe_taxonomy.json
        sae_configs: Per-model SAE config keyed by display name
        concepts_data: Optional concept analysis JSON for orphaned feature probes

    Returns:
        Complete hierarchy dict with metadata, hierarchy, feature_index, model_comparison
    """
    probe_to_cat = build_probe_to_category(taxonomy)

    # Build model config metadata
    model_metadata = {}
    for display_name in sorted(sae_configs.keys()):
        cfg = sae_configs[display_name]
        model_metadata[display_name] = {
            "hidden_dim": cfg["hidden_dim"],
            "input_dim": cfg["input_dim"],
            "bands": cfg["matryoshka_dims"],
            "topk": cfg["topk"],
        }

    # Collect all categories that will appear
    all_categories = set(taxonomy["categories"].keys())
    all_categories.update(ROW_LEVEL_CATEGORIES.keys())
    all_categories.add(UNCATEGORIZED)
    all_categories.add(UNEXPLAINED_CATEGORY)

    # Initialize hierarchy structure
    hierarchy: Dict[str, Dict[str, dict]] = {}
    for band in BAND_NAMES:
        hierarchy[band] = {}
        for cat in sorted(all_categories):
            hierarchy[band][cat] = {"groups": {}, "unmatched": {}}

    # Feature index: flat lookup for (model, feat_idx) -> path
    feature_index: Dict[str, Dict[str, dict]] = defaultdict(dict)

    # Track which groups each model participates in (for model_comparison)
    model_groups: Dict[str, Set[str]] = defaultdict(set)

    # Deduplicate groups (handles mega-group split duplication)
    concept_groups, orphaned_features = _deduplicate_groups(labels)

    # --- Process concept groups ---
    for group_id, group in concept_groups.items():
        members = group.get("members", [])
        if not members:
            continue

        top_probes = group.get("top_probes", [])
        mean_r2 = group.get("mean_r2", 0.0)
        label = group.get("label", "")
        n_models = group.get("n_models", 0)
        tier = group.get("tier", 0)

        # Determine category from probes
        category = assign_category(top_probes, probe_to_cat, r2=mean_r2)

        # Group features by model and find consensus band
        features_by_model: Dict[str, List[int]] = defaultdict(list)
        band_votes: Dict[str, int] = defaultdict(int)

        for model_name, feat_idx in members:
            features_by_model[model_name].append(feat_idx)
            if model_name in sae_configs:
                mat_dims = sae_configs[model_name]["matryoshka_dims"]
                band = assign_band(feat_idx, mat_dims)
                band_votes[band] += 1

        consensus_band = max(band_votes, key=band_votes.get) if band_votes else "S5"

        group_entry = {
            "label": label,
            "n_models": n_models,
            "mean_r2": mean_r2,
            "top_probes": top_probes,
            "dominant_category": category,
            "tier": tier,
            "features": dict(features_by_model),
        }

        hierarchy[consensus_band][category]["groups"][group_id] = group_entry

        for model_name, feat_idx in members:
            feature_index[model_name][str(feat_idx)] = {
                "band": consensus_band,
                "category": category,
                "group_id": group_id,
                "label": label,
                "matched": True,
            }
            model_groups[model_name].add(group_id)

    # --- Process orphaned features (from mega-group, not in any sub-group) ---
    for model_name, feat_idx in orphaned_features:
        mat_dims = sae_configs.get(model_name, {}).get("matryoshka_dims", [])
        band = assign_band(feat_idx, mat_dims) if mat_dims else "S5"

        # Try to get probe data from concept analysis
        top_probes = []
        r2 = 0.0
        if concepts_data:
            per_feat = concepts_data.get("models", {}).get(
                model_name, {}
            ).get("per_feature", {}).get(str(feat_idx), {})
            top_probes = per_feat.get("top_probes", [])
            r2 = per_feat.get("r2", 0.0)

        category = assign_category(top_probes, probe_to_cat, r2=r2)
        unmatched_key = f"{model_name}:{feat_idx}"
        hierarchy[band][category]["unmatched"][unmatched_key] = {
            "label": "",
            "r2": r2,
            "top_probes": top_probes,
        }
        feature_index[model_name][str(feat_idx)] = {
            "band": band,
            "category": category,
            "group_id": None,
            "label": "",
            "matched": False,
        }

    # --- Build model_comparison ---
    model_comparison = _build_model_comparison(concept_groups, model_groups)

    # --- Prune empty categories ---
    for band in BAND_NAMES:
        empty = [
            cat for cat, data in hierarchy[band].items()
            if not data["groups"] and not data["unmatched"]
        ]
        for cat in empty:
            del hierarchy[band][cat]

    # --- Count features ---
    n_features = sum(len(feats) for feats in feature_index.values())
    n_groups = len(concept_groups)
    active_categories = set()
    for band_data in hierarchy.values():
        active_categories.update(band_data.keys())

    # --- Assemble final output ---
    result = {
        "metadata": {
            "n_models": len(model_metadata),
            "n_features": n_features,
            "n_groups": n_groups,
            "matryoshka_bands": {
                "S1": "h/16",
                "S2": "h/8",
                "S3": "h/4",
                "S4": "h/2",
                "S5": "h",
            },
            "categories": sorted(active_categories),
            "models": model_metadata,
        },
        "hierarchy": hierarchy,
        "feature_index": dict(feature_index),
        "model_comparison": model_comparison,
    }

    return result


def _build_model_comparison(
    concept_groups: dict,
    model_groups: Dict[str, Set[str]],
) -> dict:
    """Pre-compute unique/shared concept sets for every model pair.

    Args:
        concept_groups: The concept_groups dict from labels
        model_groups: Mapping from model_name -> set of group_ids

    Returns:
        Dict with 'unique_to' and 'shared' keys
    """
    models = sorted(model_groups.keys())

    unique_to: Dict[str, Dict[str, dict]] = {}
    shared: Dict[str, dict] = {}

    for model_a in models:
        unique_to[model_a] = {}
        groups_a = model_groups[model_a]

        for model_b in models:
            if model_a == model_b:
                continue
            groups_b = model_groups[model_b]

            # Groups unique to A (A has it, B doesn't)
            unique_groups = groups_a - groups_b
            n_features = 0
            for gid in unique_groups:
                group = concept_groups.get(gid, {})
                members = group.get("members", [])
                n_features += sum(1 for m, f in members if m == model_a)

            unique_to[model_a][f"vs_{model_b}"] = {
                "groups": sorted(unique_groups, key=lambda x: int(x)),
                "n_features": n_features,
                "n_groups": len(unique_groups),
            }

    # Shared groups for each pair
    for model_a, model_b in combinations(models, 2):
        key = f"{model_a}__{model_b}"
        shared_groups = model_groups[model_a] & model_groups[model_b]
        n_features_a = 0
        n_features_b = 0
        for gid in shared_groups:
            group = concept_groups.get(gid, {})
            members = group.get("members", [])
            n_features_a += sum(1 for m, f in members if m == model_a)
            n_features_b += sum(1 for m, f in members if m == model_b)

        shared[key] = {
            "groups": sorted(shared_groups, key=lambda x: int(x)),
            "n_features_a": n_features_a,
            "n_features_b": n_features_b,
            "n_groups": len(shared_groups),
        }

    return {"unique_to": unique_to, "shared": shared}


def traverse(
    hierarchy: dict,
    band: Optional[str] = None,
    category: Optional[str] = None,
    model: Optional[str] = None,
    matched_only: bool = False,
) -> Dict[str, Any]:
    """Query/filter the hierarchy.

    Args:
        hierarchy: The 'hierarchy' sub-dict from the full output
        band: Filter to specific band (e.g., "S1")
        category: Filter to specific category (e.g., "Statistical")
        model: Filter to groups containing this model
        matched_only: If True, exclude unmatched features

    Returns:
        Filtered hierarchy dict with same structure
    """
    bands = [band] if band else BAND_NAMES
    result = {}

    for b in bands:
        if b not in hierarchy:
            continue
        result[b] = {}
        categories = [category] if category else list(hierarchy[b].keys())

        for cat in categories:
            if cat not in hierarchy[b]:
                continue
            cat_data = hierarchy[b][cat]

            # Filter groups
            filtered_groups = {}
            for gid, group in cat_data.get("groups", {}).items():
                if model and model not in group.get("features", {}):
                    continue
                filtered_groups[gid] = group

            # Filter unmatched
            filtered_unmatched = {}
            if not matched_only:
                for key, feat in cat_data.get("unmatched", {}).items():
                    if model:
                        feat_model = key.split(":")[0]
                        if feat_model != model:
                            continue
                    filtered_unmatched[key] = feat

            if filtered_groups or filtered_unmatched:
                result[b][cat] = {
                    "groups": filtered_groups,
                    "unmatched": filtered_unmatched,
                }

    return result


def build_feature_index(hierarchy: dict) -> Dict[str, Dict[str, dict]]:
    """Build flat lookup from hierarchy: (model, feat_idx) -> path.

    This is a convenience function if the hierarchy was loaded without
    the pre-built feature_index.
    """
    index: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for band, categories in hierarchy.items():
        for cat, cat_data in categories.items():
            for gid, group in cat_data.get("groups", {}).items():
                for model_name, feat_list in group.get("features", {}).items():
                    for feat_idx in feat_list:
                        index[model_name][str(feat_idx)] = {
                            "band": band,
                            "category": cat,
                            "group_id": gid,
                            "label": group.get("label", ""),
                            "matched": True,
                        }
            for key, feat in cat_data.get("unmatched", {}).items():
                parts = key.split(":")
                if len(parts) == 2:
                    model_name, feat_id = parts
                    index[model_name][feat_id] = {
                        "band": band,
                        "category": cat,
                        "group_id": None,
                        "label": feat.get("label", ""),
                        "matched": False,
                    }

    return dict(index)


def print_summary(result: dict) -> None:
    """Print a summary of the concept hierarchy."""
    meta = result["metadata"]
    print(f"Models: {meta['n_models']}")
    print(f"Features: {meta['n_features']}")
    print(f"Concept groups: {meta['n_groups']}")
    print()

    # Band distribution
    print("Band distribution (groups / unmatched):")
    for band in BAND_NAMES:
        n_groups = 0
        n_unmatched = 0
        for cat_data in result["hierarchy"].get(band, {}).values():
            n_groups += len(cat_data.get("groups", {}))
            n_unmatched += len(cat_data.get("unmatched", {}))
        print(f"  {band}: {n_groups:4d} groups, {n_unmatched:4d} unmatched")
    print()

    # Category distribution
    print("Category distribution (groups / unmatched):")
    cat_counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for band in BAND_NAMES:
        for cat, cat_data in result["hierarchy"].get(band, {}).items():
            g, u = cat_counts[cat]
            cat_counts[cat] = (
                g + len(cat_data.get("groups", {})),
                u + len(cat_data.get("unmatched", {})),
            )
    for cat in sorted(cat_counts):
        g, u = cat_counts[cat]
        print(f"  {cat:20s}: {g:4d} groups, {u:4d} unmatched")
    print()

    # Model comparison summary
    comp = result.get("model_comparison", {})
    unique = comp.get("unique_to", {})
    shared_pairs = comp.get("shared", {})
    print("Model comparison (unique concepts):")
    for model_a in sorted(unique.keys()):
        for vs_key, info in sorted(unique[model_a].items()):
            model_b = vs_key.replace("vs_", "")
            print(
                f"  {model_a} unique vs {model_b}: "
                f"{info['n_groups']} groups, {info['n_features']} features"
            )
    print()
    print(f"Shared pairs: {len(shared_pairs)}")
    for pair_key in sorted(shared_pairs):
        info = shared_pairs[pair_key]
        print(f"  {pair_key}: {info['n_groups']} shared groups")


def main():
    parser = argparse.ArgumentParser(description="Build concept hierarchy")
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_LABELS_PATH,
        help="Path to cross_model_concept_labels.json",
    )
    parser.add_argument(
        "--concepts", type=Path, default=DEFAULT_CONCEPTS_PATH,
        help="Path to sae_concept_analysis.json (for SAE configs and orphan probes)",
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH,
        help="Path to pymfe_taxonomy.json",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Output path for hierarchy JSON",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load inputs
    logger.info("Loading labels from %s", args.labels)
    with open(args.labels) as f:
        labels = json.load(f)

    logger.info("Loading taxonomy from %s", args.taxonomy)
    with open(args.taxonomy) as f:
        taxonomy = json.load(f)

    logger.info("Loading SAE configs from %s", args.concepts)
    sae_configs = load_sae_configs_from_concepts(args.concepts)
    logger.info("Found SAE configs for: %s", sorted(sae_configs.keys()))

    # Load concept analysis for orphaned feature probes
    with open(args.concepts) as f:
        concepts_data = json.load(f)

    # Build hierarchy
    logger.info("Building hierarchy...")
    result = build_hierarchy(labels, taxonomy, sae_configs, concepts_data)

    # Print summary
    print_summary(result)

    # Validate
    expected_features = labels["summary"]["total_alive_features"]
    actual_features = result["metadata"]["n_features"]
    if actual_features != expected_features:
        logger.warning(
            "Feature count mismatch: expected %d, got %d",
            expected_features, actual_features,
        )
    else:
        logger.info("All %d features accounted for", actual_features)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved hierarchy to %s (%.1f MB)", args.output,
                args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
