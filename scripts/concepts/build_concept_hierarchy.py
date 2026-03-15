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
    python scripts/build_concept_hierarchy.py
    python scripts/build_concept_hierarchy.py --output output/concept_hierarchy_full.json
"""

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import SAE_FILENAME

# Default paths
DEFAULT_LABELS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels.json"
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "config" / "pymfe_taxonomy.json"
DEFAULT_LAYERS_PATH = PROJECT_ROOT / "config" / "optimal_extraction_layers.json"
DEFAULT_SAE_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round5"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "concept_hierarchy_full.json"

# Band names
BAND_NAMES = ["S1", "S2", "S3", "S4", "S5"]

# Category for probes not in PyMFE taxonomy (row-level meta-features)
ROW_LEVEL_CATEGORY = "Row-Level"

# Category for features with no meaningful probes
UNEXPLAINED_CATEGORY = "Unexplained"


def load_sae_configs(sae_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load SAE configs from checkpoints to get hidden_dim and matryoshka_dims.

    Returns:
        Dict mapping model_key -> {input_dim, hidden_dim, matryoshka_dims, topk}
    """
    configs = {}
    for ckpt_path in sorted(sae_dir.glob(f"*/{SAE_FILENAME}")):
        model_key = ckpt_path.parent.name
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        if hasattr(config, "__dict__"):
            config = config.__dict__
        configs[model_key] = {
            "input_dim": config.get("input_dim"),
            "hidden_dim": config.get("hidden_dim"),
            "matryoshka_dims": config.get("matryoshka_dims", []),
            "topk": config.get("topk"),
        }
    return configs


def build_probe_to_category(taxonomy: dict) -> Dict[str, str]:
    """Build mapping from probe name to PyMFE super-category.

    Probes not in any category are mapped to ROW_LEVEL_CATEGORY.
    """
    probe_to_cat = {}
    for cat_name, cat_info in taxonomy["categories"].items():
        for feat in cat_info["features"]:
            probe_to_cat[feat] = cat_name
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
        cat = probe_to_cat.get(probe_name, ROW_LEVEL_CATEGORY)
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
        "TabDPT": "tabdpt",
        "Mitra": "mitra",
        "HyperFast": "hyperfast",
        "Tabula-8B": "tabula8b",
    }
    return mapping.get(name, name.lower().replace("-", ""))


def build_hierarchy(
    labels: dict,
    taxonomy: dict,
    sae_configs: Dict[str, dict],
) -> dict:
    """Build the full concept hierarchy from cross-model labels.

    Args:
        labels: Loaded cross_model_concept_labels.json
        taxonomy: Loaded pymfe_taxonomy.json
        sae_configs: Per-model SAE config (hidden_dim, matryoshka_dims)

    Returns:
        Complete hierarchy dict with metadata, hierarchy, feature_index, model_comparison
    """
    probe_to_cat = build_probe_to_category(taxonomy)

    # Build model config metadata
    model_metadata = {}
    for display_name in ["TabPFN", "CARTE", "TabICL", "TabDPT", "Mitra", "HyperFast", "Tabula-8B"]:
        key = _normalize_model_name(display_name)
        if key in sae_configs:
            cfg = sae_configs[key]
            model_metadata[display_name] = {
                "hidden_dim": cfg["hidden_dim"],
                "input_dim": cfg["input_dim"],
                "bands": cfg["matryoshka_dims"],
                "topk": cfg["topk"],
            }

    # Collect all categories that will appear
    all_categories = set(taxonomy["categories"].keys())
    all_categories.add(ROW_LEVEL_CATEGORY)
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
    model_groups: Dict[str, Set[str]] = defaultdict(set)  # model -> set of group_ids

    # --- Process matched concept groups ---
    concept_groups = labels.get("concept_groups", {})
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
            model_key = _normalize_model_name(model_name)
            if model_key in sae_configs:
                mat_dims = sae_configs[model_key]["matryoshka_dims"]
                band = assign_band(feat_idx, mat_dims)
                band_votes[band] += 1

        # Use majority vote for band assignment (cross-model groups span models)
        consensus_band = max(band_votes, key=band_votes.get) if band_votes else "S5"

        # Build group entry
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

        # Update feature index
        for model_name, feat_idx in members:
            feature_index[model_name][str(feat_idx)] = {
                "band": consensus_band,
                "category": category,
                "group_id": group_id,
                "label": label,
                "matched": True,
            }
            model_groups[model_name].add(group_id)

    # --- Process unmatched features (explained + unexplained) ---
    unmatched = labels.get("unmatched_features", {})
    for match_status in ["explained", "unexplained"]:
        model_features = unmatched.get(match_status, {})
        if not isinstance(model_features, dict):
            continue
        for model_name, features in model_features.items():
            if not isinstance(features, dict):
                continue
            model_key = _normalize_model_name(model_name)
            mat_dims = sae_configs.get(model_key, {}).get("matryoshka_dims", [])

            for feat_id_str, feat_info in features.items():
                feat_idx = int(feat_id_str)
                band = assign_band(feat_idx, mat_dims) if mat_dims else "S5"

                top_probes = feat_info.get("top_probes", [])
                r2 = feat_info.get("r2", 0.0)
                label = feat_info.get("label", "")

                category = assign_category(top_probes, probe_to_cat, r2=r2)

                unmatched_key = f"{model_name}:{feat_id_str}"
                hierarchy[band][category]["unmatched"][unmatched_key] = {
                    "label": label,
                    "r2": r2,
                    "top_probes": top_probes,
                }

                feature_index[model_name][feat_id_str] = {
                    "band": band,
                    "category": category,
                    "group_id": None,
                    "label": label,
                    "matched": False,
                }

    # --- Build model_comparison ---
    model_comparison = _build_model_comparison(concept_groups, model_groups)

    # --- Count features ---
    n_features = sum(len(feats) for feats in feature_index.values())
    n_groups = len(concept_groups)

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
            "pymfe_categories": sorted(all_categories),
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
            # Count features in unique groups belonging to A
            n_features = 0
            category_counts: Dict[str, int] = defaultdict(int)
            for gid in unique_groups:
                group = concept_groups.get(gid, {})
                members = group.get("members", [])
                a_features = [f for m, f in members if m == model_a]
                n_features += len(a_features)

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
        "--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH,
        help="Path to pymfe_taxonomy.json",
    )
    parser.add_argument(
        "--layers", type=Path, default=DEFAULT_LAYERS_PATH,
        help="Path to optimal_extraction_layers.json",
    )
    parser.add_argument(
        "--sae-dir", type=Path, default=DEFAULT_SAE_DIR,
        help="Path to SAE checkpoint directory",
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

    logger.info("Loading SAE configs from %s", args.sae_dir)
    sae_configs = load_sae_configs(args.sae_dir)
    logger.info("Found SAE configs for: %s", sorted(sae_configs.keys()))

    # Build hierarchy
    logger.info("Building hierarchy...")
    result = build_hierarchy(labels, taxonomy, sae_configs)

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
