#!/usr/bin/env python3
"""
Top-down concept hierarchy: PyMFE meta-features → Matryoshka bands.

Dataset-level Pearson correlation between per-dataset mean SAE activations
and PyMFE meta-features, walked down through Matryoshka bands S1→S5 to find
where dataset-level signal exhausts.

Hierarchy:
  L0: 6 PyMFE super-categories (General, Statistical, Info-Theory, ...)
  L1: 145 individual PyMFE meta-features (one value per dataset)
  Bands S1→S5: which SAE features correlate with which L1 features?

Usage:
    python scripts/analyze_concept_hierarchy.py \
        --output output/concept_hierarchy.json \
        --pymfe-cache output/pymfe_tabarena_cache.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    find_common_datasets,
    sae_sweep_dir,
)
from scripts.compare_sae_architectures import compute_activations
from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from scripts.section43.universal_concepts import pool_embeddings_with_offsets


# ---------------------------------------------------------------------------
# L0/L1 taxonomy
# ---------------------------------------------------------------------------

def load_pymfe_taxonomy(
    path: Path = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Load PyMFE feature → category mapping.

    Returns:
        category_features: {category_name: [feature_names]}
        feature_category: {feature_name: category_name}
    """
    if path is None:
        path = PROJECT_ROOT / "config" / "pymfe_taxonomy.json"
    with open(path) as f:
        data = json.load(f)

    category_features = {}
    feature_category = {}
    for cat_name, cat_info in data["categories"].items():
        features = cat_info["features"]
        category_features[cat_name] = features
        for feat in features:
            feature_category[feat] = cat_name

    return category_features, feature_category


# ---------------------------------------------------------------------------
# PyMFE dataset matrix
# ---------------------------------------------------------------------------

def load_pymfe_dataset_matrix(
    cache_path: Path,
    datasets: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build (n_datasets, n_features) matrix from PyMFE cache.

    Rank-transforms columns with skewness > 2.0 (robust to outliers with n~38).
    Filters near-constant columns (std < 1e-8) to avoid NaN correlations.

    Returns:
        matrix: (n_datasets, n_features) with NaN-free numeric values
        feature_names: sorted list of feature names (after filtering)
    """
    with open(cache_path) as f:
        cache = json.load(f)

    # Collect all feature names across all datasets in the cache
    all_feature_names = set()
    for ds in datasets:
        if ds in cache:
            all_feature_names.update(cache[ds].keys())
    feature_names = sorted(all_feature_names)

    # Build raw matrix
    n_ds = len(datasets)
    n_feat = len(feature_names)
    matrix = np.full((n_ds, n_feat), np.nan)
    for i, ds in enumerate(datasets):
        if ds not in cache:
            continue
        for j, feat in enumerate(feature_names):
            val = cache[ds].get(feat)
            if val is not None and np.isfinite(val):
                matrix[i, j] = val

    # Fill NaN with column median (some features may be missing for some datasets)
    for j in range(n_feat):
        col = matrix[:, j]
        finite_mask = np.isfinite(col)
        if finite_mask.sum() > 0:
            median_val = np.nanmedian(col)
            col[~finite_mask] = median_val
        else:
            col[:] = 0.0

    # Rank-transform highly skewed columns (equivalent to Spearman)
    for j in range(n_feat):
        col = matrix[:, j]
        if np.std(col) > 1e-8:
            skew = stats.skew(col)
            if abs(skew) > 2.0:
                matrix[:, j] = stats.rankdata(col)

    # Filter near-constant columns
    col_std = np.std(matrix, axis=0)
    keep_mask = col_std > 1e-8
    matrix = matrix[:, keep_mask]
    feature_names = [f for f, k in zip(feature_names, keep_mask) if k]

    return matrix, feature_names


# ---------------------------------------------------------------------------
# Dataset-mean activations
# ---------------------------------------------------------------------------

def compute_dataset_mean_activations(
    activations: np.ndarray,
    offsets: Dict[str, Tuple[int, int]],
    datasets: List[str],
) -> np.ndarray:
    """Average SAE activations per dataset → (n_datasets, hidden_dim).

    This is the "de-broadcasting" step: collapses row-level activations
    into one vector per dataset, yielding pure dataset-level signal.
    """
    means = []
    for ds in datasets:
        if ds not in offsets:
            means.append(np.zeros(activations.shape[1]))
            continue
        start, end = offsets[ds]
        ds_acts = activations[start:end]
        means.append(ds_acts.mean(axis=0))
    return np.array(means)


# ---------------------------------------------------------------------------
# Band-level correlations
# ---------------------------------------------------------------------------

def compute_band_correlations(
    ds_means: np.ndarray,
    pymfe_matrix: np.ndarray,
    band_start: int,
    band_end: int,
    fdr_alpha: float = 0.05,
    alive_threshold: float = 0.001,
) -> Dict:
    """Pearson r between each alive SAE feature and each PyMFE feature.

    Args:
        ds_means: (n_datasets, hidden_dim) dataset-mean activations
        pymfe_matrix: (n_datasets, n_pymfe) meta-feature values
        band_start, band_end: SAE feature index range for this band
        fdr_alpha: FDR threshold for BH correction
        alive_threshold: minimum max activation to consider a feature alive

    Returns:
        dict with r_matrix, p_matrix, q_matrix, significant_mask,
        alive_indices (relative to band), n_alive
    """
    n_ds = ds_means.shape[0]
    n_pymfe = pymfe_matrix.shape[1]
    band_acts = ds_means[:, band_start:band_end]
    band_size = band_end - band_start

    if band_size == 0:
        return {
            "r_matrix": np.empty((0, n_pymfe)),
            "p_matrix": np.empty((0, n_pymfe)),
            "q_matrix": np.empty((0, n_pymfe)),
            "significant_mask": np.empty((0, n_pymfe), dtype=bool),
            "alive_indices": np.array([], dtype=int),
            "n_alive": 0,
        }

    # Filter to alive features (max > threshold across datasets)
    alive_mask = band_acts.max(axis=0) > alive_threshold
    alive_indices = np.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive == 0:
        return {
            "r_matrix": np.empty((0, n_pymfe)),
            "p_matrix": np.empty((0, n_pymfe)),
            "q_matrix": np.empty((0, n_pymfe)),
            "significant_mask": np.empty((0, n_pymfe), dtype=bool),
            "alive_indices": alive_indices,
            "n_alive": 0,
        }

    alive_acts = band_acts[:, alive_indices]  # (n_ds, n_alive)

    # Compute Pearson r for all (alive feature, pymfe feature) pairs
    r_matrix = np.zeros((n_alive, n_pymfe))
    p_matrix = np.ones((n_alive, n_pymfe))

    for i in range(n_alive):
        feat_vec = alive_acts[:, i]
        # Skip constant features
        if np.std(feat_vec) < 1e-8:
            continue
        for j in range(n_pymfe):
            pymfe_vec = pymfe_matrix[:, j]
            r, p = stats.pearsonr(feat_vec, pymfe_vec)
            if np.isfinite(r):
                r_matrix[i, j] = r
                p_matrix[i, j] = p

    # BH FDR correction across all tests
    all_p = p_matrix.ravel()
    n_tests = len(all_p)
    q_matrix = np.ones_like(p_matrix)

    if n_tests > 0:
        sorted_idx = np.argsort(all_p)
        sorted_p = all_p[sorted_idx]
        # BH procedure
        bh_threshold = np.arange(1, n_tests + 1) / n_tests * fdr_alpha
        # Find largest k where p(k) <= k/m * alpha
        reject = sorted_p <= bh_threshold
        if reject.any():
            max_reject = np.max(np.where(reject)[0])
            # All p-values up to and including max_reject are significant
            reject_mask = np.zeros(n_tests, dtype=bool)
            reject_mask[sorted_idx[:max_reject + 1]] = True

            # Compute q-values (adjusted p-values)
            q_vals = np.ones(n_tests)
            for k in range(n_tests - 1, -1, -1):
                idx = sorted_idx[k]
                q_vals[idx] = sorted_p[k] * n_tests / (k + 1)
            # Enforce monotonicity
            for k in range(n_tests - 2, -1, -1):
                idx = sorted_idx[k]
                idx_next = sorted_idx[k + 1]
                q_vals[idx] = min(q_vals[idx], q_vals[idx_next])
            q_vals = np.clip(q_vals, 0, 1)
            q_matrix = q_vals.reshape(p_matrix.shape)

    significant_mask = q_matrix < fdr_alpha

    return {
        "r_matrix": r_matrix,
        "p_matrix": p_matrix,
        "q_matrix": q_matrix,
        "significant_mask": significant_mask,
        "alive_indices": alive_indices,
        "n_alive": n_alive,
    }


# ---------------------------------------------------------------------------
# Aggregate by L0 category
# ---------------------------------------------------------------------------

def aggregate_by_category(
    band_corr: Dict,
    pymfe_names: List[str],
    category_features: Dict[str, List[str]],
    feature_category: Dict[str, str],
) -> Dict[str, Dict]:
    """Aggregate band-level correlations by L0 super-category.

    For each category: max |r|, mean R² of significant pairs, top L1 drivers,
    fraction of significant pairs.
    """
    r_matrix = band_corr["r_matrix"]
    sig_mask = band_corr["significant_mask"]
    n_alive = band_corr["n_alive"]

    if n_alive == 0:
        return {cat: {
            "max_abs_r": 0.0, "mean_r2_significant": 0.0,
            "n_significant_pairs": 0, "frac_significant": 0.0,
            "top_l1": [],
        } for cat in category_features}

    # Map pymfe column index to category
    pymfe_to_cat = {}
    for j, name in enumerate(pymfe_names):
        cat = feature_category.get(name)
        if cat is not None:
            pymfe_to_cat[j] = cat

    results = {}
    for cat_name, cat_feats in category_features.items():
        # Find column indices belonging to this category
        cat_cols = [j for j, name in enumerate(pymfe_names) if name in cat_feats]

        if not cat_cols:
            results[cat_name] = {
                "max_abs_r": 0.0, "mean_r2_significant": 0.0,
                "n_significant_pairs": 0, "frac_significant": 0.0,
                "top_l1": [],
            }
            continue

        cat_r = r_matrix[:, cat_cols]
        cat_sig = sig_mask[:, cat_cols]

        n_pairs = n_alive * len(cat_cols)
        n_sig = int(cat_sig.sum())

        # Max |r| across all (alive feature, category feature) pairs
        max_abs_r = float(np.max(np.abs(cat_r))) if cat_r.size > 0 else 0.0

        # Mean R² of significant pairs
        if n_sig > 0:
            sig_r_vals = cat_r[cat_sig]
            mean_r2_sig = float(np.mean(sig_r_vals ** 2))
        else:
            mean_r2_sig = 0.0

        # Top L1 drivers: for each pymfe feature in this category,
        # count how many alive SAE features it significantly correlates with
        top_l1 = []
        for jj, col_idx in enumerate(cat_cols):
            feat_name = pymfe_names[col_idx]
            n_correlated = int(sig_mask[:, col_idx].sum())
            max_r = float(np.max(np.abs(r_matrix[:, col_idx])))
            if n_correlated > 0:
                top_l1.append({
                    "name": feat_name,
                    "max_r": round(max_r, 4),
                    "n_correlated": n_correlated,
                })
        top_l1.sort(key=lambda x: x["max_r"], reverse=True)

        results[cat_name] = {
            "max_abs_r": round(max_abs_r, 4),
            "mean_r2_significant": round(mean_r2_sig, 4),
            "n_significant_pairs": n_sig,
            "frac_significant": round(n_sig / n_pairs, 4) if n_pairs > 0 else 0.0,
            "top_l1": top_l1[:10],
        }

    return results


# ---------------------------------------------------------------------------
# Signal exhaustion
# ---------------------------------------------------------------------------

def compute_signal_exhaustion(band_corr: Dict) -> Dict:
    """Fraction of alive features with no significant PyMFE correlation.

    Expected: S1 low exhaustion (~15%), S5 high exhaustion (~70-90%).
    """
    sig_mask = band_corr["significant_mask"]
    n_alive = band_corr["n_alive"]

    if n_alive == 0:
        return {
            "n_explained": 0,
            "n_unexplained": 0,
            "frac_unexplained": 0.0,
        }

    # A feature is "explained" if it has ANY significant correlation
    has_any_sig = sig_mask.any(axis=1)  # (n_alive,)
    n_explained = int(has_any_sig.sum())
    n_unexplained = n_alive - n_explained

    return {
        "n_explained": n_explained,
        "n_unexplained": n_unexplained,
        "frac_unexplained": round(n_unexplained / n_alive, 4),
    }


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def compute_variance_decomposition(
    band_corr: Dict,
    pymfe_names: List[str],
    feature_category: Dict[str, str],
) -> Dict[str, float]:
    """Decompose alive feature variance by best-correlate category.

    For each alive SAE feature:
      - Find its best PyMFE correlate (highest |r|)
      - R² = r² is the fraction of variance explained
      - Assign R² to the category of that best correlate
      - Remainder (1 - R²) is unexplained

    Average across all alive features to get per-category contributions
    that sum to 1.0 (categories + Unexplained).

    Returns:
        {category_name: mean_fraction, ..., "Unexplained": mean_fraction}
    """
    r_matrix = band_corr["r_matrix"]
    n_alive = band_corr["n_alive"]

    if n_alive == 0:
        return {"Unexplained": 1.0}

    # For each alive feature, find best correlate
    category_r2_sum = {}
    total_r2_sum = 0.0

    for i in range(n_alive):
        abs_r = np.abs(r_matrix[i])
        best_j = int(np.argmax(abs_r))
        best_r2 = float(abs_r[best_j] ** 2)

        feat_name = pymfe_names[best_j]
        cat = feature_category.get(feat_name, "Other")

        category_r2_sum[cat] = category_r2_sum.get(cat, 0.0) + best_r2
        total_r2_sum += best_r2

    result = {}
    for cat, r2_sum in category_r2_sum.items():
        result[cat] = round(r2_sum / n_alive, 6)
    result["Unexplained"] = round(1.0 - total_r2_sum / n_alive, 6)

    return result


# ---------------------------------------------------------------------------
# Per-model analysis pipeline
# ---------------------------------------------------------------------------

def get_matryoshka_bands(config) -> List[Tuple[str, int, int]]:
    """Extract Matryoshka band boundaries from SAE config.

    Returns list of (label, start, end) tuples.
    """
    mat_dims = config.matryoshka_dims
    if mat_dims is None:
        mat_dims = [config.hidden_dim]
    # Ensure hidden_dim is included
    if mat_dims[-1] < config.hidden_dim:
        mat_dims = list(mat_dims) + [config.hidden_dim]

    boundaries = [0] + list(mat_dims)
    bands = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        label = f"S{i + 1} [{start},{end})"
        bands.append((label, start, end))
    return bands


def analyze_model(
    model_name: str,
    sae_path: Path,
    emb_dir: Path,
    datasets: List[str],
    pymfe_matrix: np.ndarray,
    pymfe_names: List[str],
    category_features: Dict[str, List[str]],
    feature_category: Dict[str, str],
    max_per_dataset: int = 500,
    fdr_alpha: float = 0.05,
) -> Dict:
    """Full pipeline for one model: load SAE, pool, correlate, walk bands."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    # Load SAE
    model, config, _ = load_sae_checkpoint(sae_path)
    print(f"  SAE: {config.input_dim} → {config.hidden_dim} "
          f"(topk={config.topk})")

    # Pool embeddings with offset tracking
    pooled, offsets = pool_embeddings_with_offsets(
        emb_dir, datasets, max_per_dataset=max_per_dataset,
    )
    print(f"  Pooled: {pooled.shape[0]} rows × {pooled.shape[1]} dims")

    # Compute activations (BatchNorm handles normalization)
    activations = compute_activations(model, pooled)
    alive_total = (activations.max(axis=0) > 0.001).sum()
    print(f"  Activations: {activations.shape}, alive={alive_total}/{activations.shape[1]}")

    # Dataset-mean activations
    ds_means = compute_dataset_mean_activations(activations, offsets, datasets)
    print(f"  Dataset means: {ds_means.shape}")

    # Walk Matryoshka bands
    bands = get_matryoshka_bands(config)
    band_results = {}

    for label, start, end in bands:
        print(f"\n  Band {label}:")
        band_corr = compute_band_correlations(
            ds_means, pymfe_matrix, start, end, fdr_alpha=fdr_alpha,
        )
        print(f"    Alive: {band_corr['n_alive']}")

        # Aggregate by category
        cat_agg = aggregate_by_category(
            band_corr, pymfe_names, category_features, feature_category,
        )

        # Signal exhaustion
        exhaustion = compute_signal_exhaustion(band_corr)
        print(f"    Explained: {exhaustion['n_explained']}/{band_corr['n_alive']} "
              f"({1 - exhaustion['frac_unexplained']:.0%}), "
              f"Unexplained: {exhaustion['frac_unexplained']:.0%}")

        # Variance decomposition by category
        var_decomp = compute_variance_decomposition(
            band_corr, pymfe_names, feature_category,
        )
        explained_pct = (1 - var_decomp.get("Unexplained", 1.0)) * 100
        print(f"    Variance explained: {explained_pct:.1f}% "
              f"(top: {max((c for c in var_decomp if c != 'Unexplained'), key=lambda c: var_decomp[c], default='—')})")

        # Top per-feature detail
        top_features = _extract_top_features(
            band_corr, pymfe_names, feature_category, start,
        )

        # Print top categories
        for cat, info in sorted(cat_agg.items(), key=lambda x: x[1]["max_abs_r"], reverse=True):
            if info["n_significant_pairs"] > 0:
                print(f"    {cat}: max|r|={info['max_abs_r']:.3f}, "
                      f"{info['n_significant_pairs']} sig pairs")

        band_results[label] = {
            "n_alive": band_corr["n_alive"],
            "signal_exhaustion": exhaustion,
            "variance_decomposition": var_decomp,
            "categories": cat_agg,
            "top_features": top_features[:20],
        }

    return {
        "config": {
            "hidden_dim": config.hidden_dim,
            "input_dim": config.input_dim,
            "topk": config.topk,
            "matryoshka_dims": config.matryoshka_dims,
        },
        "bands": band_results,
    }


def _extract_top_features(
    band_corr: Dict,
    pymfe_names: List[str],
    feature_category: Dict[str, str],
    band_start: int,
) -> List[Dict]:
    """Extract top SAE features by max |r| with their best PyMFE correlate."""
    r_matrix = band_corr["r_matrix"]
    alive_indices = band_corr["alive_indices"]
    n_alive = band_corr["n_alive"]

    if n_alive == 0:
        return []

    features = []
    for i in range(n_alive):
        abs_r = np.abs(r_matrix[i])
        best_j = int(np.argmax(abs_r))
        max_r = float(abs_r[best_j])
        if max_r < 0.01:
            continue
        feat_name = pymfe_names[best_j]
        features.append({
            "idx": int(alive_indices[i]) + band_start,
            "max_r": round(max_r, 4),
            "top_pymfe": feat_name,
            "category": feature_category.get(feat_name, "Unknown"),
        })

    features.sort(key=lambda x: x["max_r"], reverse=True)
    return features


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------

def _band_number(label: str) -> str:
    """Extract band number from label like 'S1 [0,96)' → 'S1'."""
    return label.split()[0]


def _model_band_by_number(model_bands: Dict, band_num: str) -> Optional[Dict]:
    """Find band data for a given Sn number regardless of exact index range."""
    for label, data in model_bands.items():
        if _band_number(label) == band_num:
            return data
    return None


def compute_cross_model_summary(
    model_results: Dict[str, Dict],
    category_features: Dict[str, List[str]],
) -> Dict:
    """Aggregate cross-model patterns: signal exhaustion staircase, universal categories."""
    models = list(model_results.keys())

    # Collect unique band numbers (S1, S2, ...) across all models
    band_numbers = set()
    for m in models:
        for label in model_results[m]["bands"]:
            band_numbers.add(_band_number(label))
    band_numbers = sorted(band_numbers, key=lambda x: int(x[1:]))

    # Signal exhaustion staircase
    exhaustion_staircase = {}
    for bn in band_numbers:
        per_model = {}
        for m in models:
            band_data = _model_band_by_number(model_results[m]["bands"], bn)
            if band_data is not None:
                frac = band_data["signal_exhaustion"]["frac_unexplained"]
                per_model[m] = frac
        if per_model:
            vals = list(per_model.values())
            exhaustion_staircase[bn] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "per_model": per_model,
            }

    # Universal categories by band: significant in ALL models
    universal_categories = {}
    for bn in band_numbers:
        cats_all_models = None
        for m in models:
            band_data = _model_band_by_number(model_results[m]["bands"], bn)
            if band_data is None:
                continue
            cats = band_data["categories"]
            sig_cats = {c for c, info in cats.items() if info["n_significant_pairs"] > 0}
            if cats_all_models is None:
                cats_all_models = sig_cats
            else:
                cats_all_models &= sig_cats
        universal_categories[bn] = sorted(cats_all_models) if cats_all_models else []

    return {
        "signal_exhaustion_staircase": exhaustion_staircase,
        "universal_categories_by_band": universal_categories,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Top-down concept hierarchy: PyMFE → Matryoshka bands"
    )
    parser.add_argument("--output", type=str,
                        default="output/concept_hierarchy.json",
                        help="Output JSON path")
    parser.add_argument("--pymfe-cache", type=str,
                        default="output/pymfe_tabarena_cache.json",
                        help="PyMFE cache JSON")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to analyze (default: all)")
    parser.add_argument("--max-per-dataset", type=int, default=500,
                        help="Max samples per dataset")
    parser.add_argument("--fdr-alpha", type=float, default=0.05,
                        help="FDR threshold for BH correction")
    args = parser.parse_args()

    print("=" * 60)
    print("Top-Down Concept Hierarchy: PyMFE → Matryoshka Bands")
    print("=" * 60)

    # Load taxonomy
    category_features, feature_category = load_pymfe_taxonomy()
    n_categories = len(category_features)
    n_features = sum(len(v) for v in category_features.values())
    print(f"\nTaxonomy: {n_categories} categories, {n_features} features")

    # Resolve model paths
    base_sae = sae_sweep_dir()
    base_emb = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    model_configs = []
    emb_dirs = {}
    for display_name, sweep_dir, emb_dir_name in DEFAULT_MODELS:
        if args.models and display_name not in args.models:
            continue
        sae_path = base_sae / sweep_dir / "sae_matryoshka_archetypal_validated.pt"
        emb_dir = base_emb / emb_dir_name
        if not sae_path.exists():
            print(f"  Warning: SAE not found for {display_name}: {sae_path}")
            continue
        if not emb_dir.exists():
            print(f"  Warning: embeddings not found for {display_name}: {emb_dir}")
            continue
        model_configs.append((display_name, sae_path, emb_dir))
        emb_dirs[display_name] = emb_dir

    if not model_configs:
        print("Error: No models found")
        sys.exit(1)

    print(f"\nModels: {[m[0] for m in model_configs]}")

    # Find common datasets
    common_datasets = find_common_datasets(emb_dirs)

    # Load PyMFE dataset matrix
    pymfe_cache_path = PROJECT_ROOT / args.pymfe_cache
    pymfe_matrix, pymfe_names = load_pymfe_dataset_matrix(
        pymfe_cache_path, common_datasets,
    )
    print(f"\nPyMFE matrix: {pymfe_matrix.shape} "
          f"({len(common_datasets)} datasets × {len(pymfe_names)} features)")
    print(f"Statistical note: n={len(common_datasets)}, "
          f"min detectable |r| ≈ {2 / np.sqrt(len(common_datasets)):.2f} at α=0.05")

    # Analyze each model
    model_results = {}
    for display_name, sae_path, emb_dir in model_configs:
        model_results[display_name] = analyze_model(
            display_name, sae_path, emb_dir, common_datasets,
            pymfe_matrix, pymfe_names, category_features, feature_category,
            max_per_dataset=args.max_per_dataset, fdr_alpha=args.fdr_alpha,
        )

    # Cross-model summary
    cross_model = compute_cross_model_summary(model_results, category_features)

    # Print staircase summary
    print("\n" + "=" * 60)
    print("SIGNAL EXHAUSTION STAIRCASE")
    print("=" * 60)
    for band, info in cross_model["signal_exhaustion_staircase"].items():
        per = ", ".join(f"{m}={v:.0%}" for m, v in info["per_model"].items())
        print(f"  {band}: mean={info['mean']:.0%} ({per})")

    print("\nUNIVERSAL CATEGORIES BY BAND:")
    for band, cats in cross_model["universal_categories_by_band"].items():
        print(f"  {band}: {cats if cats else '(none)'}")

    # Save results
    output = {
        "metadata": {
            "n_models": len(model_configs),
            "n_datasets": len(common_datasets),
            "n_pymfe_features": len(pymfe_names),
            "fdr_alpha": args.fdr_alpha,
            "max_per_dataset": args.max_per_dataset,
            "datasets": common_datasets,
            "pymfe_features": pymfe_names,
        },
        "taxonomy": {cat: feats for cat, feats in category_features.items()},
        "models": model_results,
        "cross_model": cross_model,
    }

    output_clean = convert_keys_to_native(output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_clean, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
