#!/usr/bin/env python3
"""
SAE concept analysis: probe regression, contrastive examples, and dataset profiles.

For each alive SAE feature, fits Ridge regression predicting its activation vector
from the row-level meta-feature matrix. Reports per-feature R², per-band summaries,
identifies "interpolated" concepts, and collects contrastive examples (top-activating
rows vs nearest non-activating rows in embedding space) for LLM labeling.

Usage:
    python scripts/analyze_concept_regression.py --device cuda

    # Single model only
    python scripts/analyze_concept_regression.py --models TabPFN
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT

from scripts.sae.compare_sae_architectures import (
    META_NAMES,
    compute_activations,
    compute_basic_metrics,
    compute_feature_effects,
    get_train_test_split,
    meta_features_to_array,
)
from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    DEFAULT_SAE_ROUND,
    SAE_FILENAME,
    collect_meta_for_datasets,
    find_common_datasets,
    pool_embeddings_for_datasets,
    sae_sweep_dir,
)
from scripts.sae.analyze_sae_concepts_deep import (
    NumpyEncoder,
    compute_concept_coverage,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from scripts.matching.utils import (
    compute_alive_mask,
    load_test_embeddings,
)

from analysis.sparse_autoencoder import SparseAutoencoder


def compute_activations_from_normalized(
    model: SparseAutoencoder, embeddings: np.ndarray
) -> np.ndarray:
    """Compute SAE activations from already-normalized embeddings."""
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def get_test_split_indices(n_rows: int, seed: int = 42) -> np.ndarray:
    """Get test-split row indices matching build_sae_training_data.py logic.

    The SAE training pipeline: subsample to 500 (seed=42), shuffle, split 70/30.
    Since embedding files already have <=500 rows, this just shuffles and takes
    the last 30%.

    Returns:
        Array of original row indices that went into the test split.
    """
    rng = np.random.RandomState(seed)
    if n_rows > 500:
        idx = rng.choice(n_rows, 500, replace=False)
    else:
        idx = rng.permutation(n_rows)
    n_train = int(len(idx) * 0.7)
    return idx[n_train:]


def augment_meta_with_pymfe(
    meta_array: np.ndarray,
    boundaries: np.ndarray,
    loaded_datasets: List[str],
    pymfe_cache_path: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Broadcast PyMFE dataset-level features to per-row meta_array.

    Each dataset has one PyMFE vector which is repeated for all rows from
    that dataset, using boundary indices to identify dataset membership.

    Args:
        meta_array: (n_samples, n_probes) existing row-level meta-features.
        boundaries: (n_datasets + 1,) cumulative sample counts.
        loaded_datasets: dataset names in order.
        pymfe_cache_path: path to JSON cache from compute_pymfe_cache.py.

    Returns:
        augmented: (n_samples, n_probes + n_pymfe) concatenated matrix.
        pymfe_names: list of PyMFE feature names appended.
    """
    import json

    with open(pymfe_cache_path) as f:
        cache = json.load(f)

    # Only use datasets present in cache
    available = [ds for ds in loaded_datasets if ds in cache]
    if not available:
        print("  Warning: no datasets found in PyMFE cache, skipping augmentation")
        return meta_array, []

    # Common features: intersection across available datasets in cache
    common_features = sorted(
        set.intersection(*(set(cache[ds].keys()) for ds in available))
    )
    if not common_features:
        print("  Warning: no common PyMFE features across datasets, skipping")
        return meta_array, []

    pymfe_cols = np.zeros((len(meta_array), len(common_features)))
    for i, ds_name in enumerate(loaded_datasets):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        if ds_name in cache:
            for j, feat_name in enumerate(common_features):
                pymfe_cols[start:end, j] = cache[ds_name].get(feat_name, 0.0)
        # else: rows get zeros (dataset not in cache)

    augmented = np.hstack([meta_array, pymfe_cols])
    print(f"  PyMFE augmentation: {len(common_features)} features from "
          f"{len(available)}/{len(loaded_datasets)} datasets")

    return augmented, common_features


def regress_features_on_probes(
    activations: np.ndarray,
    meta_array: np.ndarray,
    alive_indices: List[int],
    alpha: float = 1.0,
    probe_names: Optional[List[str]] = None,
) -> Dict[int, Dict]:
    """
    For each alive SAE feature, fit Ridge(alpha) predicting activation from probes.

    Args:
        probe_names: names for each column in meta_array. Defaults to META_NAMES
            (52 row-level probes). When PyMFE features are appended, pass the
            extended list (META_NAMES + pymfe_names).

    Returns:
        {feat_idx: {r2, coefficients, top_probes: [(name, coeff, rank)]}}
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if probe_names is None:
        probe_names = list(META_NAMES)

    n_probes = meta_array.shape[1]
    assert len(probe_names) == n_probes, (
        f"probe_names length ({len(probe_names)}) != meta_array columns ({n_probes})"
    )

    # Standardize meta-features (important for coefficient interpretation)
    scaler = StandardScaler()
    X = scaler.fit_transform(meta_array)

    results = {}
    for feat_idx in alive_indices:
        y = activations[:, feat_idx]

        # Skip dead features
        if y.max() < 1e-6:
            continue

        model = Ridge(alpha=alpha)
        model.fit(X, y)
        r2 = model.score(X, y)

        # Top probes by absolute standardized coefficient
        coeffs = model.coef_
        abs_coeffs = np.abs(coeffs)
        top_indices = np.argsort(-abs_coeffs)[:5]
        top_probes = [
            (probe_names[i], float(coeffs[i]), int(rank + 1))
            for rank, i in enumerate(top_indices)
        ]

        results[feat_idx] = {
            'r2': float(r2),
            'coefficients': {probe_names[i]: float(coeffs[i]) for i in range(n_probes)},
            'top_probes': top_probes,
        }

    return results


def regress_features_per_dataset(
    per_ds_acts: Dict[str, np.ndarray],
    per_ds_meta: Dict[str, np.ndarray],
    alive_indices: List[int],
    alpha: float = 1.0,
    probe_names: Optional[List[str]] = None,
    min_rows: int = 30,
) -> Dict[int, Dict]:
    """Per-dataset Ridge regression, averaged across datasets.

    For each alive feature, fits Ridge within each dataset independently
    (both activations and meta-features are on the same per-dataset scale),
    then reports mean R² and mean coefficients across datasets.

    This avoids the pooling problem where per-dataset StandardScaler
    normalization puts each dataset in its own coordinate frame.

    Args:
        per_ds_acts: {dataset_name: (n_rows, hidden_dim)} activations.
        per_ds_meta: {dataset_name: (n_rows, n_probes)} meta-features.
        alive_indices: feature indices to regress.
        min_rows: skip datasets with fewer rows than this.

    Returns:
        {feat_idx: {r2, r2_std, n_datasets, coefficients, top_probes}}
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if probe_names is None:
        probe_names = list(META_NAMES)

    # Common datasets with enough rows
    common_ds = [
        ds for ds in per_ds_acts
        if ds in per_ds_meta and per_ds_meta[ds].shape[0] >= min_rows
    ]
    n_probes = per_ds_meta[common_ds[0]].shape[1] if common_ds else 0

    results = {}
    for feat_idx in alive_indices:
        ds_r2s = []
        ds_coeffs = []

        for ds in common_ds:
            y = per_ds_acts[ds][:, feat_idx]
            if y.max() < 1e-6:
                continue

            X = per_ds_meta[ds]
            # Standardize within dataset
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_s, y)
            r2 = ridge.score(X_s, y)

            ds_r2s.append(r2)
            ds_coeffs.append(ridge.coef_)

        if not ds_r2s:
            # Feature is alive (on training data) but has no test activation
            # above 1e-6 in any dataset. Still emit it with R²=0 so it can
            # receive contrastive examples and labels downstream.
            results[feat_idx] = {
                'r2': 0.0,
                'r2_std': 0.0,
                'n_datasets': 0,
                'coefficients': {probe_names[i]: 0.0 for i in range(n_probes)},
                'top_probes': [],
            }
            continue

        mean_r2 = float(np.mean(ds_r2s))
        mean_coeffs = np.mean(ds_coeffs, axis=0)
        abs_coeffs = np.abs(mean_coeffs)
        top_indices = np.argsort(-abs_coeffs)[:5]
        top_probes = [
            (probe_names[i], float(mean_coeffs[i]), int(rank + 1))
            for rank, i in enumerate(top_indices)
        ]

        results[feat_idx] = {
            'r2': mean_r2,
            'r2_std': float(np.std(ds_r2s)),
            'n_datasets': len(ds_r2s),
            'coefficients': {
                probe_names[i]: float(mean_coeffs[i]) for i in range(n_probes)
            },
            'top_probes': top_probes,
        }

    return results


def compute_dataset_activation_profile(
    per_ds_acts: Dict[str, np.ndarray],
    alive_indices: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute per-dataset mean activation for each alive feature.

    Returns:
        {feat_idx: {dataset_name: mean_activation}}
    """
    profiles = {}
    for feat_idx in alive_indices:
        ds_means = {}
        for ds, acts in per_ds_acts.items():
            ds_means[ds] = float(acts[:, feat_idx].mean())
        profiles[feat_idx] = ds_means
    return profiles


def collect_contrastive_examples(
    per_ds_acts: Dict[str, np.ndarray],
    per_ds_embs: Dict[str, np.ndarray],
    per_ds_raw: Dict[str, 'pd.DataFrame'],
    alive_indices: List[int],
    k: int = 10,
    max_cols: int = 20,
) -> Dict[int, Dict]:
    """Collect top-k activating rows and k nearest non-activating rows per feature.

    For each alive feature:
    1. Pool activations and embeddings across datasets (with dataset labels).
    2. Select top-k rows by activation value.
    3. From non-activating rows (activation=0), find the k nearest neighbors
       to the top-k centroid in embedding space. These contrastive examples
       delineate the activation boundary better than random non-activating rows.

    Each example includes the raw tabular row data (not probe summaries), so the
    LLM labeler can reason about column-level patterns directly.

    Args:
        per_ds_embs: {dataset_name: (n_rows, emb_dim)} model embeddings.
        per_ds_raw: {dataset_name: DataFrame} of raw test-split tabular data.
        max_cols: truncate raw rows to this many columns (highest-variance first).

    Returns:
        {feat_idx: {
            'top': [{dataset, row_idx, activation, raw: {col: value}}],
            'contrast': [{dataset, row_idx, activation, raw: {col: value}}],
        }}
    """
    import torch

    # Pool across datasets with provenance
    ds_names = sorted(ds for ds in per_ds_acts if ds in per_ds_embs)
    all_acts_list = []
    all_embs_list = []
    row_provenance = []  # (dataset, local_row_idx)
    for ds in ds_names:
        acts = per_ds_acts[ds]
        embs = per_ds_embs[ds]
        n = min(acts.shape[0], embs.shape[0])
        all_acts_list.append(acts[:n])
        all_embs_list.append(embs[:n])
        for i in range(n):
            row_provenance.append((ds, i))

    all_acts = np.concatenate(all_acts_list, axis=0)
    all_embs = np.concatenate(all_embs_list, axis=0)

    # Normalize embeddings for cosine-distance neighbor search
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb_tensor = torch.tensor(all_embs / norms, dtype=torch.float32)

    # Pre-select top-variance columns per dataset for output truncation
    ds_top_cols = {}
    for ds in ds_names:
        if ds not in per_ds_raw:
            continue
        df = per_ds_raw[ds]
        if len(df.columns) <= max_cols:
            ds_top_cols[ds] = list(df.columns)
        else:
            # Keep highest-variance numeric columns + all categorical
            numeric = df.select_dtypes(include=[np.number])
            categorical = df.select_dtypes(exclude=[np.number])
            cat_cols = list(categorical.columns)
            n_numeric_slots = max_cols - len(cat_cols)
            if n_numeric_slots > 0 and len(numeric.columns) > 0:
                var = numeric.var()
                top_num = var.nlargest(n_numeric_slots).index.tolist()
            else:
                top_num = []
            ds_top_cols[ds] = top_num + cat_cols

    def _row_to_dict(idx, activation):
        ds, local_idx = row_provenance[idx]
        raw = {}
        if ds in per_ds_raw and local_idx < len(per_ds_raw[ds]):
            row = per_ds_raw[ds].iloc[local_idx]
            cols = ds_top_cols.get(ds, list(per_ds_raw[ds].columns)[:max_cols])
            for col in cols:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, float) and np.isfinite(val):
                    raw[col] = round(val, 4)
                else:
                    raw[col] = val
        return {
            'dataset': ds,
            'row_idx': int(local_idx),
            'activation': round(float(activation), 4),
            'raw': raw,
        }

    results = {}
    for feat_idx in alive_indices:
        col = all_acts[:, feat_idx]
        if col.max() < 1e-6:
            # Feature alive on train but not on test — emit empty examples
            # so downstream scripts know the feature exists
            results[feat_idx] = {'top': [], 'contrast': []}
            continue

        # Top-k activating rows
        top_indices = np.argsort(-col)[:k]
        top_examples = [_row_to_dict(i, col[i]) for i in top_indices]

        # Non-activating rows (activation == 0)
        non_active_mask = col < 1e-6
        non_active_indices = np.where(non_active_mask)[0]

        if len(non_active_indices) == 0:
            # Feature fires on every row — use bottom-k instead
            bottom_indices = np.argsort(col)[:k]
            contrast_examples = [_row_to_dict(i, col[i]) for i in bottom_indices]
        elif len(non_active_indices) <= k:
            contrast_examples = [
                _row_to_dict(i, col[i]) for i in non_active_indices
            ]
        else:
            # Centroid of top-k in embedding space
            top_centroid = emb_tensor[top_indices].mean(dim=0, keepdim=True)
            top_centroid = top_centroid / top_centroid.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Cosine distances from non-activating rows to top-k centroid
            non_active_embs = emb_tensor[non_active_indices]
            dists = 1.0 - (non_active_embs @ top_centroid.T).squeeze(1)

            # k nearest non-activating rows to the top-k centroid
            nearest_k = dists.topk(k, largest=False).indices.numpy()
            contrast_indices = non_active_indices[nearest_k]
            contrast_examples = [
                _row_to_dict(i, col[i]) for i in contrast_indices
            ]

        results[feat_idx] = {
            'top': top_examples,
            'contrast': contrast_examples,
        }

    return results


def compute_band_regression_summary(
    regression_results: Dict[int, Dict],
    config,
) -> Dict[str, Dict]:
    """
    Compute per-Matryoshka-band regression summaries.

    Returns:
        {band_label: {mean_r2, frac_explained, frac_unexplained, n_features}}
    """
    mat_dims = getattr(config, 'matryoshka_dims', None)
    if not mat_dims:
        # Non-Matryoshka: single band covering everything
        r2_vals = [r['r2'] for r in regression_results.values()]
        if not r2_vals:
            return {}
        return {
            'all': {
                'mean_r2': float(np.mean(r2_vals)),
                'frac_explained': float(np.mean([r > 0.3 for r in r2_vals])),
                'frac_unexplained': float(np.mean([r < 0.1 for r in r2_vals])),
                'n_features': len(r2_vals),
            }
        }

    summaries = {}
    band_edges = [0] + list(mat_dims)

    for i in range(len(band_edges) - 1):
        start = band_edges[i]
        end = band_edges[i + 1]
        label = f"S{i+1} [0:{end}]"

        band_r2 = [
            r['r2'] for fid, r in regression_results.items()
            if start <= fid < end
        ]

        if not band_r2:
            summaries[label] = {
                'mean_r2': 0.0, 'frac_explained': 0.0,
                'frac_unexplained': 1.0, 'n_features': 0,
            }
            continue

        summaries[label] = {
            'mean_r2': float(np.mean(band_r2)),
            'frac_explained': float(np.mean([r > 0.3 for r in band_r2])),
            'frac_unexplained': float(np.mean([r < 0.1 for r in band_r2])),
            'n_features': len(band_r2),
        }

    return summaries


def identify_interpolated_concepts(
    regression_results: Dict[int, Dict],
    feature_effects: Dict[int, Dict],
    d_threshold: float = 0.5,
    r2_threshold: float = 0.3,
) -> List[Dict]:
    """
    Find features that are poorly captured by any single probe (low max |d|)
    but well-predicted by the combination (high R²). These are "interpolated"
    concepts that sit between our probe axes.

    Returns list of {feat_idx, r2, max_d, top_probes}.
    """
    interpolated = []
    for feat_idx, reg in regression_results.items():
        if feat_idx not in feature_effects:
            continue

        effects = feature_effects[feat_idx].get('effect_sizes', {})
        max_d = max(abs(d) for d in effects.values()) if effects else 0.0

        if max_d < d_threshold and reg['r2'] > r2_threshold:
            interpolated.append({
                'feat_idx': int(feat_idx),
                'r2': reg['r2'],
                'max_d': float(max_d),
                'top_probes': reg['top_probes'],
            })

    return sorted(interpolated, key=lambda x: -x['r2'])


def analyze_model_regression(
    model_name: str,
    sae_dir: str,
    emb_model: str,
    datasets: List[str],
    meta_array: np.ndarray = None,
    max_per_dataset: int = 500,
    alpha: float = 1.0,
    probe_names: Optional[List[str]] = None,
    test_embs: Optional[Dict[str, np.ndarray]] = None,
    alive_mask: Optional[np.ndarray] = None,
    per_ds_meta: Optional[Dict[str, np.ndarray]] = None,
    per_ds_raw: Optional[Dict[str, 'pd.DataFrame']] = None,
) -> Dict:
    """Full regression analysis for one model.

    Args:
        test_embs: Pre-loaded test-split embeddings per dataset (already normalized).
            If provided, uses per-dataset regression (no pooling).
        alive_mask: Pre-computed alive mask from training data.
        per_ds_meta: Per-dataset meta-feature arrays {ds: (n_rows, n_probes)}.
            Required when test_embs is provided.
        per_ds_raw: Per-dataset raw DataFrames for contrastive examples.
        meta_array: Pooled meta-features (legacy path only).
    """
    sweep = sae_sweep_dir()
    sae_path = sweep / sae_dir / SAE_FILENAME
    emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / emb_model

    if not sae_path.exists():
        print(f"  SKIP {model_name}: {sae_path} not found")
        return {}

    print(f"\n{'─' * 60}")
    print(f"Regression analysis: {model_name}")
    print(f"{'─' * 60}")

    # Load SAE
    model, config, _ = load_sae_checkpoint(sae_path)
    print(f"  SAE: hidden={config.hidden_dim}, topk={config.topk}")

    if test_embs is not None:
        # Per-dataset regression (correct for per-dataset StandardScaler)
        test_datasets = [ds for ds in datasets if ds in test_embs]
        per_ds_acts = {}
        total_rows = 0
        for ds in test_datasets:
            acts = compute_activations_from_normalized(model, test_embs[ds])
            per_ds_acts[ds] = acts
            total_rows += acts.shape[0]
        print(f"  Test activations: {len(per_ds_acts)} datasets, {total_rows} rows")

        # Use pre-computed alive mask or compute from pooled test
        if alive_mask is not None:
            alive = list(np.where(alive_mask)[0])
        else:
            all_acts = np.concatenate(list(per_ds_acts.values()), axis=0)
            metrics = compute_basic_metrics(all_acts, config)
            alive = metrics['alive_indices']
        print(f"  Alive: {len(alive)}/{config.hidden_dim}")

        # Per-dataset Ridge regression
        reg_results = regress_features_per_dataset(
            per_ds_acts, per_ds_meta, alive, alpha=alpha,
            probe_names=probe_names,
        )
        print(f"  Regressed: {len(reg_results)} features (per-dataset avg)")

        # Cohen's d: compute per-dataset and average
        all_acts_pooled = np.concatenate(
            [per_ds_acts[ds] for ds in test_datasets], axis=0
        )
        all_meta_pooled = np.concatenate(
            [per_ds_meta[ds] for ds in test_datasets if ds in per_ds_meta],
            axis=0,
        )
        feat_effects = compute_feature_effects(
            all_acts_pooled, all_meta_pooled, alive
        )
    else:
        # Legacy path: load raw embeddings, pool, regress
        pooled = pool_embeddings_for_datasets(emb_dir, datasets, max_per_dataset)
        print(f"  Embeddings: {pooled.shape}")

        train_ds, _ = get_train_test_split(datasets)
        train_embs_list = []
        for ds in train_ds:
            path = emb_dir / f"tabarena_{ds}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            emb = data['embeddings'].astype(np.float32)
            if len(emb) > max_per_dataset:
                np.random.seed(42)
                idx = np.random.choice(len(emb), max_per_dataset, replace=False)
                emb = emb[idx]
            train_embs_list.append(emb)
        train_pooled = np.concatenate(train_embs_list)
        train_std = train_pooled.std(axis=0, keepdims=True)
        train_std[train_std < 1e-8] = 1.0
        train_norm = train_pooled / train_std
        train_mean = train_norm.mean(axis=0, keepdims=True)

        acts = compute_activations(model, pooled, train_std, train_mean)
        print(f"  Activations: {acts.shape}")

        metrics = compute_basic_metrics(acts, config)
        alive = metrics['alive_indices']
        print(f"  Alive: {len(alive)}/{config.hidden_dim}")

        feat_effects = compute_feature_effects(acts, meta_array, alive)

        reg_results = regress_features_on_probes(
            acts, meta_array, alive, alpha=alpha, probe_names=probe_names,
        )
        print(f"  Regressed: {len(reg_results)} features")

    r2_vals = [r['r2'] for r in reg_results.values()]
    print(f"  Mean R²: {np.mean(r2_vals):.3f}")
    print(f"  Explained (R²>0.3): {np.mean([r > 0.3 for r in r2_vals]):.0%}")
    print(f"  Unexplained (R²<0.1): {np.mean([r < 0.1 for r in r2_vals]):.0%}")

    # Band summary
    band_summary = compute_band_regression_summary(reg_results, config)
    for band, info in band_summary.items():
        print(f"  {band}: R²={info['mean_r2']:.3f}, "
              f"explained={info['frac_explained']:.0%}, "
              f"n={info['n_features']}")

    # Interpolated concepts
    interpolated = identify_interpolated_concepts(reg_results, feat_effects)
    print(f"  Interpolated concepts: {len(interpolated)}")
    for ic in interpolated[:5]:
        probes = ", ".join(f"{p[0]}({p[1]:.2f})" for p in ic['top_probes'][:3])
        print(f"    feat {ic['feat_idx']}: R²={ic['r2']:.3f}, max_d={ic['max_d']:.2f}, "
              f"top: {probes}")

    # Per-probe R² contribution: average absolute coefficient across features
    all_names = probe_names if probe_names is not None else list(META_NAMES)
    probe_importance = {}
    for name in all_names:
        abs_coeffs = [abs(r['coefficients'][name]) for r in reg_results.values()]
        probe_importance[name] = float(np.mean(abs_coeffs)) if abs_coeffs else 0.0

    # Dataset activation profiles and contrastive examples (for labeling)
    ds_profiles = {}
    contrastive = {}
    if test_embs is not None:
        ds_profiles = compute_dataset_activation_profile(per_ds_acts, alive)
        contrastive = collect_contrastive_examples(
            per_ds_acts, test_embs, per_ds_raw, alive, k=10,
        )
        print(f"  Contrastive examples: {len(contrastive)} features")

    result = {
        'model_name': model_name,
        'n_features_regressed': len(reg_results),
        'mean_r2': float(np.mean(r2_vals)),
        'frac_explained': float(np.mean([r > 0.3 for r in r2_vals])),
        'frac_unexplained': float(np.mean([r < 0.1 for r in r2_vals])),
        'band_summary': band_summary,
        'interpolated_concepts': interpolated,
        'probe_importance': probe_importance,
        'per_feature': {
            int(fid): {'r2': r['r2'], 'top_probes': r['top_probes']}
            for fid, r in reg_results.items()
        },
    }

    if test_embs is not None:
        for fid, r in reg_results.items():
            result['per_feature'][int(fid)]['r2_std'] = r.get('r2_std', 0.0)
            result['per_feature'][int(fid)]['n_datasets'] = r.get('n_datasets', 0)

    # Attach top/bottom datasets per feature (for labeling context)
    if ds_profiles:
        for fid, ds_means in ds_profiles.items():
            if int(fid) not in result['per_feature']:
                continue
            sorted_ds = sorted(ds_means.items(), key=lambda x: -x[1])
            result['per_feature'][int(fid)]['top_datasets'] = [
                (ds, round(v, 4)) for ds, v in sorted_ds[:5]
            ]
            result['per_feature'][int(fid)]['bottom_datasets'] = [
                (ds, round(v, 4)) for ds, v in sorted_ds[-5:]
            ]

    # Attach contrastive examples (top-activating + nearest non-activating)
    if contrastive:
        for fid, examples in contrastive.items():
            if int(fid) in result['per_feature']:
                result['per_feature'][int(fid)]['examples'] = examples

    return result


def collect_test_meta_features(
    datasets: List[str],
    test_embs: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, 'pd.DataFrame'], List[str]]:
    """Compute row-level meta-features on test-split rows.

    Uses row_indices saved in the SAE corpus NPZ files to select the exact
    same rows from the original dataset. This ensures meta-features align
    with SAE activations for Ridge regression.

    Returns:
        per_ds_meta: {dataset_name: (n_test_rows, n_meta_features)}
        per_ds_raw: {dataset_name: DataFrame of raw test-split rows}
        loaded_datasets: datasets that loaded successfully
    """
    import pandas as pd
    from data.extended_loader import load_tabarena_dataset
    from scripts.sae.analyze_sae_concepts_deep import (
        compute_column_stats,
        compute_row_meta_features,
    )
    from scripts.matching.utils import SAE_DATA_DIR

    # Load row indices from the test NPZ
    test_npz_candidates = sorted(SAE_DATA_DIR.glob("*_sae_test.npz"))
    if not test_npz_candidates:
        raise FileNotFoundError(f"No test NPZ in {SAE_DATA_DIR}")

    # Use first model's test file to get row indices (same rows for all models
    # that share the dataset — indices come from the shared splits)
    test_data = np.load(test_npz_candidates[0], allow_pickle=True)
    if "row_indices" not in test_data:
        raise ValueError(
            "Test NPZ missing row_indices. Rebuild corpus with updated "
            "06_build_sae_training_data.py"
        )
    all_row_indices = test_data["row_indices"]
    samples = test_data["samples_per_dataset"]

    # Unpool row indices per dataset
    ds_row_indices = {}
    offset = 0
    for ds_name, count in samples:
        ds_name = str(ds_name)
        count = int(count)
        ds_row_indices[ds_name] = all_row_indices[offset:offset + count]
        offset += count

    per_ds_meta = {}
    per_ds_raw = {}
    loaded = []

    for ds_name in datasets:
        if ds_name not in ds_row_indices:
            continue
        if ds_name not in test_embs:
            continue

        row_idx = ds_row_indices[ds_name]
        n_test = len(test_embs[ds_name])
        if len(row_idx) != n_test:
            print(f"    Skipping {ds_name}: row_indices ({len(row_idx)}) != "
                  f"test embs ({n_test})")
            continue

        try:
            X, y, _ = load_tabarena_dataset(ds_name)
            df = X if hasattr(X, 'iloc') else pd.DataFrame(X)
            df = df.iloc[row_idx].reset_index(drop=True)
            y_sub = y[row_idx] if y is not None else None

            numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)
            meta_features = compute_row_meta_features(
                df, y_sub, numeric_cols, categorical_cols, col_stats, dataset_stats
            )
            rows = np.array([meta_features_to_array(m) for m in meta_features])
            per_ds_meta[ds_name] = rows
            per_ds_raw[ds_name] = df
            loaded.append(ds_name)
            print(f"    {ds_name}: {rows.shape[0]} test rows, {len(df.columns)} cols")
        except Exception as e:
            print(f"    Skipping {ds_name}: {e}")

    return per_ds_meta, per_ds_raw, loaded


def main():
    parser = argparse.ArgumentParser(description="Regression analysis of SAE concepts")
    parser.add_argument("--output", type=str,
                        default=f"output/sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json")
    parser.add_argument("--models", nargs='+', default=None,
                        help="Model names to analyze (default: all 8)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    parser.add_argument("--max-per-dataset", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pymfe-cache", type=str,
        default="output/pymfe_tabarena_cache.json",
        help="PyMFE cache JSON — dataset-level context for labeling.",
    )
    parser.add_argument(
        "--use-test-split", action="store_true", default=True,
        help="Use test-split embeddings and training-derived alive masks "
             "(default: True).",
    )
    parser.add_argument(
        "--no-test-split", action="store_false", dest="use_test_split",
        help="Legacy: use raw pooled embeddings instead of test split.",
    )
    args = parser.parse_args()

    # Select models
    models = DEFAULT_MODELS
    if args.models:
        models = [(n, s, e) for n, s, e in DEFAULT_MODELS if n in args.models]
        if not models:
            print(f"No models matched {args.models}. Available: "
                  f"{[n for n, _, _ in DEFAULT_MODELS]}")
            sys.exit(1)

    # Pre-load test embeddings and alive masks if using test split
    model_test_embs = {}
    model_alive_masks = {}
    if args.use_test_split:
        print("\nLoading test-split embeddings and computing alive masks...")
        sweep = sae_sweep_dir()
        for display_name, model_key, emb_key in models:
            sae_path = sweep / model_key / SAE_FILENAME
            if not sae_path.exists():
                continue
            try:
                test_embs = load_test_embeddings(model_key)
                sae_model, _, _ = load_sae_checkpoint(sae_path)
                # Alive mask from test data — consistent with matching (step 1).
                # Every alive feature has signal in the data we analyze.
                alive = compute_alive_mask(sae_model, test_embs)
                model_test_embs[display_name] = test_embs
                model_alive_masks[display_name] = alive
                print(f"  {display_name}: {len(test_embs)} datasets, "
                      f"alive={alive.sum()}/{len(alive)}")
            except FileNotFoundError as e:
                print(f"  Skipping {display_name}: {e}")

    # Find common datasets
    if args.use_test_split and model_test_embs:
        all_ds_sets = [set(embs.keys()) for embs in model_test_embs.values()]
        datasets = sorted(set.intersection(*all_ds_sets))
    else:
        emb_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
        emb_dirs = {}
        for display_name, _, emb_model in models:
            d = emb_base / emb_model
            if d.exists():
                emb_dirs[display_name] = d
        datasets = find_common_datasets(emb_dirs)
    print(f"\nUsing {len(datasets)} common datasets")

    # Collect meta-features
    per_ds_meta = None
    per_ds_raw = None
    meta_array = None
    boundaries = None
    if args.use_test_split:
        # Per-dataset meta-features using saved row indices from corpus
        first_model_embs = list(model_test_embs.values())[0]
        per_ds_meta, per_ds_raw, loaded = collect_test_meta_features(
            datasets, first_model_embs
        )
        total_rows = sum(v.shape[0] for v in per_ds_meta.values())
        print(f"Test meta-features: {len(loaded)} datasets, {total_rows} rows, "
              f"{per_ds_meta[loaded[0]].shape[1]} probes")
    else:
        emb_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
        first_emb_dir = emb_base / models[0][2]
        meta_array, loaded, boundaries = collect_meta_for_datasets(
            datasets, first_emb_dir, args.max_per_dataset
        )
        print(f"Meta-features: {meta_array.shape}"
              f" ({len(loaded)} datasets, {len(boundaries)-1} boundaries)")

    # Use only datasets that loaded successfully for meta-features
    datasets = loaded

    # Load PyMFE cache for dataset-level context (used in labeling, not regression)
    pymfe_data = None
    if args.pymfe_cache:
        with open(args.pymfe_cache) as f:
            pymfe_data = json.load(f)
        n_pymfe_ds = sum(1 for ds in datasets if ds in pymfe_data)
        print(f"PyMFE cache: {n_pymfe_ds}/{len(datasets)} datasets (context for labeling)")

    probe_names = list(META_NAMES)
    alpha = args.alpha

    # Analyze each model
    all_results = {}
    for display_name, sae_dir, emb_model in models:
        result = analyze_model_regression(
            display_name, sae_dir, emb_model,
            datasets, meta_array, args.max_per_dataset, alpha,
            probe_names=probe_names,
            test_embs=model_test_embs.get(display_name),
            alive_mask=model_alive_masks.get(display_name),
            per_ds_meta=per_ds_meta,
            per_ds_raw=per_ds_raw,
        )
        if result:
            all_results[display_name] = result

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'REGRESSION SUMMARY':^70}")
    print(f"{'=' * 70}")
    print(f"{'Model':15s} {'N feat':>8s} {'Mean R²':>10s} {'Explained':>10s} {'Unexplained':>12s}")
    print("-" * 55)
    for name, r in all_results.items():
        print(f"{name:15s} {r['n_features_regressed']:8d} {r['mean_r2']:10.3f} "
              f"{r['frac_explained']:10.0%} {r['frac_unexplained']:12.0%}")

    # Add metadata
    metadata = {
        'n_row_probes': len(META_NAMES),
        'alpha': alpha,
        'regression_method': 'per_dataset' if args.use_test_split else 'pooled',
        'split': 'test' if args.use_test_split else 'all',
        'n_datasets': len(datasets),
        'datasets': datasets,
    }

    output_data = {'metadata': metadata, 'models': all_results}

    # Attach PyMFE dataset descriptions as labeling context
    if pymfe_data:
        ds_context = {}
        for ds in datasets:
            if ds in pymfe_data:
                ds_context[ds] = pymfe_data[ds]
        output_data['dataset_context'] = ds_context
        metadata['pymfe_cache'] = args.pymfe_cache
        metadata['n_pymfe_datasets'] = len(ds_context)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert_keys_to_native(output_data), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
