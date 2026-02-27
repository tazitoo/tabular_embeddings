#!/usr/bin/env python3
"""
Regression analysis of SAE concepts: how much activation variance do probes explain?

For each alive SAE feature, fits Ridge regression predicting its activation vector
from the 52-dim meta-feature matrix. Reports per-feature R², per-band summaries,
and identifies "interpolated" concepts (low single-probe d, high regression R²).

Usage:
    python scripts/analyze_concept_regression.py \
        --output output/concept_regression.json --device cuda

    # Single model only
    python scripts/analyze_concept_regression.py \
        --models TabPFN --output output/concept_regression_tabpfn.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_architectures import (
    META_NAMES,
    compute_activations,
    compute_basic_metrics,
    compute_feature_effects,
    get_train_test_split,
    meta_features_to_array,
)
from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    collect_meta_for_datasets,
    find_common_datasets,
    pool_embeddings_for_datasets,
    sae_sweep_dir,
)
from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    compute_concept_coverage,
    convert_keys_to_native,
    load_sae_checkpoint,
)


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
    meta_array: np.ndarray,
    max_per_dataset: int = 500,
    alpha: float = 1.0,
    probe_names: Optional[List[str]] = None,
) -> Dict:
    """Full regression analysis for one model."""
    sweep = sae_sweep_dir()
    sae_path = sweep / sae_dir / "sae_matryoshka_archetypal_validated.pt"
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

    # Pool embeddings
    pooled = pool_embeddings_for_datasets(emb_dir, datasets, max_per_dataset)
    print(f"  Embeddings: {pooled.shape}")

    # Normalize
    train_ds, _ = get_train_test_split(datasets)
    train_embs = []
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
        train_embs.append(emb)
    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)

    # Compute activations
    acts = compute_activations(model, pooled, train_std, train_mean)
    print(f"  Activations: {acts.shape}")

    # Align
    n_min = min(len(meta_array), len(acts))
    acts = acts[:n_min]
    meta = meta_array[:n_min]

    # Basic metrics for alive indices
    metrics = compute_basic_metrics(acts, config)
    alive = metrics['alive_indices']
    print(f"  Alive: {len(alive)}/{config.hidden_dim}")

    # Cohen's d feature effects (needed for interpolated concept detection)
    feat_effects = compute_feature_effects(acts, meta, alive)

    # Ridge regression
    reg_results = regress_features_on_probes(
        acts, meta, alive, alpha=alpha, probe_names=probe_names,
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

    return {
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


def main():
    parser = argparse.ArgumentParser(description="Regression analysis of SAE concepts")
    parser.add_argument("--output", type=str, default="output/concept_regression.json")
    parser.add_argument("--models", nargs='+', default=None,
                        help="Model names to analyze (default: all)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    parser.add_argument("--max-per-dataset", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pymfe-cache", type=str, default=None,
        help="Path to PyMFE cache JSON (from compute_pymfe_cache.py). "
             "Augments row-level probes with dataset-level meta-features.",
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

    # Find common datasets
    emb_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
    emb_dirs = {}
    for display_name, _, emb_model in models:
        d = emb_base / emb_model
        if d.exists():
            emb_dirs[display_name] = d
    datasets = find_common_datasets(emb_dirs)
    print(f"\nUsing {len(datasets)} common datasets")

    # Collect meta-features (use first model's embedding dir for alignment)
    first_emb_dir = list(emb_dirs.values())[0]
    meta_array, loaded, boundaries = collect_meta_for_datasets(
        datasets, first_emb_dir, args.max_per_dataset
    )
    print(f"Meta-features: {meta_array.shape}"
          f" ({len(loaded)} datasets, {len(boundaries)-1} boundaries)")

    # Optionally augment with PyMFE dataset-level features
    probe_names = list(META_NAMES)
    alpha = args.alpha
    if args.pymfe_cache:
        meta_array, pymfe_names = augment_meta_with_pymfe(
            meta_array, boundaries, loaded, args.pymfe_cache
        )
        probe_names = list(META_NAMES) + pymfe_names
        print(f"  Augmented meta: {meta_array.shape} "
              f"({len(META_NAMES)} probes + {len(pymfe_names)} PyMFE)")

        # Increase Ridge alpha for collinearity from dataset-constant features
        if pymfe_names and alpha < 10.0:
            alpha = 10.0
            print(f"  Ridge alpha increased to {alpha} (dataset-constant PyMFE features)")

        # Log condition number
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(meta_array)
        cond = np.linalg.cond(X_scaled)
        print(f"  Condition number: {cond:.1f}")

    # Analyze each model
    all_results = {}
    for display_name, sae_dir, emb_model in models:
        result = analyze_model_regression(
            display_name, sae_dir, emb_model,
            datasets, meta_array, args.max_per_dataset, alpha,
            probe_names=probe_names,
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

    # Add metadata about PyMFE augmentation
    metadata = {
        'n_row_probes': len(META_NAMES),
        'n_total_probes': len(probe_names),
        'pymfe_cache': args.pymfe_cache,
        'alpha': alpha,
    }
    if args.pymfe_cache:
        pymfe_names = probe_names[len(META_NAMES):]
        metadata['n_pymfe_features'] = len(pymfe_names)
        metadata['pymfe_feature_names'] = pymfe_names

    output_data = {'metadata': metadata, 'models': all_results}

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert_keys_to_native(output_data), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
