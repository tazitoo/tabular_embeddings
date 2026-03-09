#!/usr/bin/env python3
"""
Cross-model SAE concept comparison.

Each model has its own SAE trained on its own embedding space (different dimensions).
We compare what concepts (meta-features) each model's SAE discovers by computing
Cohen's d effect sizes against row-level meta-features.

Usage:
    python scripts/compare_sae_cross_model.py \
        --output output/cross_model_sae_comparison.json
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
    collect_meta_features,
    compute_activations,
    compute_basic_metrics,
    compute_feature_effects,
    compute_monosemanticity,
    compute_redundancy,
    compute_scale_analysis,
    load_embeddings_and_normalize,
    meta_features_to_array,
)
from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    compute_column_stats,
    compute_concept_coverage,
    compute_row_meta_features,
    convert_keys_to_native,
    load_sae_checkpoint,
)
from data.extended_loader import load_tabarena_dataset

# SAE sweep round tracking — increment when retraining with architecture changes.
# Round 4: DAE bug (TopK not enforced), fixed bands [32,64,128,256]
# Round 5: TopK enforced, proportional bands [h/16,h/8,h/4,h/2,h], ghost grads enabled
# Round 6: Per-dataset StandardScaler, b_dec subtraction, ConstrainedAdam, W_enc=W_dec.T,
#           TopK on pre_act then ReLU, no L1 on TopK types, fixed AuxK
DEFAULT_SAE_ROUND = 6


def sae_sweep_dir(round: int = None) -> Path:
    """Return the SAE sweep base directory for a given round."""
    r = round if round is not None else DEFAULT_SAE_ROUND
    return PROJECT_ROOT / "output" / f"sae_tabarena_sweep_round{r}"


# Default model configurations: (display_name, sae_sweep_dir, emb_dir)
# Round 5 SAEs are trained on all datasets per model (cls + reg pooled).
# Mitra's cls/reg checkpoints produce different embeddings, but both are
# extracted at the cls-optimal layer (10) and pooled into one SAE.
DEFAULT_MODELS = [
    ("TabPFN", "tabpfn", "tabpfn"),
    ("CARTE", "carte", "carte"),
    ("TabICL", "tabicl", "tabicl"),
    ("TabDPT", "tabdpt", "tabdpt"),
    ("Mitra", "mitra", "mitra"),
    ("HyperFast", "hyperfast", "hyperfast"),
    ("Tabula-8B", "tabula8b", "tabula8b"),
]

# Regression model configurations: models with regression-capable embeddings.
REGRESSION_MODELS = [
    ("TabPFN", "tabpfn", "tabpfn"),
    ("CARTE", "carte", "carte"),
    ("TabDPT", "tabdpt", "tabdpt"),
    ("Mitra", "mitra", "mitra"),
    ("Tabula-8B", "tabula8b", "tabula8b"),
]

# Models whose SAE only applies to datasets of a specific task type.
# Models not listed here work on all tasks.
MODEL_TASK_FILTERS = {}

REGRESSION_TASK_FILTERS = {}


def get_models_for_task(task: str = "classification"):
    """Return (model_list, task_filters) appropriate for the given task type.

    Args:
        task: "classification" or "regression"

    Returns:
        (models, task_filters) where models is a list of
        (display_name, sae_sweep_dir, emb_dir) tuples and task_filters
        restricts certain models to specific dataset task types.
    """
    if task == "regression":
        return REGRESSION_MODELS, REGRESSION_TASK_FILTERS
    return DEFAULT_MODELS, MODEL_TASK_FILTERS


def get_dataset_tasks():
    """Return {dataset_name: task_type} for all TabArena datasets."""
    from data.extended_loader import TABARENA_DATASETS
    return {name: info["task"] for name, info in TABARENA_DATASETS.items()}


def find_common_datasets(emb_dirs: Dict[str, Path]) -> List[str]:
    """Find datasets available in ALL embedding directories."""
    dataset_sets = []
    for name, emb_dir in emb_dirs.items():
        datasets = set(
            f.stem.replace("tabarena_", "")
            for f in emb_dir.glob("tabarena_*.npz")
        )
        print(f"  {name}: {len(datasets)} datasets")
        dataset_sets.append(datasets)
    common = sorted(set.intersection(*dataset_sets))
    print(f"  Common: {len(common)} datasets")
    return common


def _process_single_dataset(
    ds_name: str,
    emb_dir: Path,
    max_per_dataset: int,
) -> Optional[Tuple[str, List[List[float]]]]:
    """Process one dataset for meta-feature extraction (joblib worker function)."""
    import pandas as pd

    emb_path = emb_dir / f"tabarena_{ds_name}.npz"
    if not emb_path.exists():
        return None

    emb_data = np.load(emb_path, allow_pickle=True)
    n_emb = len(emb_data['embeddings'])

    if n_emb > max_per_dataset:
        np.random.seed(42)
        sample_indices = np.random.choice(n_emb, max_per_dataset, replace=False)
    else:
        sample_indices = np.arange(n_emb)

    try:
        X, y, _ = load_tabarena_dataset(ds_name)
        df = X if hasattr(X, 'iloc') else pd.DataFrame(X)
        df = df.iloc[sample_indices].reset_index(drop=True)
        y_sub = y[sample_indices] if y is not None else None

        numeric_cols, categorical_cols, col_stats, dataset_stats = compute_column_stats(df)
        meta_features = compute_row_meta_features(
            df, y_sub, numeric_cols, categorical_cols, col_stats, dataset_stats
        )
        rows = [meta_features_to_array(m) for m in meta_features]
        print(f"    {ds_name}: {len(rows)} rows")
        return (ds_name, rows)
    except Exception as e:
        print(f"    Skipping {ds_name}: {e}")
        return None


def collect_meta_for_datasets(
    datasets: List[str],
    emb_dir: Path,
    max_per_dataset: int = 500,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute row meta-features for specified datasets.

    Uses sample indices matching the embedding pooling logic so meta-features
    align row-by-row with SAE activations. Parallelized across datasets with
    joblib (each dataset is independent).

    Args:
        n_jobs: Number of parallel workers. -1 = all CPUs, 1 = sequential.

    Returns:
        meta_array: (n_samples, n_meta_features)
        loaded_datasets: datasets that loaded successfully
        boundaries: (n_datasets + 1,) cumulative sample counts per dataset
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_single_dataset)(ds, emb_dir, max_per_dataset)
        for ds in datasets
    )

    # Collect in original dataset order (deterministic)
    all_meta = []
    loaded = []
    boundaries = [0]
    for res in results:
        if res is not None:
            ds_name, rows = res
            all_meta.extend(rows)
            loaded.append(ds_name)
            boundaries.append(boundaries[-1] + len(rows))

    meta = np.array(all_meta) if all_meta else np.empty((0, 0))
    return meta, loaded, np.array(boundaries)


def pool_embeddings_for_datasets(
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = 500,
) -> np.ndarray:
    """Pool embeddings for specific datasets only (matching meta-feature order)."""
    all_embs = []
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        all_embs.append(emb)
    return np.concatenate(all_embs)


def analyze_single_model(
    model_name: str,
    sae_path: Path,
    emb_dir: Path,
    meta_array: np.ndarray,
    datasets: List[str],
    max_per_dataset: int = 500,
) -> Dict:
    """Run full concept analysis for one model's SAE."""
    print(f"\n{'─' * 60}")
    print(f"Analyzing {model_name}")
    print(f"{'─' * 60}")

    # Load SAE
    model, config, _ = load_sae_checkpoint(sae_path)
    print(f"  SAE: {config.sparsity_type}, hidden={config.hidden_dim}, "
          f"topk={config.topk}, input={config.input_dim}")

    # Pool embeddings for common datasets only
    pooled = pool_embeddings_for_datasets(emb_dir, datasets, max_per_dataset)
    print(f"  Embeddings: {pooled.shape}")

    # Normalize (train split stats)
    from scripts.compare_sae_architectures import get_train_test_split
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

    # Align sample counts
    n_min = min(len(meta_array), len(acts))
    acts_aligned = acts[:n_min]
    meta_aligned = meta_array[:n_min]

    # Basic metrics
    metrics = compute_basic_metrics(acts_aligned, config)
    print(f"  Alive: {metrics['alive_features']}/{config.hidden_dim} "
          f"({metrics['alive_frac']:.0%})")
    print(f"  L0: {metrics['l0_mean']:.1f}")

    # Feature effects
    feat_analysis = compute_feature_effects(
        acts_aligned, meta_aligned, metrics['alive_indices']
    )
    print(f"  Features analyzed: {len(feat_analysis)}")

    # Monosemanticity
    mono = compute_monosemanticity(feat_analysis)
    print(f"  Effects/feature: {mono['mean_strong_effects']:.1f}, "
          f"monosemantic: {mono['frac_monosemantic']:.0%}")

    # Coverage
    coverage = compute_concept_coverage(feat_analysis, META_NAMES)
    well_covered = sum(1 for c in coverage.values() if c['coverage_score'] > 1.0)
    poor_covered = sum(1 for c in coverage.values() if c['coverage_score'] < 0.5)
    print(f"  Coverage: {well_covered} well (d>1), {poor_covered} poor (d<0.5)")

    # Redundancy
    redundancy = compute_redundancy(model, metrics['alive_indices'])
    print(f"  Redundancy: {redundancy['frac_duplicate_pairs']:.2%} dup pairs, "
          f"mean sim {redundancy['mean_pairwise_similarity']:.3f}")

    # Scale analysis
    scale = compute_scale_analysis(acts_aligned, meta_aligned, config)

    return {
        'model_name': model_name,
        'sae_path': str(sae_path),
        'emb_dir': str(emb_dir),
        'config': {
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
            'topk': config.topk,
            'sparsity_type': config.sparsity_type,
        },
        'basic_metrics': {k: v for k, v in metrics.items() if k != 'alive_indices'},
        'monosemanticity': mono,
        'coverage': coverage,
        'redundancy': redundancy,
        'scale_analysis': scale,
    }


def print_cross_model_summary(results: Dict[str, Dict]):
    """Print formatted cross-model comparison."""
    models = list(results.keys())
    n = len(models)

    print("\n" + "=" * 80)
    print(f"{'CROSS-MODEL SAE CONCEPT COMPARISON':^80}")
    print("=" * 80)

    # ── Basic metrics table ──
    print(f"\n{'MODEL METRICS':^80}")
    print("-" * 80)
    col_w = 14
    header = f"  {'':22s}" + "".join(f"{m:>{col_w}s}" for m in models)
    print(header)
    print("  " + "-" * (22 + col_w * n))

    rows = [
        ("Embedding dim", lambda r: r['config']['input_dim']),
        ("Hidden dim", lambda r: r['config']['hidden_dim']),
        ("TopK", lambda r: r['config']['topk']),
        ("Alive features", lambda r: f"{r['basic_metrics']['alive_features']} ({r['basic_metrics']['alive_frac']:.0%})"),
        ("L0 sparsity", lambda r: f"{r['basic_metrics']['l0_mean']:.1f}"),
        ("Effects/feat (d>0.5)", lambda r: f"{r['monosemanticity']['mean_strong_effects']:.1f}"),
        ("Frac monosemantic", lambda r: f"{r['monosemanticity']['frac_monosemantic']:.0%}"),
        ("Dup pairs (cos>0.8)", lambda r: f"{r['redundancy']['frac_duplicate_pairs']:.2%}"),
        ("Mean pairwise sim", lambda r: f"{r['redundancy']['mean_pairwise_similarity']:.3f}"),
    ]
    for label, fn in rows:
        vals = "".join(f"{str(fn(results[m])):>{col_w}s}" for m in models)
        print(f"  {label:22s}{vals}")

    # ── Coverage comparison ──
    print(f"\n{'CONCEPT COVERAGE (max |Cohen d| per meta-feature)':^80}")
    print("-" * 80)
    header = f"  {'Meta-feature':28s}" + "".join(f"{m:>{col_w}s}" for m in models) + f"{'Universal':>{col_w}s}"
    print(header)
    print("  " + "-" * (28 + col_w * (n + 1)))

    universal_count = 0
    model_specific = {m: [] for m in models}

    for meta_name in META_NAMES:
        scores = []
        for m in models:
            cov = results[m]['coverage'].get(meta_name, {})
            score = cov.get('coverage_score', 0.0)
            scores.append(score)

        # Check if any model has meaningful coverage
        max_score = max(scores)
        if max_score < 0.3:
            continue

        # Universal: d > 1.0 in ALL models
        all_strong = all(s > 1.0 for s in scores)
        if all_strong:
            universal_count += 1
            tag = "YES"
        elif max_score > 1.0:
            # Model-specific: strong in some but not all
            tag = ""
            for i, m in enumerate(models):
                if scores[i] > 1.0:
                    model_specific[m].append(meta_name)
        else:
            tag = ""

        score_strs = "".join(f"{s:>{col_w}.2f}" for s in scores)
        print(f"  {meta_name:28s}{score_strs}{tag:>{col_w}s}")

    # ── Summary ──
    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)

    well_covered = {}
    for m in models:
        wc = sum(1 for c in results[m]['coverage'].values() if c['coverage_score'] > 1.0)
        well_covered[m] = wc

    vals = "".join(f"{well_covered[m]:>{col_w}d}" for m in models)
    print(f"  {'Well-covered (d>1.0)':28s}{vals}")

    poor_covered = {}
    for m in models:
        pc = sum(1 for c in results[m]['coverage'].values() if c['coverage_score'] < 0.5)
        poor_covered[m] = pc
    vals = "".join(f"{poor_covered[m]:>{col_w}d}" for m in models)
    print(f"  {'Poorly-covered (d<0.5)':28s}{vals}")

    print(f"\n  Universal concepts (d>1.0 in ALL models): {universal_count}/{len(META_NAMES)}")

    for m in models:
        unique = [name for name in model_specific[m]
                  if not any(name in model_specific[m2] for m2 in models if m2 != m)]
        if unique:
            print(f"  {m}-only concepts: {', '.join(unique[:5])}")

    # ── Scale analysis (if any model has it) ──
    any_scale = any(results[m].get('scale_analysis') for m in models)
    if any_scale:
        print(f"\n{'MAT-ARCH SCALE ANALYSIS':^80}")
        print("-" * 80)
        for m in models:
            scale = results[m].get('scale_analysis')
            if not scale:
                continue
            print(f"\n  {m}:")
            print(f"    {'Scale':15s} {'Alive':>8s}  {'Effects':>8s}  {'Covered':>8s}  Top concepts")
            print("    " + "-" * 65)
            for band, info in scale.items():
                concepts = ", ".join(info['well_covered_meta'][:3]) if info['well_covered_meta'] else "-"
                print(f"    {band:15s} {info['alive_features']:>5d}/{info['n_features']:<3d}"
                      f"  {info['mean_strong_effects']:8.1f}"
                      f"  {info['n_well_covered_meta']:8d}  {concepts}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model SAE concept comparison"
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--max-per-dataset", type=int, default=500,
                        help="Max samples per dataset (default: 500)")
    parser.add_argument("--round", type=int, default=None,
                        help=f"SAE sweep round (default: {DEFAULT_SAE_ROUND})")
    args = parser.parse_args()

    base_sae = sae_sweep_dir(args.round)
    base_emb = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    # Resolve model paths
    model_configs = []
    for display_name, sweep_dir, emb_dir_name in DEFAULT_MODELS:
        sae_path = base_sae / sweep_dir / "sae_matryoshka_archetypal_validated.pt"
        emb_dir = base_emb / emb_dir_name
        if not sae_path.exists():
            print(f"Warning: SAE not found for {display_name}: {sae_path}")
            continue
        if not emb_dir.exists():
            print(f"Warning: Embeddings not found for {display_name}: {emb_dir}")
            continue
        model_configs.append((display_name, sae_path, emb_dir))

    if len(model_configs) < 2:
        print("Error: Need at least 2 models for comparison")
        sys.exit(1)

    print(f"Comparing {len(model_configs)} models: {[m[0] for m in model_configs]}")

    # Find common datasets
    print("\nFinding common datasets...")
    emb_dirs = {name: emb_dir for name, _, emb_dir in model_configs}
    common_datasets = find_common_datasets(emb_dirs)

    # Compute meta-features once (using first model's emb_dir for sample indices)
    # All models use same seed(42) subsampling so indices match for common datasets
    print(f"\nComputing meta-features for {len(common_datasets)} common datasets...")
    meta_array, loaded_datasets, _boundaries = collect_meta_for_datasets(
        common_datasets, model_configs[0][2], max_per_dataset=args.max_per_dataset
    )
    print(f"  Meta-feature matrix: {meta_array.shape} from {len(loaded_datasets)} datasets")

    # Analyze each model
    results = {}
    for display_name, sae_path, emb_dir in model_configs:
        results[display_name] = analyze_single_model(
            display_name, sae_path, emb_dir, meta_array,
            loaded_datasets, max_per_dataset=args.max_per_dataset,
        )

    # Print cross-model comparison
    print_cross_model_summary(results)

    # Save JSON
    if args.output:
        report = {
            'models': results,
            'common_datasets': loaded_datasets,
            'n_common_datasets': len(loaded_datasets),
            'n_samples': len(meta_array),
            'meta_feature_names': META_NAMES,
        }
        report_clean = convert_keys_to_native(report)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report_clean, f, indent=2, cls=NumpyEncoder)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
