#!/usr/bin/env python3
"""
Deep analysis of SAE concepts: trace back to original tabular data patterns.

Key question: What properties of a tabular row cause each SAE feature to fire?
Since concepts are universal across datasets, we look for meta-patterns:
- Missing value rates
- Numeric outliers
- Categorical rarity
- Row complexity/entropy
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig
from data.extended_loader import load_tabarena_dataset


@dataclass
class RowMetaFeatures:
    """Meta-features computed for each row (dataset-agnostic)."""
    missing_rate: float          # Fraction of missing values
    numeric_mean_zscore: float   # Mean z-score of numeric values (how outlier-ish)
    numeric_max_zscore: float    # Max z-score (most extreme value)
    numeric_std: float           # Std of numeric values (row homogeneity)
    categorical_rarity: float    # Mean rarity of categorical values
    n_rare_categories: int       # Count of rare (<5%) categorical values
    row_entropy: float           # Entropy of discretized values
    n_numeric: int               # Number of numeric columns
    n_categorical: int           # Number of categorical columns


def compute_row_meta_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    col_stats: Dict,
) -> List[RowMetaFeatures]:
    """
    Compute dataset-agnostic meta-features for each row.

    These features describe the "shape" of a row without depending on
    specific column semantics.
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    meta_features = []

    for idx in range(n_rows):
        row = df.iloc[idx]

        # 1. Missing rate
        missing_rate = row.isna().sum() / n_cols

        # 2. Numeric statistics (z-scores relative to column stats)
        numeric_zscores = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                mean, std = col_stats[col]['mean'], col_stats[col]['std']
                if std > 1e-8:
                    zscore = abs((val - mean) / std)
                    numeric_zscores.append(zscore)

        if numeric_zscores:
            numeric_mean_zscore = np.mean(numeric_zscores)
            numeric_max_zscore = np.max(numeric_zscores)
            numeric_std = np.std(numeric_zscores)
        else:
            numeric_mean_zscore = 0.0
            numeric_max_zscore = 0.0
            numeric_std = 0.0

        # 3. Categorical rarity
        rarities = []
        n_rare = 0
        for col in categorical_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                freq = col_stats[col].get(val, 0.0)
                rarities.append(1.0 - freq)  # Rarity = 1 - frequency
                if freq < 0.05:
                    n_rare += 1

        categorical_rarity = np.mean(rarities) if rarities else 0.0

        # 4. Row entropy (discretize numerics, combine with categoricals)
        values = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and col in col_stats:
                # Discretize to quintiles
                pctl = col_stats[col].get('percentiles', [])
                if pctl:
                    bin_idx = np.searchsorted(pctl, val)
                    values.append(f"num_{bin_idx}")
        for col in categorical_cols:
            val = row[col]
            if pd.notna(val):
                values.append(f"cat_{val}")

        if values:
            _, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
            row_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            row_entropy = 0.0

        meta_features.append(RowMetaFeatures(
            missing_rate=missing_rate,
            numeric_mean_zscore=numeric_mean_zscore,
            numeric_max_zscore=numeric_max_zscore,
            numeric_std=numeric_std,
            categorical_rarity=categorical_rarity,
            n_rare_categories=n_rare,
            row_entropy=row_entropy,
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
        ))

    return meta_features


def compute_column_stats(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict]:
    """Compute statistics for each column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_stats = {}

    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            col_stats[col] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'percentiles': np.percentile(vals, [20, 40, 60, 80]).tolist(),
            }

    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        col_stats[col] = freq.to_dict()

    return numeric_cols, categorical_cols, col_stats


def load_sae_checkpoint(path: Path) -> Tuple[SparseAutoencoder, SAEConfig, Dict]:
    """Load SAE model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    config = SAEConfig(**checkpoint['config'])
    model = SparseAutoencoder(config)

    state_dict = checkpoint['model_state_dict']
    if 'reference_data' in state_dict and state_dict['reference_data'] is not None:
        ref_data = state_dict['reference_data']
        model.register_buffer('reference_data', ref_data)
        if 'archetype_logits' in state_dict:
            model.archetype_logits = torch.nn.Parameter(state_dict['archetype_logits'])
        if 'archetype_deviation' in state_dict:
            model.archetype_deviation = torch.nn.Parameter(state_dict['archetype_deviation'])

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, checkpoint


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split matching sae_tabarena_sweep.py."""
    train_datasets = []
    test_datasets = []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)
    return train_datasets, test_datasets


def analyze_feature_triggers(
    model: SparseAutoencoder,
    datasets: List[str],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    top_features: List[int],
    samples_per_feature: int = 100,
    max_samples_per_dataset: int = 200,
) -> Dict:
    """
    For each feature, find what tabular patterns trigger it.

    Returns per-feature analysis of row meta-features.
    """
    emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"

    # Collect all samples with their meta-features and activations
    all_meta = []  # List of RowMetaFeatures
    all_activations = []  # List of activation vectors
    all_dataset_names = []  # Track which dataset each sample came from

    print(f"Loading data and computing meta-features for {len(datasets)} datasets...")

    for ds_name in datasets:
        # Load embeddings
        emb_path = emb_dir / f"tabarena_{ds_name}.npz"
        if not emb_path.exists():
            continue

        emb_data = np.load(emb_path, allow_pickle=True)
        embeddings = emb_data['embeddings'].astype(np.float32)

        # Subsample if needed
        if len(embeddings) > max_samples_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(embeddings), max_samples_per_dataset, replace=False)
            embeddings = embeddings[idx]
            sample_indices = idx
        else:
            sample_indices = np.arange(len(embeddings))

        # Load original tabular data
        try:
            X, y, dataset_info = load_tabarena_dataset(ds_name)
            df = pd.DataFrame(X)

            # Subset to same samples
            df = df.iloc[sample_indices].reset_index(drop=True)

            # Compute column stats and row meta-features
            numeric_cols, categorical_cols, col_stats = compute_column_stats(df)
            meta_features = compute_row_meta_features(df, numeric_cols, categorical_cols, col_stats)

        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")
            continue

        # Compute SAE activations
        emb_norm = embeddings / train_std
        emb_centered = emb_norm - train_mean

        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x).numpy()

        all_meta.extend(meta_features)
        all_activations.append(h)
        all_dataset_names.extend([ds_name] * len(meta_features))

        print(f"  {ds_name}: {len(meta_features)} samples")

    if not all_activations:
        return {}

    all_activations = np.concatenate(all_activations, axis=0)
    print(f"Total: {len(all_meta)} samples from {len(set(all_dataset_names))} datasets")

    # Convert meta-features to array for analysis
    meta_array = np.array([
        [m.missing_rate, m.numeric_mean_zscore, m.numeric_max_zscore,
         m.numeric_std, m.categorical_rarity, m.n_rare_categories,
         m.row_entropy, m.n_numeric, m.n_categorical]
        for m in all_meta
    ])

    meta_names = [
        'missing_rate', 'numeric_mean_zscore', 'numeric_max_zscore',
        'numeric_std', 'categorical_rarity', 'n_rare_categories',
        'row_entropy', 'n_numeric', 'n_categorical'
    ]

    # Compute baseline statistics
    baseline_means = meta_array.mean(axis=0)
    baseline_stds = meta_array.std(axis=0)

    # Analyze each feature
    feature_analysis = {}

    for feat_idx in top_features:
        feat_acts = all_activations[:, feat_idx]

        # Find top activating samples
        top_indices = np.argsort(feat_acts)[-samples_per_feature:]
        top_acts = feat_acts[top_indices]

        # Skip if feature is dead
        if top_acts.max() < 0.01:
            continue

        # Get meta-features for top samples
        top_meta = meta_array[top_indices]
        top_means = top_meta.mean(axis=0)

        # Compute effect sizes (Cohen's d)
        effect_sizes = {}
        for i, name in enumerate(meta_names):
            if baseline_stds[i] > 1e-8:
                d = (top_means[i] - baseline_means[i]) / baseline_stds[i]
                effect_sizes[name] = float(d)
            else:
                effect_sizes[name] = 0.0

        # Find which datasets contribute most to this feature
        top_datasets = [all_dataset_names[i] for i in top_indices]
        dataset_counts = defaultdict(int)
        for ds in top_datasets:
            dataset_counts[ds] += 1
        top_dataset_list = sorted(dataset_counts.items(), key=lambda x: -x[1])[:5]

        # Identify dominant pattern
        sorted_effects = sorted(effect_sizes.items(), key=lambda x: -abs(x[1]))
        dominant_patterns = [(name, d) for name, d in sorted_effects if abs(d) > 0.3]

        feature_analysis[feat_idx] = {
            'mean_activation': float(top_acts.mean()),
            'max_activation': float(top_acts.max()),
            'effect_sizes': effect_sizes,
            'dominant_patterns': dominant_patterns,
            'top_datasets': top_dataset_list,
            'interpretation': interpret_pattern(dominant_patterns),
        }

    return {
        'baseline': {name: {'mean': float(baseline_means[i]), 'std': float(baseline_stds[i])}
                     for i, name in enumerate(meta_names)},
        'features': feature_analysis,
    }


def interpret_pattern(patterns: List[Tuple[str, float]]) -> str:
    """Generate human-readable interpretation of dominant patterns."""
    if not patterns:
        return "No strong pattern detected"

    interpretations = []
    for name, d in patterns:
        direction = "high" if d > 0 else "low"
        strength = "very " if abs(d) > 1.0 else ""

        if name == 'missing_rate':
            interpretations.append(f"{strength}{direction} missing values")
        elif name == 'numeric_mean_zscore':
            interpretations.append(f"{strength}{direction} numeric outliers (mean)")
        elif name == 'numeric_max_zscore':
            interpretations.append(f"{strength}{direction} extreme numeric values")
        elif name == 'numeric_std':
            interpretations.append(f"{strength}{direction} within-row numeric variance")
        elif name == 'categorical_rarity':
            interpretations.append(f"{strength}{direction} rare categorical values")
        elif name == 'n_rare_categories':
            interpretations.append(f"{strength}{direction} count of rare categories")
        elif name == 'row_entropy':
            interpretations.append(f"{strength}{direction} row complexity/entropy")
        elif name == 'n_numeric':
            interpretations.append(f"datasets with {direction} numeric column count")
        elif name == 'n_categorical':
            interpretations.append(f"datasets with {direction} categorical column count")

    return "; ".join(interpretations)


def main():
    parser = argparse.ArgumentParser(description="Deep SAE concept analysis")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--top-k-features", type=int, default=20, help="Number of features to analyze")
    parser.add_argument("--samples-per-feature", type=int, default=100, help="Top samples per feature")
    args = parser.parse_args()

    print("Loading SAE checkpoint...")
    model, config, checkpoint = load_sae_checkpoint(Path(args.model_path))
    print(f"  SAE type: {config.sparsity_type}")
    print(f"  Hidden dim: {config.hidden_dim}")

    # Get datasets and compute training stats
    emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / "tabpfn"
    all_datasets = [f.stem.replace("tabarena_", "") for f in emb_dir.glob("tabarena_*.npz")]
    train_datasets, test_datasets = get_train_test_split(all_datasets)

    print(f"\nComputing training normalization stats from {len(train_datasets)} datasets...")

    # Pool training embeddings
    train_embs = []
    for ds in train_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            emb = data['embeddings'].astype(np.float32)
            if len(emb) > 200:
                np.random.seed(42)
                idx = np.random.choice(len(emb), 200, replace=False)
                emb = emb[idx]
            train_embs.append(emb)

    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)

    # Find top features by mean activation
    print("\nFinding most active features...")
    train_centered = train_norm - train_mean
    with torch.no_grad():
        h = model.encode(torch.tensor(train_centered, dtype=torch.float32)).numpy()

    feature_means = h.mean(axis=0)
    top_features = np.argsort(feature_means)[-args.top_k_features:][::-1].tolist()
    print(f"  Top {args.top_k_features} features by mean activation: {top_features[:10]}...")

    # Deep analysis
    print("\n" + "="*60)
    print("DEEP CONCEPT ANALYSIS")
    print("="*60)

    results = analyze_feature_triggers(
        model=model,
        datasets=all_datasets,
        train_std=train_std,
        train_mean=train_mean,
        top_features=top_features,
        samples_per_feature=args.samples_per_feature,
    )

    # Print results
    print("\n" + "-"*60)
    print("FEATURE INTERPRETATIONS")
    print("-"*60)

    for feat_idx, analysis in results.get('features', {}).items():
        print(f"\nFeature {feat_idx}:")
        print(f"  Interpretation: {analysis['interpretation']}")
        print(f"  Top datasets: {[d[0] for d in analysis['top_datasets'][:3]]}")

        # Show top effect sizes
        effects = sorted(analysis['effect_sizes'].items(), key=lambda x: -abs(x[1]))[:3]
        print(f"  Top effects: {[(n, f'{d:.2f}') for n, d in effects]}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
