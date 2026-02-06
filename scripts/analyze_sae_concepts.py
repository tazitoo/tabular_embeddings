#!/usr/bin/env python3
"""
Analyze what concepts the SAE learned from TabPFN embeddings.

Key questions:
1. What activates each dictionary feature?
2. What input dimensions does each feature encode?
3. Do Matryoshka scales capture different abstraction levels?
4. Are there dataset-specific vs universal features?
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig


def get_train_test_split(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split matching sae_tabarena_sweep.py."""
    train_datasets = []
    test_datasets = []
    for ds in datasets:
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:  # 70% train
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)
    return train_datasets, test_datasets


def load_sae_checkpoint(path: Path) -> Tuple[SparseAutoencoder, SAEConfig, Dict]:
    """Load SAE model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    config = SAEConfig(**checkpoint['config'])
    model = SparseAutoencoder(config)

    # For archetypal SAEs, we need to initialize the archetype parameters
    # before loading state dict
    state_dict = checkpoint['model_state_dict']
    if 'reference_data' in state_dict and state_dict['reference_data'] is not None:
        # Initialize with dummy data, will be overwritten by load_state_dict
        ref_data = state_dict['reference_data']
        n_ref = len(ref_data)

        # Register the buffer and parameter with correct shapes
        model.register_buffer('reference_data', ref_data)
        if 'archetype_logits' in state_dict:
            model.archetype_logits = torch.nn.Parameter(state_dict['archetype_logits'])
        if 'archetype_deviation' in state_dict:
            model.archetype_deviation = torch.nn.Parameter(state_dict['archetype_deviation'])

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, checkpoint


def load_embeddings_by_dataset(
    model_name: str = "tabpfn",
    max_per_dataset: int = 200,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load embeddings grouped by dataset.

    Returns:
        embeddings: Dict mapping dataset name to embeddings
        train_std: Per-dimension std from TRAINING pool (for normalization)
        train_mean: Per-dimension mean from std-normalized TRAINING pool (for centering)
    """
    emb_dir = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / model_name

    dataset_embeddings = {}
    for f in emb_dir.glob("tabarena_*.npz"):
        ds_name = f.stem.replace("tabarena_", "")
        data = np.load(f, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)

        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]

        dataset_embeddings[ds_name] = emb

    # Compute normalization from training data only (matching sweep script)
    train_datasets, test_datasets = get_train_test_split(list(dataset_embeddings.keys()))
    train_emb = np.concatenate([dataset_embeddings[ds] for ds in train_datasets])

    # Step 1: Compute std for normalization
    train_std = train_emb.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0

    # Step 2: Compute mean AFTER std normalization (matching train_sae centering)
    train_emb_norm = train_emb / train_std
    train_mean = train_emb_norm.mean(axis=0, keepdims=True)

    print(f"  Train/test split: {len(train_datasets)}/{len(test_datasets)}")

    return dataset_embeddings, train_std, train_mean


def analyze_feature_activations(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    top_k: int = 10,
) -> Dict:
    """
    Analyze which samples/datasets activate each feature.

    Returns per-feature statistics:
    - Top activating datasets
    - Activation sparsity
    - Mean/max activation per dataset
    """

    # Get activations for each dataset
    dataset_activations = {}
    for ds_name, emb in embeddings.items():
        # Normalize and center using TRAINING pool stats
        emb_norm = emb / train_std
        emb_centered = emb_norm - train_mean
        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x)  # (n_samples, hidden_dim)
            dataset_activations[ds_name] = h.numpy()

    hidden_dim = model.config.hidden_dim

    # Analyze each feature
    feature_stats = []
    for feat_idx in range(hidden_dim):
        # Collect activations across all datasets
        ds_mean_acts = {}
        ds_max_acts = {}
        ds_sparsity = {}  # fraction of samples with non-zero activation

        for ds_name, acts in dataset_activations.items():
            feat_acts = acts[:, feat_idx]
            ds_mean_acts[ds_name] = float(feat_acts.mean())
            ds_max_acts[ds_name] = float(feat_acts.max())
            ds_sparsity[ds_name] = float((feat_acts > 0.01).mean())

        # Find top activating datasets
        top_datasets = sorted(ds_mean_acts.items(), key=lambda x: -x[1])[:top_k]

        # Overall statistics
        all_acts = np.concatenate([acts[:, feat_idx] for acts in dataset_activations.values()])

        feature_stats.append({
            'feature_idx': feat_idx,
            'mean_activation': float(all_acts.mean()),
            'max_activation': float(all_acts.max()),
            'sparsity': float((all_acts > 0.01).mean()),  # fraction active
            'top_datasets': top_datasets,
            'dataset_specificity': np.std(list(ds_mean_acts.values())),  # high = dataset-specific
        })

    return {
        'feature_stats': feature_stats,
        'dataset_activations': {k: v.mean(axis=0).tolist() for k, v in dataset_activations.items()},
    }


def analyze_input_correlations(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    train_std: np.ndarray,
    train_mean: np.ndarray,
) -> np.ndarray:
    """
    Correlate each SAE feature with input embedding dimensions.

    Returns: (hidden_dim, input_dim) correlation matrix
    """
    # Pool, normalize, and center using training stats
    all_emb = np.concatenate(list(embeddings.values()))
    emb_norm = all_emb / train_std
    emb_centered = emb_norm - train_mean

    # Get activations
    with torch.no_grad():
        x = torch.tensor(emb_centered, dtype=torch.float32)
        h = model.encode(x).numpy()  # (n_samples, hidden_dim)

    # Compute correlations
    # For correlation, re-center (data is already centered, but do it again for numerical precision)
    x_centered_corr = emb_centered - emb_centered.mean(axis=0)
    h_centered = h - h.mean(axis=0)

    x_std = x_centered_corr.std(axis=0, keepdims=True)
    h_std = h_centered.std(axis=0, keepdims=True)

    x_std[x_std < 1e-8] = 1.0
    h_std[h_std < 1e-8] = 1.0

    x_normed = x_centered_corr / x_std
    h_normed = h_centered / h_std

    # Correlation: (hidden_dim, input_dim)
    correlations = (h_normed.T @ x_normed) / len(x_normed)

    return correlations


def analyze_matryoshka_scales(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    scales: List[int] = None,
) -> Dict:
    """
    Analyze what each Matryoshka scale captures.

    Hypothesis: Earlier scales capture global/universal concepts,
    later scales capture dataset-specific nuances.
    """
    # Auto-generate scales if not provided (include full dimension)
    if scales is None:
        hidden_dim = model.config.hidden_dim
        scales = [32, 64, 128, 256, 512, 1024, hidden_dim]
        scales = [s for s in scales if s <= hidden_dim]

    results = {}

    for ds_name, emb in embeddings.items():
        # Normalize and center using TRAINING pool stats (matching train_sae preprocessing)
        emb_norm = emb / train_std
        emb_centered = emb_norm - train_mean

        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x)  # (n_samples, hidden_dim)

            # Reconstruction at each scale
            scale_r2 = {}
            prev_scale = 0
            for scale in scales:
                if scale > model.config.hidden_dim:
                    break

                # Proper Matryoshka truncation: decode with only first `scale` dims
                # This uses both truncated activations AND truncated dictionary
                x_hat = model.decode(h, max_dim=scale)

                # R² at this scale (centered data, so mean is ~0)
                ss_res = ((x - x_hat) ** 2).sum()
                ss_tot = (x ** 2).sum()  # Centered data
                r2 = 1 - ss_res / ss_tot

                scale_r2[scale] = float(r2)
                prev_scale = scale

            results[ds_name] = scale_r2

    # Aggregate across datasets
    aggregated = {}
    for scale in scales:
        if scale > model.config.hidden_dim:
            break
        r2_values = [r[scale] for r in results.values() if scale in r]
        aggregated[scale] = {
            'mean_r2': float(np.mean(r2_values)),
            'std_r2': float(np.std(r2_values)),
            'min_r2': float(np.min(r2_values)),
            'max_r2': float(np.max(r2_values)),
        }

    return {
        'per_dataset': results,
        'aggregated': aggregated,
    }


def cluster_features(
    model: SparseAutoencoder,
    n_clusters: int = 10,
) -> Dict:
    """
    Cluster dictionary features to find groups of related concepts.
    """
    # Get the decoder dictionary (already numpy from get_dictionary())
    dictionary = model.get_dictionary()  # (hidden_dim, input_dim)
    if hasattr(dictionary, 'detach'):
        dictionary = dictionary.detach().cpu().numpy()

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dictionary)

    # Analyze clusters
    clusters = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_dict = dictionary[mask]

        # Cluster characteristics
        clusters[i] = {
            'size': int(mask.sum()),
            'feature_indices': np.where(mask)[0].tolist(),
            'centroid_norm': float(np.linalg.norm(kmeans.cluster_centers_[i])),
            'within_cluster_variance': float(cluster_dict.var()),
        }

    return {
        'n_clusters': n_clusters,
        'labels': labels.tolist(),
        'clusters': clusters,
    }


def find_interpretable_features(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    train_std: np.ndarray,
    train_mean: np.ndarray,
    correlations: np.ndarray,
    top_k: int = 20,
) -> List[Dict]:
    """
    Find the most interpretable features - those with:
    1. High activation on specific datasets
    2. Strong correlation with specific input dimensions
    3. Clear semantic meaning
    """
    # Get activations per dataset
    dataset_mean_acts = {}
    for ds_name, emb in embeddings.items():
        emb_norm = emb / train_std
        emb_centered = emb_norm - train_mean
        with torch.no_grad():
            x = torch.tensor(emb_centered, dtype=torch.float32)
            h = model.encode(x).numpy()
            dataset_mean_acts[ds_name] = h.mean(axis=0)

    interpretable = []
    hidden_dim = model.config.hidden_dim

    for feat_idx in range(hidden_dim):
        # Dataset specificity: variance across datasets
        ds_acts = np.array([dataset_mean_acts[ds][feat_idx] for ds in dataset_mean_acts])
        dataset_specificity = ds_acts.std() / (ds_acts.mean() + 1e-8)

        # Top correlated input dims
        feat_corr = correlations[feat_idx]
        top_input_dims = np.argsort(np.abs(feat_corr))[-5:][::-1]
        top_input_corrs = feat_corr[top_input_dims]

        # Top activating datasets
        top_ds_idx = np.argsort(ds_acts)[-3:][::-1]
        top_datasets = [(list(dataset_mean_acts.keys())[i], float(ds_acts[i])) for i in top_ds_idx]

        # Interpretability score: high specificity + strong correlations
        interp_score = dataset_specificity * np.abs(top_input_corrs).max()

        interpretable.append({
            'feature_idx': feat_idx,
            'interpretability_score': float(interp_score),
            'dataset_specificity': float(dataset_specificity),
            'mean_activation': float(ds_acts.mean()),
            'top_datasets': top_datasets,
            'top_input_dims': top_input_dims.tolist(),
            'top_input_correlations': top_input_corrs.tolist(),
        })

    # Sort by interpretability
    interpretable.sort(key=lambda x: -x['interpretability_score'])

    return interpretable[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE concepts")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=20, help="Top features to analyze")
    args = parser.parse_args()

    print("Loading SAE checkpoint...")
    model, config, checkpoint = load_sae_checkpoint(Path(args.model_path))
    print(f"  SAE type: {config.sparsity_type}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Input dim: {config.input_dim}")

    print("\nLoading embeddings by dataset...")
    embeddings, train_std, train_mean = load_embeddings_by_dataset()
    print(f"  Loaded {len(embeddings)} datasets")
    print(f"  Total samples: {sum(len(e) for e in embeddings.values())}")

    print("\n" + "="*60)
    print("ANALYZING SAE CONCEPTS")
    print("="*60)

    # 1. Feature activations
    print("\n1. Analyzing feature activations...")
    activation_stats = analyze_feature_activations(model, embeddings, train_std, train_mean)

    # Summary stats
    sparsities = [f['sparsity'] for f in activation_stats['feature_stats']]
    print(f"  Mean feature sparsity: {np.mean(sparsities):.3f}")
    print(f"  Alive features (>1% active): {sum(s > 0.01 for s in sparsities)}")

    # 2. Input correlations
    print("\n2. Computing input correlations...")
    correlations = analyze_input_correlations(model, embeddings, train_std, train_mean)
    print(f"  Max absolute correlation: {np.abs(correlations).max():.3f}")
    print(f"  Mean absolute correlation: {np.abs(correlations).mean():.3f}")

    # 3. Matryoshka scale analysis
    print("\n3. Analyzing Matryoshka scales...")
    scale_analysis = analyze_matryoshka_scales(model, embeddings, train_std, train_mean)
    print("  R² by scale:")
    for scale, stats in scale_analysis['aggregated'].items():
        print(f"    {scale:4d} features: R²={stats['mean_r2']:.3f} ± {stats['std_r2']:.3f}")

    # 4. Feature clustering
    print("\n4. Clustering dictionary features...")
    cluster_results = cluster_features(model, n_clusters=10)
    print("  Cluster sizes:", [c['size'] for c in cluster_results['clusters'].values()])

    # 5. Most interpretable features
    print("\n5. Finding most interpretable features...")
    interpretable = find_interpretable_features(model, embeddings, train_std, train_mean, correlations, top_k=args.top_k)

    print(f"\nTop {args.top_k} most interpretable features:")
    print("-"*60)
    for feat in interpretable[:10]:
        print(f"  Feature {feat['feature_idx']:4d}: "
              f"score={feat['interpretability_score']:.3f}, "
              f"specificity={feat['dataset_specificity']:.2f}")
        print(f"    Top datasets: {[d[0] for d in feat['top_datasets']]}")
        print(f"    Top input dims: {feat['top_input_dims']}")

    # 6. Dataset-specific vs universal features
    print("\n6. Dataset-specific vs Universal features...")
    specificities = [f['dataset_specificity'] for f in activation_stats['feature_stats']]

    # High specificity = dataset-specific
    high_spec = sum(s > 1.0 for s in specificities)
    low_spec = sum(s < 0.3 for s in specificities)
    print(f"  Dataset-specific features (specificity > 1.0): {high_spec}")
    print(f"  Universal features (specificity < 0.3): {low_spec}")

    # Save results
    if args.output:
        results = {
            'config': checkpoint['config'],
            'metrics': checkpoint['metrics'],
            'activation_stats': activation_stats,
            'scale_analysis': scale_analysis,
            'cluster_results': cluster_results,
            'interpretable_features': interpretable,
            'summary': {
                'n_features': config.hidden_dim,
                'alive_features': sum(s > 0.01 for s in sparsities),
                'mean_sparsity': float(np.mean(sparsities)),
                'dataset_specific_features': high_spec,
                'universal_features': low_spec,
            }
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
