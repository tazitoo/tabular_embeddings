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

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig


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
) -> Dict[str, np.ndarray]:
    """Load embeddings grouped by dataset."""
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

    return dataset_embeddings


def analyze_feature_activations(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
) -> Dict:
    """
    Analyze which samples/datasets activate each feature.

    Returns per-feature statistics:
    - Top activating datasets
    - Activation sparsity
    - Mean/max activation per dataset
    """
    # Normalize embeddings (same as training)
    all_emb = np.concatenate(list(embeddings.values()))
    std = all_emb.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    # Get activations for each dataset
    dataset_activations = {}
    for ds_name, emb in embeddings.items():
        emb_norm = emb / std
        with torch.no_grad():
            x = torch.tensor(emb_norm, dtype=torch.float32)
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
) -> np.ndarray:
    """
    Correlate each SAE feature with input embedding dimensions.

    Returns: (hidden_dim, input_dim) correlation matrix
    """
    # Pool and normalize
    all_emb = np.concatenate(list(embeddings.values()))
    std = all_emb.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    emb_norm = all_emb / std

    # Get activations
    with torch.no_grad():
        x = torch.tensor(emb_norm, dtype=torch.float32)
        h = model.encode(x).numpy()  # (n_samples, hidden_dim)

    # Compute correlations
    # Normalize for correlation
    x_centered = emb_norm - emb_norm.mean(axis=0)
    h_centered = h - h.mean(axis=0)

    x_std = x_centered.std(axis=0, keepdims=True)
    h_std = h_centered.std(axis=0, keepdims=True)

    x_std[x_std < 1e-8] = 1.0
    h_std[h_std < 1e-8] = 1.0

    x_normed = x_centered / x_std
    h_normed = h_centered / h_std

    # Correlation: (hidden_dim, input_dim)
    correlations = (h_normed.T @ x_normed) / len(x_normed)

    return correlations


def analyze_matryoshka_scales(
    model: SparseAutoencoder,
    embeddings: Dict[str, np.ndarray],
    scales: List[int] = [32, 64, 128, 256],
) -> Dict:
    """
    Analyze what each Matryoshka scale captures.

    Hypothesis: Earlier scales capture global/universal concepts,
    later scales capture dataset-specific nuances.
    """
    # Pool and normalize
    all_emb = np.concatenate(list(embeddings.values()))
    std = all_emb.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    results = {}

    for ds_name, emb in embeddings.items():
        emb_norm = emb / std

        with torch.no_grad():
            x = torch.tensor(emb_norm, dtype=torch.float32)
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

                # R² at this scale
                ss_res = ((x - x_hat) ** 2).sum()
                ss_tot = ((x - x.mean(dim=0)) ** 2).sum()
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
    correlations: np.ndarray,
    top_k: int = 20,
) -> List[Dict]:
    """
    Find the most interpretable features - those with:
    1. High activation on specific datasets
    2. Strong correlation with specific input dimensions
    3. Clear semantic meaning
    """
    # Pool and normalize
    all_emb = np.concatenate(list(embeddings.values()))
    std = all_emb.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    # Get activations per dataset
    dataset_mean_acts = {}
    for ds_name, emb in embeddings.items():
        emb_norm = emb / std
        with torch.no_grad():
            x = torch.tensor(emb_norm, dtype=torch.float32)
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
    embeddings = load_embeddings_by_dataset()
    print(f"  Loaded {len(embeddings)} datasets")
    print(f"  Total samples: {sum(len(e) for e in embeddings.values())}")

    print("\n" + "="*60)
    print("ANALYZING SAE CONCEPTS")
    print("="*60)

    # 1. Feature activations
    print("\n1. Analyzing feature activations...")
    activation_stats = analyze_feature_activations(model, embeddings)

    # Summary stats
    sparsities = [f['sparsity'] for f in activation_stats['feature_stats']]
    print(f"  Mean feature sparsity: {np.mean(sparsities):.3f}")
    print(f"  Alive features (>1% active): {sum(s > 0.01 for s in sparsities)}")

    # 2. Input correlations
    print("\n2. Computing input correlations...")
    correlations = analyze_input_correlations(model, embeddings)
    print(f"  Max absolute correlation: {np.abs(correlations).max():.3f}")
    print(f"  Mean absolute correlation: {np.abs(correlations).mean():.3f}")

    # 3. Matryoshka scale analysis
    print("\n3. Analyzing Matryoshka scales...")
    scale_analysis = analyze_matryoshka_scales(model, embeddings)
    print("  R² by scale:")
    for scale, stats in scale_analysis['aggregated'].items():
        print(f"    {scale:4d} features: R²={stats['mean_r2']:.3f} ± {stats['std_r2']:.3f}")

    # 4. Feature clustering
    print("\n4. Clustering dictionary features...")
    cluster_results = cluster_features(model, n_clusters=10)
    print("  Cluster sizes:", [c['size'] for c in cluster_results['clusters'].values()])

    # 5. Most interpretable features
    print("\n5. Finding most interpretable features...")
    interpretable = find_interpretable_features(model, embeddings, correlations, top_k=args.top_k)

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
