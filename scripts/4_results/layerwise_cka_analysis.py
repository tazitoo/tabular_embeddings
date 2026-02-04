#!/usr/bin/env python3
"""
Layer-wise CKA analysis within a single model.

Extracts embeddings from each transformer layer and computes pairwise CKA
to visualize how representations evolve through the network.

Usage:
    python scripts/4_results/layerwise_cka_analysis.py --model tabpfn --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.similarity import centered_kernel_alignment


def load_dataset(dataset_name: str, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from TabArena or OpenML."""
    import openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    # TabArena suite ID
    TABARENA_SUITE_ID = 457

    # Get dataset from OpenML
    try:
        dataset = openml.datasets.get_dataset(dataset_name, download_data=True)
    except:
        # Try by ID if name doesn't work
        suite = openml.study.get_suite(TABARENA_SUITE_ID)
        # Find dataset in suite
        for did in suite.data:
            d = openml.datasets.get_dataset(did, download_data=False)
            if d.name == dataset_name:
                dataset = openml.datasets.get_dataset(did, download_data=True)
                break
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Encode categorical features
    import pandas as pd
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[[col]])

    # Convert to numpy
    X = X.values.astype(np.float32)
    y = y.values

    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels if needed
    if y.dtype == object or (hasattr(y.dtype, 'name') and y.dtype.name == 'category'):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Limit samples
    if len(X) > max_samples * 2:
        indices = np.random.permutation(len(X))[:max_samples * 2]
        X = X[indices]
        y = y[indices]

    # Split into context and query
    n = len(X)
    split = n // 2
    X_context, X_query = X[:split], X[split:]
    y_context, y_query = y[:split], y[split:]

    return X_context, y_context, X_query, y_query


def extract_tabpfn_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all TabPFN transformer layers."""
    from tabpfn import TabPFNClassifier

    # Check for local checkpoint
    worker_path = "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-classifier-v2.5_real.ckpt"
    import os
    model_path = worker_path if os.path.exists(worker_path) else None

    kwargs = dict(device=device, n_estimators=2)
    if model_path:
        kwargs["model_path"] = model_path

    clf = TabPFNClassifier(**kwargs)
    clf.fit(X_context, y_context)

    model = clf.model_
    n_layers = len(model.transformer_encoder.layers)
    print(f"TabPFN has {n_layers} transformer layers")

    # Register hooks for all layers
    captured = {}
    handles = []
    n_query = len(X_query)

    for i, layer in enumerate(model.transformer_encoder.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured[f"layer_{layer_idx}"] = output.detach().cpu().numpy()
            return hook_fn
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # Forward pass
    try:
        with torch.no_grad():
            _ = clf.predict_proba(X_query)
    finally:
        for handle in handles:
            handle.remove()

    # Process captured activations
    layer_embeddings = {}
    for key, act in captured.items():
        # Shape: (1, n_ctx+n_query+thinking, n_structure, hidden_dim)
        # Query samples are the last n_query along dim 1
        query_act = act[0, -n_query:, :, :]  # (n_query, n_structure, hidden)
        # Mean-pool over structure dimension
        emb = query_act.mean(axis=1)  # (n_query, hidden)
        layer_embeddings[key] = emb

    return layer_embeddings


def extract_mitra_all_layers(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all Mitra transformer layers."""
    from autogluon.tabular.models.mitra.mitra_model import MitraModel

    # Create and fit Mitra
    import pandas as pd
    import tempfile

    df_train = pd.DataFrame(X_context)
    df_train['target'] = y_context
    df_query = pd.DataFrame(X_query)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = MitraModel(
            path=tmpdir,
            name="mitra_layerwise",
            problem_type="binary",
        )
        model.fit(train_data=df_train, label='target')

        # Access the underlying transformer
        inner_model = model.model.model

        # Mitra uses a different architecture - need to inspect
        print(f"Mitra model structure: {type(inner_model)}")

        # For now, just extract from available hooks
        # This needs more investigation into Mitra's specific architecture

    return {}


def compute_layerwise_cka(layer_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise CKA between all layers."""
    layer_names = sorted(layer_embeddings.keys(), key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)

    cka_matrix = np.zeros((n_layers, n_layers))

    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            if i <= j:
                cka = centered_kernel_alignment(
                    layer_embeddings[name_i],
                    layer_embeddings[name_j]
                )
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    return cka_matrix, layer_names


def plot_layerwise_cka(cka_matrix: np.ndarray, layer_names: List[str], output_path: Path, model_name: str):
    """Create heatmap of layer-wise CKA."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create labels like "L0", "L1", etc.
    labels = [f"L{i}" for i in range(len(layer_names))]

    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'CKA Similarity', 'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 8}
    )

    ax.set_title(f'{model_name} Layer-wise CKA Similarity', fontsize=14, pad=15)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close()


def plot_cka_by_distance(cka_matrix: np.ndarray, output_path: Path, model_name: str):
    """Plot CKA vs layer distance to show representation drift."""
    import matplotlib.pyplot as plt

    n_layers = cka_matrix.shape[0]
    distances = []
    cka_values = []

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            distances.append(j - i)
            cka_values.append(cka_matrix[i, j])

    # Average CKA by distance
    max_dist = n_layers - 1
    avg_cka = []
    for d in range(1, max_dist + 1):
        vals = [cka_values[k] for k, dist in enumerate(distances) if dist == d]
        avg_cka.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_dist + 1), avg_cka, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer Distance', fontsize=12)
    ax.set_ylabel('Average CKA Similarity', fontsize=12)
    ax.set_title(f'{model_name}: CKA vs Layer Distance', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark the 2/3 point
    two_thirds_layer = int(n_layers * 2 / 3)
    ax.axvline(x=two_thirds_layer, color='red', linestyle='--', alpha=0.5,
               label=f'2/3 depth ({two_thirds_layer} layers)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer-wise CKA analysis")
    parser.add_argument("--model", type=str, default="tabpfn",
                        choices=["tabpfn", "mitra"],
                        help="Model to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples for analysis")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name from TabArena/OpenML (use synthetic if not specified)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    if args.dataset:
        # Load real dataset
        print(f"Loading dataset: {args.dataset}")
        X_context, y_context, X_query, _ = load_dataset(args.dataset, max_samples=args.n_samples)
        dataset_name = args.dataset
    else:
        # Generate synthetic data for analysis
        n_features = 20
        n_context = args.n_samples
        n_query = args.n_samples

        X_context = np.random.randn(n_context, n_features).astype(np.float32)
        y_context = (np.random.rand(n_context) > 0.5).astype(int)
        X_query = np.random.randn(n_query, n_features).astype(np.float32)
        dataset_name = "synthetic"

    print(f"Extracting layer-wise embeddings from {args.model}...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Context: {X_context.shape}, Query: {X_query.shape}")

    if args.model == "tabpfn":
        layer_embeddings = extract_tabpfn_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    elif args.model == "mitra":
        layer_embeddings = extract_mitra_all_layers(
            X_context, y_context, X_query, device=args.device
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if not layer_embeddings:
        print("No layer embeddings extracted!")
        return

    print(f"\nExtracted embeddings from {len(layer_embeddings)} layers:")
    for name, emb in sorted(layer_embeddings.items(), key=lambda x: int(x[0].split('_')[1])):
        print(f"  {name}: {emb.shape}")

    # Compute layer-wise CKA
    print("\nComputing layer-wise CKA...")
    cka_matrix, layer_names = compute_layerwise_cka(layer_embeddings)

    # Build output filename suffix
    suffix = f"{args.model}_{dataset_name}"

    # Save results
    np.savez(
        output_dir / f"layerwise_cka_{suffix}.npz",
        cka_matrix=cka_matrix,
        layer_names=layer_names,
        dataset=dataset_name,
    )
    print(f"Saved: {output_dir / f'layerwise_cka_{suffix}.npz'}")

    # Plot heatmap
    title = f"{args.model.upper()} Layer-wise CKA ({dataset_name})"
    plot_layerwise_cka(
        cka_matrix,
        layer_names,
        output_dir / f"layerwise_cka_heatmap_{suffix}",
        title
    )

    # Plot CKA by distance
    plot_cka_by_distance(
        cka_matrix,
        output_dir / f"layerwise_cka_distance_{suffix}.png",
        title
    )

    # Print summary
    print("\n" + "=" * 60)
    print("LAYER-WISE CKA SUMMARY")
    print("=" * 60)

    n_layers = len(layer_names)
    print(f"\nFirst layer vs others:")
    for i in range(1, n_layers):
        print(f"  L0 vs L{i}: {cka_matrix[0, i]:.3f}")

    print(f"\nLast layer vs others:")
    for i in range(n_layers - 1):
        print(f"  L{i} vs L{n_layers-1}: {cka_matrix[i, n_layers-1]:.3f}")

    # Find where representation is most stable (highest CKA with neighbors)
    neighbor_cka = []
    for i in range(1, n_layers - 1):
        avg = (cka_matrix[i, i-1] + cka_matrix[i, i+1]) / 2
        neighbor_cka.append((i, avg))

    if neighbor_cka:
        most_stable = max(neighbor_cka, key=lambda x: x[1])
        print(f"\nMost stable layer (highest neighbor CKA): L{most_stable[0]} (avg CKA={most_stable[1]:.3f})")
        print(f"2/3 depth would be layer: L{int(n_layers * 2 / 3)}")


if __name__ == "__main__":
    main()
