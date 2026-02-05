#!/usr/bin/env python3
"""
SAE Hyperparameter Sweep on TabPFN Embeddings.

Sweeps over:
- SAE type: L1, TopK, Matryoshka, Archetypal
- Expansion factor: 4x, 8x, 16x
- Sparsity penalty: 1e-4 to 1e-2
- With/without auxiliary loss

Uses wandb for experiment tracking.

Usage:
    # Single run (for testing)
    python scripts/sae_sweep.py --test

    # Full sweep with wandb
    python scripts/sae_sweep.py --sweep

    # Resume sweep
    python scripts/sae_sweep.py --sweep --sweep-id <id>
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    SAEResult,
    train_sae,
    measure_dictionary_richness,
    analyze_feature_geometry,
)

# Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# Sweep configuration
SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "richness_score", "goal": "maximize"},
    "parameters": {
        "sae_type": {"values": ["l1", "topk", "matryoshka", "archetypal"]},
        "expansion_factor": {"values": [4, 8, 16]},
        "sparsity_penalty": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "use_aux_loss": {"values": [True, False]},
        "topk": {"values": [16, 32, 64]},  # For topk SAE
        "archetypal_temp": {"distribution": "uniform", "min": 0.1, "max": 2.0},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "n_epochs": {"value": 100},
        "batch_size": {"values": [128, 256, 512]},
    },
}


def extract_tabpfn_embeddings(
    dataset_name: str = "adult",
    layer_idx: int = 17,  # Optimal layer from our analysis
    n_samples: int = 2000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract TabPFN embeddings at specified layer.

    Returns:
        embeddings: (n_samples, embedding_dim) array
    """
    import openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    if dataset_name == "adult":
        dataset = openml.datasets.get_dataset(1590, download_data=True)
    else:
        dataset = openml.datasets.get_dataset(dataset_name, download_data=True)

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Preprocess
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[[col]])

    X = X.values.astype(np.float32)
    y = y.values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if y.dtype == object or (hasattr(y.dtype, 'name') and y.dtype.name == 'category'):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Limit samples
    if len(X) > n_samples * 2:
        indices = np.random.permutation(len(X))[:n_samples * 2]
        X = X[indices]
        y = y[indices]

    # Split
    n = len(X)
    split = n // 2
    X_context, X_query = X[:split], X[split:]
    y_context, y_query = y[:split], y[split:]

    n_query = len(X_query)
    print(f"  Context: {X_context.shape}, Query: {X_query.shape}")

    # Extract embeddings with hooks
    from tabpfn import TabPFNClassifier
    import os

    worker_path = "/data/models/tabular_fm/tabpfn/tabpfn-v2.5-classifier-v2.5_real.ckpt"
    model_path = worker_path if os.path.exists(worker_path) else None

    kwargs = dict(device=device, n_estimators=1)
    if model_path:
        kwargs["model_path"] = model_path

    clf = TabPFNClassifier(**kwargs)
    clf.fit(X_context, y_context)

    # Get model - TabPFN uses transformer_encoder
    model = clf.model_
    model.eval()

    n_layers = len(model.transformer_encoder.layers)
    print(f"  TabPFN has {n_layers} transformer layers, extracting layer {layer_idx}")

    # Hook to capture activations
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            captured['embeddings'] = out.detach().float().cpu().numpy()

    handle = model.transformer_encoder.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = clf.predict_proba(X_query)
    finally:
        handle.remove()

    embeddings = captured.get('embeddings')
    if embeddings is None:
        raise ValueError("Failed to capture embeddings")

    # Handle shape - TabPFN returns (1, n_ctx+n_query+thinking, n_structure, hidden_dim)
    # Query samples are the last n_query along dim 1
    if embeddings.ndim == 4:
        # Shape: (1, seq, n_structure, hidden)
        query_act = embeddings[0, -n_query:, :, :]  # (n_query, n_structure, hidden)
        # Mean-pool over structure dimension
        embeddings = query_act.mean(axis=1)  # (n_query, hidden)
    elif embeddings.ndim == 3:
        # Shape: (1, seq, hidden)
        embeddings = embeddings[0, -n_query:, :]  # (n_query, hidden)

    print(f"  Extracted embeddings: {embeddings.shape}")
    return embeddings


def run_sae_training(config: Dict, embeddings: np.ndarray, device: str = "cpu") -> Dict:
    """
    Run a single SAE training with given config.

    Returns:
        Dict with all metrics
    """
    embedding_dim = embeddings.shape[1]
    hidden_dim = embedding_dim * config.get("expansion_factor", 4)

    # Build SAE config
    sae_type = config.get("sae_type", "l1")

    sae_config = SAEConfig(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        sparsity_penalty=config.get("sparsity_penalty", 1e-3),
        sparsity_type=sae_type,
        topk=config.get("topk", 32),
        matryoshka_dims=[hidden_dim // 8, hidden_dim // 4, hidden_dim // 2, hidden_dim],
        archetypal_simplex_temp=config.get("archetypal_temp", 1.0),
        use_aux_loss=config.get("use_aux_loss", True) and sae_type != "archetypal",
        aux_loss_coef=1e-2,
        dead_threshold=5000,
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 100),
    )

    print(f"\nTraining SAE: {sae_type}, {embedding_dim}D -> {hidden_dim}D")
    print(f"  sparsity={sae_config.sparsity_penalty:.1e}, aux_loss={sae_config.use_aux_loss}")

    # Train
    model, result = train_sae(embeddings, sae_config, device=device, verbose=True)

    # Compute metrics
    richness = measure_dictionary_richness(result)
    geometry = analyze_feature_geometry(result.dictionary, result.feature_activations)

    metrics = {
        # Basic metrics
        "reconstruction_loss": result.reconstruction_loss,
        "sparsity_loss": result.sparsity_loss,
        "total_loss": result.total_loss,
        # Feature metrics
        "alive_features": result.alive_features,
        "dead_features": result.dead_features,
        "alive_ratio": result.alive_features / hidden_dim,
        "mean_active_per_sample": result.mean_active_features,
        # Richness metrics
        "richness_score": richness["richness_score"],
        "dictionary_diversity": richness["dictionary_diversity"],
        "effective_dimensions": richness["effective_dimensions"],
        "sparsity": richness["sparsity"],
        # Geometry metrics
        "power_law_alpha": geometry["power_law_alpha"],
        "mean_clustering": geometry["mean_clustering"],
        "mean_coactivation": geometry["mean_coactivation"],
        # Config echo
        "sae_type": sae_type,
        "expansion_factor": config.get("expansion_factor", 4),
        "hidden_dim": hidden_dim,
    }

    return metrics, model, result


def wandb_sweep_train():
    """Training function for wandb sweep."""
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb not available")

    # Initialize wandb run
    run = wandb.init()
    config = dict(wandb.config)

    print(f"\n{'='*60}")
    print(f"Sweep run: {run.name}")
    print(f"Config: {config}")
    print("=" * 60)

    # Load embeddings (cached)
    embeddings_path = PROJECT_ROOT / "output" / "tabpfn_embeddings_adult_L17.npy"
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("Extracting embeddings (will cache for future runs)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = extract_tabpfn_embeddings(device=device)
        np.save(embeddings_path, embeddings)
        print(f"Cached embeddings to {embeddings_path}")

    # Train SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, model, result = run_sae_training(config, embeddings, device=device)

    # Log metrics
    wandb.log(metrics)

    # Log summary
    wandb.summary.update(metrics)

    print(f"\nResults: richness={metrics['richness_score']:.4f}, "
          f"alive={metrics['alive_ratio']:.1%}, recon={metrics['reconstruction_loss']:.4f}")


def run_sweep(sweep_id: Optional[str] = None, count: int = 50):
    """Run or resume a wandb sweep."""
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb not available. Install with: pip install wandb")

    if sweep_id is None:
        # Create new sweep
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project="tabular-sae",
            entity=None,  # Uses default entity
        )
        print(f"Created sweep: {sweep_id}")

    print(f"Running sweep {sweep_id} with {count} runs")
    wandb.agent(sweep_id, function=wandb_sweep_train, count=count, project="tabular-sae")


def run_test():
    """Run a single test training without wandb."""
    print("Running test SAE training (no wandb)")

    # Check for cached embeddings
    embeddings_path = PROJECT_ROOT / "output" / "tabpfn_embeddings_adult_L17.npy"

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("No cached embeddings found. Extracting...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            embeddings = extract_tabpfn_embeddings(device=device)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)
            print(f"Cached embeddings to {embeddings_path}")
        except Exception as e:
            print(f"Could not extract embeddings: {e}")
            print("Using random embeddings for testing...")
            embeddings = np.random.randn(1000, 192).astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    # Test each SAE type
    test_configs = [
        {"sae_type": "l1", "expansion_factor": 4, "sparsity_penalty": 1e-3, "use_aux_loss": True, "n_epochs": 30},
        {"sae_type": "topk", "expansion_factor": 4, "sparsity_penalty": 1e-3, "topk": 32, "use_aux_loss": True, "n_epochs": 30},
        {"sae_type": "matryoshka", "expansion_factor": 4, "sparsity_penalty": 1e-3, "use_aux_loss": True, "n_epochs": 30},
        {"sae_type": "archetypal", "expansion_factor": 2, "sparsity_penalty": 1e-2, "archetypal_temp": 0.5, "n_epochs": 30},
    ]

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for config in test_configs:
        try:
            metrics, _, _ = run_sae_training(config, embeddings, device=device)
            results.append(metrics)
        except Exception as e:
            print(f"Error with {config['sae_type']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"{'SAE Type':<15} {'Recon':>10} {'Alive':>10} {'Richness':>10} {'Diversity':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['sae_type']:<15} {r['reconstruction_loss']:>10.4f} {r['alive_ratio']:>10.1%} "
              f"{r['richness_score']:>10.4f} {r['dictionary_diversity']:>10.4f}")


def run_grid_search(output_dir: Path = None):
    """Run a grid search without wandb (for environments without internet)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings_path = PROJECT_ROOT / "output" / "tabpfn_embeddings_adult_L17.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = extract_tabpfn_embeddings(device=device)
        np.save(embeddings_path, embeddings)

    # Grid
    grid = {
        "sae_type": ["l1", "topk", "matryoshka"],
        "expansion_factor": [4, 8, 16],
        "sparsity_penalty": [1e-4, 1e-3, 1e-2],
        "use_aux_loss": [True, False],
    }

    # Generate all combinations
    from itertools import product
    keys = list(grid.keys())
    all_configs = [dict(zip(keys, v)) for v in product(*grid.values())]

    print(f"Running grid search with {len(all_configs)} configurations")

    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] {config}")
        config["n_epochs"] = 50
        config["topk"] = 32

        try:
            metrics, _, _ = run_sae_training(config, embeddings, device=device)
            metrics["config"] = config
            all_results.append(metrics)

            # Save incrementally
            with open(output_dir / "grid_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"Error: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Grid search complete. Results saved to {output_dir / 'grid_results.json'}")

    # Find best config
    if all_results:
        best = max(all_results, key=lambda x: x["richness_score"])
        print(f"\nBest config (richness={best['richness_score']:.4f}):")
        print(f"  {best['config']}")


def main():
    parser = argparse.ArgumentParser(description="SAE Hyperparameter Sweep")
    parser.add_argument("--test", action="store_true", help="Run single test without wandb")
    parser.add_argument("--sweep", action="store_true", help="Run wandb sweep")
    parser.add_argument("--grid", action="store_true", help="Run grid search without wandb")
    parser.add_argument("--sweep-id", type=str, default=None, help="Resume existing sweep")
    parser.add_argument("--count", type=int, default=50, help="Number of sweep runs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.sweep:
        run_sweep(sweep_id=args.sweep_id, count=args.count)
    elif args.grid:
        run_grid_search()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
