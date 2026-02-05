#!/usr/bin/env python3
"""
SAE Hyperparameter Sweep on TabPFN Embeddings using Optuna.

Sweeps over:
- SAE type: L1, TopK, Matryoshka, Archetypal
- Expansion factor: 4x, 8x, 16x
- Sparsity penalty: 1e-4 to 1e-2
- With/without auxiliary loss

Uses Optuna for Bayesian optimization (works offline).

Usage:
    # Single run (for testing)
    python scripts/sae_sweep.py --test

    # Run Optuna sweep (50 trials by default)
    python scripts/sae_sweep.py --optuna --n-trials 50

    # Resume existing study
    python scripts/sae_sweep.py --optuna --study-name my_study

    # Generate response surface plots
    python scripts/sae_sweep.py --plot --study-name my_study
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

# Check for optuna
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_parallel_coordinate,
        plot_slice,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Install with: pip install optuna")

# Optional wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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

    # Compute metrics - pass embeddings and model for proper explained variance
    richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)
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
        # New standardized metrics
        "l0_sparsity": richness["l0_sparsity"],  # Mean active features per sample
        "l0_sparsity_frac": richness["l0_sparsity_frac"],  # As fraction
        "explained_variance": richness.get("explained_variance", 0.0),  # R²
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


def create_optuna_objective(embeddings: np.ndarray, device: str = "cuda"):
    """
    Create an Optuna objective function for SAE hyperparameter optimization.

    Uses TPE (Tree-structured Parzen Estimator) for Bayesian optimization.
    """
    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters
        sae_type = trial.suggest_categorical("sae_type", ["l1", "topk", "matryoshka", "archetypal"])
        expansion_factor = trial.suggest_categorical("expansion_factor", [4, 8, 16])
        sparsity_penalty = trial.suggest_float("sparsity_penalty", 1e-4, 1e-2, log=True)
        use_aux_loss = trial.suggest_categorical("use_aux_loss", [True, False])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        # Type-specific parameters
        if sae_type == "topk":
            topk = trial.suggest_categorical("topk", [16, 32, 64])
        else:
            topk = 32

        if sae_type == "archetypal":
            archetypal_temp = trial.suggest_float("archetypal_temp", 0.1, 2.0)
            use_aux_loss = False  # Not applicable for archetypal
        else:
            archetypal_temp = 1.0

        config = {
            "sae_type": sae_type,
            "expansion_factor": expansion_factor,
            "sparsity_penalty": sparsity_penalty,
            "use_aux_loss": use_aux_loss,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "topk": topk,
            "archetypal_temp": archetypal_temp,
            "n_epochs": 100,
        }

        try:
            metrics, _, _ = run_sae_training(config, embeddings, device=device)

            # Store all metrics as user attributes
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    trial.set_user_attr(key, value)

            # Primary objective: maximize richness score
            return metrics["richness_score"]

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0  # Return worst possible score

    return objective


def run_optuna_sweep(
    study_name: str = "sae_sweep",
    n_trials: int = 50,
    output_dir: Path = None,
    device: str = "cuda",
):
    """
    Run Optuna hyperparameter sweep with Bayesian optimization.

    Args:
        study_name: Name for the Optuna study (used for persistence)
        n_trials: Number of trials to run
        output_dir: Output directory for results and plots
        device: Device to use for training
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("optuna not available. Install with: pip install optuna")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings_path = PROJECT_ROOT / "output" / "tabpfn_embeddings_adult_L17.npy"
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("Extracting embeddings...")
        embeddings = extract_tabpfn_embeddings(device=device)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"Cached embeddings to {embeddings_path}")

    print(f"Embeddings shape: {embeddings.shape}")

    # Create or load study with SQLite storage for persistence
    storage_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize richness score
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    print(f"\n{'='*60}")
    print(f"Optuna Study: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"Running {n_trials} new trials...")
    print("=" * 60)

    # Create objective function
    objective = create_optuna_objective(embeddings, device=device)

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best richness score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best config
    best_config = {
        "params": study.best_params,
        "value": study.best_value,
        "trial_number": study.best_trial.number,
        "user_attrs": study.best_trial.user_attrs,
    }
    with open(output_dir / f"{study_name}_best.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # Generate plots
    print("\nGenerating response surface plots...")
    generate_optuna_plots(study, output_dir, study_name)

    return study


def generate_optuna_plots(study: "optuna.Study", output_dir: Path, study_name: str):
    """
    Generate response surface and other visualization plots for the Optuna study.

    Creates publication-ready figures for the appendix.
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, skipping plots")
        return

    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimization history
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(plots_dir / f"{study_name}_optimization_history.png"), scale=2)
        fig.write_html(str(plots_dir / f"{study_name}_optimization_history.html"))
        print(f"  Saved: optimization_history")
    except Exception as e:
        print(f"  Failed optimization_history: {e}")

    # 2. Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.write_image(str(plots_dir / f"{study_name}_param_importances.png"), scale=2)
        fig.write_html(str(plots_dir / f"{study_name}_param_importances.html"))
        print(f"  Saved: param_importances")
    except Exception as e:
        print(f"  Failed param_importances: {e}")

    # 3. Parallel coordinate plot
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(plots_dir / f"{study_name}_parallel_coordinate.png"), scale=2)
        fig.write_html(str(plots_dir / f"{study_name}_parallel_coordinate.html"))
        print(f"  Saved: parallel_coordinate")
    except Exception as e:
        print(f"  Failed parallel_coordinate: {e}")

    # 4. Contour plots (response surfaces) for key parameter pairs
    param_pairs = [
        ("sparsity_penalty", "expansion_factor"),
        ("sparsity_penalty", "learning_rate"),
        ("expansion_factor", "learning_rate"),
    ]

    for param1, param2 in param_pairs:
        try:
            fig = plot_contour(study, params=[param1, param2])
            fig.write_image(str(plots_dir / f"{study_name}_contour_{param1}_{param2}.png"), scale=2)
            print(f"  Saved: contour_{param1}_{param2}")
        except Exception as e:
            print(f"  Failed contour_{param1}_{param2}: {e}")

    # 5. Slice plots for each parameter
    try:
        fig = plot_slice(study)
        fig.write_image(str(plots_dir / f"{study_name}_slice.png"), scale=2)
        fig.write_html(str(plots_dir / f"{study_name}_slice.html"))
        print(f"  Saved: slice")
    except Exception as e:
        print(f"  Failed slice: {e}")

    # 6. Custom matplotlib summary figure
    try:
        create_summary_figure(study, plots_dir, study_name)
        print(f"  Saved: summary figure")
    except Exception as e:
        print(f"  Failed summary figure: {e}")

    # 7. L0 sparsity and explained variance metrics figure
    try:
        create_metrics_figure(study, plots_dir, study_name)
        print(f"  Saved: metrics figure (L0 sparsity, explained variance)")
    except Exception as e:
        print(f"  Failed metrics figure: {e}")

    print(f"\nPlots saved to: {plots_dir}")


def create_summary_figure(study: "optuna.Study", output_dir: Path, study_name: str):
    """Create a matplotlib summary figure for the paper appendix."""
    import matplotlib.pyplot as plt

    # Extract data from trials
    trials_df = study.trials_dataframe()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel A: Optimization history
    ax = axes[0, 0]
    values = [t.value for t in study.trials if t.value is not None]
    best_values = [max(values[:i+1]) for i in range(len(values))]
    ax.plot(values, 'o', alpha=0.5, markersize=4, label='Trial value')
    ax.plot(best_values, '-', linewidth=2, color='red', label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Richness Score')
    ax.set_title('(A) Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: SAE type comparison
    ax = axes[0, 1]
    sae_types = trials_df['params_sae_type'].unique()
    sae_scores = [trials_df[trials_df['params_sae_type'] == t]['value'].mean() for t in sae_types]
    sae_stds = [trials_df[trials_df['params_sae_type'] == t]['value'].std() for t in sae_types]
    bars = ax.bar(sae_types, sae_scores, yerr=sae_stds, capsize=5, alpha=0.7)
    ax.set_ylabel('Richness Score')
    ax.set_title('(B) SAE Type Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Expansion factor effect
    ax = axes[0, 2]
    exp_factors = sorted(trials_df['params_expansion_factor'].unique())
    exp_scores = [trials_df[trials_df['params_expansion_factor'] == e]['value'].mean() for e in exp_factors]
    exp_stds = [trials_df[trials_df['params_expansion_factor'] == e]['value'].std() for e in exp_factors]
    ax.errorbar(exp_factors, exp_scores, yerr=exp_stds, fmt='o-', capsize=5, markersize=8)
    ax.set_xlabel('Expansion Factor')
    ax.set_ylabel('Richness Score')
    ax.set_title('(C) Expansion Factor Effect')
    ax.grid(True, alpha=0.3)

    # Panel D: Sparsity penalty vs richness (scatter)
    ax = axes[1, 0]
    valid_trials = trials_df[trials_df['value'].notna()]
    scatter = ax.scatter(
        valid_trials['params_sparsity_penalty'],
        valid_trials['value'],
        c=valid_trials['params_expansion_factor'],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    ax.set_xscale('log')
    ax.set_xlabel('Sparsity Penalty')
    ax.set_ylabel('Richness Score')
    ax.set_title('(D) Sparsity Penalty Effect')
    plt.colorbar(scatter, ax=ax, label='Expansion Factor')
    ax.grid(True, alpha=0.3)

    # Panel E: Aux loss effect
    ax = axes[1, 1]
    aux_true = trials_df[trials_df['params_use_aux_loss'] == True]['value']
    aux_false = trials_df[trials_df['params_use_aux_loss'] == False]['value']
    ax.boxplot([aux_false.dropna(), aux_true.dropna()], labels=['No Aux Loss', 'With Aux Loss'])
    ax.set_ylabel('Richness Score')
    ax.set_title('(E) Auxiliary Loss Effect')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel F: Best configuration summary
    ax = axes[1, 2]
    ax.axis('off')
    best = study.best_trial
    text = f"Best Configuration\n" + "=" * 30 + "\n\n"
    text += f"Richness Score: {best.value:.4f}\n\n"
    text += "Parameters:\n"
    for key, value in best.params.items():
        if isinstance(value, float):
            text += f"  {key}: {value:.4e}\n"
        else:
            text += f"  {key}: {value}\n"

    if best.user_attrs:
        text += "\nMetrics:\n"
        for key in ['reconstruction_loss', 'alive_ratio', 'dictionary_diversity']:
            if key in best.user_attrs:
                text += f"  {key}: {best.user_attrs[key]:.4f}\n"

    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('(F) Best Configuration')

    plt.tight_layout()
    plt.savefig(output_dir / f"{study_name}_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{study_name}_summary.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_figure(study: "optuna.Study", output_dir: Path, study_name: str):
    """
    Create response surface plots for L0 sparsity and explained variance.

    These are standard metrics from the SAE interpretability literature:
    - L0 sparsity: Anthropic (2024), "Scaling Monosemanticity"
    - Explained variance: Standard R² metric
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # Extract trial data
    trials_data = []
    for t in study.trials:
        if t.value is not None and t.user_attrs:
            trials_data.append({
                'sparsity_penalty': t.params.get('sparsity_penalty'),
                'expansion_factor': t.params.get('expansion_factor'),
                'sae_type': t.params.get('sae_type'),
                'richness': t.value,
                'l0_sparsity': t.user_attrs.get('l0_sparsity', 0),
                'explained_variance': t.user_attrs.get('explained_variance', 0),
                'alive_ratio': t.user_attrs.get('alive_ratio', 0),
            })

    if not trials_data:
        print("No trials with user_attrs found")
        return

    import pandas as pd
    df = pd.DataFrame(trials_data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: L0 sparsity analysis
    # Panel A: L0 vs richness (Pareto front)
    ax = axes[0, 0]
    for sae_type in df['sae_type'].unique():
        mask = df['sae_type'] == sae_type
        ax.scatter(df[mask]['l0_sparsity'], df[mask]['richness'],
                   label=sae_type, alpha=0.7, s=50)
    ax.set_xlabel('L0 Sparsity (mean active features)')
    ax.set_ylabel('Richness Score')
    ax.set_title('(A) L0 Sparsity vs Richness')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: L0 by SAE type
    ax = axes[0, 1]
    sae_types = df['sae_type'].unique()
    l0_by_type = [df[df['sae_type'] == t]['l0_sparsity'].values for t in sae_types]
    bp = ax.boxplot(l0_by_type, tick_labels=sae_types, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sae_types)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('L0 Sparsity')
    ax.set_title('(B) L0 Sparsity by SAE Type')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: L0 vs expansion factor
    ax = axes[0, 2]
    for exp in sorted(df['expansion_factor'].unique()):
        mask = df['expansion_factor'] == exp
        ax.scatter(df[mask]['sparsity_penalty'], df[mask]['l0_sparsity'],
                   label=f'{exp}x', alpha=0.7, s=50)
    ax.set_xscale('log')
    ax.set_xlabel('Sparsity Penalty')
    ax.set_ylabel('L0 Sparsity')
    ax.set_title('(C) L0 vs Sparsity Penalty')
    ax.legend(title='Expansion')
    ax.grid(True, alpha=0.3)

    # Row 2: Explained variance analysis
    # Panel D: Explained variance vs richness
    ax = axes[1, 0]
    for sae_type in df['sae_type'].unique():
        mask = df['sae_type'] == sae_type
        ax.scatter(df[mask]['explained_variance'], df[mask]['richness'],
                   label=sae_type, alpha=0.7, s=50)
    ax.set_xlabel('Explained Variance (R²)')
    ax.set_ylabel('Richness Score')
    ax.set_title('(D) Explained Variance vs Richness')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel E: Explained variance by SAE type
    ax = axes[1, 1]
    ev_by_type = [df[df['sae_type'] == t]['explained_variance'].values for t in sae_types]
    bp = ax.boxplot(ev_by_type, tick_labels=sae_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Explained Variance (R²)')
    ax.set_title('(E) Explained Variance by SAE Type')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel F: Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')

    # Compute summary stats
    text = "Metric Summary by SAE Type\n" + "=" * 35 + "\n\n"
    text += f"{'Type':<12} {'L0':>8} {'R²':>8} {'Rich':>8}\n"
    text += "-" * 35 + "\n"
    for sae_type in sae_types:
        mask = df['sae_type'] == sae_type
        l0_mean = df[mask]['l0_sparsity'].mean()
        ev_mean = df[mask]['explained_variance'].mean()
        rich_mean = df[mask]['richness'].mean()
        text += f"{sae_type:<12} {l0_mean:>8.1f} {ev_mean:>8.3f} {rich_mean:>8.3f}\n"

    text += "\n" + "=" * 35 + "\n"
    text += "\nReferences:\n"
    text += "- L0: Anthropic (2024)\n"
    text += "- R²: Standard metric\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.set_title('(F) Summary Statistics')

    plt.tight_layout()
    plt.savefig(output_dir / f"{study_name}_metrics.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{study_name}_metrics.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def load_and_plot_study(study_name: str, output_dir: Path = None):
    """Load an existing study and generate plots."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_sweep"

    storage_path = output_dir / f"{study_name}.db"
    if not storage_path.exists():
        raise FileNotFoundError(f"Study not found: {storage_path}")

    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    print(f"Loaded study: {study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")

    generate_optuna_plots(study, output_dir, study_name)
    return study


def main():
    parser = argparse.ArgumentParser(description="SAE Hyperparameter Sweep with Optuna")
    parser.add_argument("--test", action="store_true", help="Run single test training")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter sweep")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing study")
    parser.add_argument("--study-name", type=str, default="sae_sweep", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    # Legacy options
    parser.add_argument("--sweep", action="store_true", help="(Legacy) Run wandb sweep")
    parser.add_argument("--grid", action="store_true", help="(Legacy) Run grid search")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.optuna:
        run_optuna_sweep(
            study_name=args.study_name,
            n_trials=args.n_trials,
            device=args.device,
        )
    elif args.plot:
        load_and_plot_study(args.study_name)
    elif args.sweep:
        print("Note: Use --optuna instead of --sweep for Bayesian optimization")
        run_sweep(count=args.n_trials)
    elif args.grid:
        print("Note: Use --optuna instead of --grid for more efficient search")
        print("Grid search is deprecated. Running Optuna instead.")
        run_optuna_sweep(study_name=args.study_name, n_trials=args.n_trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
