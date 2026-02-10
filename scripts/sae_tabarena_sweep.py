#!/usr/bin/env python3
"""
SAE Architecture Comparison on TabArena.

1. Split TabArena datasets into train/test (70/30)
2. Pool embeddings from train datasets
3. Run Optuna HP sweep for each SAE type
4. Evaluate best configs on test datasets

Usage:
    # Define splits and check data
    python scripts/sae_tabarena_sweep.py --setup

    # Run HP sweep for one model
    python scripts/sae_tabarena_sweep.py --model tabpfn --n-trials 30

    # Run sweep on intermediate layer with context_size as HP
    python scripts/sae_tabarena_sweep.py --model tabpfn --layer 16 \
        --context-sizes 200,600,1000 --n-trials 30

    # Evaluate on test set
    python scripts/sae_tabarena_sweep.py --model tabpfn --evaluate
"""

import argparse
import json
import hashlib
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
    compare_dictionaries,
)

# Fixed random seed for reproducible splits
SPLIT_SEED = 42


def get_available_datasets(model_name: str) -> List[str]:
    """
    Discover available embeddings for a model.

    Dynamically finds all extracted embeddings instead of hardcoding.
    """
    path = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / model_name
    if not path.exists():
        raise ValueError(f"No embeddings directory found: {path}")

    datasets = []
    for f in path.glob("tabarena_*.npz"):
        ds_name = f.stem.replace("tabarena_", "")
        datasets.append(ds_name)

    return sorted(datasets)


def get_tabarena_splits(model_name: str = "tabpfn") -> Tuple[List[str], List[str]]:
    """
    Get train/test split of TabArena datasets.

    Split is deterministic (based on hash of dataset name).
    Roughly 70% train, 30% test.

    Dynamically discovers available embeddings for the model.
    """
    all_datasets = get_available_datasets(model_name)

    if not all_datasets:
        raise ValueError(f"No datasets found for {model_name}")

    train_datasets = []
    test_datasets = []

    for ds in all_datasets:
        # Deterministic split based on hash
        h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:  # 70% train
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)

    return train_datasets, test_datasets


def load_embeddings(model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load cached embeddings for a model/dataset."""
    path = PROJECT_ROOT / f"output/embeddings/tabarena/{model_name}/tabarena_{dataset_name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data['embeddings'].astype(np.float32)
    return None


def pool_embeddings(
    model_name: str,
    datasets: List[str],
    max_per_dataset: int = 500,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Pool embeddings from multiple datasets.

    Args:
        model_name: Model to load embeddings for
        datasets: List of dataset names
        max_per_dataset: Max samples per dataset (for balance)
        normalize: Whether to normalize embeddings

    Returns:
        (pooled_embeddings, dataset_counts)
    """
    all_embeddings = []
    dataset_counts = {}

    for ds in datasets:
        emb = load_embeddings(model_name, ds)
        if emb is None:
            print(f"  Warning: No embeddings for {ds}")
            continue

        # Subsample if too large
        if len(emb) > max_per_dataset:
            np.random.seed(SPLIT_SEED)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]

        all_embeddings.append(emb)
        dataset_counts[ds] = len(emb)

    if not all_embeddings:
        raise ValueError(f"No embeddings found for {model_name}")

    pooled = np.concatenate(all_embeddings, axis=0)

    if normalize:
        # Per-dimension normalization
        std = pooled.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        pooled = pooled / std

    return pooled, dataset_counts


def compute_stability(
    embeddings: np.ndarray,
    config: SAEConfig,
    n_runs: int = 2,
    return_per_scale: bool = False,
    device: str = "cpu",
) -> float:
    """
    Train SAE twice and measure dictionary stability.

    For Matryoshka SAEs, also measures stability at each nested scale.

    Args:
        embeddings: Training data
        config: SAE configuration
        n_runs: Number of training runs
        return_per_scale: If True, return dict with per-scale stability (Matryoshka only)

    Returns:
        Overall stability score (or dict with per-scale if return_per_scale=True)
    """
    dicts = []
    for seed in [123, 456][:n_runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, result = train_sae(embeddings, config, device=device, verbose=False)
        dicts.append(result.dictionary)

    comp = compare_dictionaries(dicts[0], dicts[1])
    overall_stability = comp['mean_best_match_a']

    # For Matryoshka, compute per-scale stability
    if config.sparsity_type == "matryoshka" and return_per_scale:
        mat_dims = config.matryoshka_dims or [32, 64, 128, 256]
        mat_dims = [d for d in mat_dims if d <= config.hidden_dim]

        per_scale = {}
        prev_dim = 0
        for dim in mat_dims:
            # Compare features in this scale only
            scale_dict1 = dicts[0][prev_dim:dim]
            scale_dict2 = dicts[1][prev_dim:dim]
            if len(scale_dict1) > 0:
                scale_comp = compare_dictionaries(scale_dict1, scale_dict2)
                per_scale[f"scale_{prev_dim}_{dim}"] = scale_comp['mean_best_match_a']
            prev_dim = dim

        # Remaining features
        if prev_dim < config.hidden_dim:
            scale_dict1 = dicts[0][prev_dim:]
            scale_dict2 = dicts[1][prev_dim:]
            scale_comp = compare_dictionaries(scale_dict1, scale_dict2)
            per_scale[f"scale_{prev_dim}_{config.hidden_dim}"] = scale_comp['mean_best_match_a']

        return {
            "overall": overall_stability,
            "per_scale": per_scale,
        }

    return overall_stability


def build_sae_config(
    embeddings: np.ndarray,
    sae_type: str,
    expansion: int = 4,
    sparsity_penalty: float = 1e-3,
    learning_rate: float = 1e-3,
    topk: int = 32,
    archetypal_n_archetypes: int = 500,
    archetypal_temp: float = 0.1,
    archetypal_relaxation: float = 0.0,
    n_epochs: int = 100,
) -> SAEConfig:
    """Build SAE config from parameters."""
    input_dim = embeddings.shape[1]
    hidden_dim = input_dim * expansion

    # Scale batch size and epochs for large hidden dims to stay within GPU memory
    # and keep training time reasonable (~30 min/trial max target).
    # For 4096-dim input: expansion=2 → 8192 hidden → 67M params → 2h+ at 100 epochs.
    # Reducing to 50 epochs halves training time; SAE convergence is stable by then.
    if hidden_dim >= 8192:
        batch_size = 64
        n_epochs = min(n_epochs, 50)
    else:
        batch_size = 128

    # Ghost grads should be disabled for Matryoshka-Archetypal SAEs.
    # Multi-scale loss keeps all features alive; ghost grads fire during
    # early warmup before features stabilize, creating garbage features
    # in higher scale bands that actively hurt reconstruction.
    use_ghost_grads = False if sae_type == "matryoshka_archetypal" else True

    return SAEConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_penalty=sparsity_penalty,
        sparsity_type=sae_type,
        topk=topk,
        archetypal_n_archetypes=archetypal_n_archetypes,
        archetypal_simplex_temp=archetypal_temp,
        archetypal_relaxation=archetypal_relaxation,
        archetypal_use_centroids=True,
        use_ghost_grads=use_ghost_grads,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def run_sae_trial(
    embeddings: np.ndarray,
    sae_type: str,
    expansion: int = 4,
    sparsity_penalty: float = 1e-3,
    learning_rate: float = 1e-3,
    topk: int = 32,
    archetypal_n_archetypes: int = 500,
    archetypal_temp: float = 0.1,
    archetypal_relaxation: float = 0.0,
    n_epochs: int = 100,
    measure_stability: bool = True,
    return_model: bool = False,
    seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """Run a single SAE training trial and return metrics (and optionally model)."""
    config = build_sae_config(
        embeddings, sae_type, expansion, sparsity_penalty, learning_rate,
        topk, archetypal_n_archetypes, archetypal_temp, archetypal_relaxation, n_epochs
    )

    # Train and evaluate
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, result = train_sae(embeddings, config, device=device, verbose=False)
    richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)

    metrics = {
        "sae_type": sae_type,
        "expansion": expansion,
        "sparsity_penalty": sparsity_penalty,
        "learning_rate": learning_rate,
        "r2": richness.get("explained_variance", 0.0),
        "l0_sparsity": richness["l0_sparsity"],
        "richness_score": richness["richness_score"],
        "reconstruction_loss": result.reconstruction_loss,
        "alive_features": result.alive_features,
    }

    if measure_stability:
        stability = compute_stability(embeddings, config, n_runs=2, device=device)
        metrics["stability"] = stability

    if return_model:
        return metrics, model, config
    return metrics


def save_sae_model(
    model,
    config: SAEConfig,
    metrics: Dict,
    params: Dict,
    save_path: Path,
):
    """Save SAE model checkpoint with config and metrics."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "params": params,
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved model to {save_path}")


def validate_and_save(
    embeddings_by_ctx: Dict[int, np.ndarray],
    sae_type: str,
    study,
    output_dir: Path,
    tolerance: float = 0.05,
    max_retries: int = 5,
    extra_trials_per_retry: int = 3,
    device: str = "cpu",
) -> Tuple[Dict, Optional[Path]]:
    """
    Validate best config by retraining, check metrics match, save if valid.

    If validation fails (metrics differ by more than tolerance), adds result
    as feedback and continues the sweep until convergence.

    Args:
        embeddings_by_ctx: Dict mapping context_size -> pooled embeddings
        sae_type: SAE architecture type
        study: Optuna study object
        output_dir: Directory to save model
        tolerance: Max allowed relative difference in score (default 5%)
        max_retries: Max validation attempts before giving up
        extra_trials_per_retry: Additional Optuna trials per failed validation

    Returns:
        (best_config_dict, model_path) - model_path is None if validation failed
    """
    import optuna

    ctx_keys = sorted(embeddings_by_ctx.keys())
    objective = create_optuna_objective(embeddings_by_ctx, sae_type, device=device)

    for attempt in range(max_retries):
        best_trial = study.best_trial
        best_params = study.best_params
        expected_score = study.best_value
        expected_r2 = best_trial.user_attrs.get("r2", 0)
        expected_stability = best_trial.user_attrs.get("stability", 0)

        print(f"\n  Validation attempt {attempt + 1}/{max_retries}")
        print(f"    Expected: score={expected_score:.4f}, R²={expected_r2:.4f}, stability={expected_stability:.4f}")

        # Select embeddings for this trial's context_size
        best_ctx = best_params.get("context_size", ctx_keys[0])
        embeddings = embeddings_by_ctx[best_ctx]
        if len(ctx_keys) > 1:
            print(f"    Context size: {best_ctx}")

        # Extract params for this SAE type
        expansion = best_params.get("expansion", 4)
        sparsity_penalty = best_params.get("sparsity_penalty", 1e-3)
        learning_rate = best_params.get("learning_rate", 1e-3)
        topk = best_params.get("topk", 32)
        archetypal_n = best_params.get("archetypal_n", 500)
        archetypal_temp = best_params.get("archetypal_temp", 0.1)
        archetypal_relax = best_params.get("archetypal_relaxation", 0.0)

        # Train with a different seed to test robustness
        validation_seed = 12345 + attempt
        metrics, model, config = run_sae_trial(
            embeddings,
            sae_type=sae_type,
            expansion=expansion,
            sparsity_penalty=sparsity_penalty,
            learning_rate=learning_rate,
            topk=topk,
            archetypal_n_archetypes=archetypal_n,
            archetypal_temp=archetypal_temp,
            archetypal_relaxation=archetypal_relax,
            n_epochs=100,
            measure_stability=True,
            return_model=True,
            seed=validation_seed,
            device=device,
        )

        # Compute validation score
        val_score = 0.4 * max(0, metrics["r2"]) + 0.6 * metrics["stability"]
        val_r2 = metrics["r2"]
        val_stability = metrics["stability"]

        print(f"    Actual:   score={val_score:.4f}, R²={val_r2:.4f}, stability={val_stability:.4f}")

        # Check if within tolerance (relative difference)
        score_diff = abs(val_score - expected_score) / max(expected_score, 0.01)
        r2_diff = abs(val_r2 - expected_r2) / max(expected_r2, 0.01)
        stability_diff = abs(val_stability - expected_stability) / max(expected_stability, 0.01)

        # Primary check: composite score within tolerance
        # Secondary: neither R² nor stability collapsed
        converged = (
            score_diff <= tolerance and
            r2_diff <= tolerance * 2 and  # Allow slightly more R² variance
            stability_diff <= tolerance * 2
        )

        if converged:
            print(f"    ✓ Validation PASSED (score diff: {score_diff:.1%})")

            # Save the validated model
            model_path = output_dir / f"sae_{sae_type}_validated.pt"
            save_sae_model(model, config, metrics, best_params, model_path)

            return {
                "params": best_params,
                "score": val_score,
                "metrics": metrics,
                "validated": True,
                "validation_attempts": attempt + 1,
            }, model_path

        else:
            print(f"    ✗ Validation FAILED (score diff: {score_diff:.1%}, r2 diff: {r2_diff:.1%}, stability diff: {stability_diff:.1%})")

            # Add this result as a new trial to inform Optuna
            # This helps the optimizer learn that this HP region has high variance
            study.add_trial(
                optuna.trial.create_trial(
                    params=best_params,
                    distributions=study.best_trial.distributions,
                    values=[val_score],
                    user_attrs=metrics,
                )
            )

            # Run additional trials to find more robust HPs
            print(f"    Running {extra_trials_per_retry} additional trials...")
            study.optimize(objective, n_trials=extra_trials_per_retry, show_progress_bar=False)

    # Max retries exceeded
    print(f"  ⚠ Validation did not converge after {max_retries} attempts")
    print(f"    Using best available config (may have high variance)")

    # Save anyway with warning flag
    model_path = output_dir / f"sae_{sae_type}_unvalidated.pt"
    save_sae_model(model, config, metrics, best_params, model_path)

    return {
        "params": best_params,
        "score": val_score,
        "metrics": metrics,
        "validated": False,
        "validation_attempts": max_retries,
    }, model_path


def create_optuna_objective(
    embeddings_by_ctx: Dict[int, np.ndarray],
    sae_type: str,
    device: str = "cpu",
):
    """Create Optuna objective for a specific SAE type.

    Args:
        embeddings_by_ctx: Dict mapping context_size -> pooled embeddings.
            When the dict has a single entry, no context_size HP is added.
            When multiple entries, context_size becomes a categorical HP.
        sae_type: SAE architecture type
    """
    import optuna

    ctx_keys = sorted(embeddings_by_ctx.keys())
    search_ctx = len(ctx_keys) > 1

    def objective(trial: optuna.Trial) -> float:
        # Context size HP (only when multiple extractions available)
        if search_ctx:
            context_size = trial.suggest_categorical("context_size", ctx_keys)
        else:
            context_size = ctx_keys[0]
        embeddings = embeddings_by_ctx[context_size]

        # Common hyperparameters — scale expansion to input dim to avoid
        # overparameterized SAEs (4096-dim × 8 = 32768 hidden = 268M params)
        input_dim = embeddings.shape[1]
        if input_dim >= 2048:
            expansion = trial.suggest_categorical("expansion", [1, 2])
        else:
            expansion = trial.suggest_categorical("expansion", [4, 8])
        sparsity_penalty = trial.suggest_float("sparsity_penalty", 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Type-specific parameters
        if sae_type in ("topk", "archetypal", "matryoshka_archetypal"):
            topk = trial.suggest_categorical("topk", [16, 32, 64, 128])
        else:
            topk = 32

        if sae_type in ("archetypal", "matryoshka_archetypal"):
            archetypal_temp = trial.suggest_float("archetypal_temp", 0.05, 0.5, log=True)
            # K-means on high-dim data is slow; limit centroids for large inputs
            if input_dim >= 2048:
                archetypal_n = trial.suggest_categorical("archetypal_n", [128, 256])
            else:
                archetypal_n = trial.suggest_categorical("archetypal_n", [256, 512, 1000])
            archetypal_relax = trial.suggest_float("archetypal_relaxation", 0.0, 2.0)
        else:
            archetypal_temp = 0.1
            archetypal_n = 500
            archetypal_relax = 0.0

        try:
            metrics = run_sae_trial(
                embeddings,
                sae_type=sae_type,
                expansion=expansion,
                sparsity_penalty=sparsity_penalty,
                learning_rate=learning_rate,
                topk=topk,
                archetypal_n_archetypes=archetypal_n,
                archetypal_temp=archetypal_temp,
                archetypal_relaxation=archetypal_relax,
                n_epochs=100,
                measure_stability=True,
                device=device,
            )

            # Store metrics (including context_size for retrieval)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    trial.set_user_attr(key, val)
            trial.set_user_attr("context_size", context_size)

            # Objective: balance R² and stability
            # Weight stability higher since that's the key differentiator
            r2 = metrics["r2"]
            stability = metrics["stability"]

            # Composite score: 0.4 * R² + 0.6 * stability
            # (stability is more important for interpretability)
            score = 0.4 * max(0, r2) + 0.6 * stability

            return score

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    return objective


def run_sweep(
    model_name: str,
    n_trials: int = 30,
    output_dir: Optional[Path] = None,
    context_sizes: Optional[List[int]] = None,
    device: str = "cuda",
    sae_type_filter: Optional[List[str]] = None,
):
    """Run HP sweep for SAE types on train datasets.

    Args:
        model_name: Base model name (e.g. 'tabpfn_layer16')
        n_trials: Number of Optuna trials per SAE type
        output_dir: Output directory for results
        context_sizes: If given, search over context sizes as an HP.
            Requires pre-extracted embeddings at each size (e.g. tabpfn_layer16_ctx200/).
            When None, uses model_name directory directly.
        device: Torch device for training
        sae_type_filter: If given, only sweep these SAE types (default: all 5)
    """
    import optuna

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pool embeddings for each context size (or just one set if no context_sizes)
    if context_sizes:
        effective_names = [f"{model_name}_ctx{c}" for c in context_sizes]
    else:
        effective_names = [model_name]

    # Discover train/test split from first available effective model
    train_datasets, test_datasets = get_tabarena_splits(effective_names[0])
    print(f"Train datasets: {len(train_datasets)}")
    print(f"Test datasets: {len(test_datasets)}")

    embeddings_by_ctx: Dict[int, np.ndarray] = {}
    for i, ename in enumerate(effective_names):
        ctx = context_sizes[i] if context_sizes else 0
        print(f"\nPooling {ename} embeddings from train datasets...")
        emb, counts = pool_embeddings(ename, train_datasets, max_per_dataset=500)
        print(f"  Total samples: {len(emb)}")
        print(f"  Embedding dim: {emb.shape[1]}")
        print(f"  Datasets loaded: {len(counts)}")
        embeddings_by_ctx[ctx] = emb

    # Save split info
    split_info = {
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "context_sizes": context_sizes or [],
        "total_train_samples": {str(k): len(v) for k, v in embeddings_by_ctx.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Run sweep for each SAE type
    all_sae_types = ["l1", "topk", "matryoshka", "archetypal", "matryoshka_archetypal"]
    sae_types = sae_type_filter if sae_type_filter else all_sae_types
    best_configs = {}

    for sae_type in sae_types:
        print(f"\n{'='*60}")
        print(f"HP Sweep: {sae_type.upper()}")
        print('='*60)

        study_name = f"{model_name}_{sae_type}"
        storage = f"sqlite:///{output_dir}/{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        objective = create_optuna_objective(embeddings_by_ctx, sae_type, device=device)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest {sae_type} (before validation):")
        print(f"  Score: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
        if study.best_trial.user_attrs:
            print(f"  R²: {study.best_trial.user_attrs.get('r2', 'N/A'):.4f}")
            print(f"  Stability: {study.best_trial.user_attrs.get('stability', 'N/A'):.4f}")

        # Validation fit: retrain with best HPs and verify metrics are reproducible
        print(f"\nValidating {sae_type} config...")
        config_result, model_path = validate_and_save(
            embeddings_by_ctx=embeddings_by_ctx,
            sae_type=sae_type,
            study=study,
            output_dir=output_dir,
            tolerance=0.05,  # 5% relative tolerance
            max_retries=5,
            extra_trials_per_retry=3,
            device=device,
        )

        config_result["model_path"] = str(model_path) if model_path else None
        best_configs[sae_type] = config_result

    # Save best configs
    with open(output_dir / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

    # Summary comparison
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON (Validated)")
    print('='*60)
    print(f"{'Type':<20} {'Score':>8} {'R²':>8} {'Stability':>10} {'L0':>8} {'Valid':>6}")
    print('-'*70)

    for sae_type, config in best_configs.items():
        m = config["metrics"]
        validated = "✓" if config.get("validated", False) else "✗"
        print(f"{sae_type:<20} {config['score']:>8.4f} {m.get('r2', 0):>8.4f} "
              f"{m.get('stability', 0):>10.4f} {m.get('l0_sparsity', 0):>8.1f} {validated:>6}")

    # List saved models
    print(f"\nSaved models:")
    for sae_type, config in best_configs.items():
        if config.get("model_path"):
            print(f"  {sae_type}: {config['model_path']}")

    return best_configs


def evaluate_on_test(
    model_name: str,
    output_dir: Optional[Path] = None,
):
    """Evaluate best configs on test datasets."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep" / model_name

    # Load best configs
    config_path = output_dir / "best_configs.json"
    if not config_path.exists():
        raise ValueError(f"No best configs found. Run --sweep first.")

    with open(config_path) as f:
        best_configs = json.load(f)

    # Get test datasets
    _, test_datasets = get_tabarena_splits(model_name)

    print(f"Evaluating on {len(test_datasets)} test datasets...")

    # Pool test embeddings
    embeddings, counts = pool_embeddings(model_name, test_datasets, max_per_dataset=200)
    print(f"  Total test samples: {len(embeddings)}")

    # Evaluate each SAE type
    results = {}

    for sae_type, config in best_configs.items():
        print(f"\nEvaluating {sae_type}...")

        params = config["params"]

        metrics = run_sae_trial(
            embeddings,
            sae_type=sae_type,
            expansion=params.get("expansion", 4),
            sparsity_penalty=params.get("sparsity_penalty", 1e-3),
            learning_rate=params.get("learning_rate", 1e-3),
            topk=params.get("topk", 32),
            archetypal_n_archetypes=params.get("archetypal_n", 500),
            archetypal_temp=params.get("archetypal_temp", 0.1),
            archetypal_relaxation=params.get("archetypal_relaxation", 0.0),
            n_epochs=100,
            measure_stability=True,
        )

        results[sae_type] = metrics
        print(f"  R²: {metrics['r2']:.4f}, Stability: {metrics['stability']:.4f}")

    # Save test results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print('='*60)
    print(f"{'Type':<12} {'R²':>8} {'Stability':>10} {'L0':>8}")
    print('-'*60)

    for sae_type, m in results.items():
        print(f"{sae_type:<12} {m['r2']:>8.4f} {m['stability']:>10.4f} {m['l0_sparsity']:>8.1f}")

    return results


def setup_check(model_name: str = "tabpfn"):
    """Check data availability and show split info."""
    train_datasets, test_datasets = get_tabarena_splits(model_name)

    print("TabArena Train/Test Split")
    print("="*60)
    print(f"Train datasets ({len(train_datasets)}):")
    for ds in train_datasets:
        emb = load_embeddings(model_name, ds)
        status = f"{len(emb)} samples" if emb is not None else "MISSING"
        print(f"  {ds}: {status}")

    print(f"\nTest datasets ({len(test_datasets)}):")
    for ds in test_datasets:
        emb = load_embeddings(model_name, ds)
        status = f"{len(emb)} samples" if emb is not None else "MISSING"
        print(f"  {ds}: {status}")


def main():
    parser = argparse.ArgumentParser(description="SAE TabArena Sweep")
    parser.add_argument("--setup", action="store_true", help="Check data and show splits")
    parser.add_argument("--model", type=str, default="tabpfn", help="Model name")
    parser.add_argument("--layer", type=int, default=None,
                        help="Intermediate layer index (e.g. 16 for TabPFN)")
    parser.add_argument("--context-sizes", type=str, default=None,
                        help="Comma-separated context sizes to search over (e.g. '200,600,1000'). "
                             "Requires pre-extracted embeddings at each size.")
    parser.add_argument("--sae-types", type=str, default=None,
                        help="Comma-separated SAE types to sweep (default: all). "
                             "Options: l1,topk,matryoshka,archetypal,matryoshka_archetypal")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device for SAE training (default: cuda)")
    parser.add_argument("--n-trials", type=int, default=30, help="Trials per SAE type")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    args = parser.parse_args()

    # Parse context sizes
    context_sizes = None
    if args.context_sizes:
        context_sizes = [int(c.strip()) for c in args.context_sizes.split(",")]

    # Build effective model name(s)
    # When --layer is given, embeddings live under {model}_layer{N}_ctx{C}
    if args.layer is not None:
        base_model = f"{args.model}_layer{args.layer}"
    else:
        base_model = args.model

    if context_sizes:
        # With context sizes, directories are {base}_ctx{C}
        effective_models = [f"{base_model}_ctx{c}" for c in context_sizes]
    else:
        effective_models = [base_model]

    if args.setup:
        for em in effective_models:
            print(f"\n--- {em} ---")
            setup_check(em)
    elif args.evaluate:
        evaluate_on_test(effective_models[0])
    else:
        # Parse SAE type filter
        sae_type_filter = None
        if args.sae_types:
            sae_type_filter = [s.strip() for s in args.sae_types.split(",")]

        run_sweep(
            base_model,
            n_trials=args.n_trials,
            context_sizes=context_sizes,
            device=args.device,
            sae_type_filter=sae_type_filter,
        )


if __name__ == "__main__":
    main()
