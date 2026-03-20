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
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Enable multi-threading for CPU operations (K-means, matrix ops)
import os
num_threads = os.cpu_count() or 8
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

from scripts.sae.compare_sae_cross_model import DEFAULT_SAE_ROUND, sae_sweep_dir

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
    compare_dictionaries,
)
from data.tabarena_utils import get_embedding_dir

# Fixed random seed for reproducible splits
SPLIT_SEED = 42


def get_available_datasets(model_name: str, task_filter: str = None) -> List[str]:
    """
    Discover available embeddings for a model.

    Dynamically finds all extracted embeddings instead of hardcoding.

    Args:
        model_name: Model identifier (e.g., 'tabpfn', 'mitra')
        task_filter: If set, only return datasets matching this task type
                     ('classification' or 'regression'). Requires TABARENA_DATASETS.
    """
    # Map model name to actual embedding directory
    emb_dir_name = get_embedding_dir(model_name)
    path = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / emb_dir_name
    if not path.exists():
        raise ValueError(f"No embeddings directory found: {path}")

    datasets = []
    for f in path.glob("tabarena_*.npz"):
        ds_name = f.stem.replace("tabarena_", "")
        if task_filter:
            from data.extended_loader import TABARENA_DATASETS
            info = TABARENA_DATASETS.get(ds_name, {})
            ds_task = info.get("task", "classification")
            if ds_task != task_filter:
                continue
        datasets.append(ds_name)

    return sorted(datasets)


def get_tabarena_splits(model_name: str = "tabpfn", task_filter: str = None) -> Tuple[List[str], List[str]]:
    """
    Get train/test dataset lists from prebuilt SAE training data.

    Row-level split: every dataset contributes to both train and test.
    Returns the same dataset list for both (since all datasets are in both splits).
    """
    all_datasets = get_available_datasets(model_name, task_filter=task_filter)

    if not all_datasets:
        raise ValueError(f"No datasets found for {model_name}")

    # With row-level split, all datasets are in both train and test
    return all_datasets, all_datasets


def load_embeddings(model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load cached embeddings for a model/dataset."""
    # Map model name to actual embedding directory
    emb_dir_name = get_embedding_dir(model_name)
    path = PROJECT_ROOT / f"output/embeddings/tabarena/{emb_dir_name}/tabarena_{dataset_name}.npz"
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
    Pool embeddings from multiple datasets with per-dataset normalization.

    Per-dataset StandardScaler removes dataset-level distributional differences
    before pooling. Stats are computed per-dataset (not globally).

    Args:
        model_name: Model to load embeddings for
        datasets: List of dataset names
        max_per_dataset: Max samples per dataset (for balance)
        normalize: Whether to apply per-dataset StandardScaler

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

        if normalize:
            # Per-dataset StandardScaler (consistent with build_sae_training_data.py)
            ds_mean = emb.mean(axis=0)
            ds_std = emb.std(axis=0)
            ds_std[ds_std < 1e-8] = 1.0
            emb = (emb - ds_mean) / ds_std

        all_embeddings.append(emb)
        dataset_counts[ds] = len(emb)

    if not all_embeddings:
        raise ValueError(f"No embeddings found for {model_name}")

    pooled = np.concatenate(all_embeddings, axis=0)
    return pooled, dataset_counts


def compute_stability(
    embeddings: np.ndarray,
    config: SAEConfig,
    n_runs: int = 2,
    return_per_scale: bool = False,
    return_models: bool = False,
    device: str = "cpu",
) -> float:
    """
    Train SAE multiple times and measure dictionary stability.

    For Matryoshka SAEs, also measures stability at each nested scale.

    Args:
        embeddings: Training data
        config: SAE configuration
        n_runs: Number of training runs
        return_per_scale: If True, return dict with per-scale stability (Matryoshka only)
        return_models: If True, include trained models in return dict

    Returns:
        Dict with stability metrics (and optionally trained models)
    """
    dicts = []
    models = []
    per_seed_metrics = []
    seeds = [123, 456, 789, 101112, 131415][:n_runs]
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, result = train_sae(embeddings, config, device=device, verbose=False)
        dicts.append(result.dictionary)
        per_seed_metrics.append({
            "seed": seed,
            "alive_features": result.alive_features,
            "reconstruction_loss": result.reconstruction_loss,
            "aux_loss": result.aux_loss,
        })
        if return_models:
            models.append(model)

    # 1. Feature alignment stability (our metric)
    comp = compare_dictionaries(dicts[0], dicts[1])
    overall_stability = comp['mean_best_match_a']

    # 2. s_n^dec (Chanin & Garriga-Alonso arXiv:2508.16560)
    # Measures decoder weight stability across retraining
    # Lower = more stable, 0 = optimal sparsity
    dicts_array = np.array(dicts)  # (n_runs, hidden_dim, input_dim)
    W_mean = dicts_array.mean(axis=0)  # (hidden_dim, input_dim)
    W_mean_norm = np.linalg.norm(W_mean, 'fro')

    distances = []
    for W in dicts:
        d = np.linalg.norm(W - W_mean, 'fro') / (W_mean_norm + 1e-8)
        distances.append(d)

    s_n_dec = float(np.mean(distances))

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

        result = {
            "alignment": overall_stability,
            "s_n_dec": s_n_dec,
            "per_scale": per_scale,
            "per_seed_metrics": per_seed_metrics,
        }
        if return_models:
            result["models"] = models
            result["seeds"] = seeds
        return result

    result = {
        "alignment": overall_stability,
        "s_n_dec": s_n_dec,
        "per_seed_metrics": per_seed_metrics,
    }
    if return_models:
        result["models"] = models
        result["seeds"] = seeds
    return result


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
    aux_loss_type: str = "auxk",
    aux_loss_alpha: float = 0.03125,
    aux_loss_warmup_epochs: int = 3,
    resample_neurons: bool = True,
    resample_interval: int = 25000,
    resample_samples: int = 1024,
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
        aux_loss_type=aux_loss_type,
        aux_loss_alpha=aux_loss_alpha,
        aux_loss_warmup_epochs=aux_loss_warmup_epochs,
        resample_dead_neurons=resample_neurons,
        resample_interval=resample_interval,
        resample_samples=resample_samples,
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
    aux_loss_type: str = "auxk",
    aux_loss_alpha: float = 0.03125,
    aux_loss_warmup: int = 3,
    resample_neurons: bool = True,
    resample_interval: int = 25000,
    resample_samples: int = 1024,
    measure_stability: bool = True,
    return_model: bool = False,
    seed: int = 42,
    device: str = "cpu",
    use_wandb: bool = False,
) -> Dict:
    """Run a single SAE training trial and return metrics (and optionally model)."""
    config = build_sae_config(
        embeddings, sae_type, expansion, sparsity_penalty, learning_rate,
        topk, archetypal_n_archetypes, archetypal_temp, archetypal_relaxation, n_epochs,
        aux_loss_type, aux_loss_alpha, aux_loss_warmup, resample_neurons, resample_interval,
        resample_samples
    )

    # Train and evaluate
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, result = train_sae(embeddings, config, device=device, verbose=False, use_wandb=use_wandb)
    richness = measure_dictionary_richness(result)

    metrics = {
        "sae_type": sae_type,
        "expansion": expansion,
        "sparsity_penalty": sparsity_penalty,
        "learning_rate": learning_rate,
        "l0_sparsity": richness["l0_sparsity"],
        "reconstruction_loss": result.reconstruction_loss,
        "sparsity_loss": result.sparsity_loss,
        "aux_loss": result.aux_loss,
        "total_loss": result.total_loss,
        "alive_features": result.alive_features,
    }

    if measure_stability:
        stability_metrics = compute_stability(
            embeddings, config, n_runs=3, return_models=return_model, device=device,
        )
        metrics["stability"] = stability_metrics["alignment"]
        metrics["s_n_dec"] = stability_metrics["s_n_dec"]
        metrics["per_seed_metrics"] = stability_metrics.get("per_seed_metrics", [])

    if return_model:
        seed_models = stability_metrics.get("models", []) if measure_stability else []
        seed_ids = stability_metrics.get("seeds", []) if measure_stability else []
        return metrics, model, config, seed_models, seed_ids
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
    use_wandb: bool = False,
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
    objective = create_optuna_objective(embeddings_by_ctx, sae_type, device=device, use_wandb=use_wandb)
    # On first attempt, validate the best from the initial sweep.
    # On subsequent attempts, validate the best from the latest surrogate-guided batch.
    candidate_trial = study.best_trial

    for attempt in range(max_retries):
        best_trial = candidate_trial
        best_params = best_trial.params
        expected_loss = best_trial.value

        print(f"\n  Validation attempt {attempt + 1}/{max_retries}")
        print(f"    Expected: loss={expected_loss:.6f}")

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
        # Dead neuron mitigation — hardcoded to match objective function.
        # These are not Optuna params; they use literature defaults.

        # Train with a different seed to test robustness
        validation_seed = 12345 + attempt
        metrics, model, config, seed_models, seed_ids = run_sae_trial(
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
            # aux/resample use hardcoded defaults from run_sae_trial
            measure_stability=True,
            return_model=True,
            seed=validation_seed,
            device=device,
        )

        # Compare on composite objective (matches objective function)
        hidden_dim = expansion * embeddings.shape[1]
        alive_frac = metrics["alive_features"] / hidden_dim
        l0 = metrics["l0_sparsity"]
        val_loss = metrics["reconstruction_loss"] * np.sqrt(hidden_dim) * np.sqrt(l0) / alive_frac

        print(f"    Actual:   loss={val_loss:.6f}")

        # Check if within tolerance (relative difference)
        loss_diff = abs(val_loss - expected_loss) / max(expected_loss, 1e-6)
        converged = loss_diff <= tolerance

        if converged:
            print(f"    ✓ Validation PASSED (loss diff: {loss_diff:.1%})")

            # Save the validated model
            model_path = output_dir / f"sae_{sae_type}_validated.pt"
            save_sae_model(model, config, metrics, best_params, model_path)

            # Save random baseline with matching dimensions for comparison
            from analysis.sparse_autoencoder import create_random_baseline
            baseline = create_random_baseline(config)
            baseline_path = output_dir / f"sae_{sae_type}_random_baseline.pt"
            save_sae_model(
                baseline, baseline.config, {"random_baseline": True},
                best_params, baseline_path,
            )

            # Save stability seed models (already trained, free repeatability)
            for seed_model, seed_id in zip(seed_models, seed_ids):
                seed_path = output_dir / f"sae_{sae_type}_seed{seed_id}.pt"
                save_sae_model(seed_model, config, metrics, best_params, seed_path)

            return {
                "params": best_params,
                "loss": val_loss,
                "metrics": metrics,
                "validated": True,
                "validation_attempts": attempt + 1,
            }, model_path

        else:
            print(f"    ✗ Validation FAILED (loss diff: {loss_diff:.1%})")

            # Add validation result to inform the surrogate about variance
            study.add_trial(
                optuna.trial.create_trial(
                    params=best_params,
                    distributions=best_trial.distributions,
                    values=[val_loss],
                    user_attrs=metrics,
                )
            )

            # Run surrogate-guided trials — the surrogate now knows this
            # region has high variance and will explore elsewhere
            print(f"    Running {extra_trials_per_retry} additional trials...")
            n_before = len(study.trials)
            study.optimize(objective, n_trials=extra_trials_per_retry, show_progress_bar=False)

            # Next validation candidate = best of the new surrogate-guided trials
            new_trials = [t for t in study.trials[n_before:]
                          if t.state == optuna.trial.TrialState.COMPLETE]
            if new_trials:
                candidate_trial = min(new_trials, key=lambda t: t.value)
            else:
                candidate_trial = study.best_trial

    # Max retries exceeded
    print(f"  ⚠ Validation did not converge after {max_retries} attempts")
    print(f"    Using best available config (may have high variance)")

    # Save anyway with warning flag
    model_path = output_dir / f"sae_{sae_type}_unvalidated.pt"
    save_sae_model(model, config, metrics, best_params, model_path)

    return {
        "params": best_params,
        "loss": val_loss,
        "metrics": metrics,
        "validated": False,
        "validation_attempts": max_retries,
    }, model_path


def create_optuna_objective(
    embeddings_by_ctx: Dict[int, np.ndarray],
    sae_type: str,
    device: str = "cpu",
    use_wandb: bool = False,
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
            expansion = trial.suggest_categorical("expansion", [1, 2, 4])
        else:
            expansion = trial.suggest_categorical("expansion", [4, 8, 16])
        sparsity_penalty = trial.suggest_float("sparsity_penalty", 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)

        # Type-specific parameters
        if sae_type in ("topk", "batchtopk", "archetypal", "batchtopk_archetypal", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal"):
            topk = trial.suggest_categorical("topk", [16, 32, 64, 128, 256])
        else:
            topk = 32

        if sae_type in ("archetypal", "batchtopk_archetypal", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal"):
            archetypal_temp = trial.suggest_float("archetypal_temp", 0.05, 0.5, log=True)
            # K-means on high-dim data is slow; limit centroids for large inputs
            # Expanded range: old max was 1000, now 2000 for breathing room
            if input_dim >= 2048:
                archetypal_n = trial.suggest_categorical("archetypal_n", [128, 256, 512])
            else:
                archetypal_n = trial.suggest_categorical("archetypal_n", [512, 1000, 2000])
            archetypal_relax = trial.suggest_float("archetypal_relaxation", 0.0, 2.0)
        else:
            archetypal_temp = 0.1
            archetypal_n = 500
            archetypal_relax = 0.0

        # Dead neuron mitigation — hardcoded, not searched.
        # Letting Optuna search aux params is counterproductive: since aux_loss
        # is part of total_loss, the surrogate learns to minimize aux_loss by
        # weakening the mitigation (long warmup, small α) rather than by
        # actually reviving neurons. Fix the values from literature instead.
        aux_loss_type = "auxk"
        aux_loss_alpha = 1 / 32  # Gao et al. (2024)
        aux_loss_warmup = 3      # Short warmup — neurons die fast without pressure
        resample_neurons = True
        resample_interval = 2500  # Every ~19 epochs for 130 steps/epoch
        resample_samples = 2048

        # Initialize wandb for this trial
        wandb_active = False
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="tabular-sae",
                    name=f"{sae_type}_trial_{trial.number}",
                    config={
                        "sae_type": sae_type,
                        "expansion": expansion,
                        "sparsity_penalty": sparsity_penalty,
                        "learning_rate": learning_rate,
                        "topk": topk,
                        "archetypal_n": archetypal_n if sae_type in ("archetypal", "batchtopk_archetypal", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal") else None,
                        "archetypal_temp": archetypal_temp if sae_type in ("archetypal", "batchtopk_archetypal", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal") else None,
                    },
                    reinit=True,
                )
                wandb_active = True
            except ImportError:
                print("Warning: wandb not available, skipping logging")

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
                # aux/resample use hardcoded defaults from run_sae_trial
                measure_stability=True,
                device=device,
                use_wandb=use_wandb,
            )

            # Store metrics (including context_size for retrieval)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    trial.set_user_attr(key, val)
            trial.set_user_attr("context_size", context_size)

            # Objective: recon_loss * sqrt(hidden_dim) * sqrt(L0) / alive_frac
            # Three-way balance: reconstruction quality, dictionary capacity,
            # and sparsity, penalized by wasted (dead) capacity.
            hidden_dim = expansion * embeddings.shape[1]
            alive_frac = metrics["alive_features"] / hidden_dim
            l0 = metrics["l0_sparsity"]
            obj = metrics["reconstruction_loss"] * np.sqrt(hidden_dim) * np.sqrt(l0) / alive_frac

            # Finish wandb run
            if wandb_active:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass

            return obj

        except Exception as e:
            print(f"Trial failed: {e}")
            # Finish wandb run on failure
            if wandb_active:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass
            return 0.0

    return objective


def _load_prebuilt_embeddings(model_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], int]]:
    """Try to load prebuilt SAE training and test data.

    Returns:
        (train_embeddings, test_embeddings, source_datasets, optimal_layer) or None
    """
    prebuilt_dir = PROJECT_ROOT / "output" / "sae_training_round10"

    # Find train file: prefer taskaware, fall back to layer-specific
    base = model_name.split("_layer")[0] if "_layer" in model_name else model_name
    train_candidates = sorted(prebuilt_dir.glob(f"{base}_taskaware_sae_training.npz"))
    if not train_candidates:
        train_candidates = sorted(prebuilt_dir.glob(f"{base}_layer*_sae_training.npz"))
    if not train_candidates:
        return None

    train_path = train_candidates[0]
    # Derive test path from train path
    test_path = Path(str(train_path).replace("_sae_training.npz", "_sae_test.npz"))

    print(f"Loading prebuilt SAE data:")
    print(f"  Train: {train_path}")

    train_data = np.load(train_path, allow_pickle=True)
    train_embeddings = train_data["embeddings"].astype(np.float32)
    optimal_layer = int(train_data["optimal_layer"])
    source_datasets = list(train_data["source_datasets"])

    test_embeddings = None
    if test_path.exists():
        print(f"  Test:  {test_path}")
        test_data = np.load(test_path, allow_pickle=True)
        test_embeddings = test_data["embeddings"].astype(np.float32)
    else:
        print(f"  Test:  NOT FOUND (run build_sae_training_data.py)")

    print(f"  Train shape: {train_embeddings.shape}")
    if test_embeddings is not None:
        print(f"  Test shape:  {test_embeddings.shape}")
    print(f"  Optimal layer: {optimal_layer}")
    print(f"  Source datasets: {len(source_datasets)}")

    return train_embeddings, test_embeddings, source_datasets, optimal_layer


def run_sweep(
    model_name: str,
    n_trials: int = 30,
    output_dir: Optional[Path] = None,
    context_sizes: Optional[List[int]] = None,
    device: str = "cuda",
    sae_type_filter: Optional[List[str]] = None,
    use_wandb: bool = False,
    task_filter: Optional[str] = None,
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
        sweep_name = f"{model_name}_{task_filter}" if task_filter else model_name
        output_dir = sae_sweep_dir() / sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prebuilt training data (required)
    prebuilt = _load_prebuilt_embeddings(model_name)
    if prebuilt is None:
        prebuilt_dir = PROJECT_ROOT / "output" / "sae_training_round10"
        raise FileNotFoundError(
            f"No prebuilt SAE training data found for '{model_name}'. "
            f"Expected: {prebuilt_dir}/{model_name}_taskaware_sae_training.npz\n"
            f"Run: python scripts/sae_corpus/06_build_sae_training_data.py --model {model_name}"
        )

    train_embeddings, test_embeddings, source_datasets, optimal_layer = prebuilt
    embeddings_by_ctx: Dict[int, np.ndarray] = {0: train_embeddings}
    print(f"Source datasets: {len(source_datasets)} (row-level 70/30 split)")

    # Save split info
    split_info = {
        "source_datasets": source_datasets,
        "split_type": "row_level_70_30",
        "context_sizes": context_sizes or [],
        "total_train_samples": {str(k): len(v) for k, v in embeddings_by_ctx.items()},
        "total_test_samples": len(test_embeddings) if test_embeddings is not None else 0,
        "prebuilt": True,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Run sweep for each SAE type
    all_sae_types = ["matryoshka_archetypal"]
    sae_types = sae_type_filter if sae_type_filter else all_sae_types
    best_configs = {}

    for sae_type in sae_types:
        print(f"\n{'='*60}")
        print(f"HP Sweep: {sae_type.upper()}")
        print('='*60)

        study_name = f"{model_name}_{sae_type}"
        storage = f"sqlite:///{output_dir}/{study_name}.db"

        # Always start fresh — delete stale study if it exists
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        objective = create_optuna_objective(embeddings_by_ctx, sae_type, device=device, use_wandb=use_wandb)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest {sae_type} (before validation):")
        print(f"  Loss: {study.best_value:.6f}")
        print(f"  Params: {study.best_params}")

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
            use_wandb=False,  # Disable wandb for validation runs
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
    print(f"{'Type':<20} {'Loss':>10} {'Recon':>10} {'Stability':>10} {'L0':>8} {'Alive':>8} {'Valid':>6}")
    print('-'*76)

    for sae_type, config in best_configs.items():
        m = config["metrics"]
        validated = "✓" if config.get("validated", False) else "✗"
        print(f"{sae_type:<20} {m.get('total_loss', 0):>10.4f} {m.get('reconstruction_loss', 0):>10.6f} "
              f"{m.get('stability', 0):>10.4f} {m.get('l0_sparsity', 0):>8.1f} "
              f"{m.get('alive_features', 0):>8} {validated:>6}")

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
        output_dir = sae_sweep_dir() / model_name

    # Load best configs
    config_path = output_dir / "best_configs.json"
    if not config_path.exists():
        raise ValueError(f"No best configs found. Run --sweep first.")

    with open(config_path) as f:
        best_configs = json.load(f)

    # Load prebuilt test embeddings
    prebuilt = _load_prebuilt_embeddings(model_name)
    if prebuilt is None or prebuilt[1] is None:
        raise ValueError(
            f"No prebuilt test data for {model_name}. "
            f"Run: python scripts/build_sae_training_data.py --model {model_name}"
        )

    _, embeddings, source_datasets, _ = prebuilt
    print(f"Evaluating on {len(source_datasets)} datasets ({len(embeddings)} test samples)...")

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
            # aux/resample use hardcoded defaults from run_sae_trial
            measure_stability=True,
        )

        results[sae_type] = metrics
        print(f"  Loss: {metrics['total_loss']:.6f}, Stability: {metrics['stability']:.4f}")

    # Save test results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print('='*60)
    print(f"{'Type':<12} {'Loss':>10} {'Recon':>10} {'Stability':>10} {'L0':>8}")
    print('-'*60)

    for sae_type, m in results.items():
        print(f"{sae_type:<12} {m['total_loss']:>10.4f} {m['reconstruction_loss']:>10.6f} "
              f"{m['stability']:>10.4f} {m['l0_sparsity']:>8.1f}")

    return results


def setup_check(model_name: str = "tabpfn"):
    """Check data availability and show split info."""
    datasets = get_available_datasets(model_name)

    print(f"TabArena datasets for {model_name} (row-level 70/30 split)")
    print("=" * 60)
    print(f"Datasets ({len(datasets)}):")
    for ds in datasets:
        emb = load_embeddings(model_name, ds)
        status = f"{len(emb)} samples" if emb is not None else "MISSING"
        print(f"  {ds}: {status}")

    # Check prebuilt files
    prebuilt = _load_prebuilt_embeddings(model_name)
    if prebuilt:
        train_emb, test_emb, _, _ = prebuilt
        print(f"\nPrebuilt train: {train_emb.shape}")
        if test_emb is not None:
            print(f"Prebuilt test:  {test_emb.shape}")
    else:
        print("\nNo prebuilt SAE training data found.")


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
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging for training visualization")
    parser.add_argument("--task", type=str, default=None,
                        choices=["classification", "regression"],
                        help="Filter datasets by task type (default: all)")
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
            use_wandb=args.wandb,
            task_filter=args.task,
        )


if __name__ == "__main__":
    main()
