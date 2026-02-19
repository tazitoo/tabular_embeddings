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

# Enable multi-threading for CPU operations (K-means, matrix ops)
import os
num_threads = os.cpu_count() or 8
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
    compare_dictionaries,
)
from data.tabarena_utils import get_embedding_dir

# Fixed random seed for reproducible splits
SPLIT_SEED = 42


def get_available_datasets(model_name: str) -> List[str]:
    """
    Discover available embeddings for a model.

    Dynamically finds all extracted embeddings instead of hardcoding.
    """
    # Map model name to actual embedding directory
    emb_dir_name = get_embedding_dir(model_name)
    path = PROJECT_ROOT / "output" / "embeddings" / "tabarena" / emb_dir_name
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
    seeds = [123, 456, 789, 101112, 131415][:n_runs]
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, result = train_sae(embeddings, config, device=device, verbose=False)
        dicts.append(result.dictionary)

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

        return {
            "alignment": overall_stability,
            "s_n_dec": s_n_dec,
            "per_scale": per_scale,
        }

    return {
        "alignment": overall_stability,
        "s_n_dec": s_n_dec,
    }


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
    aux_loss_type: str = "none",
    aux_loss_alpha: float = 0.03125,
    aux_loss_warmup_epochs: int = 3,
    resample_neurons: bool = False,
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
    aux_loss_type: str = "none",
    aux_loss_alpha: float = 0.03125,
    aux_loss_warmup: int = 3,
    resample_neurons: bool = False,
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
        stability_metrics = compute_stability(embeddings, config, n_runs=3, device=device)
        metrics["stability"] = stability_metrics["alignment"]
        metrics["s_n_dec"] = stability_metrics["s_n_dec"]

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
        aux_loss_type = best_params.get("aux_loss_type", "none")
        aux_loss_alpha = best_params.get("aux_loss_alpha", 0.03125)
        aux_loss_warmup = best_params.get("aux_warmup", 3)
        resample_neurons = best_params.get("resample_neurons", False)
        resample_interval = best_params.get("resample_interval", 25000)
        resample_samples = best_params.get("resample_samples", 1024)

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
            aux_loss_type=aux_loss_type,
            aux_loss_alpha=aux_loss_alpha,
            aux_loss_warmup=aux_loss_warmup,
            resample_neurons=resample_neurons,
            resample_interval=resample_interval,
            resample_samples=resample_samples,
            measure_stability=True,
            return_model=True,
            seed=validation_seed,
            device=device,
        )

        # Compute validation loss
        val_loss = metrics["total_loss"]

        print(f"    Actual:   loss={val_loss:.6f}")

        # Check if within tolerance (relative difference)
        loss_diff = abs(val_loss - expected_loss) / max(expected_loss, 1e-6)
        converged = loss_diff <= tolerance

        if converged:
            print(f"    ✓ Validation PASSED (loss diff: {loss_diff:.1%})")

            # Save the validated model
            model_path = output_dir / f"sae_{sae_type}_validated.pt"
            save_sae_model(model, config, metrics, best_params, model_path)

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

        # Dead neuron mitigation: residual_targeting + resampling
        # Apply to ALL architectures for fair comparison.
        # - L1 penalty can be too strict → dead neurons
        # - Matryoshka helps with splitting/absorption, not dead neurons directly
        # - Aux loss + resampling provides universal dead neuron revival
        # - No downside: if no dead neurons, aux_loss ≈ 0 and resampling doesn't trigger

        # Always use residual_targeting (modern approach from SAE research)
        aux_loss_type = "residual_targeting"

        # Search aux loss hyperparameters
        # α controls strength of dead neuron revival loss
        aux_loss_alpha = trial.suggest_float("aux_loss_alpha", 1e-3, 10.0, log=True)
        # Warmup epochs: allow initial training to stabilize before aux loss kicks in
        aux_loss_warmup = trial.suggest_categorical("aux_warmup", [3, 5, 10, 20])

        # Neuron resampling (always enabled, complementary to residual_targeting)
        resample_neurons = True

        # Resample interval in steps (Anthropic uses 25000)
        # For 100 epochs × ~120 batches = 12k steps
        # Expanded range: old max was 10k, now 25k for breathing room
        resample_interval = trial.suggest_categorical("resample_interval", [2500, 5000, 10000, 25000])

        # Number of samples to use for resampling dead neurons
        # More samples = better error distribution coverage, but higher overhead
        # For TabICL: ~16k training samples, so [512, 1024, 2048, 4096] = [3%, 6%, 12%, 25%]
        # Expanded range: old max was 2048, now 4096 for breathing room
        resample_samples = trial.suggest_categorical("resample_samples", [512, 1024, 2048, 4096])

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
                        "aux_loss_type": aux_loss_type,
                        "aux_loss_alpha": aux_loss_alpha if aux_loss_type != "none" else None,
                        "aux_loss_warmup": aux_loss_warmup if aux_loss_type != "none" else None,
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
                aux_loss_type=aux_loss_type,
                aux_loss_alpha=aux_loss_alpha,
                aux_loss_warmup=aux_loss_warmup,
                resample_neurons=resample_neurons,
                resample_interval=resample_interval,
                resample_samples=resample_samples,
                measure_stability=True,
                device=device,
                use_wandb=use_wandb,
            )

            # Store metrics (including context_size for retrieval)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    trial.set_user_attr(key, val)
            trial.set_user_attr("context_size", context_size)

            # Objective: minimize the training loss (recon_loss + sparsity_loss)
            total_loss = metrics["total_loss"]

            # Finish wandb run
            if wandb_active:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass

            return total_loss

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


def _load_prebuilt_embeddings(model_name: str) -> Optional[Tuple[np.ndarray, List[str], List[str], int]]:
    """Try to load prebuilt SAE training data.

    Returns:
        (embeddings, train_datasets, test_datasets, optimal_layer) or None if not found
    """
    prebuilt_dir = PROJECT_ROOT / "output" / "sae_training"

    # Try to find prebuilt file matching model name
    # model_name might be 'tabpfn' or 'tabpfn_layer17' — handle both
    import hashlib as _hashlib

    candidates = sorted(prebuilt_dir.glob(f"{model_name}_layer*_sae_training.npz"))
    if not candidates:
        # Try base model name (strip _layerN suffix)
        base = model_name.split("_layer")[0] if "_layer" in model_name else model_name
        candidates = sorted(prebuilt_dir.glob(f"{base}_layer*_sae_training.npz"))

    if not candidates:
        return None

    path = candidates[0]  # Use first match (train variant)
    print(f"Loading prebuilt SAE training data: {path}")

    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    optimal_layer = int(data["optimal_layer"])
    source_datasets = list(data["source_datasets"])

    # Reconstruct train/test split
    from data.extended_loader import TABARENA_DATASETS
    all_datasets = sorted(TABARENA_DATASETS.keys())
    train_datasets = []
    test_datasets = []
    for ds in all_datasets:
        h = int(_hashlib.md5(ds.encode()).hexdigest(), 16)
        if h % 10 < 7:
            train_datasets.append(ds)
        else:
            test_datasets.append(ds)

    print(f"  Shape: {embeddings.shape}")
    print(f"  Optimal layer: {optimal_layer}")
    print(f"  Source datasets: {len(source_datasets)}")

    return embeddings, train_datasets, test_datasets, optimal_layer


def run_sweep(
    model_name: str,
    n_trials: int = 30,
    output_dir: Optional[Path] = None,
    context_sizes: Optional[List[int]] = None,
    device: str = "cuda",
    sae_type_filter: Optional[List[str]] = None,
    use_wandb: bool = False,
    use_prebuilt: bool = True,
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
        use_prebuilt: If True (default), load from prebuilt SAE training files
            produced by build_sae_training_data.py. Falls back to per-dataset
            discovery if prebuilt files are not found.
    """
    import optuna

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "sae_tabarena_sweep" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try prebuilt path first (preferred — uses optimal layers)
    prebuilt = None
    if use_prebuilt and not context_sizes:
        prebuilt = _load_prebuilt_embeddings(model_name)

    if prebuilt is not None:
        embeddings, train_datasets, test_datasets, optimal_layer = prebuilt
        embeddings_by_ctx: Dict[int, np.ndarray] = {0: embeddings}
        print(f"Train datasets: {len(train_datasets)}")
        print(f"Test datasets: {len(test_datasets)}")
    else:
        if use_prebuilt and not context_sizes:
            print("No prebuilt SAE training data found, falling back to per-dataset discovery")

        # Legacy path: discover per-dataset files
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
            # Pass raw embeddings - SAE's BatchNorm will learn normalization during training
            emb, counts = pool_embeddings(ename, train_datasets, max_per_dataset=500, normalize=False)
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
        "prebuilt": prebuilt is not None,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Run sweep for each SAE type
    all_sae_types = ["l1", "topk", "matryoshka", "archetypal", "matryoshka_archetypal", "matryoshka_batchtopk_archetypal"]
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
            direction="minimize",
            load_if_exists=True,
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

    # Pool test embeddings (raw, BatchNorm will apply learned normalization)
    embeddings, counts = pool_embeddings(model_name, test_datasets, max_per_dataset=200, normalize=False)
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
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging for training visualization")
    parser.add_argument("--use-prebuilt", action="store_true", default=True,
                        help="Load from prebuilt SAE training files (default: True)")
    parser.add_argument("--no-prebuilt", action="store_true",
                        help="Disable prebuilt loading, use per-dataset discovery")
    args = parser.parse_args()
    if args.no_prebuilt:
        args.use_prebuilt = False

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
            use_prebuilt=args.use_prebuilt,
        )


if __name__ == "__main__":
    main()
