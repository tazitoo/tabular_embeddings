#!/usr/bin/env python3
"""
Re-validate SAE models with correct aux_loss and resampling settings.

Round 6 validated checkpoints were trained without residual_targeting or
dead neuron resampling due to a bug in validate_and_save() — the hardcoded
objective values (aux_loss_type="residual_targeting", resample_neurons=True)
were not stored as Optuna params, so validation fell back to wrong defaults.

This script loads each model's best params from the existing checkpoint,
retrains with the correct settings, and saves new validated + seed models.

Usage:
    python scripts/revalidate_sae_models.py --model tabpfn --device cuda
    python scripts/revalidate_sae_models.py --model all --device cuda
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
    create_random_baseline,
    train_sae,
)
from scripts.sae_tabarena_sweep import (
    build_sae_config,
    compute_stability,
    measure_dictionary_richness,
    compare_dictionaries,
    sae_sweep_dir,
)


STABILITY_SEEDS = [123, 456, 789]
SAE_TYPE = "matryoshka_archetypal"


def load_best_params(model_name: str, sweep_dir: Path) -> dict:
    """Load best params from existing validated checkpoint."""
    validated_path = sweep_dir / model_name / f"sae_{SAE_TYPE}_validated.pt"
    if not validated_path.exists():
        raise FileNotFoundError(f"No validated checkpoint: {validated_path}")

    ckpt = torch.load(validated_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", {})
    old_metrics = ckpt.get("metrics", {})

    return params, old_metrics


def load_training_data(model_name: str) -> np.ndarray:
    """Load round 6 training data."""
    from config import get_optimal_layer

    training_dir = PROJECT_ROOT / "output" / "sae_training_round6"
    layer = get_optimal_layer(model_name)
    train_path = training_dir / f"{model_name}_layer{layer}_sae_training.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    data = np.load(train_path)
    return data["embeddings"]


def save_checkpoint(model, config, metrics, params, path):
    """Save SAE model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "params": params,
    }
    torch.save(checkpoint, path)


def revalidate_model(model_name: str, device: str = "cuda"):
    """Re-validate a single model with correct aux_loss and resampling."""
    sweep_base = sae_sweep_dir()
    output_dir = sweep_base / model_name

    print(f"\n{'='*60}")
    print(f"Re-validating: {model_name}")
    print("=" * 60)

    # Load best params from existing checkpoint
    params, old_metrics = load_best_params(model_name, sweep_base)
    print(f"  Best params from checkpoint:")
    for k, v in params.items():
        print(f"    {k}: {v}")
    print(f"  Old metrics: alive={old_metrics.get('alive_features')}, "
          f"recon={old_metrics.get('reconstruction_loss'):.6f}, "
          f"stability={old_metrics.get('stability', 'N/A')}")

    # Load training data
    embeddings = load_training_data(model_name)
    print(f"  Training data: {embeddings.shape}")

    # Build config with correct defaults (residual_targeting + resampling)
    config = build_sae_config(
        embeddings,
        sae_type=SAE_TYPE,
        expansion=params.get("expansion", 4),
        sparsity_penalty=params.get("sparsity_penalty", 1e-3),
        learning_rate=params.get("learning_rate", 1e-3),
        topk=params.get("topk", 32),
        archetypal_n_archetypes=params.get("archetypal_n", 500),
        archetypal_temp=params.get("archetypal_temp", 0.1),
        archetypal_relaxation=params.get("archetypal_relaxation", 0.0),
        n_epochs=100,
        # aux/resample use hardcoded defaults from build_sae_config
    )
    print(f"  Config: aux_loss_type={config.aux_loss_type}, "
          f"resample={config.resample_dead_neurons}, "
          f"hidden_dim={config.hidden_dim}, topk={config.topk}")

    # Train validated model
    print(f"\n  Training validated model...")
    validation_seed = 12345
    torch.manual_seed(validation_seed)
    np.random.seed(validation_seed)
    model, result = train_sae(embeddings, config, device=device, verbose=True)
    richness = measure_dictionary_richness(result)

    metrics = {
        "sae_type": SAE_TYPE,
        "expansion": params.get("expansion", 4),
        "sparsity_penalty": params.get("sparsity_penalty", 1e-3),
        "learning_rate": params.get("learning_rate", 1e-3),
        "l0_sparsity": richness["l0_sparsity"],
        "reconstruction_loss": result.reconstruction_loss,
        "sparsity_loss": result.sparsity_loss,
        "aux_loss": result.aux_loss,
        "total_loss": result.total_loss,
        "alive_features": result.alive_features,
    }

    print(f"\n  Validated: alive={result.alive_features}, "
          f"recon={result.reconstruction_loss:.6f}, "
          f"total_loss={result.total_loss:.6f}")

    # Compute stability
    print(f"  Computing stability (3 seeds)...")
    stability_metrics = compute_stability(
        embeddings, config, n_runs=3, return_models=True, device=device,
    )
    metrics["stability"] = stability_metrics["alignment"]
    metrics["s_n_dec"] = stability_metrics["s_n_dec"]
    seed_models = stability_metrics.get("models", [])
    seed_ids = stability_metrics.get("seeds", STABILITY_SEEDS[:3])

    print(f"  Stability: {metrics['stability']:.4f}, s_n_dec: {metrics['s_n_dec']:.4f}")

    # Compare with old metrics
    old_alive = old_metrics.get("alive_features", 0)
    new_alive = result.alive_features
    old_recon = old_metrics.get("reconstruction_loss", 0)
    new_recon = result.reconstruction_loss
    print(f"\n  Improvement:")
    print(f"    Alive: {old_alive} -> {new_alive} ({new_alive - old_alive:+d})")
    print(f"    Recon: {old_recon:.6f} -> {new_recon:.6f}")

    # Save validated model
    validated_path = output_dir / f"sae_{SAE_TYPE}_validated.pt"
    save_checkpoint(model, config, metrics, params, validated_path)
    print(f"  Saved: {validated_path.name}")

    # Save seed models
    for seed_model, seed_id in zip(seed_models, seed_ids):
        seed_path = output_dir / f"sae_{SAE_TYPE}_seed{seed_id}.pt"
        save_checkpoint(seed_model, config, metrics, params, seed_path)
        print(f"  Saved: {seed_path.name}")

    # Save random baseline
    baseline = create_random_baseline(config)
    baseline_path = output_dir / f"sae_{SAE_TYPE}_random_baseline.pt"
    save_checkpoint(baseline, baseline.config, {"random_baseline": True}, params, baseline_path)
    print(f"  Saved: {baseline_path.name}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Re-validate SAE models with correct aux_loss/resampling")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for training (default: cuda)")
    args = parser.parse_args()

    if args.model == "all":
        sweep_base = sae_sweep_dir()
        models = sorted(
            d.name for d in sweep_base.iterdir()
            if d.is_dir() and (d / f"sae_{SAE_TYPE}_validated.pt").exists()
        )
    else:
        models = [args.model]

    print(f"Models to re-validate: {models}")

    results = {}
    for model in models:
        try:
            metrics = revalidate_model(model, device=args.device)
            results[model] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("REVALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'Alive':>8} {'Recon':>10} {'Stability':>10} {'L0':>8}")
    print("-" * 60)
    for model, m in results.items():
        if "error" in m:
            print(f"{model:<12} ERROR: {m['error']}")
        else:
            print(f"{model:<12} {m['alive_features']:>8} {m['reconstruction_loss']:>10.6f} "
                  f"{m['stability']:>10.4f} {m['l0_sparsity']:>8.1f}")


if __name__ == "__main__":
    main()
