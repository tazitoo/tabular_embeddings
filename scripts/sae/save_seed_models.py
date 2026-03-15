#!/usr/bin/env python3
"""
Save seed models for completed sweeps that finished before the seed-saving code.

Loads the validated checkpoint's config, retrains 3 stability seed models,
and saves them alongside the existing validated.pt and random_baseline.pt.

Usage:
    python scripts/save_seed_models.py --model tabdpt --device cuda
    python scripts/save_seed_models.py --model all --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT

from analysis.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
    create_random_baseline,
    train_sae,
)
from scripts.sae.compare_sae_cross_model import sae_sweep_dir


STABILITY_SEEDS = [123, 456, 789]


def save_seed_models(model_name: str, device: str = "cuda"):
    """Retrain and save 3 seed models for a completed sweep."""
    sweep_dir = sae_sweep_dir() / model_name
    sae_type = "matryoshka_archetypal"

    # Load validated config
    validated_path = sweep_dir / f"sae_{sae_type}_validated.pt"
    if not validated_path.exists():
        print(f"  No validated model found at {validated_path}")
        return

    ckpt = torch.load(validated_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    config = SAEConfig(**{k: v for k, v in cfg_dict.items() if k in SAEConfig.__dataclass_fields__})
    best_params = ckpt.get("params", {})
    metrics = ckpt.get("metrics", {})

    # Check which seeds already exist
    missing_seeds = []
    for seed in STABILITY_SEEDS:
        seed_path = sweep_dir / f"sae_{sae_type}_seed{seed}.pt"
        if seed_path.exists():
            print(f"  Seed {seed}: already exists, skipping")
        else:
            missing_seeds.append(seed)

    if not missing_seeds:
        print(f"  All seed models already exist for {model_name}")
        return

    # Load training data
    from scripts.embeddings.build_sae_training_data import OUTPUT_DIR as TRAINING_DIR
    from config import get_optimal_layer

    layer = get_optimal_layer(model_name)
    train_path = TRAINING_DIR / f"{model_name}_layer{layer}_sae_training.npz"
    if not train_path.exists():
        print(f"  Training data not found: {train_path}")
        return

    data = np.load(train_path)
    embeddings = data["embeddings"]
    print(f"  Training data: {embeddings.shape} from {train_path.name}")

    # Train and save each missing seed
    for seed in missing_seeds:
        print(f"  Training seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, result = train_sae(embeddings, config, device=device, verbose=False)

        seed_path = sweep_dir / f"sae_{sae_type}_seed{seed}.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": cfg_dict,
            "metrics": {
                "seed": seed,
                "reconstruction_loss": result.reconstruction_loss,
                "total_loss": result.total_loss,
                "alive_features": result.alive_features,
            },
            "params": best_params,
        }
        torch.save(checkpoint, seed_path)
        print(f"    Saved {seed_path.name} (recon={result.reconstruction_loss:.6f}, alive={result.alive_features})")

    # Also ensure random baseline exists
    baseline_path = sweep_dir / f"sae_{sae_type}_random_baseline.pt"
    if not baseline_path.exists():
        from dataclasses import asdict
        baseline = create_random_baseline(config)
        checkpoint = {
            "model_state_dict": baseline.state_dict(),
            "config": asdict(baseline.config),
            "metrics": {"random_baseline": True},
            "params": best_params,
        }
        torch.save(checkpoint, baseline_path)
        print(f"  Saved {baseline_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Save seed models for completed sweeps")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all' for all models with validated checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for training (default: cuda)")
    args = parser.parse_args()

    if args.model == "all":
        sweep_base = sae_sweep_dir()
        models = sorted(d.name for d in sweep_base.iterdir() if d.is_dir())
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Seed models: {model}")
        print("=" * 60)
        save_seed_models(model, device=args.device)


if __name__ == "__main__":
    main()
