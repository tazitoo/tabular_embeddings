#!/usr/bin/env python3
"""Retrain SAE with selected (floor-picked) HPs and save validated model.

For models where the floor-selected trial differs from Optuna's best,
this retrains with the selected HPs, runs stability seeds, and saves
the validated model + random baseline.

Usage:
    python scripts/sae/retrain_selected.py --model tabpfn --device cuda
    python scripts/sae/retrain_selected.py --model tabula8b --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT
from scripts.sae.sae_tabarena_sweep import (
    _load_prebuilt_embeddings, run_sae_trial, save_sae_model,
    compute_stability,
)
from scripts.sae.compare_sae_cross_model import sae_sweep_dir
from analysis.sparse_autoencoder import create_random_baseline

# Floor-selected trial HPs (from sweep analysis)
SELECTED_PARAMS = {
    "tabpfn": {
        "trial": 14,
        "expansion": 4,
        "sparsity_penalty": 0.00010360852294206857,
        "learning_rate": 0.0002516986097067781,
        "topk": 256,
        "archetypal_temp": 0.20129888873505528,
        "archetypal_n": 1000,
        "archetypal_relaxation": 1.4114268837203008,
    },
    "tabula8b": {
        "trial": 15,
        "expansion": 1,
        "sparsity_penalty": 0.0035127674530828464,
        "learning_rate": 0.00010995890162062483,
        "topk": 64,
        "archetypal_temp": 0.11473925496243119,
        "archetypal_n": 512,
        "archetypal_relaxation": 0.6950379195220894,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SELECTED_PARAMS.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_name = args.model
    params = SELECTED_PARAMS[model_name]
    output_dir = sae_sweep_dir() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    prebuilt = _load_prebuilt_embeddings(model_name)
    train_emb, test_emb, datasets, _ = prebuilt
    print(f"Train: {train_emb.shape}, Test: {test_emb.shape}")

    sae_type = "matryoshka_archetypal"

    # Train with validation seed (different from sweep seed 42)
    print(f"\nTraining {model_name} with floor-selected HPs (trial {params['trial']})...")
    print(f"  Params: {json.dumps({k: v for k, v in params.items() if k != 'trial'}, indent=4)}")

    validation_seed = 12345
    metrics, model, config, seed_models, seed_ids = run_sae_trial(
        train_emb,
        sae_type=sae_type,
        expansion=params["expansion"],
        sparsity_penalty=params["sparsity_penalty"],
        learning_rate=params["learning_rate"],
        topk=params["topk"],
        archetypal_n_archetypes=params["archetypal_n"],
        archetypal_temp=params["archetypal_temp"],
        archetypal_relaxation=params["archetypal_relaxation"],
        n_epochs=100,
        measure_stability=True,
        return_model=True,
        seed=validation_seed,
        device=args.device,
    )

    # Test set evaluation
    test_tensor = torch.tensor(test_emb, dtype=torch.float32, device=args.device)
    model.eval()
    with torch.no_grad():
        recon, _ = model(test_tensor)
        test_recon_loss = torch.nn.functional.mse_loss(recon, test_tensor).item()

    hidden_dim = params["expansion"] * train_emb.shape[1]
    alive_frac = metrics["alive_features"] / hidden_dim
    l0 = metrics["l0_sparsity"]
    obj = test_recon_loss * np.sqrt(hidden_dim) * np.sqrt(l0) / alive_frac

    print(f"\n  Results:")
    print(f"    train_recon:  {metrics['reconstruction_loss']:.6f}")
    print(f"    test_recon:   {test_recon_loss:.6f}")
    print(f"    alive:        {metrics['alive_features']}/{hidden_dim} ({alive_frac*100:.1f}%)")
    print(f"    L0:           {l0:.1f}")
    print(f"    stability:    {metrics['stability']:.4f}")
    print(f"    objective:    {obj:.4f}")

    # Save validated model
    best_params = {k: v for k, v in params.items() if k != "trial"}
    model_path = output_dir / f"sae_{sae_type}_validated.pt"
    save_sae_model(model, config, metrics, best_params, model_path)

    # Save random baseline
    baseline = create_random_baseline(config)
    baseline_path = output_dir / f"sae_{sae_type}_random_baseline.pt"
    save_sae_model(baseline, baseline.config, {"random_baseline": True}, best_params, baseline_path)

    # Save stability seed models
    for seed_model, seed_id in zip(seed_models, seed_ids):
        seed_path = output_dir / f"sae_{sae_type}_seed{seed_id}.pt"
        save_sae_model(seed_model, config, metrics, best_params, seed_path)

    print(f"\nSaved to {output_dir}/")
    print(f"  sae_{sae_type}_validated.pt")
    print(f"  sae_{sae_type}_random_baseline.pt")
    for seed_id in seed_ids:
        print(f"  sae_{sae_type}_seed{seed_id}.pt")


if __name__ == "__main__":
    main()
