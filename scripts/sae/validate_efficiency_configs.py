#!/usr/bin/env python3
"""
Validate best-efficiency SAE configs with stability repeats.

For each model, trains the SAE with the best efficiency hyperparams
(from round 8 sweep), validates convergence, and saves checkpoints
including stability seed models.

Usage:
    # Validate a single model
    python scripts/validate_efficiency_configs.py --model tabpfn --device cuda

    # Validate all models
    python scripts/validate_efficiency_configs.py --all --device cuda

    # Dry run (show configs without training)
    python scripts/validate_efficiency_configs.py --all --dry-run
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

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

from scripts.compare_sae_cross_model import DEFAULT_SAE_ROUND, sae_sweep_dir
from scripts.sae_tabarena_sweep import (
    _load_prebuilt_embeddings,
    run_sae_trial,
    save_sae_model,
    build_sae_config,
)
from analysis.sparse_autoencoder import SAEConfig, create_random_baseline


# Best efficiency configs from round 8 sweeps
# Selected by: min recon_loss * sqrt(hidden_dim) among alive >= 85%
EFFICIENCY_CONFIGS = {
    "tabpfn": {
        "expansion": 4, "topk": 256,
        "sparsity_penalty": 0.007625632012184669,
        "learning_rate": 0.00033844343421280234,
        "archetypal_temp": 0.13357811472091727,
        "archetypal_n": 2000,
        "archetypal_relaxation": 1.6580155165698034,
        "sweep_trial": 24, "sweep_recon": 0.070, "sweep_alive": 92.1,
    },
    "tabicl": {
        "expansion": 4, "topk": 256,
        "sparsity_penalty": 0.0009095103260325962,
        "learning_rate": 0.000184178980617142,
        "archetypal_temp": 0.48613886603787293,
        "archetypal_n": 2000,
        "archetypal_relaxation": 1.9836985261060902,
        "sweep_trial": 24, "sweep_recon": 0.089, "sweep_alive": 94.3,
    },
    "mitra": {
        "expansion": 4, "topk": 128,
        "sparsity_penalty": 0.0008777342664024141,
        "learning_rate": 0.0002974960980518038,
        "archetypal_temp": 0.1008183369523725,
        "archetypal_n": 1000,
        "archetypal_relaxation": 1.7363785174634265,
        "sweep_trial": 22, "sweep_recon": 0.072, "sweep_alive": 91.0,
    },
    "carte": {
        "expansion": 8, "topk": 128,
        "sparsity_penalty": 0.00013378899675167523,
        "learning_rate": 0.00020282373465875536,
        "archetypal_temp": 0.15206277227022244,
        "archetypal_n": 2000,
        "archetypal_relaxation": 1.9714779382239729,
        "sweep_trial": 11, "sweep_recon": 0.034, "sweep_alive": 90.3,
    },
    "hyperfast": {
        "expansion": 8, "topk": 128,
        "sparsity_penalty": 0.00020301359870596928,
        "learning_rate": 0.0005537314587391232,
        "archetypal_temp": 0.19198477148023135,
        "archetypal_n": 2000,
        "archetypal_relaxation": 1.97892251106423,
        "sweep_trial": 25, "sweep_recon": 0.097, "sweep_alive": 86.1,
    },
    "tabdpt": {
        "expansion": 4, "topk": 256,
        "sparsity_penalty": 0.0005892459498888461,
        "learning_rate": 0.00016881240651303825,
        "archetypal_temp": 0.3419119208480319,
        "archetypal_n": 1000,
        "archetypal_relaxation": 0.7708933990442745,
        "sweep_trial": 28, "sweep_recon": 0.143, "sweep_alive": 94.5,
    },
    "tabula8b": {
        "expansion": 1, "topk": 256,
        "sparsity_penalty": 0.006617615993454362,
        "learning_rate": 6.418831847431395e-05,
        "archetypal_temp": 0.3407874526406007,
        "archetypal_n": 512,
        "archetypal_relaxation": 1.6154689868186112,
        "sweep_trial": 27, "sweep_recon": 0.273, "sweep_alive": 95.0,
    },
    "tabicl_v2": {
        "expansion": 16, "topk": 256,
        "sparsity_penalty": 0.0005170191786366995,
        "learning_rate": 0.0001409431399338736,
        "archetypal_temp": 0.29594756677318224,
        "archetypal_n": 2000,
        "archetypal_relaxation": 1.4137146876952342,
        "sweep_trial": 4, "sweep_recon": 0.086, "sweep_alive": 88.4,
    },
}

SAE_TYPE = "matryoshka_archetypal"


def validate_model(model_name: str, device: str = "cuda"):
    """Train efficiency config with seed=42 (primary) + 4 stability seeds, report variability."""
    config = EFFICIENCY_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"Validating {model_name} (efficiency config)")
    print(f"  Sweep trial: {config['sweep_trial']}")
    print(f"  Config: {config['expansion']}x expansion, topk={config['topk']}")
    print(f"  Sweep: recon={config['sweep_recon']:.3f}, alive={config['sweep_alive']:.1f}%")
    print(f"{'='*60}")

    # Load same prebuilt data the sweep used
    prebuilt = _load_prebuilt_embeddings(model_name)
    if prebuilt is None:
        raise FileNotFoundError(
            f"No prebuilt SAE training data for '{model_name}'. "
            f"Run: python scripts/build_sae_training_data.py --model {model_name}"
        )
    embeddings, _, source_datasets, optimal_layer = prebuilt

    hidden_dim = config["expansion"] * embeddings.shape[1]
    print(f"  Hidden dim: {hidden_dim} ({config['expansion']}x × {embeddings.shape[1]})")

    # Train primary model with seed=42 (same as sweep)
    print(f"\n  Training primary model (seed=42)...")
    metrics, model, sae_config, seed_models, seed_ids = run_sae_trial(
        embeddings,
        sae_type=SAE_TYPE,
        expansion=config["expansion"],
        sparsity_penalty=config["sparsity_penalty"],
        learning_rate=config["learning_rate"],
        topk=config["topk"],
        archetypal_n_archetypes=config["archetypal_n"],
        archetypal_temp=config["archetypal_temp"],
        archetypal_relaxation=config["archetypal_relaxation"],
        n_epochs=100,
        measure_stability=True,
        return_model=True,
        seed=42,
        device=device,
    )

    # Collect per-seed metrics: primary (seed=42) + stability seeds
    primary_alive = metrics["alive_features"]
    primary_recon = metrics["reconstruction_loss"]

    all_recon = [primary_recon]
    all_alive = [primary_alive]
    all_seeds = [42]

    # Add stability seed metrics (from compute_stability's per_seed_metrics)
    stability_metrics = metrics.get("per_seed_metrics", [])
    for sm in stability_metrics:
        all_recon.append(sm["reconstruction_loss"])
        all_alive.append(sm["alive_features"])
        all_seeds.append(sm["seed"])

    recon_arr = np.array(all_recon)
    alive_arr = np.array(all_alive)
    alive_pct_arr = 100.0 * alive_arr / hidden_dim

    # Report per-seed breakdown
    print(f"\n  Per-seed results ({len(all_seeds)} seeds):")
    print(f"    {'Seed':>8} {'Recon':>8} {'Alive':>8} {'Alive%':>7}")
    for seed, recon, alive in zip(all_seeds, all_recon, all_alive):
        print(f"    {seed:>8} {recon:>8.4f} {alive:>8} {100*alive/hidden_dim:>6.1f}%")

    print(f"\n  Variability (n={len(all_seeds)}):")
    print(f"    Recon:  {recon_arr.mean():.4f} ± {recon_arr.std():.4f}  "
          f"(range {recon_arr.min():.4f}–{recon_arr.max():.4f})")
    print(f"    Alive%: {alive_pct_arr.mean():.1f} ± {alive_pct_arr.std():.1f}  "
          f"(range {alive_pct_arr.min():.1f}–{alive_pct_arr.max():.1f}%)")
    print(f"    Stability: {metrics.get('stability', 0):.4f}")
    print(f"    s_n_dec:   {metrics.get('s_n_dec', 0):.4f}")
    print(f"    L0:        {metrics['l0_sparsity']:.1f}")

    # Save checkpoints
    output_dir = sae_sweep_dir() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {k: v for k, v in config.items() if not k.startswith("sweep_")}

    # Primary model (seed=42)
    all_metrics = {**metrics, "variability": {
        "seeds": all_seeds,
        "recon_loss": all_recon,
        "alive_features": [int(a) for a in all_alive],
        "recon_mean": float(recon_arr.mean()), "recon_std": float(recon_arr.std()),
        "alive_pct_mean": float(alive_pct_arr.mean()), "alive_pct_std": float(alive_pct_arr.std()),
    }}
    model_path = output_dir / f"sae_{SAE_TYPE}_efficiency.pt"
    save_sae_model(model, sae_config, all_metrics, params, model_path)

    # Random baseline
    baseline = create_random_baseline(sae_config)
    baseline_path = output_dir / f"sae_{SAE_TYPE}_efficiency_random_baseline.pt"
    save_sae_model(baseline, baseline.config, {"random_baseline": True}, params, baseline_path)

    # Stability seed models
    for seed_model, seed_id in zip(seed_models, seed_ids):
        seed_path = output_dir / f"sae_{SAE_TYPE}_efficiency_seed{seed_id}.pt"
        save_sae_model(seed_model, sae_config, all_metrics, params, seed_path)

    print(f"\n  Saved {2 + len(seed_models)} checkpoints to {output_dir}")

    return {
        "model": model_name,
        "recon_mean": float(recon_arr.mean()),
        "recon_std": float(recon_arr.std()),
        "alive_pct_mean": float(alive_pct_arr.mean()),
        "alive_pct_std": float(alive_pct_arr.std()),
        "stability": metrics.get("stability", 0),
        "s_n_dec": metrics.get("s_n_dec", 0),
        "l0": metrics["l0_sparsity"],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate best-efficiency SAE configs")
    parser.add_argument("--model", type=str, help="Model to validate")
    parser.add_argument("--all", action="store_true", help="Validate all models")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without training")
    args = parser.parse_args()

    if args.dry_run:
        print("Best efficiency configs (round 8 sweeps):")
        print(f"{'Model':<12} {'Exp':>4} {'TopK':>5} {'Hidden':>7} {'Recon':>7} {'Alive':>7}")
        print("-" * 50)
        for name, cfg in EFFICIENCY_CONFIGS.items():
            # Need embed_dim — approximate from known values
            embed_dims = {
                "tabpfn": 192, "tabicl": 512, "mitra": 512, "carte": 300,
                "hyperfast": 776, "tabdpt": 726, "tabula8b": 4096, "tabicl_v2": 512,
            }
            ed = embed_dims.get(name, 0)
            hidden = cfg["expansion"] * ed
            print(f"{name:<12} {cfg['expansion']:>3}x {cfg['topk']:>5} {hidden:>7} "
                  f"{cfg['sweep_recon']:>7.3f} {cfg['sweep_alive']:>6.1f}%")
        return

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    models = list(EFFICIENCY_CONFIGS.keys()) if args.all else [args.model]

    results = []
    for model_name in models:
        if model_name not in EFFICIENCY_CONFIGS:
            print(f"Unknown model: {model_name}")
            print(f"Available: {list(EFFICIENCY_CONFIGS.keys())}")
            continue
        result = validate_model(model_name, device=args.device)
        results.append(result)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY (5 seeds per model)")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Recon':>12} {'Alive%':>14} {'Stab':>6} {'L0':>6}")
        print("-" * 55)
        for r in results:
            print(f"{r['model']:<12} "
                  f"{r['recon_mean']:.4f}±{r['recon_std']:.4f} "
                  f"{r['alive_pct_mean']:5.1f}±{r['alive_pct_std']:4.1f}% "
                  f"{r['stability']:>6.3f} {r['l0']:>6.1f}")


if __name__ == "__main__":
    main()
