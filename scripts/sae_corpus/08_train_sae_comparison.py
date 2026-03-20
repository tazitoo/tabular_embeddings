#!/usr/bin/env python3
"""Train SAEs for fixed-layer vs per-dataset-layer comparison.

Uses the round 8 best config (matryoshka_archetypal) to train two SAEs
on the two variants of TabPFN training data:
  A) Fixed layer 18 (tabpfn_layer18_sae_training.npz)
  B) Per-dataset critical layers (tabpfn_perds_sae_training.npz)

Then evaluates both on concept importance and ablation for two test
datasets: airfoil_self_noise (L6, regression) and
polish_companies_bankruptcy (L23, classification).

Output:
    output/sae_training_round10/tabpfn_layer18_sae.pt
    output/sae_training_round10/tabpfn_perds_sae.pt
    output/sae_training_round10/layer_comparison_results.json

Usage:
    python scripts/sae_corpus/08_train_sae_comparison.py --device cuda
    python scripts/sae_corpus/08_train_sae_comparison.py --device cuda --skip-train
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.sparse_autoencoder import SAEConfig, train_sae, SparseAutoencoder
from scripts._project_root import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round10"

# Round 8 best config for TabPFN (matryoshka_archetypal)
BEST_CONFIG = dict(
    input_dim=192,
    hidden_dim=768,
    sparsity_type="matryoshka_archetypal",
    sparsity_penalty=0.00762,
    topk=256,
    learning_rate=0.000338,
    archetypal_simplex_temp=0.134,
    archetypal_n_archetypes=2000,
    archetypal_relaxation=1.658,
    aux_loss_type="auxk",
    aux_loss_alpha=0.03125,
    aux_loss_warmup_epochs=3,
    resample_dead_neurons=True,
    resample_interval=25000,
    resample_samples=1024,
    batch_size=128,
    n_epochs=100,
    adam_eps=6.25e-10,
    weight_ema_decay=0.999,
    normalize_encoder=True,
    use_lr_schedule=True,
)

VARIANTS = {
    "task_aware": {
        "train_path": OUTPUT_DIR / "tabpfn_taskaware_sae_training.npz",
        "test_path": OUTPUT_DIR / "tabpfn_taskaware_sae_test.npz",
        "stats_path": OUTPUT_DIR / "tabpfn_taskaware_norm_stats.npz",
        "sae_path": OUTPUT_DIR / "tabpfn_taskaware_sae.pt",
    },
    "per_dataset": {
        "train_path": OUTPUT_DIR / "tabpfn_perds_sae_training.npz",
        "test_path": OUTPUT_DIR / "tabpfn_perds_sae_test.npz",
        "stats_path": OUTPUT_DIR / "tabpfn_perds_norm_stats.npz",
        "sae_path": OUTPUT_DIR / "tabpfn_perds_sae.pt",
    },
}


def train_variant(name: str, paths: dict, device: str, seed: int = 42) -> dict:
    """Train SAE for one variant and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print("=" * 60)

    train_data = np.load(str(paths["train_path"]), allow_pickle=True)
    test_data = np.load(str(paths["test_path"]), allow_pickle=True)

    train_emb = train_data["embeddings"].astype(np.float32)
    test_emb = test_data["embeddings"].astype(np.float32)

    print(f"  Train: {train_emb.shape}")
    print(f"  Test:  {test_emb.shape}")

    config = SAEConfig(**BEST_CONFIG)
    config.seed = seed

    t0 = time.time()
    sae, result = train_sae(train_emb, config, device=device, verbose=True)
    dt = time.time() - t0

    # Evaluate on test set
    sae.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_emb, dtype=torch.float32, device=device)
        h = sae.encode(test_tensor)
        recon = sae.decode(h)
        test_mse = ((test_tensor - sae.b_dec - recon + sae.b_dec) ** 2).mean().item()
        # Proper reconstruction: decode includes b_dec
        test_recon = ((test_tensor - recon) ** 2).mean().item()
        alive = (h > 0).any(dim=0).sum().item()
        l0 = (h > 0).float().sum(dim=1).mean().item()

    metrics = {
        "variant": name,
        "train_shape": list(train_emb.shape),
        "test_shape": list(test_emb.shape),
        "train_recon_loss": float(result.reconstruction_loss),
        "test_recon_loss": float(test_recon),
        "alive_features": int(alive),
        "alive_pct": float(alive / config.hidden_dim * 100),
        "l0_sparsity": float(l0),
        "training_time_s": float(dt),
    }

    # Save SAE
    torch.save({
        "state_dict": sae.state_dict(),
        "config": BEST_CONFIG,
        "metrics": metrics,
        "variant": name,
    }, str(paths["sae_path"]))

    print(f"\n  Results:")
    print(f"    Train recon: {metrics['train_recon_loss']:.4f}")
    print(f"    Test recon:  {metrics['test_recon_loss']:.4f}")
    print(f"    Alive:       {metrics['alive_features']}/{config.hidden_dim} "
          f"({metrics['alive_pct']:.1f}%)")
    print(f"    L0:          {metrics['l0_sparsity']:.1f}")
    print(f"    Time:        {dt:.1f}s")
    print(f"  Saved → {paths['sae_path'].name}")

    return metrics


def load_sae(path: Path, device: str) -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    config = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(config)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare fixed vs per-dataset layer SAEs"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing SAEs for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_metrics = {}

    # Train both variants
    for name, paths in VARIANTS.items():
        if args.skip_train and paths["sae_path"].exists():
            print(f"\nSkipping training for {name} (exists)")
            ckpt = torch.load(str(paths["sae_path"]), map_location="cpu", weights_only=False)
            all_metrics[name] = ckpt["metrics"]
        else:
            all_metrics[name] = train_variant(name, paths, args.device, seed=args.seed)

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Fixed L18':>15} {'Per-dataset':>15}")
    print("-" * 55)
    for key in ["train_recon_loss", "test_recon_loss", "alive_features", "alive_pct", "l0_sparsity"]:
        v_fixed = all_metrics["fixed"].get(key, "N/A")
        v_perds = all_metrics["per_dataset"].get(key, "N/A")
        if isinstance(v_fixed, float):
            print(f"{key:<25} {v_fixed:>15.4f} {v_perds:>15.4f}")
        else:
            print(f"{key:<25} {v_fixed:>15} {v_perds:>15}")

    # Save results
    results_path = OUTPUT_DIR / "layer_comparison_results.json"
    json.dump(all_metrics, open(str(results_path), "w"), indent=2)
    print(f"\n→ {results_path}")


if __name__ == "__main__":
    main()
