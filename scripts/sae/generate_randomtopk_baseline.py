#!/usr/bin/env python3
"""Generate data-free random top-k SAE baselines.

The existing random baseline (generate_random_baseline.py) keeps each archetypal
SAE's `reference_data` (K-means centroids of real embeddings), so its dictionary is
still a random convex combination of real data points -- data-informed. That is why
it scores close to the trained SAE, and why a reviewer found random ~= trained.

This script builds the rung *below* that one: a plain top-k SAE with the SAME
input_dim / hidden_dim / topk as the trained SAE, but whose dictionary is unit-norm
random directions carrying NO data geometry. The only thing that differs from the
archetypal-random control is the decoder, so the three-rung ladder

    purely random (this)  <  data-informed random (archetypal)  <  trained

isolates the contribution of the archetypal data scaffold.

At inference `matryoshka_archetypal` and `topk` share an identical encode (per-sample
TopK on pre-activations; the matryoshka nesting is a train-time loss only) and
`get_matryoshka_bands()` derives bands from hidden_dim alone, so demoting the type
changes nothing downstream except the (now data-free) decoder.

Checkpoints are written to BOTH conventions the pipeline consumes:
  - matching:          sweep_dir/{model}/sae_randomtopk_baseline.pt
  - ablation/transfer: random_topk_sae_dir()/{model}/<SAE_FILENAME>  (--sae-dir arm)

Usage:
    python -m scripts.sae.generate_randomtopk_baseline
    python -m scripts.sae.generate_randomtopk_baseline --models tabpfn mitra
"""
import argparse
import dataclasses
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts._project_root import PROJECT_ROOT
from scripts.round_paths import random_topk_sae_dir, sae_sweep_dir
from scripts.sae.compare_sae_cross_model import SAE_FILENAME
from analysis.sparse_autoencoder import ARCHETYPAL_TYPES, SAEConfig, SparseAutoencoder

# The 6-model matched set (HyperFast is handled by its own experiments).
DEFAULT_MODELS = ["tabpfn", "mitra", "tabicl", "tabicl_v2", "tabdpt", "carte"]

# Filename read by the matching script from the trained SAE's directory.
MATCHING_FILENAME = "sae_randomtopk_baseline.pt"


def _load_config(ckpt_path: Path) -> SAEConfig:
    config = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    if isinstance(config, dict):
        config = SAEConfig(**config)
    return config


def generate_randomtopk_sae(model: str, sweep_dir: Path, arm_dir: Path, seed: int):
    """Build a unit-norm random top-k SAE matching the trained SAE's dimensions."""
    ckpt_path = sweep_dir / model / SAE_FILENAME
    if not ckpt_path.exists():
        print(f"{model}: no validated checkpoint at {ckpt_path}, skipping")
        return

    trained_config = _load_config(ckpt_path)
    if trained_config.sparsity_type not in ARCHETYPAL_TYPES:
        raise ValueError(
            f"{model}: expected an archetypal trained SAE, got "
            f"{trained_config.sparsity_type!r}")

    # One variable changes: demote to a plain top-k SAE (free, data-free decoder).
    config = dataclasses.replace(trained_config, sparsity_type="topk")

    # Per-model reproducible stream, independent of which models are requested.
    gen = torch.Generator().manual_seed(seed ^ zlib.crc32(model.encode()))

    sae = SparseAutoencoder(config)
    # Decoder atoms (columns) = unit-norm random directions; encoder aligned (Gao 2024).
    W_dec = torch.randn(config.input_dim, config.hidden_dim, generator=gen)
    sae.W_dec.data = F.normalize(W_dec, dim=0)
    sae.W_enc.data = sae.W_dec.data.T.clone()
    sae.b_enc.data.zero_()
    sae.b_dec.data.zero_()

    col_norms = sae.W_dec.data.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), \
        f"{model}: decoder columns not unit-norm"

    payload = {
        "config": config,
        "model_state_dict": sae.state_dict(),
        "random_baseline": True,
        "random_baseline_kind": "randomtopk",
        "seed": seed,
    }

    # Ablation/transfer arm (read via --sae-dir) + matching location (trained dir).
    arm_path = arm_dir / model / SAE_FILENAME
    arm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arm_path)
    match_path = sweep_dir / model / MATCHING_FILENAME
    torch.save(payload, match_path)

    n_params = sum(p.numel() for p in sae.parameters())
    print(f"{model}: saved ({n_params:,} params, input={config.input_dim}, "
          f"hidden={config.hidden_dim}, topk={config.topk})")
    print(f"    arm      -> {arm_path.relative_to(PROJECT_ROOT)}")
    print(f"    matching -> {match_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate data-free random top-k SAE baselines")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm-dir", type=Path, default=random_topk_sae_dir())
    args = parser.parse_args()

    sweep = sae_sweep_dir()
    print(f"Source (trained SAEs): {sweep}")
    print(f"Arm dir:               {args.arm_dir}\n")

    for model in args.models:
        generate_randomtopk_sae(model, sweep, args.arm_dir, args.seed)

    print(f"\nDone. Random top-k baselines in {args.arm_dir}/ and {sweep}/*/{MATCHING_FILENAME}")


if __name__ == "__main__":
    main()
