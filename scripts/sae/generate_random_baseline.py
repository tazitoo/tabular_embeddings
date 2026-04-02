#!/usr/bin/env python3
"""Generate randomly initialized SAE checkpoints as baselines.

For each model's validated SAE, creates a new SAE with the same config
but random weights (no training). Saves to output/sae_random_baseline/{model}/.

These serve as a control: if ablation with random SAE features closes
gaps comparably to trained SAEs, our learned features aren't meaningful.

Usage:
    python -m scripts.sae.generate_random_baseline
    python -m scripts.sae.generate_random_baseline --models tabpfn mitra
"""
import argparse
from pathlib import Path

import torch

from scripts._project_root import PROJECT_ROOT
from scripts.sae.compare_sae_cross_model import sae_sweep_dir, SAE_FILENAME
from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig

DEFAULT_MODELS = [
    "tabpfn", "mitra", "tabicl", "tabicl_v2",
    "tabdpt", "carte", "hyperfast",
]

OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_random_baseline"


def generate_random_sae(model: str, sweep_dir: Path, out_dir: Path):
    """Create a randomly initialized SAE with the same config as the trained one."""
    ckpt_path = sweep_dir / model / SAE_FILENAME
    if not ckpt_path.exists():
        print(f"{model}: no validated checkpoint at {ckpt_path}, skipping")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = SAEConfig(**config)

    # Fresh SAE with random weights
    sae = SparseAutoencoder(config)

    # For archetypal SAEs, copy reference_data (needed for decode)
    # but randomize the archetype parameters
    state = ckpt["model_state_dict"]
    if "reference_data" in state and state["reference_data"] is not None:
        sae.register_buffer("reference_data", state["reference_data"])
        if "archetype_logits" in state:
            sae.archetype_logits = torch.nn.Parameter(
                torch.randn_like(state["archetype_logits"]))
        if "archetype_deviation" in state:
            sae.archetype_deviation = torch.nn.Parameter(
                torch.randn_like(state["archetype_deviation"]) * 0.01)

    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / SAE_FILENAME

    torch.save({
        "config": config,
        "model_state_dict": sae.state_dict(),
        "random_baseline": True,
    }, save_path)

    n_params = sum(p.numel() for p in sae.parameters())
    print(f"{model}: saved ({n_params:,} params, "
          f"input={config.input_dim}, hidden={config.hidden_dim}) "
          f"-> {save_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomly initialized SAE baselines")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    sweep = sae_sweep_dir()
    print(f"Source: {sweep}")
    print(f"Output: {args.output_dir}")
    print()

    for model in args.models:
        generate_random_sae(model, sweep, args.output_dir)

    print(f"\nDone. Random baselines in {args.output_dir}/")


if __name__ == "__main__":
    main()
