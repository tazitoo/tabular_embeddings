#!/usr/bin/env python3
"""Extract fundamental SAE metrics from TabICL sweep results."""
import torch
import numpy as np
from pathlib import Path

# Models to analyze
model_dir = Path("output/sae_tabarena_sweep/tabicl_layer10")
models = [
    ("L1", "sae_l1_validated.pt"),
    ("TopK", "sae_topk_validated.pt"),
    ("Matryoshka", "sae_matryoshka_validated.pt"),
    ("Archetypal", "sae_archetypal_validated.pt"),
    ("Mat-Arch", "sae_matryoshka_archetypal_validated.pt"),
    ("Mat-BatchTopK-Arch", "sae_matryoshka_batchtopk_archetypal_validated.pt"),
]

print("\n" + "="*95)
print("TABICL LAYER 10 - SAE FUNDAMENTAL METRICS")
print("="*95)
print(f"{'Architecture':<22} {'Dict':>6} {'Exp':>5} {'L0':>6} {'Dead%':>7} {'Loss':>10} {'R²':>7} {'Stab':>7}")
print("-"*95)

for arch_name, model_name in models:
    model_path = model_dir / model_name
    if not model_path.exists():
        print(f"{arch_name:<22} MODEL NOT FOUND")
        continue

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model = checkpoint['model']
    config = model.config
    metrics = checkpoint.get('metrics', {})

    # Extract metrics from checkpoint or config
    dict_size = config.hidden_dim
    input_dim = config.input_dim
    expansion = dict_size / input_dim

    # Get metrics (these were saved during validation)
    loss = metrics.get('loss', checkpoint.get('loss', 0.0))
    r2 = metrics.get('r2', checkpoint.get('r2', 0.0))
    stability = metrics.get('stability', checkpoint.get('stability', 0.0))

    # Try to get L0 and dead neuron count
    l0 = metrics.get('l0', metrics.get('mean_l0', 0))
    if l0 == 0 and hasattr(config, 'topk'):
        l0 = config.topk  # TopK architectures have fixed L0

    pct_dead = metrics.get('pct_dead', 0)
    if pct_dead == 0 and 'alive_features' in metrics:
        pct_dead = 100 * (1 - metrics['alive_features'] / dict_size)

    print(f"{arch_name:<22} {dict_size:>6} {expansion:>4.1f}x {l0:>6.0f} {pct_dead:>6.1f}% {loss:>10.6f} {r2:>7.4f} {stability:>7.4f}")

print("="*95)
print("\nColumn Definitions:")
print("  Dict:  Dictionary size (number of learned feature detectors)")
print("  Exp:   Expansion factor (dict_size / input_dim, typically 4-8x)")
print("  L0:    Average # of active features per sample (lower = more sparse)")
print("  Dead%: % of features firing <0.1% of the time (lower = better)")
print("  Loss:  Reconstruction MSE (lower = better)")
print("  R²:    Explained variance, 1.0 = perfect (higher = better)")
print("  Stab:  Stability across retraining runs (higher = more reproducible)")
print()
print("Note: L0 and Dead% require forward pass - using cached values if available.")
print("      Run full analysis with embeddings for complete metrics.")
