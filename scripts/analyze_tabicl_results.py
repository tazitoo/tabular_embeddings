#!/usr/bin/env python3
"""Extract fundamental SAE metrics from TabICL sweep results."""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.sparse_autoencoder import SparseAutoencoder
from data.extended_loader import load_tabarena_datasets

# Load TabICL embeddings
train_datasets, test_datasets = load_tabarena_datasets(task_type='classification')
from models.tabicl_model import extract_tabicl_embeddings

print("Loading TabICL embeddings from train datasets...")
embeddings_list = []
for name in train_datasets[:34]:  # First 34 for train
    try:
        X, y = train_datasets[name]
        emb = extract_tabicl_embeddings(X, y, layer=10)
        if emb is not None:
            embeddings_list.append(emb[:100])  # 100 samples per dataset
    except Exception as e:
        print(f"Skipping {name}: {e}")
        continue

embeddings = np.vstack(embeddings_list)
print(f"Total samples: {len(embeddings)}, dim: {embeddings.shape[1]}")

# Center embeddings
embeddings = embeddings - embeddings.mean(axis=0)

# Models to analyze
model_dir = Path("output/sae_tabarena_sweep/tabicl_layer10")
models = [
    "sae_l1_validated.pt",
    "sae_topk_validated.pt",
    "sae_matryoshka_validated.pt",
    "sae_archetypal_validated.pt",
    "sae_matryoshka_archetypal_validated.pt",
    "sae_matryoshka_batchtopk_archetypal_validated.pt",
]

print("\n" + "="*80)
print("TABICL LAYER 10 - SAE FUNDAMENTAL METRICS")
print("="*80)
print(f"{'Architecture':<30} {'Dict Size':>10} {'Exp':>5} {'L0':>8} {'Dead %':>8} {'Loss':>10} {'R²':>8}")
print("-"*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = torch.tensor(embeddings, dtype=torch.float32, device=device)

for model_name in models:
    model_path = model_dir / model_name
    if not model_path.exists():
        continue

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.eval()

    # Get architecture name
    arch_name = model_name.replace("sae_", "").replace("_validated.pt", "")

    # Compute metrics
    with torch.no_grad():
        x_hat, h = model(X)

        # Reconstruction loss
        loss = torch.nn.functional.mse_loss(x_hat, X).item()

        # R² (explained variance)
        ss_res = ((X - x_hat) ** 2).sum().item()
        ss_tot = ((X - X.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)

        # L0 (average sparsity)
        l0 = (h > 0).float().sum(dim=1).mean().item()

        # Dead neurons (activation freq < 1e-3)
        activation_freq = (h > 0).float().mean(dim=0)
        dead_mask = activation_freq < 1e-3
        pct_dead = 100 * dead_mask.float().mean().item()

        # Dictionary size
        dict_size = model.config.hidden_dim
        input_dim = model.config.input_dim
        expansion = dict_size / input_dim

    print(f"{arch_name:<30} {dict_size:>10} {expansion:>5.1f}x {l0:>8.1f} {pct_dead:>7.1f}% {loss:>10.6f} {r2:>8.4f}")

print("="*80)
print("\nMetrics:")
print("  Dict Size: Number of dictionary atoms (features)")
print("  Exp: Expansion factor (dict_size / input_dim)")
print("  L0: Average number of active features per sample (sparsity)")
print("  Dead %: Percentage of features that fire <0.1% of the time")
print("  Loss: Mean squared reconstruction error")
print("  R²: Explained variance (1 = perfect reconstruction)")
