#!/usr/bin/env python3
"""
Complete SAE metrics table for TabICL (matches Table 8 format from paper).
Computes L0 and Dead% by running forward passes through saved models.
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig

def load_tabicl_embeddings():
    """Load pooled TabICL embeddings from sweep cache."""
    # The sweep saves embeddings - try to find them
    cache_dirs = [
        Path("output/sae_tabarena_sweep/tabicl_layer10"),
        Path("output/embeddings/tabicl_layer10"),
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            # Check for any pt file with embeddings
            for pt_file in cache_dir.glob("*.pt"):
                try:
                    data = torch.load(pt_file, map_location='cpu')
                    if 'embeddings' in data:
                        embeddings = data['embeddings']
                        print(f"Loaded embeddings from {pt_file.name}: {embeddings.shape}")
                        return np.array(embeddings)
                except:
                    continue

    # Fallback: load from saved checkpoint
    print("Could not find cached embeddings, loading from model checkpoint...")
    model_path = Path("output/sae_tabarena_sweep/tabicl_layer10/sae_l1_validated.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'embeddings' in checkpoint:
            return np.array(checkpoint['embeddings'])

    # Last resort: generate dummy data (should not happen)
    print("WARNING: Using dummy data - results will be incorrect!")
    return np.random.randn(3400, 512).astype(np.float32)

def compute_sae_metrics(model, embeddings, device='cuda'):
    """Compute L0, Dead%, RMSE, R² for a trained SAE."""
    model.eval()
    model.to(device)

    # Convert to tensor
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Forward pass
        x_hat, h = model(X)

        # RMSE (root mean squared error)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(x_hat, X)).item()

        # R² (explained variance)
        ss_res = ((X - x_hat) ** 2).sum().item()
        ss_tot = ((X - X.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)

        # L0 (average number of active features)
        l0 = (h > 0).float().sum(dim=1).mean().item()

        # Dead% (features that never activate or activate <0.1% of time)
        activation_freq = (h > 0).float().mean(dim=0)
        dead_mask = activation_freq < 1e-3
        pct_dead = 100 * dead_mask.float().mean().item()

    return {
        'rmse': rmse,
        'r2': r2,
        'l0': l0,
        'pct_dead': pct_dead,
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load embeddings
    embeddings = load_tabicl_embeddings()
    embeddings = embeddings - embeddings.mean(axis=0)  # Center
    print(f"Embeddings shape: {embeddings.shape}")

    # Models to analyze
    model_dir = Path("output/sae_tabarena_sweep/tabicl_layer10")
    models = [
        ("Matryoshka-Archetypal", "sae_matryoshka_archetypal_validated.pt"),
        ("Archetypal", "sae_archetypal_validated.pt"),
        ("TopK", "sae_topk_validated.pt"),
        ("Matryoshka", "sae_matryoshka_validated.pt"),
        ("L1", "sae_l1_validated.pt"),
        ("Mat-BatchTopK-Arch", "sae_matryoshka_batchtopk_archetypal_validated.pt"),
    ]

    print("\n" + "="*100)
    print("Table: SAE Architecture Comparison on TabICL Embeddings (34 TabArena train datasets)")
    print("="*100)
    print(f"{'Type':<24} {'Hyperparameters':<35} {'RMSE':>8} {'Stab':>7} {'L₀':>6} {'Dead%':>7}")
    print("-"*100)

    results = []
    for arch_name, model_name in models:
        model_path = model_dir / model_name
        if not model_path.exists():
            print(f"{arch_name:<24} MODEL NOT FOUND")
            continue

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']

        # Reconstruct model from config and state dict
        if isinstance(config, dict):
            config_obj = SAEConfig(**config)
        else:
            config_obj = config

        model = SparseAutoencoder(config_obj)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get stability from checkpoint
        stability = checkpoint.get('metrics', {}).get('stability', 0.0)
        if stability == 0:
            stability = checkpoint.get('stability', 0.0)

        # Compute metrics
        metrics = compute_sae_metrics(model, embeddings, device)

        # Format hyperparameters
        m = config_obj.hidden_dim // config_obj.input_dim  # expansion
        k = getattr(config_obj, 'topk', '-')
        n = getattr(config_obj, 'archetypal_n_archetypes', '-')
        if n == '-' or n is None:
            n = '-'

        hyperparam_str = f"m={m}"
        if k != '-':
            hyperparam_str += f", k={k}"
        if n != '-':
            hyperparam_str += f", n={n}"

        # Print row
        print(f"{arch_name:<24} {hyperparam_str:<35} {metrics['rmse']:>8.3f} {stability:>7.3f} "
              f"{int(metrics['l0']):>6} {metrics['pct_dead']:>6.1f}%")

        results.append({
            'arch': arch_name,
            'rmse': metrics['rmse'],
            'stab': stability,
            'l0': int(metrics['l0']),
            'dead_pct': metrics['pct_dead'],
        })

    print("="*100)
    print("\nHyperparameters: m = expansion factor (hidden_dim = m × input_dim), k = TopK sparsity,")
    print("                 n = archetypes")
    print("\nMetrics: RMSE = reconstruction error, Stab = s_n^dec decoder stability [?],")
    print("         L₀ = avg. active features, Dead% = never-activated features.")
    print()

if __name__ == '__main__':
    main()
