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
    """Load pooled TabICL embeddings - use pre-extracted embeddings from sweep."""
    # Use exact same logic as sae_tabarena_sweep.py pool_embeddings()
    from data.extended_loader import TABARENA_DATASETS

    model_name = "tabicl_layer10"
    print(f"Pooling {model_name} embeddings from train datasets...")

    # Get train split - first 34 classification datasets
    all_datasets = sorted([k for k, v in TABARENA_DATASETS.items()
                          if v['task'] == 'classification'])
    train_datasets = all_datasets[:34]

    all_embeddings = []
    n_samples_per_ds = 100

    for ds_name in train_datasets:
        # Try to find the embedding file (check for exact match and variations)
        emb_path = Path(f"output/embeddings/tabarena/{model_name}/tabarena_{ds_name}.npz")
        if not emb_path.exists():
            continue

        data = np.load(emb_path)
        emb = data['embeddings'].astype(np.float32)

        # Take up to 100 samples
        n_samples = min(n_samples_per_ds, len(emb))
        all_embeddings.append(emb[:n_samples])

    if not all_embeddings:
        raise ValueError(f"No embeddings found for {model_name}")

    pooled = np.vstack(all_embeddings)
    print(f"  Total samples: {len(pooled)}")
    print(f"  Embedding dim: {pooled.shape[1]}")
    print(f"  Datasets loaded: {len(all_embeddings)}")

    return pooled

def compute_sae_metrics(model, embeddings, device='cuda'):
    """Compute L0, Dead%, RMSE, R², c_dec for a trained SAE."""
    from analysis.sparse_autoencoder import compute_c_dec

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

        # c_dec (decoder pairwise cosine similarity)
        # Get decoder from model (handle archetypal models)
        if hasattr(model, 'get_archetypal_dictionary'):
            decoder = model.get_archetypal_dictionary().cpu().numpy()
        else:
            decoder = model.W_dec.data.T.cpu().numpy()  # (hidden, input)

        c_dec = compute_c_dec(decoder)

    return {
        'rmse': rmse,
        'r2': r2,
        'l0': l0,
        'pct_dead': pct_dead,
        'c_dec': c_dec,
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

    print("\n" + "="*116)
    print("Table: SAE Architecture Comparison on TabICL Embeddings (34 TabArena train datasets)")
    print("="*116)
    print(f"{'Type':<24} {'Hyperparameters':<35} {'RMSE':>8} {'Stab':>7} {'s_n':>7} {'c_d':>7} {'L₀':>6} {'Dead%':>7}")
    print("-"*116)

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

        # For archetypal models, need to set reference data before loading weights
        if 'archetypal' in arch_name.lower():
            # Get n_archetypes from config (defaults to hidden_dim if not set)
            n_archetypes = getattr(config_obj, 'archetypal_n_archetypes', None)
            if n_archetypes is None:
                n_archetypes = config_obj.hidden_dim

            # Use dummy reference data for initialization (will be overwritten by loaded state)
            ref_data = torch.randn(n_archetypes, config_obj.input_dim)
            model.set_reference_data(ref_data)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Get stability metrics from checkpoint
        stability = checkpoint.get('metrics', {}).get('stability', 0.0)
        if stability == 0:
            stability = checkpoint.get('stability', 0.0)

        s_n_dec = checkpoint.get('metrics', {}).get('s_n_dec', 0.0)
        if s_n_dec == 0:
            s_n_dec = checkpoint.get('s_n_dec', 0.0)

        # Compute metrics
        metrics = compute_sae_metrics(model, embeddings, device)

        # Format hyperparameters
        m = config_obj.hidden_dim // config_obj.input_dim  # expansion
        k = getattr(config_obj, 'topk', '-')
        n = getattr(config_obj, 'archetypal_n_archetypes', '-')
        if n == '-' or n is None:
            n = '-'

        # Resampling parameters (new universal approach)
        resample_interval = getattr(config_obj, 'resample_interval', '-')
        resample_samples = getattr(config_obj, 'resample_samples', '-')

        hyperparam_str = f"m={m}"
        if k != '-':
            hyperparam_str += f", k={k}"
        if n != '-':
            hyperparam_str += f", n={n}"
        if resample_interval != '-':
            # Format in thousands (e.g., 10000 -> 10k)
            ri_k = resample_interval // 1000 if resample_interval >= 1000 else resample_interval
            hyperparam_str += f", ri={ri_k}k"
        if resample_samples != '-':
            hyperparam_str += f", rs={resample_samples}"

        # Print row
        print(f"{arch_name:<24} {hyperparam_str:<35} {metrics['rmse']:>8.3f} {stability:>7.3f} "
              f"{s_n_dec:>7.3f} {metrics['c_dec']:>7.3f} {int(metrics['l0']):>6} {metrics['pct_dead']:>6.1f}%")

        results.append({
            'arch': arch_name,
            'rmse': metrics['rmse'],
            'stab': stability,
            'l0': int(metrics['l0']),
            'dead_pct': metrics['pct_dead'],
        })

    print("="*116)
    print("\nHyperparameters: m = expansion factor (hidden_dim = m × input_dim), k = TopK sparsity,")
    print("                 n = archetypes, ri = resample_interval (×1000 steps),")
    print("                 rs = resample_samples (high-error samples for neuron revival)")
    print("\nMetrics: RMSE = reconstruction error,")
    print("         Stab = feature alignment stability [0-1, higher = more reproducible],")
    print("         s_n = s_n^dec decoder stability [0-∞, lower = better, 0 = optimal sparsity],")
    print("         c_d = c_dec pairwise cosine similarity [0-1, lower = less redundancy],")
    print("         L₀ = avg. active features, Dead% = never-activated features.")
    print("\nAll architectures use residual_targeting aux loss + neuron resampling (universal approach).")
    print()

if __name__ == '__main__':
    main()
