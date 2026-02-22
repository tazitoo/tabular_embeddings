#!/usr/bin/env python3
"""
Display comprehensive SAE metrics for a trained model.

Usage:
    python scripts/show_sae_metrics.py --model mitra_layer12
    python scripts/show_sae_metrics.py --model tabicl_layer10
    python scripts/show_sae_metrics.py --model tabpfn_layer16
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig, compute_c_dec
from data.tabarena_utils import load_embeddings_raw, get_tabarena_splits
from scripts.compare_sae_cross_model import sae_sweep_dir


def compute_sae_metrics(model, embeddings, device='cpu'):
    """Compute comprehensive SAE metrics."""
    model.eval()
    model.to(device)

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)

    with torch.no_grad():
        x_hat, h = model(X)

        # RMSE
        rmse = torch.sqrt(torch.nn.functional.mse_loss(x_hat, X)).item()

        # R²
        ss_res = ((X - x_hat) ** 2).sum().item()
        ss_tot = ((X - X.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot)

        # L0
        l0 = (h > 0).float().sum(dim=1).mean().item()

        # Dead%
        activation_freq = (h > 0).float().mean(dim=0)
        dead_mask = activation_freq < 1e-3
        pct_dead = 100 * dead_mask.float().mean().item()

        # c_dec (decoder pairwise cosine similarity)
        if hasattr(model, 'reference_data') and model.reference_data is not None:
            decoder = model.get_archetypal_dictionary().cpu().numpy()
        else:
            decoder = model.W_dec.data.T.cpu().numpy()
        c_dec = compute_c_dec(decoder)

    return {
        'rmse': rmse,
        'r2': r2,
        'l0': l0,
        'pct_dead': pct_dead,
        'c_dec': c_dec,
    }


def main():
    parser = argparse.ArgumentParser(description='Display SAE metrics')
    parser.add_argument('--model', required=True,
                       help='Model name (e.g., mitra_layer12, tabicl_layer10)')
    parser.add_argument('--architecture', default='matryoshka_archetypal',
                       help='SAE architecture (default: matryoshka_archetypal)')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    # Load RAW embeddings (SAE's BatchNorm will apply learned normalization)
    print(f'Loading {args.model} embeddings...')
    train_datasets, _ = get_tabarena_splits()
    embeddings, _ = load_embeddings_raw(args.model, train_datasets, max_per_dataset=100)
    print(f'  Raw train embeddings: {embeddings.shape}')
    print()

    # Load SAE
    model_path = sae_sweep_dir() / f'{args.model}/sae_{args.architecture}_validated.pt'
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    print(f'Loading SAE: {model_path.name}')
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    if isinstance(config, dict):
        config = SAEConfig(**config)

    model = SparseAutoencoder(config)
    # Load state dict which includes reference_data for archetypal models
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Compute metrics
    metrics = compute_sae_metrics(model, embeddings, device=args.device)

    # Get checkpoint metrics
    ckpt_metrics = checkpoint.get('metrics', {})
    stability = ckpt_metrics.get('stability', 0.0)
    s_n_dec = ckpt_metrics.get('s_n_dec', 0.0)

    # Get hyperparameters
    expansion = config.hidden_dim // config.input_dim
    topk = getattr(config, 'topk', '-')
    n_arch = getattr(config, 'archetypal_n_archetypes', '-')

    # Print results
    print('='*100)
    print(f'{args.model.upper()} - {args.architecture.replace("_", "-").title()} SAE Metrics')
    print('='*100)
    print(f'Hyperparameters: m={expansion}, k={topk}, n={n_arch}')
    print()
    print(f'{"Metric":<20} {"Value":>15}')
    print('-'*100)
    print(f'{"RMSE":<20} {metrics["rmse"]:>15.4f}')
    print(f'{"R²":<20} {metrics["r2"]:>15.4f}')
    print(f'{"Stability":<20} {stability:>15.4f}')
    print(f'{"s_n^dec":<20} {s_n_dec:>15.4f}')
    print(f'{"c_dec":<20} {metrics["c_dec"]:>15.4f}')
    print(f'{"L₀":<20} {int(metrics["l0"]):>15}')
    print(f'{"Dead%":<20} {metrics["pct_dead"]:>14.1f}%')
    print('='*100)

    # Compare to training metrics
    if 'r2' in ckpt_metrics:
        print()
        print('Training metrics (from checkpoint):')
        print(f'  R²: {ckpt_metrics["r2"]:.4f}')
        if 'l0_sparsity' in ckpt_metrics:
            print(f'  L₀: {ckpt_metrics["l0_sparsity"]:.1f}')
        if 'alive_features' in ckpt_metrics:
            print(f'  Alive features: {ckpt_metrics["alive_features"]}')
        print()


if __name__ == '__main__':
    main()
