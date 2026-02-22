#!/usr/bin/env python3
"""
Reproduce domain-level reconstruction plot for TabICL SAEs.
Shows R² by domain across Matryoshka cumulative scales.
"""
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.sparse_autoencoder import SparseAutoencoder, SAEConfig
from data.extended_loader import TABARENA_DATASETS
from scripts.compare_sae_cross_model import sae_sweep_dir

# Domain mapping for TabArena datasets
DOMAIN_COLORS = {
    'Adult': '#1f77b4',      # Blue
    'Physiology': '#ff7f0e', # Orange
    'QSAR': '#2ca02c',       # Green
    'Text': '#d62728',       # Red
    'Technology': '#9467bd', # Purple
}

def get_dataset_domain(dataset_name: str) -> str:
    """Map dataset to domain category."""
    # Biology/Medicine
    if dataset_name in ['blood-transfusion', 'breast-cancer', 'diabetes', 'heart-disease',
                        'hypothyroid', 'physiochemical']:
        return 'Physiology'

    # Chemistry/Materials
    if dataset_name in ['qsar-biodeg', 'QSAR_fish', 'superconductivity']:
        return 'QSAR'

    # Text/NLP
    if dataset_name in ['amazon', 'blastchar', 'soybean']:
        return 'Text'

    # Technology/Systems
    if dataset_name in ['APSFailure', 'Bioresponse', 'CPU_1', 'electricity', 'phoneme']:
        return 'Technology'

    # Default: Adult/Social (everything else)
    return 'Adult'


def load_embeddings_by_domain(model_name: str = "tabicl_layer10"):
    """Load TabICL embeddings grouped by domain."""
    from data.extended_loader import TABARENA_DATASETS

    # Get classification datasets
    all_datasets = sorted([k for k, v in TABARENA_DATASETS.items()
                          if v['task'] == 'classification'])
    train_datasets = all_datasets[:34]

    # Group by domain
    domain_embeddings = {domain: [] for domain in DOMAIN_COLORS.keys()}
    domain_dataset_names = {domain: [] for domain in DOMAIN_COLORS.keys()}

    for ds_name in train_datasets:
        emb_path = Path(f"output/embeddings/tabarena/{model_name}/tabarena_{ds_name}.npz")
        if not emb_path.exists():
            continue

        data = np.load(emb_path)
        emb = data['embeddings'].astype(np.float32)

        domain = get_dataset_domain(ds_name)
        domain_embeddings[domain].append(emb[:100])  # 100 samples per dataset
        domain_dataset_names[domain].append(ds_name)

    # Concatenate embeddings per domain
    domain_data = {}
    for domain in DOMAIN_COLORS.keys():
        if domain_embeddings[domain]:
            domain_data[domain] = {
                'embeddings': np.vstack(domain_embeddings[domain]),
                'datasets': domain_dataset_names[domain],
            }
            print(f"{domain}: {len(domain_data[domain]['embeddings'])} samples from "
                  f"{len(domain_data[domain]['datasets'])} datasets")

    return domain_data


def compute_reconstruction_by_scale(model, embeddings, scales, device='cuda'):
    """Compute R² at each cumulative Matryoshka scale."""
    model.eval()
    model.to(device)

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    X_centered = X - X.mean(dim=0)

    r2_by_scale = {}

    with torch.no_grad():
        # Full forward pass
        x_hat_full, h_full = model(X_centered)

        # For each scale, use only first n features
        for scale in scales:
            if scale > h_full.shape[1]:
                continue

            # Truncate activations to this scale
            h_truncated = h_full.clone()
            h_truncated[:, scale:] = 0.0  # Zero out higher-scale features

            # Reconstruct using truncated activations
            x_hat = model.decode(h_truncated)

            # Compute R²
            ss_res = ((X_centered - x_hat) ** 2).sum().item()
            ss_tot = (X_centered ** 2).sum().item()
            r2 = 1 - (ss_res / ss_tot)

            r2_by_scale[scale] = r2

    return r2_by_scale


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load embeddings by domain
    print("Loading embeddings by domain...")
    domain_data = load_embeddings_by_domain()

    # Center embeddings per domain
    for domain in domain_data:
        emb = domain_data[domain]['embeddings']
        domain_data[domain]['embeddings'] = emb - emb.mean(axis=0)

    # Models to analyze (focus on Matryoshka variants)
    models = [
        ("Matryoshka", "sae_matryoshka_validated.pt"),
        ("Mat-Arch", "sae_matryoshka_archetypal_validated.pt"),
        ("Mat-BatchTopK-Arch", "sae_matryoshka_batchtopk_archetypal_validated.pt"),
    ]

    model_dir = sae_sweep_dir() / "tabicl_layer10"

    # Matryoshka scales to evaluate
    scales = [32, 64, 128, 256]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax_idx, (model_name, model_file) in enumerate(models):
        ax = axes[ax_idx]
        ax.set_title(f"TabICL - {model_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Matryoshka cumulative scale", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Reconstruction R²", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)

        # Load model
        checkpoint = torch.load(model_dir / model_file, map_location='cpu')
        config = checkpoint['config']
        if isinstance(config, dict):
            config = SAEConfig(**config)

        model = SparseAutoencoder(config)

        # Handle archetypal models
        if 'archetypal' in model_name.lower():
            n_archetypes = getattr(config, 'archetypal_n_archetypes', None) or config.hidden_dim
            ref_data = torch.randn(n_archetypes, config.input_dim)
            model.set_reference_data(ref_data)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Compute reconstruction by domain and scale
        for domain, color in DOMAIN_COLORS.items():
            if domain not in domain_data:
                continue

            embeddings = domain_data[domain]['embeddings']
            r2_by_scale = compute_reconstruction_by_scale(model, embeddings, scales, device)

            # Plot
            scale_values = sorted(r2_by_scale.keys())
            r2_values = [r2_by_scale[s] for s in scale_values]

            ax.plot(scale_values, r2_values, 'o-', color=color, label=domain,
                   linewidth=2, markersize=6, alpha=0.8)

    # Add legend to first plot only
    axes[0].legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = Path("output/figures/tabicl_domain_reconstruction.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved to {output_path}")
    print(f"✓ Saved to {output_path.with_suffix('.png')}")


if __name__ == '__main__':
    main()
