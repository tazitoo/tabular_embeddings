#!/usr/bin/env python3
"""
Section 4.3 — Universal vs. Model-Specific Concepts.

Analyzes whether SAE-learned concepts generalize across application domains
or are domain-specific. Uses 7 trained SAEs over 36 common TabArena datasets
spanning 6 merged application domains.

Four analysis layers:
  Layer 0: Domain reconstruction R² at each Matryoshka scale
  Layer 1: Feature selectivity profiles (universal / domain-cluster / domain-specific)
  Layer 2: Data-driven domain taxonomy (ARI/NMI)
  Layer 3: (Future) Per-domain concept coverage

Outputs:
  - output/domain_concept_analysis.json (full numerical results)
  - scripts/section43/domain_concepts.pdf (2-panel figure)

Usage:
    python scripts/section43/universal_concepts.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    find_common_datasets,
    sae_sweep_dir,
)
from scripts.compare_sae_architectures import (
    compute_activations,
    get_train_test_split,
)
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint

# ---------------------------------------------------------------------------
# Domain taxonomy with merges
# ---------------------------------------------------------------------------

# Merge rules: Natural Sciences + Chemistry & Materials → Science & Materials
# Exclude singletons: Logistics & Commerce (2 datasets), Education & Social (1 dataset)
DOMAIN_MERGES = {
    "Natural Sciences": "Science & Materials",
    "Chemistry & Materials": "Science & Materials",
}
EXCLUDED_DOMAINS = {"Logistics & Commerce", "Education & Social", "Real Estate & Pricing"}

# Short display labels for plotting
DOMAIN_SHORT = {
    "Business & Marketing": "Business",
    "Finance & Insurance": "Finance",
    "Science & Materials": "Science",
    "Healthcare & Life Sciences": "Healthcare",
    "Engineering & Industry": "Engineering",
    "Technology & Security": "Technology",
    "Real Estate & Pricing": "Real Estate",
}


def load_domain_taxonomy() -> Dict[str, str]:
    """Load and apply domain merges. Returns dataset_name → merged_domain."""
    path = PROJECT_ROOT / "data" / "tabarena_domains.json"
    with open(path) as f:
        taxonomy = json.load(f)

    dataset_domain = {}
    for ds, domain in taxonomy["dataset_domain"].items():
        if domain in EXCLUDED_DOMAINS:
            continue
        merged = DOMAIN_MERGES.get(domain, domain)
        dataset_domain[ds] = merged

    return dataset_domain


def pool_embeddings_with_offsets(
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Pool embeddings and track per-dataset row offsets.

    Returns:
        pooled: (n_total, dim) concatenated embeddings
        offsets: {dataset_name: (start_row, end_row)}
    """
    all_embs = []
    offsets = {}
    cursor = 0
    for ds in datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        offsets[ds] = (cursor, cursor + len(emb))
        cursor += len(emb)
        all_embs.append(emb)
    return np.concatenate(all_embs), offsets


def compute_train_stats(
    emb_dir: Path,
    datasets: List[str],
    max_per_dataset: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalization stats from train split only.

    Returns:
        train_std: (1, dim)
        train_mean: (1, dim) — mean of std-normalized train data
    """
    train_datasets, _ = get_train_test_split(datasets)
    train_embs = []
    for ds in train_datasets:
        path = emb_dir / f"tabarena_{ds}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        if len(emb) > max_per_dataset:
            np.random.seed(42)
            idx = np.random.choice(len(emb), max_per_dataset, replace=False)
            emb = emb[idx]
        train_embs.append(emb)
    train_pooled = np.concatenate(train_embs)
    train_std = train_pooled.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    train_norm = train_pooled / train_std
    train_mean = train_norm.mean(axis=0, keepdims=True)
    return train_std, train_mean


def build_domain_row_indices(
    offsets: Dict[str, Tuple[int, int]],
    dataset_domain: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Map each domain to row indices in the pooled embedding array."""
    domain_indices: Dict[str, List[int]] = {}
    for ds, (start, end) in offsets.items():
        domain = dataset_domain.get(ds)
        if domain is None:
            continue
        domain_indices.setdefault(domain, []).extend(range(start, end))
    return {d: np.array(idx) for d, idx in domain_indices.items()}


# ---------------------------------------------------------------------------
# Layer 0: Domain reconstruction R²
# ---------------------------------------------------------------------------

def compute_domain_reconstruction_r2(
    model, activations: np.ndarray, pooled_raw: np.ndarray,
    domain_row_indices: Dict[str, np.ndarray],
    scales: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    R² of SAE reconstruction per domain at each Matryoshka scale.

    With BatchNorm in the SAE:
    - Pass raw embeddings through model.bn() to get normalized inputs
    - Compare against decoder outputs (which are in same normalized space)
    - This ensures fair comparison in BatchNorm-normalized space
    """
    results = {}
    for domain, indices in domain_row_indices.items():
        results[domain] = {}
        # Get BatchNorm-normalized inputs for this domain
        with torch.no_grad():
            x_raw = torch.tensor(pooled_raw[indices], dtype=torch.float32)
            x_true = model.bn(x_raw).numpy()  # Apply learned normalization
        h_domain = torch.tensor(activations[indices], dtype=torch.float32)
        for scale in scales:
            with torch.no_grad():
                x_hat = model.decode(h_domain, max_dim=scale).numpy()
            ss_res = np.sum((x_true - x_hat) ** 2)
            ss_tot = np.sum((x_true - x_true.mean(axis=0)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results[domain][scale] = float(r2)
    return results


# ---------------------------------------------------------------------------
# Layer 1: Feature selectivity profiles
# ---------------------------------------------------------------------------

def compute_feature_selectivity(
    activations: np.ndarray,
    domain_row_indices: Dict[str, np.ndarray],
    config,
    firing_threshold: float = 0.01,
    alive_threshold: float = 0.001,
) -> Dict[str, Dict[str, int]]:
    """Classify features as universal / domain-cluster / domain-specific.

    Returns dict mapping Matryoshka band label to selectivity counts.
    """
    n_domains = len(domain_row_indices)
    domains = sorted(domain_row_indices.keys())
    hidden_dim = activations.shape[1]

    # Compute per-feature, per-domain firing rate
    firing_rates = np.zeros((hidden_dim, n_domains))
    for di, domain in enumerate(domains):
        idx = domain_row_indices[domain]
        firing_rates[:, di] = (activations[idx] > 0).mean(axis=0)

    # Alive mask
    alive = activations.max(axis=0) > alive_threshold

    # Number of domains where feature fires above threshold
    n_active_domains = (firing_rates > firing_threshold).sum(axis=1)

    # Build Matryoshka band boundaries
    mat_dims = [d for d in config.matryoshka_dims if d <= config.hidden_dim]
    boundaries = [0] + mat_dims
    if boundaries[-1] < config.hidden_dim:
        boundaries.append(config.hidden_dim)

    band_labels = []
    for bi in range(len(boundaries) - 1):
        band_labels.append(f"S{bi+1} [{boundaries[bi]},{boundaries[bi+1]})")

    results = {}
    for bi in range(len(boundaries) - 1):
        start, end = boundaries[bi], boundaries[bi + 1]
        band_alive = alive[start:end]
        band_n_active = n_active_domains[start:end]

        n_alive = band_alive.sum()
        if n_alive == 0:
            results[band_labels[bi]] = {
                "universal": 0, "domain_cluster": 0, "domain_specific": 0,
                "dead": int((~band_alive).sum()),
            }
            continue

        # Classify among alive features only
        alive_n_active = band_n_active[band_alive]
        universal = int((alive_n_active >= n_domains).sum())
        domain_specific = int((alive_n_active <= 1).sum())
        domain_cluster = int(n_alive) - universal - domain_specific

        results[band_labels[bi]] = {
            "universal": universal,
            "domain_cluster": domain_cluster,
            "domain_specific": domain_specific,
            "dead": int((~band_alive).sum()),
        }

    return results


# ---------------------------------------------------------------------------
# Layer 2: Data-driven domain taxonomy
# ---------------------------------------------------------------------------

def compute_domain_taxonomy_agreement(
    activations: np.ndarray,
    offsets: Dict[str, Tuple[int, int]],
    dataset_domain: Dict[str, str],
) -> Dict[str, float]:
    """Cluster datasets by mean activation profile; compare to ground-truth domains."""
    ds_names = []
    ds_means = []
    ds_labels = []
    for ds, (start, end) in offsets.items():
        domain = dataset_domain.get(ds)
        if domain is None:
            continue
        ds_names.append(ds)
        ds_means.append(activations[start:end].mean(axis=0))
        ds_labels.append(domain)

    if len(ds_names) < 4:
        return {"ari": 0.0, "nmi": 0.0, "n_datasets": len(ds_names)}

    X = np.vstack(ds_means)
    Z = linkage(X, method='ward')
    n_clusters = len(set(ds_labels))
    pred_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    ari = adjusted_rand_score(ds_labels, pred_labels)
    nmi = normalized_mutual_info_score(ds_labels, pred_labels)
    return {"ari": float(ari), "nmi": float(nmi), "n_datasets": len(ds_names)}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

# Consistent model ordering and colors across the paper
MODEL_ORDER = ["TabPFN", "CARTE", "TabICL", "TabDPT", "Mitra", "HyperFast", "Tabula-8B"]
MODEL_COLORS = {
    "TabPFN": "#e41a1c",
    "CARTE": "#377eb8",
    "TabICL": "#ff7f00",
    "TabDPT": "#4daf4a",
    "Mitra": "#984ea3",
    "HyperFast": "#a65628",
    "Tabula-8B": "#f781bf",
}

DOMAIN_COLORS = {
    "Business & Marketing": "#1b9e77",
    "Finance & Insurance": "#d95f02",
    "Science & Materials": "#7570b3",
    "Healthcare & Life Sciences": "#e7298a",
    "Engineering & Industry": "#66a61e",
    "Technology & Security": "#e6ab02",
}


def _save_fig(fig, output_path: Path):
    """Save figure as PDF + PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def make_selectivity_figure(
    all_selectivity: Dict[str, Dict[str, Dict[str, int]]],
    output_path: Path,
):
    """Feature selectivity stacked bars — one per model."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    models = [m for m in MODEL_ORDER if m in all_selectivity]
    n_models = len(models)

    universal_counts = []
    cluster_counts = []
    specific_counts = []
    for m in models:
        u = c = s = 0
        for band_data in all_selectivity[m].values():
            u += band_data["universal"]
            c += band_data["domain_cluster"]
            s += band_data["domain_specific"]
        universal_counts.append(u)
        cluster_counts.append(c)
        specific_counts.append(s)

    x = np.arange(n_models)
    width = 0.6

    ax.bar(x, universal_counts, width, label='Universal (all 6 domains)',
           color='#2c7bb6', zorder=3)
    ax.bar(x, cluster_counts, width, bottom=universal_counts,
           label='Domain-cluster (2\u20135)', color='#abd9e9', zorder=3)
    bottoms = [u + c for u, c in zip(universal_counts, cluster_counts)]
    ax.bar(x, specific_counts, width, bottom=bottoms,
           label='Domain-specific (0\u20131)', color='#fdae61', zorder=3)

    for i, m in enumerate(models):
        total = universal_counts[i] + cluster_counts[i] + specific_counts[i]
        if total > 0:
            pct = 100 * universal_counts[i] / total
            ax.text(i, total + 15, f'{pct:.0f}%', ha='center', va='bottom',
                    fontsize=7, color='#2c7bb6')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('Alive SAE features', fontsize=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save_fig(fig, output_path)


def make_reconstruction_figure(
    all_r2: Dict[str, Dict[str, Dict[int, float]]],
    output_path: Path,
):
    """Domain reconstruction R² vs Matryoshka scale — faceted by model."""
    # Use only the standard Matryoshka dims shared by all models (32-256).
    # Full-dim R² varies wildly because models have different hidden_dim
    # and TopK sparsity, making cross-model averaging meaningless at full scale.
    scales = [32, 64, 128, 256]
    scale_labels = [str(s) for s in scales]

    domains = sorted(
        d for d in DOMAIN_COLORS.keys()
        if any(d in model_r2 for model_r2 in all_r2.values())
    )

    # Filter to models that exist in the data
    models = [m for m in MODEL_ORDER if m in all_r2]
    n_models = len(models)

    # Layout: 2 rows × 4 cols for up to 7 models
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for mi, model_name in enumerate(models):
        ax = axes[mi]
        model_r2 = all_r2[model_name]

        for domain in domains:
            if domain not in model_r2:
                continue
            vals = [model_r2[domain].get(s, np.nan) for s in scales]
            if any(np.isnan(v) for v in vals):
                continue

            short = DOMAIN_SHORT.get(domain, domain)
            color = DOMAIN_COLORS[domain]
            ax.plot(range(len(scales)), vals, 'o-', color=color, label=short,
                    markersize=3, linewidth=1.2, zorder=3)

        ax.set_title(model_name, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

        # Only show x-labels on bottom row
        if mi >= n_models - ncols:
            ax.set_xticks(range(len(scales)))
            ax.set_xticklabels(scale_labels, fontsize=7)

        # Only show y-labels on leftmost column
        if mi % ncols == 0:
            ax.set_ylabel('Reconstruction R²', fontsize=8)

    # Hide unused subplots
    for mi in range(n_models, len(axes)):
        axes[mi].axis('off')

    # Add shared legend above the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc='upper center',
               bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False)

    # Add shared x-axis label
    fig.text(0.5, -0.02, 'Matryoshka cumulative scale', ha='center', fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print("=" * 60)
    print("Section 4.3: Universal vs. Model-Specific Concepts")
    print("=" * 60)

    # Load domain taxonomy
    dataset_domain = load_domain_taxonomy()
    domains_used = sorted(set(dataset_domain.values()))
    print(f"\nDomains ({len(domains_used)}): {domains_used}")
    print(f"Datasets with domain labels: {len(dataset_domain)}")

    # Resolve model paths
    base_sae = sae_sweep_dir()
    base_emb = PROJECT_ROOT / "output" / "embeddings" / "tabarena"

    model_configs = []
    emb_dirs = {}
    for display_name, sweep_dir, emb_dir_name in DEFAULT_MODELS:
        sae_path = base_sae / sweep_dir / "sae_matryoshka_archetypal_validated.pt"
        emb_dir = base_emb / emb_dir_name
        if not sae_path.exists():
            print(f"  Warning: SAE not found for {display_name}: {sae_path}")
            continue
        if not emb_dir.exists():
            print(f"  Warning: embeddings not found for {display_name}: {emb_dir}")
            continue
        model_configs.append((display_name, sae_path, emb_dir))
        emb_dirs[display_name] = emb_dir

    # Find common datasets across all models
    common_datasets = find_common_datasets(emb_dirs)
    print(f"Common datasets across {len(emb_dirs)} models: {len(common_datasets)}")

    # Filter to datasets that have domain labels (excluding singletons)
    domain_datasets = [ds for ds in common_datasets if ds in dataset_domain]
    excluded = [ds for ds in common_datasets if ds not in dataset_domain]
    print(f"Datasets with domain labels: {len(domain_datasets)} "
          f"(excluded {len(excluded)}: {excluded})")

    # Matryoshka cumulative scales for Layer 0
    cumulative_scales = [32, 64, 128, 256]

    # Results accumulators
    all_selectivity = {}
    all_r2 = {}
    all_taxonomy = {}

    for display_name, sae_path, emb_dir in model_configs:
        print(f"\n--- {display_name} ---")

        # Load SAE
        model, config, _ = load_sae_checkpoint(sae_path)
        print(f"  SAE: {config.input_dim} → {config.hidden_dim} "
              f"(topk={config.topk})")

        # Add full hidden dim to scales if not already there
        scales = [s for s in cumulative_scales if s <= config.hidden_dim]
        if config.hidden_dim not in scales:
            scales.append(config.hidden_dim)

        # Pool embeddings with offset tracking
        pooled, offsets = pool_embeddings_with_offsets(
            emb_dir, domain_datasets, max_per_dataset=500
        )
        print(f"  Pooled: {pooled.shape[0]} samples × {pooled.shape[1]} dims")

        # Compute activations from raw embeddings (BatchNorm handles normalization)
        activations = compute_activations(model, pooled)
        alive = (activations.max(axis=0) > 0.001).sum()
        print(f"  Activations: {activations.shape}, alive={alive}/{activations.shape[1]}")

        # Build domain → row indices
        domain_row_indices = build_domain_row_indices(offsets, dataset_domain)
        for d, idx in sorted(domain_row_indices.items()):
            print(f"    {d}: {len(idx)} rows")

        # Layer 0: Domain reconstruction R² (using raw embeddings, BatchNorm in model handles normalization)
        r2 = compute_domain_reconstruction_r2(
            model, activations, pooled,  # Pass raw embeddings
            domain_row_indices, scales,
        )
        all_r2[display_name] = r2
        for domain in sorted(r2.keys()):
            r2_full = r2[domain][scales[-1]]
            r2_32 = r2[domain][32] if 32 in r2[domain] else float('nan')
            print(f"    R² {DOMAIN_SHORT.get(domain, domain)}: "
                  f"scale=32 → {r2_32:.3f}, full → {r2_full:.3f}")

        # Layer 1: Feature selectivity
        selectivity = compute_feature_selectivity(
            activations, domain_row_indices, config
        )
        all_selectivity[display_name] = selectivity
        total_u = sum(v["universal"] for v in selectivity.values())
        total_c = sum(v["domain_cluster"] for v in selectivity.values())
        total_s = sum(v["domain_specific"] for v in selectivity.values())
        total_alive = total_u + total_c + total_s
        print(f"  Selectivity: {total_u} universal, {total_c} cluster, "
              f"{total_s} specific ({total_alive} alive)")

        # Layer 2: Data-driven domain taxonomy
        taxonomy = compute_domain_taxonomy_agreement(
            activations, offsets, dataset_domain
        )
        all_taxonomy[display_name] = taxonomy
        print(f"  Taxonomy: ARI={taxonomy['ari']:.3f}, NMI={taxonomy['nmi']:.3f}")

    # --- Save results ---
    output_json = PROJECT_ROOT / "output" / "domain_concept_analysis.json"
    results = {
        "metadata": {
            "n_models": len(model_configs),
            "n_domains": len(domains_used),
            "domains": domains_used,
            "n_datasets": len(domain_datasets),
            "excluded_datasets": excluded,
            "cumulative_scales": cumulative_scales,
        },
        "reconstruction_r2": all_r2,
        "feature_selectivity": all_selectivity,
        "domain_taxonomy": all_taxonomy,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {output_json}")

    # --- Generate figures ---
    fig_dir = PROJECT_ROOT / "output" / "paper_figures" / "appendix_d"
    make_selectivity_figure(all_selectivity, fig_dir / "feature_selectivity.pdf")
    make_reconstruction_figure(all_r2, fig_dir / "domain_reconstruction.pdf")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nLayer 2 — Data-driven taxonomy agreement:")
    aris = []
    nmis = []
    for m in MODEL_ORDER:
        if m in all_taxonomy:
            t = all_taxonomy[m]
            aris.append(t["ari"])
            nmis.append(t["nmi"])
            print(f"  {m}: ARI={t['ari']:.3f}, NMI={t['nmi']:.3f}")
    print(f"  Mean: ARI={np.mean(aris):.3f} ± {np.std(aris):.3f}, "
          f"NMI={np.mean(nmis):.3f} ± {np.std(nmis):.3f}")

    print("\nLayer 1 — Universal feature fraction per model:")
    for m in MODEL_ORDER:
        if m in all_selectivity:
            s = all_selectivity[m]
            total_u = sum(v["universal"] for v in s.values())
            total_alive = sum(
                v["universal"] + v["domain_cluster"] + v["domain_specific"]
                for v in s.values()
            )
            pct = 100 * total_u / total_alive if total_alive > 0 else 0
            print(f"  {m}: {total_u}/{total_alive} ({pct:.1f}%) universal")


if __name__ == '__main__':
    main()
