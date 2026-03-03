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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    sae_sweep_dir,
)
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint

# Map display names to prebuilt training data base names
_MODEL_BASE_NAMES = {
    'TabPFN': 'tabpfn', 'CARTE': 'carte', 'TabICL': 'tabicl',
    'TabDPT': 'tabdpt', 'Mitra': 'mitra', 'HyperFast': 'hyperfast',
    'Tabula-8B': 'tabula8b',
}

PREBUILT_DIR = PROJECT_ROOT / "output" / "sae_training_round5"

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
    """Pool embeddings from per-dataset NPZ files and track row offsets.

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


def load_prebuilt_sae_data(
    display_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
    """Load prebuilt SAE training/test data with known provenance.

    The SAE was trained on centered data: X_centered = X - X.mean(dim=0).
    We load that same training data to compute the exact centering mean,
    and use the test split (same extraction pipeline) for evaluation.

    Returns:
        test_embeddings: (n_test, dim) raw test embeddings
        centering_mean: (dim,) mean of training data (for centering)
        train_embeddings: (n_train, dim) raw training embeddings
        test_offsets: {dataset_name: (start, end)} per-dataset row offsets in test
        source_datasets: list of dataset names
    """
    base = _MODEL_BASE_NAMES.get(display_name, display_name.lower())
    candidates = sorted(PREBUILT_DIR.glob(f"{base}_layer*_sae_training.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No prebuilt training data for {display_name} in {PREBUILT_DIR}"
        )

    # Load training data → centering mean
    train_data = np.load(candidates[0], allow_pickle=True)
    train_emb = train_data['embeddings'].astype(np.float32)
    centering_mean = train_emb.mean(axis=0)
    source_datasets = list(train_data['source_datasets'])

    # Load test data → evaluation embeddings with per-dataset offsets
    test_path = Path(str(candidates[0]).replace('_sae_training.npz', '_sae_test.npz'))
    if not test_path.exists():
        raise FileNotFoundError(f"No test data: {test_path}")

    test_data = np.load(test_path, allow_pickle=True)
    test_emb = test_data['embeddings'].astype(np.float32)
    samples_per_dataset = test_data['samples_per_dataset']

    # Reconstruct per-dataset offsets from structured array
    test_offsets = {}
    cursor = 0
    for entry in samples_per_dataset:
        ds_name = str(entry['dataset'])
        count = int(entry['count'])
        test_offsets[ds_name] = (cursor, cursor + count)
        cursor += count

    return test_emb, centering_mean, train_emb, test_offsets, source_datasets


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

def compute_domain_reconstruction_fve(
    model, pooled_raw: np.ndarray,
    domain_row_indices: Dict[str, np.ndarray],
    scales: List[int],
    centering_mean: np.ndarray,
) -> Dict[str, Dict[int, float]]:
    """
    Cumulative fraction of variance explained per domain per Matryoshka scale.

    FVE(scale) = (MSE_null - MSE_scale) / (MSE_null - MSE_full)

    where MSE_null is the error from predicting the domain mean (no SAE),
    MSE_full is the error using all features, and MSE_scale uses only the
    first `scale` features. Goes from ~0 (S1) to 1.0 (full) monotonically.

    Args:
        centering_mean: Mean of the SAE training data. The SAE was trained on
            X_centered = X - X.mean(), so we must center eval data identically.
    """
    model.eval()
    mean_t = torch.tensor(centering_mean, dtype=torch.float32)

    results = {}
    for domain, indices in domain_row_indices.items():
        results[domain] = {}
        with torch.no_grad():
            x = torch.tensor(pooled_raw[indices], dtype=torch.float32) - mean_t
            h = model.encode(x)

            # Null model: predict domain mean
            mse_null = ((x - x.mean(dim=0)) ** 2).mean().item()

            # Full reconstruction
            x_hat_full = model.decode(h)
            mse_full = ((x - x_hat_full) ** 2).mean().item()

            denom = mse_null - mse_full
            if denom <= 0:
                for scale in scales:
                    results[domain][scale] = 0.0
                continue

            for scale in scales:
                h_trunc = h.clone()
                h_trunc[:, scale:] = 0.0
                x_hat = model.decode(h_trunc)
                mse_scale = ((x - x_hat) ** 2).mean().item()
                results[domain][scale] = float((mse_null - mse_scale) / denom)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for mi, model_name in enumerate(models):
        ax = axes[mi]
        model_r2 = all_r2[model_name]

        # Derive per-model scales from the data
        first_domain_r2 = next(iter(model_r2.values()))
        model_scales = sorted(first_domain_r2.keys())
        model_scale_labels = [str(s) for s in model_scales]

        for domain in domains:
            if domain not in model_r2:
                continue
            vals = [model_r2[domain].get(s, np.nan) for s in model_scales]
            if any(np.isnan(v) for v in vals):
                continue

            short = DOMAIN_SHORT.get(domain, domain)
            color = DOMAIN_COLORS[domain]
            ax.plot(range(len(model_scales)), vals, 'o-', color=color, label=short,
                    markersize=3, linewidth=1.2, zorder=3)

        ax.set_title(model_name, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(model_scales)))
        ax.set_xticklabels(model_scale_labels, fontsize=6, rotation=45)

        # Only show y-labels on leftmost column
        if mi % ncols == 0:
            ax.set_ylabel('Fraction of variance explained', fontsize=8)

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


def compute_centered_activations(
    model, embeddings: np.ndarray, centering_mean: np.ndarray,
) -> np.ndarray:
    """Compute SAE activations on centered embeddings.

    Centers with the training mean to match the distribution the SAE
    was trained on (train_sae centers with X.mean before fitting).
    """
    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32)
        x = x - torch.tensor(centering_mean, dtype=torch.float32)
        h = model.encode(x).numpy()
    return h


def main():
    print("=" * 60)
    print("Section 4.3: Universal vs. Model-Specific Concepts")
    print("=" * 60)

    # Load domain taxonomy
    dataset_domain = load_domain_taxonomy()
    domains_used = sorted(set(dataset_domain.values()))
    print(f"\nDomains ({len(domains_used)}): {domains_used}")
    print(f"Datasets with domain labels: {len(dataset_domain)}")

    # Resolve model paths (SAE checkpoints only — embeddings come from prebuilt data)
    base_sae = sae_sweep_dir()

    model_configs = []
    for display_name, sweep_dir, _ in DEFAULT_MODELS:
        sae_path = base_sae / sweep_dir / "sae_matryoshka_archetypal_validated.pt"
        if not sae_path.exists():
            print(f"  Warning: SAE not found for {display_name}: {sae_path}")
            continue
        # Verify prebuilt data exists
        base = _MODEL_BASE_NAMES.get(display_name, display_name.lower())
        if not list(PREBUILT_DIR.glob(f"{base}_layer*_sae_training.npz")):
            print(f"  Warning: no prebuilt data for {display_name}")
            continue
        model_configs.append((display_name, sae_path))

    print(f"Models: {[m[0] for m in model_configs]}")

    # Matryoshka cumulative scales: derived per-model from config below
    cumulative_scales = None  # set from first model's config

    # Results accumulators
    all_selectivity = {}
    all_r2 = {}
    all_taxonomy = {}

    for display_name, sae_path in model_configs:
        print(f"\n--- {display_name} ---")

        # Load SAE
        model, config, _ = load_sae_checkpoint(sae_path)
        print(f"  SAE: {config.input_dim} → {config.hidden_dim} "
              f"(topk={config.topk})")

        # Derive cumulative scales from config matryoshka_dims
        mat_dims = getattr(config, 'matryoshka_dims', None)
        if mat_dims:
            scales = sorted([d for d in mat_dims if d <= config.hidden_dim])
        else:
            scales = [config.hidden_dim]
        if config.hidden_dim not in scales:
            scales.append(config.hidden_dim)
        if cumulative_scales is None:
            cumulative_scales = scales
            print(f"  Matryoshka scales: {scales}")

        # Load prebuilt test data (same extraction pipeline as SAE training)
        test_emb, centering_mean, _, offsets, source_ds = load_prebuilt_sae_data(
            display_name
        )
        print(f"  Test data: {test_emb.shape[0]} samples × {test_emb.shape[1]} dims")
        print(f"  Source datasets: {len(source_ds)}")

        # Filter offsets to datasets with domain labels
        domain_offsets = {
            ds: span for ds, span in offsets.items() if ds in dataset_domain
        }
        excluded = [ds for ds in offsets if ds not in dataset_domain]
        if excluded:
            print(f"  Excluded (no domain label): {excluded}")

        # Compute activations with training centering
        activations = compute_centered_activations(model, test_emb, centering_mean)
        alive = (activations.max(axis=0) > 0.001).sum()
        print(f"  Activations: {activations.shape}, alive={alive}/{activations.shape[1]}")

        # Build domain → row indices
        domain_row_indices = build_domain_row_indices(domain_offsets, dataset_domain)
        for d, idx in sorted(domain_row_indices.items()):
            print(f"    {d}: {len(idx)} rows")

        # Layer 0: Domain reconstruction FVE
        r2 = compute_domain_reconstruction_fve(
            model, test_emb, domain_row_indices, scales, centering_mean,
        )
        all_r2[display_name] = r2
        for domain in sorted(r2.keys()):
            fve_s1 = r2[domain][scales[0]]
            fve_full = r2[domain][scales[-1]]
            print(f"    FVE {DOMAIN_SHORT.get(domain, domain)}: "
                  f"S1={fve_s1:.3f}, full={fve_full:.3f}")

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
            activations, domain_offsets, dataset_domain
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
            "data_source": "output/sae_training_round5 (prebuilt test split)",
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
    fig_dir = PROJECT_ROOT / "scripts" / "section43"
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
