#!/usr/bin/env python3
"""
Figure: Regression explainability of SAE concepts across models and bands.

Panel A: Mean R² per Matryoshka band per model (grouped bars).
         Expectation: S1 (coarse) high, S5 (fine) low.
Panel B: Scatter of max-single-|d| vs R² per feature.
         Quadrants: top-right = well-covered, top-left = interpolated,
         bottom-left = unexplained. Colored by band.

Usage:
    python scripts/paper/appendix_c/fig_concept_regression.py \
        --output output/paper_figures/concept_regression.pdf

    # From pre-computed JSON (faster):
    python scripts/paper/appendix_c/fig_concept_regression.py \
        --from-json output/concept_regression.json \
        --output output/paper_figures/concept_regression.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_architectures import (
    META_NAMES,
    compute_activations,
    compute_basic_metrics,
    compute_feature_effects,
    get_train_test_split,
    meta_features_to_array,
)
from scripts.compare_sae_cross_model import (
    DEFAULT_MODELS,
    collect_meta_for_datasets,
    find_common_datasets,
    pool_embeddings_for_datasets,
    sae_sweep_dir,
)
from scripts.analyze_sae_concepts_deep import load_sae_checkpoint
from scripts.analyze_concept_regression import (
    compute_band_regression_summary,
    identify_interpolated_concepts,
    regress_features_on_probes,
)

# Colors for models (consistent with other figures)
MODEL_COLORS = {
    'TabPFN': '#1f77b4',
    'CARTE': '#ff7f0e',
    'TabICL': '#2ca02c',
    'TabDPT': '#d62728',
    'Mitra': '#9467bd',
    'HyperFast': '#8c564b',
    'Tabula-8B': '#e377c2',
}

BAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
BAND_LABELS_SHORT = ['S1', 'S2', 'S3', 'S4', 'S5']


def compute_from_models(models, max_per_dataset=500, alpha=1.0):
    """Compute regression data from scratch for all models."""
    emb_base = PROJECT_ROOT / "output" / "embeddings" / "tabarena"
    sweep = sae_sweep_dir()

    # Find common datasets
    emb_dirs = {}
    for name, _, emb_model in models:
        d = emb_base / emb_model
        if d.exists():
            emb_dirs[name] = d
    datasets = find_common_datasets(emb_dirs)

    # Meta-features
    first_dir = list(emb_dirs.values())[0]
    meta_array, loaded, _boundaries = collect_meta_for_datasets(datasets, first_dir, max_per_dataset)

    results = {}
    for display_name, sae_dir, emb_model in models:
        sae_path = sweep / sae_dir / "sae_matryoshka_archetypal_validated.pt"
        emb_dir = emb_base / emb_model
        if not sae_path.exists():
            continue

        model, config, _ = load_sae_checkpoint(sae_path)
        pooled = pool_embeddings_for_datasets(emb_dir, datasets, max_per_dataset)

        # Normalize
        train_ds, _ = get_train_test_split(datasets)
        train_embs = []
        for ds in train_ds:
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

        acts = compute_activations(model, pooled, train_std, train_mean)
        n_min = min(len(meta_array), len(acts))
        acts = acts[:n_min]
        meta = meta_array[:n_min]

        metrics = compute_basic_metrics(acts, config)
        alive = metrics['alive_indices']
        feat_effects = compute_feature_effects(acts, meta, alive)
        reg_results = regress_features_on_probes(acts, meta, alive, alpha)
        band_summary = compute_band_regression_summary(reg_results, config)

        # Collect per-feature (r2, max_d, band_idx) for scatter
        mat_dims = getattr(config, 'matryoshka_dims', None) or [config.hidden_dim]
        band_edges = [0] + list(mat_dims)
        per_feature = []
        for fid, r in reg_results.items():
            # Determine band
            band_idx = 0
            for bi in range(len(band_edges) - 1):
                if band_edges[bi] <= fid < band_edges[bi + 1]:
                    band_idx = bi
                    break

            effects = feat_effects.get(fid, {}).get('effect_sizes', {})
            max_d = max(abs(d) for d in effects.values()) if effects else 0.0
            per_feature.append({
                'feat_idx': int(fid),
                'r2': r['r2'],
                'max_d': float(max_d),
                'band_idx': band_idx,
            })

        results[display_name] = {
            'band_summary': band_summary,
            'per_feature': per_feature,
        }

    return results


def load_from_json(path):
    """Load pre-computed results from JSON."""
    with open(path) as f:
        data = json.load(f)

    results = {}
    for model_name, model_data in data.items():
        band_summary = model_data.get('band_summary', {})
        per_feature_raw = model_data.get('per_feature', {})

        per_feature = []
        for fid_str, fdata in per_feature_raw.items():
            per_feature.append({
                'feat_idx': int(fid_str),
                'r2': fdata['r2'],
                'max_d': 0.0,  # Not stored per-feature in summary JSON
                'band_idx': 0,
            })

        results[model_name] = {
            'band_summary': band_summary,
            'per_feature': per_feature,
        }

    return results


def plot_figure(results, output_path):
    """Create the two-panel figure."""
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    # --- Panel A: Mean R² per band per model ---
    ax_a = fig.add_subplot(gs[0])

    model_names = list(results.keys())
    n_models = len(model_names)

    # Collect band data
    all_bands = set()
    for res in results.values():
        all_bands.update(res['band_summary'].keys())
    band_labels = sorted(all_bands)
    n_bands = len(band_labels)

    if n_bands == 0:
        ax_a.text(0.5, 0.5, 'No band data', ha='center', va='center',
                  transform=ax_a.transAxes)
    else:
        bar_width = 0.8 / n_models
        x = np.arange(n_bands)

        for mi, model_name in enumerate(model_names):
            r2_vals = []
            for bl in band_labels:
                bs = results[model_name]['band_summary'].get(bl, {})
                r2_vals.append(bs.get('mean_r2', 0.0))

            color = MODEL_COLORS.get(model_name, f'C{mi}')
            ax_a.bar(x + mi * bar_width, r2_vals, bar_width,
                     label=model_name, color=color, alpha=0.85)

        # Short labels
        short_labels = []
        for bl in band_labels:
            if bl.startswith('S'):
                short_labels.append(bl.split(' ')[0])
            else:
                short_labels.append(bl)

        ax_a.set_xticks(x + bar_width * (n_models - 1) / 2)
        ax_a.set_xticklabels(short_labels)
        ax_a.set_ylabel('Mean R² (Ridge)')
        ax_a.set_xlabel('Matryoshka Scale Band')
        ax_a.set_title('A. Probe Explainability by Band', fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=7, ncol=2, loc='upper right')
        ax_a.set_ylim(0, 1.0)
        ax_a.axhline(0.3, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_a.text(ax_a.get_xlim()[1] * 0.98, 0.31, 'R²=0.3', ha='right',
                  fontsize=7, color='gray')

    # --- Panel B: Scatter max-|d| vs R² (first model with per_feature data) ---
    ax_b = fig.add_subplot(gs[1])

    # Use first model that has full per_feature data with max_d
    scatter_model = None
    for name, res in results.items():
        pf = res['per_feature']
        if pf and any(p.get('max_d', 0) > 0 for p in pf):
            scatter_model = name
            break

    if scatter_model is None:
        # Fall back to any model
        scatter_model = model_names[0] if model_names else None

    if scatter_model and results[scatter_model]['per_feature']:
        pf = results[scatter_model]['per_feature']
        max_ds = [p['max_d'] for p in pf]
        r2s = [p['r2'] for p in pf]
        bands = [p['band_idx'] for p in pf]

        for bi in range(max(bands) + 1):
            mask = [b == bi for b in bands]
            x_bi = [max_ds[i] for i in range(len(mask)) if mask[i]]
            y_bi = [r2s[i] for i in range(len(mask)) if mask[i]]
            label = BAND_LABELS_SHORT[bi] if bi < len(BAND_LABELS_SHORT) else f'S{bi+1}'
            color = BAND_COLORS[bi] if bi < len(BAND_COLORS) else f'C{bi}'
            ax_b.scatter(x_bi, y_bi, s=8, alpha=0.4, label=label, color=color)

        # Quadrant lines
        ax_b.axhline(0.3, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_b.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # Quadrant labels
        ax_b.text(0.05, 0.95, 'Interpolated', transform=ax_b.transAxes,
                  fontsize=8, color='#666', ha='left', va='top')
        ax_b.text(0.95, 0.95, 'Well-covered', transform=ax_b.transAxes,
                  fontsize=8, color='#666', ha='right', va='top')
        ax_b.text(0.05, 0.05, 'Unexplained', transform=ax_b.transAxes,
                  fontsize=8, color='#666', ha='left', va='bottom')
        ax_b.text(0.95, 0.05, 'Single-probe', transform=ax_b.transAxes,
                  fontsize=8, color='#666', ha='right', va='bottom')

        ax_b.set_xlabel('Max single-probe |Cohen\'s d|')
        ax_b.set_ylabel('Ridge R²')
        ax_b.set_title(f'B. Feature Explainability ({scatter_model})',
                       fontsize=12, fontweight='bold')
        ax_b.legend(fontsize=8, markerscale=2)
        ax_b.set_xlim(-0.1, max(max_ds) * 1.05 if max_ds else 5)
        ax_b.set_ylim(-0.05, 1.05)
    else:
        ax_b.text(0.5, 0.5, 'No per-feature data', ha='center', va='center',
                  transform=ax_b.transAxes)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Concept regression figure")
    parser.add_argument("--output", type=str,
                        default="output/paper_figures/concept_regression.pdf")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Load pre-computed JSON instead of recomputing")
    parser.add_argument("--models", nargs='+', default=None)
    parser.add_argument("--max-per-dataset", type=int, default=500)
    args = parser.parse_args()

    if args.from_json:
        results = load_from_json(args.from_json)
    else:
        models = DEFAULT_MODELS
        if args.models:
            models = [(n, s, e) for n, s, e in DEFAULT_MODELS if n in args.models]
        results = compute_from_models(models, args.max_per_dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_figure(results, output_path)


if __name__ == "__main__":
    main()
