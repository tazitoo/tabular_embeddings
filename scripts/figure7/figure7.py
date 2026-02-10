#!/usr/bin/env python3
"""
Figure 7: CKA vs SAE concept overlap scatter.

The punchline figure — do geometrically similar models also share concepts?
Each point is a model pair. x-axis = CKA, y-axis = Jaccard.
S1 (coarse) and S5 (fine) shown together: coarse concepts track geometry,
fine concepts don't.

Usage:
    python scripts/figure7/figure7.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Data: model pairs with CKA (from Table 1, 36 common datasets) and Jaccard per band (from Figure 1)
PAIRS = {
    # Transformer cluster (TabPFN, TabICL, TabDPT, Mitra)
    'TabPFN–TabDPT':  {'cka': 0.67, 's1': 0.524, 's2': 0.148, 's3': 0.055, 's4': 0.100, 's5': 0.074},
    'TabPFN–TabICL':  {'cka': 0.66, 's1': 0.360, 's2': 0.042, 's3': 0.021, 's4': 0.071, 's5': 0.066},
    'TabPFN–Mitra':   {'cka': 0.68, 's1': 0.391, 's2': 0.179, 's3': 0.091, 's4': 0.159, 's5': 0.087},
    'TabDPT–TabICL':  {'cka': 0.61, 's1': 0.391, 's2': 0.103, 's3': 0.117, 's4': 0.070, 's5': 0.077},
    'TabDPT–Mitra':   {'cka': 0.83, 's1': 0.429, 's2': 0.212, 's3': 0.100, 's4': 0.125, 's5': 0.078},
    'TabICL–Mitra':   {'cka': 0.62, 's1': 0.333, 's2': 0.214, 's3': 0.140, 's4': 0.102, 's5': 0.067},
    # CARTE pairs (GNN outlier)
    'TabPFN–CARTE':   {'cka': 0.43, 's1': 0.250, 's2': 0.056, 's3': 0.000, 's4': 0.059, 's5': 0.060},
    'TabDPT–CARTE':   {'cka': 0.52, 's1': 0.278, 's2': 0.040, 's3': 0.021, 's4': 0.061, 's5': 0.086},
    'TabICL–CARTE':   {'cka': 0.39, 's1': 0.190, 's2': 0.111, 's3': 0.054, 's4': 0.085, 's5': 0.059},
    'Mitra–CARTE':    {'cka': 0.53, 's1': 0.353, 's2': 0.077, 's3': 0.056, 's4': 0.147, 's5': 0.090},
    # HyperFast pairs (hypernetwork outlier)
    'TabPFN–HyperFast':  {'cka': 0.27, 's1': 0.200, 's2': 0.083, 's3': 0.025, 's4': 0.029, 's5': 0.048},
    'TabDPT–HyperFast':  {'cka': 0.22, 's1': 0.308, 's2': 0.222, 's3': 0.054, 's4': 0.072, 's5': 0.079},
    'TabICL–HyperFast':  {'cka': 0.24, 's1': 0.200, 's2': 0.038, 's3': 0.042, 's4': 0.086, 's5': 0.057},
    'Mitra–HyperFast':   {'cka': 0.21, 's1': 0.308, 's2': 0.129, 's3': 0.195, 's4': 0.149, 's5': 0.060},
    'CARTE–HyperFast':   {'cka': 0.13, 's1': 0.286, 's2': 0.105, 's3': 0.069, 's4': 0.122, 's5': 0.048},
}

TRANSFORMER_PAIRS = {
    'TabPFN–TabDPT', 'TabPFN–TabICL', 'TabPFN–Mitra',
    'TabDPT–TabICL', 'TabDPT–Mitra', 'TabICL–Mitra',
}
CARTE_PAIRS = {'TabPFN–CARTE', 'TabDPT–CARTE', 'TabICL–CARTE', 'Mitra–CARTE'}
HYPERFAST_PAIRS = {
    'TabPFN–HyperFast', 'TabDPT–HyperFast', 'TabICL–HyperFast',
    'Mitra–HyperFast', 'CARTE–HyperFast',
}


def _pair_style(label):
    """Return (color, marker) for a model pair."""
    if label in TRANSFORMER_PAIRS:
        return '#e41a1c', 'o'
    elif label in CARTE_PAIRS:
        return '#377eb8', 's'
    else:  # HyperFast
        return '#4daf4a', 'D'


def make_figure(output_path: Path):
    fig, (ax, ax_lower) = plt.subplots(2, 1, figsize=(5.5, 5.5),
                                        height_ratios=[3, 1.2],
                                        gridspec_kw={'hspace': 0.4})

    # Plot S5 first (behind, faded)
    for label, vals in PAIRS.items():
        color, marker = _pair_style(label)
        ax.scatter(vals['cka'], vals['s5'],
                   c=color, marker=marker, s=40, zorder=2, alpha=0.3,
                   edgecolors='none')

    # Plot S1 (foreground, solid)
    for label, vals in PAIRS.items():
        color, marker = _pair_style(label)
        ax.scatter(vals['cka'], vals['s1'],
                   c=color, marker=marker, s=70, zorder=3, edgecolors='white',
                   linewidth=0.8)

        # Connect S1 to S5 with a thin line
        ax.plot([vals['cka'], vals['cka']], [vals['s1'], vals['s5']],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

        # Label only a subset to avoid clutter (highest/lowest CKA per group)
        show_label = label in {
            'TabDPT–Mitra', 'TabPFN–TabDPT', 'TabPFN–TabICL',
            'TabPFN–CARTE', 'Mitra–CARTE',
            'TabPFN–HyperFast', 'CARTE–HyperFast',
        }
        if not show_label:
            continue

        ha, va, dx, dy = 'left', 'bottom', 0.015, 0.015
        if label == 'TabPFN–TabICL':
            ha, va, dx, dy = 'left', 'top', 0.015, -0.015
        elif label == 'TabPFN–CARTE':
            ha, va, dx, dy = 'right', 'bottom', -0.015, 0.015
        elif label == 'Mitra–CARTE':
            ha, va, dx, dy = 'left', 'top', 0.015, -0.015
        elif label == 'CARTE–HyperFast':
            ha, va, dx, dy = 'right', 'top', -0.015, -0.015

        ax.annotate(label, (vals['cka'] + dx, vals['s1'] + dy),
                    fontsize=6.5, ha=ha, va=va, color=color)

    # Trend lines
    x = np.array([v['cka'] for v in PAIRS.values()])
    y_s1 = np.array([v['s1'] for v in PAIRS.values()])
    y_s5 = np.array([v['s5'] for v in PAIRS.values()])
    x_fit = np.linspace(0.05, 0.90, 100)

    # S1 trend
    coeffs_s1 = np.polyfit(x, y_s1, 1)
    r_s1 = np.corrcoef(x, y_s1)[0, 1]
    ax.plot(x_fit, np.polyval(coeffs_s1, x_fit), '--', color='gray', alpha=0.5,
            linewidth=1, zorder=1)

    # Lower panel: r and mean Jaccard by band
    bands = ['s1', 's2', 's3', 's4', 's5']
    band_labels = ['S1\n[0,32)', 'S2\n[32,64)', 'S3\n[64,128)', 'S4\n[128,256)', 'S5\n[256,N)']
    r_vals = []
    mean_j = []
    for b in bands:
        y_b = np.array([v[b] for v in PAIRS.values()])
        r_vals.append(np.corrcoef(x, y_b)[0, 1])
        mean_j.append(y_b.mean())

    ax2 = ax_lower.twinx()
    ax_lower.plot(range(5), r_vals, 'o-', color='#333333', markersize=5,
                  linewidth=1.5, label='r (CKA vs Jaccard)', zorder=3)
    ax2.bar(range(5), mean_j, 0.5, color='#cccccc', alpha=0.6,
            label='Mean Jaccard', zorder=1)

    ax_lower.set_xticks(range(5))
    ax_lower.set_xticklabels(band_labels, fontsize=7)
    ax_lower.set_ylabel('Correlation (r)', fontsize=9)
    ax_lower.set_ylim(-0.1, 1.1)
    ax_lower.set_yticks([0, 0.5, 1.0])
    ax2.set_ylabel('Mean Jaccard', fontsize=9, color='#888888')
    ax2.set_ylim(0, 0.40)
    ax2.tick_params(labelsize=7, colors='#888888')
    ax_lower.tick_params(labelsize=7)
    ax_lower.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax_lower.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_lower.legend(lines1 + lines2, labels1 + labels2,
                    fontsize=7, loc='upper right')

    # Legend
    ax.scatter([], [], c='#e41a1c', marker='o', s=50, label='Transformer pairs')
    ax.scatter([], [], c='#377eb8', marker='s', s=50, label='CARTE pairs')
    ax.scatter([], [], c='#4daf4a', marker='D', s=50, label='HyperFast pairs')
    ax.scatter([], [], c='gray', marker='o', s=30, alpha=0.3, label='S5 (fine)')
    ax.scatter([], [], c='gray', marker='o', s=50, label='S1 (coarse)')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    ax.set_xlabel('CKA Similarity', fontsize=10)
    ax.set_ylabel('Concept Overlap (Jaccard)', fontsize=10)
    ax.set_xlim(0.05, 0.90)
    ax.set_ylim(-0.02, 0.58)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.14, right=0.88, top=0.97, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


if __name__ == '__main__':
    output = Path('scripts/figure7/geometric_vs_concept.pdf')
    make_figure(output)
