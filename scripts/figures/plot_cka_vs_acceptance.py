#!/usr/bin/env python3
"""Plot CKA vs transfer concept acceptance rate.

Tests whether geometric similarity (CKA) predicts functional
transferability (fraction of concepts accepted by the weak model).
A positive correlation links the geometric convergence story to
the causal transfer story.

Usage:
    python -m scripts.figures.plot_cka_vs_acceptance
    python -m scripts.figures.plot_cka_vs_acceptance --transfer-dir output/transfer_guided
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

from scripts._project_root import PROJECT_ROOT

# CKA values from the updated 6-model table (38 cls datasets)
CKA = {
    ("carte", "mitra"): 0.54,
    ("carte", "tabdpt"): 0.51,
    ("carte", "tabicl"): 0.51,
    ("carte", "tabicl_v2"): 0.54,
    ("carte", "tabpfn"): 0.35,
    ("mitra", "tabdpt"): 0.78,
    ("mitra", "tabicl"): 0.76,
    ("mitra", "tabicl_v2"): 0.79,
    ("mitra", "tabpfn"): 0.55,
    ("tabdpt", "tabicl"): 0.82,
    ("tabdpt", "tabicl_v2"): 0.78,
    ("tabdpt", "tabpfn"): 0.56,
    ("tabicl", "tabicl_v2"): 0.78,
    ("tabicl", "tabpfn"): 0.54,
    ("tabicl_v2", "tabpfn"): 0.45,
}

DISPLAY = {
    "tabpfn": "TabPFN", "mitra": "Mitra", "tabicl": "TabICL",
    "tabicl_v2": "TabICL-v2", "tabdpt": "TabDPT", "carte": "CARTE",
}

DEFAULT_TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_sweep_v2"
OUTPUT_PATH = PROJECT_ROOT / "output" / "figures" / "cka_vs_acceptance.pdf"


def get_pair_stats(transfer_dir):
    """Compute mean acceptance rate and gc per pair."""
    results = {}
    for pair_dir in sorted(transfer_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        accept_rates = []
        gap_closeds = []
        for f in pair_dir.glob("*.npz"):
            d = np.load(f, allow_pickle=True)
            if "acceptance_rate" in d.files:
                accept_rates.append(float(d["acceptance_rate"]))
            if "strong_wins" in d.files:
                sw = d["strong_wins"]
                if sw.sum() > 0:
                    gc = d["gap_closed"][sw]
                    gc = gc[~np.isnan(gc)]
                    if len(gc) > 0:
                        gap_closeds.append(float(gc.mean()))
        if accept_rates:
            results[pair_dir.name] = {
                "acceptance": np.mean(accept_rates),
                "gc": np.mean(gap_closeds) if gap_closeds else 0.0,
                "n_datasets": len(accept_rates),
            }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot CKA vs transfer acceptance rate")
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pair_stats = get_pair_stats(args.transfer_dir)

    # Match CKA values to pair stats
    cka_vals, acc_vals, gc_vals, labels = [], [], [], []
    for pair_name, stats in pair_stats.items():
        parts = pair_name.split("_vs_")
        if len(parts) != 2:
            continue
        a, b = parts
        key = tuple(sorted([a, b]))
        cka = CKA.get(key)
        if cka is None:
            continue
        cka_vals.append(cka)
        acc_vals.append(stats["acceptance"])
        gc_vals.append(stats["gc"])
        labels.append(f"{DISPLAY.get(a, a)}\nvs\n{DISPLAY.get(b, b)}")

    cka_vals = np.array(cka_vals)
    acc_vals = np.array(acc_vals)
    gc_vals = np.array(gc_vals)

    rho_acc, p_acc = spearmanr(cka_vals, acc_vals)
    rho_gc, p_gc = spearmanr(cka_vals, gc_vals)

    # Two-panel figure: CKA vs acceptance, CKA vs gc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: CKA vs Acceptance Rate
    ax1.scatter(cka_vals, acc_vals * 100, s=40, c="#2c7bb6", edgecolors="white",
                linewidths=0.5, zorder=3)

    # Add labels
    for i, label in enumerate(labels):
        short = label.replace("\nvs\n", "–")
        ax1.annotate(short, (cka_vals[i], acc_vals[i] * 100),
                     fontsize=5, ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points")

    # Trend line
    z = np.polyfit(cka_vals, acc_vals * 100, 1)
    x_line = np.linspace(cka_vals.min() - 0.05, cka_vals.max() + 0.05, 100)
    ax1.plot(x_line, np.polyval(z, x_line), "k--", lw=0.8, alpha=0.5)

    ax1.set_xlabel("CKA Similarity", fontsize=10)
    ax1.set_ylabel("Concept Acceptance Rate (%)", fontsize=10)
    ax1.set_title("Geometric Similarity vs Transferability", fontsize=11,
                  fontweight="bold")
    ax1.set_xlim(0.25, 0.90)
    ax1.set_ylim(55, 105)
    ax1.text(0.05, 0.05,
             f"Spearman ρ = {rho_acc:.2f}\np = {p_acc:.3f}",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel 2: CKA vs Gap Closed
    ax2.scatter(cka_vals, gc_vals, s=40, c="#d7191c", edgecolors="white",
                linewidths=0.5, zorder=3)

    for i, label in enumerate(labels):
        short = label.replace("\nvs\n", "–")
        ax2.annotate(short, (cka_vals[i], gc_vals[i]),
                     fontsize=5, ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points")

    z2 = np.polyfit(cka_vals, gc_vals, 1)
    ax2.plot(x_line, np.polyval(z2, x_line), "k--", lw=0.8, alpha=0.5)

    ax2.set_xlabel("CKA Similarity", fontsize=10)
    ax2.set_ylabel("Transfer Gap Closed", fontsize=10)
    ax2.set_title("Geometric Similarity vs Transfer Efficacy", fontsize=11,
                  fontweight="bold")
    ax2.set_xlim(0.25, 0.90)
    ax2.set_ylim(0.2, 1.05)
    ax2.text(0.05, 0.05,
             f"Spearman ρ = {rho_gc:.2f}\np = {p_gc:.3f}",
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("CKA Predicts Cross-Model Concept Transfer",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    output = args.output or OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")
    print(f"\nCKA vs Acceptance: ρ={rho_acc:.3f}, p={p_acc:.3f}")
    print(f"CKA vs Gap Closed: ρ={rho_gc:.3f}, p={p_gc:.3f}")

    # Print table
    print(f"\n{'Pair':<30} {'CKA':>6} {'Accept%':>8} {'GC':>6}")
    print("-" * 54)
    for i in np.argsort(-acc_vals):
        short = labels[i].replace("\nvs\n", " vs ")
        print(f"{short:<30} {cka_vals[i]:>6.2f} {acc_vals[i]*100:>7.1f}% {gc_vals[i]:>6.2f}")


if __name__ == "__main__":
    main()
