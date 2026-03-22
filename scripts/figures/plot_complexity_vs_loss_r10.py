#!/usr/bin/env python3
"""Plot complexity (hidden_dim) vs loss for round 10 SAE sweeps.

Reads trial data from Optuna SQLite DBs. Shows test reconstruction loss
(used for model selection) colored by alive%, with annotations for
best recon, best alive, and best efficiency configs.

Usage:
    python scripts/figures/plot_complexity_vs_loss_r10.py
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._project_root import PROJECT_ROOT

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("optuna required: pip install optuna")

SWEEP_DIR = PROJECT_ROOT / "output" / "sae_tabarena_sweep_round10"

EMBED_DIMS = {
    "tabpfn": 192, "tabicl": 512, "tabicl_v2": 512, "mitra": 512,
    "carte": 150, "hyperfast": 784, "tabdpt": 768, "tabula8b": 4096,
}

DISPLAY_NAMES = {
    "tabpfn": "TabPFN", "tabicl": "TabICL", "tabicl_v2": "TabICL v2",
    "mitra": "Mitra", "carte": "CARTE", "hyperfast": "HyperFast",
    "tabdpt": "TabDPT", "tabula8b": "Tabula-8B",
}

COLORS = {
    "TabPFN": "#1f77b4", "TabICL": "#ff7f0e", "TabICL v2": "#bcbd22",
    "Mitra": "#2ca02c", "CARTE": "#d62728", "HyperFast": "#9467bd",
    "TabDPT": "#8c564b", "Tabula-8B": "#e377c2",
}


def load_trials(model_name):
    """Load completed trials from Optuna DB."""
    db_path = SWEEP_DIR / model_name / f"{model_name}_matryoshka_archetypal.db"
    if not db_path.exists():
        return None

    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(
        study_name=f"{model_name}_matryoshka_archetypal", storage=storage,
    )

    trials = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(t.params)
        row["trial"] = t.number
        row["objective"] = t.value
        for k, v in t.user_attrs.items():
            if isinstance(v, (int, float)):
                row[k] = v
        trials.append(row)
    return trials


def main():
    # Discover available models
    available = []
    for model_name in EMBED_DIMS:
        trials = load_trials(model_name)
        if trials:
            available.append((model_name, trials))

    if not available:
        print("No completed sweeps found.")
        return

    nmodels = len(available)
    ncols = min(3, nmodels)
    nrows = (nmodels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nmodels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i in range(nmodels, len(axes)):
        axes[i].set_visible(False)

    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlBu

    for idx, (model_name, trials) in enumerate(available):
        ax = axes[idx]
        ed = EMBED_DIMS[model_name]
        display = DISPLAY_NAMES[model_name]

        complexities = [t["expansion"] * ed for t in trials]
        # Use test recon (model selection metric) if available, else train
        recon = [t.get("test_reconstruction_loss", t["reconstruction_loss"]) for t in trials]
        train_recon = [t["reconstruction_loss"] for t in trials]
        alive_pcts = [t["alive_features"] / (t["expansion"] * ed) * 100 for t in trials]
        l0_vals = [t["l0_sparsity"] for t in trials]

        # Plot test recon loss (filled) and train recon (hollow x)
        sc = ax.scatter(complexities, recon, c=alive_pcts, cmap=cmap, norm=norm,
                        s=80, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3,
                        label="Test recon")
        ax.scatter(complexities, train_recon, c=alive_pcts, cmap=cmap, norm=norm,
                   s=40, alpha=0.4, edgecolors="k", linewidths=0.3, marker="x",
                   zorder=2, label="Train recon")

        ax.set_xlabel(f"Hidden dim (expansion x {ed})")
        ax.set_ylabel("Loss")
        ax.set_title(f"{display} (embed={ed})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        xmax = max(complexities) * 1.1
        ax.set_xlim(-xmax * 0.03, xmax)
        tick_vals = sorted(set(complexities))
        ax.set_xticklabels([f"{v // 1000}K" if v >= 1000 else str(v) for v in tick_vals],
                           fontsize=7, rotation=45)
        ax.set_xticks(tick_vals)

        # Annotations
        annotated = set()

        # 1. Best recon
        best_idx = int(np.argmin(recon))
        t = trials[best_idx]
        ax.annotate(f"best recon\n{t['expansion']}x, k={t.get('topk','?')}, "
                    f"{alive_pcts[best_idx]:.0f}%",
                    xy=(complexities[best_idx], recon[best_idx]),
                    xytext=(15, 15), textcoords="offset points", fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        annotated.add(best_idx)

        # 2. Best alive recon (>85%)
        alive_mask = [a >= 85 for a in alive_pcts]
        if any(alive_mask):
            alive_recons = [r if m else 999 for r, m in zip(recon, alive_mask)]
            best_alive_idx = int(np.argmin(alive_recons))
            if best_alive_idx not in annotated:
                t2 = trials[best_alive_idx]
                ax.annotate(f"best alive\n{t2['expansion']}x, k={t2.get('topk','?')}, "
                            f"{alive_pcts[best_alive_idx]:.0f}%",
                            xy=(complexities[best_alive_idx], recon[best_alive_idx]),
                            xytext=(-60, 25), textcoords="offset points", fontsize=7,
                            arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="green",
                                      alpha=0.8))
                annotated.add(best_alive_idx)

        # 3. Best efficiency: recon * sqrt(hidden) * sqrt(L0) / alive_frac
        efficiency = []
        for r, c, a, l0 in zip(recon, complexities, alive_pcts, l0_vals):
            if a >= 80:
                efficiency.append(r * np.sqrt(c) * np.sqrt(l0) / (a / 100))
            else:
                efficiency.append(999)
        best_eff_idx = int(np.argmin(efficiency))
        if best_eff_idx not in annotated:
            t3 = trials[best_eff_idx]
            ax.annotate(f"best efficiency\n{t3['expansion']}x, k={t3.get('topk','?')}, "
                        f"{alive_pcts[best_eff_idx]:.0f}%",
                        xy=(complexities[best_eff_idx], recon[best_eff_idx]),
                        xytext=(-20, -30), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="blue",
                                  alpha=0.8))
            annotated.add(best_eff_idx)

        # 4. Optuna best trial (actual winner, no alive filter)
        objectives = [t["objective"] for t in trials]
        best_optuna_idx = int(np.argmin(objectives))
        ax.scatter([complexities[best_optuna_idx]], [recon[best_optuna_idx]],
                   c="purple", s=50, marker="D", edgecolors="black",
                   linewidths=1, zorder=10, label="optuna best")
        if best_optuna_idx not in annotated:
            t4 = trials[best_optuna_idx]
            ax.annotate(f"{t4['expansion']}x, k={t4.get('topk','?')}, "
                        f"{alive_pcts[best_optuna_idx]:.0f}%",
                        xy=(complexities[best_optuna_idx], recon[best_optuna_idx]),
                        xytext=(20, -20), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="->", color="purple", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", fc="lavender", ec="purple",
                                  alpha=0.8))

        # 5. Selected: best efficiency among alive >= 80% AND stability >= 0.75
        stab_vals = [t.get("stability", 0) for t in trials]
        floor_eff = []
        for r, c, a, l0, s in zip(recon, complexities, alive_pcts, l0_vals, stab_vals):
            if a >= 80 and s >= 0.75:
                floor_eff.append(r * np.sqrt(c) * np.sqrt(l0) / (a / 100))
            else:
                floor_eff.append(999)
        best_floor_idx = int(np.argmin(floor_eff))
        if floor_eff[best_floor_idx] < 999:
            ax.scatter([complexities[best_floor_idx]], [recon[best_floor_idx]],
                       c="lime", s=100, marker="*", edgecolors="darkgreen",
                       linewidths=1, zorder=11, label="selected")
        else:
            ax.annotate("no trials pass\nalive≥80% & stab≥0.75",
                        xy=(0.5, 0.95), xycoords="axes fraction", fontsize=7,
                        ha="center", color="red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="mistyrose", ec="red", alpha=0.8))

        ax.legend(fontsize=7, loc="upper right")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Alive %", fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.92, 1.0])

    out_dir = PROJECT_ROOT / "output" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "complexity_vs_loss_r10.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(out_dir / "complexity_vs_loss_r10.png", bbox_inches="tight", dpi=150)
    print(f"Saved to {out_dir / 'complexity_vs_loss_r10.{{pdf,png}}'}")


if __name__ == "__main__":
    main()
