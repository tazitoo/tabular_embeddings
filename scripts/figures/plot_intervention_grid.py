#!/usr/bin/env python3
"""Per-dataset combined ablation + transfer grid.

Replaces the two separate per-dataset grids in the appendix (one for
ablation, one for transfer) with a single grid where every cell shows the
L-shape used by plot_intervention_example_3panel:

    - grey baseline dots at (strong P(correct), weak P(correct))
    - horizontal segment to the ablated strong P(correct)
    - vertical segment to the transferred weak P(correct)
    - small black dots at both post-intervention endpoints

Each file lays out all SAE-eligible model pairs for one dataset in a
5x3 grid (15 pairs for classification, 10 pairs for regression).

Usage:
    python -m scripts.figures.plot_intervention_grid --dataset credit-g
    python -m scripts.figures.plot_intervention_grid --dataset credit-g \\
        --ablation-dir output/ablation_sweep \\
        --transfer-dir output/transfer_global_mnnp90_trained
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt

from scripts._project_root import PROJECT_ROOT
from scripts.figures.plot_intervention_example_3panel import (
    DISPLAY_NAMES,
    draw_intervention_cell,
)
from scripts.paper._paper_repo import paper_figure_path

MODELS_CLS = ["tabpfn", "mitra", "tabicl", "tabicl_v2", "tabdpt", "carte"]
MODELS_REG = ["tabpfn", "mitra", "tabicl_v2", "tabdpt", "carte"]

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures" / "intervention_grid"
SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"


MITRA_MAX_FEATURES = 500  # autogluon/tabular/models/mitra/mitra_model.py:194


def _pair_npz(sweep_dir: Path, a: str, b: str, dataset: str) -> Path | None:
    for pair_name in (f"{min(a, b)}_vs_{max(a, b)}",
                      f"{a}_vs_{b}", f"{b}_vs_{a}"):
        p = sweep_dir / pair_name / f"{dataset}.npz"
        if p.exists():
            return p
    return None


def _dataset_n_features(dataset: str) -> int | None:
    """Return n_features for a TabArena dataset, or None if unavailable."""
    try:
        from data.extended_loader import load_tabarena_dataset
    except Exception:
        return None
    try:
        X, *_ = load_tabarena_dataset(dataset)
        return int(X.shape[1])
    except Exception:
        return None


def _degenerate_reason(dataset: str, model_a: str, model_b: str) -> str:
    """Return a human-readable reason string for a degenerate pair.

    Priority:
    1. Mitra + n_features > 500: exceeds AutoGluon's documented cap
       (autogluon/tabular/models/mitra/mitra_model.py:194).
    2. Identify which specific model has constant output and name it —
       without claiming a known cause when we don't have one.
    3. Fall back to generic "degenerate" when neither applies.
    """
    from scripts.figures.plot_intervention_example_3panel import _broken_model

    if "mitra" in (model_a, model_b):
        nf = _dataset_n_features(dataset)
        if nf is not None and nf > MITRA_MAX_FEATURES:
            return (f"Mitra requires n_features < {MITRA_MAX_FEATURES}"
                    f"\n(dataset has {nf})")
    broken = _broken_model(dataset, model_a, model_b)
    if broken is not None:
        return f"{broken} collapsed\n(constant output)"
    return "degenerate"


def _draw_missing(ax, label: str, why: str):
    ax.text(0.5, 0.5, why, ha="center", va="center",
            fontsize=6, color="#999999", transform=ax.transAxes)
    ax.set_title(label, fontsize=6, pad=2, color="#999999")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset combined intervention grid")
    parser.add_argument("--dataset", default=None,
                        help="Single dataset. Omit with --all to run every "
                             "SAE-eligible dataset in the splits file.")
    parser.add_argument("--all", action="store_true",
                        help="Render every dataset in tabarena_splits.json")
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--transfer-dir", type=Path, default=TRANSFER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())

    if args.all:
        datasets = sorted(splits.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("--dataset or --all is required")

    for dataset in datasets:
        task_type = splits.get(dataset, {}).get("task_type", "classification")
        _render_one(dataset, task_type, args.ablation_dir, args.transfer_dir,
                    args.output if len(datasets) == 1 else None)


def _render_one(dataset: str, task_type: str,
                ablation_dir: Path, transfer_dir: Path,
                output_override: Path | None):

    if task_type == "regression":
        model_list = MODELS_REG
        ncols = 3
        fig_height = 6.5
    else:
        model_list = MODELS_CLS
        ncols = 3
        fig_height = 8.0

    pairs = list(combinations(model_list, 2))
    n_pairs = len(pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, fig_height))
    axes = axes.flatten()

    found = 0
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx]
        label = f"{DISPLAY_NAMES.get(a, a)} vs {DISPLAY_NAMES.get(b, b)}"
        ap = _pair_npz(ablation_dir, a, b, dataset)
        tp = _pair_npz(transfer_dir, a, b, dataset)
        if ap is None or tp is None:
            _draw_missing(ax, label, "no data")
            continue

        try:
            n_drawn, n_wins, strong, weak, _ = draw_intervention_cell(
                ax, ap, tp, marker_size=4.0, base_size=6.0,
            )
        except KeyError:
            _draw_missing(ax, label, _degenerate_reason(dataset, a, b))
            continue

        disp_s = DISPLAY_NAMES.get(strong, strong)
        disp_w = DISPLAY_NAMES.get(weak, weak)
        # n_wins=0 indicates a tied/degenerate pair: scatter is rendered
        # from baseline preds but there are no intervention arrows.
        if n_wins == 0:
            title = f"{disp_s} vs {disp_w}  (tied)"
        else:
            title = f"{disp_s} \u2192 {disp_w}  ({n_drawn}/{n_wins})"
        ax.set_title(title, fontsize=6, pad=2)
        ax.tick_params(labelsize=4, length=2, pad=1)
        found += 1

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"{dataset} — pairwise ablation + transfer "
        f"({found}/{n_pairs} pairs)",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.8, w_pad=0.5)

    # Dual-write: local (under output/figures/intervention_grid) and
    # paper repo (under figures/E_appendix), keeping the existing
    # `ablation_grid_{dataset}.pdf` naming for drop-in replacement.
    if output_override is not None:
        outputs = [output_override]
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        outputs = [
            OUTPUT_DIR / f"{dataset}.pdf",
            paper_figure_path("E_appendix", f"ablation_grid_{dataset}.pdf"),
        ]
    for path in outputs:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {dataset}: {found}/{n_pairs} pairs → {outputs[-1]}")


if __name__ == "__main__":
    main()
