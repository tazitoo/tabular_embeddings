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


def _pair_npz(sweep_dir: Path, a: str, b: str, dataset: str) -> Path | None:
    for pair_name in (f"{min(a, b)}_vs_{max(a, b)}",
                      f"{a}_vs_{b}", f"{b}_vs_{a}"):
        p = sweep_dir / pair_name / f"{dataset}.npz"
        if p.exists():
            return p
    return None


MITRA_DIM_EMBEDDING = 512  # autogluon mitra dim_embedding default; SelectKBest target


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
    """Return a human-readable reason for a stub/degenerate pair.

    Mitra's `MitraClassifier` (sklearn interface) handles datasets with
    >512 features via SelectKBest truncation, so the framework-level cap
    in mitra_model.py does NOT block extraction. But on some high-feature
    datasets (e.g. Bioresponse, 1776 features) the SelectKBest-truncated
    input is sufficiently out-of-distribution for Mitra that its
    predictions collapse to a near-constant value (AUC ≈ 0.5). When that
    happens we name the mechanism explicitly.
    """
    from scripts.figures.plot_intervention_example_3panel import _broken_model
    broken = _broken_model(dataset, model_a, model_b)
    if broken is None:
        return "degenerate"
    if broken.lower() == "mitra":
        nf = _dataset_n_features(dataset)
        if nf is not None and nf > MITRA_DIM_EMBEDDING:
            return (f"Mitra constant after SelectKBest"
                    f"\n(truncation {nf} \u2192 {MITRA_DIM_EMBEDDING})")
    return f"{broken} constant output"


def _draw_missing(ax, label: str, why: str):
    """Blank panel with a small 'see caption' note.

    Matches the square 0-1 aspect of populated panels so the grid layout
    stays uniform. The actual reason (`why`) is written to a sidecar
    manifest by the grid renderer so it can be embedded in the LaTeX
    figure caption.
    """
    ax.text(0.5, 0.5, "see caption", ha="center", va="center",
            fontsize=5, color="#bbbbbb", style="italic",
            transform=ax.transAxes)
    ax.set_title(label, fontsize=6, pad=2, color="#999999")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
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
    blanks = []  # list of (label, reason) for the sidecar manifest
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx]
        label = f"{DISPLAY_NAMES.get(a, a)} vs {DISPLAY_NAMES.get(b, b)}"
        ap = _pair_npz(ablation_dir, a, b, dataset)
        tp = _pair_npz(transfer_dir, a, b, dataset)
        if ap is None and tp is None:
            _draw_missing(ax, label, "no data")
            blanks.append((label, "no ablation or transfer NPZ available"))
            continue

        try:
            n_drawn, n_wins, strong, weak, _ = draw_intervention_cell(
                ax, ap, tp, marker_size=4.0, base_size=6.0,
            )
        except KeyError:
            reason = _degenerate_reason(dataset, a, b)
            _draw_missing(ax, label, reason)
            blanks.append((label, reason.replace("\n", " ")))
            continue

        disp_s = DISPLAY_NAMES.get(strong, strong)
        disp_w = DISPLAY_NAMES.get(weak, weak)
        # n_wins=0 indicates a tied/degenerate pair: scatter is rendered
        # from baseline preds but there are no intervention arrows.
        if n_wins == 0:
            title = f"{disp_s} vs {disp_w}  (tied)"
        elif ap is None:
            title = f"{disp_s} \u2192 {disp_w}  ({n_drawn}/{n_wins}, transfer only)"
        elif tp is None:
            title = f"{disp_s} \u2192 {disp_w}  ({n_drawn}/{n_wins}, ablation only)"
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

    # Sidecar manifest: { dataset, blanks: [{pair, reason}, ...] }
    # Written next to each PDF so the LaTeX caption can be edited from it.
    manifest = {"dataset": dataset, "blanks": [
        {"pair": label, "reason": reason} for label, reason in blanks
    ]}
    for path in outputs:
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {dataset}: {found}/{n_pairs} pairs → {outputs[-1]}")


if __name__ == "__main__":
    main()
