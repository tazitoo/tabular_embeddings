#!/usr/bin/env python3
"""Two-by-two importance decay grid.

Top row (LOO): per-feature |Δpred| ranked descending across all
(model, dataset) pairs, split into MNN-matched (left) and unmatched
(right) features. Same data as plot_importance_decay.py.

Bottom row (greedy step): per-step |Δpred| from the greedy ablation
(left) and transfer (right) sweeps. For each (pair, dataset, row), the
sweep records the strong model's prediction after each greedy step in
`step_preds`; we take successive pairwise deltas (baseline → step 1,
step 1 → step 2, …), reduce over the class axis with total-variation
distance for classification or absolute value for regression, then
average per-step across all rows grouped by strong model.

Usage:
    python -m scripts.figures.4_results.plot_importance_decay_grid
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import importlib

from scripts._project_root import PROJECT_ROOT
from scripts.paper._paper_repo import paper_figure_path

# Python identifiers can't start with a digit, so `4_results` can't appear
# in a regular import statement — use importlib to pull the helpers from
# the sibling plot_importance_decay module.
_decay = importlib.import_module(
    "scripts.figures.4_results.plot_importance_decay"
)
MODELS = _decay.MODELS
DISPLAY = _decay.DISPLAY
MODEL_COLORS = _decay.MODEL_COLORS
MODEL_LINESTYLE = _decay.MODEL_LINESTYLE
MATCHING_FILE = _decay.MATCHING_FILE
CLS_DATASETS = _decay.CLS_DATASETS
build_matched_sets = _decay.build_matched_sets
accumulate_rank_curve = _decay.accumulate_rank_curve

OUT_DIR = PROJECT_ROOT / "output" / "figures"
ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep_tols"
TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols"


def _style_axes(ax, title: str, xlabel: str = "Rank descending") -> None:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-5, 1.5e0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def _step_delta_row(pred_row: np.ndarray, baseline: np.ndarray,
                    step_sizes_row: np.ndarray) -> np.ndarray:
    """Compute per-step |Δpred| for one row's greedy walk, returned
    sorted descending by magnitude so aggregation-by-rank yields a
    monotonically decreasing curve (consistent with the top-row LOO
    panels, which also sort per-row |Δ| descending before averaging).

    `pred_row` has shape (MAX_STEPS,) for regression or
    (MAX_STEPS, n_classes) for classification. `baseline` is the
    pre-intervention prediction (shape matches one step).
    `step_sizes_row[s] > 0` marks valid steps; NaN steps are dropped.

    Returns an array of length equal to the number of valid steps.
    """
    n_steps = int((step_sizes_row > 0).sum())
    if n_steps == 0:
        return np.empty(0, dtype=np.float32)
    seq = pred_row[:n_steps]
    prev = np.concatenate([baseline[None, ...], seq[:-1]], axis=0)
    diff = seq - prev
    if diff.ndim == 1:
        mags = np.abs(diff)
    else:
        # Multi-class: total variation distance = 0.5 * L1 norm over classes
        mags = 0.5 * np.abs(diff).sum(axis=-1)
    return -np.sort(-mags)


def _accumulate_step_curves(sweep_dir) -> dict[str, list[list[float]]]:
    """Walk a sweep dir and collect per-rank |Δpred| arrays grouped by
    strong model. Each row's deltas are sorted descending, then padded
    with zeros to MAX_STEPS — so every row contributes a value at every
    rank, and the rank-k mean is a strict descent. The zero-pad
    interpretation: rows that stopped early didn't need rank-k
    concepts, so those concepts had |Δ|=0 impact on that row.

    Classification datasets only — regression step_preds are raw scalars
    at target scale (e.g. house prices in thousands) which would dominate
    the mean and push the log-scale curves off-chart.
    """
    MAX_STEPS = 20
    per_model: dict[str, list[list[float]]] = {}
    for pair_dir in sorted(sweep_dir.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        for npz_path in sorted(pair_dir.glob("*.npz")):
            if npz_path.stem not in CLS_DATASETS:
                continue
            try:
                d = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            needed = {"step_preds", "step_sizes", "preds_strong",
                      "strong_model", "strong_wins"}
            if not needed.issubset(set(d.files)):
                continue
            strong = str(d["strong_model"])
            sw = d["strong_wins"].astype(bool)
            if not sw.any():
                continue
            step_preds = d["step_preds"][sw]
            step_sizes = d["step_sizes"][sw]
            baseline = d["preds_strong"][sw]
            bucket = per_model.setdefault(strong, [[] for _ in range(MAX_STEPS)])
            for r in range(step_preds.shape[0]):
                deltas = _step_delta_row(step_preds[r], baseline[r],
                                         step_sizes[r])
                # Pad to MAX_STEPS so every row contributes at every rank.
                for s in range(MAX_STEPS):
                    v = float(deltas[s]) if s < len(deltas) else 0.0
                    if np.isfinite(v):
                        bucket[s].append(v)
    return per_model


def _step_delta_row_vs_weak(
    pred_row: np.ndarray, weak_baseline: np.ndarray,
    step_sizes_row: np.ndarray,
) -> np.ndarray:
    """Transfer variant: deltas are (injected prediction) − (previous
    injected prediction) with the baseline being the weak model's
    pre-injection prediction. Sorted descending per row (see
    `_step_delta_row`).
    """
    n_steps = int((step_sizes_row > 0).sum())
    if n_steps == 0:
        return np.empty(0, dtype=np.float32)
    seq = pred_row[:n_steps]
    prev = np.concatenate([weak_baseline[None, ...], seq[:-1]], axis=0)
    diff = seq - prev
    if diff.ndim == 1:
        mags = np.abs(diff)
    else:
        mags = 0.5 * np.abs(diff).sum(axis=-1)
    return -np.sort(-mags)


def _accumulate_transfer_curves(sweep_dir) -> dict[str, list[list[float]]]:
    """Like `_accumulate_step_curves` but groups by STRONG model (the
    source of the injected concepts). Baseline is `preds_weak`.
    Same zero-pad monotonicity treatment as `_accumulate_step_curves`.
    Classification datasets only.
    """
    MAX_STEPS = 20
    per_model: dict[str, list[list[float]]] = {}
    for pair_dir in sorted(sweep_dir.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        for npz_path in sorted(pair_dir.glob("*.npz")):
            if npz_path.stem not in CLS_DATASETS:
                continue
            try:
                d = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            needed = {"step_preds", "step_sizes", "preds_weak",
                      "strong_model", "strong_wins"}
            if not needed.issubset(set(d.files)):
                continue
            strong = str(d["strong_model"])
            sw = d["strong_wins"].astype(bool)
            if not sw.any():
                continue
            step_preds = d["step_preds"][sw]
            step_sizes = d["step_sizes"][sw]
            baseline = d["preds_weak"][sw]
            bucket = per_model.setdefault(strong, [[] for _ in range(MAX_STEPS)])
            for r in range(step_preds.shape[0]):
                deltas = _step_delta_row_vs_weak(step_preds[r], baseline[r],
                                                 step_sizes[r])
                for s in range(MAX_STEPS):
                    v = float(deltas[s]) if s < len(deltas) else 0.0
                    if np.isfinite(v):
                        bucket[s].append(v)
    return per_model


def _plot_step_curves(ax, per_model: dict[str, list[list[float]]],
                      order: list[str], title: str) -> None:
    _style_axes(ax, title, xlabel="Rank descending")
    for model in order:
        if model not in per_model:
            continue
        buckets = per_model[model]
        xs, ys = [], []
        for s, vals in enumerate(buckets, start=1):
            if not vals:
                continue
            xs.append(s)
            ys.append(float(np.mean(vals)))
        if not xs:
            continue
        ax.plot(xs, ys, color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLE[model], lw=1.6,
                label=DISPLAY[model])


def main() -> None:
    matched_sets = build_matched_sets(MATCHING_FILE)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        m, u = accumulate_rank_curve(model, matched_sets[model])
        curves[model] = (m, u)
        print(
            f"{DISPLAY[model]:10s}  matched curve len {m.size}  "
            f"unmatched curve len {u.size}"
        )

    order = sorted(
        MODELS,
        key=lambda m: curves[m][0][0] if curves[m][0].size else -1.0,
        reverse=True,
    )
    print("Plot order:", ", ".join(DISPLAY[m] for m in order))

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharey=True, sharex="row")
    top_left, top_right = axes[0]
    bot_left, bot_right = axes[1]

    for model in order:
        m_curve, u_curve = curves[model]
        color = MODEL_COLORS[model]
        ls = MODEL_LINESTYLE[model]
        label = DISPLAY[model]
        if m_curve.size:
            x = np.arange(1, m_curve.size + 1)
            top_left.plot(x, m_curve, color=color, linestyle=ls,
                          lw=1.6, label=label)
        if u_curve.size:
            x = np.arange(1, u_curve.size + 1)
            top_right.plot(x, u_curve, color=color, linestyle=ls,
                           lw=1.6, label=label)

    _style_axes(top_left, "(a) LOO decay — matched concepts")
    _style_axes(top_right, "(b) LOO decay — unmatched concepts")

    print("Computing ablation step |Δpred| curves…")
    ablation_curves = _accumulate_step_curves(ABLATION_DIR)
    _plot_step_curves(bot_left, ablation_curves, order,
                      "(c) Ablation step $|\\Delta\\text{pred}|$")

    print("Computing transfer step |Δpred| curves…")
    transfer_curves = _accumulate_transfer_curves(TRANSFER_DIR)
    _plot_step_curves(bot_right, transfer_curves, order,
                      "(d) Transfer step $|\\Delta\\text{pred}|$")

    axes[0, 0].set_ylabel("Mean $|\\Delta\\text{pred}|$ (LOO)")
    axes[1, 0].set_ylabel("Mean $|\\Delta\\text{pred}|$ (step)")
    top_left.legend(fontsize=8, loc="upper right", ncol=1, frameon=True)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUT_DIR / "importance_decay_grid.pdf"
    paper_path = paper_figure_path("4_results", "importance_decay_grid.pdf")
    for path in (local_path, paper_path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
