#!/usr/bin/env python3
"""Analyze SAE corpus sampling strategies on a single dataset.

Computes per-row difficulty (log-loss or MSE) from TabPFN predictions on the
holdout set, then visualizes how 4 sampling strategies select from the
difficulty distribution:

  1. Random — uniform random sample
  2. Target-stratified — proportional class representation
  3. Difficulty-stratified — equal easy/medium/hard terciles
  4. Target × difficulty — stratify on both axes

Output:
    output/sae_training_round9/sampling_analysis_{dataset}.png

Usage:
    python scripts/sae_corpus/05_sampling_analysis.py --dataset churn
    python scripts/sae_corpus/05_sampling_analysis.py --dataset wine_quality --n-sample 500
    python scripts/sae_corpus/05_sampling_analysis.py --dataset credit-g --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.preprocessing import CACHE_DIR, load_preprocessed
from models.layer_extraction import load_and_fit, predict
from scripts._project_root import PROJECT_ROOT

SPLITS_PATH = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_splits.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9"


def per_row_loss(y_true: np.ndarray, proba: np.ndarray, task_type: str) -> np.ndarray:
    """Compute per-row loss: log-loss for classification, squared error for regression."""
    eps = 1e-15
    if task_type == "classification":
        if proba.ndim == 1:
            p = np.clip(proba, eps, 1 - eps)
            return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        n = len(y_true)
        p = np.clip(proba[np.arange(n), y_true], eps, 1 - eps)
        return -np.log(p)
    else:
        return (y_true - proba) ** 2


def tercile_labels(losses: np.ndarray) -> np.ndarray:
    """Assign each row to easy (0), medium (1), or hard (2) tercile."""
    t1 = np.percentile(losses, 33.3)
    t2 = np.percentile(losses, 66.7)
    labels = np.zeros(len(losses), dtype=int)
    labels[losses > t1] = 1
    labels[losses > t2] = 2
    return labels


def sample_random(n_total, n_sample, seed=42):
    """Uniform random sample."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=min(n_sample, n_total), replace=False)


def sample_target_stratified(y, n_sample, seed=42):
    """Sample proportional to class frequency."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    indices = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(n_sample * count / len(y)))
        indices.append(rng.choice(cls_idx, size=min(n_take, len(cls_idx)), replace=False))
    indices = np.concatenate(indices)
    # Trim to exactly n_sample
    if len(indices) > n_sample:
        indices = rng.choice(indices, size=n_sample, replace=False)
    return indices


def sample_difficulty_stratified(losses, n_sample, seed=42):
    """Equal representation from easy/medium/hard terciles."""
    rng = np.random.RandomState(seed)
    terciles = tercile_labels(losses)
    per_tercile = n_sample // 3
    indices = []
    for t in range(3):
        t_idx = np.where(terciles == t)[0]
        n_take = min(per_tercile, len(t_idx))
        indices.append(rng.choice(t_idx, size=n_take, replace=False))
    return np.concatenate(indices)


def sample_target_x_difficulty(y, losses, n_sample, seed=42):
    """Stratify on target class × difficulty tercile."""
    rng = np.random.RandomState(seed)
    terciles = tercile_labels(losses)
    classes = np.unique(y)
    n_strata = len(classes) * 3
    per_stratum = max(1, n_sample // n_strata)
    indices = []
    for cls in classes:
        for t in range(3):
            mask = (y == cls) & (terciles == t)
            stratum_idx = np.where(mask)[0]
            n_take = min(per_stratum, len(stratum_idx))
            if n_take > 0:
                indices.append(rng.choice(stratum_idx, size=n_take, replace=False))
    return np.concatenate(indices) if indices else np.array([], dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE sampling strategies")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-sample", type=int, default=500,
                        help="Target sample size per strategy (default: 500)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context", type=int, default=1024)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    split_info = splits[args.dataset]
    task_type = split_info["task_type"]

    data = load_preprocessed("tabpfn", args.dataset, CACHE_DIR)
    print(f"Dataset: {args.dataset}")
    print(f"  task={task_type}, holdout={len(data.X_test)}/{len(data.X_train)+len(data.X_test)} rows")

    # Context sampling
    X_ctx, y_ctx = data.X_train, data.y_train
    if len(X_ctx) > args.max_context:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_ctx), args.max_context, replace=False)
        X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]

    # Fit TabPFN and predict on holdout
    fit_kwargs = {}
    if data.cat_indices:
        fit_kwargs["cat_indices"] = data.cat_indices
    clf = load_and_fit("tabpfn", X_ctx, y_ctx, task=task_type, device=args.device, **fit_kwargs)
    preds = predict(clf, data.X_test, task=task_type)
    losses = per_row_loss(data.y_test, preds, task_type)

    print(f"  loss: mean={losses.mean():.4f}  std={losses.std():.4f}  "
          f"p5={np.percentile(losses,5):.4f}  p95={np.percentile(losses,95):.4f}")

    terciles = tercile_labels(losses)
    for t, label in enumerate(["easy", "medium", "hard"]):
        t_losses = losses[terciles == t]
        print(f"  {label}: n={len(t_losses)}  loss=[{t_losses.min():.4f}, {t_losses.max():.4f}]")

    # Apply 4 sampling strategies
    n = args.n_sample
    strategies = {
        "1. Random": sample_random(len(losses), n),
        "2. Target-stratified": sample_target_stratified(data.y_test, n),
        "3. Difficulty-stratified": sample_difficulty_stratified(losses, n),
        "4. Target × Difficulty": sample_target_x_difficulty(data.y_test, losses, n),
    }

    # --- Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for col, (name, idx) in enumerate(strategies.items()):
        ax_hist = axes[0, col]
        ax_bar = axes[1, col]

        # Top row: loss distribution — full vs sampled
        bins = np.linspace(0, np.percentile(losses, 99), 40)
        ax_hist.hist(losses, bins=bins, alpha=0.3, color="gray", label=f"all (n={len(losses)})",
                     density=True)
        ax_hist.hist(losses[idx], bins=bins, alpha=0.7, color="#4C72B0",
                     label=f"sampled (n={len(idx)})", density=True)
        ax_hist.set_title(name, fontsize=10, fontweight="bold")
        ax_hist.set_xlabel("per-row loss" if col == 0 else "", fontsize=8)
        ax_hist.set_ylabel("density" if col == 0 else "", fontsize=8)
        ax_hist.legend(fontsize=7)
        ax_hist.tick_params(labelsize=7)

        # Bottom row: target class distribution — full vs sampled
        if task_type == "classification":
            classes = np.unique(data.y_test)
            full_counts = np.array([(data.y_test == c).sum() for c in classes])
            sample_counts = np.array([(data.y_test[idx] == c).sum() for c in classes])
            x = np.arange(len(classes))
            w = 0.35
            ax_bar.bar(x - w/2, full_counts / full_counts.sum(), w, alpha=0.5,
                      color="gray", label="all")
            ax_bar.bar(x + w/2, sample_counts / sample_counts.sum() if sample_counts.sum() > 0 else sample_counts,
                      w, alpha=0.8, color="#4C72B0", label="sampled")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels([str(c) for c in classes], fontsize=7)
            ax_bar.set_xlabel("class" if col == 0 else "", fontsize=8)
            ax_bar.set_ylabel("proportion" if col == 0 else "", fontsize=8)
        else:
            ax_bar.hist(data.y_test, bins=30, alpha=0.3, color="gray",
                       label="all", density=True)
            ax_bar.hist(data.y_test[idx], bins=30, alpha=0.7, color="#4C72B0",
                       label="sampled", density=True)
            ax_bar.set_xlabel("target value" if col == 0 else "", fontsize=8)
            ax_bar.set_ylabel("density" if col == 0 else "", fontsize=8)
        ax_bar.legend(fontsize=7)
        ax_bar.tick_params(labelsize=7)

    fig.suptitle(
        f"SAE corpus sampling strategies — {args.dataset}\n"
        f"holdout={len(data.X_test)} rows, sample={args.n_sample}, "
        f"task={task_type}, loss from TabPFN",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"sampling_analysis_{args.dataset}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure → {out_path}")


if __name__ == "__main__":
    main()
