#!/usr/bin/env python3
"""
Plot per-row log-loss differences: our TabPFN 2.5 pipeline vs TabArena's.

Re-runs our inference on the 5 validation test folds, computes per-row log-loss
for both pipelines, and plots the distribution of differences (ours − TabArena).
Aggregate similarity can hide large per-row deviations; this checks for that.

Output:
    output/sae_training_round9/validation_logloss_dist.png
    output/sae_training_round9/validation_logloss_perrow.json

Usage:
    python scripts/sae_corpus/plot_validation_logloss.py
    python scripts/sae_corpus/plot_validation_logloss.py --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.extended_loader import _load_tabarena_cached_v2
from scripts._project_root import PROJECT_ROOT

OOF_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_oof_predictions"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9"
SPLITS_PATH = OUTPUT_DIR / "tabarena_splits.json"

DATASETS = [
    "blood-transfusion-service-center",
    "diabetes",
    "website_phishing",
    "anneal",
    "credit-g",
]


def perrow_logloss(proba: np.ndarray, y_int: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Per-row log-loss.

    Args:
        proba: (n, n_classes) full probability matrix, or (n,) P(positive) for binary.
        y_int: integer class labels, 0-indexed.
    """
    if proba.ndim == 1:
        # Binary stored as P(positive class)
        p = np.clip(proba, eps, 1 - eps)
        return -(y_int * np.log(p) + (1 - y_int) * np.log(1 - p))
    else:
        n = len(y_int)
        p = np.clip(proba[np.arange(n), y_int], eps, 1 - eps)
        return -np.log(p)


def run_our_pipeline(
    name: str,
    split_info: dict,
    model,
) -> tuple[np.ndarray, np.ndarray]:
    """Run our TabPFN 2.5 on the test fold.

    Returns:
        our_proba: (n_test, n_classes) predicted probabilities
        y_test_enc: integer-encoded ground truth labels (LabelEncoder order)
    """
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    cached = _load_tabarena_cached_v2(name)
    if cached is None:
        raise ValueError(f"No v2 cache for {name}. Run 01_validate_inference.py first.")
    X_df, y = cached

    train_idx = np.array(split_info["train_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include="number").columns
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if len(cat_cols):
        enc.fit(X_train[cat_cols])

    def to_numpy(df):
        parts = []
        if len(num_cols):
            parts.append(df[num_cols].values.astype(np.float32))
        if len(cat_cols):
            parts.append(enc.transform(df[cat_cols]).astype(np.float32))
        return np.nan_to_num(np.concatenate(parts, axis=1))

    model.fit(to_numpy(X_train), y_train_enc)
    our_proba = model.predict_proba(to_numpy(X_test))  # always (n_test, n_classes)
    return our_proba, y_test_enc


def main():
    parser = argparse.ArgumentParser(description="Plot per-row log-loss differences")
    parser.add_argument("--device", default="cuda", help="Torch device for TabPFN")
    args = parser.parse_args()

    from models.tabpfn_utils import load_tabpfn
    splits = json.loads(SPLITS_PATH.read_text())

    print(f"Loading TabPFN 2.5 on {args.device}...")
    model = load_tabpfn(task="classification", device=args.device, n_estimators=4)

    all_diffs = {}

    for name in DATASETS:
        split_info = splits[name]
        ref_path = OOF_DIR / f"{name}.json"
        if not ref_path.exists():
            print(f"  SKIP {name}: no reference file")
            continue

        ref = json.loads(ref_path.read_text())
        tabarena_proba = np.array(ref["y_pred_proba_test"])
        y_test_tabarena = np.array(ref["y_test"], dtype=int)

        print(f"\n[{name}] n_test={len(y_test_tabarena)}")
        our_proba, y_test_our = run_our_pipeline(name, split_info, model)

        # Sanity check: label encodings should match
        if not np.array_equal(y_test_our, y_test_tabarena):
            n_mismatch = (y_test_our != y_test_tabarena).sum()
            print(f"  WARNING: {n_mismatch}/{len(y_test_our)} label mismatches between "
                  f"LabelEncoder and LabelCleaner — log-loss comparison may be invalid")

        tabarena_ll = perrow_logloss(tabarena_proba, y_test_tabarena)
        our_ll = perrow_logloss(our_proba, y_test_our)
        diffs = our_ll - tabarena_ll
        all_diffs[name] = diffs.tolist()

        print(f"  mean={diffs.mean():+.4f}  std={diffs.std():.4f}  "
              f"p5={np.percentile(diffs, 5):+.4f}  p95={np.percentile(diffs, 95):+.4f}  "
              f"max_abs={np.abs(diffs).max():.4f}")

    # Save per-row data
    perrow_path = OUTPUT_DIR / "validation_logloss_perrow.json"
    perrow_path.write_text(json.dumps(all_diffs, indent=2))
    print(f"\nPer-row data → {perrow_path}")

    # Plot: 5 panels, one per dataset
    n = len(all_diffs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, diffs_list) in zip(axes, all_diffs.items()):
        diffs = np.array(diffs_list)
        ax.hist(diffs, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, zorder=5)
        ax.axvline(diffs.mean(), color="darkorange", linestyle="-", linewidth=1.5, zorder=5,
                   label=f"mean={diffs.mean():+.3f}")
        short = name.replace("-service-center", "").replace("_", " ")
        ax.set_title(short, fontsize=9, pad=4)
        ax.set_xlabel("Δ log-loss (ours − TabArena)", fontsize=8)
        ax.set_ylabel("Count" if ax is axes[0] else "", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.7)

    fig.suptitle(
        "Per-row log-loss difference: our TabPFN 2.5 pipeline vs TabArena's\n"
        "(positive = we are worse, negative = we are better, red = no difference)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "validation_logloss_dist.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure → {fig_path}")


if __name__ == "__main__":
    main()
