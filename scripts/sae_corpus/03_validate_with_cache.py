#!/usr/bin/env python3
"""Validate TabPFN 2.5 inference using the preprocessed cache against TabArena's pipeline.

For each of the 5 validation datasets, loads the preprocessed data from cache
(AutoMLPipelineFeatureGenerator, NaN preserved), runs TabPFN 2.5, and compares
aggregate and per-row log-loss against TabArena's reference predictions.

Plots:
  1. Aggregate scatter — one point per dataset: x=TabArena log-loss, y=ours, y=x line
  2. Per-dataset per-row scatter — each panel: x=TabArena per-row log-loss, y=ours

Output:
    output/sae_training_round9/validation_scatter_aggregate.png
    output/sae_training_round9/validation_scatter_perrow.png
    output/sae_training_round9/validation_scatter_report.json

Usage:
    python scripts/sae_corpus/03_validate_with_cache.py
    python scripts/sae_corpus/03_validate_with_cache.py --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.preprocessing import CACHE_DIR, load_preprocessed
from scripts._project_root import PROJECT_ROOT

OOF_DIR = PROJECT_ROOT / "output" / "sae_training_round9" / "tabarena_oof_predictions"
OUTPUT_DIR = PROJECT_ROOT / "output" / "sae_training_round9"

DATASETS = [
    "blood-transfusion-service-center",
    "diabetes",
    "website_phishing",
    "anneal",
    "credit-g",
]


def perrow_logloss(proba: np.ndarray, y_int: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    if proba.ndim == 1:
        p = np.clip(proba, eps, 1 - eps)
        return -(y_int * np.log(p) + (1 - y_int) * np.log(1 - p))
    n = len(y_int)
    p = np.clip(proba[np.arange(n), y_int], eps, 1 - eps)
    return -np.log(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from models.tabpfn_utils import load_tabpfn
    print(f"Loading TabPFN 2.5 on {args.device}...")
    model = load_tabpfn(task="classification", device=args.device, n_estimators=4)

    agg_results = {}   # dataset → {our_logloss, tabarena_logloss, diff}
    perrow_data = {}   # dataset → {our_ll, tabarena_ll}

    for name in DATASETS:
        ref_path = OOF_DIR / f"{name}.json"
        if not ref_path.exists():
            print(f"  SKIP {name}: no reference file")
            continue

        ref = json.loads(ref_path.read_text())
        tabarena_proba = np.array(ref["y_pred_proba_test"])
        y_test_tabarena = np.array(ref["y_test"], dtype=int)

        data = load_preprocessed("tabpfn", name, CACHE_DIR)
        print(f"\n[{name}] n_train={len(data.X_train)} n_test={len(data.X_test)}")

        model.fit(data.X_train, data.y_train)
        our_proba = model.predict_proba(data.X_test)

        our_ll_agg = float(log_loss(data.y_test, our_proba))
        tab_ll_agg = float(log_loss(y_test_tabarena, tabarena_proba))
        diff = our_ll_agg - tab_ll_agg

        our_ll_row = perrow_logloss(our_proba, data.y_test)
        tab_ll_row = perrow_logloss(tabarena_proba, y_test_tabarena)

        agg_results[name] = {
            "our_logloss": our_ll_agg,
            "tabarena_logloss": tab_ll_agg,
            "diff": diff,
        }
        perrow_data[name] = {
            "our_ll": our_ll_row.tolist(),
            "tabarena_ll": tab_ll_row.tolist(),
        }

        sym = "✓" if abs(diff) <= 0.10 else "⚠"
        print(f"  {sym} our={our_ll_agg:.4f}  tabarena={tab_ll_agg:.4f}  diff={diff:+.4f}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Plot 1: per-row scatter, all datasets combined, colored by dataset ---
    fig1, ax = plt.subplots(figsize=(6, 6))
    all_tab = np.concatenate([np.array(v["tabarena_ll"]) for v in perrow_data.values()])
    all_our = np.concatenate([np.array(v["our_ll"]) for v in perrow_data.values()])
    lo = min(all_tab.min(), all_our.min()) * 0.9
    hi = max(all_tab.max(), all_our.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x", zorder=3)
    for i, (name, rows) in enumerate(perrow_data.items()):
        our = np.array(rows["our_ll"])
        tab = np.array(rows["tabarena_ll"])
        short = name.replace("-service-center", "").replace("_", " ")
        ax.scatter(tab, our, alpha=0.35, s=12, color=colors[i % len(colors)],
                   label=f"{short} (n={len(our)})", zorder=2)
    ax.set_xlabel("TabArena per-row log-loss", fontsize=11)
    ax.set_ylabel("Our per-row log-loss (cache preprocessing)", fontsize=11)
    ax.set_title("TabPFN 2.5 — per-row log-loss, all test samples\nour pipeline vs TabArena's",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    plt.tight_layout()
    out1 = OUTPUT_DIR / "validation_scatter_aggregate.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nPer-row scatter (combined) → {out1}")

    # --- Plot 2: per-dataset per-row scatter (one panel each) ---
    n = len(perrow_data)
    fig2, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, perrow_data.items()):
        our = np.array(rows["our_ll"])
        tab = np.array(rows["tabarena_ll"])
        lo = min(our.min(), tab.min()) * 0.9
        hi = max(our.max(), tab.max()) * 1.1
        ax.scatter(tab, our, alpha=0.25, s=8, color="#4C72B0")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        short = name.replace("-service-center", "").replace("_", " ")
        ax.set_title(f"{short}\n(n={len(our)})", fontsize=9, pad=4)
        ax.set_xlabel("TabArena per-row log-loss", fontsize=8)
        ax.set_ylabel("Our per-row log-loss" if ax is axes[0] else "", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    fig2.suptitle("Per-row log-loss: our pipeline (cache) vs TabArena's  |  y=x = identical predictions",
                  fontsize=10, y=1.02)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "validation_scatter_perrow.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Per-row scatter (panels)   → {out2}")

    # --- Save JSON report ---
    report = {ds: agg_results[ds] for ds in agg_results}
    out3 = OUTPUT_DIR / "validation_scatter_report.json"
    out3.write_text(json.dumps(report, indent=2))
    print(f"Report            → {out3}")


if __name__ == "__main__":
    main()
