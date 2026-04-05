#!/usr/bin/env python3
"""Stage 4: Single-concept scatter plots.

For each top concept × its top dataset, generates a focused scatter:
- Gray: all test rows (x=strong P(correct), y=weak P(correct))
- Blue: rows where this concept fires
- Black: strong model predictions after ablating ONLY this concept

This is the "smoking gun" — one concept's causal effect on one dataset.

Usage:
    python -m scripts.causal_chain_analysis.04_plot_concept_scatter --pair tabicl_vs_tabicl_v2
    python -m scripts.causal_chain_analysis.04_plot_concept_scatter --pair tabicl_vs_tabicl_v2 --top-k 3
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._project_root import PROJECT_ROOT
from scripts.intervention.concept_performance_diagnostic import DISPLAY_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATION_DIR = PROJECT_ROOT / "output" / "ablation_sweep"
EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures" / "concept_evidence"


def plot_concept_on_dataset(pair_name, model_key, feature_idx, dataset,
                            feature_label, feature_band, is_unmatched):
    """Generate a single-concept scatter for one dataset."""
    ablation_path = ABLATION_DIR / pair_name / f"{dataset}.npz"
    if not ablation_path.exists():
        return None

    d = np.load(ablation_path, allow_pickle=True)
    strong = str(d["strong_model"])
    weak = str(d["weak_model"])

    if strong != model_key:
        return None  # This concept's model isn't strong on this dataset

    if "strong_wins" not in d.files or "selected_features" not in d.files:
        return None

    sw = d["strong_wins"]
    if sw.sum() == 0:
        return None

    preds_s = d["preds_strong"]
    preds_w = d["preds_weak"]
    preds_i = d["preds_intervened"]
    y = d["y_query"].astype(int)
    sel = d["selected_features"]

    # Scalar scores
    def to_scalar(preds):
        if preds.ndim == 2 and preds.shape[1] > 1:
            return preds[np.arange(len(y)), y]
        return preds.ravel()

    p_s = to_scalar(preds_s)
    p_w = to_scalar(preds_w)
    p_i = to_scalar(preds_i)

    is_reg = preds_s.ndim == 1

    # Find rows where this specific concept was selected
    concept_rows = np.zeros(len(y), dtype=bool)
    if sel.ndim == 2:
        for r in range(len(sel)):
            if sw[r]:
                for fi in sel[r]:
                    if int(fi) == feature_idx:
                        concept_rows[r] = True
                        break

    n_concept = int(concept_rows.sum())
    if n_concept == 0:
        return None

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))

    if is_reg:
        all_vals = np.concatenate([p_s, p_w, p_i])
        lo = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
        hi = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
    else:
        lo, hi = 0, 1

    # All rows: light gray
    ax.scatter(p_s, p_w, c="#dddddd", s=15, alpha=0.5, edgecolors="none",
               zorder=1, label="All rows")

    # Rows where concept fires (before ablation): blue
    ax.scatter(p_s[concept_rows], p_w[concept_rows], c="#4a90d9", s=20,
               alpha=0.7, edgecolors="none", zorder=2,
               label=f"Concept fires (n={n_concept})")

    # After ablation on concept rows: black (moves left — strong degrades)
    ax.scatter(p_i[concept_rows], p_w[concept_rows], c="black", s=12,
               alpha=0.8, edgecolors="none", zorder=3,
               label="After ablation")

    # Arrows from blue to black
    for r in np.where(concept_rows)[0]:
        if abs(p_s[r] - p_i[r]) > 0.01:
            ax.annotate("", xy=(p_i[r], p_w[r]), xytext=(p_s[r], p_w[r]),
                        arrowprops=dict(arrowstyle="-", color="#999999",
                                        lw=0.5, alpha=0.5))

    # y=x line
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    disp_s = DISPLAY_NAMES.get(strong, strong)
    disp_w = DISPLAY_NAMES.get(weak, weak)
    unm_tag = "unmatched" if is_unmatched else "matched"

    ax.set_xlabel(f"{disp_s} P(correct)", fontsize=9)
    ax.set_ylabel(f"{disp_w} P(correct)", fontsize=9)
    ax.set_title(f"{dataset}\n"
                 f"f{feature_idx} [Band {feature_band}, {unm_tag}]\n"
                 f"{feature_label}",
                 fontsize=8, pad=4)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Single-concept scatter plots")
    parser.add_argument("--pair", default=None)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top K concepts per model (default: 5)")
    parser.add_argument("--top-datasets", type=int, default=3,
                        help="Top datasets per concept (default: 3)")
    args = parser.parse_args()

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted(d.name for d in EVIDENCE_DIR.iterdir()
                        if d.is_dir() and (d / "concept_dataset_matrix.json").exists())

    for pair in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair}")

        matrix_path = EVIDENCE_DIR / pair / "concept_dataset_matrix.json"
        if not matrix_path.exists():
            logger.warning(f"  No concept_dataset_matrix.json, run Stage 3 first")
            continue

        matrix_data = json.loads(matrix_path.read_text())

        for model_key, data in matrix_data.items():
            disp = data["display_name"]
            features = data["feature_indices"][:args.top_k]

            for fi in features:
                top_ds = data["concept_top_datasets"].get(str(fi), [])
                label = data["feature_labels"].get(str(fi), "")
                band = data["feature_bands"].get(str(fi), -1)
                unmatched = data["feature_unmatched"].get(str(fi), False)

                for ds_info in top_ds[:args.top_datasets]:
                    ds = ds_info["dataset"]
                    logger.info(f"  {disp} f{fi} on {ds}")

                    fig = plot_concept_on_dataset(
                        pair, model_key, fi, ds,
                        label, band, unmatched,
                    )

                    if fig is None:
                        logger.info(f"    skipped (no data)")
                        continue

                    out_dir = OUTPUT_DIR / pair
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{model_key}_f{fi}_{ds}.pdf"
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"    saved {out_path.name}")


if __name__ == "__main__":
    main()
