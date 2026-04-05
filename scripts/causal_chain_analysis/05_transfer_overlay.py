#!/usr/bin/env python3
"""Stage 5: Transfer overlay on concept scatter plots.

For concepts that appear in both ablation and transfer results, overlays
the transfer effect (green dots) on the ablation scatter. Shows whether
the concept that hurts the strong model when removed also helps the weak
model when transferred.

Usage:
    python -m scripts.causal_chain_analysis.05_transfer_overlay --pair mitra_vs_tabpfn
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
TRANSFER_DIR = PROJECT_ROOT / "output" / "transfer_sweep_v2"
EVIDENCE_DIR = PROJECT_ROOT / "output" / "concept_evidence"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures" / "concept_evidence"


def plot_ablation_vs_transfer(pair_name, model_key, feature_idx, dataset,
                               feature_label, feature_band, is_unmatched):
    """Two-panel: ablation effect (left) and transfer effect (right)."""
    abl_path = ABLATION_DIR / pair_name / f"{dataset}.npz"
    trf_path = TRANSFER_DIR / pair_name / f"{dataset}.npz"

    if not abl_path.exists() or not trf_path.exists():
        return None

    abl = np.load(abl_path, allow_pickle=True)
    trf = np.load(trf_path, allow_pickle=True)

    strong = str(abl["strong_model"])
    weak = str(abl["weak_model"])
    if strong != model_key:
        return None

    sw = abl["strong_wins"]
    if sw.sum() == 0:
        return None

    y = abl["y_query"].astype(int)

    def to_scalar(preds):
        if preds.ndim == 2 and preds.shape[1] > 1:
            return preds[np.arange(len(y)), y]
        return preds.ravel()

    p_s = to_scalar(abl["preds_strong"])
    p_w = to_scalar(abl["preds_weak"])
    p_abl = to_scalar(abl["preds_intervened"])

    # Transfer predictions
    trf_key = "preds_transferred" if "preds_transferred" in trf.files else "preds_intervened"
    p_trf = to_scalar(trf[trf_key])

    is_reg = abl["preds_strong"].ndim == 1

    # Find rows where this concept was selected in ablation
    sel = abl["selected_features"]
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

    if is_reg:
        all_vals = np.concatenate([p_s, p_w, p_abl, p_trf])
        lo = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
        hi = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
    else:
        lo, hi = 0, 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    disp_s = DISPLAY_NAMES.get(strong, strong)
    disp_w = DISPLAY_NAMES.get(weak, weak)
    unm_tag = "unmatched" if is_unmatched else "matched"

    for ax, title, p_mod, mod_label, color in [
        (ax1, "Ablation (remove from strong)", p_abl, "After removal", "black"),
        (ax2, "Transfer (add to weak)", p_trf, "After transfer", "#2ca02c"),
    ]:
        ax.scatter(p_s, p_w, c="#dddddd", s=15, alpha=0.5, edgecolors="none", zorder=1)
        ax.scatter(p_s[concept_rows], p_w[concept_rows], c="#4a90d9", s=20,
                   alpha=0.7, edgecolors="none", zorder=2,
                   label=f"Concept fires (n={n_concept})")

        if ax == ax1:
            # Ablation: strong model moves left (x-axis)
            ax.scatter(p_mod[concept_rows], p_w[concept_rows], c=color, s=12,
                       alpha=0.8, edgecolors="none", zorder=3, label=mod_label)
        else:
            # Transfer: weak model moves up (y-axis)
            ax.scatter(p_s[concept_rows], p_mod[concept_rows], c=color, s=12,
                       alpha=0.8, edgecolors="none", zorder=3, label=mod_label)

        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{disp_s} P(correct)", fontsize=9)
        ax.set_ylabel(f"{disp_w} P(correct)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{dataset} — f{feature_idx} [Band {feature_band}, {unm_tag}]\n"
                 f"{feature_label}",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Transfer overlay on concept scatter plots")
    parser.add_argument("--pair", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-datasets", type=int, default=2)
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
            continue

        # Check if transfer results exist
        transfer_pair_dir = TRANSFER_DIR / pair
        if not transfer_pair_dir.exists():
            logger.info(f"  No transfer results for {pair}")
            continue

        matrix_data = json.loads(matrix_path.read_text())

        for model_key, data in matrix_data.items():
            features = data["feature_indices"][:args.top_k]

            for fi in features:
                top_ds = data["concept_top_datasets"].get(str(fi), [])
                label = data["feature_labels"].get(str(fi), "")
                band = data["feature_bands"].get(str(fi), -1)
                unmatched = data["feature_unmatched"].get(str(fi), False)

                for ds_info in top_ds[:args.top_datasets]:
                    ds = ds_info["dataset"]

                    fig = plot_ablation_vs_transfer(
                        pair, model_key, fi, ds, label, band, unmatched)

                    if fig is None:
                        continue

                    out_dir = OUTPUT_DIR / pair
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{model_key}_f{fi}_{ds}_transfer.pdf"
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"  {data['display_name']} f{fi} on {ds}: saved")


if __name__ == "__main__":
    main()
