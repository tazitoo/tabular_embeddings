#!/usr/bin/env python3
"""Plot row-level intervention diagram from precomputed data.

Reads row_intervention_data.json (produced by compute_row_intervention_data.py).
No GPU needed.

Usage:
    python scripts/paper/sec4/row_intervention_figure.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "row_intervention_data.json"
OUTPUT_DIR = Path(__file__).parent


def main():
    with open(DATA_FILE) as f:
        d = json.load(f)

    p_strong = d["p_strong"]
    p_weak = d["p_weak"]
    ab_feats = d["ablation"]["features"]
    ab_preds = d["ablation"]["step_preds"]
    tr_feats = d["transfer"]["features"]
    tr_preds = d["transfer"]["step_preds"]
    pool = [(p["feature"], p["importance"]) for p in d["pool"]]

    ablation_feats = set(ab_feats)
    transfer_feats = set(tr_feats)
    pool_dict = {f: imp for f, imp in pool}

    c = "black"
    fs_label = 7.5
    pred_xlim = (0.75, 0.95)
    pool_xlim = (-0.03, 0.04)
    pred_xlabel = "$P(\\mathrm{good\\ credit})$"
    y = 0.5

    fig, axes = plt.subplots(3, 1, figsize=(7, 2.2),
                             gridspec_kw={"hspace": 1.3,
                                          "height_ratios": [1, 1, 1]})

    def setup_axis(ax, xlim, xlabel):
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for sp in ["top", "left", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_position(("axes", 0.5))
        ax.tick_params(axis="x", labelsize=7, pad=2)
        ax.set_xlabel(xlabel, fontsize=7, labelpad=3)

    # ── Ablation ──
    ax = axes[0]
    setup_axis(ax, pred_xlim, pred_xlabel)
    ax.plot(p_strong, y, "o", color=c, ms=7, zorder=10, clip_on=False)
    ax.plot(p_weak, y, "o", color=c, ms=7, zorder=10, clip_on=False)

    prev = p_strong
    for feat, pred in zip(ab_feats, ab_preds):
        delta = pred - prev
        mid = (prev + pred) / 2
        ax.annotate("", xy=(pred, y), xytext=(prev, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5,
                                    shrinkA=4, shrinkB=3))
        ax.plot(pred, y, "D", color=c, ms=4, zorder=8, clip_on=False)
        ax.annotate(f"$f_{{{feat}}}$  {delta:+.03f}", (mid, y),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=fs_label, color=c)
        prev = pred

    # ── Pool ──
    ax = axes[1]
    setup_axis(ax, pool_xlim, "LOO importance")
    for feat, imp in pool:
        selected = feat in ablation_feats or feat in transfer_feats
        alpha = 1.0 if selected else 0.2
        ax.plot(imp, y, "o", color=c, ms=3, zorder=3, clip_on=False,
                alpha=alpha)
        if selected:
            label = f"$f_{{{feat}}}$"
            ax.annotate(label, (imp, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=fs_label, color=c, fontweight="bold")
    ax.axvline(0, color=c, lw=0.4, ls=":", alpha=0.3)

    # ── Transfer ──
    ax = axes[2]
    setup_axis(ax, pred_xlim, pred_xlabel)
    ax.plot(p_weak, y, "o", color=c, ms=7, zorder=10, clip_on=False)
    ax.plot(p_strong, y, "o", color=c, ms=7, zorder=10, clip_on=False)

    if tr_preds:
        prev = p_weak
        for i, (feat, pred) in enumerate(zip(tr_feats, tr_preds)):
            delta = pred - prev
            mid = (prev + pred) / 2
            ax.annotate("", xy=(pred, y), xytext=(prev, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5,
                                        shrinkA=4, shrinkB=3))
            ax.plot(pred, y, "D", color=c, ms=4, zorder=8, clip_on=False)
            if i == 1:  # f_86: right-align to its diamond (prev is f_86's diamond)
                ax.annotate(f"$f_{{{feat}}}$  {delta:+.03f}", (prev, y),
                            xytext=(0, 8), textcoords="offset points",
                            ha="right", fontsize=fs_label, color=c)
            elif i == 2:  # f_36: raised to avoid overlap with f_86
                ax.annotate(f"$f_{{{feat}}}$  {delta:+.03f}", (pred, y),
                            xytext=(0, 24), textcoords="offset points",
                            ha="center", fontsize=fs_label, color=c,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2",
                                            color="#999999",
                                            lw=0.6, alpha=0.2, linestyle="-",
                                            shrinkA=1, shrinkB=2))
            else:
                ax.annotate(f"$f_{{{feat}}}$  {delta:+.03f}", (mid, y),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=fs_label, color=c)
            prev = pred

    # ── Cross-panel arrows ──
    # Use ConnectionPatch which handles cross-axes coordinates correctly
    # regardless of bbox_inches="tight"
    from matplotlib.patches import ConnectionPatch
    arrow_col = "#999999"
    arrow_alpha = 0.2

    cross_arrows = []
    # Compute midpoints for ablation labels (where the f_* text sits)
    ab_mids = []
    prev = p_strong
    for pred in ab_preds:
        ab_mids.append((prev + pred) / 2)
        prev = pred

    # Small x-offset to start arrows just left of the label text
    label_offset = 0.001  # in pool axis data coords

    for i, feat in enumerate(ab_feats):
        if feat not in pool_dict:
            continue
        imp = pool_dict[feat]
        mid = ab_mids[i]
        start_x = imp - label_offset
        con = ConnectionPatch(
            xyA=(start_x, 0.9), coordsA=axes[1].transData,
            xyB=(mid, 0.7), coordsB=axes[0].transData,
            arrowstyle="-|>", color=arrow_col, lw=0.6,
            alpha=arrow_alpha, linestyle="-")
        fig.add_artist(con)
        cross_arrows.append(con)

    # Compute label anchor positions for transfer annotations
    # f_92 (i=0): centered on mid
    # f_86 (i=1): right-aligned to prev (label center is left of prev)
    # f_36 (i=2): centered on mid, raised
    tr_label_xs = []
    prev_t = p_weak
    for i, pred in enumerate(tr_preds):
        mid = (prev_t + pred) / 2
        if i == 1:
            tr_label_xs.append(prev_t)  # right-aligned to prev, arrow to left of that
        else:
            tr_label_xs.append(mid)
        prev_t = pred

    for i, feat in enumerate(tr_feats):
        if feat not in pool_dict:
            continue
        imp = pool_dict[feat]
        label_x = tr_label_xs[i]
        if i == 0:  # f_92: shared with ablation, same start point (left of label)
            start_x = imp - label_offset
            end_x = label_x + 0.012  # right of centered label
            end_y = 1.2
        elif i == 1:  # f_86: start right of label, end left of right-aligned text
            start_x = imp + label_offset
            end_x = label_x - 0.025
            end_y = 1.2
        else:  # f_36: start right of label, end left of raised centered text
            start_x = imp + label_offset
            end_x = label_x - 0.012
            end_y = 1.6
        con = ConnectionPatch(
            xyA=(start_x, 0.9), coordsA=axes[1].transData,
            xyB=(end_x, end_y), coordsB=axes[2].transData,
            arrowstyle="-|>", color=arrow_col, lw=0.6,
            alpha=arrow_alpha, linestyle="-")
        fig.add_artist(con)
        cross_arrows.append(con)

    plt.savefig(OUTPUT_DIR / "row_intervention_figure.pdf",
                bbox_inches="tight", dpi=300,
                bbox_extra_artists=cross_arrows)
    plt.savefig(OUTPUT_DIR / "row_intervention_figure.png",
                bbox_inches="tight", dpi=150,
                bbox_extra_artists=cross_arrows)
    print("Saved row_intervention_figure.pdf/png")


if __name__ == "__main__":
    main()
