#!/usr/bin/env python3
"""
Table 3: Per-model SAE summary with matching statistics.

Computes: Emb dim, Alive features, Matched %, and Unmatched %FVE
using the noise-floor-filtered MNN matching and concept graph.

Unmatched %FVE = fraction of total SAE activation variance explained
by unmatched features. This is the ceiling for ablation: if unmatched
features carry X% of the variance, ablation can remove at most X%.

Usage:
    python -m scripts.tables.table3.table3
"""

import json
from pathlib import Path

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.matching.utils import (
    compute_sae_activations,
    load_test_embeddings,
)
from scripts.sae.analyze_sae_concepts_deep import load_sae_checkpoint
from scripts.sae.compare_sae_cross_model import (
    DEFAULT_MODELS,
    SAE_FILENAME,
    sae_sweep_dir,
)

SWEEP_DIR = sae_sweep_dir()
GRAPH_PATH = PROJECT_ROOT / "output" / "sae_feature_match_graph_p90.json"
OUTPUT_TEX = Path(__file__).parent / "matching_summary.tex"

# Models to include (excludes HyperFast and Tabula-8B)
MODELS = ["TabPFN", "Mitra", "TabICL", "TabICL-v2", "TabDPT", "CARTE"]

# Map display name -> (model_key, emb_key)
MODEL_LOOKUP = {
    display: (key, emb) for display, key, emb in DEFAULT_MODELS
}


def compute_unmatched_fve(
    model_key: str,
    display_name: str,
    unmatched_indices: set,
) -> float:
    """Compute fraction of SAE activation variance from unmatched features.

    FVE = sum of variance across unmatched features / total variance.
    Activations are pooled across all test datasets.
    """
    ckpt_path = SWEEP_DIR / model_key / SAE_FILENAME
    sae, _, _ = load_sae_checkpoint(ckpt_path)
    sae.eval()

    test_embs = load_test_embeddings(model_key)

    all_acts = []
    for ds in sorted(test_embs.keys()):
        acts = compute_sae_activations(sae, test_embs[ds])
        all_acts.append(acts)
    pooled = np.concatenate(all_acts, axis=0)

    # Variance per feature across all rows
    var_per_feat = pooled.var(axis=0)
    total_var = var_per_feat.sum()

    if total_var == 0:
        return 0.0

    unmatched_var = sum(var_per_feat[i] for i in unmatched_indices if i < len(var_per_feat))
    return unmatched_var / total_var * 100


def main():
    # Load concept graph for tier classifications
    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    alive_counts = graph["model_alive_counts"]
    tier_counts = graph["model_tier_counts"]

    # Collect unmatched feature indices per model
    unmatched_by_model = {m: set() for m in MODELS}
    for key, feat in graph["features"].items():
        model = feat["model"]
        if model in unmatched_by_model and feat["tier"] == "unmatched":
            unmatched_by_model[model].add(feat["feat_idx"])

    # Compute stats per model
    rows = {}
    for display in MODELS:
        model_key, emb_key = MODEL_LOOKUP[display]
        ckpt_path = SWEEP_DIR / model_key / SAE_FILENAME
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        emb_dim = config.get("input_dim", "?")

        alive = alive_counts.get(display, 0)
        tc = tier_counts.get(display, {})
        mnn = tc.get("mnn", 0)
        thresh = tc.get("threshold", 0)
        mnn_pct = mnn / alive * 100 if alive else 0
        thresh_pct = thresh / alive * 100 if alive else 0

        print(f"{display}: emb_dim={emb_dim}, alive={alive}, "
              f"MNN={mnn} ({mnn_pct:.1f}%), threshold={thresh} ({thresh_pct:.1f}%), "
              f"unmatched={len(unmatched_by_model[display])}")

        fve = compute_unmatched_fve(
            model_key, display, unmatched_by_model[display]
        )
        print(f"  unmatched FVE = {fve:.1f}%")

        rows[display] = {
            "emb_dim": emb_dim,
            "alive": alive,
            "mnn_pct": mnn_pct,
            "thresh_pct": thresh_pct,
            "unmatched_fve": fve,
        }

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Per-model SAE summary. \emph{Alive}: concepts that fire on at "
        r"least one test row. \emph{MNN \%}: mutual nearest neighbor matches above "
        r"the per-pair p90 noise floor. \emph{Threshold \%}: additional features "
        r"with a best correlate above the noise floor (one-sided). "
        r"\emph{Unmatched \%FVE}: activation variance in unmatched "
        r"concepts---the ceiling for ablation.}"
    )
    lines.append(r"\label{tab:matching_summary}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODELS) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(MODELS) + r" \\")
    lines.append(r"\midrule")

    # Emb dim
    vals = [str(rows[m]["emb_dim"]) if m in rows else "---" for m in MODELS]
    lines.append("Emb dim & " + " & ".join(vals) + r" \\")

    # Alive
    vals = [f"{rows[m]['alive']:,}" if m in rows else "---" for m in MODELS]
    lines.append("Alive & " + " & ".join(vals) + r" \\")

    # MNN %
    vals = [f"{rows[m]['mnn_pct']:.1f}" if m in rows else "---" for m in MODELS]
    lines.append(r"MNN \% & " + " & ".join(vals) + r" \\")

    # Threshold %
    vals = [f"{rows[m]['thresh_pct']:.1f}" if m in rows else "---" for m in MODELS]
    lines.append(r"Threshold \% & " + " & ".join(vals) + r" \\")

    # Unmatched %FVE
    vals = [f"{rows[m]['unmatched_fve']:.1f}" if m in rows else "---" for m in MODELS]
    lines.append(r"Unmatched \%FVE & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    OUTPUT_TEX.write_text(tex + "\n")
    print(f"\nSaved to {OUTPUT_TEX}")
    print(tex)


if __name__ == "__main__":
    main()
