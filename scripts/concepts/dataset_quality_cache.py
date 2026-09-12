#!/usr/bin/env python3
"""Helpers for dataset-quality caching and dataset selection.

The cache ranks each (model, feature, dataset) pair for labeling usefulness.
It is intended to improve which datasets are chosen upstream for contrastive
example generation without changing the downstream labeling prompts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

from scripts.round_paths import QUALITY_CACHE_FILE

CACHE_VERSION = 3
DEFAULT_CACHE_PATH = QUALITY_CACHE_FILE
CACHE_DIR = DEFAULT_CACHE_PATH.parent

DEFAULT_SCORE_WEIGHTS = {
    "activation_support": 0.30,
    "activation_tail": 0.20,
    "contrastive_separation": 0.30,
    "prediction_usefulness": 0.20,
}

DEFAULT_SCORE_CONFIG = {
    "min_active_rows": 6,
    "contrastive_active_rows": 6,
    "contrastive_inactive_rows": 6,
    "validator_active_rows": 10,
    "validator_inactive_rows": 10,
    "feasibility_margin_rows": 0,
    "top_quantile": 0.99,
    "low_conf_threshold": 0.60,
    "prediction_std_floor": 1e-6,
    "diversity_tie_margin": 0.03,
}


def load_quality_cache(path: Optional[Path] = None) -> Optional[dict]:
    """Load the global dataset-quality cache if it exists."""
    cache_path = Path(path) if path is not None else DEFAULT_CACHE_PATH
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f)


def cache_entry_for_feature(cache: dict, model: str, feat_idx: int) -> Optional[dict]:
    """Return the per-feature cache block for one model."""
    if not cache:
        return None
    model_block = cache.get("models", {}).get(model)
    if not model_block:
        return None
    return model_block.get("features", {}).get(str(int(feat_idx)))


def minmax_scale(values: Iterable[float]) -> list[float]:
    """Min-max scale a list; return zeros when the range collapses."""
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def normalized_entropy_from_counts(counts: Iterable[float]) -> float:
    """Return entropy normalized to [0, 1]."""
    vals = [float(v) for v in counts if float(v) > 0]
    if len(vals) <= 1:
        return 0.0
    total = sum(vals)
    probs = [v / total for v in vals]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    max_ent = math.log(len(probs))
    if max_ent <= 0:
        return 0.0
    return float(ent / max_ent)


def _activation_shape_bucket(entry: dict) -> str:
    band = entry.get("activation_band_mass", {})
    pct_top = float(band.get("pct_top", 0.0))
    pct_p90 = float(band.get("pct_p90", 0.0))
    pct_p80 = float(band.get("pct_p80", 0.0))
    pct_p70 = float(band.get("pct_p70", 0.0))
    if pct_top >= max(pct_p90, pct_p80, pct_p70):
        return "top_heavy"
    if pct_p90 >= max(pct_p80, pct_p70):
        return "upper_tail"
    if pct_p80 >= pct_p70:
        return "broad_tail"
    return "diffuse"


def _prediction_spread_bucket(entry: dict) -> str:
    score = float(entry.get("quality_components", {}).get("prediction_usefulness", 0.0))
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _diversity_bonus(selected: list[dict], candidate: dict) -> tuple[int, int, int]:
    if not selected:
        return (1, 1, 1)
    task_type = candidate.get("task_type", "unknown")
    shape = _activation_shape_bucket(candidate)
    pred_bucket = _prediction_spread_bucket(candidate)
    seen_tasks = {s.get("task_type", "unknown") for s in selected}
    seen_shapes = {_activation_shape_bucket(s) for s in selected}
    seen_pred = {_prediction_spread_bucket(s) for s in selected}
    return (
        1 if task_type not in seen_tasks else 0,
        1 if shape not in seen_shapes else 0,
        1 if pred_bucket not in seen_pred else 0,
    )


def select_top_datasets(
    feature_entries: dict,
    max_datasets: int,
    diversity_tie_margin: float = DEFAULT_SCORE_CONFIG["diversity_tie_margin"],
) -> list[str]:
    """Select top datasets by quality score with a mild diversity tie-break."""
    candidates = []
    for dataset, entry in feature_entries.items():
        if entry.get("filtered_out"):
            continue
        candidates.append((dataset, entry))
    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (
            -float(item[1].get("labeling_quality_score", 0.0)),
            -int(item[1].get("n_active", 0)),
            item[0],
        )
    )

    selected: list[tuple[str, dict]] = []
    remaining = candidates[:]
    while remaining and len(selected) < max_datasets:
        top_score = float(remaining[0][1].get("labeling_quality_score", 0.0))
        tie_group = [
            (dataset, entry)
            for dataset, entry in remaining
            if top_score - float(entry.get("labeling_quality_score", 0.0)) <= diversity_tie_margin
        ]
        prior_entries = [entry for _, entry in selected]
        chosen = max(
            tie_group,
            key=lambda item: (
                _diversity_bonus(prior_entries, item[1]),
                float(item[1].get("labeling_quality_score", 0.0)),
                int(item[1].get("n_active", 0)),
                item[0],
            ),
        )
        selected.append(chosen)
        remaining = [(dataset, entry) for dataset, entry in remaining if dataset != chosen[0]]
    return [dataset for dataset, _ in selected]
