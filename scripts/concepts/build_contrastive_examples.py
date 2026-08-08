#!/usr/bin/env python3
"""Build contrastive example CSVs for SAE feature labeling.

For each (model, feature, dataset): find stratified activating rows and
matched non-activating rows. Save as CSV with the raw data values,
activation diagnostics, and per-row stats (PMI, surprise, compression).

Also saves a preprocessing_context.json describing how each model
transforms raw data before embedding, so labeling agents understand
what the SAE actually sees.

Output:
    output/contrastive_examples/{model}/f{feat}_{dataset}.csv
    output/contrastive_examples/{model}/preprocessing_context.json

Usage:
    python -m scripts.concepts.build_contrastive_examples \
        --model mitra --features 6 11 36 86 92

    # All features for a model (slow)
    python -m scripts.concepts.build_contrastive_examples --model mitra --all
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from scripts.concepts.dataset_quality_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SCORE_CONFIG,
    cache_entry_for_feature,
    load_quality_cache,
    select_top_datasets,
)
from scripts.concepts.row_source import (
    AUTO_SOURCE,
    DEFAULT_ROW_SOURCE_MODE,
    SAE_TEST_SOURCE,
    VALID_ROW_SOURCE_MODES,
    explicit_row_source_or_default,
    feature_block_row_source,
    load_row_source_baseline_predictions,
    load_row_source_embeddings,
    load_row_source_row_indices,
)
from scripts._project_root import PROJECT_ROOT
from scripts.intervention.intervene_lib import (
    load_sae, load_test_embeddings, SPLITS_PATH,
)

SAE_DATA_DIR = PROJECT_ROOT / "output" / "sae_training_round10"
PMI_DIR = PROJECT_ROOT / "output" / "pmi_cache"
SURPRISE_DIR = PROJECT_ROOT / "output" / "surprise_cache"
COMPRESSION_DIR = PROJECT_ROOT / "output" / "compression_cache"
PYMFE_PATH = PROJECT_ROOT / "output" / "pymfe_tabarena_cache.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "contrastive_examples"
VALID_CONTRAST_POLICIES = {"nearest_non_d", "cofire_hard_negative", "stratified_nonfire"}
DEFAULT_CONTRAST_POLICY = "nearest_non_d"
DEFAULT_COFIRE_TOP_K = 8
STRATIFIED_NONFIRE_BANDS = ("near_boundary", "decoder_aligned", "far_control")
VALID_ACTIVATING_POLICIES = {"activation_bands", "scored_tiers"}
DEFAULT_ACTIVATING_POLICY = "activation_bands"
VALID_VALIDATOR_POLICIES = {"random", "stratified_nonfire", "scored_tiers"}
DEFAULT_VALIDATOR_POLICY = "random"

PREPROCESSING_CONTEXT = {
    "mitra": {
        "model": "Mitra",
        "architecture": "2D attention transformer (72M params), in-context learning with 1-epoch finetune",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator: auto-detects column types, "
            "ordinal-encodes categoricals (codes as integers), passes numerics through",
            "Median imputation for all NaN values (not NaN-safe)",
            "Category codes converted to float32 (ordinal, not one-hot)",
            "SAE is trained on the output of layer 12 (final_layer_norm)",
        ],
        "implication": (
            "The SAE sees ordinal-encoded categoricals and median-imputed numerics. "
            "A feature that fires on 'high values in column X' in the raw data may "
            "actually be responding to the ordinal code or the imputed median pattern. "
            "Missing values are replaced by the training-set median before the model sees them."
        ),
    },
    "tabpfn": {
        "model": "TabPFN",
        "architecture": "Transformer (12M params), pure in-context learning (no finetuning)",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator (same as Mitra)",
            "NaN-safe: missing values preserved as NaN (TabPFN handles them natively)",
            "Category codes preserved as integers with NaN for missing",
            "SAE is trained on the output of layer 16 (of 24 transformer layers)",
        ],
        "implication": (
            "The SAE sees the same ordinal encoding as Mitra but with NaN preserved. "
            "Features responding to missingness patterns are possible."
        ),
    },
    "tabicl": {
        "model": "TabICL",
        "architecture": "Column-then-row transformer, in-context learning",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "Median imputation for NaN (not NaN-safe)",
            "SAE is trained on the output of layer 10 (of 12 blocks)",
        ],
        "implication": (
            "Same preprocessing as Mitra. Column-level attention means the model "
            "can learn per-feature patterns before combining across features."
        ),
    },
    "tabicl_v2": {
        "model": "TabICL-v2",
        "architecture": "Column-then-row transformer v2, in-context learning",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "Median imputation for NaN",
            "SAE is trained on the output of layer 12 (of 12 blocks)",
        ],
        "implication": "Same preprocessing as TabICL/Mitra.",
    },
    "tabdpt": {
        "model": "TabDPT",
        "architecture": "Transformer with retrieval augmentation",
        "preprocessing": [
            "AutoGluon AutoMLPipelineFeatureGenerator",
            "NaN-safe: missing values preserved",
            "SAE is trained on the output of layer 14 (of 18 encoder layers)",
        ],
        "implication": (
            "NaN-safe like TabPFN. Retrieval augmentation means the model's "
            "representation incorporates nearest-neighbor context from training data."
        ),
    },
    "carte": {
        "model": "CARTE",
        "architecture": "Graph transformer (star graph GNN), trains per-dataset",
        "preprocessing": [
            "CARTE handles its own preprocessing internally",
            "Builds a star graph per row: central node + one node per feature",
            "Uses pretrained fastText embeddings for column names",
            "RobustScaler applied to numeric features",
            "SAE is trained on the output of layer 1 (of GNN layers)",
        ],
        "implication": (
            "CARTE's representation is fundamentally different: it embeds column "
            "names via language model, so the same numeric value in different columns "
            "gets different representations. The SAE sees graph-level embeddings, "
            "not tabular features directly."
        ),
    },
    "hyperfast": {
        "model": "HyperFast",
        "architecture": "Hypernetwork that generates MLP weights",
        "preprocessing": [
            "Ordinal encoding for categoricals",
            "Mean imputation for missing numerics, mode for missing categoricals",
            "StandardScaler: zero mean, unit variance per feature",
            "SAE is trained on the output of layer 2",
        ],
        "implication": (
            "Standardized features mean the SAE sees z-scores, not raw values. "
            "A feature firing on 'high values' means high z-score (>2σ from mean)."
        ),
    },
}


def load_row_stats():
    """Load per-row PMI, surprise, compression caches."""
    caches = {}
    for name, dirp, key in [
        ("pmi", PMI_DIR, "row_pmi"),
        ("surprise", SURPRISE_DIR, "row_surprise"),
        ("compression", COMPRESSION_DIR, "row_compress_delta"),
    ]:
        summary_path = dirp / f"{name}_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for ds in summary.get("datasets", summary.keys()):
            ds_name = ds if isinstance(ds, str) else ds
            npz_path = dirp / f"{ds_name}.npz"
            if npz_path.exists():
                vals = np.load(npz_path, allow_pickle=True)
                if key in vals:
                    caches.setdefault(ds_name, {})[name] = vals[key]
    return caches


def load_baseline_predictions(model: str, dataset: str, row_source: str = SAE_TEST_SOURCE) -> Optional[dict]:
    """Back-compat wrapper around the shared row-source prediction loader."""
    return load_row_source_baseline_predictions(model, dataset, row_source)


def _stratified_activating(active_idx, active_acts, per_band=2, rng=None):
    """Sample `per_band` rows from each of 3 activation strata: top, p90, p80.

    Returns (indices, bands) where indices are into the embedding matrix and
    bands is a parallel list of stratum names ('top'|'p90'|'p80'). Falls back
    to top-(3*per_band) labelled 'top' if too few activating rows to stratify.
    """
    if rng is None:
        rng = np.random.default_rng(13)
    n_target = 3 * per_band
    if len(active_idx) <= n_target:
        order = np.argsort(-active_acts)
        idx = active_idx[order]
        return idx, ["top"] * len(idx)

    order = np.argsort(-active_acts)  # descending by activation
    sorted_idx = active_idx[order]
    sorted_acts = active_acts[order]

    top = sorted_idx[:per_band]

    p90_val = np.percentile(active_acts, 90)
    p80_val = np.percentile(active_acts, 80)

    used = set(int(i) for i in top)
    p90 = _pick_near(sorted_idx, sorted_acts, p90_val, per_band, used, rng)
    used.update(int(i) for i in p90)
    p80 = _pick_near(sorted_idx, sorted_acts, p80_val, per_band, used, rng)

    indices = np.concatenate([top, p90, p80])
    bands = ["top"] * per_band + ["p90"] * per_band + ["p80"] * per_band
    return indices, bands


def _pick_near(sorted_idx, sorted_acts, target_val, k, used, rng):
    """Pick `k` distinct indices whose activation is closest to `target_val`."""
    dist = np.abs(sorted_acts - target_val)
    order = np.argsort(dist)
    picks = []
    for j in order:
        ix = int(sorted_idx[j])
        if ix in used:
            continue
        picks.append(ix)
        if len(picks) == k:
            break
    return np.array(picks, dtype=sorted_idx.dtype)


def _take_ranked_candidates(ranked_idx, k, used):
    """Take up to `k` indices from ranked_idx, skipping already-used rows."""
    if k <= 0:
        return []
    picks = []
    for ix in ranked_idx:
        ix = int(ix)
        if ix in used:
            continue
        picks.append(ix)
        used.add(ix)
        if len(picks) == k:
            break
    return picks


def _decoder_dictionary(sae) -> np.ndarray:
    """Return the effective decoder dictionary as (hidden_dim, input_dim)."""
    config = getattr(sae, "config", None)
    sparsity_type = getattr(config, "sparsity_type", "")
    if "archetypal" in sparsity_type and getattr(sae, "archetype_logits", None) is not None:
        return sae.get_archetypal_dictionary().detach().cpu().numpy()
    if getattr(config, "tied_weights", False):
        return sae.W_enc.detach().cpu().numpy()
    return sae.W_dec.detach().cpu().numpy().T


def _feature_decoder_cosine(sae, emb_t: torch.Tensor, feat_idx: int) -> np.ndarray:
    """Cosine between SAE input residuals and the feature's decoder direction."""
    decoder = _decoder_dictionary(sae)
    direction = decoder[feat_idx]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    residual = (emb_t - sae.b_dec).detach().cpu().numpy()
    residual_norm = residual / (np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8)
    return residual_norm @ direction


def _select_stratified_nonfire(
    *,
    inactive_idx: np.ndarray,
    pre_act_np: np.ndarray,
    decoder_cosine: np.ndarray,
    feat_idx: int,
    n: int,
) -> tuple[np.ndarray, list[str]]:
    """Select non-firing rows from near-boundary, decoder-aligned, and far-control bands."""
    if n <= 0 or len(inactive_idx) == 0:
        return np.array([], dtype=int), []

    n_bands = len(STRATIFIED_NONFIRE_BANDS)
    base = n // n_bands
    extras = n % n_bands
    quotas = {
        band: base + (1 if i < extras else 0)
        for i, band in enumerate(STRATIFIED_NONFIRE_BANDS)
    }

    inactive_pre = pre_act_np[inactive_idx, feat_idx]
    inactive_cos = decoder_cosine[inactive_idx]
    near_ranked = inactive_idx[np.argsort(-inactive_pre)]
    aligned_ranked = inactive_idx[np.argsort(-inactive_cos)]
    far_score = (
        (inactive_pre - np.nanmean(inactive_pre)) / (np.nanstd(inactive_pre) + 1e-8)
        + (inactive_cos - np.nanmean(inactive_cos)) / (np.nanstd(inactive_cos) + 1e-8)
    )
    far_ranked = inactive_idx[np.argsort(far_score)]

    used: set[int] = set()
    band_to_ranked = {
        "near_boundary": near_ranked,
        "decoder_aligned": aligned_ranked,
        "far_control": far_ranked,
    }
    selected: list[int] = []
    bands: list[str] = []
    for band in STRATIFIED_NONFIRE_BANDS:
        picks = _take_ranked_candidates(band_to_ranked[band], quotas[band], used)
        selected.extend(picks)
        bands.extend([band] * len(picks))

    if len(selected) < n:
        fallback = _take_ranked_candidates(near_ranked, n - len(selected), used)
        selected.extend(fallback)
        bands.extend(["near_boundary"] * len(fallback))
    return np.array(selected, dtype=int), bands


def _percentile_scores(values: np.ndarray) -> np.ndarray:
    """Deterministic rank-percentile scores in [0, 1], stable under ties."""
    values = np.asarray(values)
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.float64)
    if len(values) <= 1:
        ranks.fill(1.0)
        return ranks
    ranks[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1)
    return ranks


def _tier_quotas(n: int, tiers: tuple[str, ...]) -> dict[str, int]:
    """Split n rows across tiers, assigning remainder to harder boundary tiers first."""
    if n <= 0:
        return {tier: 0 for tier in tiers}
    base = n // len(tiers)
    extras = n % len(tiers)
    return {
        tier: base + (1 if i < extras else 0)
        for i, tier in enumerate(tiers)
    }


def _take_ranked_positions(ranked: np.ndarray, k: int, used: set[int]) -> list[int]:
    picks: list[int] = []
    if k <= 0:
        return picks
    for ix in ranked:
        ix = int(ix)
        if ix in used:
            continue
        picks.append(ix)
        used.add(ix)
        if len(picks) == k:
            break
    return picks


def _select_scored_tiers(
    *,
    candidates: np.ndarray,
    scores: np.ndarray,
    n: int,
    positive: bool,
) -> tuple[np.ndarray, list[str]]:
    """Deterministically select rows from score-defined difficulty tiers.

    Positives are strong/medium/weak by active-likeness score. Negatives are
    hard/medium/easy by the same score. Medium rows are nearest the candidate
    score median.
    """
    candidates = np.asarray(candidates, dtype=int)
    if n <= 0 or len(candidates) == 0:
        return np.array([], dtype=int), []
    n = min(n, len(candidates))
    cand_scores = scores[candidates]
    if positive:
        tiers = ("positive_weak", "positive_medium", "positive_strong")
        ranked_by_tier = {
            "positive_weak": candidates[np.lexsort((candidates, cand_scores))],
            "positive_medium": candidates[np.lexsort((candidates, np.abs(cand_scores - np.median(cand_scores))))],
            "positive_strong": candidates[np.lexsort((candidates, -cand_scores))],
        }
        fallback_order = ("positive_weak", "positive_medium", "positive_strong")
    else:
        tiers = ("negative_hard", "negative_medium", "negative_easy")
        ranked_by_tier = {
            "negative_hard": candidates[np.lexsort((candidates, -cand_scores))],
            "negative_medium": candidates[np.lexsort((candidates, np.abs(cand_scores - np.median(cand_scores))))],
            "negative_easy": candidates[np.lexsort((candidates, cand_scores))],
        }
        fallback_order = ("negative_hard", "negative_medium", "negative_easy")

    quotas = _tier_quotas(n, tiers)
    used: set[int] = set()
    selected: list[int] = []
    selected_tiers: list[str] = []
    for tier in tiers:
        picks = _take_ranked_positions(ranked_by_tier[tier], quotas[tier], used)
        selected.extend(picks)
        selected_tiers.extend([tier] * len(picks))

    for tier in fallback_order:
        if len(selected) >= n:
            break
        picks = _take_ranked_positions(ranked_by_tier[tier], n - len(selected), used)
        selected.extend(picks)
        selected_tiers.extend([tier] * len(picks))
    return np.array(selected, dtype=int), selected_tiers


def _classify_column(series: pd.Series) -> str:
    """Classify column as 'numeric', 'binary', or 'categorical'.

    String-typed columns (including pandas `string[pyarrow]` / `ArrowStringArray`
    used by newer TabArena cache builds) must bucket as categorical; they are
    neither `object` dtype nor `CategoricalDtype`, so the older check
    `is_categorical_dtype(series) or series.dtype == object` silently missed
    them and they were then annotated with `(pXX)` percentiles from an
    alphabetical sort — garbage.
    """
    if (isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)):
        return "categorical"
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:
        return "categorical"
    if n_unique <= 2:
        return "binary"
    if n_unique <= 20 and pd.api.types.is_integer_dtype(series):
        return "categorical"  # small-cardinality integers: likely ordinal-encoded
    return "numeric"


def _compute_marginals(X_train: pd.DataFrame) -> dict:
    """Per-column marginal stats from the SAE training split."""
    marginals = {}
    for col in X_train.columns:
        col_type = _classify_column(X_train[col])
        if col_type == "numeric":
            vals = X_train[col].dropna().to_numpy()
            marginals[col] = {
                "type": "numeric",
                "sorted": np.sort(vals),
                "n": int(len(vals)),
            }
        else:
            freq = X_train[col].value_counts(dropna=False, normalize=True).to_dict()
            marginals[col] = {"type": col_type, "freq": freq}
    return marginals


def _annotate_value(col: str, val, marginals: dict) -> str:
    """Attach marginal-distribution position to a raw cell value."""
    m = marginals.get(col)
    if pd.isna(val):
        return "NaN"
    if m is None:
        return str(val)
    if m["type"] == "numeric":
        sorted_vals = m["sorted"]
        if len(sorted_vals) == 0:
            return _fmt_num(val)
        pct = int(round(100 * np.searchsorted(sorted_vals, val, side="right") / len(sorted_vals)))
        return f"{_fmt_num(val)} (p{pct})"
    freq = m["freq"].get(val, 0.0)
    return f"{val} (freq {freq:.2f})"


def _fmt_num(val) -> str:
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    try:
        return f"{float(val):.4g}"
    except (ValueError, TypeError):
        return str(val)


def _target_summary(y_train, task_type: str) -> dict:
    """Target distribution from the SAE training split."""
    y = np.asarray(y_train).ravel()
    if task_type == "classification":
        unique, counts = np.unique(y, return_counts=True)
        return {
            "task": "classification",
            "n": int(y.size),
            "class_freq": {str(u): round(float(c / y.size), 4) for u, c in zip(unique, counts)},
        }
    return {
        "task": "regression",
        "n": int(y.size),
        "mean": round(float(np.mean(y)), 4),
        "std": round(float(np.std(y)), 4),
        "p25": round(float(np.percentile(y, 25)), 4),
        "p50": round(float(np.percentile(y, 50)), 4),
        "p75": round(float(np.percentile(y, 75)), 4),
    }


def _select_contrast_indices(
    *,
    feat_idx: int,
    top_idx: np.ndarray,
    emb: np.ndarray,
    acts_np: np.ndarray,
    pre_act_np: Optional[np.ndarray],
    decoder_cosine: Optional[np.ndarray],
    active_mask: np.ndarray,
    contrast_policy: str,
    cofire_top_k: int,
) -> tuple[np.ndarray, list[str]]:
    """Pick matched contrast rows for the activating sample.

    `nearest_non_d` reproduces the current behavior: nearest non-activating
    rows in embedding space.

    `cofire_hard_negative` prefers non-activating rows that preserve as many
    of the activating row's strongest co-firing features as possible, then
    breaks ties by embedding distance. This asks a harder "what makes D
    unique?" question than any-nearest non-D negatives.

    `stratified_nonfire` samples three non-firing bands:
      - near_boundary: largest pre-activation for this feature without firing
      - decoder_aligned: highest cosine to the feature decoder direction
      - far_control: lowest pre-activation/decoder-cosine controls
    """
    inactive_idx = np.where(~active_mask)[0]
    if len(inactive_idx) == 0:
        return np.array([], dtype=int), []

    top_emb = emb[top_idx]
    inactive_emb = emb[inactive_idx]
    dists = cdist(top_emb, inactive_emb, metric="cosine")
    contrast_set = set()
    n_contrast = len(top_idx)

    if contrast_policy == "nearest_non_d":
        for i in range(len(top_idx)):
            nearest = inactive_idx[np.argsort(dists[i])[:n_contrast]]
            contrast_set.update(nearest.tolist())
        contrast_idx = np.array(sorted(contrast_set)[:n_contrast], dtype=int)
        return contrast_idx, ["contrast"] * len(contrast_idx)

    if contrast_policy == "stratified_nonfire":
        if pre_act_np is None or decoder_cosine is None:
            raise ValueError("stratified_nonfire requires pre_act_np and decoder_cosine")
        return _select_stratified_nonfire(
            inactive_idx=inactive_idx,
            pre_act_np=pre_act_np,
            decoder_cosine=decoder_cosine,
            feat_idx=feat_idx,
            n=n_contrast,
        )

    if contrast_policy != "cofire_hard_negative":
        raise ValueError(
            f"contrast_policy={contrast_policy!r} not one of {sorted(VALID_CONTRAST_POLICIES)}"
        )

    fired = acts_np > 0
    inactive_fired = fired[inactive_idx]
    for i, ri in enumerate(top_idx):
        cofire = np.flatnonzero(fired[ri])
        cofire = cofire[cofire != feat_idx]
        if cofire.size:
            cofire_strength = acts_np[ri, cofire]
            take = min(cofire_top_k, cofire.size) if cofire_top_k > 0 else cofire.size
            strongest = np.argsort(-cofire_strength)[:take]
            cofire = cofire[strongest]
            overlap = inactive_fired[:, cofire].sum(axis=1)
        else:
            overlap = np.zeros(len(inactive_idx), dtype=int)
        ranked = np.lexsort((dists[i], -overlap))
        nearest = inactive_idx[ranked[:n_contrast]]
        contrast_set.update(nearest.tolist())
    contrast_idx = np.array(sorted(contrast_set)[:n_contrast], dtype=int)
    return contrast_idx, ["contrast"] * len(contrast_idx)


def build_contrastive(model, feat_idx, dataset, sae, test_embs, splits,
                      row_stats, top_k=5, device="cpu", per_band=2,
                      activating_policy: str = DEFAULT_ACTIVATING_POLICY,
                      contrast_policy: str = DEFAULT_CONTRAST_POLICY,
                      cofire_top_k: int = DEFAULT_COFIRE_TOP_K,
                      row_source: str = SAE_TEST_SOURCE):
    """Build contrastive examples for one (model, feature, dataset).

    Activating rows:
        - `activation_bands`: stratified sample — `per_band` each from top,
          p90, p80 activation bands (6 rows by default)
        - `scored_tiers`: deterministic weak/medium/strong positives by
          active-likeness score percentile

    Contrast rows:
        - `nearest_non_d`: nearest non-activating rows in embedding space
        - `cofire_hard_negative`: non-activating rows that preserve the
          activating row's strongest co-firing features, tie-broken by
          embedding distance
        - `stratified_nonfire`: non-activating near-boundary,
          decoder-aligned, and far-control rows
    Size matched to the activating set.

    Returns a list of dicts, each with:
        label: "activating" or "contrast"
        activation: float
        row_idx: int
        + all raw data columns
        + pmi, surprise, compression (if available)
    """
    from data.extended_loader import load_tabarena_dataset

    if dataset not in test_embs:
        return []

    emb = test_embs[dataset]
    n_emb = len(emb)
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        acts, pre_act = sae.encode(emb_t, return_pre_act=True)
    feat_acts = acts[:, feat_idx].cpu().numpy()
    acts_np = acts.cpu().numpy()
    pre_act_np = pre_act.cpu().numpy()
    decoder_cosine = _feature_decoder_cosine(sae, emb_t, feat_idx)
    active_likeness_score = _percentile_scores(pre_act_np[:, feat_idx])

    # Load raw data. The SAE test NPZ stores absolute row indices under
    # `row_indices` — use those directly so X_test, y_test, and per-row stats
    # align with the embeddings the SAE actually saw (test_indices[:n_emb]
    # is WRONG: the SAE training split shuffles test_indices before picking).
    ds_splits = splits.get(dataset)
    if not ds_splits:
        return []
    result = load_tabarena_dataset(dataset, max_samples=999999)
    if result is None:
        return []
    X, y = result[0], result[1]
    source_row_indices = load_row_source_row_indices(model, dataset, row_source)
    if source_row_indices is None or len(source_row_indices) != n_emb:
        return []
    train_idx = ds_splits.get("train_indices", ds_splits.get("train"))
    try:
        X_test = X.iloc[source_row_indices].reset_index(drop=True)
        y_test = np.asarray(y)[source_row_indices]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = np.asarray(y)[train_idx]
    except (IndexError, KeyError):
        return []

    # Marginal stats + target distribution from SAE training split
    marginals = _compute_marginals(X_train)
    task_type = ds_splits.get("task_type", "classification")
    target_summary = _target_summary(y_train, task_type)

    # Stratified activating rows: 2 top + 2 near-p90 + 2 near-p80 (6 total)
    active_mask = feat_acts > 0
    if active_mask.sum() < 1:
        return []
    active_idx = np.where(active_mask)[0]
    active_acts = feat_acts[active_idx]
    if activating_policy == "activation_bands":
        top_idx, top_bands = _stratified_activating(active_idx, active_acts, per_band=per_band)
    elif activating_policy == "scored_tiers":
        top_idx, top_bands = _select_scored_tiers(
            candidates=active_idx,
            scores=active_likeness_score,
            n=3 * per_band,
            positive=True,
        )
    else:
        raise ValueError(
            f"activating_policy={activating_policy!r} not one of {sorted(VALID_ACTIVATING_POLICIES)}"
        )

    contrast_idx, contrast_bands = _select_contrast_indices(
        feat_idx=feat_idx,
        top_idx=top_idx,
        emb=emb,
        acts_np=acts_np,
        pre_act_np=pre_act_np,
        decoder_cosine=decoder_cosine,
        active_mask=active_mask,
        contrast_policy=contrast_policy,
        cofire_top_k=cofire_top_k,
    )
    if len(contrast_idx) == 0:
        return []

    # Build rows (raw values annotated with marginal position from training split)
    ds_stats = row_stats.get(dataset, {})
    rows = []
    groups = [
        (top_idx, ["activating"] * len(top_idx), top_bands),
        (contrast_idx, ["contrast"] * len(contrast_idx), contrast_bands),
    ]
    for idx, labels, bands in groups:
        for ri, lab, band in zip(idx, labels, bands):
            row = {
                "label": lab,
                "band": band,
                "dataset": dataset,
                "row_idx": int(ri),
                "activation": float(feat_acts[ri]),
                "feature_pre_act": float(pre_act_np[ri, feat_idx]),
                "decoder_cosine": float(decoder_cosine[ri]),
                "active_likeness_score": float(active_likeness_score[ri]),
                "target": float(y_test[ri]) if ri < len(y_test) else None,
            }
            # Raw data values, annotated with marginal (training-split) position
            if ri < len(X_test):
                for col in X_test.columns:
                    row[col] = _annotate_value(col, X_test.iloc[ri][col], marginals)
            # Per-row stats
            for stat_name in ["pmi", "surprise", "compression"]:
                vals = ds_stats.get(stat_name)
                if vals is not None and ri < len(vals):
                    row[stat_name] = float(vals[ri])
            rows.append(row)

    return {"rows": rows, "target_summary": target_summary}


def build_validator_examples(
    model: str,
    feat_idx: int,
    n_act: int = 5,
    n_con: int = 5,
    device: str = "cpu",
    validator_policy: str = DEFAULT_VALIDATOR_POLICY,
    validator_tag: str = "",
) -> dict:
    """Build held-out validator CSVs + truth file for one (model, feat).

    Samples n_act activating + n_con non-activating rows per dataset from the
    SAE test set, *excluding* row_idx values already used in the contrastive
    CSVs. Writes per-dataset CSVs stripped of label/band/activation/row_idx
    giveaways and a truth JSON with per-row metadata for grading.
    """
    from scripts.intervention.intervene_lib import load_sae as _load_sae
    from data.extended_loader import load_tabarena_dataset

    out_dir = OUTPUT_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    if validator_policy not in VALID_VALIDATOR_POLICIES:
        raise ValueError(
            f"validator_policy={validator_policy!r} not one of {sorted(VALID_VALIDATOR_POLICIES)}"
        )
    tag_part = f"_{validator_tag}" if validator_tag else ""

    sae, _ = _load_sae(model, device=device)
    splits = json.loads(SPLITS_PATH.read_text())

    truth: dict = {}
    written: list = []

    model_dir = OUTPUT_DIR / model
    ds_to_csv = {}
    ctx_path = model_dir / f"f{feat_idx}_context.json"
    if not ctx_path.exists():
        raise FileNotFoundError(
            f"Missing {ctx_path}. Run the contrastive builder first so validator "
            f"generation can use the current datasets_used for f_{feat_idx}."
        )
    feat_context = json.loads(ctx_path.read_text())
    datasets_used = feat_context.get("datasets_used") or []
    row_source = feat_context.get("row_source", SAE_TEST_SOURCE)
    test_embs = load_row_source_embeddings(model, row_source)
    for ds in datasets_used:
        p = model_dir / f"f{feat_idx}_{ds}.csv"
        if p.exists() and ds in test_embs:
            ds_to_csv[ds] = p

    if not ds_to_csv:
        raise FileNotFoundError(
            f"No current contrastive CSVs for f_{feat_idx} in {model_dir}. "
            "Run the contrastive builder first."
        )

    for ds, contrastive_csv in ds_to_csv.items():
        emb = test_embs[ds]
        used_row_idx = set(int(r) for r in pd.read_csv(contrastive_csv).row_idx.tolist())

        with torch.no_grad():
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
            acts, pre_act = sae.encode(emb_t, return_pre_act=True)
        feat_acts = acts[:, feat_idx].cpu().numpy()
        pre_act_np = pre_act.cpu().numpy()
        decoder_cosine = _feature_decoder_cosine(sae, emb_t, feat_idx)
        active_likeness_score = _percentile_scores(pre_act_np[:, feat_idx])

        active_pos = np.where(feat_acts > 0)[0]
        inactive_pos = np.where(feat_acts == 0)[0]
        held_act = np.array([i for i in active_pos if int(i) not in used_row_idx])
        held_con = np.array([i for i in inactive_pos if int(i) not in used_row_idx])

        rng = np.random.default_rng(seed=(42 + feat_idx) * 1000 + hash(ds) % (2**31))
        n_act_take = min(n_act, len(held_act))
        n_con_take = min(n_con, len(held_con))
        if validator_policy == "scored_tiers":
            picked_act, picked_act_bands = _select_scored_tiers(
                candidates=held_act,
                scores=active_likeness_score,
                n=n_act_take,
                positive=True,
            )
            picked_con, picked_con_bands = _select_scored_tiers(
                candidates=held_con,
                scores=active_likeness_score,
                n=n_con_take,
                positive=False,
            )
        else:
            picked_act = rng.choice(held_act, size=n_act_take, replace=False) if n_act_take else np.array([], dtype=int)
            picked_act_bands = ["activating"] * len(picked_act)
        if validator_policy == "stratified_nonfire":
            picked_con, picked_con_bands = _select_stratified_nonfire(
                inactive_idx=held_con,
                pre_act_np=pre_act_np,
                decoder_cosine=decoder_cosine,
                feat_idx=feat_idx,
                n=n_con_take,
            )
        elif validator_policy == "random":
            picked_con = rng.choice(held_con, size=n_con_take, replace=False) if n_con_take else np.array([], dtype=int)
            picked_con_bands = ["random_nonfire"] * len(picked_con)

        ds_splits = splits.get(ds)
        if not ds_splits:
            continue
        train_idx = ds_splits.get("train_indices", ds_splits.get("train"))
        loaded = load_tabarena_dataset(ds, max_samples=999999)
        if loaded is None:
            continue
        X, y = loaded[0], loaded[1]
        source_row_indices = load_row_source_row_indices(model, ds, row_source)
        if source_row_indices is None or len(source_row_indices) != len(emb):
            continue
        X_test = X.iloc[source_row_indices].reset_index(drop=True)
        y_test = np.asarray(y)[source_row_indices]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        marginals = _compute_marginals(X_train)

        rows = []
        ds_truth: dict = {}
        groups = [
            (picked_act, True, picked_act_bands),
            (picked_con, False, picked_con_bands),
        ]
        for positions, is_active, row_bands in groups:
            for ri, row_band in zip(positions, row_bands):
                ri = int(ri)
                row_id = f"r{len(rows):03d}"
                row = {"row_id": row_id}
                row["target"] = float(y_test[ri])
                for col in X_test.columns:
                    row[col] = _annotate_value(col, X_test.iloc[ri][col], marginals)
                rows.append(row)
                ds_truth[row_id] = {
                    "fires": bool(is_active),
                    "kind": row_band,
                    "row_idx": ri,
                    "activation": float(feat_acts[ri]),
                    "feature_pre_act": float(pre_act_np[ri, feat_idx]),
                    "decoder_cosine": float(decoder_cosine[ri]),
                    "active_likeness_score": float(active_likeness_score[ri]),
                }

        # Shuffle so the validator can't infer class from row order
        shuffle_rng = np.random.default_rng(seed=777 + feat_idx)
        order = list(range(len(rows)))
        shuffle_rng.shuffle(order)
        rows = [rows[i] for i in order]
        # Re-assign opaque row_ids after shuffle so they run 0..N-1 in display order
        new_truth = {}
        for new_i, r in enumerate(rows):
            old_id = r["row_id"]
            new_id = f"r{new_i:03d}"
            r["row_id"] = new_id
            new_truth[new_id] = ds_truth[old_id]
        ds_truth = new_truth

        out_path = out_dir / f"f{feat_idx}_validator{tag_part}_{ds}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        truth[ds] = ds_truth
        written.append(out_path)
        print(f"  {ds}: {len(picked_act)} act + {len(picked_con)} con -> {out_path.name}")

    truth_payload = {
        "_meta": {
            "model": model,
            "feature_idx": feat_idx,
            "row_source": row_source,
            "validator_policy": validator_policy,
            "validator_tag": validator_tag,
        },
        **truth,
    }
    truth_path = out_dir / f"f{feat_idx}_validator{tag_part}_truth.json"
    truth_path.write_text(json.dumps(truth_payload, indent=2))
    print(f"Wrote {len(written)} validator CSVs + {truth_path.name}")
    return truth_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", nargs="+", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-datasets", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--row-source",
        choices=sorted(VALID_ROW_SOURCE_MODES),
        default=DEFAULT_ROW_SOURCE_MODE,
        help="Row source for contrastive/validator examples. 'auto' uses the quality-cache feature source when available.",
    )
    parser.add_argument(
        "--contrast-policy",
        choices=sorted(VALID_CONTRAST_POLICIES),
        default=DEFAULT_CONTRAST_POLICY,
        help="How to choose non-activating contrast rows for each activating sample.",
    )
    parser.add_argument(
        "--activating-policy",
        choices=sorted(VALID_ACTIVATING_POLICIES),
        default=DEFAULT_ACTIVATING_POLICY,
        help="How to choose activating rows for each contrastive example CSV.",
    )
    parser.add_argument(
        "--cofire-top-k",
        type=int,
        default=DEFAULT_COFIRE_TOP_K,
        help="When using cofire_hard_negative, rank overlap using the row's top-K strongest co-firing features.",
    )
    parser.add_argument(
        "--dataset-selection",
        choices=["auto", "n_active", "quality"],
        default="auto",
        help="How to choose datasets per feature. 'auto' prefers the quality cache when available.",
    )
    parser.add_argument(
        "--quality-cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the global dataset-quality cache.",
    )
    parser.add_argument(
        "--require-quality-cache",
        action="store_true",
        help="Fail if quality-based dataset selection is requested but the cache is missing.",
    )
    parser.add_argument("--validator", action="store_true",
                        help="Build validator (held-out) CSVs instead of contrastive examples")
    parser.add_argument("--n-act", type=int, default=5,
                        help="Number of held-out activating rows per dataset (--validator mode)")
    parser.add_argument("--n-con", type=int, default=5,
                        help="Number of held-out non-activating rows per dataset (--validator mode)")
    parser.add_argument(
        "--validator-policy",
        choices=sorted(VALID_VALIDATOR_POLICIES),
        default=DEFAULT_VALIDATOR_POLICY,
        help="How to choose held-out non-activating validator rows.",
    )
    parser.add_argument(
        "--validator-tag",
        default="",
        help="Optional tag for validator artifacts, e.g. 'hard' writes fN_validator_hard_*.csv.",
    )
    args = parser.parse_args()

    if args.validator:
        if not args.features:
            parser.error("--validator requires --features")
        for feat in args.features:
            build_validator_examples(
                args.model,
                feat,
                args.n_act,
                args.n_con,
                args.device,
                validator_policy=args.validator_policy,
                validator_tag=args.validator_tag,
            )
        return

    splits = json.loads(SPLITS_PATH.read_text())
    sae, _ = load_sae(args.model, device=args.device)
    row_stats = load_row_stats()
    row_source_embs: dict[str, dict[str, np.ndarray]] = {}

    def _embeddings_for(row_source: str) -> dict[str, np.ndarray]:
        if row_source not in row_source_embs:
            row_source_embs[row_source] = load_row_source_embeddings(args.model, row_source)
        return row_source_embs[row_source]

    default_test_embs = _embeddings_for(explicit_row_source_or_default(args.row_source))

    # Load dataset-level metafeatures
    dataset_meta = {}
    if PYMFE_PATH.exists():
        with open(PYMFE_PATH) as f:
            dataset_meta = json.load(f)

    print(f"Loaded SAE, {len(default_test_embs)} datasets from {explicit_row_source_or_default(args.row_source)}, row stats for {len(row_stats)} datasets")

    if args.all:
        # All alive features
        with torch.no_grad():
            sample_emb = next(iter(default_test_embs.values()))
            sample_acts = sae.encode(torch.tensor(sample_emb[:1], dtype=torch.float32,
                                                   device=args.device))
            n_features = sample_acts.shape[1]
        features = list(range(n_features))
    elif args.features:
        features = args.features
    else:
        parser.error("Specify --features or --all")

    out_dir = OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessing context
    ctx = PREPROCESSING_CONTEXT.get(args.model, {})
    if ctx:
        ctx_path = out_dir / "preprocessing_context.json"
        with open(ctx_path, "w") as f:
            json.dump(ctx, f, indent=2)
        print(f"Saved preprocessing context to {ctx_path}")

    # Rank datasets by activation count for each feature
    quality_cache = load_quality_cache(args.quality_cache_path)
    if args.dataset_selection == "quality" and not quality_cache:
        if args.require_quality_cache or args.quality_cache_path != DEFAULT_CACHE_PATH:
            parser.error(f"Quality cache not found at {args.quality_cache_path}")
    if args.dataset_selection == "auto" and args.require_quality_cache and not quality_cache:
        parser.error(f"Required quality cache not found at {args.quality_cache_path}")

    for feat in features:
        # Find datasets where this feature fires most
        ds_counts = []
        ds_quality_entries = None
        selected_via = "n_active"
        feature_block = cache_entry_for_feature(quality_cache, args.model, feat) if quality_cache else None
        if args.row_source == AUTO_SOURCE:
            feature_row_source = feature_block_row_source(feature_block, SAE_TEST_SOURCE)
        else:
            feature_row_source = explicit_row_source_or_default(args.row_source)
        test_embs = _embeddings_for(feature_row_source)
        datasets = sorted(test_embs.keys())
        for ds in datasets:
            emb = test_embs[ds]
            with torch.no_grad():
                acts = sae.encode(torch.tensor(emb, dtype=torch.float32,
                                               device=args.device))
            n_active = int((acts[:, feat] > 0).sum())
            if n_active > 0:
                ds_counts.append((ds, n_active))
        ds_counts.sort(key=lambda x: -x[1])

        if not ds_counts:
            print(f"f_{feat}: no activations across any dataset")
            continue

        if quality_cache and args.dataset_selection in {"auto", "quality"}:
            if feature_block:
                cache_source = feature_block_row_source(
                    feature_block,
                    quality_cache.get("metadata", {}).get("source_split", SAE_TEST_SOURCE),
                )
                if cache_source == feature_row_source:
                    ds_quality_entries = feature_block.get("datasets", {})
                    selected_ds = select_top_datasets(
                        ds_quality_entries,
                        args.max_datasets,
                        diversity_tie_margin=quality_cache.get("metadata", {})
                        .get("score_config", {})
                        .get("diversity_tie_margin", DEFAULT_SCORE_CONFIG["diversity_tie_margin"]),
                    )
                    if selected_ds:
                        selected_via = f"quality_cache:{feature_row_source}"
                    elif args.dataset_selection == "quality" and args.require_quality_cache:
                        parser.error(
                            f"Quality cache has no selectable datasets for model={args.model} f_{feat} row_source={feature_row_source}"
                        )
                elif args.dataset_selection == "quality" and args.require_quality_cache:
                    parser.error(
                        f"Quality cache row_source mismatch for model={args.model} f_{feat}: cache={cache_source}, requested={feature_row_source}"
                    )
            elif args.dataset_selection == "quality" and args.require_quality_cache:
                parser.error(
                    f"Quality cache missing model={args.model} f_{feat} at {args.quality_cache_path}"
                )
            else:
                selected_ds = []
        else:
            selected_ds = []

        if not selected_ds:
            selected_ds = [ds for ds, _ in ds_counts[:args.max_datasets]]
            selected_via = "n_active"

        print(
            f"f_{feat}: {len(ds_counts)} datasets fire on {feature_row_source}, using top {len(selected_ds)} via {selected_via}"
        )

        ds_contexts = {}
        for ds in selected_ds:
            result = build_contrastive(args.model, feat, ds, sae, test_embs,
                                       splits, row_stats, args.top_k, args.device,
                                       activating_policy=args.activating_policy,
                                       contrast_policy=args.contrast_policy,
                                       cofire_top_k=args.cofire_top_k,
                                       row_source=feature_row_source)
            if not result or not result.get("rows"):
                continue
            rows = result["rows"]
            target_summary = result.get("target_summary", {})

            import pandas as pd
            df = pd.DataFrame(rows)
            out_path = out_dir / f"f{feat}_{ds}.csv"
            df.to_csv(out_path, index=False)

            # Collect dataset metafeatures
            meta = dataset_meta.get(ds, {})
            ds_info = splits.get(ds, {})
            ds_contexts[ds] = {
                "task_type": ds_info.get("task_type", "unknown"),
                "n_train": len(ds_info.get("train_indices", [])),
                "n_test": len(ds_info.get("test_indices", [])),
                "n_activating": int((pd.Series([r["label"] for r in rows]) == "activating").sum()),
                "csv_file": f"f{feat}_{ds}.csv",
                "target_summary": target_summary,
                "dataset_selection_method": selected_via,
                "activating_policy": args.activating_policy,
                "contrast_policy": args.contrast_policy,
                "row_source": feature_row_source,
            }
            if ds_quality_entries and ds in ds_quality_entries:
                ds_contexts[ds]["dataset_quality"] = ds_quality_entries[ds]
            for key in ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat",
                         "inst_to_attr", "cat_to_num", "nr_bin"]:
                if key in meta:
                    ds_contexts[ds][key] = meta[key]

        # Save per-feature context
        feat_context = {
            "model": args.model,
            "feature_idx": feat,
            "n_datasets_firing": len(ds_counts),
            "datasets_used": list(ds_contexts.keys()),
            "dataset_selection_method": selected_via,
            "activating_policy": args.activating_policy,
            "contrast_policy": args.contrast_policy,
            "row_source": feature_row_source,
            "row_source_mode_requested": args.row_source,
            "preprocessing": ctx,
            "dataset_stats": ds_contexts,
        }
        ctx_path = out_dir / f"f{feat}_context.json"
        with open(ctx_path, "w") as f:
            json.dump(feat_context, f, indent=2)

        print(f"  -> {len(ds_contexts)} CSVs + context saved")


if __name__ == "__main__":
    main()
