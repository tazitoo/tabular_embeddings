#!/usr/bin/env python3
"""Build a global dataset-quality cache for contrastive-example selection.

The cache scores each (model, feature, dataset) pair for labeling usefulness
using activation support, activation tail structure, contrastive separation,
and prediction spread. It is used upstream of contrastive example generation
to choose datasets that are likely to yield better labeling evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.dataset_quality_cache import (
    CACHE_VERSION,
    DEFAULT_CACHE_PATH,
    DEFAULT_SCORE_CONFIG,
    DEFAULT_SCORE_WEIGHTS,
    minmax_scale,
    normalized_entropy_from_counts,
)
from scripts.concepts.row_source import (
    AUTO_SOURCE,
    BACKUP_SOURCE,
    DEFAULT_ROW_SOURCE_MODE,
    SAE_TEST_SOURCE,
    VALID_ROW_SOURCE_MODES,
    feature_block_row_source,
    load_row_source_baseline_predictions,
    load_row_source_embeddings,
)
from scripts.intervention.intervene_lib import load_sae
from scripts.round_paths import sae_training_dir


SAE_DATA_DIR = sae_training_dir()


def carry_over_models(cache: dict, previous: Path, rebuilt: list[str]) -> None:
    """Copy every model block of a previous round's cache that this run did not rebuild.

    A round regenerates one model's SAE; the other models' entries are unchanged, so
    their blocks are carried over verbatim and recorded in metadata["carried_over"].
    A rebuilt model's fresh block always wins.
    """
    with open(previous) as f:
        old = json.load(f)
    carried = cache["metadata"].setdefault("carried_over", {})
    for model, block in old.get("models", {}).items():
        if model in rebuilt or model in cache["models"]:
            continue
        cache["models"][model] = block
        carried[model] = str(previous)


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        if np.isnan(value):
            return float(default)
    except TypeError:
        pass
    return float(value)


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _detect_models(requested: Optional[list[str]]) -> list[str]:
    if requested:
        return requested
    models = set()
    for path in SAE_DATA_DIR.glob("*_sae_test.npz"):
        name = path.name
        if name.endswith("_taskaware_sae_test.npz"):
            model = name[: -len("_taskaware_sae_test.npz")]
        elif name.endswith("_sae_test.npz"):
            model = name[: -len("_sae_test.npz")]
        else:
            continue
        models.add(model)
    return sorted(models)


def _prediction_spread_metrics(preds: Optional[dict], active_mask: np.ndarray, cfg: dict) -> tuple[dict, float, list[str]]:
    metrics = {}
    reasons = []
    if preds is None:
        metrics["available"] = False
        return metrics, 0.0, ["missing_predictions"]

    metrics["available"] = True
    task_type = str(preds["task_type"])
    active_idx = np.where(active_mask)[0]
    if active_idx.size == 0:
        return metrics, 0.0, ["no_active_rows"]

    if task_type == "regression":
        pred_vals = np.asarray(preds["pred_probs"]).reshape(-1)
        y_true = np.asarray(preds["y_true"]).reshape(-1)
        err = np.abs(pred_vals - y_true)
        active_pred = pred_vals[active_idx]
        active_err = err[active_idx]
        target_active = y_true[active_idx]
        metrics.update(
            {
                "pred_value_std_active": _safe_float(np.std(active_pred)),
                "pred_abs_err_std_active": _safe_float(np.std(active_err)),
                "pred_abs_err_p90_active": _safe_float(np.quantile(active_err, 0.90)),
                "target_std_active": _safe_float(np.std(target_active)),
            }
        )
        score = float(
            np.mean(
                [
                    metrics["pred_value_std_active"],
                    metrics["pred_abs_err_std_active"],
                    metrics["target_std_active"],
                ]
            )
        )
        if score <= cfg["prediction_std_floor"]:
            reasons.append("degenerate_prediction_spread")
        return metrics, score, reasons

    probs = np.asarray(preds["pred_probs"])
    pred_class = np.asarray(preds["pred_class"]).reshape(-1)
    y_true = np.asarray(preds["y_true"]).reshape(-1)
    pred_conf = probs.max(axis=1)
    active_conf = pred_conf[active_idx]
    active_pred = pred_class[active_idx]
    active_true = y_true[active_idx]
    incorrect = active_pred != active_true
    low_conf = active_conf < cfg["low_conf_threshold"]
    counts = np.bincount(active_pred.astype(int), minlength=int(active_pred.max()) + 1 if active_pred.size else 0)
    metrics.update(
        {
            "pred_conf_std_active": _safe_float(np.std(active_conf)),
            "pred_conf_mean_active": _safe_float(np.mean(active_conf)),
            "fraction_incorrect_active": _safe_float(np.mean(incorrect)),
            "fraction_low_conf_active": _safe_float(np.mean(low_conf)),
            "pred_class_entropy_active": normalized_entropy_from_counts(counts),
            "pred_class_diversity_active": _safe_float(np.count_nonzero(counts) / max(len(counts), 1)),
        }
    )
    score = float(
        np.mean(
            [
                metrics["pred_conf_std_active"],
                metrics["fraction_incorrect_active"],
                metrics["fraction_low_conf_active"],
                metrics["pred_class_entropy_active"],
            ]
        )
    )
    if (
        metrics["pred_conf_std_active"] <= cfg["prediction_std_floor"]
        and metrics["fraction_incorrect_active"] <= cfg["prediction_std_floor"]
        and metrics["pred_class_entropy_active"] <= cfg["prediction_std_floor"]
    ):
        reasons.append("degenerate_prediction_spread")
    return metrics, score, reasons


def _raw_metrics_for_dataset_feature(
    feat_acts: np.ndarray,
    preds: Optional[dict],
    cfg: dict,
) -> tuple[dict, dict]:
    n_rows = int(len(feat_acts))
    active_mask = feat_acts > 0
    inactive_mask = ~active_mask
    active_vals = feat_acts[active_mask]
    inactive_vals = feat_acts[inactive_mask]
    n_active = int(active_mask.sum())
    n_inactive = int(inactive_mask.sum())
    fire_rate = float(n_active / n_rows) if n_rows else 0.0
    inactive_rate = float(inactive_mask.sum() / n_rows) if n_rows else 0.0

    required_active_rows = int(
        cfg["contrastive_active_rows"]
        + cfg["validator_active_rows"]
        + cfg.get("feasibility_margin_rows", 0)
    )
    required_inactive_rows = int(
        cfg["contrastive_inactive_rows"]
        + cfg["validator_inactive_rows"]
        + cfg.get("feasibility_margin_rows", 0)
    )

    filtered = []
    if n_active < cfg["min_active_rows"]:
        filtered.append("too_few_active_rows")
    if n_active < required_active_rows:
        filtered.append("insufficient_active_pool")
    if n_inactive < required_inactive_rows:
        filtered.append("insufficient_inactive_pool")

    if n_active > 0:
        p70 = _quantile(active_vals, 0.70)
        p80 = _quantile(active_vals, 0.80)
        p90 = _quantile(active_vals, 0.90)
        p99 = _quantile(active_vals, cfg["top_quantile"])
        mean_positive = _safe_float(np.mean(active_vals))
        max_positive = _safe_float(np.max(active_vals))
    else:
        p70 = p80 = p90 = p99 = mean_positive = max_positive = 0.0

    band_mass = {
        "pct_top": float(np.mean(feat_acts >= p99)) if n_rows and n_active else 0.0,
        "pct_p90": float(np.mean(feat_acts >= p90)) if n_rows and n_active else 0.0,
        "pct_p80": float(np.mean(feat_acts >= p80)) if n_rows and n_active else 0.0,
        "pct_p70": float(np.mean(feat_acts >= p70)) if n_rows and n_active else 0.0,
    }

    inactive_p99 = _quantile(inactive_vals, 0.99) if inactive_vals.size else 0.0
    inactive_overlap = float(np.mean(inactive_vals >= p80)) if inactive_vals.size and n_active else 0.0
    active_above_inactive_p99 = float(np.mean(active_vals >= inactive_p99)) if n_active else 0.0
    contrastive_separation = {
        "mean_gap": mean_positive - _safe_float(np.mean(inactive_vals)),
        "p80_gap": p80 - inactive_p99,
        "inactive_above_active_p80": inactive_overlap,
        "active_above_inactive_p99": active_above_inactive_p99,
    }

    prediction_spread, pred_score_raw, pred_reasons = _prediction_spread_metrics(preds, active_mask, cfg)
    filtered.extend(pred_reasons)

    task_type = str(preds["task_type"]) if preds is not None else "unknown"

    metrics = {
        "task_type": task_type,
        "n_rows_test": n_rows,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "fire_rate": fire_rate,
        "inactive_rate": inactive_rate,
        "activation_band_mass": band_mass,
        "activation_summary": {
            "max_positive": max_positive,
            "mean_positive": mean_positive,
            "p90_positive": p90,
            "p99_positive": p99,
        },
        "contrastive_separation": contrastive_separation,
        "prediction_spread": prediction_spread,
        "filtered_out": bool(filtered),
        "filter_reasons": sorted(set(filtered)),
    }

    raw_components = {
        "activation_support": 0.5 * fire_rate + 0.5 * np.log1p(n_active),
        "activation_tail": 0.10 * band_mass["pct_top"] + 0.35 * band_mass["pct_p90"] + 0.35 * band_mass["pct_p80"] + 0.20 * band_mass["pct_p70"],
        "contrastive_separation": (
            0.45 * contrastive_separation["mean_gap"]
            + 0.30 * contrastive_separation["p80_gap"]
            + 0.15 * (1.0 - contrastive_separation["inactive_above_active_p80"])
            + 0.10 * contrastive_separation["active_above_inactive_p99"]
        ),
        "prediction_usefulness": pred_score_raw,
    }
    return metrics, raw_components


def _apply_scoring(entries: dict[str, dict], raw_components: dict[str, dict], weights: dict) -> None:
    components = list(weights.keys())
    normalized = {component: {} for component in components}
    for component in components:
        datasets = list(raw_components.keys())
        vals = [raw_components[dataset][component] for dataset in datasets]
        scaled = minmax_scale(vals)
        for dataset, value in zip(datasets, scaled):
            normalized[component][dataset] = float(value)

    for dataset, entry in entries.items():
        quality_components = {
            component: normalized[component][dataset]
            for component in components
        }
        entry["quality_components"] = quality_components
        if entry.get("filtered_out"):
            entry["labeling_quality_score"] = 0.0
        else:
            entry["labeling_quality_score"] = float(
                sum(weights[component] * quality_components[component] for component in components)
            )


def build_cache_for_model(model: str, device: str, score_cfg: dict, weights: dict, row_source: str) -> dict:
    sae, _ = load_sae(model, device=device)
    test_embs = load_row_source_embeddings(model, row_source)
    if not test_embs:
        raise ValueError(f"No embeddings found for model '{model}' row_source={row_source!r}")

    sample_ds = next(iter(sorted(test_embs.keys())))
    with torch.no_grad():
        sample_acts = sae.encode(
            torch.tensor(test_embs[sample_ds][:1], dtype=torch.float32, device=device)
        )
    n_features = int(sample_acts.shape[1])

    per_feature_entries: dict[str, dict] = {
        str(feat_idx): {"datasets": {}} for feat_idx in range(n_features)
    }
    per_feature_raw: dict[str, dict] = {
        str(feat_idx): {} for feat_idx in range(n_features)
    }

    for dataset in sorted(test_embs.keys()):
        emb = test_embs[dataset]
        with torch.no_grad():
            acts = sae.encode(torch.tensor(emb, dtype=torch.float32, device=device))
        acts_np = acts.cpu().numpy()
        preds = load_row_source_baseline_predictions(model, dataset, row_source)

        for feat_idx in range(n_features):
            metrics, raw_components = _raw_metrics_for_dataset_feature(acts_np[:, feat_idx], preds, score_cfg)
            feat_key = str(feat_idx)
            per_feature_entries[feat_key]["datasets"][dataset] = metrics
            per_feature_raw[feat_key][dataset] = raw_components

    for feat_key in per_feature_entries:
        _apply_scoring(per_feature_entries[feat_key]["datasets"], per_feature_raw[feat_key], weights)

    return {
        "n_features": n_features,
        "n_datasets": len(test_embs),
        "datasets": sorted(test_embs.keys()),
        "row_source": row_source,
        "features": per_feature_entries,
    }


def _feature_has_selectable_datasets(feature_block: dict) -> bool:
    datasets = feature_block.get("datasets", {})
    return any(not entry.get("filtered_out") for entry in datasets.values())


def build_cache_for_model_auto(model: str, device: str, score_cfg: dict, weights: dict) -> dict:
    source_results: dict[str, dict] = {}
    for row_source in [SAE_TEST_SOURCE, BACKUP_SOURCE]:
        try:
            source_results[row_source] = build_cache_for_model(
                model, device, score_cfg, weights, row_source
            )
        except FileNotFoundError:
            continue
        except ValueError:
            continue

    if not source_results:
        raise ValueError(f"No embeddings found for model '{model}' in any row source")

    first_result = next(iter(source_results.values()))
    n_features = int(first_result["n_features"])
    features: dict[str, dict] = {}
    for feat_idx in range(n_features):
        feat_key = str(feat_idx)
        preferred_source = None
        for candidate in [SAE_TEST_SOURCE, BACKUP_SOURCE]:
            feature_block = source_results.get(candidate, {}).get("features", {}).get(feat_key)
            if feature_block and _feature_has_selectable_datasets(feature_block):
                preferred_source = candidate
                break
        if preferred_source is None:
            preferred_source = SAE_TEST_SOURCE if SAE_TEST_SOURCE in source_results else next(iter(source_results.keys()))
        chosen = copy.deepcopy(source_results[preferred_source]["features"][feat_key])
        chosen["selected_row_source"] = preferred_source
        chosen["row_source_availability"] = {
            source: {
                "n_datasets": len(source_results[source]["features"][feat_key].get("datasets", {})),
                "n_selectable": sum(
                    1
                    for entry in source_results[source]["features"][feat_key].get("datasets", {}).values()
                    if not entry.get("filtered_out")
                ),
            }
            for source in sorted(source_results.keys())
        }
        features[feat_key] = chosen

    return {
        "n_features": n_features,
        "n_datasets": sum(result["n_datasets"] for result in source_results.values()),
        "datasets": {source: result["datasets"] for source, result in source_results.items()},
        "row_source": AUTO_SOURCE,
        "row_sources_evaluated": sorted(source_results.keys()),
        "features": features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Models to include. Defaults to all cached SAE models.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--row-source",
        choices=sorted(VALID_ROW_SOURCE_MODES),
        default=DEFAULT_ROW_SOURCE_MODE,
        help="Row source to score. 'auto' prefers sae_test and falls back per-feature to outer_context_backup when sae_test has no selectable datasets.",
    )
    parser.add_argument("--min-active-rows", type=int, default=DEFAULT_SCORE_CONFIG["min_active_rows"])
    parser.add_argument("--contrastive-active-rows", type=int, default=DEFAULT_SCORE_CONFIG["contrastive_active_rows"])
    parser.add_argument("--contrastive-inactive-rows", type=int, default=DEFAULT_SCORE_CONFIG["contrastive_inactive_rows"])
    parser.add_argument("--validator-active-rows", type=int, default=DEFAULT_SCORE_CONFIG["validator_active_rows"])
    parser.add_argument("--validator-inactive-rows", type=int, default=DEFAULT_SCORE_CONFIG["validator_inactive_rows"])
    parser.add_argument("--feasibility-margin-rows", type=int, default=DEFAULT_SCORE_CONFIG["feasibility_margin_rows"])
    parser.add_argument("--top-quantile", type=float, default=DEFAULT_SCORE_CONFIG["top_quantile"])
    parser.add_argument("--low-conf-threshold", type=float, default=DEFAULT_SCORE_CONFIG["low_conf_threshold"])
    parser.add_argument("--prediction-std-floor", type=float, default=DEFAULT_SCORE_CONFIG["prediction_std_floor"])
    parser.add_argument("--carry-over", type=Path, default=None,
                        help="previous round's cache; its blocks for models not in --models are copied over")
    args = parser.parse_args()

    score_cfg = {
        **DEFAULT_SCORE_CONFIG,
        "min_active_rows": args.min_active_rows,
        "contrastive_active_rows": args.contrastive_active_rows,
        "contrastive_inactive_rows": args.contrastive_inactive_rows,
        "validator_active_rows": args.validator_active_rows,
        "validator_inactive_rows": args.validator_inactive_rows,
        "feasibility_margin_rows": args.feasibility_margin_rows,
        "top_quantile": args.top_quantile,
        "low_conf_threshold": args.low_conf_threshold,
        "prediction_std_floor": args.prediction_std_floor,
    }
    models = _detect_models(args.models)
    if not models:
        raise SystemExit("No models found in SAE cache directory.")

    cache = {
        "metadata": {
            "cache_version": CACHE_VERSION,
            "source_split": args.row_source,
            "sae_data_dir": str(SAE_DATA_DIR.relative_to(PROJECT_ROOT)),
            "weights": DEFAULT_SCORE_WEIGHTS,
            "score_config": score_cfg,
            "selection_target": "top_k_datasets_per_feature",
        },
        "models": {},
    }

    for model in models:
        print(f"Building dataset-quality cache for {model}")
        if args.row_source == AUTO_SOURCE:
            cache["models"][model] = build_cache_for_model_auto(
                model, args.device, score_cfg, DEFAULT_SCORE_WEIGHTS
            )
        else:
            model_block = build_cache_for_model(
                model, args.device, score_cfg, DEFAULT_SCORE_WEIGHTS, args.row_source
            )
            for feature_block in model_block["features"].values():
                feature_block["selected_row_source"] = args.row_source
            cache["models"][model] = model_block

    if args.carry_over is not None:
        carry_over_models(cache, args.carry_over, rebuilt=models)
        print(f"Carried over {sorted(cache['metadata']['carried_over'])} from {args.carry_over}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Wrote dataset-quality cache to {args.output}")


if __name__ == "__main__":
    main()
