#!/usr/bin/env python3
"""Replay panel causal patches through strong and weak predictors.

The panel patch artifact measures SAE activation changes under raw donor-row
patches. This script adds model predictions for each original and patched row,
so the raw input patch can be compared against the intervention scatter plot's
strong/weak prediction behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from models.mitra_embeddings import MitraEmbeddingExtractor
from scripts._project_root import PROJECT_ROOT
from scripts.concepts.patch_activation_probe import (
    MITRA_CONTEXT_ROWS,
    MITRA_SEED,
    _load_raw_mitra_context_query,
)
from scripts.intervention.intervene_lib import (
    SPLITS_PATH,
    get_extraction_layer_taskaware,
    load_dataset_context,
)
from scripts.intervention.intervene_sae import build_tail


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _prediction_record(pred: np.ndarray, *, task: str) -> dict[str, Any]:
    pred = np.asarray(pred)
    if task == "regression":
        return {"value": float(pred.reshape(-1)[0])}
    probs = pred.astype(float).reshape(-1)
    return {
        "probs": [_json_value(x) for x in probs],
        "pred_class": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
    }


def _prediction_delta(orig: np.ndarray, patched: np.ndarray, *, task: str) -> Any:
    orig = np.asarray(orig)
    patched = np.asarray(patched)
    if task == "regression":
        return float(patched.reshape(-1)[0] - orig.reshape(-1)[0])
    return [_json_value(x) for x in (patched.astype(float).reshape(-1) - orig.astype(float).reshape(-1))]


def _predict_mitra(
    *,
    X_context: pd.DataFrame,
    y_context: np.ndarray,
    X_rows: pd.DataFrame,
    task: str,
    device: str,
    seed: int,
) -> np.ndarray:
    extractor = MitraEmbeddingExtractor(
        device=device,
        n_estimators=1,
        fine_tune=False,
        seed=seed,
    )
    result = extractor.extract_embeddings(
        X_context,
        y_context,
        X_rows,
        task=task,
    )
    key = "final_preds" if task == "regression" else "final_probs"
    if key not in result.layer_embeddings:
        raise RuntimeError(f"Mitra result missing {key}")
    return np.asarray(result.layer_embeddings[key])


def _predict_tail_model(
    *,
    model: str,
    dataset: str,
    X_rows: pd.DataFrame,
    task: str,
    device: str,
    seed: int,
) -> np.ndarray:
    splits = json.loads(SPLITS_PATH.read_text())
    X_context, y_context, _, _, _, _ = load_dataset_context(
        model,
        dataset,
        splits=splits,
        max_context=MITRA_CONTEXT_ROWS,
    )
    if hasattr(y_context, "dtype") and y_context.dtype == np.int32:
        y_context = y_context.astype(np.int64)
    _seed_everything(seed)
    layer = get_extraction_layer_taskaware(model, dataset=dataset)
    target_name = splits.get(dataset, {}).get("target", "target")
    tail = build_tail(
        model,
        X_context,
        y_context,
        X_rows.reset_index(drop=True),
        layer,
        task,
        device,
        target_name=target_name,
    )
    return np.asarray(tail.baseline_preds)


def _rows_for_panel(panel: dict) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    explanation_indices = []
    for idx, item in enumerate(panel["explanations"]):
        if item.get("status") != "found":
            continue
        rows.append(item["active_row"])
        rows.append(item["patched_row"])
        explanation_indices.append(idx)
    return pd.DataFrame(rows), explanation_indices


def replay(args: argparse.Namespace) -> dict:
    payload = json.loads(args.input.read_text())
    for panel in payload["panels"]:
        dataset = panel["dataset"]
        strong = panel["strong_model"]
        weak = panel["weak_model"]
        X_rows, indices = _rows_for_panel(panel)
        print(f"{dataset}: replaying {len(indices)} patches through {strong}/{weak}", flush=True)
        if not indices:
            continue

        splits = json.loads(SPLITS_PATH.read_text())
        if strong == "mitra":
            X_context_s, y_context_s, _, _, _, task = _load_raw_mitra_context_query(
                strong,
                dataset,
            )
        else:
            X_context_s, y_context_s, _, _, _, task = load_dataset_context(
                strong,
                dataset,
                splits=splits,
                max_context=MITRA_CONTEXT_ROWS,
            )
            if hasattr(y_context_s, "dtype") and y_context_s.dtype == np.int32:
                y_context_s = y_context_s.astype(np.int64)

        if strong == "mitra":
            strong_preds = _predict_mitra(
                X_context=X_context_s,
                y_context=y_context_s,
                X_rows=X_rows,
                task=task,
                device=args.device,
                seed=args.seed,
            )
        else:
            strong_preds = _predict_tail_model(
                model=strong,
                dataset=dataset,
                X_rows=X_rows,
                task=task,
                device=args.device,
                seed=args.seed,
            )

        weak_preds = _predict_tail_model(
            model=weak,
            dataset=dataset,
            X_rows=X_rows,
            task=task,
            device=args.device,
            seed=args.seed,
        )

        for pos, item_idx in enumerate(indices):
            orig_i = 2 * pos
            patched_i = orig_i + 1
            item = panel["explanations"][item_idx]
            item["prediction_replay"] = {
                "task": task,
                "strong_model": strong,
                "weak_model": weak,
                "strong_original": _prediction_record(strong_preds[orig_i], task=task),
                "strong_patched": _prediction_record(strong_preds[patched_i], task=task),
                "strong_delta": _prediction_delta(strong_preds[orig_i], strong_preds[patched_i], task=task),
                "weak_original": _prediction_record(weak_preds[orig_i], task=task),
                "weak_patched": _prediction_record(weak_preds[patched_i], task=task),
                "weak_delta": _prediction_delta(weak_preds[orig_i], weak_preds[patched_i], task=task),
            }
    return payload


def _scalar_for_csv(record: dict[str, Any], *, task: str, y_true: Any = None) -> Any:
    if task == "regression":
        return record.get("value")
    probs = record.get("probs") or []
    if y_true is not None:
        y = int(y_true)
        if 0 <= y < len(probs):
            return probs[y]
    return record.get("confidence")


def write_csv(payload: dict, path: Path) -> None:
    rows = []
    for panel in payload["panels"]:
        for item in panel["explanations"]:
            replay = item.get("prediction_replay")
            patch = "; ".join(
                f"{field['column']}: {field['active_value']} -> {field['donor_value']}"
                for field in item.get("patch_fields", [])
            )
            row = {
                "dataset": panel["dataset"],
                "rank": item["rank"],
                "feature": item["feat"],
                "status": item["status"],
                "patch": patch,
                "original_activation": item.get("original_activation"),
                "patched_activation": item.get("patched_activation"),
            }
            if replay:
                task = replay["task"]
                row.update(
                    {
                        "task": task,
                        "strong_model": replay["strong_model"],
                        "weak_model": replay["weak_model"],
                        "strong_original_score": _scalar_for_csv(replay["strong_original"], task=task),
                        "strong_patched_score": _scalar_for_csv(replay["strong_patched"], task=task),
                        "weak_original_score": _scalar_for_csv(replay["weak_original"], task=task),
                        "weak_patched_score": _scalar_for_csv(replay["weak_patched"], task=task),
                        "strong_delta": replay["strong_delta"],
                        "weak_delta": replay["weak_delta"],
                    }
                )
            rows.append(row)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "concept_patch_probes"
        / "panel_patch_explanations_seeded_context1024_v1.worker.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "concept_patch_probes"
        / "panel_patch_explanations_seeded_context1024_with_predictions_v1.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=MITRA_SEED)
    args = parser.parse_args()

    payload = replay(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    csv_path = args.out.with_suffix(".csv")
    write_csv(payload, csv_path)
    print(f"Wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
