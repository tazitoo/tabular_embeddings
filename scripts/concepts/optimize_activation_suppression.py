#!/usr/bin/env python3
"""Optimize donor-column replacements that suppress SAE feature activation.

For each activating evidence row, this script chooses a matched contrast donor
row from the same dataset, runs leave-one-column-out donor replacement, then
greedily accumulates the most suppressive donor values until the SAE activation
falls below a tolerance or a maximum patch size is reached.

The output is intended to become causal context for concept labeling: original
activating row, matched contrast donor, optimized patched row, and the columns
whose donor replacements most reduced firing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.patch_activation_probe import (
    _column_separation_scores,
    _datasets_for_feature,
    _encode_frame_for_matching,
    _extract_feature_activations,
    _load_evidence_rows,
    _load_raw_mitra_context_query,
)


@dataclass
class SuppressionStep:
    step: int
    column_index: int
    column_name: str
    activation_before: float
    activation_after: float
    delta: float
    cumulative_drop: float
    cumulative_drop_frac: float
    donor_value: str
    original_value: str


@dataclass
class LooImportance:
    direction: str
    column_index: int
    column_name: str
    original_activation: float
    patched_activation: float
    delta: float
    active_value: str
    donor_value: str


@dataclass
class SuppressionResult:
    model: str
    feat: int
    dataset: str
    active_row_idx: int
    donor_row_idx: int
    donor_distance: float
    original_activation: float
    final_activation: float
    total_drop: float
    total_drop_frac: float
    n_selected: int
    stop_reason: str
    selected_columns: list[str]
    selected_column_indices: list[int]
    loo_remove: list[LooImportance]
    loo_add: list[LooImportance]
    steps: list[SuppressionStep]


def _scaled_query(X_query: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X_query, axis=0)
    scale = np.nanstd(X_query, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (np.nan_to_num(X_query, nan=0.0) - med) / scale


def _nearest_donor_except(
    X_scaled: np.ndarray,
    recipient: int,
    donors: np.ndarray,
    *,
    exclude_cols: set[int] | None = None,
) -> tuple[int, float]:
    mask = np.ones(X_scaled.shape[1], dtype=bool)
    for col in exclude_cols or set():
        if 0 <= col < len(mask):
            mask[col] = False
    diff = X_scaled[donors][:, mask] - X_scaled[recipient, mask]
    dist = np.sqrt(np.nanmean(diff * diff, axis=1))
    best = int(np.nanargmin(dist))
    return int(donors[best]), float(dist[best])


def _string_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _evaluate_rows(
    *,
    model: str,
    feat: int,
    dataset: str,
    task: str,
    device: str,
    X_context: pd.DataFrame,
    y_context: np.ndarray,
    rows: list[pd.Series],
) -> np.ndarray:
    return _extract_feature_activations(
        model=model,
        feat=feat,
        X_context=X_context,
        y_context=y_context,
        X_query_batch=pd.DataFrame(rows).reset_index(drop=True),
        task=task,
        dataset=dataset,
        device=device,
    )


def _optimize_one_row(
    *,
    model: str,
    feat: int,
    dataset: str,
    task: str,
    device: str,
    X_context: pd.DataFrame,
    y_context: np.ndarray,
    X_query_raw: pd.DataFrame,
    X_scaled: np.ndarray,
    col_names: list[str],
    active_row: int,
    contrast_rows: np.ndarray,
    candidate_cols: list[int],
    max_cols: int,
    target_drop_frac: float,
    activation_tol: float,
    min_step_drop: float,
) -> SuppressionResult:
    donor_row, donor_distance = _nearest_donor_except(
        X_scaled,
        active_row,
        contrast_rows,
    )
    original_row = X_query_raw.iloc[active_row]
    donor = X_query_raw.iloc[donor_row]

    loo_rows: list[pd.Series] = [original_row, donor]
    remove_cols: list[int] = []
    add_cols: list[int] = []
    for col in candidate_cols:
        if _string_value(original_row.iloc[col]) == _string_value(donor.iloc[col]):
            continue
        remove_patched = original_row.copy()
        remove_patched.iloc[col] = donor.iloc[col]
        loo_rows.append(remove_patched)
        remove_cols.append(col)
    for col in remove_cols:
        add_patched = donor.copy()
        add_patched.iloc[col] = original_row.iloc[col]
        loo_rows.append(add_patched)
        add_cols.append(col)
    acts = _evaluate_rows(
        model=model,
        feat=feat,
        dataset=dataset,
        task=task,
        device=device,
        X_context=X_context,
        y_context=y_context,
        rows=loo_rows,
    )
    original_activation = float(acts[0])
    donor_activation = float(acts[1])
    remove_acts = acts[2:2 + len(remove_cols)]
    add_acts = acts[2 + len(remove_cols):]
    loo_drops = {
        col: original_activation - float(act)
        for col, act in zip(remove_cols, remove_acts)
    }
    loo_remove = [
        LooImportance(
            direction="remove_from_active",
            column_index=int(col),
            column_name=col_names[col],
            original_activation=original_activation,
            patched_activation=float(act),
            delta=original_activation - float(act),
            active_value=_string_value(original_row.iloc[col]),
            donor_value=_string_value(donor.iloc[col]),
        )
        for col, act in zip(remove_cols, remove_acts)
    ]
    loo_add = [
        LooImportance(
            direction="add_to_contrast",
            column_index=int(col),
            column_name=col_names[col],
            original_activation=donor_activation,
            patched_activation=float(act),
            delta=float(act) - donor_activation,
            active_value=_string_value(original_row.iloc[col]),
            donor_value=_string_value(donor.iloc[col]),
        )
        for col, act in zip(add_cols, add_acts)
    ]
    loo_remove.sort(key=lambda item: item.delta, reverse=True)
    loo_add.sort(key=lambda item: item.delta, reverse=True)
    remaining = [
        col
        for col, _ in sorted(loo_drops.items(), key=lambda item: item[1], reverse=True)
        if np.isfinite(loo_drops[col]) and loo_drops[col] > min_step_drop
    ]

    current = original_row.copy()
    current_activation = original_activation
    selected: list[int] = []
    steps: list[SuppressionStep] = []
    stop_reason = "max_cols"
    if current_activation <= activation_tol:
        stop_reason = "already_below_tol"
        remaining = []

    for step_idx in range(1, max_cols + 1):
        if not remaining:
            stop_reason = "no_remaining_columns"
            break
        trial_rows = []
        trial_cols = []
        for col in remaining:
            trial = current.copy()
            trial.iloc[col] = donor.iloc[col]
            trial_rows.append(trial)
            trial_cols.append(col)
        trial_acts = _evaluate_rows(
            model=model,
            feat=feat,
            dataset=dataset,
            task=task,
            device=device,
            X_context=X_context,
            y_context=y_context,
            rows=trial_rows,
        )
        deltas = current_activation - trial_acts
        best_i = int(np.nanargmax(deltas))
        best_col = int(trial_cols[best_i])
        old_activation = current_activation
        best_activation = float(trial_acts[best_i])
        best_delta = float(deltas[best_i])
        if not np.isfinite(best_delta):
            stop_reason = "nonfinite_delta"
            break
        if best_delta <= min_step_drop:
            stop_reason = "no_suppressive_column"
            break
        current.iloc[best_col] = donor.iloc[best_col]
        selected.append(best_col)
        current_activation = best_activation
        cumulative_drop = original_activation - current_activation
        drop_frac = cumulative_drop / original_activation if original_activation > 0 else 0.0
        steps.append(
            SuppressionStep(
                step=step_idx,
                column_index=best_col,
                column_name=col_names[best_col],
                activation_before=float(old_activation),
                activation_after=current_activation,
                delta=best_delta,
                cumulative_drop=float(cumulative_drop),
                cumulative_drop_frac=float(drop_frac),
                donor_value=_string_value(donor.iloc[best_col]),
                original_value=_string_value(original_row.iloc[best_col]),
            )
        )
        remaining = [col for col in remaining if col != best_col]
        if current_activation <= activation_tol:
            stop_reason = "activation_tol"
            break
        if drop_frac >= target_drop_frac:
            stop_reason = "target_drop_frac"
            break

    total_drop = original_activation - current_activation
    total_drop_frac = total_drop / original_activation if original_activation > 0 else 0.0
    return SuppressionResult(
        model=model,
        feat=feat,
        dataset=dataset,
        active_row_idx=int(active_row),
        donor_row_idx=int(donor_row),
        donor_distance=float(donor_distance),
        original_activation=original_activation,
        final_activation=float(current_activation),
        total_drop=float(total_drop),
        total_drop_frac=float(total_drop_frac),
        n_selected=len(selected),
        stop_reason=stop_reason,
        selected_columns=[col_names[col] for col in selected],
        selected_column_indices=selected,
        loo_remove=loo_remove,
        loo_add=loo_add,
        steps=steps,
    )


def optimize_feature_dataset(
    *,
    model: str,
    feat: int,
    dataset: str,
    rows_per_dataset: int,
    candidate_cols: int,
    max_cols: int,
    target_drop_frac: float,
    activation_tol: float,
    min_step_drop: float,
    device: str,
) -> list[SuppressionResult]:
    X_context, y_context, X_query_raw, _, _, task = _load_raw_mitra_context_query(
        model,
        dataset,
    )
    X_query, col_names = _encode_frame_for_matching(X_query_raw)
    evidence = _load_evidence_rows(model, feat, dataset)
    active_rows = evidence[evidence.label == "activating"].row_idx.astype(int).to_numpy()
    contrast_rows = evidence[evidence.label == "contrast"].row_idx.astype(int).to_numpy()
    active_rows = active_rows[(active_rows >= 0) & (active_rows < len(X_query))]
    contrast_rows = contrast_rows[(contrast_rows >= 0) & (contrast_rows < len(X_query))]
    if len(active_rows) == 0 or len(contrast_rows) == 0:
        return []

    X_scaled = _scaled_query(X_query)
    scores = _column_separation_scores(X_query, active_rows, contrast_rows)
    order = np.argsort(-scores)
    cols = [
        int(col)
        for col in order[:candidate_cols]
        if np.isfinite(scores[col]) and scores[col] > 0
    ]
    results = []
    for active_row in active_rows[:rows_per_dataset]:
        results.append(
            _optimize_one_row(
                model=model,
                feat=feat,
                dataset=dataset,
                task=task,
                device=device,
                X_context=X_context,
                y_context=y_context,
                X_query_raw=X_query_raw,
                X_scaled=X_scaled,
                col_names=col_names,
                active_row=int(active_row),
                contrast_rows=contrast_rows,
                candidate_cols=cols,
                max_cols=max_cols,
                target_drop_frac=target_drop_frac,
                activation_tol=activation_tol,
                min_step_drop=min_step_drop,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--features", nargs="+", type=int, required=True)
    parser.add_argument("--datasets-per-feature", type=int, default=2)
    parser.add_argument(
        "--task-filter",
        choices=["all", "classification", "regression"],
        default="all",
    )
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--candidate-cols", type=int, default=12)
    parser.add_argument("--max-cols", type=int, default=5)
    parser.add_argument("--target-drop-frac", type=float, default=0.8)
    parser.add_argument("--activation-tol", type=float, default=1e-4)
    parser.add_argument(
        "--min-step-drop",
        type=float,
        default=0.0,
        help="Minimum absolute activation decrease required to accept a greedy patch step.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "concept_patch_probes"
        / "mitra_suppression_optimized.json",
    )
    args = parser.parse_args()

    all_results: list[SuppressionResult] = []
    errors: list[dict] = []
    for feat in args.features:
        for dataset in _datasets_for_feature(
            args.model,
            feat,
            args.datasets_per_feature,
            args.task_filter,
        ):
            print(f"Optimizing {args.model} f{feat} {dataset}...", flush=True)
            try:
                all_results.extend(
                    optimize_feature_dataset(
                        model=args.model,
                        feat=feat,
                        dataset=dataset,
                        rows_per_dataset=args.rows_per_dataset,
                        candidate_cols=args.candidate_cols,
                        max_cols=args.max_cols,
                        target_drop_frac=args.target_drop_frac,
                        activation_tol=args.activation_tol,
                        min_step_drop=args.min_step_drop,
                        device=args.device,
                    )
                )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR {msg}", flush=True)
                errors.append({"feat": feat, "dataset": dataset, "error": msg})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model": args.model,
            "features": args.features,
            "datasets_per_feature": args.datasets_per_feature,
            "task_filter": args.task_filter,
            "rows_per_dataset": args.rows_per_dataset,
            "candidate_cols": args.candidate_cols,
            "max_cols": args.max_cols,
            "target_drop_frac": args.target_drop_frac,
            "activation_tol": args.activation_tol,
            "min_step_drop": args.min_step_drop,
            "device": args.device,
        },
        "results": [
            {
                **asdict(result),
                "steps": [asdict(step) for step in result.steps],
                "loo_remove": [asdict(item) for item in result.loo_remove],
                "loo_add": [asdict(item) for item in result.loo_add],
            }
            for result in all_results
        ],
        "errors": errors,
    }
    args.out.write_text(json.dumps(payload, indent=2))

    flat_rows = []
    for result in all_results:
        row = asdict(result)
        row.pop("steps")
        row.pop("loo_remove")
        row.pop("loo_add")
        row["selected_columns"] = "|".join(result.selected_columns)
        row["selected_column_indices"] = "|".join(
            str(col) for col in result.selected_column_indices
        )
        flat_rows.append(row)
    df = pd.DataFrame(flat_rows)
    if not df.empty:
        csv_path = args.out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        summary = (
            df.groupby(["feat", "dataset"])
            .agg(
                rows=("active_row_idx", "count"),
                mean_drop_frac=("total_drop_frac", "mean"),
                median_drop_frac=("total_drop_frac", "median"),
                mean_selected=("n_selected", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
        print(f"Wrote {args.out} and {csv_path}")
    else:
        print(f"No optimization results. Wrote {args.out}")


if __name__ == "__main__":
    main()
