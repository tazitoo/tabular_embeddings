#!/usr/bin/env python3
"""Search live Mitra/SAE row patches for concrete feature explanations.

Unlike the contrastive-evidence optimizer, this script does not rely on cached
feature activations. It first scores all query rows through the same live Mitra
inference path used for patches, then searches active/donor pairs and
multi-column donor replacements for patches that suppress a feature.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
from scripts.concepts.optimize_activation_suppression import _scaled_query
from scripts.concepts.patch_activation_probe import (
    _encode_frame_for_matching,
    _load_raw_mitra_context_query,
)
from scripts.intervention.intervene_lib import load_norm_stats, load_sae


@dataclass(frozen=True)
class PairColumn:
    feat: int
    active_row: int
    donor_row: int
    column_index: int
    column_name: str


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def _row_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): _json_value(v) for k, v in row.items()}


def _format_num(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


class LiveMitraSaeEvaluator:
    def __init__(self, *, model: str, dataset: str, task: str, device: str, features: list[int]):
        if model != "mitra":
            raise NotImplementedError("Only --model mitra is supported")
        self.model = model
        self.dataset = dataset
        self.task = task
        self.device = device
        self.features = features
        self.extractor = MitraEmbeddingExtractor(device=device, n_estimators=1, fine_tune=False)
        self.mean, self.std = load_norm_stats(model, dataset, device=device)
        self.sae, _ = load_sae(model, device=device)

    def activations(
        self,
        *,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        rows: pd.DataFrame,
    ) -> np.ndarray:
        result = self.extractor.extract_embeddings(
            X_context,
            y_context,
            rows.reset_index(drop=True),
            task=self.task,
        )
        emb = np.asarray(result.embeddings, dtype=np.float32)
        with torch.no_grad():
            x = torch.tensor(emb, dtype=torch.float32, device=self.device)
            x = (x - self.mean) / self.std
            acts = self.sae.encode(x)[:, self.features].detach().cpu().numpy()
        return acts


def _eval_in_chunks(
    evaluator: LiveMitraSaeEvaluator,
    *,
    X_context: pd.DataFrame,
    y_context: np.ndarray,
    rows: list[pd.Series],
    chunk_size: int,
) -> np.ndarray:
    outs = []
    for start in range(0, len(rows), chunk_size):
        batch = pd.DataFrame(rows[start:start + chunk_size]).reset_index(drop=True)
        outs.append(
            evaluator.activations(
                X_context=X_context,
                y_context=y_context,
                rows=batch,
            )
        )
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(evaluator.features)))


def _nearest_donors(
    X_scaled: np.ndarray,
    active_row: int,
    donor_rows: np.ndarray,
    n: int,
) -> list[tuple[int, float]]:
    diff = X_scaled[donor_rows] - X_scaled[active_row]
    dist = np.sqrt(np.nanmean(diff * diff, axis=1))
    order = np.argsort(dist)
    return [(int(donor_rows[i]), float(dist[i])) for i in order[:n]]


def _different_columns(
    active: pd.Series,
    donor: pd.Series,
    max_cols: int | None,
    exclude_cols: set[int] | None = None,
) -> list[int]:
    cols = [
        i
        for i, (a, d) in enumerate(zip(active.iloc[:], donor.iloc[:]))
        if str(a) != str(d) and i not in (exclude_cols or set())
    ]
    return cols[:max_cols] if max_cols else cols


def _patched_row(active: pd.Series, donor: pd.Series, cols: list[int]) -> pd.Series:
    patched = active.copy()
    for col in cols:
        patched.iloc[col] = donor.iloc[col]
    return patched


def _render_markdown(payload: dict) -> str:
    lines = [
        f"# {payload['model']} {payload['dataset']} Causal Patch Explanations",
        "",
        "Rows and patches were selected from live Mitra/SAE inference, not cached activations.",
        "",
    ]
    for item in payload["explanations"]:
        lines.extend(
            [
                f"## f{item['feat']}",
                "",
                f"- status: {item['status']}",
            ]
        )
        if item["status"] != "found":
            lines.append(f"- reason: {item.get('reason', '')}")
            lines.append("")
            continue
        lines.extend(
            [
                f"- active row: {item['active_row_idx']}",
                f"- donor row: {item['donor_row_idx']}",
                f"- original activation: {_format_num(item['original_activation'])}",
                f"- patched activation: {_format_num(item['patched_activation'])}",
                f"- drop fraction: {_format_num(item['drop_frac'])}",
                f"- patch size: {len(item['patch_fields'])}",
                "",
                "| field | active value | donor value |",
                "| --- | --- | --- |",
            ]
        )
        for field in item["patch_fields"]:
            lines.append(
                f"| `{field['column']}` | {field['active_value']} | {field['donor_value']} |"
            )
        lines.extend(
            [
                "",
                "Top one-field suppressions from the selected active/donor pair:",
                "",
                "| field | original act | patched act | delta | active value | donor value |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for loo in item["top_loo_remove"]:
            lines.append(
                f"| `{loo['column']}` | {_format_num(loo['original_activation'])} | "
                f"{_format_num(loo['patched_activation'])} | {_format_num(loo['delta'])} | "
                f"{loo['active_value']} | {loo['donor_value']} |"
            )
        lines.extend(["", "Row differences:", ""])
        lines.extend(
            [
                "| field | active row | donor row | patched active row |",
                "| --- | --- | --- | --- |",
            ]
        )
        for diff in item["row_differences"]:
            marker = "**" if diff["patched"] else ""
            lines.append(
                f"| {marker}`{diff['column']}`{marker} | {diff['active_value']} | "
                f"{diff['donor_value']} | {diff['patched_value']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def search(args: argparse.Namespace) -> dict:
    X_context, y_context, X_query_raw, _, _, task = _load_raw_mitra_context_query(
        args.model,
        args.dataset,
    )
    X_query_num, col_names = _encode_frame_for_matching(X_query_raw)
    X_scaled = _scaled_query(X_query_num)
    excluded_columns = set(args.exclude_columns or [])
    excluded_col_indices = {
        idx for idx, name in enumerate(col_names) if name in excluded_columns
    }

    evaluator = LiveMitraSaeEvaluator(
        model=args.model,
        dataset=args.dataset,
        task=task,
        device=args.device,
        features=args.features,
    )
    live_acts = _eval_in_chunks(
        evaluator,
        X_context=X_context,
        y_context=y_context,
        rows=[row for _, row in X_query_raw.iterrows()],
        chunk_size=args.eval_chunk_size,
    )

    explanations = []
    for feat_pos, feat in enumerate(args.features):
        feat_acts = live_acts[:, feat_pos]
        active_candidates = np.flatnonzero(feat_acts > args.activation_tol)
        if len(active_candidates) == 0:
            explanations.append(
                {
                    "feat": feat,
                    "status": "not_found",
                    "reason": "no live activating credit-g query rows above activation tolerance",
                    "max_live_activation": float(np.max(feat_acts)) if len(feat_acts) else None,
                }
            )
            continue

        donor_candidates = np.flatnonzero(feat_acts <= args.donor_activation_max)
        if len(donor_candidates) < args.min_donors:
            donor_candidates = np.argsort(feat_acts)[: max(args.min_donors, args.donors_per_active)]

        active_order = active_candidates[np.argsort(-feat_acts[active_candidates])]
        active_rows = active_order[:args.active_rows]

        patch_rows: list[pd.Series] = []
        patch_meta: list[PairColumn] = []
        pair_distances: dict[tuple[int, int], float] = {}
        for active_row in active_rows:
            donors = _nearest_donors(
                X_scaled,
                int(active_row),
                donor_candidates,
                args.donors_per_active,
            )
            for donor_row, distance in donors:
                pair_distances[(int(active_row), int(donor_row))] = distance
                active = X_query_raw.iloc[int(active_row)]
                donor = X_query_raw.iloc[int(donor_row)]
                for col in _different_columns(
                    active,
                    donor,
                    args.candidate_cols,
                    excluded_col_indices,
                ):
                    patch_rows.append(_patched_row(active, donor, [col]))
                    patch_meta.append(
                        PairColumn(
                            feat=feat,
                            active_row=int(active_row),
                            donor_row=int(donor_row),
                            column_index=int(col),
                            column_name=col_names[col],
                        )
                    )

        if not patch_rows:
            explanations.append(
                {
                    "feat": feat,
                    "status": "not_found",
                    "reason": "no differing active/donor columns to patch",
                    "max_live_activation": float(np.max(feat_acts)),
                }
            )
            continue

        patched_acts_all = _eval_in_chunks(
            evaluator,
            X_context=X_context,
            y_context=y_context,
            rows=patch_rows,
            chunk_size=args.eval_chunk_size,
        )
        by_pair: dict[tuple[int, int], list[dict]] = {}
        for meta, patched_acts in zip(patch_meta, patched_acts_all):
            original_activation = float(feat_acts[meta.active_row])
            patched_activation = float(patched_acts[feat_pos])
            delta = original_activation - patched_activation
            active = X_query_raw.iloc[meta.active_row]
            donor = X_query_raw.iloc[meta.donor_row]
            by_pair.setdefault((meta.active_row, meta.donor_row), []).append(
                {
                    "column_index": meta.column_index,
                    "column": meta.column_name,
                    "original_activation": original_activation,
                    "patched_activation": patched_activation,
                    "delta": float(delta),
                    "active_value": _json_value(active.iloc[meta.column_index]),
                    "donor_value": _json_value(donor.iloc[meta.column_index]),
                }
            )

        best_pair = None
        best_initial_score = -np.inf
        for pair, items in by_pair.items():
            positive = [item for item in items if item["delta"] > args.min_step_drop]
            if not positive:
                continue
            original_activation = float(feat_acts[pair[0]])
            best_delta = max(item["delta"] for item in positive)
            pair_score = best_delta / max(original_activation, 1e-8)
            pair_score -= 0.01 * pair_distances.get(pair, 0.0)
            if pair_score > best_initial_score:
                best_initial_score = pair_score
                best_pair = pair

        if best_pair is None:
            explanations.append(
                {
                    "feat": feat,
                    "status": "not_found",
                    "reason": "no one-field donor replacement reduced live activation",
                    "max_live_activation": float(np.max(feat_acts)),
                    "n_live_active_rows": int(len(active_candidates)),
                }
            )
            continue

        active_row, donor_row = best_pair
        active = X_query_raw.iloc[active_row]
        donor = X_query_raw.iloc[donor_row]
        original_activation = float(feat_acts[active_row])
        ranked = sorted(by_pair[best_pair], key=lambda item: item["delta"], reverse=True)
        selected_cols: list[int] = []
        selected_fields: list[dict] = []
        current_activation = original_activation
        current = active.copy()
        for item in ranked:
            if len(selected_cols) >= args.max_patch_cols:
                break
            if item["delta"] <= args.min_step_drop:
                continue
            trial = current.copy()
            trial.iloc[item["column_index"]] = donor.iloc[item["column_index"]]
            trial_act = _eval_in_chunks(
                evaluator,
                X_context=X_context,
                y_context=y_context,
                rows=[trial],
                chunk_size=1,
            )[0, feat_pos]
            if float(trial_act) < current_activation - args.min_step_drop:
                current = trial
                current_activation = float(trial_act)
                selected_cols.append(item["column_index"])
                selected_fields.append(
                    {
                        "column": item["column"],
                        "active_value": item["active_value"],
                        "donor_value": item["donor_value"],
                        "activation_after_step": current_activation,
                    }
                )
            drop_frac = (original_activation - current_activation) / max(original_activation, 1e-8)
            if current_activation <= args.activation_tol or drop_frac >= args.target_drop_frac:
                break

        row_diffs = []
        selected_set = set(selected_cols)
        for col, name in enumerate(col_names):
            if str(active.iloc[col]) == str(donor.iloc[col]) and col not in selected_set:
                continue
            row_diffs.append(
                {
                    "column": name,
                    "active_value": _json_value(active.iloc[col]),
                    "donor_value": _json_value(donor.iloc[col]),
                    "patched_value": _json_value(current.iloc[col]),
                    "patched": col in selected_set,
                }
            )

        explanations.append(
            {
                "feat": feat,
                "status": "found" if selected_fields else "not_found",
                "reason": "" if selected_fields else "no greedy multi-column patch improved activation",
                "active_row_idx": int(active_row),
                "donor_row_idx": int(donor_row),
                "donor_distance": pair_distances.get(best_pair),
                "original_activation": original_activation,
                "patched_activation": current_activation,
                "drop": original_activation - current_activation,
                "drop_frac": (original_activation - current_activation)
                / max(original_activation, 1e-8),
                "patch_fields": selected_fields,
                "top_loo_remove": ranked[:args.top_loo],
                "active_row": _row_dict(active),
                "donor_row": _row_dict(donor),
                "patched_row": _row_dict(current),
                "row_differences": row_diffs,
                "n_live_active_rows": int(len(active_candidates)),
                "max_live_activation": float(np.max(feat_acts)),
            }
        )

    return {
        "model": args.model,
        "dataset": args.dataset,
        "features": args.features,
        "config": {
            "active_rows": args.active_rows,
            "donors_per_active": args.donors_per_active,
            "candidate_cols": args.candidate_cols,
            "max_patch_cols": args.max_patch_cols,
            "target_drop_frac": args.target_drop_frac,
            "activation_tol": args.activation_tol,
            "donor_activation_max": args.donor_activation_max,
            "min_step_drop": args.min_step_drop,
            "eval_chunk_size": args.eval_chunk_size,
            "device": args.device,
            "exclude_columns": sorted(excluded_columns),
        },
        "explanations": explanations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", nargs="+", type=int, required=True)
    parser.add_argument("--active-rows", type=int, default=25)
    parser.add_argument("--donors-per-active", type=int, default=25)
    parser.add_argument("--min-donors", type=int, default=10)
    parser.add_argument("--candidate-cols", type=int, default=None)
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=None,
        help="Column names to exclude from candidate patches.",
    )
    parser.add_argument("--max-patch-cols", type=int, default=8)
    parser.add_argument("--target-drop-frac", type=float, default=0.8)
    parser.add_argument("--activation-tol", type=float, default=1e-4)
    parser.add_argument("--donor-activation-max", type=float, default=1e-4)
    parser.add_argument("--min-step-drop", type=float, default=0.0)
    parser.add_argument("--top-loo", type=int, default=8)
    parser.add_argument("--eval-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "causal_patch_search.json",
    )
    args = parser.parse_args()

    payload = search(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    md_path = args.out.with_suffix(".md")
    md_path.write_text(_render_markdown(payload))
    for item in payload["explanations"]:
        if item["status"] == "found":
            print(
                f"f{item['feat']}: row {item['active_row_idx']} -> donor {item['donor_row_idx']} "
                f"{_format_num(item['original_activation'])} -> {_format_num(item['patched_activation'])} "
                f"cols={','.join(field['column'] for field in item['patch_fields'])}"
            )
        else:
            print(f"f{item['feat']}: not_found ({item.get('reason')})")
    print(f"Wrote {args.out} and {md_path}")


if __name__ == "__main__":
    main()
