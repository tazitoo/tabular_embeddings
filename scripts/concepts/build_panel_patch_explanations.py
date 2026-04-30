#!/usr/bin/env python3
"""Build one live causal patch explanation per top panel feature.

Consumes `rank_panel_intervention_features.py` output. For each panel dataset
and each ranked feature, the script tries the representative intervened rows in
rank order and keeps the first high-quality donor-replacement patch found.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.optimize_activation_suppression import _scaled_query
from scripts.concepts.patch_activation_probe import (
    _encode_frame_for_matching,
    _load_raw_mitra_context_query,
)
from scripts.concepts.search_causal_patch_explanations import (
    LiveMitraSaeEvaluator,
    _different_columns,
    _eval_in_chunks,
    _format_num,
    _json_value,
    _nearest_donors,
    _patched_row,
    _row_dict,
)


def _candidate_positions(example_rows: list[dict], abs_to_pos: dict[int, int]) -> list[dict]:
    out = []
    seen = set()
    for row in example_rows:
        abs_idx = int(row["row_idx"])
        if abs_idx not in abs_to_pos:
            continue
        pos = abs_to_pos[abs_idx]
        if pos in seen:
            continue
        seen.add(pos)
        out.append(
            {
                "row_idx": abs_idx,
                "query_pos": int(pos),
                "rank_in_row": int(row.get("rank_in_row", -1)),
            }
        )
    return out


def _row_differences(
    *,
    active: pd.Series,
    donor: pd.Series,
    patched: pd.Series,
    col_names: list[str],
    selected_cols: set[int],
) -> list[dict]:
    diffs = []
    for col, name in enumerate(col_names):
        if str(active.iloc[col]) == str(donor.iloc[col]) and col not in selected_cols:
            continue
        diffs.append(
            {
                "column": name,
                "active_value": _json_value(active.iloc[col]),
                "donor_value": _json_value(donor.iloc[col]),
                "patched_value": _json_value(patched.iloc[col]),
                "patched": col in selected_cols,
            }
        )
    return diffs


def _search_one_feature(
    *,
    feat: int,
    feat_pos: int,
    candidates: list[dict],
    feat_acts: np.ndarray,
    evaluator: LiveMitraSaeEvaluator,
    X_context: pd.DataFrame,
    y_context: np.ndarray,
    X_query_raw: pd.DataFrame,
    X_scaled: np.ndarray,
    col_names: list[str],
    args: argparse.Namespace,
) -> dict:
    donor_candidates = np.flatnonzero(feat_acts <= args.donor_activation_max)
    if len(donor_candidates) < args.min_donors:
        donor_candidates = np.argsort(feat_acts)[: max(args.min_donors, args.donors_per_row)]

    attempts = []
    for candidate in candidates[: args.fallback_rows]:
        active_pos = int(candidate["query_pos"])
        original_activation = float(feat_acts[active_pos])
        if original_activation <= args.activation_tol:
            attempts.append(
                {
                    **candidate,
                    "status": "skipped",
                    "reason": "representative row is not live-active",
                    "activation": original_activation,
                }
            )
            continue

        active = X_query_raw.iloc[active_pos]
        donors = _nearest_donors(
            X_scaled,
            active_pos,
            donor_candidates,
            args.donors_per_row,
        )

        patch_rows = []
        patch_meta = []
        pair_distance = {}
        for donor_pos, distance in donors:
            donor = X_query_raw.iloc[donor_pos]
            pair_distance[(active_pos, donor_pos)] = distance
            for col in _different_columns(active, donor, args.candidate_cols):
                patch_rows.append(_patched_row(active, donor, [col]))
                patch_meta.append((donor_pos, col))

        if not patch_rows:
            attempts.append(
                {
                    **candidate,
                    "status": "skipped",
                    "reason": "no differing donor columns",
                    "activation": original_activation,
                }
            )
            continue

        patched_acts = _eval_in_chunks(
            evaluator,
            X_context=X_context,
            y_context=y_context,
            rows=patch_rows,
            chunk_size=args.eval_chunk_size,
        )

        by_pair: dict[tuple[int, int], list[dict]] = {}
        for (donor_pos, col), act_row in zip(patch_meta, patched_acts):
            patched_activation = float(act_row[feat_pos])
            donor = X_query_raw.iloc[donor_pos]
            item = {
                "column_index": int(col),
                "column": col_names[col],
                "original_activation": original_activation,
                "patched_activation": patched_activation,
                "delta": original_activation - patched_activation,
                "active_value": _json_value(active.iloc[col]),
                "donor_value": _json_value(donor.iloc[col]),
            }
            by_pair.setdefault((active_pos, donor_pos), []).append(item)

        best_pair = None
        best_score = -math.inf
        for pair, items in by_pair.items():
            positive = [item for item in items if item["delta"] > args.min_step_drop]
            if not positive:
                continue
            best_delta = max(item["delta"] for item in positive)
            score = best_delta / max(original_activation, 1e-8)
            score -= 0.01 * pair_distance.get(pair, 0.0)
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            attempts.append(
                {
                    **candidate,
                    "status": "skipped",
                    "reason": "no suppressive one-field patch",
                    "activation": original_activation,
                }
            )
            continue

        _, donor_pos = best_pair
        donor = X_query_raw.iloc[donor_pos]
        ranked = sorted(by_pair[best_pair], key=lambda item: item["delta"], reverse=True)
        selected_cols = []
        selected_fields = []
        current = active.copy()
        current_activation = original_activation
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

        drop_frac = (original_activation - current_activation) / max(original_activation, 1e-8)
        if selected_fields and drop_frac >= args.min_drop_frac:
            return {
                "feat": int(feat),
                "status": "found",
                "source_row_idx": int(candidate["row_idx"]),
                "active_query_pos": active_pos,
                "donor_query_pos": int(donor_pos),
                "donor_distance": pair_distance.get(best_pair),
                "original_activation": original_activation,
                "patched_activation": current_activation,
                "drop": original_activation - current_activation,
                "drop_frac": drop_frac,
                "patch_fields": selected_fields,
                "top_loo_remove": ranked[: args.top_loo],
                "active_row": _row_dict(active),
                "donor_row": _row_dict(donor),
                "patched_row": _row_dict(current),
                "row_differences": _row_differences(
                    active=active,
                    donor=donor,
                    patched=current,
                    col_names=col_names,
                    selected_cols=set(selected_cols),
                ),
                "attempts": attempts,
            }

        attempts.append(
            {
                **candidate,
                "status": "skipped",
                "reason": "patch did not meet minimum drop fraction",
                "activation": original_activation,
                "best_patched_activation": current_activation,
                "best_drop_frac": drop_frac,
            }
        )

    return {
        "feat": int(feat),
        "status": "not_found",
        "reason": "no representative row produced a qualifying patch",
        "attempts": attempts,
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Panel Patch Explanations",
        "",
        "One live Mitra/SAE donor-replacement patch is selected per top panel feature.",
        "",
    ]
    for panel in payload["panels"]:
        lines.extend(
            [
                f"## {panel['dataset']}",
                "",
                "| rank | feature | status | patch | original act | patched act | drop frac | row | donor |",
                "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in panel["explanations"]:
            if item["status"] == "found":
                patch = "; ".join(
                    f"`{field['column']}`: {field['active_value']} -> {field['donor_value']}"
                    for field in item["patch_fields"]
                )
                lines.append(
                    f"| {item['rank']} | f{item['feat']} | found | {patch} | "
                    f"{_format_num(item['original_activation'])} | "
                    f"{_format_num(item['patched_activation'])} | "
                    f"{_format_num(item['drop_frac'])} | "
                    f"{item['source_row_idx']} | {item['donor_query_pos']} |"
                )
            else:
                lines.append(
                    f"| {item['rank']} | f{item['feat']} | not found | {item.get('reason', '')} | | | | | |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build(args: argparse.Namespace) -> dict:
    ranking = json.loads(args.ranking.read_text())
    panels = []
    for panel in ranking["panels"]:
        dataset = panel["dataset"]
        top_features = panel["top_features"][: args.top_k]
        features = [int(item["feature"]) for item in top_features]
        print(f"{dataset}: loading live activations for {len(features)} features", flush=True)

        X_context, y_context, X_query_raw, _, row_indices, task = _load_raw_mitra_context_query(
            args.model,
            dataset,
        )
        abs_to_pos = {int(abs_idx): int(pos) for pos, abs_idx in enumerate(row_indices)}
        X_query_num, col_names = _encode_frame_for_matching(X_query_raw)
        X_scaled = _scaled_query(X_query_num)
        evaluator = LiveMitraSaeEvaluator(
            model=args.model,
            dataset=dataset,
            task=task,
            device=args.device,
            features=features,
        )
        live_acts = _eval_in_chunks(
            evaluator,
            X_context=X_context,
            y_context=y_context,
            rows=[row for _, row in X_query_raw.iterrows()],
            chunk_size=args.eval_chunk_size,
        )

        explanations = []
        for feat_pos, feature_item in enumerate(top_features):
            feat = int(feature_item["feature"])
            candidates = _candidate_positions(feature_item["example_rows"], abs_to_pos)
            result = _search_one_feature(
                feat=feat,
                feat_pos=feat_pos,
                candidates=candidates,
                feat_acts=live_acts[:, feat_pos],
                evaluator=evaluator,
                X_context=X_context,
                y_context=y_context,
                X_query_raw=X_query_raw,
                X_scaled=X_scaled,
                col_names=col_names,
                args=args,
            )
            result.update(
                {
                    "rank": len(explanations) + 1,
                    "selection_count": int(feature_item["count"]),
                    "selection_first_count": int(feature_item["first_count"]),
                    "selection_mean_rank": float(feature_item["mean_rank"]),
                }
            )
            explanations.append(result)
            if result["status"] == "found":
                patch = ", ".join(field["column"] for field in result["patch_fields"])
                print(
                    f"  f{feat}: found {patch} "
                    f"{_format_num(result['original_activation'])}->{_format_num(result['patched_activation'])}",
                    flush=True,
                )
            else:
                print(f"  f{feat}: not found", flush=True)

        panels.append(
            {
                "dataset": dataset,
                "strong_model": panel["strong_model"],
                "weak_model": panel["weak_model"],
                "drawn_rows": panel["drawn_rows"],
                "explanations": explanations,
            }
        )

    return {
        "model": args.model,
        "ranking": str(args.ranking),
        "config": {
            "top_k": args.top_k,
            "fallback_rows": args.fallback_rows,
            "donors_per_row": args.donors_per_row,
            "candidate_cols": args.candidate_cols,
            "max_patch_cols": args.max_patch_cols,
            "target_drop_frac": args.target_drop_frac,
            "min_drop_frac": args.min_drop_frac,
            "activation_tol": args.activation_tol,
            "donor_activation_max": args.donor_activation_max,
            "eval_chunk_size": args.eval_chunk_size,
            "device": args.device,
        },
        "panels": panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "panel_intervention_feature_rankings_v1.json",
    )
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fallback-rows", type=int, default=10)
    parser.add_argument("--donors-per-row", type=int, default=40)
    parser.add_argument("--min-donors", type=int, default=10)
    parser.add_argument("--candidate-cols", type=int, default=None)
    parser.add_argument("--max-patch-cols", type=int, default=3)
    parser.add_argument("--target-drop-frac", type=float, default=0.8)
    parser.add_argument("--min-drop-frac", type=float, default=0.5)
    parser.add_argument("--activation-tol", type=float, default=1e-4)
    parser.add_argument("--donor-activation-max", type=float, default=1e-4)
    parser.add_argument("--min-step-drop", type=float, default=0.0)
    parser.add_argument("--top-loo", type=int, default=5)
    parser.add_argument("--eval-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "panel_patch_explanations_v1.json",
    )
    args = parser.parse_args()

    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    md_path = args.out.with_suffix(".md")
    md_path.write_text(_render_markdown(payload))
    print(f"Wrote {args.out} and {md_path}")


if __name__ == "__main__":
    main()
