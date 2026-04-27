#!/usr/bin/env python3
"""Run one fresh contrastive-mesh experiment via Codex-driven microcalls.

This is the hardened fresh-generation path for environments that should use the
top-level Codex CLI only. It drives the existing state machine in
`scripts.concepts.label_contrastive_mesh` via:

- `init`
- `next-task`
- `record`
- `record-judge`
- `record-validator`

Each worker / judge / validator step is a separate `codex exec` call with a
task-local JSON schema, prompt artifact, response artifact, and retry loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_WORKER_MODEL = "gpt-5.4-mini"
DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_VALIDATOR_MODEL = "gpt-5.4"
VALID_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
VALID_JUDGE_PROMPT_FAMILIES = (
    "baseline_v2",
    "judge_concrete_heterogeneity_fallback",
    "judge_positive_anchor_v1",
    "judge_positive_boundary_v1",
    "judge_generalization_v1",
)


@dataclass
class CallRecord:
    role: str
    model: str
    reasoning_effort: str | None
    round_num: int | None
    dataset: str | None
    prompt_chars: int
    tokens_used: int | None
    attempts: int
    task_dir: str
    response_file: str


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "task"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _mesh_cmd(model: str, feat: int, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.concepts.label_contrastive_mesh",
        "--model",
        model,
        "--feat",
        str(feat),
        *extra,
    ]


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        input=stdin,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _run_mesh_json(model: str, feat: int, *extra: str) -> dict:
    proc = _run_cmd(_mesh_cmd(model, feat, *extra))
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Expected JSON from mesh CLI for {' '.join(extra)}.\nstdout:\n{proc.stdout}"
        ) from exc


def _extract_turn_usage(jsonl_stdout: str) -> int | None:
    total: int | None = None
    for raw in jsonl_stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "turn.completed":
            continue
        usage = payload.get("usage") or {}
        if "total_tokens" in usage and usage["total_tokens"] is not None:
            total = int(usage["total_tokens"])
            continue
        inp = usage.get("input_tokens") or 0
        out = usage.get("output_tokens") or 0
        total = int(inp) + int(out)
    return total


def _worker_schema(
    agent_band_prediction: bool = False,
    causal_patch_plan: bool = False,
) -> dict:
    base_props: dict[str, object] = {
        "label": {
            "type": "string",
            "minLength": 1,
        }
    }
    required = ["label"]
    if causal_patch_plan:
        base_props.update(
            {
                "local_hypothesis": {"type": "string"},
                "patch_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "patch_rationale": {"type": "string"},
            }
        )
        required.extend(["local_hypothesis", "patch_columns", "patch_rationale"])
    if agent_band_prediction:
        strength_item = {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_id", "bin"],
            "properties": {
                "evidence_id": {"type": "string"},
                "bin": {
                    "type": "string",
                    "enum": ["strong_positive", "medium_positive", "weak_positive"],
                },
            },
        }
        contrast_item = {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_id", "bin"],
            "properties": {
                "evidence_id": {"type": "string"},
                "bin": {
                    "type": "string",
                    "enum": ["near_contrast", "mid_contrast", "far_contrast"],
                },
            },
        }
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *required,
                "positive_strength",
                "contrast_closeness",
                "rank_rationale",
            ],
            "properties": {
                **base_props,
                "positive_strength": {"type": "array", "items": strength_item},
                "contrast_closeness": {"type": "array", "items": contrast_item},
                "rank_rationale": {"type": "string"},
            },
        }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": base_props,
    }


def _judge_schema(
    datasets: list[str],
    pairings_mode: str,
    judge_prompt_family: str = "baseline_v2",
    require_final_label: bool = False,
    causal_patch_plan: bool = False,
) -> dict:
    claim = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claim", "certainty"],
        "properties": {
            "claim": {"type": "string"},
            "certainty": {
                "type": "string",
                "enum": [
                    "local observation",
                    "emerging pattern",
                    "stable cross-round pattern",
                ],
            },
        },
    }
    portability = {
        "type": "object",
        "additionalProperties": False,
        "required": ["level", "justification"],
        "properties": {
            "level": {"type": "string", "enum": ["low", "medium", "high"]},
            "justification": {"type": "string"},
        },
    }
    dataset_block = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "verdict",
            "supported_claims",
            "unsupported_claims",
            "contradicted_claims",
            "missing_signal",
            "portability_risk",
            "suggested_revision",
        ],
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "revise", "split"]},
            "supported_claims": {"type": "array", "items": claim},
            "unsupported_claims": {"type": "array", "items": claim},
            "contradicted_claims": {"type": "array", "items": claim},
            "missing_signal": {
                "anyOf": [
                    claim,
                    {"type": "null"},
                ]
            },
            "portability_risk": portability,
            "suggested_revision": {"type": "string"},
        },
    }
    props: dict[str, object] = {
        "overall_verdict": {"type": "string", "enum": ["done", "continue"]},
        "overall_note": {"type": "string"},
        "final_label": {"type": "string"} if require_final_label else {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "per_dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": datasets,
            "properties": {ds: dataset_block for ds in datasets},
        },
    }
    required = ["overall_verdict", "overall_note", "final_label", "per_dataset"]
    if causal_patch_plan:
        props["causal_patch_plan"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "hypothesis_id",
                    "dataset",
                    "agent_id",
                    "source_dataset",
                    "portable_hypothesis",
                    "local_hypothesis",
                    "columns",
                    "expected_add_delta",
                    "expected_remove_delta",
                ],
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "dataset": {"type": "string", "enum": datasets},
                    "agent_id": {"type": "string"},
                    "source_dataset": {"type": "string", "enum": datasets},
                    "portable_hypothesis": {"type": "string"},
                    "local_hypothesis": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "expected_add_delta": {
                        "type": "string",
                        "enum": ["increase", "decrease", "no_change", "unclear"],
                    },
                    "expected_remove_delta": {
                        "type": "string",
                        "enum": ["increase", "decrease", "no_change", "unclear"],
                    },
                },
            },
        }
        required.append("causal_patch_plan")
    if judge_prompt_family == "judge_positive_boundary_v1":
        props["positive_evidence"] = {"type": "string"}
        props["contrast_boundary"] = {"type": "string"}
        required.extend(["positive_evidence", "contrast_boundary"])
    if judge_prompt_family == "judge_generalization_v1":
        props["positive_evidence"] = {"type": "string"}
        props["contrast_calibration"] = {"type": "string"}
        props["generalization_rationale"] = {"type": "string"}
        required.extend(
            [
                "positive_evidence",
                "contrast_calibration",
                "generalization_rationale",
            ]
        )
    if pairings_mode == "on":
        props["next_round_pairings"] = {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": datasets},
        }
        required.append("next_round_pairings")
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def _validator_schema(datasets: list[str]) -> dict:
    pred = {
        "type": "object",
        "additionalProperties": False,
        "required": ["row_id", "prediction"],
        "properties": {
            "row_id": {"type": "string"},
            "prediction": {"type": "string", "enum": ["fires", "does_not_fire"]},
        },
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": ["per_dataset", "overall_note"],
        "properties": {
            "overall_note": {"type": "string"},
            "per_dataset": {
                "type": "object",
                "additionalProperties": False,
                "required": datasets,
                "properties": {ds: {"type": "array", "items": pred} for ds in datasets},
            },
        },
    }


def _f1(tp: int, fp: int, fn: int) -> float | None:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or not (precision + recall):
        return None
    return 2 * precision * recall / (precision + recall)


def _dev_validator_packet(
    *,
    model: str,
    feat: int,
    label: str,
    round_num: int,
    n_act: int,
    n_con: int,
) -> tuple[str, dict[str, dict[str, bool]]]:
    from scripts.concepts.label_contrastive_mesh import ContrastiveMeshPipeline

    pipe = ContrastiveMeshPipeline(model, feat)
    sections: list[str] = []
    truth: dict[str, dict[str, bool]] = {}
    for ds in pipe.datasets:
        samples = pipe._sample_rows_for(
            round_num,
            ds,
            n_act,
            n_con,
            purpose="dev_validator",
        )
        rows: list[tuple[bool, dict]] = []
        if samples["act"].strip():
            act_df = pd.read_csv(StringIO(samples["act"]))
            rows.extend((True, row.to_dict()) for _, row in act_df.iterrows())
        if samples["con"].strip():
            con_df = pd.read_csv(StringIO(samples["con"]))
            rows.extend((False, row.to_dict()) for _, row in con_df.iterrows())

        ds_truth: dict[str, bool] = {}
        prompt_rows: list[dict] = []
        for idx, (fires, row) in enumerate(rows):
            row_id = f"r{idx:03d}"
            ds_truth[row_id] = fires
            clean = {"row_id": row_id}
            clean.update({k: v for k, v in row.items() if k != "label"})
            prompt_rows.append(clean)
        truth[ds] = ds_truth
        sections.append(f"=== {ds} ===\n{pd.DataFrame(prompt_rows).to_csv(index=False)}")

    data_block = "\n\n".join(sections)
    prompt = (
        f"You are a development validator for SAE feature f_{feat}.\n"
        "These rows are sampled from the current concept-labeling evidence, not from the final "
        "holdout validator. Your job is to estimate whether the provisional label classifies "
        "concept-present and concept-absent examples correctly.\n\n"
        f"PROVISIONAL LABEL\n\"{label}\"\n\n"
        "Row cells use the same marginal annotations as the labeling prompt. Rows are shuffled "
        "within each dataset. Decide each row independently.\n\n"
        f"{data_block}\n\n"
        "TASK\n"
        "For each row_id, predict whether the row fits the PROVISIONAL LABEL:\n"
        "  * \"fires\" if the row matches the label's concept-present structure.\n"
        "  * \"does_not_fire\" otherwise.\n\n"
        "OUTPUT FORMAT (strict)\n"
        "Single fenced ```json``` block. Nothing before or after.\n\n"
        "```json\n"
        "{\n"
        '  "overall_note": "<brief note>",\n'
        '  "per_dataset": {\n'
        '    "<dataset>": [\n'
        '      {"row_id": "r000", "prediction": "fires"}\n'
        "    ]\n"
        "  }\n"
        "}\n"
        "```"
    )
    return prompt, truth


def _grade_dev_validator_response(response: dict, truth: dict[str, dict[str, bool]]) -> dict:
    per_ds = response.get("per_dataset") or {}
    per_dataset: dict[str, dict] = {}
    total_tp = total_fp = total_tn = total_fn = 0
    for ds, ds_truth in truth.items():
        preds = {
            item.get("row_id"): str(item.get("prediction", "")).strip().lower() == "fires"
            for item in per_ds.get(ds, [])
        }
        tp = fp = tn = fn = 0
        false_positives: list[str] = []
        false_negatives: list[str] = []
        for row_id, actual in ds_truth.items():
            pred = bool(preds.get(row_id, False))
            if actual and pred:
                tp += 1
            elif actual and not pred:
                fn += 1
                false_negatives.append(row_id)
            elif (not actual) and pred:
                fp += 1
                false_positives.append(row_id)
            else:
                tn += 1
        total = tp + fp + tn + fn
        per_dataset[ds] = {
            "accuracy": (tp + tn) / total if total else 0.0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "f1": _f1(tp, fp, fn),
            "false_positive_row_ids": false_positives,
            "false_negative_row_ids": false_negatives,
        }
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
    total = total_tp + total_fp + total_tn + total_fn
    return {
        "overall": {
            "accuracy": (total_tp + total_tn) / total if total else 0.0,
            "accuracy_macro": (
                sum(s["accuracy"] for s in per_dataset.values()) / len(per_dataset)
                if per_dataset else 0.0
            ),
            "f1": _f1(total_tp, total_fp, total_fn),
            "tp": total_tp,
            "fp": total_fp,
            "tn": total_tn,
            "fn": total_fn,
        },
        "per_dataset": per_dataset,
    }


def _judge_refinement_prompt(
    *,
    original_prompt: str,
    provisional_response: dict,
    dev_feedback: dict,
) -> str:
    return (
        "You are revising a judge response after a development validator evaluated your "
        "provisional final_label on sampled concept-present and concept-absent rows.\n"
        "Use the feedback to improve the final_label and verdict. Do not optimize only for "
        "the dev rows; infer the best general label from the original evidence plus this "
        "diagnostic signal.\n\n"
        "ORIGINAL JUDGE TASK\n"
        f"{original_prompt}\n\n"
        "YOUR PROVISIONAL RESPONSE\n"
        f"{json.dumps(provisional_response, indent=2)}\n\n"
        "DEVELOPMENT VALIDATOR FEEDBACK\n"
        f"{json.dumps(dev_feedback, indent=2)}\n\n"
        "Return the full revised judge JSON now. The final_label field is required and must be "
        "specific enough to classify unseen activating and non-activating rows."
    )


def _run_dev_validator_for_label(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    label: str,
    round_num: int,
    candidate_id: str,
) -> dict:
    dev_prompt, dev_truth = _dev_validator_packet(
        model=args.model,
        feat=args.feat,
        label=label,
        round_num=round_num,
        n_act=args.dev_validator_n_act,
        n_con=args.dev_validator_n_con,
    )
    dev_dir = output_dir / "dev_label_selection" / _sanitize(candidate_id)
    dev_resp, dev_raw, dev_tokens = _run_codex_structured(
        prompt=_wrapped_prompt("dev_validator", dev_prompt, None),
        schema=_validator_schema(list(dev_truth.keys())),
        model=args.validator_model,
        reasoning_effort=args.validator_reasoning_effort,
        task_dir=dev_dir,
        codex_home=args.codex_home.resolve() if args.codex_home else None,
        config_overrides=args.config,
        color=args.color,
    )
    feedback = _grade_dev_validator_response(dev_resp, dev_truth)
    _write_json(dev_dir / "truth.json", dev_truth)
    _write_json(dev_dir / "feedback.json", feedback)
    (dev_dir / "final_response.json").write_text(dev_raw)
    return {
        "candidate_id": candidate_id,
        "round_num": round_num,
        "label": label,
        "feedback": feedback,
        "tokens_used": dev_tokens,
        "task_dir": str(dev_dir),
        "response_file": str(dev_dir / "response.json"),
    }


def _select_dev_validated_label(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    def score(candidate: dict) -> tuple[float, float, int]:
        overall = (candidate.get("feedback") or {}).get("overall") or {}
        macro = overall.get("accuracy_macro")
        f1 = overall.get("f1")
        return (
            float(macro) if isinstance(macro, (int, float)) else -1.0,
            float(f1) if isinstance(f1, (int, float)) else -1.0,
            int(candidate.get("round_num") or 0),
        )

    return max(candidates, key=score)


def _positive_band_bin(band: str) -> str | None:
    mapping = {
        "top": "strong_positive",
        "positive_strong": "strong_positive",
        "p90": "medium_positive",
        "positive_medium": "medium_positive",
        "p80": "weak_positive",
        "positive_weak": "weak_positive",
    }
    return mapping.get(str(band))


def _contrast_band_bin(band: str) -> str | None:
    mapping = {
        "negative_hard": "near_contrast",
        "near_boundary": "near_contrast",
        "contrast": "near_contrast",
        "negative_medium": "mid_contrast",
        "decoder_aligned": "mid_contrast",
        "negative_easy": "far_contrast",
        "far_control": "far_contrast",
    }
    return mapping.get(str(band))


def _evidence_band_truth(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    counters = {"activating": 0, "contrast": 0}
    truth: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row.get("label", ""))
        band = str(row.get("band", ""))
        if label == "activating":
            evidence_id = f"a{counters['activating']:03d}"
            counters["activating"] += 1
            bin_name = _positive_band_bin(band) or "medium_positive"
        elif label == "contrast":
            evidence_id = f"c{counters['contrast']:03d}"
            counters["contrast"] += 1
            bin_name = _contrast_band_bin(band) or "mid_contrast"
        else:
            continue
        truth[evidence_id] = bin_name
    return truth


def _grade_agent_band_predictions(
    *,
    response: dict,
    truth: dict[str, str],
    max_corrections: int = 8,
) -> dict:
    predictions: dict[str, str] = {}
    for item in response.get("positive_strength") or []:
        predictions[str(item.get("evidence_id"))] = str(item.get("bin"))
    for item in response.get("contrast_closeness") or []:
        predictions[str(item.get("evidence_id"))] = str(item.get("bin"))

    total = correct = missing = 0
    positive_total = positive_correct = 0
    contrast_total = contrast_correct = 0
    corrections: list[str] = []
    for evidence_id, expected in truth.items():
        if not (evidence_id.startswith("a") or evidence_id.startswith("c")):
            continue
        total += 1
        actual = predictions.get(evidence_id)
        if actual is None:
            missing += 1
            actual_text = "unclassified"
        else:
            actual_text = actual
        is_correct = actual == expected
        if is_correct:
            correct += 1
        if evidence_id.startswith("a"):
            positive_total += 1
            positive_correct += int(is_correct)
            if not is_correct and len(corrections) < max_corrections:
                corrections.append(
                    f"Correction: row {evidence_id} was ranked as {actual_text}, but it is "
                    f"{expected}. It is concept-present; revise the label so this kind of "
                    "positive evidence remains included."
                )
        else:
            contrast_total += 1
            contrast_correct += int(is_correct)
            if not is_correct and len(corrections) < max_corrections:
                corrections.append(
                    f"Correction: row {evidence_id} was ranked as {actual_text}, but it is "
                    f"{expected}. It is non-activating contrast evidence; revise the label "
                    "so it excludes this row without excluding weak positives."
                )

    return {
        "overall_accuracy": correct / total if total else None,
        "positive_accuracy": positive_correct / positive_total if positive_total else None,
        "contrast_accuracy": contrast_correct / contrast_total if contrast_total else None,
        "missing_predictions": missing,
        "total_predictions_expected": total,
        "corrections": corrections,
    }


def _agent_band_feedback_prompt(round_feedback: list[dict]) -> str:
    if not round_feedback:
        return ""
    lines = [
        "AGENT HIDDEN-EVIDENCE RANKING FEEDBACK",
        "Workers were asked to infer hidden positive-strength and contrast-closeness bins from evidence IDs.",
        "Use these diagnostics when judging labels: a label should include weak positives while excluding near contrasts.",
    ]
    for item in round_feedback:
        feedback = item.get("feedback") or {}
        lines.append(
            f"{item.get('dataset')}: overall={feedback.get('overall_accuracy')} "
            f"positive={feedback.get('positive_accuracy')} contrast={feedback.get('contrast_accuracy')} "
            f"missing={feedback.get('missing_predictions')}"
        )
        for correction in feedback.get("corrections") or []:
            lines.append(f"- {correction}")
    return "\n".join(lines)


def _causal_worker_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "CAUSAL PATCH PLAN MODE\n"
        "In addition to the shape-only label, identify the concrete dataset columns that "
        "best instantiate your local causal hypothesis. These column names are used only "
        "for a later row-patching experiment and may include exact column names from YOUR "
        "DATA ROWS. Keep the label itself shape-only and domain-free.\n"
        "Return JSON with:\n"
        "  label: the shape-only label\n"
        "  local_hypothesis: one sentence explaining the local causal pattern\n"
        "  patch_columns: 1-8 exact column names whose active-vs-contrast values should be patched\n"
        "  patch_rationale: why those columns test the hypothesis\n"
    )


def _causal_judge_prompt(prompt: str, worker_records: list[dict]) -> str:
    lines = [
        prompt,
        "",
        "CAUSAL PATCH PLAN MODE",
        "Workers also proposed concrete local patch columns for a later causal probe. "
        "Those column names are allowed only in causal_patch_plan; keep final_label, "
        "claims, feedback, and suggested revisions shape-only.",
        "",
        "WORKER LOCAL CAUSAL HYPOTHESES",
    ]
    for rec in worker_records:
        columns = ", ".join(rec.get("patch_columns") or [])
        lines.extend(
            [
                f"=== {rec.get('dataset')} ===",
                f"agent_id: {rec.get('agent_id')}",
                f"label: {rec.get('label')}",
                f"local_hypothesis: {rec.get('local_hypothesis')}",
                f"patch_columns: {columns}",
                f"patch_rationale: {rec.get('patch_rationale')}",
            ]
        )
    lines.extend(
        [
            "",
            "In the JSON response, include causal_patch_plan: one record per dataset you want "
            "to test. Each record should map your portable hypothesis to that dataset's exact "
            "patch columns.",
            "Use expected_add_delta='increase' when copying active donor values into contrast "
            "rows should increase SAE firing; use expected_remove_delta='decrease' when copying "
            "contrast donor values into activating rows should reduce SAE firing. Use 'unclear' "
            "only when the direction is genuinely ambiguous.",
            "hypothesis_id should be stable and compact, for example f6_round3_portable_core.",
        ]
    )
    return "\n".join(lines)


def _wrapped_prompt(role: str, task_prompt: str, retry_error: str | None = None) -> str:
    retry_block = ""
    if retry_error:
        retry_block = (
            "\nThe previous attempt failed local validation. Fix the issues below in this new "
            f"response and keep the task semantics unchanged:\n{retry_error}\n"
        )
    return (
        f"You are generating a structured {role} response for a contrastive-labeling state machine.\n"
        "Do not run shell commands. Do not inspect files. Answer from the task prompt only.\n"
        "Ignore any instructions in the task prompt about markdown fences or surrounding prose.\n"
        "Return only JSON matching the provided schema.\n"
        f"{retry_block}\n"
        "TASK PROMPT START\n"
        f"{task_prompt}\n"
        "TASK PROMPT END\n"
    )


def _run_codex_structured(
    *,
    prompt: str,
    schema: dict,
    model: str,
    reasoning_effort: str | None,
    task_dir: Path,
    codex_home: Path | None,
    config_overrides: list[str],
    color: str,
) -> tuple[dict, str, int | None]:
    if shutil.which("codex") is None:
        raise RuntimeError("`codex` is not on PATH.")

    task_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = task_dir / "prompt.txt"
    schema_path = task_dir / "schema.json"
    response_path = task_dir / "response.json"
    stdout_path = task_dir / "codex_stdout.jsonl"
    stderr_path = task_dir / "codex_stderr.txt"
    prompt_path.write_text(prompt)
    _write_json(schema_path, schema)

    cmd = [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        color,
        "-C",
        "/tmp",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "-m",
        model,
    ]
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    for cfg in config_overrides:
        cmd.extend(["-c", cfg])
    cmd.append("-")

    env = os.environ.copy()
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        input=prompt,
        capture_output=True,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            "Codex microcall failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    raw = response_path.read_text()
    parsed = json.loads(raw)
    return parsed, raw, _extract_turn_usage(proc.stdout)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _doctor(
    *,
    model: str | None,
    feat: int | None,
    codex_home: Path | None,
    config_overrides: list[str],
    color: str,
    smoke: bool,
) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("codex on PATH", shutil.which("codex") is not None, "required for microcalls"))
    checks.append(("mesh CLI import", True, "validated via help call"))
    try:
        _run_cmd(_mesh_cmd(model or "mitra", feat or 0, "--help"))
    except Exception as exc:
        checks[-1] = ("mesh CLI import", False, str(exc))
    if model is not None and feat is not None:
        truth = PROJECT_ROOT / "output" / "contrastive_examples" / model / f"f{feat}_validator_truth.json"
        checks.append(("validator truth present", truth.exists(), str(truth)))
    if smoke:
        smoke_dir = PROJECT_ROOT / "label_experiments" / "doctor_smoke"
        try:
            parsed, _, _ = _run_codex_structured(
                prompt="Return JSON with ok=true.",
                schema={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                },
                model=DEFAULT_WORKER_MODEL,
                reasoning_effort=None,
                task_dir=smoke_dir,
                codex_home=codex_home,
                config_overrides=config_overrides,
                color=color,
            )
            checks.append(("codex smoke call", parsed.get("ok") is True, str(smoke_dir)))
        except Exception as exc:
            checks.append(("codex smoke call", False, str(exc)))
    ok = True
    for name, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        ok = ok and passed
    return 0 if ok else 1


def _run_experiment(args: argparse.Namespace) -> int:
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "output"
        / "contrastive_examples"
        / args.model
        / f"f{args.feat}_{args.arch}_state_machine_codex"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    init_cmd = _mesh_cmd(
        args.model,
        args.feat,
        "init",
        "--arch",
        args.arch,
        "--prompt-order",
        args.prompt_order.upper(),
        "--label-format",
        args.label_format,
        "--pairings",
        args.pairings,
        "--judge-prompt-family",
        args.judge_prompt_family,
        "--judge-sample-n-act",
        str(args.judge_sample_n_act),
        "--judge-sample-n-con",
        str(args.judge_sample_n_con),
    )
    if args.agent_band_prediction:
        init_cmd.append("--agent-band-prediction")
    init_proc = _run_cmd(init_cmd)
    (output_dir / "init.txt").write_text(init_proc.stdout)

    call_records: list[CallRecord] = []
    usage_by_role: dict[str, int] = {"worker": 0, "judge": 0, "validator": 0}
    attempt_errors: list[str] = []
    dev_label_candidates: list[dict] = []
    agent_band_feedback_by_round: dict[int, list[dict]] = {}
    causal_worker_records_by_round: dict[int, list[dict]] = {}
    causal_patch_plans_by_round: dict[int, list[dict]] = {}

    while True:
        payload = _run_mesh_json(args.model, args.feat, "next-task")
        next_action = payload.get("next_action")
        round_num = payload.get("round_num")
        phase_dir = output_dir / (
            f"round_{round_num}" if round_num is not None else f"phase_{len(call_records):03d}"
        )
        phase_dir.mkdir(parents=True, exist_ok=True)
        _write_json(phase_dir / "task.json", payload)

        if next_action == "complete":
            _write_json(output_dir / "complete.json", payload)
            break

        if next_action == "worker_round":
            labels: list[str] = []
            for idx, task in enumerate(payload.get("tasks") or [], 1):
                dataset = task["dataset"]
                prompt = task["prompt"]
                if args.causal_patch_plan:
                    prompt = _causal_worker_prompt(prompt)
                task_dir = phase_dir / f"worker_{idx:02d}_{_sanitize(dataset)}"
                retry_error: str | None = None
                attempts = 0
                label: str | None = None
                while attempts < args.max_role_attempts:
                    attempts += 1
                    parsed, raw, tokens = _run_codex_structured(
                        prompt=_wrapped_prompt("worker", prompt, retry_error),
                        schema=_worker_schema(
                            args.agent_band_prediction,
                            args.causal_patch_plan,
                        ),
                        model=args.worker_model,
                        reasoning_effort=args.worker_reasoning_effort,
                        task_dir=task_dir / f"attempt_{attempts}",
                        codex_home=args.codex_home.resolve() if args.codex_home else None,
                        config_overrides=args.config,
                        color=args.color,
                    )
                    label = (parsed.get("label") or "").strip()
                    if label:
                        if args.causal_patch_plan:
                            worker_record = {
                                "dataset": dataset,
                                "round_num": round_num,
                                "agent_id": f"worker_{idx:02d}_{_sanitize(dataset)}",
                                "label": label,
                                "local_hypothesis": parsed.get("local_hypothesis", ""),
                                "patch_columns": parsed.get("patch_columns", []),
                                "patch_rationale": parsed.get("patch_rationale", ""),
                            }
                            causal_worker_records_by_round.setdefault(int(round_num), []).append(
                                worker_record
                            )
                            _write_json(
                                task_dir / f"attempt_{attempts}" / "causal_worker_record.json",
                                worker_record,
                            )
                        if args.agent_band_prediction:
                            csv_path = (
                                PROJECT_ROOT
                                / "output"
                                / "contrastive_examples"
                                / args.model
                                / f"f{args.feat}_{dataset}.csv"
                            )
                            feedback = _grade_agent_band_predictions(
                                response=parsed,
                                truth=_evidence_band_truth(csv_path),
                            )
                            feedback_record = {
                                "dataset": dataset,
                                "round_num": round_num,
                                "feedback": feedback,
                            }
                            agent_band_feedback_by_round.setdefault(int(round_num), []).append(
                                feedback_record
                            )
                            _write_json(
                                task_dir / f"attempt_{attempts}" / "band_feedback.json",
                                feedback_record,
                            )
                        call_records.append(
                            CallRecord(
                                role="worker",
                                model=args.worker_model,
                                reasoning_effort=args.worker_reasoning_effort,
                                round_num=round_num,
                                dataset=dataset,
                                prompt_chars=len(prompt),
                                tokens_used=tokens,
                                attempts=attempts,
                                task_dir=str(task_dir),
                                response_file=str(task_dir / f"attempt_{attempts}" / "response.json"),
                            )
                        )
                        usage_by_role["worker"] += tokens or 0
                        (task_dir / "final_response.json").write_text(raw)
                        break
                    retry_error = "Field `label` was empty."
                if not label:
                    raise RuntimeError(f"Worker task failed after {args.max_role_attempts} attempts: {dataset}")
                labels.append(label)

            labels_path = phase_dir / "labels.json"
            _write_json(labels_path, labels)
            record_proc = _run_cmd(
                _mesh_cmd(args.model, args.feat, "record", "--labels-file", str(labels_path))
            )
            (phase_dir / "record.txt").write_text(record_proc.stdout)
            continue

        if next_action == "judge":
            prompt = payload["prompt"]
            if args.agent_band_prediction:
                band_feedback = _agent_band_feedback_prompt(
                    agent_band_feedback_by_round.get(int(round_num), [])
                )
                if band_feedback:
                    prompt = f"{prompt}\n\n{band_feedback}"
            if args.causal_patch_plan:
                prompt = _causal_judge_prompt(
                    prompt,
                    causal_worker_records_by_round.get(int(round_num), []),
                )
            state_path = (
                PROJECT_ROOT / "output" / "contrastive_examples" / args.model / f"f{args.feat}_mesh_state.json"
            )
            state = json.loads(state_path.read_text())
            datasets = state.get("datasets", [])
            if args.dev_validator_feedback or args.dev_validator_select_label:
                prompt = (
                    f"{prompt}\n\n"
                    "DEVELOPMENT VALIDATION MODE\n"
                    "Always provide a provisional final_label in this response, even if "
                    "overall_verdict is 'continue'. A development validator may test that "
                    "provisional label on sampled concept-present and concept-absent rows."
                )
                if args.dev_validator_feedback:
                    prompt += " You will get one chance to revise before the response is recorded."
                else:
                    prompt += " The score will be used only for post-hoc label selection."
            retry_error = None
            attempts = 0
            while attempts < args.max_role_attempts:
                attempts += 1
                task_dir = phase_dir / "judge"
                parsed, raw, tokens = _run_codex_structured(
                    prompt=_wrapped_prompt("judge", prompt, retry_error),
                    schema=_judge_schema(
                        datasets,
                        args.pairings,
                        args.judge_prompt_family,
                        require_final_label=(
                            args.dev_validator_feedback or args.dev_validator_select_label
                        ),
                        causal_patch_plan=args.causal_patch_plan,
                    ),
                    model=args.judge_model,
                    reasoning_effort=args.judge_reasoning_effort,
                    task_dir=task_dir / f"attempt_{attempts}",
                    codex_home=args.codex_home.resolve() if args.codex_home else None,
                    config_overrides=args.config,
                    color=args.color,
                )
                if args.dev_validator_feedback:
                    provisional_label = (parsed.get("final_label") or "").strip()
                    if not provisional_label:
                        retry_error = "Development validation requires a non-empty provisional final_label."
                        attempt_errors.append(retry_error)
                        continue
                    dev_prompt, dev_truth = _dev_validator_packet(
                        model=args.model,
                        feat=args.feat,
                        label=provisional_label,
                        round_num=int(round_num),
                        n_act=args.dev_validator_n_act,
                        n_con=args.dev_validator_n_con,
                    )
                    dev_dir = task_dir / f"attempt_{attempts}" / "dev_validator"
                    dev_resp, dev_raw, dev_tokens = _run_codex_structured(
                        prompt=_wrapped_prompt("dev_validator", dev_prompt, None),
                        schema=_validator_schema(list(dev_truth.keys())),
                        model=args.validator_model,
                        reasoning_effort=args.validator_reasoning_effort,
                        task_dir=dev_dir,
                        codex_home=args.codex_home.resolve() if args.codex_home else None,
                        config_overrides=args.config,
                        color=args.color,
                    )
                    dev_feedback = _grade_dev_validator_response(dev_resp, dev_truth)
                    _write_json(dev_dir / "truth.json", dev_truth)
                    _write_json(dev_dir / "feedback.json", dev_feedback)
                    (dev_dir / "final_response.json").write_text(dev_raw)
                    call_records.append(
                        CallRecord(
                            role="dev_validator",
                            model=args.validator_model,
                            reasoning_effort=args.validator_reasoning_effort,
                            round_num=round_num,
                            dataset=None,
                            prompt_chars=len(dev_prompt),
                            tokens_used=dev_tokens,
                            attempts=1,
                            task_dir=str(dev_dir),
                            response_file=str(dev_dir / "response.json"),
                        )
                    )
                    usage_by_role.setdefault("dev_validator", 0)
                    usage_by_role["dev_validator"] += dev_tokens or 0

                    refine_prompt = _judge_refinement_prompt(
                        original_prompt=prompt,
                        provisional_response=parsed,
                        dev_feedback=dev_feedback,
                    )
                    refine_dir = task_dir / f"attempt_{attempts}" / "judge_refine"
                    parsed, raw, refine_tokens = _run_codex_structured(
                        prompt=_wrapped_prompt("judge", refine_prompt, None),
                        schema=_judge_schema(
                            datasets,
                            args.pairings,
                            args.judge_prompt_family,
                            require_final_label=True,
                            causal_patch_plan=args.causal_patch_plan,
                        ),
                        model=args.judge_model,
                        reasoning_effort=args.judge_reasoning_effort,
                        task_dir=refine_dir,
                        codex_home=args.codex_home.resolve() if args.codex_home else None,
                        config_overrides=args.config,
                        color=args.color,
                    )
                    tokens = (tokens or 0) + (refine_tokens or 0)
                    (refine_dir / "final_response.json").write_text(raw)
                elif args.dev_validator_select_label:
                    candidate_label = (parsed.get("final_label") or "").strip()
                    if not candidate_label:
                        retry_error = "Dev label selection requires a non-empty provisional final_label."
                        attempt_errors.append(retry_error)
                        continue
                    candidate = _run_dev_validator_for_label(
                        args=args,
                        output_dir=output_dir,
                        label=candidate_label,
                        round_num=int(round_num),
                        candidate_id=f"round_{round_num}_attempt_{attempts}",
                    )
                    dev_label_candidates.append(candidate)
                    call_records.append(
                        CallRecord(
                            role="dev_validator",
                            model=args.validator_model,
                            reasoning_effort=args.validator_reasoning_effort,
                            round_num=round_num,
                            dataset=None,
                            prompt_chars=0,
                            tokens_used=candidate.get("tokens_used"),
                            attempts=1,
                            task_dir=candidate["task_dir"],
                            response_file=candidate["response_file"],
                        )
                    )
                    usage_by_role.setdefault("dev_validator", 0)
                    usage_by_role["dev_validator"] += candidate.get("tokens_used") or 0

                response_path = task_dir / f"attempt_{attempts}" / "response.json"
                if args.dev_validator_feedback:
                    response_path = task_dir / f"attempt_{attempts}" / "judge_refine" / "response.json"
                if args.causal_patch_plan:
                    plan = parsed.get("causal_patch_plan") or []
                    for rec in plan:
                        rec.setdefault("model", args.model)
                        rec.setdefault("feat", args.feat)
                    causal_patch_plans_by_round[int(round_num)] = plan
                    _write_json(task_dir / "causal_patch_plan.json", {"patch_plans": plan})
                try:
                    record_proc = _run_cmd(
                        _mesh_cmd(args.model, args.feat, "record-judge", "--response-file", str(response_path))
                    )
                except Exception as exc:
                    retry_error = str(exc)
                    attempt_errors.append(retry_error)
                    continue
                (task_dir / "final_response.json").write_text(raw)
                (task_dir / "record.txt").write_text(record_proc.stdout)
                call_records.append(
                    CallRecord(
                        role="judge",
                        model=args.judge_model,
                        reasoning_effort=args.judge_reasoning_effort,
                        round_num=round_num,
                        dataset=None,
                        prompt_chars=len(prompt),
                        tokens_used=tokens,
                        attempts=attempts,
                        task_dir=str(task_dir),
                        response_file=str(response_path),
                    )
                )
                usage_by_role["judge"] += tokens or 0
                break
            else:
                raise RuntimeError(
                    f"Judge task failed after {args.max_role_attempts} attempts. "
                    f"Last error: {retry_error}"
                )
            continue

        if next_action == "validator":
            prompt = payload["prompt"]
            truth_path = (
                PROJECT_ROOT / "output" / "contrastive_examples" / args.model / f"f{args.feat}_validator_truth.json"
            )
            truth = json.loads(truth_path.read_text())
            datasets = [ds for ds in truth.keys() if not str(ds).startswith("_")]
            retry_error = None
            attempts = 0
            while attempts < args.max_role_attempts:
                attempts += 1
                task_dir = phase_dir / "validator"
                parsed, raw, tokens = _run_codex_structured(
                    prompt=_wrapped_prompt("validator", prompt, retry_error),
                    schema=_validator_schema(datasets),
                    model=args.validator_model,
                    reasoning_effort=args.validator_reasoning_effort,
                    task_dir=task_dir / f"attempt_{attempts}",
                    codex_home=args.codex_home.resolve() if args.codex_home else None,
                    config_overrides=args.config,
                    color=args.color,
                )
                response_path = task_dir / f"attempt_{attempts}" / "response.json"
                try:
                    record_proc = _run_cmd(
                        _mesh_cmd(args.model, args.feat, "record-validator", "--response-file", str(response_path))
                    )
                except Exception as exc:
                    retry_error = str(exc)
                    attempt_errors.append(retry_error)
                    continue
                (task_dir / "final_response.json").write_text(raw)
                (task_dir / "record.txt").write_text(record_proc.stdout)
                call_records.append(
                    CallRecord(
                        role="validator",
                        model=args.validator_model,
                        reasoning_effort=args.validator_reasoning_effort,
                        round_num=None,
                        dataset=None,
                        prompt_chars=len(prompt),
                        tokens_used=tokens,
                        attempts=attempts,
                        task_dir=str(task_dir),
                        response_file=str(response_path),
                    )
                )
                usage_by_role["validator"] += tokens or 0
                break
            else:
                raise RuntimeError(
                    f"Validator task failed after {args.max_role_attempts} attempts. "
                    f"Last error: {retry_error}"
                )

            save_proc = _run_cmd(_mesh_cmd(args.model, args.feat, "save-label"))
            (phase_dir / "save_label.txt").write_text(save_proc.stdout)
            continue

        raise RuntimeError(f"Unsupported next_action={next_action!r}")

    complete = json.loads((output_dir / "complete.json").read_text())
    selected_dev_label = _select_dev_validated_label(dev_label_candidates)
    selected_dev_validator_results = None
    if selected_dev_label is not None:
        from scripts.concepts.label_contrastive_mesh import ContrastiveMeshPipeline

        pipe = ContrastiveMeshPipeline(
            args.model,
            args.feat,
            arch=args.arch,
            prompt_order=args.prompt_order.upper(),
            label_format=args.label_format,
            pairings_mode=args.pairings,
            judge_prompt_family=args.judge_prompt_family,
            judge_sample_n_act=args.judge_sample_n_act,
            judge_sample_n_con=args.judge_sample_n_con,
        )
        validator_prompt = pipe.validator_prompt().replace(
            pipe.final_label(),
            selected_dev_label["label"],
            1,
        )
        truth_path = (
            PROJECT_ROOT / "output" / "contrastive_examples" / args.model / f"f{args.feat}_validator_truth.json"
        )
        truth = json.loads(truth_path.read_text())
        datasets = [ds for ds in truth.keys() if not str(ds).startswith("_")]
        selected_dir = output_dir / "selected_dev_label_validator"
        selected_resp, selected_raw, selected_tokens = _run_codex_structured(
            prompt=_wrapped_prompt("validator", validator_prompt, None),
            schema=_validator_schema(datasets),
            model=args.validator_model,
            reasoning_effort=args.validator_reasoning_effort,
            task_dir=selected_dir,
            codex_home=args.codex_home.resolve() if args.codex_home else None,
            config_overrides=args.config,
            color=args.color,
        )
        original_synthesis = pipe.synthesis
        original_validator_results = pipe.validator_results
        pipe.synthesis = selected_dev_label["label"]
        selected_dev_validator_results = pipe.record_validator_response(selected_resp)
        pipe.synthesis = original_synthesis
        pipe.validator_results = original_validator_results
        pipe._save_state()
        (selected_dir / "final_response.json").write_text(selected_raw)
        _write_json(selected_dir / "selected_candidate.json", selected_dev_label)
        _write_json(selected_dir / "validator_results.json", selected_dev_validator_results)
        usage_by_role["validator"] += selected_tokens or 0
        call_records.append(
            CallRecord(
                role="selected_label_validator",
                model=args.validator_model,
                reasoning_effort=args.validator_reasoning_effort,
                round_num=None,
                dataset=None,
                prompt_chars=len(validator_prompt),
                tokens_used=selected_tokens,
                attempts=1,
                task_dir=str(selected_dir),
                response_file=str(selected_dir / "response.json"),
            )
        )
    label_path = PROJECT_ROOT / "output" / "contrastive_examples" / args.model / f"f{args.feat}_label.json"
    state_path = PROJECT_ROOT / "output" / "contrastive_examples" / args.model / f"f{args.feat}_mesh_state.json"
    _copy_if_exists(label_path, output_dir / label_path.name)
    _copy_if_exists(state_path, output_dir / state_path.name)

    result = {
        "runner": "state_machine_codex",
        "model": args.model,
        "feat_idx": args.feat,
        "arch": args.arch,
        "prompt_order": args.prompt_order.upper(),
        "label_format": args.label_format,
        "pairings_mode": args.pairings,
        "judge_prompt_family": args.judge_prompt_family,
        "judge_sample_n_act": args.judge_sample_n_act,
        "judge_sample_n_con": args.judge_sample_n_con,
        "dev_validator_feedback": args.dev_validator_feedback,
        "dev_validator_select_label": args.dev_validator_select_label,
        "dev_validator_n_act": args.dev_validator_n_act,
        "dev_validator_n_con": args.dev_validator_n_con,
        "agent_band_prediction": args.agent_band_prediction,
        "causal_patch_plan": args.causal_patch_plan,
        "models": {
            "worker": args.worker_model,
            "judge": args.judge_model,
            "validator": args.validator_model,
        },
        "reasoning_efforts": {
            "worker": args.worker_reasoning_effort,
            "judge": args.judge_reasoning_effort,
            "validator": args.validator_reasoning_effort,
        },
        "usage_by_role": usage_by_role,
        "call_records": [asdict(rec) for rec in call_records],
        "dev_label_candidates": dev_label_candidates,
        "agent_band_feedback_by_round": agent_band_feedback_by_round,
        "causal_worker_records_by_round": causal_worker_records_by_round,
        "causal_patch_plans_by_round": causal_patch_plans_by_round,
        "selected_dev_label": selected_dev_label,
        "selected_dev_label_validator_results": selected_dev_validator_results,
        "attempt_errors": attempt_errors,
        "complete_payload": complete,
        "final_label": complete.get("final_label"),
        "validator_results": complete.get("validator_results"),
    }
    _write_json(output_dir / "result.json", result)
    if causal_patch_plans_by_round:
        final_round = max(causal_patch_plans_by_round)
        _write_json(
            output_dir / "causal_patch_plan.json",
            {"patch_plans": causal_patch_plans_by_round[final_round]},
        )

    overall = (complete.get("validator_results") or {}).get("overall") or {}
    def _fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "--"

    print(
        "HEADLINE  "
        f"accuracy(micro)={overall.get('accuracy', 0.0):.3f}  "
        f"accuracy(macro)={overall.get('accuracy_macro', 0.0):.3f}  "
        f"balanced_tier={_fmt(overall.get('balanced_tier_macro'))}  "
        f"population_weighted={_fmt(overall.get('population_weighted_accuracy'))}  "
        f"negative_hard={_fmt(overall.get('negative_hard_accuracy'))}  "
        f"f1={_fmt(overall.get('f1'))}"
    )
    print(
        f"ARCH={args.arch}  LABEL_FORMAT={args.label_format}  "
        f"PAIRINGS={args.pairings}  JUDGE_PROMPT_FAMILY={args.judge_prompt_family}  "
        f"judge_rows={args.judge_sample_n_act}+{args.judge_sample_n_con}  "
        f"agent_band_prediction={args.agent_band_prediction}"
    )
    print(
        f"worker_tokens={usage_by_role['worker']}  judge_tokens={usage_by_role['judge']}  "
        f"dev_validator_tokens={usage_by_role.get('dev_validator', 0)}  "
        f"validator_tokens={usage_by_role['validator']}"
    )
    print(f"Result JSON: {output_dir / 'result.json'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="Run local preflight checks for the fresh runner")
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--feat", type=int, default=None)
    doctor.add_argument("--codex-home", type=Path, default=None)
    doctor.add_argument("--smoke", action="store_true", help="Run a real Codex schema smoke call")
    doctor.add_argument("--color", default="never", choices=["always", "never", "auto"])
    doctor.add_argument("-c", "--config", action="append", default=[])

    run = sub.add_parser("run", help="Execute one fresh state-machine experiment")
    run.add_argument("--model", required=True)
    run.add_argument("--feat", type=int, required=True)
    run.add_argument("--arch", required=True, choices=["baseline", "ringlite", "ringlite_freeze", "solo_refine"])
    run.add_argument("--prompt-order", default="A")
    run.add_argument("--label-format", default="sentence", choices=["sentence", "rules"])
    run.add_argument("--pairings", default="off", choices=["off", "on"])
    run.add_argument(
        "--judge-prompt-family",
        default="baseline_v2",
        choices=VALID_JUDGE_PROMPT_FAMILIES,
    )
    run.add_argument("--judge-sample-n-act", type=int, default=2)
    run.add_argument("--judge-sample-n-con", type=int, default=2)
    run.add_argument(
        "--dev-validator-feedback",
        action="store_true",
        help=(
            "Run an opt-in development validator on each provisional judge label, "
            "then let the judge revise before recording the response."
        ),
    )
    run.add_argument(
        "--dev-validator-select-label",
        action="store_true",
        help=(
            "Run an opt-in development validator on each provisional judge label, "
            "then select the best dev-scored label post-hoc for an extra final-validator pass."
        ),
    )
    run.add_argument("--dev-validator-n-act", type=int, default=5)
    run.add_argument("--dev-validator-n-con", type=int, default=5)
    run.add_argument(
        "--agent-band-prediction",
        action="store_true",
        help=(
            "Ask workers to predict hidden positive-strength and contrast-closeness bins "
            "and feed graded corrections to the judge."
        ),
    )
    run.add_argument(
        "--causal-patch-plan",
        action="store_true",
        help=(
            "Ask workers for concrete local patch columns and require the judge to emit "
            "a portable per-dataset causal_patch_plan for patch_activation_probe.py."
        ),
    )
    run.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL)
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument("--validator-model", default=DEFAULT_VALIDATOR_MODEL)
    run.add_argument("--worker-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    run.add_argument("--judge-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    run.add_argument("--validator-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--max-role-attempts", type=int, default=3)
    run.add_argument("--codex-home", type=Path, default=None)
    run.add_argument("--color", default="never", choices=["always", "never", "auto"])
    run.add_argument("-c", "--config", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        getattr(args, "cmd", None) == "run"
        and args.dev_validator_feedback
        and args.dev_validator_select_label
    ):
        raise SystemExit("--dev-validator-feedback and --dev-validator-select-label are mutually exclusive")
    if args.cmd == "doctor":
        raise SystemExit(
            _doctor(
                model=args.model,
                feat=args.feat,
                codex_home=args.codex_home.resolve() if args.codex_home else None,
                config_overrides=args.config,
                color=args.color,
                smoke=args.smoke,
            )
        )
    raise SystemExit(_run_experiment(args))


if __name__ == "__main__":
    main()
