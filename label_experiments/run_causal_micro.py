#!/usr/bin/env python3
"""Run a causal-micro labeling experiment from bidirectional LOO packets.

This experiment intentionally avoids the full contrastive mesh. Each dataset
agent sees only compact causal patch evidence for its dataset, then a synthesizer
blends the local labels into one portable feature label. The existing held-out
validator grades that final label.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from label_experiments.run_fresh_state_machine import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_VALIDATOR_MODEL,
    DEFAULT_WORKER_MODEL,
    VALID_REASONING_EFFORTS,
    _run_codex_structured,
    _validator_schema,
    _wrapped_prompt,
)
from scripts._project_root import PROJECT_ROOT
from scripts.concepts.label_contrastive_mesh import ContrastiveMeshPipeline
from scripts.concepts.run_label_contrastive_mesh_codex import _grade_validator_response


@dataclass
class CallRecord:
    role: str
    model: str
    reasoning_effort: str | None
    dataset: str | None
    prompt_chars: int
    tokens_used: int | None
    task_dir: str
    response_file: str


def _format_num(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def _worker_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": ["local_label", "causal_pattern", "boundary"],
        "properties": {
            "local_label": {"type": "string", "minLength": 1},
            "causal_pattern": {"type": "string", "minLength": 1},
            "boundary": {"type": "string", "minLength": 1},
        },
    }


def _synth_schema(datasets: list[str]) -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": ["final_label", "synthesis_rationale", "per_dataset_notes"],
        "properties": {
            "final_label": {"type": "string", "minLength": 1},
            "synthesis_rationale": {"type": "string"},
            "per_dataset_notes": {
                "type": "object",
                "additionalProperties": False,
                "required": datasets,
                "properties": {ds: {"type": "string"} for ds in datasets},
            },
        },
    }


def _top_items(items: list[dict], n: int, positive_only: bool = True) -> list[dict]:
    out = sorted(items or [], key=lambda item: float(item.get("delta") or 0.0), reverse=True)
    if positive_only:
        out = [item for item in out if float(item.get("delta") or 0.0) > 0]
    return out[:n]


def _example_block(example: dict, top_loo: int) -> str:
    lines = [
        f"active_row={example['active_row_idx']} donor_contrast_row={example['donor_row_idx']}",
        (
            "activation "
            f"{_format_num(example.get('original_activation'))} -> "
            f"{_format_num(example.get('final_activation'))}; "
            f"drop_frac={_format_num(example.get('total_drop_frac'))}; "
            f"tier={example.get('suppression_tier')}; stop={example.get('stop_reason')}"
        ),
    ]
    changed = example.get("changed_fields") or []
    if changed:
        lines.append("optimized suppressive fields:")
        for field in changed:
            lines.append(
                f"- {field['column']}: active={field.get('active_value')} "
                f"contrast={field.get('donor_value')}"
            )
    remove = _top_items(example.get("top_loo_remove") or [], top_loo)
    add = _top_items(example.get("top_loo_add") or [], top_loo)
    lines.append("top remove-from-active drops:")
    if remove:
        for item in remove:
            lines.append(
                f"- {item['column_name']}: {_format_num(item.get('original_activation'))} -> "
                f"{_format_num(item.get('patched_activation'))}, drop={_format_num(item.get('delta'))}, "
                f"active={item.get('active_value')} contrast={item.get('donor_value')}"
            )
    else:
        lines.append("- none positive")
    lines.append("top add-to-contrast increases:")
    if add:
        for item in add:
            lines.append(
                f"- {item['column_name']}: {_format_num(item.get('original_activation'))} -> "
                f"{_format_num(item.get('patched_activation'))}, increase={_format_num(item.get('delta'))}, "
                f"active={item.get('active_value')} contrast={item.get('donor_value')}"
            )
    else:
        lines.append("- none positive")
    return "\n".join(lines)


def _worker_prompt(feat: int, dataset_packet: dict, top_loo: int) -> str:
    dataset = dataset_packet["dataset"]
    summary = dataset_packet["summary"]
    examples = dataset_packet["examples"]
    blocks = "\n\n".join(_example_block(ex, top_loo) for ex in examples)
    return (
        f"You are labeling SAE feature f_{feat} for dataset '{dataset}' using only causal perturbation evidence.\n\n"
        "Each example is a matched activating row and non-activating donor row. "
        "remove-from-active copies one donor value into the activating row and measures activation drop. "
        "add-to-contrast copies one activating value into the non-activating donor row and measures activation increase.\n\n"
        "Your task: describe the local row-level firing pattern implied by these causal swaps. "
        "Use field names only as evidence in your reasoning; the local_label should be shape-level and portable within this dataset.\n\n"
        f"Dataset summary: rows={summary['rows']} mean_drop={_format_num(summary['mean_drop_frac'])} "
        f"strong_rate={_format_num(summary['strong_rate'])} common_fields={', '.join(summary['common_selected_columns']) or '(none)'}\n\n"
        f"{blocks}\n\n"
        "Return JSON only. local_label should be concise and describe what makes rows fire; "
        "causal_pattern should mention the field roles in natural language; boundary should state what non-firing donors lack."
    )


def _synth_prompt(feat: int, packet: dict, worker_outputs: dict[str, dict]) -> str:
    sections = []
    for ds in packet["datasets"]:
        name = ds["dataset"]
        out = worker_outputs[name]
        summary = ds["summary"]
        sections.append(
            f"=== {name} ===\n"
            f"summary: rows={summary['rows']} mean_drop={_format_num(summary['mean_drop_frac'])} "
            f"strong_rate={_format_num(summary['strong_rate'])} common_fields={', '.join(summary['common_selected_columns']) or '(none)'}\n"
            f"local_label: {out['local_label']}\n"
            f"causal_pattern: {out['causal_pattern']}\n"
            f"boundary: {out['boundary']}"
        )
    return (
        f"You are synthesizing a portable label for SAE feature f_{feat} from five dataset-local causal labels.\n\n"
        "The local agents saw only bidirectional perturbation evidence. Your job is to abstract across datasets. "
        "Do not include column names or domain terms in final_label. Prefer the shortest positive label that is specific enough "
        "to classify unseen activating vs non-activating rows.\n\n"
        f"{chr(10).join(sections)}\n\n"
        "Return JSON only. final_label must be one sentence and shape-level. "
        "If the evidence is heterogeneous, choose the tightest shared structural role rather than saying it is heterogeneous."
    )


def run(args: argparse.Namespace) -> int:
    packet_path = args.packet_dir / f"{args.model}_f{args.feat}_suppression_label_packet.json"
    packet = json.loads(packet_path.read_text())
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "output"
        / "contrastive_examples"
        / args.model
        / f"f{args.feat}_causal_micro_v1_codex"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    call_records: list[CallRecord] = []
    usage_by_role = {"worker": 0, "judge": 0, "validator": 0}
    worker_outputs: dict[str, dict] = {}
    for ds in packet["datasets"]:
        dataset = ds["dataset"]
        prompt = _worker_prompt(args.feat, ds, args.top_loo)
        task_dir = output_dir / "workers" / dataset
        parsed, raw, tokens = _run_codex_structured(
            prompt=_wrapped_prompt("causal_micro_worker", prompt, None),
            schema=_worker_schema(),
            model=args.worker_model,
            reasoning_effort=args.worker_reasoning_effort,
            task_dir=task_dir,
            codex_home=args.codex_home.resolve() if args.codex_home else None,
            config_overrides=args.config,
            color=args.color,
        )
        worker_outputs[dataset] = parsed
        (task_dir / "final_response.json").write_text(raw)
        usage_by_role["worker"] += tokens or 0
        call_records.append(
            CallRecord(
                role="worker",
                model=args.worker_model,
                reasoning_effort=args.worker_reasoning_effort,
                dataset=dataset,
                prompt_chars=len(prompt),
                tokens_used=tokens,
                task_dir=str(task_dir),
                response_file=str(task_dir / "response.json"),
            )
        )

    datasets = [ds["dataset"] for ds in packet["datasets"]]
    synth_prompt = _synth_prompt(args.feat, packet, worker_outputs)
    synth_dir = output_dir / "synthesizer"
    synth, synth_raw, synth_tokens = _run_codex_structured(
        prompt=_wrapped_prompt("causal_micro_synthesizer", synth_prompt, None),
        schema=_synth_schema(datasets),
        model=args.judge_model,
        reasoning_effort=args.judge_reasoning_effort,
        task_dir=synth_dir,
        codex_home=args.codex_home.resolve() if args.codex_home else None,
        config_overrides=args.config,
        color=args.color,
    )
    (synth_dir / "final_response.json").write_text(synth_raw)
    usage_by_role["judge"] += synth_tokens or 0
    call_records.append(
        CallRecord(
            role="judge",
            model=args.judge_model,
            reasoning_effort=args.judge_reasoning_effort,
            dataset=None,
            prompt_chars=len(synth_prompt),
            tokens_used=synth_tokens,
            task_dir=str(synth_dir),
            response_file=str(synth_dir / "response.json"),
        )
    )

    pipe = ContrastiveMeshPipeline(
        args.model,
        args.feat,
        arch="ringlite_freeze",
        label_format="sentence",
        judge_prompt_family="baseline_v2",
        judge_sample_n_act=5,
        judge_sample_n_con=5,
    )
    pipe.rounds = []
    pipe.judge_verdicts = []
    pipe.synthesis = synth["final_label"]
    validator_prompt = pipe.validator_prompt()
    truth = pipe._validator_truth()
    validator_dir = output_dir / "validator"
    validator, validator_raw, validator_tokens = _run_codex_structured(
        prompt=_wrapped_prompt("validator", validator_prompt, None),
        schema=_validator_schema(list(truth.keys())),
        model=args.validator_model,
        reasoning_effort=args.validator_reasoning_effort,
        task_dir=validator_dir,
        codex_home=args.codex_home.resolve() if args.codex_home else None,
        config_overrides=args.config,
        color=args.color,
    )
    (validator_dir / "final_response.json").write_text(validator_raw)
    usage_by_role["validator"] += validator_tokens or 0
    call_records.append(
        CallRecord(
            role="validator",
            model=args.validator_model,
            reasoning_effort=args.validator_reasoning_effort,
            dataset=None,
            prompt_chars=len(validator_prompt),
            tokens_used=validator_tokens,
            task_dir=str(validator_dir),
            response_file=str(validator_dir / "response.json"),
        )
    )
    validator_results = _grade_validator_response(pipe, validator)

    result = {
        "runner": "causal_micro_v1",
        "model": args.model,
        "feat_idx": args.feat,
        "packet_path": str(packet_path),
        "top_loo": args.top_loo,
        "worker_outputs": worker_outputs,
        "synthesizer": synth,
        "final_label": synth["final_label"],
        "validator_results": validator_results,
        "usage_by_role": usage_by_role,
        "call_records": [asdict(rec) for rec in call_records],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    overall = validator_results["overall"]
    print(
        "HEADLINE  "
        f"accuracy(micro)={overall.get('accuracy', 0.0):.3f}  "
        f"accuracy(macro)={overall.get('accuracy_macro', 0.0):.3f}  "
        f"balanced_tier={overall.get('balanced_tier_macro', 0.0):.3f}  "
        f"population_weighted={overall.get('population_weighted_accuracy', 0.0):.3f}  "
        f"f1={overall.get('f1') if overall.get('f1') is not None else '--'}"
    )
    print(f"Label: {synth['final_label']}")
    print(f"Result JSON: {output_dir / 'result.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--feat", type=int, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "suppression_label_packets_v2",
    )
    parser.add_argument("--top-loo", type=int, default=3)
    parser.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--validator-model", default=DEFAULT_VALIDATOR_MODEL)
    parser.add_argument("--worker-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    parser.add_argument("--judge-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    parser.add_argument("--validator-reasoning-effort", choices=VALID_REASONING_EFFORTS, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--color", default="never", choices=["always", "never", "auto"])
    parser.add_argument("-c", "--config", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
