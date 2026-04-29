#!/usr/bin/env python3
"""Build compact one-patch causal exemplars from suppression label packets.

This is an evidence packaging step, not a labeling experiment. For each feature
and dataset, it selects one high-confidence activating/donor pair and writes the
actual row values plus the patch effect that changed SAE activation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT


def _format_num(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _positive_best(items: list[dict]) -> dict | None:
    positive = [item for item in items or [] if float(item.get("delta") or 0.0) > 0.0]
    if not positive:
        return None
    return max(positive, key=lambda item: float(item.get("delta") or 0.0))


def _delta_frac(delta: float, denom: float | None) -> float:
    if not denom or denom <= 0:
        return 0.0
    return max(0.0, min(delta / denom, 1.0))


def _score_example(example: dict) -> dict:
    remove = _positive_best(example.get("top_loo_remove") or [])
    add = _positive_best(example.get("top_loo_add") or [])
    original_activation = float(example.get("original_activation") or 0.0)
    drop_frac = max(0.0, float(example.get("total_drop_frac") or 0.0))
    remove_delta = float(remove.get("delta") or 0.0) if remove else 0.0
    add_delta = float(add.get("delta") or 0.0) if add else 0.0
    same_field = bool(
        remove
        and add
        and remove.get("column_name") == add.get("column_name")
    )
    n_selected = int(example.get("n_selected") or len(example.get("selected_columns") or []))

    # Prefer examples that are causal in both directions and easy to inspect.
    score = (
        drop_frac
        + _delta_frac(remove_delta, original_activation)
        + _delta_frac(add_delta, original_activation)
        + (0.5 if same_field else 0.0)
        - (0.05 * max(n_selected - 1, 0))
    )
    return {
        "score": score,
        "drop_frac": drop_frac,
        "remove": remove,
        "add": add,
        "same_top_field": same_field,
        "n_selected": n_selected,
    }


def _select_exemplar(dataset_packet: dict) -> dict:
    scored = []
    for example in dataset_packet.get("examples") or []:
        score = _score_example(example)
        scored.append((score["score"], score, example))
    if not scored:
        raise ValueError(f"No examples found for dataset {dataset_packet.get('dataset')}")
    _, score, example = max(scored, key=lambda item: item[0])
    return {
        "dataset": dataset_packet["dataset"],
        "selection": {
            "score": score["score"],
            "drop_frac": score["drop_frac"],
            "same_top_field": score["same_top_field"],
            "n_selected": score["n_selected"],
            "top_remove_field": (score["remove"] or {}).get("column_name"),
            "top_add_field": (score["add"] or {}).get("column_name"),
        },
        "active_row_idx": example["active_row_idx"],
        "donor_row_idx": example["donor_row_idx"],
        "donor_distance": example.get("donor_distance"),
        "original_activation": example.get("original_activation"),
        "final_activation": example.get("final_activation"),
        "total_drop": example.get("total_drop"),
        "total_drop_frac": example.get("total_drop_frac"),
        "suppression_tier": example.get("suppression_tier"),
        "stop_reason": example.get("stop_reason"),
        "selected_columns": example.get("selected_columns") or [],
        "changed_fields": example.get("changed_fields") or [],
        "best_remove_from_active": score["remove"],
        "best_add_to_contrast": score["add"],
        "active_row": example.get("active_row") or {},
        "donor_contrast_row": example.get("donor_contrast_row") or {},
        "optimized_patched_row": example.get("optimized_patched_row") or {},
    }


def _row_table(exemplar: dict) -> list[str]:
    active = exemplar.get("active_row") or {}
    donor = exemplar.get("donor_contrast_row") or {}
    patched = exemplar.get("optimized_patched_row") or {}
    fields = list(active.keys())
    changed = {field["column"] for field in exemplar.get("changed_fields") or []}
    differing = {field for field in fields if active.get(field) != donor.get(field)}
    shown = changed | differing
    ordered = [field for field in fields if field in changed] + [
        field for field in fields if field not in changed and field in shown
    ]
    lines = [
        "| field | active row | donor row | patched active row |",
        "| --- | --- | --- | --- |",
    ]
    for field in ordered:
        marker = "**" if field in changed else ""
        lines.append(
            f"| {marker}`{field}`{marker} | "
            f"{active.get(field)} | {donor.get(field)} | {patched.get(field)} |"
        )
    hidden = len(fields) - len(ordered)
    if hidden > 0:
        lines.append(f"| ({hidden} identical fields omitted; full rows in JSON) | | | |")
    return lines


def _render_feature_markdown(packet: dict, exemplars: list[dict]) -> str:
    lines = [
        f"# Mitra f{packet['feat']} Causal Exemplars",
        "",
        "Each section shows one selected activating row, one matched non-activating donor row,",
        "and the donor-replacement patch that changed SAE activation. This artifact is for",
        "inspection and evidence review, not validation scoring.",
        "",
    ]
    for ex in exemplars:
        remove = ex.get("best_remove_from_active") or {}
        add = ex.get("best_add_to_contrast") or {}
        lines.extend(
            [
                f"## Dataset: {ex['dataset']}",
                "",
                f"- active row: {ex['active_row_idx']}",
                f"- donor contrast row: {ex['donor_row_idx']}",
                f"- activation: {_format_num(ex.get('original_activation'))} -> {_format_num(ex.get('final_activation'))}",
                f"- drop fraction: {_format_num(ex.get('total_drop_frac'))}",
                f"- selected patch fields: {', '.join(ex.get('selected_columns') or []) or '(none)'}",
                f"- selection score: {_format_num(ex['selection']['score'])}",
                "",
                "Patch fields:",
                "",
                "| field | active value | donor value |",
                "| --- | --- | --- |",
            ]
        )
        for field in ex.get("changed_fields") or []:
            lines.append(
                f"| `{field['column']}` | {field.get('active_value')} | {field.get('donor_value')} |"
            )
        if not ex.get("changed_fields"):
            lines.append("| (none) | | |")
        lines.extend(
            [
                "",
                "Bidirectional one-field checks:",
                "",
                "| direction | field | original act | patched act | delta | active value | donor value |",
                "| --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        if remove:
            lines.append(
                f"| remove from active | `{remove['column_name']}` | "
                f"{_format_num(remove.get('original_activation'))} | "
                f"{_format_num(remove.get('patched_activation'))} | "
                f"{_format_num(remove.get('delta'))} | "
                f"{remove.get('active_value')} | {remove.get('donor_value')} |"
            )
        else:
            lines.append("| remove from active | (none positive) | | | | | |")
        if add:
            lines.append(
                f"| add to contrast | `{add['column_name']}` | "
                f"{_format_num(add.get('original_activation'))} | "
                f"{_format_num(add.get('patched_activation'))} | "
                f"{_format_num(add.get('delta'))} | "
                f"{add.get('active_value')} | {add.get('donor_value')} |"
            )
        else:
            lines.append("| add to contrast | (none positive) | | | | | |")
        lines.extend(["", "Row differences:", ""])
        lines.extend(_row_table(ex))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_packets(packet_dir: Path, model: str, features: list[int] | None) -> list[Path]:
    if features:
        return [packet_dir / f"{model}_f{feat}_suppression_label_packet.json" for feat in features]
    return sorted(packet_dir.glob(f"{model}_f*_suppression_label_packet.json"))


def build_exemplars(
    *,
    packet_dir: Path,
    out_dir: Path,
    model: str,
    features: list[int] | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_packets = []
    for packet_path in _load_packets(packet_dir, model, features):
        packet = json.loads(packet_path.read_text())
        exemplars = [_select_exemplar(ds) for ds in packet.get("datasets") or []]
        payload = {
            "model": packet.get("model", model),
            "feat": packet["feat"],
            "source_packet": str(packet_path),
            "purpose": "concrete causal exemplar evidence; not a validation-scored label artifact",
            "exemplars": exemplars,
        }
        json_path = out_dir / f"{model}_f{packet['feat']}_causal_exemplars.json"
        md_path = out_dir / f"{model}_f{packet['feat']}_causal_exemplars.md"
        json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
        md_path.write_text(_render_feature_markdown(packet, exemplars))
        manifest_packets.append(
            {
                "feat": packet["feat"],
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "datasets": len(exemplars),
                "mean_selection_score": sum(ex["selection"]["score"] for ex in exemplars)
                / len(exemplars)
                if exemplars
                else math.nan,
                "same_top_field_rate": sum(
                    1 for ex in exemplars if ex["selection"]["same_top_field"]
                )
                / len(exemplars)
                if exemplars
                else math.nan,
            }
        )
    manifest = {
        "packet_dir": str(packet_dir),
        "out_dir": str(out_dir),
        "model": model,
        "packets": manifest_packets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "suppression_label_packets_v2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "causal_exemplars_v1",
    )
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--features", type=int, nargs="*", default=None)
    args = parser.parse_args()

    manifest = build_exemplars(
        packet_dir=args.packet_dir,
        out_dir=args.out_dir,
        model=args.model,
        features=args.features,
    )
    for packet in manifest["packets"]:
        print(
            f"f{packet['feat']}: datasets={packet['datasets']} "
            f"mean_score={_format_num(packet['mean_selection_score'])} "
            f"same_top_field_rate={_format_num(packet['same_top_field_rate'])} "
            f"md={packet['markdown_path']}"
        )
    print(f"Wrote manifest {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
