#!/usr/bin/env python3
"""Build label-ready packets from optimized activation suppression results.

The optimizer output contains row indices, donor indices, selected donor
replacement columns, and activation drops. This script reconstructs the raw
activating, donor contrast, and optimized patched rows, then writes:

- one JSON packet per feature with full row audit trail
- one Markdown packet per feature with compact causal evidence for label agents
- a combined JSON manifest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.patch_activation_probe import _load_raw_mitra_context_query


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


def _patched_row(active: pd.Series, donor: pd.Series, selected: list[str]) -> pd.Series:
    patched = active.copy()
    for col in selected:
        if col in patched.index:
            patched[col] = donor[col]
    return patched


def _changed_fields(active: pd.Series, donor: pd.Series, selected: list[str]) -> list[dict]:
    fields = []
    for col in selected:
        if col not in active.index:
            fields.append(
                {
                    "column": col,
                    "active_value": "<missing-column>",
                    "donor_value": "<missing-column>",
                    "changed": False,
                }
            )
            continue
        active_value = _json_value(active[col])
        donor_value = _json_value(donor[col])
        fields.append(
            {
                "column": col,
                "active_value": active_value,
                "donor_value": donor_value,
                "changed": active_value != donor_value,
            }
        )
    return fields


def _success_tier(drop_frac: float) -> str:
    if drop_frac >= 0.8:
        return "strong"
    if drop_frac >= 0.5:
        return "partial"
    if drop_frac > 0:
        return "weak"
    return "none"


def _format_num(value: float | int | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def _top_loo(items: list[dict], limit: int) -> list[dict]:
    return sorted(items or [], key=lambda item: float(item.get("delta") or 0.0), reverse=True)[:limit]


def _render_loo_table(lines: list[str], title: str, items: list[dict]) -> None:
    lines.extend(
        [
            title,
            "",
            "| field | original act | patched act | delta | active value | donor value |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if not items:
        lines.append("| (none) | | | | | |")
    for item in items:
        lines.append(
            f"| `{item['column_name']}` | "
            f"{_format_num(item.get('original_activation'))} | "
            f"{_format_num(item.get('patched_activation'))} | "
            f"{_format_num(item.get('delta'))} | "
            f"{item.get('active_value')} | {item.get('donor_value')} |"
        )
    lines.append("")


def _render_markdown(packet: dict) -> str:
    lines = [
        f"# Mitra f{packet['feat']} Suppression Evidence",
        "",
        "Rows are optimized donor-replacement tests. For each activating row, selected fields",
        "were copied from a matched non-activating donor row until SAE activation dropped or no",
        "suppressive field remained.",
        "",
        "Use this as causal context: fields that repeatedly suppress firing are candidates for",
        "what the SAE feature depends on. Failed or weak suppression is also evidence that the",
        "concept may be heterogeneous or not captured by the tested fields.",
        "",
        "## Feature Summary",
        "",
        f"- rows tested: {packet['summary']['rows']}",
        f"- mean drop fraction: {_format_num(packet['summary']['mean_drop_frac'])}",
        f"- median drop fraction: {_format_num(packet['summary']['median_drop_frac'])}",
        f"- strong suppression rate: {_format_num(packet['summary']['strong_rate'])}",
        f"- common selected fields: {', '.join(packet['summary']['common_selected_columns']) or '(none)'}",
        "",
    ]
    for ds in packet["datasets"]:
        lines.extend(
            [
                f"## Dataset: {ds['dataset']}",
                "",
                f"- rows tested: {ds['summary']['rows']}",
                f"- mean drop fraction: {_format_num(ds['summary']['mean_drop_frac'])}",
                f"- strong suppression rate: {_format_num(ds['summary']['strong_rate'])}",
                f"- common selected fields: {', '.join(ds['summary']['common_selected_columns']) or '(none)'}",
                "",
            ]
        )
        for ex in ds["examples"]:
            selected = ", ".join(ex["selected_columns"]) or "(none)"
            lines.extend(
                [
                    f"### Example active row {ex['active_row_idx']} -> donor {ex['donor_row_idx']}",
                    "",
                    f"- suppression tier: {ex['suppression_tier']}",
                    f"- activation: {_format_num(ex['original_activation'])} -> {_format_num(ex['final_activation'])}",
                    f"- drop fraction: {_format_num(ex['total_drop_frac'])}",
                    f"- selected fields: {selected}",
                    f"- stop reason: {ex['stop_reason']}",
                    "",
                    "Top LOO remove-from-active drops copy one donor value into the activating row.",
                    "Top LOO add-to-contrast increases copy one active value into the donor contrast row.",
                    "",
                    "| field | active value | donor value |",
                    "| --- | --- | --- |",
                ]
            )
            if ex["changed_fields"]:
                for field in ex["changed_fields"]:
                    lines.append(
                        f"| `{field['column']}` | {field['active_value']} | {field['donor_value']} |"
                    )
            else:
                lines.append("| (none) | | |")
            lines.append("")
            _render_loo_table(
                lines,
                "Top LOO remove-from-active drops",
                ex.get("top_loo_remove", []),
            )
            _render_loo_table(
                lines,
                "Top LOO add-to-contrast increases",
                ex.get("top_loo_add", []),
            )
    return "\n".join(lines).rstrip() + "\n"


def _summarize_examples(examples: list[dict]) -> dict:
    drops = [float(ex["total_drop_frac"]) for ex in examples]
    selected_counter = Counter()
    for ex in examples:
        selected_counter.update(ex["selected_columns"])
    return {
        "rows": len(examples),
        "mean_drop_frac": float(np.mean(drops)) if drops else None,
        "median_drop_frac": float(np.median(drops)) if drops else None,
        "strong_rate": float(np.mean([drop >= 0.8 for drop in drops])) if drops else None,
        "common_selected_columns": [
            col for col, _ in selected_counter.most_common(10)
        ],
    }


def build_packets(
    *,
    suppression_path: Path,
    out_dir: Path,
    include_full_rows: bool,
    top_loo: int,
) -> dict:
    payload = json.loads(suppression_path.read_text())
    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for result in payload.get("results") or []:
        grouped[int(result["feat"])][str(result["dataset"])].append(result)

    packets = []
    raw_cache: dict[tuple[str, str], pd.DataFrame] = {}
    model = str(payload.get("config", {}).get("model", "mitra"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for feat, by_dataset in sorted(grouped.items()):
        dataset_packets = []
        all_examples = []
        for dataset, results in sorted(by_dataset.items()):
            cache_key = (model, dataset)
            if cache_key not in raw_cache:
                _, _, X_query_raw, _, _, _ = _load_raw_mitra_context_query(model, dataset)
                raw_cache[cache_key] = X_query_raw
            X_query_raw = raw_cache[cache_key]

            examples = []
            for result in results:
                active_idx = int(result["active_row_idx"])
                donor_idx = int(result["donor_row_idx"])
                active = X_query_raw.iloc[active_idx]
                donor = X_query_raw.iloc[donor_idx]
                selected = [str(col) for col in result.get("selected_columns") or []]
                patched = _patched_row(active, donor, selected)
                example = {
                    "model": model,
                    "feat": feat,
                    "dataset": dataset,
                    "active_row_idx": active_idx,
                    "donor_row_idx": donor_idx,
                    "donor_distance": result.get("donor_distance"),
                    "original_activation": result.get("original_activation"),
                    "final_activation": result.get("final_activation"),
                    "total_drop": result.get("total_drop"),
                    "total_drop_frac": result.get("total_drop_frac"),
                    "suppression_tier": _success_tier(float(result.get("total_drop_frac") or 0.0)),
                    "n_selected": result.get("n_selected"),
                    "stop_reason": result.get("stop_reason"),
                    "selected_columns": selected,
                    "selected_column_indices": result.get("selected_column_indices") or [],
                    "changed_fields": _changed_fields(active, donor, selected),
                    "top_loo_remove": _top_loo(result.get("loo_remove") or [], top_loo),
                    "top_loo_add": _top_loo(result.get("loo_add") or [], top_loo),
                    "steps": result.get("steps") or [],
                }
                if include_full_rows:
                    example.update(
                        {
                            "active_row": _row_dict(active),
                            "donor_contrast_row": _row_dict(donor),
                            "optimized_patched_row": _row_dict(patched),
                        }
                    )
                examples.append(example)
                all_examples.append(example)
            dataset_packets.append(
                {
                    "dataset": dataset,
                    "summary": _summarize_examples(examples),
                    "examples": examples,
                }
            )

        packet = {
            "model": model,
            "feat": feat,
            "source": str(suppression_path),
            "summary": _summarize_examples(all_examples),
            "datasets": dataset_packets,
        }
        json_path = out_dir / f"{model}_f{feat}_suppression_label_packet.json"
        md_path = out_dir / f"{model}_f{feat}_suppression_label_packet.md"
        json_path.write_text(json.dumps(packet, indent=2))
        md_path.write_text(_render_markdown(packet))
        packets.append(
            {
                "feat": feat,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "summary": packet["summary"],
            }
        )

    manifest = {
        "source": str(suppression_path),
        "out_dir": str(out_dir),
        "packets": packets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suppression", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "label_packets",
    )
    parser.add_argument(
        "--no-full-rows",
        action="store_true",
        help="Omit full active/donor/patched row payloads from JSON packets.",
    )
    parser.add_argument("--top-loo", type=int, default=5)
    args = parser.parse_args()

    manifest = build_packets(
        suppression_path=args.suppression,
        out_dir=args.out_dir,
        include_full_rows=not args.no_full_rows,
        top_loo=args.top_loo,
    )
    for packet in manifest["packets"]:
        summary = packet["summary"]
        print(
            f"f{packet['feat']}: rows={summary['rows']} "
            f"mean_drop={_format_num(summary['mean_drop_frac'])} "
            f"strong_rate={_format_num(summary['strong_rate'])} "
            f"md={packet['markdown_path']}"
        )
    print(f"Wrote manifest {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
