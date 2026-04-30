#!/usr/bin/env python3
"""Rank intervention-selected strong-model features for figure panels.

For the combined intervention figure, a row is drawn only when both the
strong-model ablation and weak-model transfer moved the prediction. This script
uses that same row mask, then ranks the strong model's selected features over
the union of drawn rows for each panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT


DEFAULT_DATASETS = [
    "credit-g",
    "students_dropout_and_academic_success",
    "houses",
]


def _rank_dataset(
    *,
    dataset: str,
    pair: str,
    ablation_dir: Path,
    transfer_dir: Path,
    top_k: int,
) -> dict:
    ablation_path = ablation_dir / pair / f"{dataset}.npz"
    transfer_path = transfer_dir / pair / f"{dataset}.npz"
    if not ablation_path.exists():
        raise FileNotFoundError(ablation_path)
    if not transfer_path.exists():
        raise FileNotFoundError(transfer_path)

    ablation = np.load(ablation_path, allow_pickle=True)
    transfer = np.load(transfer_path, allow_pickle=True)
    drawn_mask = (
        ablation["strong_wins"]
        & (ablation["optimal_k"] > 0)
        & (transfer["optimal_k"] > 0)
    )
    selected = ablation["selected_features"][drawn_mask]
    row_indices = ablation["row_indices"][drawn_mask]

    counts: Counter[int] = Counter()
    first_counts: Counter[int] = Counter()
    rank_sum: Counter[int] = Counter()
    rows_by_feature: dict[int, list[dict]] = defaultdict(list)
    for local_row, feature_row in enumerate(selected):
        features = [int(feat) for feat in feature_row if int(feat) >= 0]
        if not features:
            continue
        first_counts[features[0]] += 1
        for rank, feat in enumerate(features, start=1):
            counts[feat] += 1
            rank_sum[feat] += rank
            rows_by_feature[feat].append(
                {
                    "row_idx": int(row_indices[local_row]),
                    "rank_in_row": int(rank),
                }
            )

    ranked = []
    for feature, count in counts.most_common(top_k):
        ranked.append(
            {
                "feature": int(feature),
                "count": int(count),
                "first_count": int(first_counts[feature]),
                "mean_rank": float(rank_sum[feature] / count),
                "example_rows": rows_by_feature[feature][:10],
            }
        )

    return {
        "dataset": dataset,
        "pair": pair,
        "strong_model": str(ablation["strong_model"]),
        "weak_model": str(ablation["weak_model"]),
        "drawn_rows": int(drawn_mask.sum()),
        "strong_wins": int(ablation["strong_wins"].sum()),
        "feature_instances": int(sum(counts.values())),
        "unique_features": int(len(counts)),
        "top_features": ranked,
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Panel Intervention Feature Rankings",
        "",
        "Features are ranked over the union of rows drawn in the combined intervention panel.",
        "Rows are included only when both ablation and transfer moved the prediction.",
        "",
    ]
    for panel in payload["panels"]:
        lines.extend(
            [
                f"## {panel['dataset']}",
                "",
                f"- strong model: {panel['strong_model']}",
                f"- weak model: {panel['weak_model']}",
                f"- drawn rows: {panel['drawn_rows']} / {panel['strong_wins']} strong-win rows",
                f"- selected feature instances: {panel['feature_instances']}",
                f"- unique selected features: {panel['unique_features']}",
                "",
                "| rank | feature | count | first-count | mean rank | example row ids |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for rank, item in enumerate(panel["top_features"], start=1):
            rows = ", ".join(str(row["row_idx"]) for row in item["example_rows"][:5])
            lines.append(
                f"| {rank} | f{item['feature']} | {item['count']} | "
                f"{item['first_count']} | {item['mean_rank']:.2f} | {rows} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="carte_vs_mitra")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "ablation_sweep_tols",
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "transfer_global_mnnp90_trained_tols",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "concept_patch_probes"
        / "panel_intervention_feature_rankings.json",
    )
    args = parser.parse_args()

    payload = {
        "pair": args.pair,
        "ablation_dir": str(args.ablation_dir),
        "transfer_dir": str(args.transfer_dir),
        "top_k": args.top_k,
        "panels": [
            _rank_dataset(
                dataset=dataset,
                pair=args.pair,
                ablation_dir=args.ablation_dir,
                transfer_dir=args.transfer_dir,
                top_k=args.top_k,
            )
            for dataset in args.datasets
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    md_path = args.out.with_suffix(".md")
    md_path.write_text(_render_markdown(payload))

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "rank",
                "feature",
                "count",
                "first_count",
                "mean_rank",
                "drawn_rows",
                "unique_features",
            ],
        )
        writer.writeheader()
        for panel in payload["panels"]:
            for rank, item in enumerate(panel["top_features"], start=1):
                writer.writerow(
                    {
                        "dataset": panel["dataset"],
                        "rank": rank,
                        "feature": item["feature"],
                        "count": item["count"],
                        "first_count": item["first_count"],
                        "mean_rank": f"{item['mean_rank']:.6f}",
                        "drawn_rows": panel["drawn_rows"],
                        "unique_features": panel["unique_features"],
                    }
                )

    for panel in payload["panels"]:
        top = ", ".join(f"f{item['feature']}({item['count']})" for item in panel["top_features"][:5])
        print(f"{panel['dataset']}: {top}")
    print(f"Wrote {args.out}, {md_path}, and {csv_path}")


if __name__ == "__main__":
    main()
