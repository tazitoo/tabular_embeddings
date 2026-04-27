#!/usr/bin/env python3
"""Within-dataset row patch probes for SAE feature activation.

This is a lightweight empirical test loop for concept labeling. It starts from
the current contrastive evidence, picks columns that best separate activating
and contrast rows in the model's preprocessed query space, patches those columns
between matched same-dataset donor/recipient rows, re-extracts Mitra embeddings,
and measures target SAE feature activation deltas.

The goal is not to produce final labels. It is to verify whether row-level
patching can supply causal evidence for or against candidate mechanisms.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts._project_root import PROJECT_ROOT
from scripts.concepts.label_contrastive_mesh import CONTRASTIVE_DIR
from scripts.intervention.intervene_lib import (
    load_norm_stats,
    load_sae,
    SPLITS_PATH,
)
from scripts.concepts.row_source import load_row_source_row_indices
from scripts.intervention.context_sampling import select_context_indices
from models.mitra_embeddings import MitraEmbeddingExtractor


@dataclass
class PatchResult:
    model: str
    feat: int
    dataset: str
    hypothesis_id: str
    agent_id: str
    source_dataset: str
    portable_hypothesis: str
    local_hypothesis: str
    patch_kind: str
    expected_direction: str
    candidate_column_index: int
    candidate_column_name: str
    column_index: int
    column_name: str
    direction: str
    recipient_row_idx: int
    donor_row_idx: int
    context_distance: float
    original_activation: float
    patched_activation: float
    delta: float


def _load_evidence_rows(model: str, feat: int, dataset: str) -> pd.DataFrame:
    path = CONTRASTIVE_DIR / model / f"f{feat}_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "row_idx" not in df.columns:
        raise ValueError(f"{path} is missing row_idx")
    return df


def _encode_frame_for_matching(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert raw mixed-type rows to numeric codes for distance/ranking only."""
    names = list(df.columns)
    cols: list[np.ndarray] = []
    for name in names:
        s = df[name]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            vals = numeric.to_numpy(dtype=np.float32)
        else:
            vals = pd.factorize(s.astype("string").fillna("<NA>"), sort=True)[0].astype(
                np.float32
            )
        if np.isnan(vals).any():
            med = np.nanmedian(vals)
            vals = np.nan_to_num(vals, nan=float(med) if np.isfinite(med) else 0.0)
        cols.append(vals)
    return np.column_stack(cols).astype(np.float32), names


def _load_raw_mitra_context_query(model: str, dataset: str):
    """Load raw rows aligned with the row source used by contrastive evidence."""
    if model != "mitra":
        raise NotImplementedError("raw context/query loader currently supports mitra")

    from data.extended_loader import load_tabarena_dataset
    from sklearn.preprocessing import LabelEncoder

    splits = json.loads(SPLITS_PATH.read_text())
    split_info = splits[dataset]
    task = split_info["task_type"]
    train_idx = np.asarray(split_info["train_indices"], dtype=np.int64)
    row_indices = load_row_source_row_indices(model, dataset, "sae_test")
    if row_indices is None:
        raise FileNotFoundError(f"No sae_test row indices for {model}/{dataset}")

    loaded = load_tabarena_dataset(dataset, max_samples=999999)
    if loaded is None:
        raise ValueError(f"Failed to load dataset {dataset}")
    X, y = loaded[0], np.asarray(loaded[1])

    y_encoded = y
    if task == "classification":
        y_encoded = LabelEncoder().fit_transform(y)

    ctx_local = select_context_indices(
        n_rows=len(train_idx),
        y_train=y_encoded[train_idx],
        max_context=600,
        task=task,
        dataframe_style=True,
    )
    X_context = X.iloc[train_idx[ctx_local]].reset_index(drop=True)
    y_context = y_encoded[train_idx[ctx_local]]
    X_query = X.iloc[row_indices].reset_index(drop=True)
    y_query = y_encoded[row_indices]
    return X_context, y_context, X_query, y_query, row_indices, task


def _column_separation_scores(
    X_query: np.ndarray,
    active_rows: np.ndarray,
    contrast_rows: np.ndarray,
) -> np.ndarray:
    """Active-vs-contrast standardized mean difference by column."""
    X = np.asarray(X_query, dtype=np.float32)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    active_mean = np.nanmean(X[active_rows], axis=0)
    contrast_mean = np.nanmean(X[contrast_rows], axis=0)
    return np.abs(active_mean - contrast_mean) / std


def _candidate_columns(
    X_query: np.ndarray,
    active_rows: np.ndarray,
    contrast_rows: np.ndarray,
    *,
    top_k: int,
) -> tuple[list[int], np.ndarray]:
    """Rank columns by active-vs-contrast standardized mean difference."""
    score = _column_separation_scores(X_query, active_rows, contrast_rows)
    order = np.argsort(-score)
    cols = [int(i) for i in order[:top_k] if np.isfinite(score[i]) and score[i] > 0]
    return cols, score


def _control_column(
    candidate_col: int,
    candidate_cols: set[int],
    scores: np.ndarray,
) -> int | None:
    """Pick a low-separation non-candidate column as an off-hypothesis control."""
    valid = [
        int(i)
        for i, score in enumerate(scores)
        if i != candidate_col and i not in candidate_cols and np.isfinite(score)
    ]
    if not valid:
        return None
    order = sorted(valid, key=lambda i: (scores[i], i))
    return order[0]


def _load_patch_plan(path: Path | None) -> list[dict]:
    if path is None:
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("patch_plans", "plans", "records"):
            if isinstance(raw.get(key), list):
                return raw[key]
        if isinstance(raw.get("per_dataset_patch_plan"), dict):
            records = []
            common = {
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "per_dataset_patch_plan",
                    "patch_plans",
                    "plans",
                    "records",
                }
            }
            for dataset, plan in raw["per_dataset_patch_plan"].items():
                record = dict(common)
                record.update(plan)
                record.setdefault("dataset", dataset)
                records.append(record)
            return records
    raise ValueError(
        "--patch-plan must be a JSON list, or a dict with patch_plans/plans/records/per_dataset_patch_plan"
    )


def _plans_for_dataset(
    patch_plans: list[dict],
    *,
    model: str,
    feat: int,
    dataset: str,
) -> list[dict]:
    selected = []
    for plan in patch_plans:
        if plan.get("model", model) != model:
            continue
        if int(plan.get("feat", plan.get("feature", feat))) != feat:
            continue
        if plan.get("dataset", plan.get("target_dataset")) != dataset:
            continue
        selected.append(plan)
    return selected


def _column_indices_from_plan(plan: dict, col_names: list[str]) -> list[int]:
    requested = plan.get("columns", plan.get("patch_columns", []))
    by_name = {name: i for i, name in enumerate(col_names)}
    cols = []
    for item in requested:
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str) and item in by_name:
            idx = by_name[item]
        elif isinstance(item, str) and item.isdigit():
            idx = int(item)
        else:
            raise ValueError(f"Unknown patch column {item!r}")
        if idx < 0 or idx >= len(col_names):
            raise ValueError(f"Patch column index out of range: {idx}")
        cols.append(idx)
    return list(dict.fromkeys(cols))


def _nearest_donor(
    X_scaled: np.ndarray,
    recipient: int,
    donors: np.ndarray,
    *,
    exclude_col: int,
) -> tuple[int, float]:
    cols = np.arange(X_scaled.shape[1]) != exclude_col
    diff = X_scaled[donors][:, cols] - X_scaled[recipient, cols]
    dist = np.sqrt(np.nanmean(diff * diff, axis=1))
    best = int(np.nanargmin(dist))
    return int(donors[best]), float(dist[best])


def _extract_feature_activations(
    *,
    model: str,
    feat: int,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query_batch: np.ndarray,
    task: str,
    dataset: str,
    device: str,
) -> np.ndarray:
    if model != "mitra":
        raise NotImplementedError("patch_activation_probe currently supports --model mitra")

    extractor = MitraEmbeddingExtractor(device=device, n_estimators=1, fine_tune=False)
    result = extractor.extract_embeddings(
        X_context,
        y_context,
        X_query_batch,
        task=task,
    )
    emb = np.asarray(result.embeddings, dtype=np.float32)
    mean, std = load_norm_stats(model, dataset, device=device)
    sae, _ = load_sae(model, device=device)
    with torch.no_grad():
        x = torch.tensor(emb, dtype=torch.float32, device=device)
        x = (x - mean) / std
        acts = sae.encode(x)[:, feat].detach().cpu().numpy()
    return acts


def probe_feature_dataset(
    *,
    model: str,
    feat: int,
    dataset: str,
    top_cols: int,
    rows_per_direction: int,
    device: str,
    patch_plans: list[dict] | None = None,
) -> list[PatchResult]:
    X_context, y_context, X_query_raw, _, _, task = _load_raw_mitra_context_query(
        model, dataset
    )
    X_query, col_names = _encode_frame_for_matching(X_query_raw)
    evidence = _load_evidence_rows(model, feat, dataset)
    active_rows = evidence[evidence.label == "activating"].row_idx.astype(int).to_numpy()
    contrast_rows = evidence[evidence.label == "contrast"].row_idx.astype(int).to_numpy()
    active_rows = active_rows[(active_rows >= 0) & (active_rows < len(X_query))]
    contrast_rows = contrast_rows[(contrast_rows >= 0) & (contrast_rows < len(X_query))]
    if len(active_rows) == 0 or len(contrast_rows) == 0:
        return []

    med = np.nanmedian(X_query, axis=0)
    scale = np.nanstd(X_query, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    X_scaled = (np.nan_to_num(X_query, nan=0.0) - med) / scale

    separation_scores = _column_separation_scores(X_query, active_rows, contrast_rows)
    dataset_plans = _plans_for_dataset(
        patch_plans or [],
        model=model,
        feat=feat,
        dataset=dataset,
    )
    if dataset_plans:
        candidate_cols = []
        for plan in dataset_plans:
            candidate_cols.extend(_column_indices_from_plan(plan, col_names))
        candidate_cols = list(dict.fromkeys(candidate_cols))
    else:
        candidate_cols, separation_scores = _candidate_columns(
        X_query,
        active_rows,
        contrast_rows,
        top_k=top_cols,
        )
        dataset_plans = [
            {
                "hypothesis_id": "auto_separator",
                "agent_id": "auto",
                "source_dataset": dataset,
                "portable_hypothesis": "",
                "local_hypothesis": "",
                "columns": candidate_cols,
                "expected_add_delta": "increase",
                "expected_remove_delta": "decrease",
            }
        ]
    candidate_set = set(candidate_cols)
    recipients_active = active_rows[:rows_per_direction]
    recipients_contrast = contrast_rows[:rows_per_direction]

    batch_rows: list[pd.Series] = []
    metadata: list[dict] = []
    for plan in dataset_plans:
        plan_cols = _column_indices_from_plan(plan, col_names)
        for col in plan_cols:
            ctrl_col = _control_column(col, candidate_set, separation_scores)
            base_meta = {
                "hypothesis_id": str(plan.get("hypothesis_id", "manual")),
                "agent_id": str(plan.get("agent_id", "")),
                "source_dataset": str(plan.get("source_dataset", dataset)),
                "portable_hypothesis": str(plan.get("portable_hypothesis", "")),
                "local_hypothesis": str(plan.get("local_hypothesis", plan.get("label", ""))),
                "candidate_col": col,
            }
            expected_remove = str(plan.get("expected_remove_delta", "decrease"))
            expected_add = str(plan.get("expected_add_delta", "increase"))
            for recipient in recipients_active:
                donor, dist = _nearest_donor(
                    X_scaled,
                    int(recipient),
                    contrast_rows,
                    exclude_col=col,
                )
                patched = X_query_raw.iloc[int(recipient)].copy()
                patched.iloc[col] = X_query_raw.iloc[donor, col]
                batch_rows.extend([X_query_raw.iloc[int(recipient)], patched])
                metadata.append(
                    {
                        **base_meta,
                        "patch_kind": "candidate",
                        "patch_col": col,
                        "direction": "remove_from_active",
                        "expected_direction": expected_remove,
                        "recipient": int(recipient),
                        "donor": donor,
                        "dist": dist,
                    }
                )
                if ctrl_col is not None:
                    control = X_query_raw.iloc[int(recipient)].copy()
                    control.iloc[ctrl_col] = X_query_raw.iloc[donor, ctrl_col]
                    batch_rows.extend([X_query_raw.iloc[int(recipient)], control])
                    metadata.append(
                        {
                            **base_meta,
                            "patch_kind": "control",
                            "patch_col": ctrl_col,
                            "direction": "remove_from_active",
                            "expected_direction": expected_remove,
                            "recipient": int(recipient),
                            "donor": donor,
                            "dist": dist,
                        }
                    )
            for recipient in recipients_contrast:
                donor, dist = _nearest_donor(
                    X_scaled,
                    int(recipient),
                    active_rows,
                    exclude_col=col,
                )
                patched = X_query_raw.iloc[int(recipient)].copy()
                patched.iloc[col] = X_query_raw.iloc[donor, col]
                batch_rows.extend([X_query_raw.iloc[int(recipient)], patched])
                metadata.append(
                    {
                        **base_meta,
                        "patch_kind": "candidate",
                        "patch_col": col,
                        "direction": "add_to_contrast",
                        "expected_direction": expected_add,
                        "recipient": int(recipient),
                        "donor": donor,
                        "dist": dist,
                    }
                )
                if ctrl_col is not None:
                    control = X_query_raw.iloc[int(recipient)].copy()
                    control.iloc[ctrl_col] = X_query_raw.iloc[donor, ctrl_col]
                    batch_rows.extend([X_query_raw.iloc[int(recipient)], control])
                    metadata.append(
                        {
                            **base_meta,
                            "patch_kind": "control",
                            "patch_col": ctrl_col,
                            "direction": "add_to_contrast",
                            "expected_direction": expected_add,
                            "recipient": int(recipient),
                            "donor": donor,
                            "dist": dist,
                        }
                    )

    if not batch_rows:
        return []

    activations = _extract_feature_activations(
        model=model,
        feat=feat,
        X_context=X_context,
        y_context=y_context,
        X_query_batch=pd.DataFrame(batch_rows).reset_index(drop=True),
        task=task,
        dataset=dataset,
        device=device,
    )

    results: list[PatchResult] = []
    for i, meta in enumerate(metadata):
        original = float(activations[2 * i])
        patched = float(activations[2 * i + 1])
        candidate_col = int(meta["candidate_col"])
        patch_col = int(meta["patch_col"])
        results.append(
            PatchResult(
                model=model,
                feat=feat,
                dataset=dataset,
                hypothesis_id=meta["hypothesis_id"],
                agent_id=meta["agent_id"],
                source_dataset=meta["source_dataset"],
                portable_hypothesis=meta["portable_hypothesis"],
                local_hypothesis=meta["local_hypothesis"],
                patch_kind=meta["patch_kind"],
                expected_direction=meta["expected_direction"],
                candidate_column_index=candidate_col,
                candidate_column_name=col_names[candidate_col],
                column_index=patch_col,
                column_name=col_names[patch_col],
                direction=meta["direction"],
                recipient_row_idx=int(meta["recipient"]),
                donor_row_idx=int(meta["donor"]),
                context_distance=float(meta["dist"]),
                original_activation=original,
                patched_activation=patched,
                delta=patched - original,
            )
        )
    return results


def _datasets_for_feature(
    model: str,
    feat: int,
    limit: int | None,
    task_filter: str,
) -> list[str]:
    ctx_path = CONTRASTIVE_DIR / model / f"f{feat}_context.json"
    ctx = json.loads(ctx_path.read_text())
    datasets = []
    stats = ctx.get("dataset_stats") or {}
    for ds in ctx.get("datasets_used") or []:
        task = (stats.get(ds) or {}).get("task_type")
        if task_filter != "all" and task != task_filter:
            continue
        datasets.append(ds)
    return datasets[:limit] if limit else datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mitra")
    parser.add_argument("--features", nargs="+", type=int, required=True)
    parser.add_argument("--datasets-per-feature", type=int, default=1)
    parser.add_argument(
        "--task-filter",
        choices=["all", "classification", "regression"],
        default="all",
        help="Limit probed datasets by task type. Useful when only classifier/regressor checkpoint is cached.",
    )
    parser.add_argument("--top-cols", type=int, default=3)
    parser.add_argument("--rows-per-direction", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--patch-plan",
        type=Path,
        default=None,
        help=(
            "Optional JSON patch plan. Accepts a list of records or a dict with "
            "patch_plans/plans/records/per_dataset_patch_plan. Each record should "
            "include feat, dataset, hypothesis_id, agent_id, columns."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "concept_patch_probes" / "mitra_patch_probe.json",
    )
    args = parser.parse_args()

    patch_plans = _load_patch_plan(args.patch_plan)
    all_results: list[PatchResult] = []
    errors: list[dict] = []
    for feat in args.features:
        for dataset in _datasets_for_feature(
            args.model,
            feat,
            args.datasets_per_feature,
            args.task_filter,
        ):
            print(f"Probing {args.model} f{feat} {dataset}...", flush=True)
            try:
                all_results.extend(
                    probe_feature_dataset(
                        model=args.model,
                        feat=feat,
                        dataset=dataset,
                        top_cols=args.top_cols,
                        rows_per_direction=args.rows_per_direction,
                        device=args.device,
                        patch_plans=patch_plans,
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
            "top_cols": args.top_cols,
            "rows_per_direction": args.rows_per_direction,
            "device": args.device,
            "patch_plan": str(args.patch_plan) if args.patch_plan else None,
        },
        "patch_plans": patch_plans,
        "results": [asdict(r) for r in all_results],
        "errors": errors,
    }
    args.out.write_text(json.dumps(payload, indent=2))

    df = pd.DataFrame(payload["results"])
    if not df.empty:
        csv_path = args.out.with_suffix(".csv")
        df["expected_match"] = np.select(
            [
                df.expected_direction.eq("increase"),
                df.expected_direction.eq("decrease"),
            ],
            [
                df.delta > 0,
                df.delta < 0,
            ],
            default=np.nan,
        )
        df.to_csv(csv_path, index=False)
        summary = (
            df.groupby(["feat", "hypothesis_id", "dataset", "direction", "patch_kind"])
            .agg(
                count=("delta", "count"),
                mean=("delta", "mean"),
                median=("delta", "median"),
                min=("delta", "min"),
                max=("delta", "max"),
                expected_match_rate=("expected_match", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
        pair_key = [
            "feat",
            "dataset",
            "hypothesis_id",
            "direction",
            "recipient_row_idx",
            "donor_row_idx",
            "candidate_column_index",
        ]
        paired = (
            df.pivot_table(
                index=pair_key,
                columns="patch_kind",
                values="delta",
                aggfunc="first",
            )
            .dropna(subset=["candidate", "control"])
            .reset_index()
        )
        if not paired.empty:
            paired["candidate_minus_control"] = paired["candidate"] - paired["control"]
            paired_summary = (
                paired.groupby(["feat", "hypothesis_id", "dataset", "direction"])[
                    "candidate_minus_control"
                ]
                .agg(["count", "mean", "median", "min", "max"])
                .reset_index()
            )
            print("\nCandidate minus control delta:")
            print(paired_summary.to_string(index=False))
        print(f"Wrote {args.out} and {csv_path}")
    else:
        print(f"No probe results. Wrote {args.out}")


if __name__ == "__main__":
    main()
