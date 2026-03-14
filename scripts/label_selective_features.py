#!/usr/bin/env python3
"""Label ultra-selective SAE features from their activating rows.

Features that fire on <20 rows are self-interpreting: the activating rows
ARE the explanation. This script collects those rows, finds what's
distinctive about them (column-level z-scores vs dataset distribution),
and generates descriptive labels.

Two modes:
  Rule-based (default): generates labels from column z-scores
  --llm: sends actual row data to Claude Haiku for real-world interpretation

Usage:
    python scripts/label_selective_features.py --model tabdpt
    python scripts/label_selective_features.py --model tabdpt --llm
    python scripts/label_selective_features.py --model tabdpt --threshold 20 --top-cols 3
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.extended_loader import TABARENA_DATASETS
from scripts.concept_fingerprint import load_per_dataset_embeddings
from scripts.concept_importance import (
    get_alive_features,
    get_feature_labels,
    get_matryoshka_bands,
    feature_to_band,
)
from scripts.intervene_sae import load_sae
from scripts.row_level_probes import (
    get_column_names,
    get_train_indices,
    reconstruct_query_data,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "row_level_probes"


# ── LLM labeling ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You label neurons in a Sparse Autoencoder trained on tabular foundation model \
embeddings. Each neuron fires on specific data rows across tabular datasets.

You will see the ACTUAL DATA ROWS that activate the neuron, with real column \
names and values, plus which columns are statistically distinctive (z-scores \
vs the dataset). The old label came from dataset-level probe regressions and \
is too generic — ignore it.

Respond with ONLY a concept label (2-6 words). Describe the real-world pattern \
these rows represent, not statistics. \
Examples: "high-leverage bankrupt firms", "frequent repeat donors", \
"large diamond outliers", "apps requesting dangerous permissions", \
"low-income multi-policy holders"."""

_llm_disabled = False


def _init_llm_client():
    """Initialize Anthropic client from environment."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set, LLM labeling disabled")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("LLM client initialized (Claude Haiku)")
        return client
    except ImportError:
        print("Warning: anthropic package not installed")
        return None


def _call_llm(prompt: str, client) -> Optional[str]:
    """Call Claude Haiku for a concept label."""
    global _llm_disabled
    if _llm_disabled or client is None:
        return None
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        label = response.content[0].text.strip().strip('"').strip("'")
        return label if label else None
    except Exception as e:
        err_str = str(e)
        if "credit balance" in err_str or "authentication" in err_str.lower():
            print(f"  LLM fatal error (disabling): {e}")
            _llm_disabled = True
        else:
            print(f"  LLM error: {e}")
        return None


def format_llm_prompt(feat: Dict) -> str:
    """Format a selective feature for LLM labeling.

    Shows the actual rows, distinctive columns, and old label as context
    for what NOT to do.
    """
    lines = [
        f"=== SAE Feature {feat['feature_idx']} (fires on {feat['n_active_rows']} "
        f"of 16,608 rows) ===",
        "",
    ]

    # Old label (as negative example)
    if feat.get("old_label", "none") != "none":
        lines.append(f"Old label (from dataset-level probes, too generic): "
                      f'"{feat["old_label"]}"')
        lines.append("")

    # Distinctive columns
    if feat.get("distinctive_columns"):
        lines.append("DISTINCTIVE COLUMNS (z-score vs dataset distribution):")
        for dc in feat["distinctive_columns"][:5]:
            lines.append(f"  {dc['column']}: z={dc['mean_z']:+.1f} ({dc['direction']})")
        lines.append("")

    # Actual data rows (the key evidence)
    lines.append("ACTIVATING ROWS:")
    for s in feat.get("samples", [])[:10]:  # cap at 10 rows for prompt size
        ds = s["dataset"]
        domain = TABARENA_DATASETS.get(ds, {}).get("domain", "unknown")
        y_str = f", y={s['y']}" if s.get("y") is not None else ""
        lines.append(f"  [{ds} ({domain}){y_str}]")
        if "columns" in s:
            # Show up to 8 most distinctive columns
            cols = s["columns"]
            col_strs = [f"{k}={v}" for k, v in list(cols.items())[:8]]
            lines.append(f"    {', '.join(col_strs)}")
            if len(cols) > 8:
                lines.append(f"    ... ({len(cols) - 8} more columns)")
        lines.append("")

    lines.append("What real-world pattern do these rows represent? (2-6 words only)")
    return "\n".join(lines)


# ── Core logic ────────────────────────────────────────────────────────────────


def find_selective_features(
    activations: np.ndarray,
    alive_features: List[int],
    threshold: int = 20,
) -> List[Tuple[int, np.ndarray]]:
    """Find features that fire on <= threshold rows.

    Returns list of (feat_idx, active_row_indices) sorted by n_active ascending.
    """
    results = []
    for feat_idx in alive_features:
        active = np.where(activations[:, feat_idx] > 0)[0]
        if 0 < len(active) <= threshold:
            results.append((feat_idx, active))
    results.sort(key=lambda x: len(x[1]))
    return results


def compute_dataset_stats(
    X: np.ndarray,
    col_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and std for z-scoring."""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds < 1e-8] = 1.0  # avoid division by zero
    return means, stds


def describe_rows(
    row_values: np.ndarray,
    col_names: List[str],
    ds_means: np.ndarray,
    ds_stds: np.ndarray,
    top_n: int = 3,
) -> List[Tuple[str, float, str]]:
    """Find the most distinctive columns for a set of rows.

    Computes mean z-score of the rows relative to dataset distribution.
    Returns top_n columns by |z|: (col_name, mean_z, direction).
    """
    # Z-score each row relative to dataset
    z_scores = (row_values - ds_means) / ds_stds
    mean_z = np.nanmean(z_scores, axis=0)

    # Rank by absolute z-score
    ranked = np.argsort(np.abs(mean_z))[::-1]

    results = []
    for idx in ranked[:top_n]:
        z = float(mean_z[idx])
        if abs(z) < 0.3:
            break  # nothing distinctive left
        direction = "high" if z > 0 else "low"
        results.append((col_names[idx], z, direction))
    return results


def generate_label(
    feat_idx: int,
    active_rows: np.ndarray,
    activations: np.ndarray,
    row_datasets: np.ndarray,
    dataset_X: Dict[str, np.ndarray],
    dataset_y: Dict[str, np.ndarray],
    dataset_col_names: Dict[str, List[str]],
    dataset_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    top_cols: int = 3,
) -> Dict:
    """Generate a descriptive label for one selective feature.

    Strategy:
    - Single dataset: describe distinctive columns relative to that dataset
    - Multi-dataset: find cross-dataset column patterns (by position, not name)
    """
    feat_acts = activations[:, feat_idx]
    n_active = len(active_rows)

    # Group active rows by dataset
    ds_groups: Dict[str, List[int]] = defaultdict(list)
    for row_idx in active_rows:
        ds_groups[row_datasets[row_idx]].append(row_idx)

    # Per-dataset analysis
    ds_descriptions = []
    all_distinctive = []

    for ds_name, row_indices in sorted(ds_groups.items(), key=lambda x: -len(x[1])):
        if ds_name not in dataset_X:
            continue

        X_ds = dataset_X[ds_name]
        col_names = dataset_col_names[ds_name]
        ds_means, ds_stds = dataset_stats[ds_name]

        # Get the actual values for active rows in this dataset
        # row_indices are global; we need the local dataset row indices
        # We stored which global rows belong to which dataset during build
        # For now, directly use the X values from the row metadata
        # Actually, we need to recover the local values. Let's use the
        # train_indices to map back.
        # Simpler: we already have X_ds (all train rows for this dataset).
        # We need to know which of those rows are the active ones.
        # The global row_indices map into the concatenated array. We need
        # the offset for this dataset.
        # This is passed in via row_datasets matching.

        # Get distinctive columns
        # We need the actual X values for the active rows. These are in X_ds
        # at positions relative to the dataset's slice of the global array.
        distinctive = describe_rows(
            X_ds[row_indices] if len(row_indices) <= len(X_ds) else X_ds[:1],
            col_names, ds_means, ds_stds, top_n=top_cols,
        )
        all_distinctive.extend(distinctive)

        if distinctive:
            col_desc = ", ".join(
                f"{'high' if d == 'high' else 'low'} {c}" for c, z, d in distinctive
            )
            ds_descriptions.append(f"{ds_name}: {col_desc}")
        else:
            ds_descriptions.append(ds_name)

    # Generate label
    datasets = sorted(ds_groups.keys())
    n_datasets = len(datasets)

    if n_datasets == 1:
        ds = datasets[0]
        if ds in dataset_col_names and all_distinctive:
            parts = [f"{d} {c}" for c, z, d in all_distinctive[:top_cols]]
            label = f"{ds} — {', '.join(parts)}"
        else:
            label = f"{ds} specific"
    elif n_datasets <= 3:
        # Check for common column patterns across datasets
        if all_distinctive:
            # Take most common direction+position patterns
            parts = [f"{d} {c}" for c, z, d in all_distinctive[:2]]
            label = f"{'+'.join(datasets[:2])}: {', '.join(parts)}"
        else:
            label = f"selective across {'+'.join(datasets[:3])}"
    else:
        label = f"selective across {n_datasets} datasets"

    # Target analysis
    target_values = []
    for ds_name, row_indices in ds_groups.items():
        if ds_name in dataset_y:
            y_ds = dataset_y[ds_name]
            for ri in row_indices:
                if ri < len(y_ds):
                    target_values.append(float(y_ds[ri]))

    target_info = {}
    if target_values:
        unique_targets = set(target_values)
        if len(unique_targets) == 1:
            target_info = {"all_same_class": target_values[0]}
        elif len(unique_targets) <= 5:
            target_info = {"target_values": sorted(unique_targets)}

    # Build sample table (all rows, since there are so few)
    samples = []
    for ds_name, row_indices in sorted(ds_groups.items()):
        col_names = dataset_col_names.get(ds_name, [])
        X_ds = dataset_X.get(ds_name)
        y_ds = dataset_y.get(ds_name)

        for ri in row_indices:
            sample = {
                "dataset": ds_name,
                "activation": float(feat_acts[active_rows[
                    list(active_rows).index(
                        next(r for r in active_rows if row_datasets[r] == ds_name and r == active_rows[list(active_rows).index(ri if ri in active_rows else active_rows[0])])
                    )
                ]]) if False else float(feat_acts[ri]),  # simplified
                "y": float(y_ds[ri]) if y_ds is not None and ri < len(y_ds) else None,
            }
            if X_ds is not None and ri < len(X_ds):
                sample["columns"] = {
                    col_names[j]: float(X_ds[ri, j])
                    for j in range(min(len(col_names), X_ds.shape[1]))
                }
            samples.append(sample)

    return {
        "feature_idx": feat_idx,
        "n_active_rows": n_active,
        "n_datasets": n_datasets,
        "datasets": datasets,
        "label": label,
        "distinctive_columns": [
            {"column": c, "mean_z": round(z, 2), "direction": d}
            for c, z, d in all_distinctive
        ],
        "target": target_info,
        "per_dataset": ds_descriptions,
        "samples": samples,
    }


# ── Build pipeline ────────────────────────────────────────────────────────────


def build_dataset_cache(
    samples_per_dataset: List[Tuple[str, int]],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[str]],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    Dict[str, Tuple[int, int]],
]:
    """Reconstruct original data for all datasets.

    Returns:
        dataset_X: ds_name -> (n_train, n_cols) original feature values
        dataset_y: ds_name -> (n_train,) target values
        dataset_col_names: ds_name -> column names
        dataset_stats: ds_name -> (means, stds) for z-scoring
        row_datasets: (n_total,) array of dataset names per global row
        ds_offsets: ds_name -> (start, end) in global array
    """
    dataset_X = {}
    dataset_y = {}
    dataset_col_names = {}
    dataset_stats = {}
    row_datasets = []
    ds_offsets = {}

    offset = 0
    for ds_name, count in samples_per_dataset:
        ds_offsets[ds_name] = (offset, offset + count)
        row_datasets.extend([ds_name] * count)
        offset += count

        try:
            X_query, y_query, task = reconstruct_query_data(ds_name)
            train_idx = get_train_indices(len(X_query))

            if len(train_idx) != count:
                train_idx = train_idx[:count] if len(train_idx) > count else train_idx

            X_train = X_query[train_idx]
            y_train = y_query[train_idx]

            # Column names
            real_names = get_column_names(ds_name)
            if real_names and len(real_names) == X_train.shape[1]:
                col_names = real_names
            else:
                col_names = [f"col_{i}" for i in range(X_train.shape[1])]

            dataset_X[ds_name] = X_train
            dataset_y[ds_name] = y_train
            dataset_col_names[ds_name] = col_names
            dataset_stats[ds_name] = compute_dataset_stats(X_train, col_names)

        except Exception as e:
            print(f"  Warning: {ds_name}: {e}")

    return (
        dataset_X, dataset_y, dataset_col_names,
        dataset_stats, np.array(row_datasets), ds_offsets,
    )


def label_selective_features(
    model_key: str,
    threshold: int = 20,
    top_cols: int = 3,
    device: str = "cpu",
    use_llm: bool = False,
) -> Dict:
    """Main pipeline: find and label all ultra-selective features."""

    # Load embeddings and encode through SAE
    print(f"Loading SAE for {model_key}...")
    per_ds = load_per_dataset_embeddings(model_key)
    ds_order = list(per_ds.keys())
    all_emb = np.concatenate([per_ds[ds] for ds in ds_order], axis=0)
    samples_per_dataset = [(ds, len(per_ds[ds])) for ds in ds_order]

    sae, config = load_sae(model_key, device=device)
    sae.eval()
    with torch.no_grad():
        x = torch.tensor(all_emb, dtype=torch.float32, device=device)
        activations = sae.encode(x).cpu().numpy()

    print(f"  {activations.shape[0]} rows, {activations.shape[1]} features")

    # Find selective features
    try:
        alive_features = get_alive_features(model_key)
    except Exception:
        alive_features = [
            i for i in range(activations.shape[1])
            if activations[:, i].max() > 0
        ]

    selective = find_selective_features(activations, alive_features, threshold)
    print(f"  {len(selective)} features fire on <= {threshold} rows")

    if not selective:
        print("  No ultra-selective features found.")
        return {"model": model_key, "threshold": threshold, "features": []}

    # Distribution
    counts = [len(rows) for _, rows in selective]
    print(f"  Active row counts: 1={counts.count(1)}, "
          f"2-5={sum(1 for c in counts if 2 <= c <= 5)}, "
          f"6-{threshold}={sum(1 for c in counts if 6 <= c <= threshold)}")

    # Reconstruct original data
    print(f"\nReconstructing original data for {len(samples_per_dataset)} datasets...")
    (dataset_X, dataset_y, dataset_col_names,
     dataset_stats, row_datasets, ds_offsets) = build_dataset_cache(samples_per_dataset)

    # Remap active row indices to local dataset indices
    # global_idx -> (ds_name, local_idx_within_dataset_slice)
    def global_to_local(global_idx: int) -> Tuple[str, int]:
        ds = row_datasets[global_idx]
        start, _ = ds_offsets[ds]
        return ds, global_idx - start

    # For each selective feature, build local index maps and label
    print(f"\nLabeling {len(selective)} features...")
    try:
        bands = get_matryoshka_bands(model_key)
        existing_labels = get_feature_labels(model_key)
    except Exception:
        bands = None
        existing_labels = {}

    results = []
    for i, (feat_idx, active_global) in enumerate(selective):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(selective)}...")

        # Group by dataset with LOCAL indices
        ds_local: Dict[str, List[int]] = defaultdict(list)
        for gi in active_global:
            ds, local_idx = global_to_local(gi)
            ds_local[ds].append(local_idx)

        # Per-dataset distinctive columns
        all_distinctive = []
        ds_descriptions = []

        for ds_name in sorted(ds_local.keys(), key=lambda d: -len(ds_local[d])):
            if ds_name not in dataset_X:
                ds_descriptions.append(ds_name)
                continue

            local_indices = np.array(ds_local[ds_name])
            X_ds = dataset_X[ds_name]
            col_names = dataset_col_names[ds_name]
            ds_means, ds_stds = dataset_stats[ds_name]

            # Clamp indices
            valid = local_indices[local_indices < len(X_ds)]
            if len(valid) == 0:
                ds_descriptions.append(ds_name)
                continue

            distinctive = describe_rows(
                X_ds[valid], col_names, ds_means, ds_stds, top_n=top_cols,
            )
            all_distinctive.extend(distinctive)

            if distinctive:
                col_desc = ", ".join(f"{d} {c}" for c, z, d in distinctive)
                ds_descriptions.append(f"{ds_name}: {col_desc}")
            else:
                ds_descriptions.append(ds_name)

        # Generate label
        datasets = sorted(ds_local.keys())
        n_datasets = len(datasets)

        if n_datasets == 1 and all_distinctive:
            ds = datasets[0]
            parts = [f"{d} {c}" for c, z, d in all_distinctive[:top_cols]]
            label = f"{ds} — {', '.join(parts)}"
        elif n_datasets <= 3 and all_distinctive:
            parts = [f"{d} {c}" for c, z, d in all_distinctive[:2]]
            label = f"{'+'.join(datasets[:3])}: {', '.join(parts)}"
        elif all_distinctive:
            parts = [f"{d} {c}" for c, z, d in all_distinctive[:2]]
            label = f"{n_datasets} datasets: {', '.join(parts)}"
        else:
            label = f"selective across {n_datasets} datasets"

        # Target analysis
        target_info = {}
        target_vals = []
        for ds_name, local_indices in ds_local.items():
            if ds_name in dataset_y:
                y_ds = dataset_y[ds_name]
                for li in local_indices:
                    if li < len(y_ds):
                        target_vals.append(float(y_ds[li]))
        if target_vals:
            unique = set(target_vals)
            if len(unique) == 1:
                target_info["all_same_class"] = target_vals[0]

        # Build sample table
        samples = []
        for ds_name in sorted(ds_local.keys()):
            local_indices = ds_local[ds_name]
            col_names = dataset_col_names.get(ds_name, [])
            X_ds = dataset_X.get(ds_name)
            y_ds = dataset_y.get(ds_name)

            for li in sorted(local_indices):
                s = {"dataset": ds_name}
                # Find global index for activation value
                start, _ = ds_offsets[ds_name]
                gi = start + li
                s["activation"] = float(activations[gi, feat_idx])
                if y_ds is not None and li < len(y_ds):
                    s["y"] = float(y_ds[li])
                if X_ds is not None and li < len(X_ds):
                    s["columns"] = {
                        col_names[j]: float(X_ds[li, j])
                        for j in range(min(len(col_names), X_ds.shape[1]))
                    }
                samples.append(s)

        band = feature_to_band(feat_idx, bands) if bands else "unknown"
        old_label = existing_labels.get(feat_idx, "none")

        results.append({
            "feature_idx": feat_idx,
            "band": band,
            "old_label": old_label,
            "label": label,
            "n_active_rows": len(active_global),
            "n_datasets": n_datasets,
            "datasets": datasets,
            "distinctive_columns": [
                {"column": c, "mean_z": round(z, 2), "direction": d}
                for c, z, d in all_distinctive
            ],
            "target": target_info,
            "per_dataset": ds_descriptions,
            "samples": samples,
        })

    # LLM labeling pass
    if use_llm:
        client = _init_llm_client()
        if client:
            print(f"\nLLM labeling {len(results)} features...")
            n_labeled = 0
            for i, feat in enumerate(results):
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(results)} ({n_labeled} labeled)...")
                prompt = format_llm_prompt(feat)
                llm_label = _call_llm(prompt, client)
                if llm_label:
                    feat["rule_label"] = feat["label"]
                    feat["label"] = llm_label
                    feat["label_method"] = "llm"
                    n_labeled += 1
                else:
                    feat["label_method"] = "rule"
            print(f"  LLM labeled {n_labeled}/{len(results)} features")

    return {
        "model": model_key,
        "threshold": threshold,
        "n_selective": len(results),
        "n_alive": len(alive_features),
        "features": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Label ultra-selective SAE features from their activating rows",
    )
    parser.add_argument("--model", required=True, help="Model key (e.g. tabdpt)")
    parser.add_argument(
        "--threshold", type=int, default=20,
        help="Max active rows to count as ultra-selective (default: 20)",
    )
    parser.add_argument(
        "--top-cols", type=int, default=3,
        help="Number of distinctive columns to report (default: 3)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--llm", action="store_true",
        help="Use Claude Haiku for real-world interpretation labels",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    t0 = time.time()
    result = label_selective_features(
        args.model,
        threshold=args.threshold,
        top_cols=args.top_cols,
        device=args.device,
        use_llm=args.llm,
    )

    # Save
    out_dir = output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selective_feature_labels.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\nSaved {result['n_selective']} labels to {out_path}")
    print(f"Done in {elapsed:.0f}s")

    # Print a few examples
    print("\n=== Sample labels ===")
    for feat in result["features"][:10]:
        print(
            f"  Feature {feat['feature_idx']:4d} "
            f"({feat['n_active_rows']:2d} rows, {feat['n_datasets']} ds): "
            f"{feat['label']}"
        )
        if feat["old_label"] != "none":
            print(f"    old: {feat['old_label']}")


if __name__ == "__main__":
    main()
