#!/usr/bin/env python3
"""
Generate a complete concept dictionary for all SAE features.

Combines LLM-generated labels (from Claude Code agents) with rule-based
fallback labels from effect size profiles. Outputs a structured JSON
dictionary suitable for paper figures/tables.

Usage:
    # Collect labels from completed agent output files, then generate dictionary
    python scripts/generate_concept_dictionary.py --collect-agents --generate

    # Just generate dictionary from existing merged labels
    python scripts/generate_concept_dictionary.py --generate

    # Just collect agent outputs into merged.json
    python scripts/generate_concept_dictionary.py --collect-agents
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

from scripts.sae.compare_sae_cross_model import SAE_FILENAME, sae_sweep_dir


# Concise label rules: maps (meta_feature, direction) to short descriptors
LABEL_VOCAB = {
    # Sparsity / density
    ("frac_zeros", "+"): "sparse",
    ("frac_zeros", "-"): "dense",
    # Sign patterns
    ("frac_negative", "+"): "negative-heavy",
    ("frac_negative", "-"): "positive-heavy",
    # Distribution shape
    ("numeric_skewness", "+"): "right-skewed",
    ("numeric_skewness", "-"): "symmetric",
    ("numeric_kurtosis", "+"): "heavy-tailed",
    ("numeric_kurtosis", "-"): "light-tailed",
    # Outlier patterns
    ("numeric_max_zscore", "+"): "extreme-values",
    ("numeric_max_zscore", "-"): "bounded",
    ("numeric_mean_zscore", "+"): "outlier-rich",
    ("numeric_mean_zscore", "-"): "typical-values",
    ("numeric_min_zscore", "+"): "all-deviant",
    ("numeric_min_zscore", "-"): "has-typical",
    ("frac_positive_outliers", "+"): "positive-outliers",
    ("frac_negative_outliers", "+"): "negative-outliers",
    # Spread
    ("numeric_range", "+"): "wide-range",
    ("numeric_range", "-"): "narrow-range",
    ("numeric_iqr", "+"): "spread-out",
    ("numeric_iqr", "-"): "concentrated",
    ("numeric_std", "+"): "high-variance",
    ("numeric_std", "-"): "low-variance",
    # Geometry / position
    ("nearest_neighbor_dist", "+"): "isolated",
    ("nearest_neighbor_dist", "-"): "clustered",
    ("centroid_distance", "+"): "peripheral",
    ("centroid_distance", "-"): "central",
    ("local_density", "+"): "dense-region",
    ("local_density", "-"): "sparse-region",
    # Structure
    ("pca_residual", "+"): "nonlinear",
    ("pca_residual", "-"): "linear",
    ("numeric_correlation_mean", "+"): "correlated",
    ("numeric_correlation_mean", "-"): "uncorrelated",
    # Complexity
    ("row_entropy", "+"): "complex",
    ("row_entropy", "-"): "simple",
    ("row_uniformity", "+"): "uniform",
    ("row_uniformity", "-"): "varied",
    ("n_distinct_values", "+"): "diverse-values",
    ("n_distinct_values", "-"): "repetitive",
    # Dataset properties
    ("n_numeric", "+"): "high-dimensional",
    ("n_numeric", "-"): "low-dimensional",
    ("n_cols_total", "+"): "wide-table",
    ("n_cols_total", "-"): "narrow-table",
    ("n_rows_total", "+"): "large-dataset",
    ("n_rows_total", "-"): "small-dataset",
    # PCA position
    ("pca_pc1", "+"): "high-PC1",
    ("pca_pc1", "-"): "low-PC1",
    ("pca_pc2", "+"): "high-PC2",
    ("pca_pc2", "-"): "low-PC2",
    # Target
    ("target_is_minority", "+"): "minority-class",
    ("target_zscore", "+"): "high-target",
    ("target_zscore", "-"): "low-target",
    # Missing / categorical (likely zero for TabPFN but include for completeness)
    ("missing_rate", "+"): "missing-heavy",
    ("dataset_sparsity", "+"): "sparse-dataset",
}


def label_feature(effects: dict) -> dict:
    """Generate a concise rule-based label from a feature's effect size profile."""
    sorted_e = sorted(effects.items(), key=lambda x: -abs(x[1]))

    # Get significant effects (|d| > 0.3)
    significant = [(name, d) for name, d in sorted_e if abs(d) > 0.3]

    if not significant:
        if sorted_e:
            name, d = sorted_e[0]
            direction = "+" if d > 0 else "-"
            word = LABEL_VOCAB.get((name, direction), name.replace("_", "-"))
            return {
                "label": f"weak {word}",
                "components": [(name, round(d, 2))],
                "confidence": 0.2,
                "max_d": round(abs(d), 2),
            }
        return {"label": "uncharacterized", "components": [], "confidence": 0.0, "max_d": 0.0}

    # Build label from top 1-2 effects
    parts = []
    for name, d in significant[:2]:
        direction = "+" if d > 0 else "-"
        word = LABEL_VOCAB.get((name, direction))
        if word is None:
            word = ("high-" if d > 0 else "low-") + name.replace("_", "-")[:12]
        parts.append(word)

    max_d = abs(significant[0][1])
    if max_d > 2.0:
        label = f"very {parts[0]}"
    else:
        label = parts[0]

    if len(parts) > 1 and abs(significant[1][1]) > 0.5:
        label = f"{label}, {parts[1]}"

    n_strong = sum(1 for _, d in significant if abs(d) > 1.0)
    if max_d > 2.0:
        confidence = 0.95
    elif max_d > 1.0:
        confidence = 0.8
    elif max_d > 0.5:
        confidence = 0.6
    else:
        confidence = 0.4

    return {
        "label": label,
        "components": [(n, round(d, 2)) for n, d in significant[:5]],
        "confidence": round(confidence, 2),
        "max_d": round(max_d, 2),
    }


def extract_json_from_agent_output(output_path: Path) -> dict:
    """Extract JSON labels from a Claude Code agent JSONL output file."""
    labels = {}
    try:
        text = output_path.read_text()
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Look for assistant messages with text content containing JSON
            if msg.get("type") != "assistant":
                continue
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            for block in content:
                if block.get("type") != "text":
                    continue
                text_content = block.get("text", "")

                # Extract JSON from markdown code blocks
                json_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text_content, re.DOTALL)
                if json_match:
                    try:
                        batch_labels = json.loads(json_match.group(1))
                        labels.update(batch_labels)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"  Warning: failed to parse {output_path.name}: {e}")
    return labels


def collect_agent_outputs(tasks_dir: Path, merged_path: Path) -> dict:
    """Collect labels from all agent output files and merge with existing."""
    # Load existing merged labels
    existing = {}
    if merged_path.exists():
        with open(merged_path) as f:
            existing = json.load(f)

    # Find and parse all agent output files
    output_files = sorted(tasks_dir.glob("*.output"))
    new_count = 0
    for output_file in output_files:
        batch_labels = extract_json_from_agent_output(output_file)
        for fid, label in batch_labels.items():
            if str(fid) not in existing:
                new_count += 1
            existing[str(fid)] = label

    # Save merged
    with open(merged_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Collected from {len(output_files)} agent files: {len(existing)} total labels ({new_count} new)")
    return existing


def matryoshka_level(fid: int, matryoshka_dims: list, hidden_dim: int) -> int:
    """Return the Matryoshka nesting level for a feature index.

    Level 0 = features in the innermost (most global) nested set,
    higher levels = progressively finer detail added at each scale.
    """
    boundaries = [0] + sorted(matryoshka_dims) + [hidden_dim]
    for level, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        if lo <= fid < hi:
            return level
    return len(matryoshka_dims)  # beyond all defined levels


def generate_dictionary(analysis_path: Path, llm_labels: dict, out_path: Path,
                        sae_checkpoint: Path = None):
    """Generate the complete concept dictionary combining LLM + rule-based labels."""
    with open(analysis_path) as f:
        data = json.load(f)

    features = data["features"]
    clusters = data.get("clustering", {}).get("clusters", {})
    f2c = data.get("clustering", {}).get("feature_to_cluster", {})

    # Load Matryoshka config from SAE checkpoint
    mat_dims = [32, 64, 128, 256]  # default
    hidden_dim = 1536
    if sae_checkpoint:
        ckpt_path = Path(sae_checkpoint)
        if ckpt_path.exists():
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            mat_dims = cfg.get("matryoshka_dims", mat_dims)
            hidden_dim = cfg.get("hidden_dim", hidden_dim)
            print(f"Loaded Matryoshka dims from checkpoint: {mat_dims}, hidden={hidden_dim}")

    # Label every feature
    dictionary = {}
    llm_count = 0
    rule_count = 0

    for fid, info in features.items():
        rule_info = label_feature(info["effect_sizes"])
        cluster = f2c.get(str(fid), f2c.get(fid, -1))

        # Prefer LLM label when available
        llm_label = llm_labels.get(str(fid))
        if llm_label:
            label = llm_label
            label_source = "llm"
            llm_count += 1
        else:
            label = rule_info["label"]
            label_source = "rule"
            rule_count += 1

        dictionary[int(fid)] = {
            "label": label,
            "label_source": label_source,
            "rule_label": rule_info["label"],
            "cluster": cluster,
            "matryoshka_level": matryoshka_level(int(fid), mat_dims, hidden_dim),
            "confidence": rule_info["confidence"],
            "max_d": rule_info["max_d"],
            "components": rule_info["components"],
            "top_datasets": info.get("top_datasets", [])[:3],
            "mean_activation": round(info.get("mean_activation", 0), 4),
            "max_activation": round(info.get("max_activation", 0), 4),
        }

    # Label clusters using LLM labels
    cluster_labels = {}
    for cid, cinfo in clusters.items():
        cluster_features = [dictionary[int(fid)] for fid in cinfo["feature_ids"]
                           if int(fid) in dictionary]
        if cluster_features:
            primary_components = []
            for f in cluster_features:
                if f["components"]:
                    primary_components.append(f["components"][0][0])
            if primary_components:
                most_common = Counter(primary_components).most_common(3)
                theme_parts = []
                for comp_name, count in most_common:
                    pct = 100 * count / len(cluster_features)
                    theme_parts.append(f"{comp_name} ({pct:.0f}%)")
                theme = ", ".join(theme_parts)
            else:
                theme = "mixed"
        else:
            theme = "empty"

        cluster_labels[int(cid)] = {
            "n_features": cinfo["n_features"],
            "theme": theme,
            "interpretation": cinfo.get("interpretation", ""),
        }

    # Sort clusters by size
    sorted_clusters = sorted(cluster_labels.items(), key=lambda x: -x[1]["n_features"])

    # Print complete dictionary
    print("=" * 90)
    print(f"TABPFN SAE CONCEPT DICTIONARY: {len(dictionary)} features, {len(cluster_labels)} clusters")
    print(f"  LLM labels: {llm_count}, Rule-based: {rule_count}")
    print("=" * 90)

    for cid, cinfo in sorted_clusters:
        n = cinfo["n_features"]
        theme = cinfo["theme"][:65]
        print(f"\n{'─' * 90}")
        print(f"CLUSTER {cid} ({n} features) — {theme}")
        print(f"{'─' * 90}")

        cluster_feats = [(fid, info) for fid, info in dictionary.items() if info["cluster"] == cid]
        cluster_feats.sort(key=lambda x: -x[1]["max_d"])

        for fid, info in cluster_feats:
            label = info["label"]
            src = "L" if info["label_source"] == "llm" else "R"
            max_d = info["max_d"]
            act = info["max_activation"]
            ds = [d[0][:12] for d in info["top_datasets"]]
            ds_str = ", ".join(ds) if ds else ""

            if info["confidence"] >= 0.8:
                conf_mark = "■"
            elif info["confidence"] >= 0.6:
                conf_mark = "◧"
            else:
                conf_mark = "□"

            print(f"  {conf_mark}{src} F{fid:4d} │ {label:35s} │ d={max_d:4.2f} │ act={act:.3f} │ {ds_str}")

    # Statistics
    print(f"\n{'=' * 90}")
    print("STATISTICS")
    print(f"{'=' * 90}")

    confidences = [d["confidence"] for d in dictionary.values()]
    max_ds = [d["max_d"] for d in dictionary.values()]

    print(f"  Total features: {len(dictionary)}")
    print(f"  LLM-labeled: {llm_count} ({100*llm_count/len(dictionary):.0f}%)")
    print(f"  Rule-labeled: {rule_count} ({100*rule_count/len(dictionary):.0f}%)")
    print(f"  High confidence (>=0.8): {sum(1 for c in confidences if c >= 0.8)}")
    print(f"  Medium confidence (0.4-0.8): {sum(1 for c in confidences if 0.4 <= c < 0.8)}")
    print(f"  Low confidence (<0.4): {sum(1 for c in confidences if c < 0.4)}")
    print(f"  Strong effect (d>1.0): {sum(1 for d in max_ds if d > 1.0)}")
    print(f"  Moderate effect (0.5-1.0): {sum(1 for d in max_ds if 0.5 <= d < 1.0)}")
    print(f"  Weak effect (d<0.5): {sum(1 for d in max_ds if d < 0.5)}")

    # Label diversity
    label_counts = Counter(d["label"] for d in dictionary.values())
    print(f"\n  Unique labels: {len(label_counts)}")
    print(f"  Most common labels:")
    for label, count in label_counts.most_common(10):
        print(f"    {label:35s} x {count}")

    # Matryoshka level summary
    boundaries = [0] + sorted(mat_dims) + [hidden_dim]
    level_summary = {}
    for level in range(len(boundaries) - 1):
        lo, hi = boundaries[level], boundaries[level + 1]
        level_feats = [v for fid, v in dictionary.items() if lo <= fid < hi]
        level_summary[level] = {
            "range": [lo, hi],
            "n_total": hi - lo,
            "n_alive": len(level_feats),
            "mean_activation": round(sum(f["mean_activation"] for f in level_feats) / max(len(level_feats), 1), 4),
            "mean_max_d": round(sum(f["max_d"] for f in level_feats) / max(len(level_feats), 1), 2),
        }

    # Save
    output = {
        "n_features": len(dictionary),
        "n_clusters": len(cluster_labels),
        "n_llm_labeled": llm_count,
        "n_rule_labeled": rule_count,
        "matryoshka_dims": mat_dims,
        "matryoshka_levels": {str(k): v for k, v in level_summary.items()},
        "features": {str(k): v for k, v in dictionary.items()},
        "clusters": {str(k): v for k, v in cluster_labels.items()},
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    return dictionary


def main():
    parser = argparse.ArgumentParser(description="Generate SAE concept dictionary")
    parser.add_argument("--collect-agents", action="store_true",
                        help="Collect labels from agent output files")
    parser.add_argument("--generate", action="store_true",
                        help="Generate the complete dictionary")
    parser.add_argument("--tasks-dir", type=Path,
                        default=Path("/private/tmp/claude-501/-Volumes-Samsung2TB-src-tabular-embeddings/tasks"),
                        help="Directory containing agent output files")
    parser.add_argument("--sae-checkpoint", type=Path,
                        default=sae_sweep_dir() / "tabpfn" / SAE_FILENAME,
                        help="SAE checkpoint for Matryoshka level metadata")
    args = parser.parse_args()

    if not args.collect_agents and not args.generate:
        args.generate = True  # Default to generate

    merged_path = Path("output/concept_labels/llm_results/merged.json")
    analysis_path = Path("output/concept_labels_full.json")
    out_path = Path("output/concept_dictionary_complete.json")

    llm_labels = {}

    if args.collect_agents:
        llm_labels = collect_agent_outputs(args.tasks_dir, merged_path)
    elif merged_path.exists():
        with open(merged_path) as f:
            llm_labels = json.load(f)
        print(f"Loaded {len(llm_labels)} LLM labels from {merged_path}")

    if args.generate:
        if not analysis_path.exists():
            print(f"Error: {analysis_path} not found. Run analyze_sae_concepts_deep.py first.")
            return
        generate_dictionary(analysis_path, llm_labels, out_path,
                            sae_checkpoint=args.sae_checkpoint)


if __name__ == "__main__":
    main()
