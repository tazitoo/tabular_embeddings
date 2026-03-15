#!/usr/bin/env python3
"""
Cross-model SAE concept grouping and prompt generation.

Groups features identified by cross-model matching into concept groups, then
generates labeling prompts with contrastive examples and dataset context.
Labeling is done externally in Claude Code, not via API.

Runs in three phases:
  Phase 1: MNN-matched groups (union-find on tier-1 edges)
  Phase 2: Extend groups via cross-correlation direct assignment
  Phase 3: Cluster unmatched features by probe signature

Usage:
    # Run all phases (default)
    python scripts/label_cross_model_concepts.py

    # Single phase
    python scripts/label_cross_model_concepts.py --phase 1
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_sae_concepts_deep import (
    NumpyEncoder,
    convert_keys_to_native,
)
from scripts.compare_sae_cross_model import DEFAULT_SAE_ROUND


# ── Data loading ──────────────────────────────────────────────────────────


def load_concept_data(
    concepts: dict, top_k: int = 5
) -> Tuple[Dict[str, Dict[int, dict]], dict]:
    """Load probe lookup AND contrastive examples from sae_concept_analysis.

    Returns:
        probe_lookup: model -> {feat_idx: {r2, probes, examples}}
        dataset_context: dataset_name -> {pymfe_features}
    """
    lookup = {}
    for model_name, model_data in concepts["models"].items():
        per_feature = model_data.get("per_feature", {})
        features = {}
        for feat_idx_str, feat_data in per_feature.items():
            feat_idx = int(feat_idx_str)
            probes = feat_data.get("top_probes", [])[:top_k]
            features[feat_idx] = {
                "r2": feat_data["r2"],
                "probes": [(p[0], float(p[1])) for p in probes],
                "probe_names": {p[0] for p in probes},
                "examples": feat_data.get("examples", {}),
            }
        lookup[model_name] = features

    dataset_context = concepts.get("dataset_context", {})
    return lookup, dataset_context

# ── Union-Find ────────────────────────────────────────────────────────────


class UnionFind:
    """Weighted union-find with path compression for (model, feat_idx) nodes."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: Tuple[str, int]) -> Tuple[str, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Tuple[str, int], b: Tuple[str, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
        """Return {root: [members]} for all components with 2+ members."""
        groups = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return {r: sorted(members) for r, members in groups.items() if len(members) >= 2}


# ── Graph building ────────────────────────────────────────────────────────


def build_match_graph(
    matching: dict,
    tier: int,
    min_r: float = 0.0,
    uf: Optional[UnionFind] = None,
) -> UnionFind:
    """Add edges of given tier with |r| >= min_r to union-find."""
    if uf is None:
        uf = UnionFind()

    n_edges = 0
    for pair_key, pair_data in matching["pairs"].items():
        model_a = pair_data.get("model_a", pair_key.split("__")[0])
        model_b = pair_data.get("model_b", pair_key.split("__")[1])

        for m in pair_data["matches"]:
            if m.get("tier", 1) != tier:
                continue
            if abs(m["r"]) < min_r:
                continue
            uf.union((model_a, m["idx_a"]), (model_b, m["idx_b"]))
            n_edges += 1

    print(f"  Added {n_edges} tier-{tier} edges (min_r={min_r:.2f})")
    return uf


# ── Probe aggregation ─────────────────────────────────────────────────────


def aggregate_group_probes(
    members: List[Tuple[str, int]],
    probe_lookup: Dict[str, Dict[int, dict]],
) -> dict:
    """Aggregate probe profiles across all features in a concept group."""
    models = set()
    r2_vals = []
    probe_votes = defaultdict(lambda: {"count": 0, "coeffs": [], "models": set()})
    per_member = []

    for model, feat_idx in members:
        models.add(model)
        feat_data = probe_lookup.get(model, {}).get(feat_idx)
        if feat_data is None:
            per_member.append({
                "model": model, "feature_idx": feat_idx,
                "r2": 0.0, "top_probes": [], "examples": {},
            })
            continue

        r2_vals.append(feat_data["r2"])
        probes = feat_data.get("probes", [])
        per_member.append({
            "model": model, "feature_idx": feat_idx,
            "r2": feat_data["r2"],
            "top_probes": [(p[0], round(p[1], 3)) for p in probes[:5]],
            "examples": feat_data.get("examples", {}),
        })
        for name, coeff in probes[:5]:
            probe_votes[name]["count"] += 1
            probe_votes[name]["coeffs"].append(coeff)
            probe_votes[name]["models"].add(model)

    top_probes = sorted(
        probe_votes.items(),
        key=lambda x: (-x[1]["count"], -abs(sum(x[1]["coeffs"]) / len(x[1]["coeffs"]))),
    )
    top_probes_list = [
        (name, data["count"], round(sum(data["coeffs"]) / len(data["coeffs"]), 3))
        for name, data in top_probes[:10]
    ]

    return {
        "n_models": len(models),
        "models": sorted(models),
        "mean_r2": round(sum(r2_vals) / len(r2_vals), 4) if r2_vals else 0.0,
        "probe_votes": {
            name: {"count": d["count"], "mean_coeff": round(sum(d["coeffs"]) / len(d["coeffs"]), 3)}
            for name, d in top_probes
        },
        "top_probes": top_probes_list,
        "per_member": per_member,
    }


# ── Prompt generation ─────────────────────────────────────────────────────


SYSTEM_PROMPT = """\
You are an expert at analyzing tabular data patterns. You are labeling \
universal concepts found by Sparse Autoencoders trained on different tabular \
foundation models. A "concept group" means features from multiple independent \
models that activate on the same data rows, indicating they detect the same \
underlying tabular pattern.

You will see:
- Probe regression coefficients (statistical summaries of row-level properties)
- Contrastive examples: raw data rows where the feature fires vs nearby rows \
where it doesn't. Look for patterns in the actual column values.
- Dataset context: PyMFE meta-features describing each dataset.

Respond with ONLY a concept label (2-5 words). Focus on data properties, \
not semantics. Examples: "extreme outliers", "high feature correlation", \
"sparse rows", "right-skewed distribution", "isolated points"."""


def _format_raw_row(raw: dict, max_cols: int = 12) -> str:
    """Format a raw data row compactly."""
    items = list(raw.items())[:max_cols]
    parts = []
    for col, val in items:
        if isinstance(val, float):
            parts.append(f"col{col}={val:.3g}")
        else:
            parts.append(f"col{col}={val}")
    return "  " + ", ".join(parts)


def _format_examples(examples: dict, n: int = 5) -> List[str]:
    """Format contrastive examples (top-activating vs nearest inactive)."""
    lines = []
    top = examples.get("top", [])[:n]
    contrast = examples.get("contrast", [])[:n]

    if top:
        lines.append("TOP-ACTIVATING ROWS (feature fires strongly):")
        for ex in top:
            ds = ex.get("dataset", "?")
            act = ex.get("activation", 0)
            lines.append(f"  [{ds}] activation={act:.1f}")
            if "raw" in ex:
                lines.append(_format_raw_row(ex["raw"]))
        lines.append("")

    if contrast:
        lines.append("NEAREST NON-ACTIVATING ROWS (similar data, feature silent):")
        for ex in contrast:
            ds = ex.get("dataset", "?")
            act = ex.get("activation", 0)
            lines.append(f"  [{ds}] activation={act:.1f}")
            if "raw" in ex:
                lines.append(_format_raw_row(ex["raw"]))
        lines.append("")

    return lines


def _format_dataset_context(
    examples: dict, dataset_context: dict, max_datasets: int = 3
) -> List[str]:
    """Add PyMFE dataset descriptions for datasets appearing in examples."""
    lines = []
    datasets_seen = set()
    for ex in examples.get("top", []) + examples.get("contrast", []):
        ds = ex.get("dataset")
        if ds and ds not in datasets_seen:
            datasets_seen.add(ds)

    if not datasets_seen or not dataset_context:
        return lines

    lines.append("DATASET CONTEXT:")
    for ds in sorted(datasets_seen)[:max_datasets]:
        ctx = dataset_context.get(ds, {})
        if not ctx:
            continue
        parts = []
        for key in ["nr_inst", "nr_attr", "nr_class", "nr_num", "nr_cat",
                     "inst_to_attr", "cat_to_num"]:
            if key in ctx:
                parts.append(f"{key}={ctx[key]:.2g}" if isinstance(ctx[key], float) else f"{key}={ctx[key]}")
        if parts:
            lines.append(f"  {ds}: {', '.join(parts)}")
    if lines:
        lines.append("")
    return lines


def format_group_prompt(
    group_id: int, agg: dict, dataset_context: dict = None,
    domain_lookup: dict = None,
) -> str:
    """Format prompt for a matched concept group.

    Structure: contrastive examples first (primary evidence), then probe
    statistics as secondary guidance, then monosemantic framing.
    """
    n_members = len(agg['per_member'])
    lines = [
        f"=== Concept Group {group_id} ===",
        f"Models: {agg['n_models']}/8 ({', '.join(agg['models'])})",
        f"Members: {n_members} features",
        f"Mean R² (probe-explained): {agg['mean_r2']:.3f}",
        "",
    ]

    # Primary evidence: contrastive examples from the highest-R² member
    best_member = max(agg["per_member"], key=lambda m: m.get("r2", 0))
    examples = best_member.get("examples", {})
    if examples:
        lines.append(f"CONTRASTIVE EXAMPLES (from {best_member['model']} #{best_member['feature_idx']}):")
        lines.extend(_format_examples(examples))
        if dataset_context:
            lines.extend(_format_dataset_context(examples, dataset_context))

    # Secondary guidance: probe statistics
    consensus = [(n, d["count"], d["mean_coeff"])
                 for n, d in agg["probe_votes"].items() if d["count"] >= 2]
    consensus.sort(key=lambda x: (-x[1], -abs(x[2])))
    if consensus:
        lines.append("STATISTICAL GUIDANCE (probe correlations — explain ~20% of variance):")
        lines.append("These are partial hints, not the full story. Use them to check your")
        lines.append("interpretation of the contrastive examples above, not as primary evidence.")
        for name, count, mc in consensus[:8]:
            sign = "+" if mc > 0 else "-"
            lines.append(f"  {name}: {count}/{n_members} members, coeff={sign}{abs(mc):.3f}")
        lines.append("")

    # Per-member detail
    lines.append("PER-MEMBER DETAIL:")
    for m in agg["per_member"][:10]:
        probes_str = ", ".join(f"{p[0]}({p[1]:+.2f})" for p in m["top_probes"][:4])
        lines.append(f"  {m['model']} #{m['feature_idx']} (R²={m['r2']:.2f}): {probes_str}")
    lines.append("")

    # Monosemantic framing
    lines.append(
        "The rows above come from a Sparse Autoencoder (SAE) feature — a single "
        "learned concept that should be MONOSEMANTIC (encoding exactly one coherent "
        "meaning). The \"top-activating\" rows are where this concept fires most "
        "strongly; the \"non-activating\" rows are nearby data points where it "
        "stays silent."
    )
    lines.append("")
    lines.append(
        f"This concept was independently discovered by {agg['n_models']} different "
        "tabular foundation models, confirming it captures something real and "
        "universal — not a model artifact."
    )
    lines.append("")

    # Domain context
    if domain_lookup and examples:
        ds_names = set()
        for ex in examples.get("top", []) + examples.get("contrast", []):
            ds = ex.get("dataset")
            if ds:
                ds_names.add(ds)
        if ds_names:
            domain_parts = [f"{ds} ({domain_lookup.get(ds, 'Unknown')})"
                            for ds in sorted(ds_names)]
            lines.append(f"The contrastive examples come from: {', '.join(domain_parts)}")
            lines.append("")

    lines.append(
        "In 1-2 sentences, describe the single coherent monosemantic meaning this "
        "SAE concept encodes. What is the one specific data pattern that causes it "
        "to fire? Focus on what makes activating rows concretely different from "
        "their non-activating neighbors."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: The concept is UNIVERSAL — it fires across multiple models and "
        "domains. Describe the abstract structural pattern, not the domain-specific "
        "interpretation. Never cite column names (col0, col3, etc.) or probe names "
        "(numeric_std, frac_zeros, etc.) in your description — the output should "
        "read as a natural, technically oriented sentence that a data scientist "
        "would understand without seeing the raw data."
    )
    return "\n".join(lines)


def format_individual_prompt(
    model: str, feat_idx: int, feat_data: dict, dataset_context: dict = None
) -> str:
    """Format prompt for an unmatched feature."""
    lines = [
        f"=== Unmatched Feature: {model} #{feat_idx} ===",
        f"R² (probe-explained): {feat_data.get('r2', 0):.3f}",
        "",
        "Top probes:",
    ]
    for name, coeff in feat_data.get("probes", [])[:5]:
        lines.append(f"  {name}: coeff={coeff:+.3f}")
    lines.append("")

    examples = feat_data.get("examples", {})
    if examples:
        lines.extend(_format_examples(examples))
        if dataset_context:
            lines.extend(_format_dataset_context(examples, dataset_context))

    lines.append("This feature was not matched in other models. What tabular pattern does it detect? (2-5 words only)")
    return "\n".join(lines)


def format_signature_prompt(
    signature: str, count: int, example_probes: list,
    rep_examples: dict = None, dataset_context: dict = None
) -> str:
    """Format prompt for a probe-signature cluster of unmatched features."""
    lines = [
        f"=== Unmatched Feature Cluster ({count} features) ===",
        f"Probe signature: {signature}",
        "",
        "These features all share the same top probes:",
    ]
    for name, coeff in example_probes[:5]:
        lines.append(f"  {name}: coeff={coeff:+.3f}")
    lines.append("")

    if rep_examples:
        lines.extend(_format_examples(rep_examples))
        if dataset_context:
            lines.extend(_format_dataset_context(rep_examples, dataset_context))

    lines.append("What tabular pattern do these features detect? (2-5 words only)")
    return "\n".join(lines)


# ── Phase execution ───────────────────────────────────────────────────────


def run_phase1(
    matching: dict,
    probe_lookup: Dict[str, Dict[int, dict]],
    min_r: float,
    max_group_size: int,
    dataset_context: dict = None,
    domain_lookup: dict = None,
) -> dict:
    """Phase 1: Build tier-1 (MNN) graph, find components, generate prompts."""
    print("\n── Phase 1: MNN concept groups ──")
    uf = build_match_graph(matching, tier=1, min_r=min_r)
    components = uf.components()
    print(f"  Found {len(components)} groups (2+ members)")

    concept_groups = {}
    n_skip = 0

    for i, (root, members) in enumerate(sorted(components.items(), key=lambda x: -len(x[1]))):
        agg = aggregate_group_probes(members, probe_lookup)

        if len(members) > max_group_size:
            concept_groups[str(i)] = {
                "members": [[m, f] for m, f in members],
                "n_models": agg["n_models"],
                "tier": 1,
                "mean_r2": agg["mean_r2"],
                "label": "unlabeled",
                "top_probes": agg["top_probes"][:5],
                "phase_added": 1,
                "prompt": None,  # Too large for single prompt
            }
            n_skip += 1
            continue

        prompt = format_group_prompt(i, agg, dataset_context=dataset_context,
                                            domain_lookup=domain_lookup)

        concept_groups[str(i)] = {
            "members": [[m, f] for m, f in members],
            "n_models": agg["n_models"],
            "tier": 1,
            "mean_r2": agg["mean_r2"],
            "label": "unlabeled",
            "top_probes": agg["top_probes"][:5],
            "phase_added": 1,
            "prompt": prompt,
        }

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(components)} groups...")

    print(f"  Generated prompts for {len(components) - n_skip} groups, {n_skip} too large")
    return {
        "metadata": {"phase": 1},
        "concept_groups": concept_groups,
        "_uf_edges": _serialize_uf(uf),
    }


def _load_cross_correlations(
    corr_dir: Path,
) -> Dict[str, dict]:
    """Load pre-computed cross-correlation matrices for all model pairs."""
    import numpy as _np

    corr_data = {}
    for npz_path in sorted(corr_dir.glob("*.npz")):
        d = _np.load(npz_path, allow_pickle=True)
        pair_key = npz_path.stem
        corr_data[pair_key] = {
            "corr_matrix": d["corr_matrix"],
            "indices_a": d["indices_a"],
            "indices_b": d["indices_b"],
            "model_a": str(d["model_a"]),
            "model_b": str(d["model_b"]),
        }
    return corr_data


def _build_corr_lookup(
    corr_data: Dict[str, dict],
) -> Dict[Tuple[str, str], dict]:
    """Build bidirectional lookup: (model_x, model_y) -> correlation info."""
    lookup = {}
    for pair_key, info in corr_data.items():
        ma, mb = info["model_a"], info["model_b"]
        lookup[(ma, mb)] = info
        lookup[(mb, ma)] = {
            "corr_matrix": info["corr_matrix"].T,
            "indices_a": info["indices_b"],
            "indices_b": info["indices_a"],
            "model_a": mb,
            "model_b": ma,
        }
    return lookup


def run_phase2(
    prev_output: dict,
    matching: dict,
    probe_lookup: Dict[str, Dict[int, dict]],
    min_r: float,
    max_group_size: int,
    relabel_threshold: float,
    corr_dir: Optional[Path] = None,
    dataset_context: dict = None,
    domain_lookup: dict = None,
) -> dict:
    """Phase 2: Extend groups using cross-correlation direct assignment."""
    print("\n── Phase 2: Extending groups via cross-correlation ──")

    if corr_dir is None:
        corr_dir = Path(__file__).parent.parent / "output" / "sae_cross_correlations"

    if not corr_dir.exists():
        print(f"  ERROR: No cross-correlation directory at {corr_dir}")
        print(f"  Run: python scripts/match_sae_features.py --save-correlations")
        return {
            "metadata": {"phase": 2},
            "concept_groups": dict(prev_output["concept_groups"]),
        }

    corr_data = _load_cross_correlations(corr_dir)
    if not corr_data:
        print(f"  ERROR: No .npz files in {corr_dir}")
        return {
            "metadata": {"phase": 2},
            "concept_groups": dict(prev_output["concept_groups"]),
        }

    corr_lookup = _build_corr_lookup(corr_data)
    print(f"  Loaded {len(corr_data)} cross-correlation matrices")

    import numpy as _np

    abs_to_rel = {}
    for pair_key, info in corr_lookup.items():
        abs_to_rel[pair_key] = {
            int(idx): i for i, idx in enumerate(info["indices_a"])
        }

    concept_groups = dict(prev_output["concept_groups"])
    node_to_group = {}
    for gid, group in concept_groups.items():
        for model, feat_idx in group["members"]:
            node_to_group[(model, feat_idx)] = gid

    grouped_mask_cache = {}
    for pair_key, info in corr_lookup.items():
        other_model = info["model_b"]
        idx_b = info["indices_b"]
        mask = _np.zeros(len(idx_b), dtype=bool)
        group_ids = [None] * len(idx_b)
        for j, abs_idx in enumerate(idx_b):
            abs_j = int(abs_idx)
            gid = node_to_group.get((other_model, abs_j))
            if gid is not None:
                mask[j] = True
                group_ids[j] = gid
        grouped_mask_cache[pair_key] = (mask, group_ids)

    all_models = set()
    alive_per_model = {}
    for model, features in probe_lookup.items():
        all_models.add(model)
        alive_per_model[model] = set(int(k) for k in features.keys())

    unmatched = []
    for model in sorted(all_models):
        for feat_idx in sorted(alive_per_model.get(model, set())):
            if (model, feat_idx) not in node_to_group:
                unmatched.append((model, feat_idx))

    print(f"  {len(unmatched)} unmatched features across {len(all_models)} models")

    # --- Pass 1: Assign unmatched features to existing groups ---
    next_id = max((int(k) for k in concept_groups), default=-1) + 1
    n_extended = 0
    n_new = 0
    n_skip_size = 0
    n_skip_r = 0

    still_unmatched = []

    for model, feat_idx in unmatched:
        best_r = 0.0
        best_group = None

        for other_model in sorted(all_models):
            if other_model == model:
                continue

            pair = (model, other_model)
            if pair not in corr_lookup:
                continue

            pos_a = abs_to_rel.get(pair, {}).get(feat_idx)
            if pos_a is None:
                continue

            info = corr_lookup[pair]
            corr_row = info["corr_matrix"][pos_a]
            mask, group_ids = grouped_mask_cache[pair]

            grouped_corr = _np.where(mask, corr_row, 0.0)
            j_best = int(grouped_corr.argmax())
            r = float(grouped_corr[j_best])
            if r > best_r and group_ids[j_best] is not None:
                best_r = r
                best_group = group_ids[j_best]

        if best_r >= min_r and best_group is not None:
            current_size = len(concept_groups[best_group]["members"])
            if current_size >= max_group_size:
                n_skip_size += 1
                still_unmatched.append((model, feat_idx))
            else:
                concept_groups[best_group]["members"].append([model, feat_idx])
                node_to_group[(model, feat_idx)] = best_group
                n_extended += 1
        else:
            n_skip_r += 1
            still_unmatched.append((model, feat_idx))

    print(f"  Pass 1 (extend existing groups): {n_extended} assigned, "
          f"{n_skip_r} below threshold, {n_skip_size} hit size limit")

    # --- Pass 2: Pair remaining unmatched features into new groups ---
    still_unmatched_set = set(still_unmatched)
    assigned_pass2 = set()

    unmatched_mask_cache = {}
    for pair_key, info in corr_lookup.items():
        other_model = info["model_b"]
        idx_b = info["indices_b"]
        mask = _np.zeros(len(idx_b), dtype=bool)
        for j, abs_idx in enumerate(idx_b):
            if (other_model, int(abs_idx)) in still_unmatched_set:
                mask[j] = True
        unmatched_mask_cache[pair_key] = mask

    candidate_pairs = []
    for model, feat_idx in still_unmatched:
        for other_model in sorted(all_models):
            if other_model == model:
                continue
            pair = (model, other_model)
            if pair not in corr_lookup:
                continue

            pos_a = abs_to_rel.get(pair, {}).get(feat_idx)
            if pos_a is None:
                continue

            info = corr_lookup[pair]
            corr_row = info["corr_matrix"][pos_a]
            um_mask = unmatched_mask_cache[pair]
            idx_b = info["indices_b"]

            masked_corr = _np.where(um_mask, corr_row, 0.0)
            j_best = int(masked_corr.argmax())
            r = float(masked_corr[j_best])
            if r >= min_r:
                abs_j = int(idx_b[j_best])
                candidate_pairs.append((r, model, feat_idx, other_model, abs_j))

    candidate_pairs.sort(reverse=True)
    for r, m_a, f_a, m_b, f_b in candidate_pairs:
        if (m_a, f_a) in assigned_pass2 or (m_b, f_b) in assigned_pass2:
            continue
        if (m_a, f_a) in node_to_group or (m_b, f_b) in node_to_group:
            continue

        gid = str(next_id)
        next_id += 1

        members = [(m_a, f_a), (m_b, f_b)]
        agg = aggregate_group_probes(members, probe_lookup)
        prompt = format_group_prompt(int(gid), agg, dataset_context=dataset_context,
                                            domain_lookup=domain_lookup)

        concept_groups[gid] = {
            "members": [[m, f] for m, f in members],
            "n_models": agg["n_models"],
            "tier": 2,
            "mean_r2": agg["mean_r2"],
            "label": "unlabeled",
            "top_probes": agg["top_probes"][:5],
            "phase_added": 2,
            "match_r": round(r, 4),
            "prompt": prompt,
        }

        node_to_group[(m_a, f_a)] = gid
        node_to_group[(m_b, f_b)] = gid
        assigned_pass2.add((m_a, f_a))
        assigned_pass2.add((m_b, f_b))
        n_new += 1

    # --- Pass 3: Extend new tier-2 groups with more unmatched features ---
    remaining = [
        (m, f) for m, f in still_unmatched
        if (m, f) not in node_to_group
    ]
    n_extended_t2 = 0

    for pair_key, info in corr_lookup.items():
        other_model = info["model_b"]
        idx_b = info["indices_b"]
        mask = _np.zeros(len(idx_b), dtype=bool)
        group_ids = [None] * len(idx_b)
        for j, abs_idx in enumerate(idx_b):
            abs_j = int(abs_idx)
            gid = node_to_group.get((other_model, abs_j))
            if gid is not None:
                mask[j] = True
                group_ids[j] = gid
        grouped_mask_cache[pair_key] = (mask, group_ids)

    for model, feat_idx in remaining:
        best_r = 0.0
        best_group = None

        for other_model in sorted(all_models):
            if other_model == model:
                continue
            pair = (model, other_model)
            if pair not in corr_lookup:
                continue

            pos_a = abs_to_rel.get(pair, {}).get(feat_idx)
            if pos_a is None:
                continue

            info = corr_lookup[pair]
            corr_row = info["corr_matrix"][pos_a]
            mask, group_ids = grouped_mask_cache[pair]

            grouped_corr = _np.where(mask, corr_row, 0.0)
            j_best = int(grouped_corr.argmax())
            r = float(grouped_corr[j_best])
            if r > best_r and group_ids[j_best] is not None:
                best_r = r
                best_group = group_ids[j_best]

        if best_r >= min_r and best_group is not None:
            current_size = len(concept_groups[best_group]["members"])
            if current_size < max_group_size:
                concept_groups[best_group]["members"].append([model, feat_idx])
                node_to_group[(model, feat_idx)] = best_group
                n_extended_t2 += 1

    # Re-generate prompts for groups that grew significantly
    n_relabel = 0
    for gid, group in concept_groups.items():
        if group.get("phase_added") == 2:
            continue
        prev_members = prev_output["concept_groups"].get(gid, {}).get("members", [])
        if not prev_members:
            continue
        growth = (len(group["members"]) - len(prev_members)) / len(prev_members)
        if growth > relabel_threshold and len(group["members"]) <= max_group_size:
            agg = aggregate_group_probes(
                [(m, f) for m, f in group["members"]], probe_lookup
            )
            group["prompt"] = format_group_prompt(
                int(gid), agg, dataset_context=dataset_context,
                domain_lookup=domain_lookup
            )
            group["n_models"] = agg["n_models"]
            group["mean_r2"] = agg["mean_r2"]
            group["top_probes"] = agg["top_probes"][:5]
            group["phase_relabeled"] = 2
            n_relabel += 1

    total_grouped = sum(len(g["members"]) for g in concept_groups.values())
    max_size = max((len(g["members"]) for g in concept_groups.values()), default=0)
    print(f"  Pass 2 (new tier-2 pairs): {n_new} new groups")
    print(f"  Pass 3 (extend tier-2): {n_extended_t2} more assigned")
    print(f"  Re-prompted: {n_relabel}")
    print(f"  Total grouped: {total_grouped}, max group size: {max_size}")

    return {
        "metadata": {"phase": 2},
        "concept_groups": concept_groups,
    }


def run_phase3(
    prev_output: dict,
    probe_lookup: Dict[str, Dict[int, dict]],
    r2_threshold: float,
    dataset_context: dict = None,
) -> dict:
    """Phase 3: Cluster unmatched features by probe signature, generate prompts."""
    print("\n── Phase 3: Unmatched features ──")

    grouped = set()
    for gid, group in prev_output["concept_groups"].items():
        for model, feat_idx in group["members"]:
            grouped.add((model, feat_idx))

    explained = defaultdict(dict)
    unexplained = defaultdict(dict)
    sig_clusters = defaultdict(list)

    for model, features in probe_lookup.items():
        for feat_idx, feat_data in features.items():
            if (model, feat_idx) in grouped:
                continue
            if feat_data["r2"] >= r2_threshold:
                probes = feat_data.get("probes", [])[:2]
                sig = "__".join(p[0] for p in probes) if probes else "none"
                sig_clusters[sig].append((model, feat_idx, feat_data))
            else:
                unexplained[model][str(feat_idx)] = {
                    "r2": round(feat_data["r2"], 4),
                    "label": "unexplained",
                    "top_probes": [(p[0], round(p[1], 3)) for p in feat_data.get("probes", [])[:3]],
                }

    print(f"  {len(sig_clusters)} probe-signature clusters for "
          f"{sum(len(v) for v in sig_clusters.values())} explained features")

    for sig, members in sorted(sig_clusters.items(), key=lambda x: -len(x[1])):
        _, _, rep_data = members[0]
        rep_probes = [(p[0], p[1]) for p in rep_data.get("probes", [])[:5]]
        rep_examples = rep_data.get("examples", {})

        prompt = format_signature_prompt(
            sig, len(members), rep_probes,
            rep_examples=rep_examples, dataset_context=dataset_context,
        )

        for model, feat_idx, feat_data in members:
            explained[model][str(feat_idx)] = {
                "r2": round(feat_data["r2"], 4),
                "label": "unlabeled",
                "signature": sig,
                "top_probes": [(p[0], round(p[1], 3)) for p in feat_data.get("probes", [])[:3]],
                "prompt": prompt,
            }

    n_explained = sum(len(v) for v in explained.values())
    n_unexplained = sum(len(v) for v in unexplained.values())
    print(f"  Explained: {n_explained}, Unexplained: {n_unexplained}")
    print(f"  Unique signature prompts: {len(sig_clusters)}")

    return {
        "metadata": {"phase": 3},
        "unmatched_features": {
            "explained": dict(explained),
            "unexplained": dict(unexplained),
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────


def _serialize_uf(uf: UnionFind) -> list:
    """Serialize union-find edges for reproducibility."""
    edges = []
    for node in uf.parent:
        root = uf.find(node)
        if root != node:
            edges.append([list(node), list(root)])
    return edges


def build_feature_lookup(result: dict) -> dict:
    """Build per-model, per-feature label lookup from complete results."""
    lookup = defaultdict(dict)

    for gid, group in result.get("concept_groups", {}).items():
        for model, feat_idx in group["members"]:
            lookup[model][str(feat_idx)] = {
                "group_id": int(gid),
                "label": group["label"],
                "category": "matched",
            }

    unmatched = result.get("unmatched_features", {})
    for category in ("explained", "unexplained"):
        for model, features in unmatched.get(category, {}).items():
            for feat_idx, feat_data in features.items():
                lookup[model][feat_idx] = {
                    "group_id": None,
                    "label": feat_data["label"],
                    "category": f"unmatched_{category}",
                }

    return dict(lookup)


def build_summary(result: dict) -> dict:
    """Build summary statistics."""
    in_groups = sum(len(g["members"]) for g in result.get("concept_groups", {}).values())
    n_groups = len(result.get("concept_groups", {}))
    unmatched = result.get("unmatched_features", {})
    n_explained = sum(len(v) for v in unmatched.get("explained", {}).values())
    n_unexplained = sum(len(v) for v in unmatched.get("unexplained", {}).values())

    # Count labels
    n_labeled = 0
    n_unlabeled = 0
    for g in result.get("concept_groups", {}).values():
        if g.get("label", "unlabeled") != "unlabeled":
            n_labeled += 1
        else:
            n_unlabeled += 1
    for cat in ("explained", "unexplained"):
        for feats in unmatched.get(cat, {}).values():
            for f in feats.values():
                if f.get("label", "unlabeled") != "unlabeled":
                    n_labeled += 1
                else:
                    n_unlabeled += 1

    return {
        "total_alive_features": in_groups + n_explained + n_unexplained,
        "in_groups": in_groups,
        "n_groups": n_groups,
        "unmatched_explained": n_explained,
        "unmatched_unexplained": n_unexplained,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-model concept groups and generate labeling prompts"
    )
    parser.add_argument(
        "--phase", type=int, default=None, choices=[1, 2, 3],
        help="Run a single phase (default: run all 1->2->3)",
    )
    parser.add_argument("--min-r", type=float, default=0.20)
    parser.add_argument("--r2-threshold", type=float, default=0.1)
    parser.add_argument("--max-group-size", type=int, default=100)
    parser.add_argument("--relabel-threshold", type=float, default=0.3)
    parser.add_argument(
        "--matching", type=str,
        default="output/sae_feature_matching_mnn_t0.001_test.json",
    )
    parser.add_argument(
        "--concepts", type=str,
        default=f"output/sae_concept_analysis_round{DEFAULT_SAE_ROUND}.json",
    )
    parser.add_argument(
        "--output", type=str,
        default=f"output/cross_model_concept_labels_round{DEFAULT_SAE_ROUND}.json",
    )
    parser.add_argument(
        "--corr-dir", type=str,
        default="output/sae_cross_correlations",
    )
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3]
    t0 = time.time()

    # Load data
    print(f"Loading matching: {args.matching}")
    with open(PROJECT_ROOT / args.matching) as f:
        matching = json.load(f)

    print(f"Loading concept regression: {args.concepts}")
    with open(PROJECT_ROOT / args.concepts) as f:
        concepts = json.load(f)
    probe_lookup, dataset_context = load_concept_data(concepts, top_k=5)
    print(f"  {len(probe_lookup)} models, {len(dataset_context)} datasets with context")

    # Load dataset-to-domain mapping for prompt context
    domain_path = PROJECT_ROOT / "data" / "tabarena_domains.json"
    domain_lookup = {}
    if domain_path.exists():
        with open(domain_path) as f:
            domain_lookup = json.load(f).get("dataset_domain", {})
        print(f"  Loaded {len(domain_lookup)} dataset-to-domain mappings")

    out_path = PROJECT_ROOT / args.output
    result = None

    for phase in phases:
        if phase == 1:
            result = run_phase1(
                matching, probe_lookup, args.min_r,
                args.max_group_size, dataset_context=dataset_context,
                domain_lookup=domain_lookup,
            )
        elif phase == 2:
            if result is None:
                if out_path.exists():
                    with open(out_path) as f:
                        result = json.load(f)
                    print(f"Loaded prior output ({len(result['concept_groups'])} groups)")
                else:
                    print(f"No prior output at {out_path}, running phase 1 first")
                    result = run_phase1(
                        matching, probe_lookup, args.min_r,
                        args.max_group_size, dataset_context=dataset_context,
                        domain_lookup=domain_lookup,
                    )
            result = run_phase2(
                result, matching, probe_lookup, args.min_r,
                args.max_group_size, args.relabel_threshold,
                corr_dir=PROJECT_ROOT / args.corr_dir,
                dataset_context=dataset_context,
                domain_lookup=domain_lookup,
            )
        elif phase == 3:
            if result is None:
                if out_path.exists():
                    with open(out_path) as f:
                        result = json.load(f)
                    print(f"Loaded prior output ({len(result['concept_groups'])} groups)")
                else:
                    print(f"ERROR: Phase 3 requires prior output at {out_path}")
                    sys.exit(1)
            phase3 = run_phase3(
                result, probe_lookup, args.r2_threshold,
                dataset_context=dataset_context,
            )
            result["unmatched_features"] = phase3["unmatched_features"]
            result["metadata"]["phase"] = 3

    # Build lookup and summary
    result["feature_lookup"] = build_feature_lookup(result)
    result["summary"] = build_summary(result)
    result["metadata"]["sae_round"] = DEFAULT_SAE_ROUND
    result["metadata"]["min_r"] = args.min_r
    result["metadata"]["matching_file"] = args.matching
    result["metadata"]["concepts_file"] = args.concepts
    result["metadata"]["n_models"] = len(matching["metadata"]["models"])
    result["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    result["metadata"]["system_prompt"] = SYSTEM_PROMPT

    # Remove internal fields
    result.pop("_uf_edges", None)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(convert_keys_to_native(result), f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")

    # Print summary
    s = result["summary"]
    print(f"\n── Summary ──")
    print(f"  Total alive features: {s['total_alive_features']}")
    print(f"  In concept groups:    {s['in_groups']} ({s['n_groups']} groups)")
    print(f"  Unmatched explained:  {s['unmatched_explained']}")
    print(f"  Unmatched unexplained:{s['unmatched_unexplained']}")
    print(f"  Labeled: {s['n_labeled']}, Unlabeled: {s['n_unlabeled']}")
    print(f"  Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
