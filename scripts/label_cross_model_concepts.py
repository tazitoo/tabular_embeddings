#!/usr/bin/env python3
"""
Cross-model SAE concept labeling with LLM.

Labels concept groups identified by cross-model feature matching. Runs in three
phases: Phase 1 labels MNN-matched groups, Phase 2 extends groups via
cross-correlation direct assignment, Phase 3 labels unmatched features.

Phase 2 uses pre-computed cross-correlation matrices (row-level detail) with
greedy direct assignment — no union-find transitive closure, preventing
mega-groups. Requires running match_sae_features.py --save-correlations first.

Usage:
    # Phase 1: MNN groups
    python scripts/label_cross_model_concepts.py --phase 1

    # Phase 2: Extend via cross-correlation (requires pre-computed matrices)
    python scripts/label_cross_model_concepts.py --phase 2

    # Phase 3: Label unmatched features
    python scripts/label_cross_model_concepts.py --phase 3

    # Dry run (no LLM calls)
    python scripts/label_cross_model_concepts.py --phase 1 --dry-run

    # Rule-based only (no API key needed)
    python scripts/label_cross_model_concepts.py --phase 1 --no-llm
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
    load_api_key,
)
from scripts.annotate_feature_matches import load_concept_probes
from scripts.generate_concept_dictionary import LABEL_VOCAB

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
    """
    Add edges of given tier with |r| >= min_r to union-find.

    Args:
        matching: Loaded tiered matching JSON.
        tier: 1 for MNN, 2 for Hungarian.
        min_r: Minimum |r| for an edge to be added.
        uf: Existing union-find to extend (or creates new one).

    Returns:
        Updated UnionFind.
    """
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
    """
    Aggregate probe profiles across all features in a concept group.

    Returns dict with n_models, mean_r2, probe_votes, top_probes, per_member.
    """
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
                "r2": 0.0, "top_probes": [],
            })
            continue

        r2_vals.append(feat_data["r2"])
        probes = feat_data.get("probes", [])
        per_member.append({
            "model": model, "feature_idx": feat_idx,
            "r2": feat_data["r2"],
            "top_probes": [(p[0], round(p[1], 3)) for p in probes[:5]],
        })
        for name, coeff in probes[:5]:
            probe_votes[name]["count"] += 1
            probe_votes[name]["coeffs"].append(coeff)
            probe_votes[name]["models"].add(model)

    # Sort probes by vote count (descending), then by mean |coeff|
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


# ── LLM interaction ──────────────────────────────────────────────────────


SYSTEM_PROMPT = """\
You are an expert at analyzing tabular data patterns. You are labeling \
universal concepts found by Sparse Autoencoders trained on different tabular \
foundation models. A "concept group" means features from multiple independent \
models that activate on the same data rows, indicating they detect the same \
underlying tabular pattern.

Respond with ONLY a concept label (2-5 words). Focus on data properties, \
not semantics. Examples: "extreme outliers", "high feature correlation", \
"sparse rows", "right-skewed distribution", "isolated points"."""


def format_group_prompt(group_id: int, agg: dict) -> str:
    """Format LLM prompt for a matched concept group."""
    lines = [
        f"=== Concept Group {group_id} ===",
        f"Models: {agg['n_models']}/7 ({', '.join(agg['models'])})",
        f"Members: {len(agg['per_member'])} features",
        f"Mean R² (probe-explained): {agg['mean_r2']:.3f}",
        "",
    ]

    # Probe consensus
    consensus = [(n, d["count"], d["mean_coeff"])
                 for n, d in agg["probe_votes"].items() if d["count"] >= 2]
    consensus.sort(key=lambda x: (-x[1], -abs(x[2])))
    if consensus:
        lines.append("PROBE CONSENSUS (appearing in 2+ members):")
        for name, count, mc in consensus[:8]:
            sign = "+" if mc > 0 else "-"
            lines.append(f"  {name}: {count}/{len(agg['per_member'])} members, coeff={sign}{abs(mc):.3f}")
        lines.append("")

    # Per-member detail
    lines.append("PER-MEMBER DETAIL:")
    for m in agg["per_member"][:10]:
        probes_str = ", ".join(f"{p[0]}({p[1]:+.2f})" for p in m["top_probes"][:4])
        lines.append(f"  {m['model']} #{m['feature_idx']} (R²={m['r2']:.2f}): {probes_str}")

    lines.append("")
    lines.append("What universal tabular pattern does this concept capture? (2-5 words only)")

    return "\n".join(lines)


def format_individual_prompt(model: str, feat_idx: int, feat_data: dict) -> str:
    """Format LLM prompt for an unmatched feature."""
    lines = [
        f"=== Unmatched Feature: {model} #{feat_idx} ===",
        f"R² (probe-explained): {feat_data.get('r2', 0):.3f}",
        "",
        "Top probes:",
    ]
    for name, coeff in feat_data.get("probes", [])[:5]:
        lines.append(f"  {name}: coeff={coeff:+.3f}")
    lines.append("")
    lines.append("This feature was not matched in other models. What tabular pattern does it detect? (2-5 words only)")
    return "\n".join(lines)


def format_signature_prompt(signature: str, count: int, example_probes: list) -> str:
    """Format LLM prompt for a probe-signature cluster of unmatched features."""
    lines = [
        f"=== Unmatched Feature Cluster ({count} features) ===",
        f"Probe signature: {signature}",
        "",
        "These features all share the same top probes:",
    ]
    for name, coeff in example_probes[:5]:
        lines.append(f"  {name}: coeff={coeff:+.3f}")
    lines.append("")
    lines.append("What tabular pattern do these features detect? (2-5 words only)")
    return "\n".join(lines)


_llm_disabled = False  # Set True after fatal API errors to stop retrying


def label_with_llm(prompt: str, client) -> Optional[str]:
    """Call Claude Haiku for a 2-5 word concept label."""
    global _llm_disabled
    if _llm_disabled:
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
            print(f"    LLM fatal error (disabling): {e}")
            _llm_disabled = True
        else:
            print(f"    LLM error: {e}")
        return None


def label_with_rules(top_probes: list) -> str:
    """
    Rule-based fallback label from probe coefficients.

    Args:
        top_probes: List of (probe_name, coefficient) tuples.
    """
    if not top_probes:
        return "uncharacterized"

    parts = []
    for name, coeff in top_probes[:2]:
        direction = "+" if coeff > 0 else "-"
        word = LABEL_VOCAB.get((name, direction))
        if word is None:
            word = ("high-" if coeff > 0 else "low-") + name.replace("_", "-")[:12]
        parts.append(word)

    return ", ".join(parts) if parts else "uncharacterized"


# ── Phase execution ───────────────────────────────────────────────────────


def _label_group(group_id, agg, client, dry_run, no_llm):
    """Label a single concept group. Returns (label, method, rule_label)."""
    # Rule-based label (always computed as fallback)
    rule_probes = [(name, mc) for name, _, mc in agg["top_probes"][:3]]
    rule_label = label_with_rules(rule_probes)

    if dry_run:
        return "[DRY RUN]", "dry_run", rule_label
    if no_llm or client is None:
        return rule_label, "rule", rule_label

    prompt = format_group_prompt(group_id, agg)
    llm_label = label_with_llm(prompt, client)
    if llm_label:
        return llm_label, "llm", rule_label
    return rule_label, "rule", rule_label


def run_phase1(
    matching: dict,
    probe_lookup: Dict[str, Dict[int, dict]],
    min_r: float,
    client,
    dry_run: bool,
    no_llm: bool,
    max_group_size: int,
) -> dict:
    """Phase 1: Build tier-1 (MNN) graph, find components, label each group."""
    print("\n── Phase 1: MNN concept groups ──")
    uf = build_match_graph(matching, tier=1, min_r=min_r)
    components = uf.components()
    print(f"  Found {len(components)} groups (2+ members)")

    concept_groups = {}
    n_llm = n_rule = n_skip = 0

    for i, (root, members) in enumerate(sorted(components.items(), key=lambda x: -len(x[1]))):
        agg = aggregate_group_probes(members, probe_lookup)

        if len(members) > max_group_size:
            concept_groups[str(i)] = {
                "members": [[m, f] for m, f in members],
                "n_models": agg["n_models"],
                "tier": 1,
                "mean_r2": agg["mean_r2"],
                "label": "too_large",
                "label_method": "skip",
                "rule_label": "",
                "top_probes": agg["top_probes"][:5],
                "phase_added": 1,
            }
            n_skip += 1
            continue

        label, method, rule_label = _label_group(i, agg, client, dry_run, no_llm)

        concept_groups[str(i)] = {
            "members": [[m, f] for m, f in members],
            "n_models": agg["n_models"],
            "tier": 1,
            "mean_r2": agg["mean_r2"],
            "label": label,
            "label_method": method,
            "rule_label": rule_label,
            "top_probes": agg["top_probes"][:5],
            "phase_added": 1,
        }

        if method == "llm":
            n_llm += 1
        elif method == "rule":
            n_rule += 1

        if (i + 1) % 100 == 0:
            print(f"  Labeled {i + 1}/{len(components)} groups...")

    print(f"  Labeled: {n_llm} LLM, {n_rule} rule, {n_skip} skipped")
    return {
        "metadata": {"phase": 1, "n_llm_calls": n_llm},
        "concept_groups": concept_groups,
        "_uf_edges": _serialize_uf(uf),
    }


def _load_cross_correlations(
    corr_dir: Path,
) -> Dict[str, dict]:
    """Load pre-computed cross-correlation matrices for all model pairs.

    Returns dict keyed by pair_key (e.g. 'Mitra__TabPFN') with:
        corr_matrix: (n_alive_a, n_alive_b) ndarray
        indices_a: absolute feature indices for model A
        indices_b: absolute feature indices for model B
        model_a, model_b: model display names
    """
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
    """Build bidirectional lookup: (model_x, model_y) -> correlation info.

    For pair (A, B) with corr_matrix shape (n_alive_A, n_alive_B):
    - (A, B) returns corr_matrix, indices_a=A's indices, indices_b=B's indices
    - (B, A) returns corr_matrix.T, indices_a=B's indices, indices_b=A's indices
    """
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
    client,
    dry_run: bool,
    no_llm: bool,
    max_group_size: int,
    relabel_threshold: float,
    corr_dir: Optional[Path] = None,
) -> dict:
    """Phase 2: Extend groups using cross-correlation direct assignment.

    Unlike the original approach (Hungarian + union-find), this uses:
    1. Pre-computed cross-correlation matrices (row-level detail)
    2. Direct greedy assignment (no transitive closure)

    For each unmatched feature:
    - Find its best correlate across ALL other models
    - If that correlate is in a tier-1 group, add the feature to that group
    - If no grouped correlate found, pair with best unmatched counterpart
    - Respects max_group_size constraint
    """
    print("\n── Phase 2: Extending groups via cross-correlation ──")

    if corr_dir is None:
        corr_dir = Path(__file__).parent.parent / "output" / "sae_cross_correlations"

    if not corr_dir.exists():
        print(f"  ERROR: No cross-correlation directory at {corr_dir}")
        print(f"  Run: python scripts/match_sae_features.py --save-correlations")
        return {
            "metadata": {"phase": 2, "n_llm_calls": 0},
            "concept_groups": dict(prev_output["concept_groups"]),
        }

    corr_data = _load_cross_correlations(corr_dir)
    if not corr_data:
        print(f"  ERROR: No .npz files in {corr_dir}")
        return {
            "metadata": {"phase": 2, "n_llm_calls": 0},
            "concept_groups": dict(prev_output["concept_groups"]),
        }

    corr_lookup = _build_corr_lookup(corr_data)
    print(f"  Loaded {len(corr_data)} cross-correlation matrices")

    # Pre-build absolute→relative index maps for each pair (avoids linear scans)
    import numpy as _np

    abs_to_rel = {}  # (model_x, model_y) → {abs_feat_idx_in_x: relative_position}
    for pair_key, info in corr_lookup.items():
        abs_to_rel[pair_key] = {
            int(idx): i for i, idx in enumerate(info["indices_a"])
        }

    # Build node → group_id mapping from phase 1
    concept_groups = dict(prev_output["concept_groups"])
    node_to_group = {}  # (model, feat_idx) → group_id
    for gid, group in concept_groups.items():
        for model, feat_idx in group["members"]:
            node_to_group[(model, feat_idx)] = gid

    # Pre-build per-model grouped feature masks for each pair
    # grouped_mask[(model_x, model_y)][j] = True if indices_b[j] is in a group
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

    # Collect all models and all alive features per model from probe lookup
    all_models = set()
    alive_per_model = {}  # model → set of alive feature indices
    for model, features in probe_lookup.items():
        all_models.add(model)
        alive_per_model[model] = set(int(k) for k in features.keys())

    # Identify unmatched features (alive but not in any group)
    unmatched = []  # list of (model, feat_idx)
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
    n_llm = 0

    still_unmatched = []

    for model, feat_idx in unmatched:
        best_r = 0.0
        best_group = None

        # Search across all other models for the best correlating grouped feature
        for other_model in sorted(all_models):
            if other_model == model:
                continue

            pair = (model, other_model)
            if pair not in corr_lookup:
                continue

            # Fast lookup: is this feature in the alive set for this pair?
            pos_a = abs_to_rel.get(pair, {}).get(feat_idx)
            if pos_a is None:
                continue

            info = corr_lookup[pair]
            corr_row = info["corr_matrix"][pos_a]  # (n_alive_b,)
            mask, group_ids = grouped_mask_cache[pair]

            # Vectorized: find best grouped correlate
            grouped_corr = _np.where(mask, corr_row, 0.0)
            j_best = int(grouped_corr.argmax())
            r = float(grouped_corr[j_best])
            if r > best_r and group_ids[j_best] is not None:
                best_r = r
                best_group = group_ids[j_best]

        if best_r >= min_r and best_group is not None:
            # Check group size
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
    # For each still-unmatched feature, find its best still-unmatched counterpart
    still_unmatched_set = set(still_unmatched)
    assigned_pass2 = set()

    # Build unmatched masks for each pair
    unmatched_mask_cache = {}
    for pair_key, info in corr_lookup.items():
        other_model = info["model_b"]
        idx_b = info["indices_b"]
        mask = _np.zeros(len(idx_b), dtype=bool)
        for j, abs_idx in enumerate(idx_b):
            if (other_model, int(abs_idx)) in still_unmatched_set:
                mask[j] = True
        unmatched_mask_cache[pair_key] = mask

    # Collect all candidate pairs with their correlations
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

            # Find best unmatched correlate
            masked_corr = _np.where(um_mask, corr_row, 0.0)
            j_best = int(masked_corr.argmax())
            r = float(masked_corr[j_best])
            if r >= min_r:
                abs_j = int(idx_b[j_best])
                candidate_pairs.append((r, model, feat_idx, other_model, abs_j))

    # Greedy: process highest-correlation pairs first
    candidate_pairs.sort(reverse=True)
    for r, m_a, f_a, m_b, f_b in candidate_pairs:
        if (m_a, f_a) in assigned_pass2 or (m_b, f_b) in assigned_pass2:
            continue

        # Check if either was assigned to a group in the meantime (by a previous pair)
        if (m_a, f_a) in node_to_group or (m_b, f_b) in node_to_group:
            continue

        # Create new tier-2 group
        gid = str(next_id)
        next_id += 1

        members = [(m_a, f_a), (m_b, f_b)]
        agg = aggregate_group_probes(members, probe_lookup)
        label, method, rule_label = _label_group(int(gid), agg, client, dry_run, no_llm)
        if method == "llm":
            n_llm += 1

        concept_groups[gid] = {
            "members": [[m, f] for m, f in members],
            "n_models": agg["n_models"],
            "tier": 2,
            "mean_r2": agg["mean_r2"],
            "label": label,
            "label_method": method,
            "rule_label": rule_label,
            "top_probes": agg["top_probes"][:5],
            "phase_added": 2,
            "match_r": round(r, 4),
        }

        node_to_group[(m_a, f_a)] = gid
        node_to_group[(m_b, f_b)] = gid
        assigned_pass2.add((m_a, f_a))
        assigned_pass2.add((m_b, f_b))
        n_new += 1

    # --- Pass 3: Extend new tier-2 groups with more unmatched features ---
    # Features still unmatched after pass 2 can join tier-2 groups
    remaining = [
        (m, f) for m, f in still_unmatched
        if (m, f) not in node_to_group
    ]
    n_extended_t2 = 0

    # Rebuild grouped masks to include pass-2 assignments
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

    # Re-label extended groups that grew significantly
    n_relabel = 0
    for gid, group in concept_groups.items():
        if group.get("phase_added") == 2:
            continue  # Already labeled in this phase
        prev_members = prev_output["concept_groups"].get(gid, {}).get("members", [])
        if not prev_members:
            continue
        growth = (len(group["members"]) - len(prev_members)) / len(prev_members)
        if growth > relabel_threshold and len(group["members"]) <= max_group_size:
            agg = aggregate_group_probes(
                [(m, f) for m, f in group["members"]], probe_lookup
            )
            label, method, rule_label = _label_group(int(gid), agg, client, dry_run, no_llm)
            group["label"] = label
            group["label_method"] = method
            group["rule_label"] = rule_label
            group["n_models"] = agg["n_models"]
            group["mean_r2"] = agg["mean_r2"]
            group["top_probes"] = agg["top_probes"][:5]
            group["phase_relabeled"] = 2
            n_relabel += 1
            if method == "llm":
                n_llm += 1

    # Summary
    total_grouped = sum(
        len(g["members"]) for g in concept_groups.values()
    )
    max_size = max(
        (len(g["members"]) for g in concept_groups.values()), default=0
    )
    print(f"  Pass 2 (new tier-2 pairs): {n_new} new groups")
    print(f"  Pass 3 (extend tier-2): {n_extended_t2} more assigned")
    print(f"  Re-labeled: {n_relabel}, LLM calls: {n_llm}")
    print(f"  Total grouped: {total_grouped}, max group size: {max_size}")

    return {
        "metadata": {"phase": 2, "n_llm_calls": n_llm},
        "concept_groups": concept_groups,
    }


def run_phase3(
    prev_output: dict,
    probe_lookup: Dict[str, Dict[int, dict]],
    r2_threshold: float,
    client,
    dry_run: bool,
    no_llm: bool,
) -> dict:
    """Phase 3: Label unmatched features."""
    print("\n── Phase 3: Unmatched features ──")

    # Find all features already in groups
    grouped = set()
    for gid, group in prev_output["concept_groups"].items():
        for model, feat_idx in group["members"]:
            grouped.add((model, feat_idx))

    # Find unmatched features
    explained = defaultdict(dict)  # model → {feat_idx: {...}}
    unexplained = defaultdict(dict)
    sig_clusters = defaultdict(list)  # signature → [(model, feat_idx, feat_data)]

    for model, features in probe_lookup.items():
        for feat_idx, feat_data in features.items():
            if (model, feat_idx) in grouped:
                continue
            if feat_data["r2"] >= r2_threshold:
                # Cluster by top-2 probe signature for efficient labeling
                probes = feat_data.get("probes", [])[:2]
                sig = "__".join(p[0] for p in probes) if probes else "none"
                sig_clusters[sig].append((model, feat_idx, feat_data))
            else:
                unexplained[model][str(feat_idx)] = {
                    "r2": round(feat_data["r2"], 4),
                    "label": "unexplained",
                    "label_method": "threshold",
                    "top_probes": [(p[0], round(p[1], 3)) for p in feat_data.get("probes", [])[:3]],
                }

    # Label signature clusters
    n_llm = n_rule = 0
    print(f"  {len(sig_clusters)} probe-signature clusters for {sum(len(v) for v in sig_clusters.values())} explained features")

    for sig, members in sorted(sig_clusters.items(), key=lambda x: -len(x[1])):
        # Use first member's probes as representative
        _, _, rep_data = members[0]
        rep_probes = [(p[0], p[1]) for p in rep_data.get("probes", [])[:5]]
        rule_label = label_with_rules(rep_probes)

        if dry_run:
            label, method = "[DRY RUN]", "dry_run"
        elif no_llm or client is None:
            label, method = rule_label, "rule"
        else:
            prompt = format_signature_prompt(sig, len(members), rep_probes)
            llm_label = label_with_llm(prompt, client)
            if llm_label:
                label, method = llm_label, "llm"
                n_llm += 1
            else:
                label, method = rule_label, "rule"
                n_rule += 1

        if method == "rule":
            n_rule += 1

        for model, feat_idx, feat_data in members:
            explained[model][str(feat_idx)] = {
                "r2": round(feat_data["r2"], 4),
                "label": label,
                "label_method": method,
                "rule_label": rule_label,
                "signature": sig,
                "top_probes": [(p[0], round(p[1], 3)) for p in feat_data.get("probes", [])[:3]],
            }

    n_explained = sum(len(v) for v in explained.values())
    n_unexplained = sum(len(v) for v in unexplained.values())
    print(f"  Explained: {n_explained}, Unexplained: {n_unexplained}")
    print(f"  LLM calls: {n_llm}, Rule: {n_rule}")

    return {
        "metadata": {"phase": 3, "n_llm_calls": n_llm},
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

    # From concept groups
    for gid, group in result.get("concept_groups", {}).items():
        for model, feat_idx in group["members"]:
            lookup[model][str(feat_idx)] = {
                "group_id": int(gid),
                "label": group["label"],
                "category": "matched",
            }

    # From unmatched
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

    methods = defaultdict(int)
    for g in result.get("concept_groups", {}).values():
        methods[g.get("label_method", "unknown")] += 1
    for cat in ("explained", "unexplained"):
        for feats in unmatched.get(cat, {}).values():
            for f in feats.values():
                methods[f.get("label_method", "unknown")] += 1

    return {
        "total_alive_features": in_groups + n_explained + n_unexplained,
        "in_groups": in_groups,
        "n_groups": n_groups,
        "unmatched_explained": n_explained,
        "unmatched_unexplained": n_unexplained,
        "labels_by_method": dict(methods),
    }


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Label cross-model SAE concept groups with LLM"
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--min-r", type=float, default=0.20)
    parser.add_argument("--r2-threshold", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-group-size", type=int, default=100)
    parser.add_argument("--relabel-threshold", type=float, default=0.3)
    parser.add_argument(
        "--matching", type=str,
        default="output/sae_feature_matching_mnn_t0.001_n500.json",
    )
    parser.add_argument(
        "--concepts", type=str,
        default="output/concept_regression_with_pymfe.json",
    )
    parser.add_argument(
        "--output", type=str,
        default="output/cross_model_concept_labels.json",
    )
    parser.add_argument(
        "--corr-dir", type=str,
        default="output/sae_cross_correlations",
        help="Directory with pre-computed cross-correlation NPZ files (for phase 2)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    print(f"Loading matching: {args.matching}")
    with open(PROJECT_ROOT / args.matching) as f:
        matching = json.load(f)

    print(f"Loading concept regression: {args.concepts}")
    with open(PROJECT_ROOT / args.concepts) as f:
        concepts = json.load(f)
    probe_lookup = load_concept_probes(concepts, top_k=5)

    # Initialize LLM client
    client = None
    if not args.no_llm and not args.dry_run:
        api_key = load_api_key()
        if api_key:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("LLM client initialized (Claude Haiku)")
        else:
            print("WARNING: No API key found, falling back to rule-based labels")

    out_path = PROJECT_ROOT / args.output

    if args.phase == 1:
        result = run_phase1(
            matching, probe_lookup, args.min_r, client, args.dry_run,
            args.no_llm, args.max_group_size,
        )
    elif args.phase == 2:
        if out_path.exists():
            with open(out_path) as f:
                prev = json.load(f)
            print(f"Loaded phase {prev['metadata']['phase']} output ({len(prev['concept_groups'])} groups)")
        else:
            print(f"No prior output at {out_path}, running phase 1 first")
            prev = run_phase1(
                matching, probe_lookup, args.min_r, client, args.dry_run,
                args.no_llm, args.max_group_size,
            )
        result = run_phase2(
            prev, matching, probe_lookup, args.min_r, client, args.dry_run,
            args.no_llm, args.max_group_size, args.relabel_threshold,
            corr_dir=PROJECT_ROOT / args.corr_dir,
        )
    elif args.phase == 3:
        if out_path.exists():
            with open(out_path) as f:
                prev = json.load(f)
            print(f"Loaded phase {prev['metadata']['phase']} output ({len(prev['concept_groups'])} groups)")
        else:
            print(f"ERROR: Phase 3 requires prior output at {out_path}")
            sys.exit(1)
        phase3 = run_phase3(
            prev, probe_lookup, args.r2_threshold, client, args.dry_run,
            args.no_llm,
        )
        result = prev
        result["unmatched_features"] = phase3["unmatched_features"]
        result["metadata"]["phase"] = 3
        result["metadata"]["n_llm_calls"] = (
            result["metadata"].get("n_llm_calls", 0) + phase3["metadata"]["n_llm_calls"]
        )

    # Build lookup and summary
    result["feature_lookup"] = build_feature_lookup(result)
    result["summary"] = build_summary(result)
    result["metadata"]["min_r"] = args.min_r
    result["metadata"]["matching_file"] = args.matching
    result["metadata"]["concepts_file"] = args.concepts
    result["metadata"]["n_models"] = len(matching["metadata"]["models"])
    result["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)

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
    print(f"  Labels by method:     {s['labels_by_method']}")
    print(f"  Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
