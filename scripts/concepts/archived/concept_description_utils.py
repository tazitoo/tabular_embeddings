"""Utilities for generating concept descriptions.

Extracts activating/non-activating row samples and formats prompts
for LLM-based concept labeling.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts._project_root import PROJECT_ROOT


def get_activating_samples(
    activations: np.ndarray,
    feat_idx: int,
    top_k: int = 5,
    bottom_k: int = 5,
    active_threshold: float = 0.001,
) -> Tuple[List[int], List[int]]:
    """Get indices of top-K activating and bottom-K non-activating rows.

    Args:
        activations: (n_samples, hidden_dim) SAE activation matrix.
        feat_idx: Feature index to analyze.
        top_k: Number of highest-activating rows to return.
        bottom_k: Number of non-activating rows to return.
        active_threshold: Below this, a row is considered non-activating.

    Returns:
        (high_indices, low_indices) — sorted by activation descending / ascending.
    """
    feat_acts = activations[:, feat_idx]

    # High: top-k by activation value (only rows that actually activate)
    active_mask = feat_acts > active_threshold
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return [], list(np.argsort(feat_acts)[:bottom_k])

    n_high = min(top_k, len(active_indices))
    high_indices = active_indices[np.argsort(feat_acts[active_indices])[-n_high:][::-1]]

    # Low: bottom-k from non-activating rows
    inactive_indices = np.where(~active_mask)[0]
    n_low = min(bottom_k, len(inactive_indices))
    low_indices = inactive_indices[np.argsort(feat_acts[inactive_indices])[:n_low]]

    return list(high_indices), list(low_indices)


def format_haiku_prompt(
    group_id: int,
    probes: List[Tuple[str, int, float]],
    n_models: int,
    n_members: int,
    per_member_detail: Optional[List[dict]] = None,
) -> str:
    """Format prompt for Haiku brief label (2-5 words).

    Args:
        probes: List of (probe_name, count, mean_coefficient) tuples.
        per_member_detail: Optional list of {model, feature_idx, r2, top_probes}.
    """
    lines = [
        f"=== Concept Group {group_id} ({n_models} models, {n_members} members) ===",
        "",
    ]

    if probes:
        lines.append("PROBE CONSENSUS:")
        for name, count, mc in probes[:8]:
            sign = "+" if mc > 0 else "-"
            lines.append(f"  {name}: {count}/{n_members} members, coeff={sign}{abs(mc):.3f}")
        lines.append("")

    if per_member_detail:
        lines.append("PER-MEMBER DETAIL:")
        for m in per_member_detail[:10]:
            probes_str = ", ".join(
                f"{p[0]}({p[1]:+.2f})" for p in m.get("top_probes", [])[:4]
            )
            lines.append(f"  {m['model']} #{m['feature_idx']} (R²={m['r2']:.2f}): {probes_str}")
        lines.append("")

    lines.append("What universal tabular pattern does this concept capture? (2-5 words only)")
    return "\n".join(lines)


def _format_rows_block(rows: List[dict], label: str) -> List[str]:
    """Format a block of sample rows for inclusion in a prompt.

    Each row dict may contain:
      - 'activation': float (SAE activation value)
      - 'dataset': str (dataset name)
      - Other keys are feature columns.
    """
    lines = [f"{label} ROWS (feature fires strongly):" if "ACTIVATING" in label.upper()
             else f"{label} ROWS (similar data, feature silent):"]
    for row in rows[:5]:
        activation = row.get("activation")
        dataset = row.get("dataset")
        # Separate metadata from feature columns
        feature_cols = {k: v for k, v in row.items()
                        if k not in ("activation", "dataset")}
        vals = ", ".join(f"{k}={v}" for k, v in list(feature_cols.items())[:12])
        prefix_parts = []
        if dataset:
            prefix_parts.append(f"[{dataset}]")
        if activation is not None:
            prefix_parts.append(f"activation={activation}")
        prefix = " ".join(prefix_parts)
        if prefix:
            lines.append(f"  {prefix}")
            lines.append(f"  {vals}")
        else:
            lines.append(f"  {vals}")
    return lines


def format_sonnet_group_prompt(
    group_id: int,
    probes: List[Tuple[str, int, float]],
    n_models: int,
    n_members: int,
    high_rows: List[dict],
    low_rows: List[dict],
    per_member_detail: Optional[List[dict]] = None,
    dataset_context: Optional[str] = None,
    domain_context: Optional[str] = None,
    mean_r2: Optional[float] = None,
    model_names: Optional[List[str]] = None,
) -> str:
    """Format prompt for Sonnet rich description of a concept group.

    Contrastive examples come first (primary evidence), then probe statistics
    as secondary guidance. Asks for 1-2 sentence monosemantic description.
    """
    model_str = f" ({', '.join(model_names)})" if model_names else ""
    r2_str = f", mean R²={mean_r2:.3f}" if mean_r2 is not None else ""
    lines = [
        f"=== Concept Group {group_id} ===",
        f"Models: {n_models}/8{model_str}",
        f"Members: {n_members} features{r2_str}",
        "",
    ]

    # Primary evidence: contrastive examples
    lines.extend(_format_rows_block(high_rows, "TOP-ACTIVATING"))
    lines.append("")
    lines.extend(_format_rows_block(low_rows, "NEAREST NON-ACTIVATING"))
    lines.append("")

    if dataset_context:
        lines.append(f"DATASET CONTEXT:\n{dataset_context}")
        lines.append("")

    # Secondary guidance: probe statistics
    if probes:
        lines.append("STATISTICAL GUIDANCE (probe correlations — explain ~20% of variance):")
        lines.append("These are partial hints, not the full story. Use them to check your")
        lines.append("interpretation of the contrastive examples above, not as primary evidence.")
        for name, count, mc in probes[:8]:
            sign = "+" if mc > 0 else "-"
            lines.append(f"  {name}: {count}/{n_members} members, coeff={sign}{abs(mc):.3f}")
        lines.append("")

    if per_member_detail:
        lines.append("PER-MEMBER DETAIL:")
        for m in per_member_detail[:6]:
            probes_str = ", ".join(
                f"{p[0]}({p[1]:+.2f})" for p in m.get("top_probes", [])[:4]
            )
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
        f"This concept was independently discovered by {n_models} different tabular "
        "foundation models, confirming it captures something real and universal — "
        "not a model artifact."
    )
    lines.append("")
    if domain_context:
        lines.append(f"The contrastive examples come from: {domain_context}")
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
        "read as a natural, technically oriented sentence that a data scientist would "
        "understand without seeing the raw data."
    )
    return "\n".join(lines)


def format_sonnet_unexplained_prompt(
    model: str,
    feat_idx: int,
    high_rows: List[dict],
    low_rows: List[dict],
    landmarks: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """Format prompt for Sonnet description of an unexplained feature.

    No probe guidance available. Uses activating/non-activating samples
    plus nearest described neighbor landmarks for context.
    """
    lines = [
        f"=== Unmatched Feature: {model} #{feat_idx} ===",
        "This feature has no strong statistical probe signal (low R²).",
        "",
    ]

    if landmarks:
        lines.append("NEAREST DESCRIBED NEIGHBOR LANDMARKS:")
        for desc, corr in landmarks[:5]:
            lines.append(f'  "{desc}" (correlation r={corr:.3f})')
        lines.append("")

    lines.extend(_format_rows_block(high_rows, "HIGH-ACTIVATING"))
    lines.append("")
    lines.extend(_format_rows_block(low_rows, "NON-ACTIVATING"))
    lines.append("")

    if landmarks:
        lines.append(
            "Describe what tabular pattern this feature detects in 1-2 sentences. "
            "The nearest neighbor landmarks above are related but distinct concepts — "
            "explain how this feature differs from or interpolates between them."
        )
    else:
        lines.append(
            "Describe what tabular pattern this feature detects in 1-2 sentences. "
            "Focus on what distinguishes the activating rows from the non-activating rows."
        )
    return "\n".join(lines)
