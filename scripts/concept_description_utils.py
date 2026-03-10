"""Utilities for generating concept descriptions.

Extracts activating/non-activating row samples and formats prompts
for LLM-based concept labeling.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    """Format a block of sample rows for inclusion in a prompt."""
    lines = [f"{label} ROWS:"]
    for i, row in enumerate(rows[:5]):
        vals = ", ".join(f"{k}={v}" for k, v in list(row.items())[:10])
        lines.append(f"  Row {i+1}: {vals}")
    return lines


def format_sonnet_group_prompt(
    group_id: int,
    probes: List[Tuple[str, int, float]],
    n_models: int,
    n_members: int,
    high_rows: List[dict],
    low_rows: List[dict],
    per_member_detail: Optional[List[dict]] = None,
) -> str:
    """Format prompt for Sonnet rich description of a concept group.

    Includes probe consensus + activating/non-activating row samples.
    Asks for 1-2 sentence description.
    """
    lines = [
        f"=== Concept Group {group_id} ({n_models} models, {n_members} members) ===",
        "",
    ]

    if probes:
        lines.append("PROBE CONSENSUS (statistical meta-features of activating rows):")
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

    lines.extend(_format_rows_block(high_rows, "HIGH-ACTIVATING"))
    lines.append("")
    lines.extend(_format_rows_block(low_rows, "NON-ACTIVATING"))
    lines.append("")

    lines.append(
        "Describe the tabular data pattern this concept detects in 1-2 sentences. "
        "Focus on what distinguishes the activating rows from the non-activating rows. "
        "Be specific about data properties (distributions, correlations, magnitudes)."
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
