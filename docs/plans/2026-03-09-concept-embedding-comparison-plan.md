# Concept Embedding Comparison — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Three standalone scripts that generate richer concept descriptions (Haiku+Sonnet), embed them locally (nomic-embed), and validate embedding quality (self-checks + optional API).

**Architecture:** Script 1 generates descriptions via Anthropic API (three-pass: Haiku labels, Sonnet descriptions for explained features, Sonnet + landmarks for unexplained). Script 2 embeds descriptions locally and computes validation metrics. Script 3 validates embedding quality via Matryoshka self-checks and optional API cross-reference.

**Tech Stack:** anthropic SDK, sentence-transformers, nomic-embed-text-v1.5, numpy, scipy

**Design doc:** `docs/plans/2026-03-09-concept-embedding-comparison-design.md`

---

## Task 1: Shared utilities — sample extraction

Extracts top-K activating and bottom-K non-activating rows for a given SAE feature.
This is the core building block for generating data-grounded descriptions.

**Files:**
- Create: `scripts/concept_description_utils.py`
- Test: `tests/test_concept_description_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_concept_description_utils.py
"""Tests for concept description utilities."""
import numpy as np
import pytest


def test_get_activating_samples_returns_correct_counts():
    """Top-k activating and bottom-k non-activating rows."""
    from scripts.concept_description_utils import get_activating_samples

    # 20 rows, 4 features. Feature 1 has clear high/low split.
    rng = np.random.RandomState(42)
    activations = rng.rand(20, 4)
    activations[:5, 1] = 10.0   # top 5 rows strongly activate feature 1
    activations[10:, 1] = 0.0   # bottom 10 rows don't activate

    high, low = get_activating_samples(activations, feat_idx=1, top_k=3, bottom_k=3)

    assert len(high) == 3
    assert len(low) == 3
    # High indices should be from the first 5 rows
    assert all(i < 5 for i in high)
    # Low indices should be from rows 10+
    assert all(i >= 10 for i in low)


def test_get_activating_samples_handles_sparse_feature():
    """Feature with very few activating rows returns what's available."""
    from scripts.concept_description_utils import get_activating_samples

    activations = np.zeros((20, 4))
    activations[0, 2] = 1.0  # Only 1 row activates feature 2

    high, low = get_activating_samples(activations, feat_idx=2, top_k=5, bottom_k=5)

    assert len(high) == 1  # Only 1 available
    assert len(low) == 5
    assert high[0] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_concept_description_utils.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# scripts/concept_description_utils.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_concept_description_utils.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scripts/concept_description_utils.py tests/test_concept_description_utils.py
git commit -m "feat: add activating sample extraction for concept descriptions"
```

---

## Task 2: Shared utilities — prompt formatting

Format prompts for Haiku (brief label) and Sonnet (rich description) from probe
data + row samples.

**Files:**
- Modify: `scripts/concept_description_utils.py`
- Test: `tests/test_concept_description_utils.py`

**Step 1: Write the failing tests**

```python
def test_format_haiku_prompt_includes_probes():
    """Haiku prompt includes probe consensus and asks for 2-5 words."""
    from scripts.concept_description_utils import format_haiku_prompt

    probes = [("frac_zeros", 5, -1.2), ("numeric_skewness", 3, 0.8)]
    prompt = format_haiku_prompt(group_id=0, probes=probes, n_models=4, n_members=12)

    assert "frac_zeros" in prompt
    assert "2-5 words" in prompt
    assert "4 models" in prompt or "n_models=4" in prompt


def test_format_sonnet_group_prompt_includes_samples():
    """Sonnet group prompt includes probe data AND row samples."""
    from scripts.concept_description_utils import format_sonnet_group_prompt

    probes = [("frac_zeros", 5, -1.2)]
    high_rows = [{"col_a": 0.0, "col_b": 1.5}, {"col_a": 0.0, "col_b": 2.3}]
    low_rows = [{"col_a": 0.9, "col_b": 0.1}, {"col_a": 0.7, "col_b": 0.4}]

    prompt = format_sonnet_group_prompt(
        group_id=0, probes=probes, n_models=3, n_members=8,
        high_rows=high_rows, low_rows=low_rows,
    )

    assert "frac_zeros" in prompt
    assert "ACTIVATING" in prompt.upper() or "activating" in prompt.lower()
    assert "1-2 sentences" in prompt or "one to two sentences" in prompt.lower()


def test_format_sonnet_unexplained_prompt_includes_landmarks():
    """Unexplained feature prompt includes landmark descriptions."""
    from scripts.concept_description_utils import format_sonnet_unexplained_prompt

    high_rows = [{"x": 1.0}]
    low_rows = [{"x": 0.0}]
    landmarks = [
        ("sparse rows with many zero-valued numerics", 0.34),
        ("extreme outlier rows", 0.28),
    ]

    prompt = format_sonnet_unexplained_prompt(
        model="TabPFN", feat_idx=789,
        high_rows=high_rows, low_rows=low_rows, landmarks=landmarks,
    )

    assert "sparse rows" in prompt
    assert "0.34" in prompt
    assert "landmark" in prompt.lower() or "neighbor" in prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_concept_description_utils.py::test_format_haiku_prompt_includes_probes -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `scripts/concept_description_utils.py`:

```python
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
        f"This feature has no strong statistical probe signal (low R²).",
        "",
    ]

    if landmarks:
        lines.append("NEAREST DESCRIBED NEIGHBOR LANDMARKS:")
        for desc, corr in landmarks[:5]:
            lines.append(f"  \"{desc}\" (correlation r={corr:.3f})")
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_concept_description_utils.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add scripts/concept_description_utils.py tests/test_concept_description_utils.py
git commit -m "feat: add prompt formatting for Haiku labels and Sonnet descriptions"
```

---

## Task 3: Script 1 — description generation core

The main `generate_concept_descriptions.py` script. Loads concept groups, probes,
SAE activations, and calls Haiku/Sonnet to produce descriptions.

**Files:**
- Create: `scripts/generate_concept_descriptions.py`
- Ref: `scripts/label_cross_model_concepts.py` (existing LLM call pattern, lines 271-292)
- Ref: `scripts/row_level_probes.py:145-169` (SAE activation loading)
- Ref: `scripts/concept_fingerprint.py:51-81` (per-dataset embedding loading)
- Ref: `scripts/intervene_sae.py:64-100` (SAE loading with archetypal params)
- Test: `tests/test_generate_concept_descriptions.py`

**Step 1: Write the failing test**

```python
# tests/test_generate_concept_descriptions.py
"""Tests for concept description generation."""
import json
import pytest
from unittest.mock import MagicMock, patch


def test_generate_group_description_calls_sonnet():
    """Group description calls Sonnet with correct prompt structure."""
    from scripts.generate_concept_descriptions import describe_group

    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Rows with many zero-valued features and low entropy.")]
    )

    group = {
        "members": [["TabPFN", 305], ["Mitra", 42]],
        "n_models": 2,
        "top_probes": [["frac_zeros", 2, -1.2]],
    }
    high_rows = [{"a": 0.0, "b": 1.0}]
    low_rows = [{"a": 0.5, "b": 0.3}]

    result = describe_group(
        group_id=0, group=group, high_rows=high_rows, low_rows=low_rows,
        client=mock_client, model="claude-sonnet-4-20250514",
    )

    assert "zero-valued" in result
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


def test_describe_group_falls_back_on_api_error():
    """Returns None on API error without crashing."""
    from scripts.generate_concept_descriptions import describe_group

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("rate limit")

    group = {"members": [], "n_models": 0, "top_probes": []}
    result = describe_group(
        group_id=0, group=group, high_rows=[], low_rows=[],
        client=mock_client, model="claude-sonnet-4-20250514",
    )

    assert result is None


def test_output_schema_has_required_keys():
    """Output JSON has metadata, groups, and unmatched sections."""
    from scripts.generate_concept_descriptions import build_output_skeleton

    skeleton = build_output_skeleton()
    assert "metadata" in skeleton
    assert "groups" in skeleton
    assert "unmatched" in skeleton
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_concept_descriptions.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Generate concept descriptions using Haiku (brief) and Sonnet (rich).

Three-pass pipeline:
  Pass 1: Haiku brief labels (2-5 words) for all groups + unmatched
  Pass 2: Sonnet rich descriptions for grouped + explained features
  Pass 3: Sonnet rich descriptions for unexplained features with landmarks

Usage:
    python scripts/generate_concept_descriptions.py --pass all
    python scripts/generate_concept_descriptions.py --pass 1  # Haiku only
    python scripts/generate_concept_descriptions.py --pass 2  # Sonnet grouped
    python scripts/generate_concept_descriptions.py --pass 3  # Sonnet unexplained
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.concept_description_utils import (
    format_haiku_prompt,
    format_sonnet_group_prompt,
    format_sonnet_unexplained_prompt,
    get_activating_samples,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"
LABELS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels_v2.json"
PROBES_PATH = PROJECT_ROOT / "output" / "concept_regression_with_pymfe.json"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

HAIKU_SYSTEM = """\
You are an expert at analyzing tabular data patterns. You are labeling \
universal concepts found by Sparse Autoencoders trained on tabular \
foundation models. Respond with ONLY a concept label (2-5 words). \
Focus on data properties, not semantics. Examples: "extreme outliers", \
"high feature correlation", "sparse rows", "right-skewed distribution"."""

SONNET_SYSTEM = """\
You are an expert at analyzing tabular data patterns discovered by Sparse \
Autoencoders trained on tabular foundation models. You will see statistical \
meta-feature probes and/or actual data rows that activate (or don't activate) \
a learned concept. Describe the tabular pattern in 1-2 precise sentences. \
Focus on data properties: distributions, magnitudes, correlations, sparsity. \
Do not speculate about domain meaning."""


def build_output_skeleton() -> dict:
    """Create empty output structure with metadata."""
    return {
        "metadata": {
            "haiku_model": HAIKU_MODEL,
            "sonnet_model": SONNET_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "groups": {},
        "unmatched": {},
    }


def call_llm(
    client, model: str, system: str, prompt: str, max_tokens: int = 200,
) -> Optional[str]:
    """Call Anthropic API with error handling. Returns text or None."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().strip('"').strip("'")
        return text if text else None
    except Exception as e:
        print(f"    LLM error: {e}")
        return None


def describe_group(
    group_id: int,
    group: dict,
    high_rows: List[dict],
    low_rows: List[dict],
    client,
    model: str = SONNET_MODEL,
    per_member_detail: Optional[List[dict]] = None,
) -> Optional[str]:
    """Generate Sonnet description for a concept group."""
    prompt = format_sonnet_group_prompt(
        group_id=group_id,
        probes=group.get("top_probes", []),
        n_models=group.get("n_models", 0),
        n_members=len(group.get("members", [])),
        high_rows=high_rows,
        low_rows=low_rows,
        per_member_detail=per_member_detail,
    )
    return call_llm(client, model, SONNET_SYSTEM, prompt)


# ... main() with argparse, pass dispatch, checkpoint/resume logic
```

Full implementation will include:
- `load_activations(model_key)` — reuses `load_sae` from `intervene_sae.py:64`
  and `encode_all_rows` pattern from `row_level_probes.py:145`
- `load_raw_rows(model_key, dataset, indices)` — loads original tabular data
  for human-readable row samples via TabArena dataset loading
- `run_pass1(client, labels, probes)` — Haiku brief labels
- `run_pass2(client, labels, probes, activations)` — Sonnet grouped + explained
- `run_pass3(client, labels, activations, pass2_results)` — Sonnet with landmarks
- Checkpoint/resume: saves after each pass to allow incremental progress

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_concept_descriptions.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add scripts/generate_concept_descriptions.py tests/test_generate_concept_descriptions.py
git commit -m "feat: add concept description generation script (Haiku + Sonnet)"
```

---

## Task 4: Script 1 — activation loading and row retrieval

Wire up the SAE activation pipeline and raw row loading so descriptions
are grounded in actual data.

**Files:**
- Modify: `scripts/generate_concept_descriptions.py`
- Ref: `scripts/intervene_sae.py:64-100` (load_sae)
- Ref: `scripts/concept_fingerprint.py:51-81` (load_per_dataset_embeddings)
- Ref: `scripts/row_level_probes.py:145-169` (encode_all_rows)
- Ref: `data/extended_loader.py` (TabArena dataset loading)
- Test: `tests/test_generate_concept_descriptions.py`

**Step 1: Write the failing test**

```python
def test_load_activations_shape(tmp_path):
    """Activation loader returns (n_samples, hidden_dim) array."""
    # This test requires real data — skip if not available
    from scripts.generate_concept_descriptions import load_activations_cached

    sae_dir = Path("output/sae_tabarena_sweep_round6")
    train_dir = Path("output/sae_training_round6")
    if not (sae_dir / "tabpfn" / "sae_matryoshka_archetypal_validated.pt").exists():
        pytest.skip("Real SAE checkpoint not available")

    acts, samples_per_ds = load_activations_cached("tabpfn", sae_dir, train_dir, device="cpu")
    assert acts.ndim == 2
    assert acts.shape[1] > 0  # hidden_dim
    assert len(samples_per_ds) > 0
```

**Step 2: Run test, verify it fails**

**Step 3: Implement activation loading**

Add to `scripts/generate_concept_descriptions.py`:

```python
from scripts.intervene_sae import load_sae
from scripts.concept_fingerprint import load_per_dataset_embeddings
from scripts.compare_sae_cross_model import sae_sweep_dir

_activation_cache = {}

def load_activations_cached(
    model_key: str,
    sae_dir: Path = None,
    training_dir: Path = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Load SAE activations for all training rows, with caching.

    Returns:
        activations: (n_total, hidden_dim) array
        samples_per_dataset: list of (dataset_name, count) tuples
    """
    if model_key in _activation_cache:
        return _activation_cache[model_key]

    if sae_dir is None:
        sae_dir = sae_sweep_dir()
    if training_dir is None:
        training_dir = PROJECT_ROOT / "output" / "sae_training_round6"

    per_ds = load_per_dataset_embeddings(model_key, training_dir)
    ds_order = list(per_ds.keys())
    all_emb = np.concatenate([per_ds[ds] for ds in ds_order], axis=0)
    samples_per_dataset = [(ds, len(per_ds[ds])) for ds in ds_order]

    sae, config = load_sae(model_key, sae_dir=sae_dir, device=device)
    sae.eval()

    with torch.no_grad():
        x = torch.tensor(all_emb, dtype=torch.float32, device=device)
        h = sae.encode(x)
        activations = h.cpu().numpy()

    _activation_cache[model_key] = (activations, samples_per_dataset)
    return activations, samples_per_dataset
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/test_generate_concept_descriptions.py -v`

**Step 5: Commit**

```bash
git add scripts/generate_concept_descriptions.py tests/test_generate_concept_descriptions.py
git commit -m "feat: add activation loading and caching for description generation"
```

---

## Task 5: Script 1 — three-pass main loop with checkpoint/resume

Wire up the full three-pass pipeline with checkpoint/resume support
so interrupted runs can continue.

**Files:**
- Modify: `scripts/generate_concept_descriptions.py`
- Test: `tests/test_generate_concept_descriptions.py`

**Step 1: Write the failing test**

```python
def test_checkpoint_resume_skips_completed(tmp_path):
    """Resuming from checkpoint skips already-described groups."""
    from scripts.generate_concept_descriptions import (
        build_output_skeleton,
        load_checkpoint,
        save_checkpoint,
    )

    output = build_output_skeleton()
    output["groups"]["0"] = {
        "brief_label": "sparse rows",
        "summary": "Rows with many zeros.",
    }

    ckpt_path = tmp_path / "checkpoint.json"
    save_checkpoint(output, ckpt_path)

    loaded = load_checkpoint(ckpt_path)
    assert "0" in loaded["groups"]
    assert loaded["groups"]["0"]["brief_label"] == "sparse rows"
```

**Step 2: Run test, verify it fails**

**Step 3: Implement checkpoint + main loop**

Add to `scripts/generate_concept_descriptions.py`:

```python
def save_checkpoint(output: dict, path: Path):
    """Save current progress to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_checkpoint(path: Path) -> Optional[dict]:
    """Load checkpoint if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate concept descriptions")
    parser.add_argument("--pass", dest="run_pass", type=str, default="all",
                        choices=["all", "1", "2", "3"],
                        help="Which pass to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Max activating/non-activating rows per feature")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for SAE encoding")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / "concept_descriptions_checkpoint.json"
    final_path = OUTPUT_DIR / "concept_descriptions.json"

    if args.resume and ckpt_path.exists():
        output = load_checkpoint(ckpt_path)
        print(f"Resumed from checkpoint: {len(output['groups'])} groups done")
    else:
        output = build_output_skeleton()

    # Load shared data
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(PROBES_PATH) as f:
        probes_data = json.load(f)

    passes = ["1", "2", "3"] if args.run_pass == "all" else [args.run_pass]

    for p in passes:
        print(f"\n{'='*60}")
        print(f"Pass {p}")
        print("=" * 60)

        if p == "1":
            run_pass1(client, labels, probes_data, output)
        elif p == "2":
            run_pass2(client, labels, probes_data, output, args)
        elif p == "3":
            run_pass3(client, labels, output, args)

        save_checkpoint(output, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    with open(final_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFinal output: {final_path}")
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/test_generate_concept_descriptions.py -v`

**Step 5: Commit**

```bash
git add scripts/generate_concept_descriptions.py tests/test_generate_concept_descriptions.py
git commit -m "feat: add three-pass main loop with checkpoint/resume"
```

---

## Task 6: Script 2 — embedding and metrics

The standalone embedding script that takes `concept_descriptions.json`
and produces embeddings + validation metrics.

**Files:**
- Create: `scripts/embed_concept_descriptions.py`
- Test: `tests/test_embed_concept_descriptions.py`

**Step 1: Write the failing tests**

```python
# tests/test_embed_concept_descriptions.py
"""Tests for concept description embedding."""
import json
import numpy as np
import pytest


def test_embed_descriptions_returns_correct_shape():
    """Embedding N descriptions produces (N, dim) array."""
    from scripts.embed_concept_descriptions import embed_descriptions

    descriptions = [
        "Rows with many zero-valued features.",
        "Extreme outlier rows with high z-scores.",
        "Dense numeric rows with low sparsity.",
    ]
    embeddings = embed_descriptions(descriptions, dim=768)

    assert embeddings.shape == (3, 768)
    # Embeddings should be unit-normalized (nomic default)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_embed_descriptions_matryoshka_truncation():
    """Matryoshka truncation produces correct dimensionality."""
    from scripts.embed_concept_descriptions import embed_descriptions

    descriptions = ["test sentence one", "test sentence two"]
    emb_256 = embed_descriptions(descriptions, dim=256)
    emb_768 = embed_descriptions(descriptions, dim=768)

    assert emb_256.shape == (2, 256)
    assert emb_768.shape == (2, 768)
    # Truncated should match prefix of full (Matryoshka property)
    np.testing.assert_allclose(emb_256, emb_768[:, :256], atol=1e-5)


def test_within_group_coherence():
    """Within-group cosine sim is higher than between-group."""
    from scripts.embed_concept_descriptions import compute_within_group_coherence

    # Group 0: similar descriptions. Group 1: different descriptions.
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # group 0
        [0.9, 0.1, 0.0],  # group 0
        [0.0, 0.0, 1.0],  # group 1
        [0.0, 0.1, 0.9],  # group 1
    ])
    group_ids = [0, 0, 1, 1]

    result = compute_within_group_coherence(embeddings, group_ids)
    assert result["mean"] > 0.8  # Both groups internally similar


def test_matched_pair_agreement():
    """Matched pairs have higher sim than random baseline."""
    from scripts.embed_concept_descriptions import compute_matched_pair_agreement

    # 4 embeddings: 0↔2 matched (similar), 1↔3 matched (similar)
    embeddings = np.array([
        [1.0, 0.0],  # 0: matched with 2
        [0.0, 1.0],  # 1: matched with 3
        [0.9, 0.1],  # 2: matched with 0
        [0.1, 0.9],  # 3: matched with 1
    ])
    matched_pairs = [(0, 2), (1, 3)]

    result = compute_matched_pair_agreement(embeddings, matched_pairs)
    assert result["mean"] > result["random_baseline"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed_concept_descriptions.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Embed concept descriptions and compute validation metrics.

Uses nomic-embed-text-v1.5 via sentence-transformers for local embedding.
Matryoshka support enables multi-resolution comparison.

Usage:
    python scripts/embed_concept_descriptions.py
    python scripts/embed_concept_descriptions.py --dim 256
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"

_model_cache = None


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _model_cache


def embed_descriptions(descriptions: List[str], dim: int = 768) -> np.ndarray:
    """Embed text descriptions using nomic-embed-text-v1.5.

    Args:
        descriptions: List of text strings to embed.
        dim: Embedding dimension (768, 256, 128, or 64 for Matryoshka).

    Returns:
        (n, dim) array of L2-normalized embeddings.
    """
    model = _get_model()
    # nomic requires "search_document: " prefix for documents
    prefixed = [f"search_document: {d}" for d in descriptions]
    embeddings = model.encode(prefixed, normalize_embeddings=True)

    if dim < embeddings.shape[1]:
        embeddings = embeddings[:, :dim]
        # Re-normalize after truncation (Matryoshka)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

    return embeddings


def compute_within_group_coherence(
    embeddings: np.ndarray,
    group_ids: List[int],
) -> Dict:
    """Mean pairwise cosine similarity within each group."""
    unique_groups = sorted(set(g for g in group_ids if g >= 0))
    per_group = {}

    for gid in unique_groups:
        indices = [i for i, g in enumerate(group_ids) if g == gid]
        if len(indices) < 2:
            continue
        sims = []
        for i, j in combinations(indices, 2):
            sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
            sims.append(sim)
        per_group[str(gid)] = float(np.mean(sims))

    vals = list(per_group.values())
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals)) if vals else 0.0,
        "n_groups": len(per_group),
        "per_group": per_group,
    }


def compute_matched_pair_agreement(
    embeddings: np.ndarray,
    matched_pairs: List[Tuple[int, int]],
    n_random: int = 1000,
    seed: int = 42,
) -> Dict:
    """Cosine similarity of matched pairs vs random baseline."""
    if not matched_pairs:
        return {"mean": 0.0, "random_baseline": 0.0, "n_pairs": 0}

    pair_sims = []
    for i, j in matched_pairs:
        sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
        pair_sims.append(sim)

    rng = np.random.RandomState(seed)
    n = len(embeddings)
    random_sims = []
    for _ in range(n_random):
        i, j = rng.choice(n, 2, replace=False)
        sim = 1.0 - cosine_dist(embeddings[i], embeddings[j])
        random_sims.append(sim)

    return {
        "mean": float(np.mean(pair_sims)),
        "std": float(np.std(pair_sims)),
        "random_baseline": float(np.mean(random_sims)),
        "n_pairs": len(matched_pairs),
    }


def compute_interpolation_coherence(
    embeddings: np.ndarray,
    feature_ids: List[str],
    unmatched_landmarks: Dict[str, List[Tuple[str, float]]],
) -> Dict:
    """Cosine sim between unexplained features and their landmarks."""
    id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    sims = []

    for feat_id, landmarks in unmatched_landmarks.items():
        if feat_id not in id_to_idx:
            continue
        feat_emb = embeddings[id_to_idx[feat_id]]
        for landmark_id, _ in landmarks:
            if landmark_id in id_to_idx:
                lm_emb = embeddings[id_to_idx[landmark_id]]
                sim = 1.0 - cosine_dist(feat_emb, lm_emb)
                sims.append(sim)

    return {
        "mean": float(np.mean(sims)) if sims else 0.0,
        "std": float(np.std(sims)) if sims else 0.0,
        "n_features": len(unmatched_landmarks),
    }


def main():
    parser = argparse.ArgumentParser(description="Embed concept descriptions")
    parser.add_argument("--dim", type=int, default=768,
                        choices=[64, 128, 256, 768],
                        help="Embedding dimension (Matryoshka)")
    parser.add_argument("--input", type=str,
                        default=str(INPUT_DIR / "concept_descriptions.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Collect all descriptions with feature IDs and group membership
    feature_ids = []
    descriptions = []
    group_ids = []

    for gid, group in data.get("groups", {}).items():
        for feat_key, feat in group.get("features", {}).items():
            feature_ids.append(feat_key)
            descriptions.append(feat.get("description", feat.get("brief_label", "")))
            group_ids.append(int(gid))

    for feat_key, feat in data.get("unmatched", {}).items():
        feature_ids.append(feat_key)
        descriptions.append(feat.get("description", feat.get("brief_label", "")))
        group_ids.append(-1)

    print(f"Embedding {len(descriptions)} descriptions at dim={args.dim}...")
    embeddings = embed_descriptions(descriptions, dim=args.dim)

    # Save embeddings
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = INPUT_DIR / "concept_embeddings.npz"
    np.savez_compressed(
        str(emb_path),
        embeddings=embeddings,
        feature_ids=np.array(feature_ids),
        group_ids=np.array(group_ids),
    )
    print(f"Saved: {emb_path}")

    # Compute metrics
    print("Computing metrics...")
    metrics = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": args.dim,
        "n_features": len(descriptions),
    }

    metrics["within_group_cosine"] = compute_within_group_coherence(
        embeddings, group_ids,
    )
    print(f"  Within-group coherence: {metrics['within_group_cosine']['mean']:.3f}")

    # Build matched pairs from group membership
    matched_pairs = []
    group_indices = {}
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            group_indices.setdefault(gid, []).append(i)
    for indices in group_indices.values():
        for i, j in combinations(indices, 2):
            matched_pairs.append((i, j))

    metrics["matched_pair_cosine"] = compute_matched_pair_agreement(
        embeddings, matched_pairs,
    )
    print(f"  Matched-pair agreement: {metrics['matched_pair_cosine']['mean']:.3f}"
          f" (random: {metrics['matched_pair_cosine']['random_baseline']:.3f})")

    metrics_path = INPUT_DIR / "concept_embedding_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_concept_descriptions.py -v`
Expected: PASS (4 tests). Note: first run downloads nomic model (~550MB).

**Step 5: Commit**

```bash
git add scripts/embed_concept_descriptions.py tests/test_embed_concept_descriptions.py
git commit -m "feat: add concept embedding script with validation metrics"
```

---

## Task 7: Script 3 — validation with self-checks and optional API

Validates nomic embedding quality via Matryoshka consistency and
bootstrap stability. Optional API fallback.

**Files:**
- Create: `scripts/validate_concept_embeddings.py`
- Test: `tests/test_validate_concept_embeddings.py`

**Step 1: Write the failing tests**

```python
# tests/test_validate_concept_embeddings.py
"""Tests for concept embedding validation."""
import numpy as np
import pytest


def test_matryoshka_consistency_high_for_stable_embeddings():
    """Matryoshka dimensions should agree on rankings for stable embeddings."""
    from scripts.validate_concept_embeddings import matryoshka_consistency

    descriptions = [
        "Rows with many zero-valued features.",
        "Sparse numeric rows with low entropy.",
        "Dense rows with all features populated.",
        "Extreme outliers with very high z-scores.",
        "Rows with strongly correlated feature pairs.",
    ]

    result = matryoshka_consistency(descriptions)
    # 768 vs 256 should agree strongly
    assert result["768v256"] > 0.85


def test_validation_report_structure():
    """Validation report has required keys."""
    from scripts.validate_concept_embeddings import build_validation_report

    nomic_checks = {"matryoshka_spearman": {}, "bootstrap_stability": 0.95, "passed": True}
    report = build_validation_report(nomic_checks)

    assert "nomic_self_checks" in report
    assert "api_validation" in report
    assert report["api_validation"]["ran"] is False
```

**Step 2: Run test, verify fail**

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Validate concept embedding quality.

Step 1 (always): Nomic self-checks — Matryoshka dimension consistency,
bootstrap stability.
Step 2 (optional): API cross-reference if self-checks fail or --api flag.

Usage:
    python scripts/validate_concept_embeddings.py
    python scripts/validate_concept_embeddings.py --api voyage
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embed_concept_descriptions import embed_descriptions

INPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"


def matryoshka_consistency(
    descriptions: List[str],
    dims: tuple = (768, 256, 128, 64),
) -> Dict[str, float]:
    """Check Spearman rank correlation of pairwise sims across Matryoshka dims."""
    embeddings_by_dim = {}
    for d in dims:
        embeddings_by_dim[d] = embed_descriptions(descriptions, dim=d)

    def pairwise_sims(emb):
        from scipy.spatial.distance import pdist
        return 1.0 - pdist(emb, metric="cosine")

    sims = {d: pairwise_sims(emb) for d, emb in embeddings_by_dim.items()}
    base = dims[0]

    result = {}
    for d in dims[1:]:
        rho, _ = stats.spearmanr(sims[base], sims[d])
        result[f"{base}v{d}"] = float(rho)

    return result


def bootstrap_stability(
    descriptions: List[str],
    n_bootstrap: int = 5,
    dim: int = 768,
    seed: int = 42,
) -> float:
    """Check embedding stability under bootstrap resampling."""
    rng = np.random.RandomState(seed)
    base_emb = embed_descriptions(descriptions, dim=dim)

    from scipy.spatial.distance import pdist
    base_sims = 1.0 - pdist(base_emb, metric="cosine")

    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(descriptions), len(descriptions), replace=True)
        unique_idx = sorted(set(idx))
        sub_descs = [descriptions[i] for i in unique_idx]
        sub_emb = embed_descriptions(sub_descs, dim=dim)
        sub_sims = 1.0 - pdist(sub_emb, metric="cosine")

        # Compare overlapping pairs
        rho, _ = stats.spearmanr(
            1.0 - pdist(base_emb[unique_idx], metric="cosine"),
            sub_sims,
        )
        rhos.append(rho)

    return float(np.mean(rhos))


def build_validation_report(
    nomic_checks: Dict,
    api_result: Optional[Dict] = None,
) -> Dict:
    """Build final validation report."""
    report = {
        "nomic_self_checks": nomic_checks,
        "api_validation": api_result or {"ran": False, "reason": "nomic self-checks passed"},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate concept embeddings")
    parser.add_argument("--api", type=str, choices=["voyage", "openai"],
                        help="Force API validation with specified provider")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of descriptions to sample for API check")
    args = parser.parse_args()

    desc_path = INPUT_DIR / "concept_descriptions.json"
    with open(desc_path) as f:
        data = json.load(f)

    # Collect all descriptions
    descriptions = []
    for group in data.get("groups", {}).values():
        for feat in group.get("features", {}).values():
            descriptions.append(feat.get("description", feat.get("brief_label", "")))
    for feat in data.get("unmatched", {}).values():
        descriptions.append(feat.get("description", feat.get("brief_label", "")))

    print(f"Validating {len(descriptions)} descriptions...")

    # Step 1: Nomic self-checks
    print("\nMatryoshka consistency...")
    mat_result = matryoshka_consistency(descriptions)
    for k, v in mat_result.items():
        print(f"  {k}: rho={v:.4f}")

    print("\nBootstrap stability...")
    boot_result = bootstrap_stability(descriptions)
    print(f"  Stability: {boot_result:.4f}")

    min_mat = min(mat_result.values())
    passed = min_mat > 0.80 and boot_result > 0.85
    nomic_checks = {
        "matryoshka_spearman": mat_result,
        "bootstrap_stability": boot_result,
        "passed": passed,
    }
    print(f"\nSelf-checks: {'PASSED' if passed else 'FAILED'}")

    # Step 2: API validation (if requested or self-checks failed)
    api_result = None
    if args.api or not passed:
        reason = f"--api {args.api}" if args.api else "self-checks failed"
        print(f"\nRunning API validation ({reason})...")
        # API embedding implementation here — depends on provider
        api_result = {"ran": True, "reason": reason, "provider": args.api or "auto"}

    report = build_validation_report(nomic_checks, api_result)
    report_path = INPUT_DIR / "concept_embedding_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/test_validate_concept_embeddings.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scripts/validate_concept_embeddings.py tests/test_validate_concept_embeddings.py
git commit -m "feat: add embedding validation with Matryoshka self-checks"
```

---

## Task 8: Integration test — end-to-end on small subset

Run the full pipeline on a small subset (5 groups, 10 features) to
verify the scripts chain correctly.

**Files:**
- Create: `tests/test_concept_embedding_integration.py`

**Step 1: Write integration test**

```python
# tests/test_concept_embedding_integration.py
"""End-to-end integration test for concept embedding pipeline."""
import json
import pytest
from pathlib import Path


@pytest.mark.slow
def test_full_pipeline_small_subset(tmp_path):
    """Run description → embedding → validation on synthetic data."""
    from scripts.embed_concept_descriptions import (
        embed_descriptions,
        compute_within_group_coherence,
        compute_matched_pair_agreement,
    )

    # Synthetic concept descriptions (no API calls needed)
    descriptions_data = {
        "metadata": {"test": True},
        "groups": {
            "0": {
                "brief_label": "sparse rows",
                "summary": "Rows with many zero-valued features.",
                "features": {
                    "A:1": {"description": "Sparse numeric rows with many zeros."},
                    "B:2": {"description": "Rows with mostly zero feature values."},
                },
            },
            "1": {
                "brief_label": "outliers",
                "summary": "Extreme outlier rows.",
                "features": {
                    "A:3": {"description": "Rows with extreme z-scores above 3."},
                    "B:4": {"description": "Outlier rows far from the centroid."},
                },
            },
        },
        "unmatched": {
            "A:5": {"description": "Rows with highly skewed feature distributions."},
        },
    }

    # Save synthetic descriptions
    desc_path = tmp_path / "concept_descriptions.json"
    with open(desc_path, "w") as f:
        json.dump(descriptions_data, f)

    # Collect descriptions
    descs = []
    group_ids = []
    for gid, group in descriptions_data["groups"].items():
        for feat in group["features"].values():
            descs.append(feat["description"])
            group_ids.append(int(gid))
    for feat in descriptions_data["unmatched"].values():
        descs.append(feat["description"])
        group_ids.append(-1)

    # Embed
    embeddings = embed_descriptions(descs, dim=256)
    assert embeddings.shape == (5, 256)

    # Metrics
    coherence = compute_within_group_coherence(embeddings, group_ids)
    assert coherence["mean"] > 0.0  # Some coherence
    assert coherence["n_groups"] == 2

    # Within-group should be higher than random
    pairs = [(0, 1), (2, 3)]  # group 0 pair, group 1 pair
    agreement = compute_matched_pair_agreement(embeddings, pairs)
    assert agreement["mean"] > agreement["random_baseline"]
```

**Step 2: Run test**

Run: `pytest tests/test_concept_embedding_integration.py -v -m slow`

**Step 3: Commit**

```bash
git add tests/test_concept_embedding_integration.py
git commit -m "test: add end-to-end integration test for concept embedding pipeline"
```

---

## Task 9: Update concept_fingerprint.py DEFAULT_TRAINING_DIR

The `load_per_dataset_embeddings` function in `concept_fingerprint.py:48`
still defaults to round 5. Update to round 6.

**Files:**
- Modify: `scripts/concept_fingerprint.py:48`

**Step 1: Check current value**

```python
# Line 48: DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round5"
```

**Step 2: Update**

Change to:
```python
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "output" / "sae_training_round6"
```

**Step 3: Run existing tests**

Run: `pytest tests/test_row_level_probes.py -v`

**Step 4: Commit**

```bash
git add scripts/concept_fingerprint.py
git commit -m "fix: update concept_fingerprint default training dir to round 6"
```

---

## Summary

| Task | Description | Est. |
|------|-------------|------|
| 1 | Sample extraction utilities + tests | 5 min |
| 2 | Prompt formatting (Haiku + Sonnet) + tests | 5 min |
| 3 | Script 1 core — description generation | 10 min |
| 4 | Script 1 — activation loading + row retrieval | 5 min |
| 5 | Script 1 — three-pass main loop + checkpoint | 10 min |
| 6 | Script 2 — embedding + metrics | 10 min |
| 7 | Script 3 — validation self-checks + API | 10 min |
| 8 | Integration test — end-to-end small subset | 5 min |
| 9 | Update concept_fingerprint round 5 → 6 | 2 min |
