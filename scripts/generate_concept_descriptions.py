#!/usr/bin/env python3
"""Generate rich concept descriptions via LLM (Haiku brief + Sonnet detailed).

Three-pass pipeline:
  Pass 1 (Haiku):  Brief 2-5 word labels for all concept groups.
  Pass 2 (Sonnet): Rich 1-2 sentence descriptions for grouped concepts,
                    using probe consensus and activating/non-activating samples.
  Pass 3 (Sonnet): Descriptions for unexplained features using activating
                    samples and nearest-described-neighbor landmarks.

Supports checkpoint/resume: every group/feature result is saved incrementally
so interrupted runs can pick up where they left off.

Usage:
    # Run all three passes
    python scripts/generate_concept_descriptions.py --device cpu

    # Run only pass 1 (Haiku labels)
    python scripts/generate_concept_descriptions.py --pass 1

    # Resume interrupted run
    python scripts/generate_concept_descriptions.py --resume

    # Limit activating samples per feature
    python scripts/generate_concept_descriptions.py --max-samples 10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.concept_description_utils import (
    format_haiku_prompt,
    format_sonnet_group_prompt,
    format_sonnet_unexplained_prompt,
    get_activating_samples,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

OUTPUT_DIR = PROJECT_ROOT / "output" / "concept_descriptions"
LABELS_PATH = PROJECT_ROOT / "output" / "cross_model_concept_labels_v2.json"
PROBES_PATH = PROJECT_ROOT / "output" / "sae_concept_analysis_round8.json"
CROSS_CORR_DIR = PROJECT_ROOT / "output" / "sae_cross_correlations"

# Display name -> checkpoint directory key
LABEL_KEY_TO_MODEL_KEY = {
    "TabPFN": "tabpfn",
    "Mitra": "mitra",
    "TabICL": "tabicl",
    "TabICL-v2": "tabicl_v2",
    "TabDPT": "tabdpt",
    "HyperFast": "hyperfast",
    "CARTE": "carte",
    "Tabula-8B": "tabula8b",
}


# ---------------------------------------------------------------------------
# Output skeleton
# ---------------------------------------------------------------------------

def build_output_skeleton() -> dict:
    """Create empty output dict with metadata."""
    return {
        "metadata": {
            "haiku_model": HAIKU_MODEL,
            "sonnet_model": SONNET_MODEL,
            "labels_path": str(LABELS_PATH),
            "probes_path": str(PROBES_PATH),
            "created": datetime.now(timezone.utc).isoformat(),
            "n_haiku_calls": 0,
            "n_sonnet_calls": 0,
        },
        "groups": {},
        "unmatched": {},
    }


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

def call_llm(
    client,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 256,
) -> Optional[str]:
    """Call Anthropic API with error handling. Returns text or None on failure."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("LLM call failed (%s): %s", model, e)
        return None


# ---------------------------------------------------------------------------
# Group description helper
# ---------------------------------------------------------------------------

def describe_group(
    group_id: int,
    group: dict,
    high_rows: List[dict],
    low_rows: List[dict],
    client,
    model: str,
    per_member_detail: Optional[List[dict]] = None,
) -> Optional[str]:
    """Generate a Sonnet description for one concept group.

    Args:
        group_id: Numeric group identifier.
        group: Group dict with members, n_models, top_probes.
        high_rows: Dicts of activating row data.
        low_rows: Dicts of non-activating row data.
        client: Anthropic client.
        model: Sonnet model name.
        per_member_detail: Optional per-member probe detail.

    Returns:
        Description string, or None on API failure.
    """
    probes = [tuple(p) for p in group.get("top_probes", [])]
    n_models = group.get("n_models", 0)
    n_members = len(group.get("members", []))

    prompt = format_sonnet_group_prompt(
        group_id=group_id,
        probes=probes,
        n_models=n_models,
        n_members=n_members,
        high_rows=high_rows,
        low_rows=low_rows,
        per_member_detail=per_member_detail,
    )
    return call_llm(client, model, SONNET_SYSTEM, prompt, max_tokens=512)


# ---------------------------------------------------------------------------
# Activation loading with cache
# ---------------------------------------------------------------------------

# Module-level cache for loaded activations: model_key -> np.ndarray
_activation_cache: Dict[str, np.ndarray] = {}


def load_activations_cached(
    model_key: str,
    sae_dir: Path,
    training_dir: Path,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Load SAE, encode training embeddings, cache result.

    Returns:
        (n_samples, hidden_dim) activation matrix, or None on failure.
    """
    if model_key in _activation_cache:
        return _activation_cache[model_key]

    try:
        import torch
        from scripts.intervene_sae import load_sae, load_training_mean

        sae, config = load_sae(model_key, sae_dir=sae_dir, device=device)

        # Load raw training embeddings
        from scripts.intervene_sae import get_extraction_layer
        layer = get_extraction_layer(model_key)
        training_path = training_dir / f"{model_key}_layer{layer}_sae_training.npz"
        if not training_path.exists():
            logger.warning("Training data not found: %s", training_path)
            return None

        data = np.load(training_path, allow_pickle=True)
        embeddings = data["embeddings"]

        # Center embeddings (SAE trained on centered data)
        emb_mean = embeddings.mean(axis=0)
        embeddings_centered = embeddings - emb_mean

        # Encode through SAE in batches
        batch_size = 1024
        all_acts = []
        with torch.no_grad():
            for i in range(0, len(embeddings_centered), batch_size):
                batch = torch.tensor(
                    embeddings_centered[i : i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                acts = sae.encode(batch)
                all_acts.append(acts.cpu().numpy())

        activations = np.concatenate(all_acts, axis=0)
        _activation_cache[model_key] = activations
        logger.info(
            "Cached activations for %s: shape %s", model_key, activations.shape
        )
        return activations

    except Exception as e:
        logger.warning("Failed to load activations for %s: %s", model_key, e)
        return None


def _embedding_rows_as_dicts(
    model_key: str,
    indices: List[int],
    training_dir: Path,
) -> List[dict]:
    """Convert embedding rows at given indices to dicts.

    TODO: Load original tabular data for richer row content. For now we use
    raw embedding dimensions as a proxy.
    """
    try:
        from scripts.intervene_sae import get_extraction_layer

        layer = get_extraction_layer(model_key)
        training_path = training_dir / f"{model_key}_layer{layer}_sae_training.npz"
        data = np.load(training_path, allow_pickle=True)
        embeddings = data["embeddings"]

        rows = []
        for idx in indices:
            if idx < len(embeddings):
                row = embeddings[idx]
                # Use top-10 dimensions with highest absolute value
                top_dims = np.argsort(np.abs(row))[-10:][::-1]
                rows.append({f"dim_{d}": round(float(row[d]), 4) for d in top_dims})
        return rows
    except Exception as e:
        logger.warning("Failed to load embedding rows for %s: %s", model_key, e)
        return []


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(output: dict, path: Path) -> None:
    """Save output dict as JSON checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    tmp.rename(path)


def load_checkpoint(path: Path) -> dict:
    """Load output dict from JSON checkpoint."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pass 1: Haiku brief labels
# ---------------------------------------------------------------------------

def run_pass1(
    client,
    labels: dict,
    probes_data: dict,
    output: dict,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> int:
    """Generate Haiku brief labels for all concept groups.

    Args:
        client: Anthropic API client.
        labels: Loaded cross_model_concept_labels_v2.json.
        probes_data: Loaded sae_concept_analysis JSON.
        output: Output dict (mutated in place).
        checkpoint_path: Path to save incremental progress.
        checkpoint_every: Save checkpoint every N groups.

    Returns:
        Number of new labels generated.
    """
    concept_groups = labels["concept_groups"]
    n_generated = 0

    for gid, group in concept_groups.items():
        # Skip already-completed groups
        if gid in output["groups"] and output["groups"][gid].get("brief_label"):
            continue

        probes = [tuple(p) for p in group.get("top_probes", [])]
        n_models = group.get("n_models", 0)
        members = group.get("members", [])
        n_members = len(members)

        # Build per-member detail from probes_data
        per_member_detail = _build_per_member_detail(members, probes_data)

        prompt = format_haiku_prompt(
            group_id=int(gid),
            probes=probes,
            n_models=n_models,
            n_members=n_members,
            per_member_detail=per_member_detail,
        )

        label = call_llm(client, HAIKU_MODEL, HAIKU_SYSTEM, prompt, max_tokens=64)

        if gid not in output["groups"]:
            output["groups"][gid] = {}
        output["groups"][gid]["brief_label"] = label
        output["groups"][gid]["n_models"] = n_models
        output["groups"][gid]["n_members"] = n_members

        # Populate per-feature entries (design schema expects features dict)
        if "features" not in output["groups"][gid]:
            output["groups"][gid]["features"] = {}
        for model_name, feat_idx_raw in members:
            feat_key = f"{model_name}:{feat_idx_raw}"
            if feat_key not in output["groups"][gid]["features"]:
                output["groups"][gid]["features"][feat_key] = {}
            output["groups"][gid]["features"][feat_key]["brief_label"] = label

        output["metadata"]["n_haiku_calls"] += 1
        n_generated += 1

        if n_generated % checkpoint_every == 0:
            logger.info("Pass 1: %d/%d groups labeled", n_generated, len(concept_groups))
            if checkpoint_path:
                save_checkpoint(output, checkpoint_path)

    return n_generated


def _build_per_member_detail(
    members: List[list],
    probes_data: dict,
    top_k: int = 4,
) -> List[dict]:
    """Build per-member probe detail for prompt formatting."""
    detail = []
    for model_name, feat_idx in members:
        feat_idx = int(feat_idx)
        model_probes = probes_data.get("models", {}).get(model_name, {})
        per_feature = model_probes.get("per_feature", {})
        feat_data = per_feature.get(str(feat_idx), {})

        if feat_data:
            top_probes = feat_data.get("top_probes", [])[:top_k]
            detail.append({
                "model": model_name,
                "feature_idx": feat_idx,
                "r2": feat_data.get("r2", 0.0),
                "top_probes": [(p[0], float(p[1])) for p in top_probes],
            })
    return detail


# ---------------------------------------------------------------------------
# Pass 2: Sonnet grouped descriptions with activating samples
# ---------------------------------------------------------------------------

def run_pass2(
    client,
    labels: dict,
    probes_data: dict,
    output: dict,
    args: argparse.Namespace,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> int:
    """Generate Sonnet descriptions for concept groups.

    For each group, picks the highest-R2 member to get activating samples,
    then calls Sonnet with probe consensus + sample rows.

    Returns:
        Number of new descriptions generated.
    """
    concept_groups = labels["concept_groups"]
    n_generated = 0
    max_samples = getattr(args, "max_samples", 5)
    device = getattr(args, "device", "cpu")
    training_dir = Path(getattr(args, "training_dir", PROJECT_ROOT / "output" / "sae_training_round6"))

    for gid, group in concept_groups.items():
        # Skip already-completed groups
        if gid in output["groups"] and output["groups"][gid].get("summary"):
            continue

        members = group.get("members", [])
        if not members:
            continue

        # Find highest-R2 member
        best_member = _find_best_member(members, probes_data)
        if best_member is None:
            best_member = (members[0][0], int(members[0][1]))

        model_name, feat_idx = best_member
        model_key = LABEL_KEY_TO_MODEL_KEY.get(model_name)
        if model_key is None:
            logger.warning("Unknown model name: %s", model_name)
            continue

        # Determine SAE dir: try round 6 first, fall back to round 5
        sae_dir = _resolve_sae_dir(model_key)

        # Load activations
        activations = load_activations_cached(
            model_key, sae_dir, training_dir, device=device
        )

        high_rows, low_rows = [], []
        if activations is not None:
            high_idx, low_idx = get_activating_samples(
                activations, feat_idx, top_k=max_samples, bottom_k=max_samples
            )
            high_rows = _embedding_rows_as_dicts(model_key, high_idx, training_dir)
            low_rows = _embedding_rows_as_dicts(model_key, low_idx, training_dir)

        per_member_detail = _build_per_member_detail(members, probes_data)

        desc = describe_group(
            group_id=int(gid),
            group=group,
            high_rows=high_rows,
            low_rows=low_rows,
            client=client,
            model=SONNET_MODEL,
            per_member_detail=per_member_detail,
        )

        if gid not in output["groups"]:
            output["groups"][gid] = {}
        output["groups"][gid]["summary"] = desc
        output["groups"][gid]["sample_model"] = model_name
        output["groups"][gid]["sample_feature"] = feat_idx

        # Populate per-feature descriptions (same as group summary for now;
        # could be refined with per-member Sonnet calls if needed)
        if "features" not in output["groups"][gid]:
            output["groups"][gid]["features"] = {}
        for m_name, m_idx in members:
            feat_key = f"{m_name}:{m_idx}"
            if feat_key not in output["groups"][gid]["features"]:
                output["groups"][gid]["features"][feat_key] = {}
            output["groups"][gid]["features"][feat_key]["description"] = desc

        output["metadata"]["n_sonnet_calls"] += 1
        n_generated += 1

        if checkpoint_path and n_generated % checkpoint_every == 0 and n_generated > 0:
            save_checkpoint(output, checkpoint_path)
        if n_generated % 20 == 0:
            logger.info("Pass 2: %d groups described", n_generated)

    return n_generated


def _find_best_member(
    members: List[list],
    probes_data: dict,
) -> Optional[Tuple[str, int]]:
    """Find the group member with highest R2 in probe regressions."""
    best_r2 = -1.0
    best = None
    for model_name, feat_idx in members:
        feat_idx = int(feat_idx)
        model_probes = probes_data.get("models", {}).get(model_name, {})
        per_feature = model_probes.get("per_feature", {})
        feat_data = per_feature.get(str(feat_idx), {})
        r2 = feat_data.get("r2", 0.0)
        if r2 > best_r2:
            best_r2 = r2
            best = (model_name, feat_idx)
    return best


def _resolve_sae_dir(model_key: str) -> Path:
    """Return the SAE sweep directory for a model, preferring round 6."""
    from scripts.compare_sae_cross_model import sae_sweep_dir

    round6 = sae_sweep_dir(6)
    ckpt = round6 / model_key / "sae_matryoshka_archetypal_validated.pt"
    if ckpt.exists():
        return round6

    round5 = sae_sweep_dir(5)
    ckpt5 = round5 / model_key / "sae_matryoshka_archetypal_validated.pt"
    if ckpt5.exists():
        return round5

    # Fallback to round 6 (will raise FileNotFoundError in load_sae)
    return round6


# ---------------------------------------------------------------------------
# Pass 3: Sonnet unexplained features with landmarks
# ---------------------------------------------------------------------------

def run_pass3(
    client,
    labels: dict,
    output: dict,
    args: argparse.Namespace,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> int:
    """Generate Sonnet descriptions for unexplained unmatched features.

    Uses nearest described groups as landmarks via cross-correlation files.

    Returns:
        Number of new descriptions generated.
    """
    unmatched = labels.get("unmatched_features", {}).get("unexplained", {})
    if not unmatched:
        logger.info("Pass 3: no unexplained features to describe.")
        return 0

    max_samples = getattr(args, "max_samples", 5)
    device = getattr(args, "device", "cpu")
    training_dir = Path(getattr(args, "training_dir", PROJECT_ROOT / "output" / "sae_training_round6"))
    n_generated = 0

    for model_name, features in unmatched.items():
        model_key = LABEL_KEY_TO_MODEL_KEY.get(model_name)
        if model_key is None:
            logger.warning("Unknown model name: %s", model_name)
            continue

        sae_dir = _resolve_sae_dir(model_key)
        activations = load_activations_cached(
            model_key, sae_dir, training_dir, device=device
        )

        for feat_idx_str, feat_info in features.items():
            feat_key = f"{model_name}:{feat_idx_str}"

            # Skip already-completed
            if feat_key in output["unmatched"] and output["unmatched"][feat_key].get("description"):
                continue

            feat_idx = int(feat_idx_str)

            high_rows, low_rows = [], []
            if activations is not None:
                high_idx, low_idx = get_activating_samples(
                    activations, feat_idx, top_k=max_samples, bottom_k=max_samples
                )
                high_rows = _embedding_rows_as_dicts(model_key, high_idx, training_dir)
                low_rows = _embedding_rows_as_dicts(model_key, low_idx, training_dir)

            # Try to find nearest described landmarks
            landmarks = _find_landmarks(model_name, feat_idx, labels, output)

            prompt = format_sonnet_unexplained_prompt(
                model=model_name,
                feat_idx=feat_idx,
                high_rows=high_rows,
                low_rows=low_rows,
                landmarks=landmarks,
            )

            desc = call_llm(client, SONNET_MODEL, SONNET_SYSTEM, prompt, max_tokens=512)

            output["unmatched"][feat_key] = {
                "model": model_name,
                "feature_idx": feat_idx,
                "r2": feat_info.get("r2", 0.0),
                "description": desc,
                "n_landmarks": len(landmarks) if landmarks else 0,
            }
            output["metadata"]["n_sonnet_calls"] += 1
            n_generated += 1

            if checkpoint_path and n_generated % checkpoint_every == 0 and n_generated > 0:
                save_checkpoint(output, checkpoint_path)
            if n_generated % 50 == 0:
                logger.info("Pass 3: %d unexplained features described", n_generated)

    return n_generated


def _find_landmarks(
    model_name: str,
    feat_idx: int,
    labels: dict,
    output: dict,
    top_n: int = 5,
) -> Optional[List[Tuple[str, float]]]:
    """Find nearest described concept groups as landmarks.

    Uses cross-correlation files to find features in other models that are
    correlated with this feature, then maps them to described groups.

    Returns:
        List of (description, correlation) tuples, or None if unavailable.
    """
    # Look for same-model features in described groups
    feature_lookup = labels.get("feature_lookup", {}).get(model_name, {})
    if not feature_lookup:
        return None

    # Find all features from this model that are in concept groups
    described = []
    for other_feat_str, info in feature_lookup.items():
        other_feat = int(other_feat_str)
        if other_feat == feat_idx:
            continue
        gid = str(info.get("group_id", ""))
        if gid in output.get("groups", {}):
            group_data = output["groups"][gid]
            group_desc = group_data.get("summary") or group_data.get("brief_label")
            if group_desc:
                described.append((other_feat, gid, group_desc))

    if not described:
        return None

    # If we have cross-correlation data, use it to rank
    # Otherwise just return first few described features
    model_key = LABEL_KEY_TO_MODEL_KEY.get(model_name)
    if model_key is None:
        return [(desc, 0.0) for _, _, desc in described[:top_n]]

    # Try loading within-model correlation from activation cache
    if model_key in _activation_cache:
        acts = _activation_cache[model_key]
        feat_acts = acts[:, feat_idx]
        feat_std = feat_acts.std()
        if feat_std > 1e-8:
            correlations = []
            for other_feat, gid, desc in described:
                if other_feat < acts.shape[1]:
                    other_acts = acts[:, other_feat]
                    other_std = other_acts.std()
                    if other_std > 1e-8:
                        r = np.corrcoef(feat_acts, other_acts)[0, 1]
                        correlations.append((desc, float(r)))
            if correlations:
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                return correlations[:top_n]

    return [(desc, 0.0) for _, _, desc in described[:top_n]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM concept descriptions (Haiku + Sonnet)."
    )
    parser.add_argument(
        "--pass",
        dest="passes",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which passes to run (1=Haiku labels, 2=Sonnet groups, 3=Sonnet unexplained). "
        "Default: all three.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max activating/non-activating rows per feature (default: 5).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for SAE encoding (default: cpu, inference only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N groups (default: 50).",
    )
    parser.add_argument(
        "--training-dir",
        default=str(PROJECT_ROOT / "output" / "sae_training_round6"),
        help="Directory with SAE training data (.npz files).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load input data
    logger.info("Loading labels from %s", LABELS_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    logger.info("Loading probes from %s", PROBES_PATH)
    with open(PROBES_PATH) as f:
        probes_data = json.load(f)

    # Load or create output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "descriptions_checkpoint.json"
    final_path = OUTPUT_DIR / "concept_descriptions.json"

    if args.resume and checkpoint_path.exists():
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        output = load_checkpoint(checkpoint_path)
    else:
        output = build_output_skeleton()

    # Create Anthropic client
    import anthropic
    client = anthropic.Anthropic()

    t0 = time.time()

    ckpt_every = args.checkpoint_every

    # Pass 1: Haiku brief labels
    if 1 in args.passes:
        logger.info("=== Pass 1: Haiku brief labels ===")
        n = run_pass1(client, labels, probes_data, output, checkpoint_path, ckpt_every)
        logger.info("Pass 1 complete: %d new labels", n)
        save_checkpoint(output, checkpoint_path)

    # Pass 2: Sonnet grouped descriptions
    if 2 in args.passes:
        logger.info("=== Pass 2: Sonnet group descriptions ===")
        n = run_pass2(client, labels, probes_data, output, args, checkpoint_path, ckpt_every)
        logger.info("Pass 2 complete: %d new descriptions", n)
        save_checkpoint(output, checkpoint_path)

    # Pass 3: Sonnet unexplained features
    if 3 in args.passes:
        logger.info("=== Pass 3: Sonnet unexplained features ===")
        n = run_pass3(client, labels, output, args, checkpoint_path, ckpt_every)
        logger.info("Pass 3 complete: %d new descriptions", n)
        save_checkpoint(output, checkpoint_path)

    elapsed = time.time() - t0
    output["metadata"]["runtime_seconds"] = round(elapsed, 1)
    output["metadata"]["completed"] = datetime.now(timezone.utc).isoformat()

    # Save final output
    save_checkpoint(output, final_path)
    logger.info(
        "Done in %.1fs. Haiku calls: %d, Sonnet calls: %d. Output: %s",
        elapsed,
        output["metadata"]["n_haiku_calls"],
        output["metadata"]["n_sonnet_calls"],
        final_path,
    )


if __name__ == "__main__":
    main()
