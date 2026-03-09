# Concept Embedding Comparison Design

**Date:** 2026-03-09
**Goal:** Validate SAE concept groupings and cross-model alignment by comparing
natural language concept descriptions via text embeddings.

## Motivation

The interpretability pipeline groups SAE features into ~866 concept groups based
on activation correlations (MNN matching + cross-correlation). We need to verify
these groupings make semantic sense: features grouped together should describe
similar patterns, matched features across models should agree, and unmatched
features near a group should interpolate coherently.

Text embeddings of richer concept descriptions provide an independent validation
signal that complements the activation-based matching.

## Three Use Cases

1. **Within-group coherence** — Do features grouped by activation correlation
   also describe the same tabular pattern?
2. **Cross-model matched-pair agreement** — Do MNN-matched features across
   different TFMs have semantically similar descriptions?
3. **Interpolation coherence** — For unmatched features near a group, do their
   descriptions interpolate sensibly with the group's description?

## Architecture: Three Standalone Scripts

```
SAE activations + probes + raw data
        │
        ▼
Script 1: generate_concept_descriptions.py
        │  (Haiku brief labels + Sonnet rich descriptions)
        ▼
concept_descriptions.json
        │
        ├──► Script 2: embed_concept_descriptions.py
        │         │  (nomic-embed-text-v1.5, local)
        │         ▼
        │    concept_embeddings.npz + concept_embedding_metrics.json
        │
        └──► Script 3: validate_concept_embeddings_api.py
                  │  (nomic self-checks; API fallback if needed)
                  ▼
             concept_embedding_validation.json
```

All outputs go to `output/concept_descriptions/`.

---

## Script 1: `generate_concept_descriptions.py`

### Approach

Three-pass generation using the `anthropic` Python SDK (API key from environment,
same pattern as existing `label_cross_model_concepts.py`):

**Pass 1 — Haiku brief labels** (~1,600 calls)
- Upgraded prompt (from existing code), still "2-5 words only"
- Cheap, fast; useful for figure labels, table cells, quick scanning
- Also serves as sanity check against Sonnet descriptions

**Pass 2 — Sonnet rich descriptions for grouped + explained features** (~1,100 calls)
- One Sonnet call per group (866 groups) + per explained-unmatched cluster (~200)
- Prompt includes:
  - Probe consensus with signs and strengths (when available)
  - Top-5 activating rows + bottom-5 non-activating rows (actual tabular data)
  - Model names and feature indices for context
- Output: 1-2 sentence group summary + per-feature description where behavior differs

**Pass 3 — Sonnet rich descriptions for unexplained features** (~500 cluster calls)
- Two-pass dependency: needs descriptions from pass 2 as landmarks
- Prompt includes:
  - Activating/non-activating row samples (no probe guidance available)
  - Nearest described neighbors as landmarks:
    "This feature's closest described concepts are: 'sparse rows with many
    zero-valued numerics' (r=0.34), 'extreme outlier rows' (r=0.28).
    How does this feature differ?"
- Sonnet interpolates/contrasts against known territory

### Input

- `cross_model_concept_labels_v2.json` (groups + feature lookup)
- `concept_regression_with_pymfe.json` (probe coefficients)
- SAE training/test data (`output/sae_training_round6/`)
- SAE checkpoints (`output/sae_tabarena_sweep_round6/`)

### Output

`output/concept_descriptions/concept_descriptions.json`:
```json
{
  "metadata": {
    "haiku_model": "claude-haiku-4-5-20251001",
    "sonnet_model": "claude-sonnet-4-20250514",
    "timestamp": "..."
  },
  "groups": {
    "0": {
      "brief_label": "sparse rows",
      "summary": "Activates on rows with many zero-valued numeric features...",
      "features": {
        "TabPFN:305": {
          "brief_label": "sparse numerics",
          "description": "Sparse numeric rows with concentrated zeros..."
        },
        "Mitra:42": {
          "brief_label": "zero-heavy rows",
          "description": "Similar to group but stronger activation on..."
        }
      }
    }
  },
  "unmatched": {
    "TabPFN:789": {
      "brief_label": "extreme outliers",
      "description": "Rows with extreme outliers in the upper tail...",
      "landmarks": ["group:12", "group:45"],
      "r2": 0.256
    }
  }
}
```

### CLI

```bash
python scripts/generate_concept_descriptions.py --model all --max-samples 5
python scripts/generate_concept_descriptions.py --model tabpfn  # single model unmatched only
python scripts/generate_concept_descriptions.py --pass 1  # Haiku labels only (cheap test)
```

### Cost Estimate

~1,600 Haiku calls (~$0.10) + ~1,600 Sonnet calls (~$2-3) ≈ $3 total.

---

## Script 2: `embed_concept_descriptions.py`

### Approach

Embed all Sonnet descriptions locally using `nomic-embed-text-v1.5` via
`sentence-transformers`. Matryoshka embedding support enables multi-resolution
comparison (768d, 256d, 128d, 64d).

### Process

1. Load all Sonnet descriptions from `concept_descriptions.json`
2. Embed with nomic-embed-text-v1.5 (runs on CPU in seconds for ~10K descriptions)
3. Compute validation metrics for the three use cases

### Validation Metrics

| Metric | What it tests | Method |
|--------|--------------|--------|
| Within-group coherence | Grouped features describe same thing | Mean pairwise cosine sim within each group; distribution across 866 groups |
| Matched-pair agreement | MNN-matched features agree semantically | Cosine sim of matched pairs vs random-pair baseline |
| Interpolation coherence | Unmatched features near groups sound similar | Cosine sim between unexplained description and its landmark neighbors |

### Output

`output/concept_descriptions/concept_embeddings.npz`:
```
embeddings: (n_features, 768)       # full nomic-embed vectors
feature_ids: ["TabPFN:305", ...]    # matching order
group_ids: [0, 0, 5, 5, -1, ...]   # group membership (-1 = unmatched)
```

`output/concept_descriptions/concept_embedding_metrics.json`:
```json
{
  "within_group_cosine": {"mean": 0.72, "std": 0.15, "per_group": {...}},
  "matched_pair_cosine": {"mean": 0.68, "random_baseline": 0.12},
  "interpolation_cosine": {"mean": 0.45, "n_features": 2147},
  "model": "nomic-embed-text-v1.5",
  "embedding_dim": 768
}
```

### CLI

```bash
python scripts/embed_concept_descriptions.py
python scripts/embed_concept_descriptions.py --dim 256  # Matryoshka truncation
```

### Dependency

`pip install sentence-transformers` (pulls in transformers + torch, already in tfm env).

---

## Script 3: `validate_concept_embeddings_api.py`

### Approach

Validate that nomic embeddings are trustworthy using self-consistency checks first.
Escalate to an API embedding model only if internal checks flag concerns.

### Process

**Step 1 — Nomic internal checks (always run):**
- **Matryoshka dimension consistency**: Do nearest-neighbor rankings agree across
  768d vs 256d vs 128d? Report Spearman correlation on pairwise similarities.
- **Bootstrap stability**: Resample descriptions with replacement, re-embed,
  compare ranking stability.

**Step 2 — API fallback (only if step 1 flags issues, or `--api` flag):**
- Sample ~200 descriptions stratified across groups/unmatched/unexplained
- Embed with API model (Voyage AI `voyage-3` or OpenAI `text-embedding-3-large`)
- Compare rank-order agreement with nomic: Spearman rho, recall@5
- Report worst disagreements for manual inspection

### Output

`output/concept_descriptions/concept_embedding_validation.json`:
```json
{
  "nomic_self_checks": {
    "matryoshka_spearman": {"768v256": 0.97, "768v128": 0.94, "768v64": 0.88},
    "bootstrap_stability": 0.96,
    "passed": true
  },
  "api_validation": {
    "ran": false,
    "reason": "nomic self-checks passed"
  }
}
```

Or if API validation was triggered:
```json
{
  "api_validation": {
    "ran": true,
    "api_model": "voyage-3",
    "n_sampled": 200,
    "spearman_rho": 0.91,
    "recall_at_5": 0.84,
    "worst_disagreements": [...]
  }
}
```

### CLI

```bash
python scripts/validate_concept_embeddings_api.py           # nomic self-checks only
python scripts/validate_concept_embeddings_api.py --api voyage  # force API comparison
```

---

## Key Design Decisions

1. **Separate scripts** — Description generation, embedding, and validation are
   independent. Can iterate on prompts without re-embedding, and vice versa.

2. **Two description lengths** — Haiku brief labels (2-5 words) for display,
   Sonnet rich descriptions (1-2 sentences) for embedding. Both stored per feature.

3. **Landmark-guided unexplained descriptions** — Pass 3 gives Sonnet semantic
   anchors from pass 2, enabling interpolation rather than labeling in a vacuum.

4. **Local-first embedding** — nomic-embed-text-v1.5 is free, fast, reproducible,
   and has Matryoshka support. API models only as validation fallback.

5. **Three validation metrics** map directly to the three use cases: within-group
   coherence, matched-pair agreement, interpolation coherence.

## Dependencies

- `anthropic` (already installed, API key from environment)
- `sentence-transformers` (pip install, pulls in transformers + torch)
- `nomic-embed-text-v1.5` model (auto-downloaded on first use, ~550MB)
