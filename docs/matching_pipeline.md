# Cross-Model SAE Concept Matching & Labeling Pipeline

This document describes the pipeline for matching Sparse Autoencoder (SAE)
features across tabular foundation models, grouping them into universal
concepts, and labeling those concepts with natural-language descriptions.

## Overview

Tabular foundation models (TabPFN, TabICL, Mitra, CARTE, etc.) are trained
independently, so their SAE feature spaces are not aligned. This pipeline
discovers which features across different models respond to the same data
patterns by correlating their row-level activations on shared datasets, then
groups and labels the resulting cross-model correspondences.

```
 01_match        02_build_graph     03_pymfe      04_regression
 (MNN pairs) --> (tier classify)    (dataset     --> (probe R²,
                                     meta-feat)      examples)
                        \               |              /
                         \              v             /
                      05_label_cross_model_concepts
                       (union-find → groups → prompts)
                                    |
                                    v
                      label_concept_groups.py
                       (batch → dispatch → LLM label → merge)
```

## Prerequisites

Each model needs:
- Extracted embeddings in `output/embeddings/tabarena/{model}/`
- A trained SAE checkpoint (`validated.pt`) in `output/sae_tabarena_sweep_round{N}/`
- Normalized training/test splits in `output/sae_training_round{N}/`

These are produced by the embedding extraction and SAE training pipelines
(see `scripts/embeddings/` and `scripts/sae/`).

---

## Step 1: Cross-Model Feature Matching

**Script:** `scripts/matching/01_match_sae_concepts_mnn.py`

For each pair of models, computes row-level Pearson correlation between all
alive SAE features on shared TabArena datasets, then finds feature
correspondences via mutual nearest neighbors (MNN).

```bash
PYTHONPATH=. python scripts/matching/01_match_sae_concepts_mnn.py \
    --method mnn --alive-threshold 0.001 --save-correlations
```

**Inputs:**
- Test-split embeddings and SAE checkpoints for each model
- Per-dataset normalization stats

**Outputs:**
- `output/sae_feature_matching_mnn_t0.001.json` — pairwise MNN matches
- `output/sae_cross_correlations/*.npz` — full correlation matrices (28 model pairs)

**Also needed (control):**
```bash
PYTHONPATH=. python scripts/matching/01_match_sae_concepts_mnn.py \
    --cross-model-baseline
```
Produces `output/sae_cross_model_random_baseline.json` — trained-A vs
random-B null distribution for per-pair significance thresholds.

### Key parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--method` | `mnn` | Matching algorithm (mnn recommended) |
| `--alive-threshold` | 0.001 | Min max-activation for a feature to be "alive" |
| `--save-correlations` | false | Save full NxM correlation matrices (needed for steps 2, 5) |

---

## Step 2: Build Feature Match Graph

**Script:** `scripts/matching/02_build_concept_graph.py`

Classifies every alive feature into a tier based on its best cross-model
correlate and the per-pair random baseline threshold:

| Tier | Meaning |
|------|---------|
| `mnn` | Mutual nearest neighbor AND above pair-specific p90 baseline |
| `threshold` | Best correlate exceeds baseline but not a mutual nearest neighbor |
| `mnn_below_threshold` | MNN match but below noise floor |
| `unmatched` | No correlate above noise in any partner model |

```bash
PYTHONPATH=. python scripts/matching/02_build_concept_graph.py
```

**Inputs:**
- MNN matching results (step 1)
- Cross-model random baseline (step 1)
- Cross-correlation matrices (step 1)

**Output:** `output/sae_feature_match_graph_p90.json`

---

## Step 3: Compute Dataset Meta-Features

**Script:** `scripts/matching/03_compute_pymfe_cache.py`

Extracts ~80-140 PyMFE meta-features per TabArena dataset (statistical,
information-theoretic, complexity, landmarking). These provide dataset-level
context for concept labeling prompts.

```bash
PYTHONPATH=. python scripts/matching/03_compute_pymfe_cache.py
```

**Output:** `output/pymfe_tabarena_cache.json`

---

## Step 4: Concept Probe Regression

**Script:** `scripts/matching/04_analyze_concept_regression.py`

For each alive SAE feature in each model, fits Ridge regression predicting
the feature's activation vector from row-level meta-features. Produces:

- **R² score** — how well meta-features explain the feature's firing pattern
- **Top probes** — which meta-features are most predictive (with coefficients)
- **Contrastive examples** — raw data rows where the feature fires strongly
  vs nearby rows where it stays silent (primary evidence for labeling)

```bash
PYTHONPATH=. python scripts/matching/04_analyze_concept_regression.py --device cuda
```

**Inputs:**
- Test-split embeddings, SAE checkpoints
- PyMFE cache (step 3)

**Output:** `output/sae_concept_analysis_round{N}.json` (~1GB, contains
per-feature statistics and contrastive examples for all models)

### Important notes
- Runs on GPU (computes SAE activations)
- Uses test-split embeddings (held out from SAE training)
- Alive masks computed from training data (authoritative source)

---

## Step 5: Concept Grouping

**Script:** `scripts/matching/05_label_cross_model_concepts.py`

Groups matched features into concept groups and generates labeling prompts.
Runs in phases:

### Phase 1: MNN groups (union-find)
Applies transitive closure on MNN edges — if feature A matches B, and B
matches C, all three join one group.

```bash
PYTHONPATH=. python scripts/matching/05_label_cross_model_concepts.py --phase 1
```

This typically produces one oversized "mega-group" (group 0) containing
thousands of features chained together through loose transitive connections.
This is a graph artifact, not a real concept.

### Splitting the mega-group (Leiden)
Decomposes the mega-group into coherent sub-communities using Leiden
community detection on the cross-correlation subgraph:

```bash
PYTHONPATH=. python scripts/matching/05_label_cross_model_concepts.py \
    --split-megagroup 0
```

Communities exceeding `--max-group-size` (default: 100) are recursively
split at doubled resolution. The original group is retained with split
metadata; new sub-groups get sequential IDs.

### Phase 2: Extend via cross-correlation (optional)
Assigns unmatched features to existing groups if their best cross-model
correlate exceeds the pair-specific threshold.

### Phase 3: Probe-signature clustering (optional)
Clusters remaining unmatched features by their top-probe signature.

**Inputs:**
- MNN matching results (step 1)
- Concept analysis with probes and examples (step 4)
- Cross-correlation matrices (step 1)

**Output:** `output/cross_model_concept_labels_round{N}.json`

Each concept group contains:
- `members` — list of (model, feature_idx) tuples
- `n_models` — how many independent models discovered this concept
- `prompt` — LLM labeling prompt with contrastive examples
- `label` — initially "unlabeled", filled by step 6

---

## Step 6: LLM Labeling

**Script:** `scripts/concepts/label_concept_groups.py`

Five-step workflow that labels concept groups using LLM agents.

### 6a. Prepare batches
Sorts groups by confidence (n_models, fraction with good contrastive
examples, mean R²) and writes numbered batch files of 100 groups each.
Group 0 (mega-group) is skipped.

```bash
PYTHONPATH=. python scripts/concepts/label_concept_groups.py prepare
```

**Output:** `output/concept_labeling/batch_{00..N}.json` + `manifest.json`

### 6b. Dispatch into agent chunks
Splits each batch into 50-group chunks small enough for an LLM agent to
process in a single context window (~250KB each).

```bash
PYTHONPATH=. python scripts/concepts/label_concept_groups.py dispatch --batch 0
```

**Output:** `output/concept_labeling/chunks/batch_NN_chunk_M.json` + prompt files

### 6c. Label (LLM agents)
Each agent reads a chunk file containing prompts with contrastive examples,
and writes a JSON file mapping group IDs to natural-language labels.

**Bootstrapping:** Batch 0 is labeled first *without* few-shot examples
(it's the highest-confidence batch). Its labels then serve as style
reference for all subsequent batches, ensuring consistency without leaking
labels from prior rounds.

Labels describe abstract structural patterns (e.g., "rows with a single
extreme outlier against an otherwise tight numeric spread") rather than
domain-specific interpretations.

### 6d. Combine chunks
Merges chunk label files back into a single per-batch label file.

```bash
PYTHONPATH=. python scripts/concepts/label_concept_groups.py combine --batch 0
```

### 6e. Merge into groups JSON
Writes all labels back into the concept groups file with provenance metadata.

```bash
PYTHONPATH=. python scripts/concepts/label_concept_groups.py merge
```

---

## Output Files

| File | Size | Contents |
|------|------|----------|
| `sae_feature_matching_mnn_t0.001.json` | ~2MB | Pairwise MNN matches (28 pairs) |
| `sae_cross_correlations/*.npz` | ~300MB | Full correlation matrices |
| `sae_cross_model_random_baseline.json` | ~12KB | Per-pair null thresholds |
| `sae_feature_match_graph_p90.json` | ~5MB | Per-feature tier classification |
| `pymfe_tabarena_cache.json` | ~1MB | Dataset meta-features |
| `sae_concept_analysis_round{N}.json` | ~1GB | Per-feature R², probes, examples |
| `cross_model_concept_labels_round{N}.json` | ~8MB | Concept groups with labels |
| `concept_labeling/` | ~5MB | Batch files, chunks, label files |

## Design Decisions

**Why MNN?** Mutual nearest neighbors is conservative — both features must
consider each other their best match. This avoids false positives from
one-sided high correlations at the cost of lower recall (addressed by
phases 2-3).

**Why Leiden for mega-groups?** Union-find's transitive closure chains
loosely related features: A↔B, B↔C, ... all end up in one component.
Leiden (RBConfigurationVertexPartition) finds dense sub-communities in
the actual correlation graph, producing coherent groups.

**Why test-split?** Matching and analysis use held-out test embeddings
(30% split) never seen during SAE training. This prevents SAE
reconstruction artifacts from inflating correlations.

**Why per-pair thresholds?** Different model pairs have different noise
floors (e.g., HyperFast vs TabPFN baseline correlation differs from
Mitra vs TabICL). Per-pair p90 baselines from trained-vs-random controls
set pair-specific significance thresholds.

**Why confidence-ordered batches?** High-confidence groups (more models,
better examples, higher R²) produce more reliable labels. Labeling these
first and using them as few-shot examples for lower-confidence batches
improves overall label quality.
