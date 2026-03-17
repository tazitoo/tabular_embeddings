# Plan: Preprocessing Pipeline + SAE Rebuild

**Date:** 2026-03-17
**Status:** Planned
**Blocking:** All intervention pipeline work (Phase 1/2/3)

## Context

Session on 2026-03-17 discovered two foundational issues:

1. **Preprocessing is wrong.** All 8 models receive raw float32 numpy arrays with categoricals pre-encoded as integer codes. Each model has its own expected preprocessing (StandardScaler, PowerTransformer, one-hot encoding, etc.) that we bypass entirely. 82% of columns across TabArena are categorical.

2. **SAE training data is insufficient.** 350 rows per dataset (1.72% of available data). Reconstruction tests show TabICL at 25-95x degradation, TabDPT/Mitra at 1.5-2.5x. Partially caused by bad norm stats from small samples.

These must be fixed in sequence — preprocessing first, then SAE rebuild.

## Phase 1: Preprocessing Pipeline

### Goal
Each model receives input data in the format its library expects.

### Approach
- `load_tabarena_dataset()` returns a **DataFrame** with proper dtypes (categorical columns as `object` or `category`, not integer codes)
- Each model wrapper in `models/` handles its own preprocessing OR we add a per-model preprocessor
- Reference: TabArena repo (`github.com/autogluon/tabarena/tree/main/tabarena/tabarena/models`) for TabPFN, TabICL, Mitra, TabDPT configs

### Per-model preprocessing:

| Model | Preprocessing needed |
|-------|---------------------|
| TabPFN | Let TabPFN library handle it (internal StandardScaler + power scaling + categorical handling) |
| TabICL | Normalization methods (power/robust/quantile) — check TabICL library defaults |
| TabICL v2 | Same as TabICL |
| Mitra | AutoGluon's AutoMLPipelineFeatureGenerator (already using autogluon) |
| TabDPT | Check TabDPT library for internal preprocessing |
| HyperFast | Pass `cat_features` indices, let library do mean imputation + one-hot + StandardScaler |
| CARTE | Pass raw DataFrame with `object`-dtype categoricals, library does PowerTransformer + graph construction |
| Tabula-8B | Text serialization — likely already correct |

### Validation
- Compare model accuracy with old vs new preprocessing on a few datasets
- If accuracy improves, embeddings were degraded

### Files to modify
- `data/extended_loader.py` — return DataFrame with proper dtypes alongside numpy arrays
- `models/*_embeddings.py` — each wrapper accepts DataFrame, handles preprocessing before extraction
- May need new `data/preprocessing.py` for shared utilities

## Phase 2: Re-extract Layerwise Embeddings

### Goal
Extract embeddings with correct preprocessing across all 51 TabArena datasets × 8 models.

### Approach
- Update `scripts/embeddings/extract_layer_embeddings.py` to use new preprocessing
- Re-run layerwise extraction on all 4 GPU workers
- Re-run optimal layer analysis — the optimal extraction layer may change with proper preprocessing

### Output
- `output/embeddings/tabarena_layerwise_round7/` (new round)
- Updated `config/optimal_extraction_layers.json`

## Phase 3: Doubly Stratified SAE Training Corpus

### Goal
Build SAE training data that covers both the target distribution and the model's difficulty spectrum.

### Approach
1. Run each model on each full dataset with correct preprocessing, collect per-row losses
2. For each model × dataset: stratify by target (class labels or quantile bins for regression) × loss quartile
3. Sample `min(N_i, 5000)` rows per dataset, 70/30 train/test split
4. Train/test split is sacred — test set becomes the intervention holdout

### Per-model training sets
Each model gets its own training corpus (different rows have different loss values per model).

### Output
- `output/sae_training_round9/{model}_layer{N}_sae_training.npz`
- `output/sae_training_round9/{model}_layer{N}_sae_test.npz`
- `output/sae_training_round9/{model}_layer{N}_norm_stats.npz`

## Phase 4: Retrain SAEs

### Goal
Train SAEs on the new corpus.

### Approach
- Use same architecture/configs from round 8 (validated efficiency configs)
- Train on new data with correct preprocessing
- 4-seed validation as before

### Output
- `output/sae_tabarena_sweep_round9/`

## Phase 5: Validate

### Goal
Confirm SAEs generalize to full dataset.

### Approach
- Run `sae_recon_test.py` — MSE ratio should be ~1.0 across all models
- Run `embedding_energy_distance.py` — energy distance should be low
- Compare concept dictionaries old vs new — are there new concepts? Did we lose any?

## Phase 6: Resume Intervention Pipeline

Once SAEs are validated:
- Phase 1 (concept importance) with correct embeddings
- Phase 2 (cross-model comparison)
- Phase 3 (transfer + ablation)

## Open Questions

- Should we use TabArena's default preprocessing configs or sweep? (Principled approach: use defaults, not sweeps)
- Does the optimal extraction layer change with correct preprocessing?
- Will TabICL's SAE reconstruction fix with proper preprocessing alone, or does it also need more data?
- Tabula-8B preprocessing — verify text serialization is actually correct
