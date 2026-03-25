# Output Directory Structure

## SAE Pipeline Stages

The `sae_training_round*` directories are confusingly named. They represent
different pipeline stages, not iterations:

### `sae_training_round9/` — Raw data & embeddings
- `tabarena_splits.json` — Train/test row indices for all 51 TabArena datasets
- `preprocessed/{model}/{dataset}.npz` — Per-model preprocessed features (numpy)
- `embeddings/{model}/{dataset}.npz` — Per-layer raw embeddings from extraction

### `sae_training_round10/` — SAE training corpus & checkpoints
- `{model}_taskaware_sae_training.npz` — Pooled training embeddings (70% of rows)
- `{model}_taskaware_sae_test.npz` — Pooled test embeddings (30% of rows)
- `{model}_taskaware_norm_stats.npz` — Per-dataset normalization stats

### `sae_tabarena_sweep_round10/` — Trained SAE models
- `{model}/sae_matryoshka_archetypal_validated.pt` — Best SAE checkpoint per model

## TODO
- [ ] Rename directories to reflect their actual purpose (e.g. `sae_corpus/`,
      `sae_embeddings/`, `sae_checkpoints/`) rather than historical round numbers.
      This requires updating ~30 scripts that reference these paths.

## Other Output Directories

- `perrow_importance/{model}/` — Per-row feature importance from SAE ablation
- `ablation_sweep/{model_a}_vs_{model_b}/` — Cross-model ablation results
- `transfer_sweep/{model_a}_vs_{model_b}/` — Cross-model transfer results
- `sae_cross_correlations/` — Pairwise SAE feature correlation matrices
- `figures/` — Generated plots
- `preprocessing_cache/` — Legacy (replaced by `sae_training_round9/preprocessed/`)
