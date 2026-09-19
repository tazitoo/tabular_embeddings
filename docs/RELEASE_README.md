# INCEPT: Infusing Novel Concepts for Explaining Pretrained Tabular Models

Code for the paper. Anonymized for review; the full release (with the trained
SAE checkpoints, matching files, and per-row intervention outputs from which
every table and figure is generated) follows upon publication.

## Layout

| directory | contents |
|---|---|
| `models/` | embedding extraction for each tabular foundation model (TabPFN, Mitra, TabICL, TabICL-v2, TabDPT, CARTE, HyperFast, Tabula-8B) and the pinned, deterministic fitting paths (`layer_extraction.py`) |
| `data/` | TabArena loading and the per-model preprocessing cache |
| `scripts/sae_corpus/` | SAE training corpus construction (`04_extract_all_layers.py`, `07_build_sae_training_data.py`, `promote_round.py`) |
| `scripts/sae/` | SAE architecture, hyperparameter sweep, selection rule, retraining, random-SAE controls |
| `scripts/matching/` | cross-model concept matching (`01`–`05`), transfer caches |
| `scripts/intervention/` | per-row concept importance, intervention library, baseline prediction cache |
| `scripts/rebuttal/` | ablation and transfer sweeps (both directions and both SAE arms), functional decomposition, off-manifold stratification, patch search and its dispatcher |
| `scripts/tables/`, `scripts/figures/`, `scripts/paper/` | table and figure generators |
| `scripts/round_paths.py` | where a round's artifacts and results live; bumping `DEFAULT_SAE_ROUND` moves the whole pipeline to a fresh tree |
| `envs/` | frozen environments (`tfm`, and `tfm2` for TabICL-v2) |
| `docs/reproducibility.md` | what reproduces bit-exactly, and across which hardware |
| `tests/` | unit tests (`pytest tests`) |

## Pipeline

Every script's defaults are the canonical path; run with no arguments unless noted.
Stages in order, per SAE round:

```bash
# 1. embeddings and SAE corpus
python -m scripts.sae_corpus.04_extract_all_layers
python -m scripts.sae_corpus.07_build_sae_training_data

# 2. SAEs: sweep, select by the paper's rule, retrain, random-SAE controls
python -m scripts.sae.sae_tabarena_sweep
python -m scripts.sae.select_sae_config
python -m scripts.sae.retrain_selected --trial N --expansion E --topk K
python -m scripts.sae.generate_random_baseline

# 3. concept matching and transfer caches
python -m scripts.matching.01_match_sae_features
python -m scripts.matching.02_build_match_graph
python -m scripts.matching.04_regress_group_features
python -m scripts.matching.05_label_cross_model_concepts
python -m scripts.matching.build_transfer_caches

# 4. per-row importances and baseline predictions
python -m scripts.intervention.cache_baseline_predictions
python -m scripts.intervention.perrow_importance --model <m>

# 5. interventions (forward = the paper's direction; reverse = the above-diagonal check)
python -m scripts.rebuttal.ablation_sweep_symmetric --models a b --forward
python -m scripts.rebuttal.ablation_sweep_symmetric --models a b
python -m scripts.rebuttal.transfer_sweep_symmetric --models a b --forward
python -m scripts.rebuttal.transfer_sweep_symmetric --models a b
#    random-SAE arm: add --sae-dir/--importance-dir/--matching-file/--output-dir
#    (see scripts/rebuttal/launch_reverse_queue.sh for the exact arguments)

# 6. decomposition, patching population, patch search
python -m scripts.rebuttal.functional_decomposition --models a b
python -m scripts.rebuttal.off_manifold_concept_stratification --dump
python -m scripts.rebuttal.build_patching_burndown
python -m scripts.rebuttal.patch_search            # or patch_queue.py to dispatch over GPUs

# 7. tables and figures: scripts/tables, scripts/figures, scripts/paper
```

Cluster launchers under `scripts/rebuttal/launch_*.sh` and `scripts/intervention/launch_*.sh`
assume worker hosts named `worker1`…`worker5` and a multi-GPU host `gpuhost`; edit the host
lists for your setup.

## Determinism

Seeds, the CUDA allocator configuration, and the model fitting paths are pinned
(`models/layer_extraction.py`); every output records the host, GPU, and commit that
produced it. Reproduction is bit-exact on a given GPU architecture; see
`docs/reproducibility.md` for the measured cross-architecture differences.

## License

AGPL-3.0 (see `LICENSE`).
