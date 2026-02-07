# CLAUDE.md - Tabular Embeddings Geometry
## Project Overview

Research project investigating the universal geometry of embeddings in tabular foundation models. Target: ML/AI conference publication. This repo will be open sourced and heavily scrutinized by experts in the field.  The code should be concise, well written, well documented, and test coverage of at least 80%.  Research & experimentation code should not safe guard against edge cases - we need to know why it fails, and fix it.  Short cuts to get results undermine the validity of the paper.

## Research Goals

1. Test if tabular FMs share universal latent structure (Platonic hypothesis)
2. Apply vec2vec-style embedding translation between model spaces
3. Use SAEs to extract and compare learned "concepts" across models
4. Relate embedding geometry to downstream task performance

## Key Scripts

```bash
# Embedding geometry comparison
python compare_embeddings.py --dataset adult --models tabpfn

# SAE concept analysis
python compare_sae_concepts.py --dataset iris --dict-expansion 4

# Run on benchmark suite
python compare_embeddings.py --suite quick
python compare_embeddings.py --suite tabarena --max-datasets 10
python compare_embeddings.py --suite relbench
```

## Benchmark Suites

- **TabZilla**: 36 "hard" datasets from OpenML
- **OpenML-CC18**: 72 curated classification datasets
- **Regression**: 12 standard regression tasks
- **Quick**: 9 datasets for fast iteration
- **TabArena**: 51 curated datasets (NeurIPS 2025), OpenML suite 457, `data/extended_loader.py`
- **RelBench**: 14 relational tasks across 7 databases (Stanford), flattened to tabular, `data/extended_loader.py`
- **PMLB**: 110 datasets across 33 domains, `data/extended_loader.py`
- **Probing**: 16 controlled synthetic generators, `data/extended_loader.py`

## Code Conventions

- **Never default to CPU for PyTorch training.** All scripts that call `train_sae`, model extraction, or any torch training loop must accept `--device` and default to `cuda` (or auto-detect). Running on CPU is never acceptable.
- Embeddings: `(n_samples, embedding_dim)` shape
- `EmbeddingResult` dataclass for extraction results
- `SimilarityResult` for pairwise comparisons
- All models implement `EmbeddingExtractor` base class

## Supported Models

| Model | Architecture | Install | Type |
|-------|-------------|---------|------|
| TabPFN | Transformer (ICL) | `pip install tabpfn` (requires HF token) | Classification |
| HyperFast | Hypernetwork | Manual weight download | Classification |
| TabICL | Column-then-row transformer | `pip install tabicl` | Classification |
| TabDPT | Transformer + retrieval | `git clone Layer6/TabDPT && pip install -e .` | Classification + Regression |
| Mitra | 2D attention transformer (72M) | `pip install 'autogluon.tabular[mitra]'` | Classification + Regression |
| MotherNet | Hypernetwork (generates MLP) | `git clone microsoft/ticl && pip install -e .` | Classification |
| CARTE | Graph transformer (star graph GNN) | `pip install carte-ai` | Classification + Regression |

## Dependencies

- tabpfn (requires HF token)
- hyperfast (manual weight download)
- tabicl, tabdpt, carte-ai (pip install)
- autogluon.tabular[mitra] (for Mitra model)
- mothernet (from microsoft/ticl repo)
- torch, numpy, pandas, scikit-learn, matplotlib
- openml (for TabArena datasets)
- relbench (for RelBench datasets, `pip install relbench`)
- pmlb (for PMLB datasets, `pip install pmlb`)

## Distributed Execution
- Running on CPU is never a good idea.
- Workers have two hostnames: surfer (1GbE) and surfer4 (40GbE, 192.168.10.x)                                                               
- Always SSH to the 4 variant for data transfers                                                                                            
- socket.gethostname() returns the short name (no 4), so code checking hostname must accept both   

GPU worker pool using the `tfm` conda env (has torch+CUDA+model deps).

- **Workers**: surfer4 (3090), terrax4 (2080 Ti), octo4 (3070), firelord4 (4090)
- **Conda env**: `tfm` on each worker (may need to be created on each worker)
- **Worker repo path**: `/home/brian/src/tabular_embeddings`
- **Worker python**: `/home/brian/anaconda3/envs/tfm/bin/python` 
- **Code sync**: `git pull --ff-only` on each worker before cluster start.  Using rsync is never a good idea.
- **Module**: `cluster.py` (not `distributed.py` — renamed to avoid shadowing `dask.distributed`)

```bash
# Check worker health
python cluster.py --check

# Sync code to workers
python cluster.py --sync

# Distributed embedding comparison (multi-dataset)
python compare_embeddings.py --suite tabarena --models tabpfn hyperfast tabicl --distributed

# Distributed SAE extraction
python compare_sae_concepts.py --dataset adult --models tabpfn hyperfast --distributed
```

galactus orchestrates (Dask scheduler), workers extract embeddings on GPU, similarity analysis runs locally on galactus.

## Key Findings

- [x] Full TabArena sweep (51 datasets, 4 models): see `output/geometric_sweep_full.csv`
- [x] Transformer ICL cluster: Mitra-TabPFN CKA=0.81, Mitra-TabICL=0.66, TabICL-TabPFN=0.64
- [x] HyperFast (hypernetwork) geometrically distinct: CKA ~0.21-0.25 vs all transformer models
- [x] Intrinsic dimensionality: TabPFN ~7 of 192 dims, TabICL ~17 of 512 dims
- [ ] MotherNet — would test if hypernetwork geometric distance is architecture-class or HyperFast-specific
- [ ] SAE richness comparison across models
- [ ] Correlation: embedding geometry vs task performance

See `PROJECT_STATUS.md` for detailed extraction status, blocked items, and next steps.

## Known Pitfalls

- **`train_sae()` defaults to CPU.** The function signature is `device="cpu"`. Any script wrapping it must explicitly pass `device="cuda"` or accept `--device` from the CLI. Easy to miss and silently 25%+ slower.
- **TabPFN has two model variants with different architectures.** The classifier has 24 transformer layers; the regressor has 18. Layer indices (e.g. layer 16) land at different relative depths (67% vs 89%). Any layer-specific analysis must account for task type.
- **TabPFN layer extraction requires task type.** `extract_tabpfn_all_layers()` needs `task="regression"` to load `TabPFNRegressor` and call `predict()` instead of `predict_proba()`. Without it, regression and many-class datasets fail silently.
- **TabArena has mixed task types.** 11 of 51 datasets are regression or many-class. Always check `TABARENA_DATASETS[name]["task"]` rather than assuming classification.
