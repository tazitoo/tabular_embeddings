# CLAUDE.md - Tabular Embeddings Geometry

## Project Overview

Research project investigating the universal geometry of embeddings in tabular foundation models. Target: ML/AI conference publication.

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

Same GPU worker pool as the finance project, using the `finance` conda env (has torch+CUDA+model deps).

- **Workers**: surfer4 (3090), terrax4 (2080 Ti), octo4 (3070), firelord4 (4090)
- **Conda env**: `finance` on each worker
- **Worker repo path**: `/home/brian/src/tabular_embeddings`
- **Worker python**: `/home/brian/anaconda3/envs/finance/bin/python`
- **Code sync**: `git pull --ff-only` on each worker before cluster start
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

## Key Findings (Update as research progresses)

- [x] CKA between TabPFN (192d) and TabICL (512d) internal representations: **0.705** (5 datasets)
- [x] CKA between TabPFN and TabICL output probabilities: **0.879** (higher convergence at output)
- [x] Intrinsic dimensionality: TabPFN ~7 of 192 dims, TabICL ~17 of 512 dims
- [ ] CKA scores with HyperFast and Mitra (pending GPU worker testing)
- [ ] SAE richness comparison across models
- [ ] Correlation: embedding geometry vs task performance
- [ ] Full TabArena sweep (51 datasets)
