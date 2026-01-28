# Universal Geometry of Tabular Embeddings

Investigating whether tabular foundation models learn universal latent representations, inspired by [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540) (Jha et al., NeurIPS 2025).

## Research Questions

Do tabular foundation models (TabPFN, HyperFast, TabICL, etc.) share universal geometric structure despite different architectures and pretraining approaches? We investigate:

1. **Geometric Similarity**: Do embeddings from different models align in representation space?
2. **Embedding Translation**: Can we translate between model spaces without paired data (vec2vec)?
3. **Concept Richness**: Do more capable models learn richer dictionaries of features (via SAE)?
4. **Transfer Implications**: Does embedding geometry predict downstream task performance?

## Key Differences from LLM Setting

| Aspect | LLM (Original Paper) | Tabular FMs (This Work) |
|--------|---------------------|-------------------------|
| Input | Text sequences | Tabular rows (numerical/categorical) |
| Architecture | Transformers | Hypernetworks, Transformers, Mixtures |
| Embedding source | Layer activations | Context encodings, predicted weights |
| Pretraining | Next token prediction | Synthetic tabular priors, meta-learning |

## Models Under Investigation

| Model | Architecture | Pretraining | Reference |
|-------|-------------|-------------|-----------|
| TabPFN v2.5 | Transformer | Synthetic prior | [arxiv](https://arxiv.org/abs/2511.08667) |
| HyperFast | Hypernetwork | Meta-learning | [arxiv](https://arxiv.org/abs/2402.14335) |
| TabICL | In-context learning | Synthetic | [ICML 2025] |
| iLTM | GBDT + Hypernetwork | Meta-learning | [arxiv](https://arxiv.org/html/2511.15941) |

## Hypotheses

1. **Platonic Hypothesis for Tabular**: Models pretrained on synthetic priors converge to similar representations
2. **Concept Richness**: More capable models learn richer SAE dictionaries (more features, higher diversity)
3. **Geometry Predicts Performance**: Embedding similarity correlates with task performance similarity
4. **Pretraining Signature**: Synthetic vs real-data pretraining creates distinct geometric signatures

## Benchmarks

We evaluate on standard tabular benchmarks:

| Suite | Datasets | Description |
|-------|----------|-------------|
| [TabZilla](https://github.com/naszilla/tabzilla) | 36 | "Hard" datasets where simple baselines fail |
| [OpenML-CC18](https://www.openml.org/search?type=study&id=99) | 72 | Curated classification benchmark |
| Regression | 12 | Standard regression tasks |

```python
from data.loader import load_dataset, load_benchmark_suite

# Load single dataset
X, y, meta = load_dataset("adult")

# Load entire benchmark suite
datasets = load_benchmark_suite("tabzilla", max_samples=5000)
```

## Methods

### 1. Embedding Extraction

Extract internal representations from each model:
- **TabPFN**: Transformer hidden states via forward hooks
- **HyperFast**: Context encoding + predicted NN weights
- **TabICL**: Attention-based representations

### 2. Geometric Analysis

Following the original vec2vec paper:

| Metric | Description |
|--------|-------------|
| **CKA** | Centered Kernel Alignment - structural similarity independent of dimension |
| **Procrustes** | Distance after optimal orthogonal alignment |
| **Cosine Similarity** | Pairwise embedding similarity |

```bash
# Compare embedding geometry
python compare_embeddings.py --dataset adult --models tabpfn hyperfast
```

### 3. Sparse Autoencoder Analysis

Inspired by mechanistic interpretability ([Anthropic, 2023](https://transformer-circuits.pub/2023/monosemantic-features)), we train SAEs to extract interpretable "concepts" from embeddings.

**Richness Metrics:**
- **Alive features**: Non-dead dictionary elements
- **Effective dimensions**: Entropy-based feature count
- **Dictionary diversity**: Average pairwise feature distance
- **Sparsity**: Activation selectivity (interpretability proxy)

```bash
# Compare SAE-extracted concepts
python compare_sae_concepts.py --dataset adult --dict-expansion 4
```

### 4. Feature Geometry

Following [The Geometry of Concepts](https://arxiv.org/abs/2410.19750):
- Eigenvalue spectrum (power law)
- Feature clustering
- Co-activation patterns

## Directory Structure

```
tabular_embeddings/
├── models/                 # Embedding extraction wrappers
│   ├── base.py            # EmbeddingExtractor base class
│   ├── tabpfn_embeddings.py
│   └── hyperfast_embeddings.py
├── analysis/              # Analysis tools
│   ├── similarity.py      # CKA, Procrustes, cosine
│   ├── sparse_autoencoder.py  # SAE training & analysis
│   └── visualization.py   # Plots and figures
├── data/
│   └── loader.py          # TabZilla, OpenML, synthetic loaders
├── compare_embeddings.py  # Main geometry comparison script
├── compare_sae_concepts.py # SAE concept analysis script
├── output/                # Results and figures
└── notebooks/             # Exploratory analysis
```

## Installation

```bash
# Create environment
conda create -n tabular_emb python=3.10
conda activate tabular_emb

# Core dependencies
pip install torch numpy pandas scikit-learn matplotlib

# Tabular foundation models
pip install tabpfn  # TabPFN v2.5 (requires HF token for gated model)
pip install hyperfast  # HyperFast (manual weight download required)
```

### Model Setup

**TabPFN v2.5** (gated model):
```bash
# Get HuggingFace token and accept model terms
# https://huggingface.co/Prior-Labs/tabpfn_2_5
export HF_TOKEN="hf_..."
```

**HyperFast** (manual download):
```bash
# Download from https://figshare.com/articles/software/hyperfast_ckpt/24749838
# Save to ~/.hyperfast/hyperfast.ckpt
```

## Quick Start

```bash
# Test with synthetic data
python compare_embeddings.py --synthetic --n-samples 1000

# Run on OpenML dataset
python compare_embeddings.py --dataset adult

# SAE concept analysis
python compare_sae_concepts.py --dataset iris --dict-expansion 8

# Run on benchmark suite
python compare_embeddings.py --suite quick
```

## Related Work

- [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540) - vec2vec paper
- [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) - Theoretical foundation
- [The Geometry of Concepts](https://arxiv.org/abs/2410.19750) - SAE feature structure
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - SAE foundations
- [TabZilla](https://github.com/naszilla/tabzilla) - Tabular benchmark suite
- [TabPFN v2.5](https://arxiv.org/abs/2511.08667) - Primary tabular FM
- [HyperFast](https://arxiv.org/abs/2402.14335) - Hypernetwork approach

## Citation

```bibtex
@misc{tabular_embeddings,
  title={Universal Geometry of Tabular Embeddings},
  author={...},
  year={2026},
  note={Work in progress}
}
```
