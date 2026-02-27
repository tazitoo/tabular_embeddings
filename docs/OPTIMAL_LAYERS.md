# Optimal Extraction Layers - Integration Guide

## Overview

We've established a single source of truth for optimal embedding extraction layers based on layerwise CKA critical depth analysis. This ensures consistency across:
1. **Table 5** (extraction layers table)
2. **Layerwise CKA plots** (3-panel visualizations)
3. **Embedding extraction** (compare_embeddings.py)
4. **SAE training** (sae_tabarena_sweep.py)

## Configuration File

**Location:** `config/optimal_extraction_layers.json`

**Structure:**
```json
{
  "tabpfn": {
    "optimal_layer": 17,              // Extract from layer 17
    "n_layers": 24,                    // Total model depth
    "critical_layer_mean": 18,         // Where CKA < 0.5 (the cliff)
    "critical_layer_std": 1.64,        // Variance across datasets
    "optimal_depth_fraction": 0.708,   // 17/24 = 70.8%
    "extraction_point": "layer_17",    // Directory name suffix
    "rationale": "..."                 // Why this layer is optimal
  },
  ...
}
```

## Key Findings

| Model | Old Layer | Optimal Layer | CKA at Old | CKA at Optimal | Improvement |
|-------|-----------|---------------|------------|----------------|-------------|
| TabPFN | L23 (96%) | L17 (71%) | 0.16 | 0.68 | **4.3x better** |
| TabICL | L12 (92%) | L8 (62%) | - | 0.66 | - |
| Mitra | L12 (92%) | L10 (77%) | - | 0.87 | - |
| TabDPT | L15 (88%) | L13 (76%) | - | 0.73 | - |
| CARTE | L4 (80%) | L1 (20%) | - | 0.50 | - |
| HyperFast | L1 (33%) | L1 (33%) | 1.00 | 1.00 | ✓ Already optimal |
| Tabula-8B | L21 (64%) | L18 (55%) | - | 0.38 | - |

**Problem:** We trained SAEs on over-processed representations (too close to output) that had already diverged significantly from the input.

**Solution:** Re-extract at optimal layers (critical - 1) where representation is transformed but not yet diverged.

## Usage

### 1. In Python Code

```python
from config import get_optimal_layer, load_optimal_layers

# Get optimal layer for a model
layer = get_optimal_layer('tabpfn')  # Returns: 17

# Load full config
config = load_optimal_layers()
print(config['tabpfn']['rationale'])
```

### 2. Embedding Directory Lookup

```python
from data.tabarena_utils import get_embedding_dir

# Automatically uses optimal layer from config
dir_name = get_embedding_dir('tabpfn')  # Returns: 'tabpfn_layer17'

# Or use legacy behavior
dir_name = get_embedding_dir('tabpfn', use_optimal=False)  # Returns: 'tabpfn'
```

### 3. Extract Embeddings at Optimal Layers

```bash
# For all models
python scripts/extract_optimal_layers.py --suite tabarena --device cuda

# For specific models
python scripts/extract_optimal_layers.py --suite tabarena --models tabpfn mitra

# Distributed across GPU workers
python scripts/extract_optimal_layers.py --suite tabarena --distributed
```

### 4. SAE Training (Automatic)

The SAE sweep script now automatically uses optimal layers:

```bash
# Train SAE on optimal embeddings
python scripts/sae_tabarena_sweep.py --model tabpfn --n-trials 30

# This will automatically look for embeddings in:
# output/embeddings/tabarena/tabpfn_layer17/
```

## Workflow

### Phase 1: Re-extract Embeddings (Required)

```bash
# Extract embeddings at optimal layers for all models
python scripts/extract_optimal_layers.py --suite tabarena --distributed
```

This creates:
```
output/embeddings/tabarena/
├── tabpfn_layer17/          # Optimal (was using layer 23)
├── tabicl_layer8/           # Optimal (was using layer 12)
├── mitra_layer10/           # Optimal (was using layer 12)
├── tabdpt_layer13/          # Optimal (was using layer 15)
├── carte_layer1/            # Optimal (was using layer 4)
├── hyperfast_layer1/        # Optimal (already correct)
└── tabula8b_layer18_ctx600/ # Optimal (was using layer 21)
```

### Phase 2: Retrain SAEs

```bash
# Train SAEs on new optimal embeddings
for model in tabpfn tabicl mitra tabdpt carte hyperfast tabula8b; do
    python scripts/sae_tabarena_sweep.py --model $model --n-trials 30 --device cuda
done
```

### Phase 3: Compare Quality

```python
# Compare old vs new SAE quality
python scripts/compare_sae_quality.py \
    --old output/sae_tabarena_sweep_old/ \
    --new output/sae_tabarena_sweep/ \
    --metrics richness stability reconstruction
```

## Benefits

1. **Consistency**: One config file → all scripts synchronized
2. **Reproducibility**: Clear documentation of extraction decisions
3. **Maintainability**: Update one file to change all behaviors
4. **Validation**: Table 5 and plots derived from same source
5. **Quality**: SAEs trained on better representations (higher CKA with input)

## Expected SAE Improvements

Training on optimal layers should improve:
- **Feature interpretability**: Less over-processed = more grounded in input
- **Reconstruction quality**: Better representations = better SAE fit
- **Dictionary richness**: More alive features capturing meaningful variations
- **Stability**: Representations less sensitive to dataset variations

## Files Modified

1. **Created:**
   - `config/optimal_extraction_layers.json` - Source of truth
   - `config/__init__.py` - Config loading utilities
   - `scripts/extract_optimal_layers.py` - Extraction helper
   - `docs/OPTIMAL_LAYERS.md` - This documentation

2. **Updated:**
   - `data/tabarena_utils.py` - `get_embedding_dir()` now uses config
   - `scripts/tables/extraction_layers_detailed.tex` - Generated from config
   - `output/layerwise_cka_appendix_*.png` - 3-panel plots showing cliffs

## Next Steps

1. ✅ Config file created
2. ✅ Integration code written
3. ✅ Visualizations generated
4. ⏳ **Extract embeddings at optimal layers** (1-2 hours, distributed)
5. ⏳ **Retrain SAEs** (4-6 hours per model, can parallelize)
6. ⏳ **Compare old vs new quality**
7. ⏳ **Update paper figures and tables**
