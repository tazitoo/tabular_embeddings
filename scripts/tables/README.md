# LaTeX Tables

Paper-ready LaTeX tables for SAE sweep results.

## Usage

```bash
python scripts/tables/sae_tables.py
```

Generates:
- `sae_hyperparameters.tex` - Best hyperparameters per architecture per model
- `sae_metrics.tex` - R², L0, alive features, stability per architecture per model

## Data Source

Reads from: `output/sae_tabarena_sweep/*/best_configs.json`

## Tables

**Table 1: Hyperparameters** (`\label{tab:sae_hyperparameters}`)
- Expansion factor
- Learning rate
- Sparsity penalty
- TopK value (where applicable)

**Table 2: Metrics** (`\label{tab:sae_metrics}`)
- R² (reconstruction quality)
- L0 (average sparsity)
- Alive features
- Stability ($s_n^{dec}$)
