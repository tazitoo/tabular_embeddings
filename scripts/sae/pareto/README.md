# Pareto Frontier Plots

SAEBench-style Pareto frontier analysis for SAE architectures.

## Usage

```bash
python scripts/pareto/sae_pareto_frontier.py
```

Generates:
- `sae_pareto_r2.pdf` - L0 sparsity vs R² trade-off
- `sae_pareto_stability.pdf` - L0 sparsity vs Stability trade-off

Both plots include:
- Pareto-optimal points (circles) vs non-optimal (squares)
- SAEBench optimal L0 range (50-150) highlighted in green
- All validated SAE architectures across models

## Data Source

Reads from: `output/sae_tabarena_sweep/*/best_configs.json`
