# Project Status

Last updated: 2026-02-04

## Extraction Pipeline

Single-model extraction (`extract_embeddings.py`) runs on GPU workers via SSH, saves `.npz` per dataset. CKA computed locally (`compute_cka_from_saved.py`) from rsynced files.

```bash
# Extract on worker
ssh surfer4 "source /home/brian/anaconda3/etc/profile.d/conda.sh && conda activate tfm && \
  cd /home/brian/src/tabular_embeddings && \
  python extract_embeddings.py --model tabpfn --suite tabarena --output-dir /data/embeddings/tabarena --device cuda"

# Rsync to local
rsync -avz surfer4:/data/embeddings/tabarena/ 3_output/embeddings/tabarena/

# Compute CKA
python compute_cka_from_saved.py --embedding-dir 3_output/embeddings/tabarena --output 3_output/geometric_sweep_tabarena.csv
```

## Completed Extractions (TabArena, 51 datasets)

| Model | Datasets | Embedding Dim | Notes |
|-------|----------|---------------|-------|
| TabPFN | 51/51 | 192 | Transformer ICL |
| TabICL | 51/51 | 512 | Column-then-row transformer |
| Mitra | 50/51 | 256 | 2D attention transformer |
| HyperFast | 51/51 | varies | Hypernetwork |
| TabDPT | 51/51 | 192 | Transformer + retrieval |
| CARTE | 51/51 | 300 | GNN (FastText node embeddings) |
| Tabula-8B | 51/51 | 4096 | Llama-3 8B fine-tuned for tabular |

## 7-Model CKA Results

| Model Pair | CKA (mean ± std) | n |
|---|---|---|
| Mitra vs TabDPT | 0.82 ± 0.14 | 50 |
| Mitra vs TabPFN | 0.79 ± 0.14 | 50 |
| TabDPT vs TabPFN | 0.76 ± 0.15 | 51 |
| Mitra vs TabICL | 0.65 ± 0.19 | 50 |
| TabDPT vs TabICL | 0.64 ± 0.16 | 51 |
| TabICL vs TabPFN | 0.61 ± 0.18 | 51 |
| CARTE vs Mitra | 0.45 ± 0.23 | 50 |
| CARTE vs TabDPT | 0.44 ± 0.23 | 51 |
| CARTE vs TabPFN | 0.42 ± 0.21 | 51 |
| CARTE vs TabICL | 0.37 ± 0.20 | 51 |
| HyperFast vs Mitra | 0.28 ± 0.22 | 50 |
| HyperFast vs TabDPT | 0.27 ± 0.18 | 51 |
| HyperFast vs TabICL | 0.26 ± 0.19 | 51 |
| HyperFast vs TabPFN | 0.22 ± 0.17 | 51 |
| TabICL vs Tabula | 0.17 ± 0.17 | 51 |
| CARTE vs HyperFast | 0.16 ± 0.14 | 51 |
| Mitra vs Tabula | 0.16 ± 0.17 | 50 |
| TabDPT vs Tabula | 0.16 ± 0.17 | 51 |
| TabPFN vs Tabula | 0.15 ± 0.17 | 51 |
| HyperFast vs Tabula | 0.12 ± 0.15 | 51 |
| CARTE vs Tabula | 0.12 ± 0.17 | 51 |

### Key Findings

**Three distinct geometric clusters:**

1. **Transformer ICL cluster** (CKA 0.61-0.82): Mitra, TabDPT, TabPFN, TabICL
   - These models share similar latent geometry despite different architectures
   - Strongest: Mitra-TabDPT (0.82), Mitra-TabPFN (0.79)

2. **Intermediate**: CARTE (GNN) and HyperFast (hypernetwork)
   - CARTE: CKA ~0.37-0.45 with transformers
   - HyperFast: CKA ~0.22-0.28 with transformers
   - CARTE vs HyperFast: 0.16

3. **Tabula-8B (LLM)**: Most geometrically distinct
   - CKA ~0.12-0.17 with all other models
   - Text serialization approach yields fundamentally different representations

## Storage Layout

```
/data/embeddings/tabarena/          (on workers)
  tabpfn/*.npz
  tabicl/*.npz
  mitra/*.npz
  hyperfast/*.npz
  tabdpt/*.npz
  carte/*.npz
  tabula/*.npz

3_output/embeddings/tabarena/       (local, rsynced)
  same structure (~240MB total)
```

Embeddings are NOT in git. CSVs are.

## Package Availability by Worker

| Package | surfer4 |
|---------|---------|
| tabpfn | yes |
| tabicl | yes |
| hyperfast | yes |
| tabdpt | yes |
| carte-ai | yes |
| tabula-8b | yes (model at /data/models/tabula-8b) |

Note: CARTE requires FastText model at `/home/brian/cc.en.300.bin`

## Blocked

- **MotherNet checkpoint**: Download host unreachable. Would test whether hypernetwork geometric distance is architecture-class or HyperFast-specific.

## Next Steps

- [ ] Per-dataset CKA analysis: do regression vs classification datasets show different geometric patterns?
- [ ] Intrinsic dimensionality analysis across all 7 models
- [ ] SAE concept extraction and comparison
- [ ] Correlation between CKA structure and downstream task performance
- [ ] vec2vec embedding translation experiments
