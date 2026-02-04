# Project Status

Last updated: 2026-02-03

## Extraction Pipeline

Single-model extraction (`extract_embeddings.py`) runs on GPU workers via SSH, saves `.npz` per dataset. CKA computed locally (`compute_cka_from_saved.py`) from rsynced files.

```bash
# Extract on worker
ssh surfer4 "source /home/brian/anaconda3/etc/profile.d/conda.sh && conda activate finance && \
  cd /home/brian/src/tabular_embeddings && \
  python extract_embeddings.py --model tabpfn --suite tabarena --output-dir /data/embeddings/tabarena --device cuda"

# Rsync to local
rsync -avz surfer4:/data/embeddings/tabarena/ 3_output/embeddings/tabarena/

# Compute CKA
python compute_cka_from_saved.py --embedding-dir 3_output/embeddings/tabarena --output 3_output/geometric_sweep_tabarena.csv
```

## Completed Extractions (TabArena, 51 datasets)

| Model | Datasets | Notes |
|-------|----------|-------|
| TabPFN | 51/51 | Regressor checkpoint synced to workers |
| TabICL | 51/51 | Handles both tasks natively |
| Mitra | 50/51 | 1 fail: QSAR-TID-11 (class label issue) |
| HyperFast | 51/51 | Regression via target discretization (bins continuous y for pseudo-classification) |

## 4-Model CKA Results

| Model Pair | CKA (mean ± std) | n datasets |
|---|---|---|
| Mitra vs TabPFN | 0.789 ± 0.144 | 50 |
| Mitra vs TabICL | 0.655 ± 0.194 | 50 |
| TabICL vs TabPFN | 0.615 ± 0.178 | 51 |
| HyperFast vs Mitra | 0.277 ± 0.215 | 50 |
| HyperFast vs TabICL | 0.262 ± 0.189 | 51 |
| HyperFast vs TabPFN | 0.225 ± 0.166 | 51 |

Transformer ICL models cluster at CKA 0.62-0.79. HyperFast (hypernetwork) is geometrically distinct at ~0.22-0.28.

## Storage Layout

```
/data/embeddings/tabarena/          (on workers)
  tabpfn/*.npz
  tabicl/*.npz
  mitra/*.npz
  hyperfast/*.npz

3_output/embeddings/tabarena/       (local, rsynced)
  same structure (~52MB total)
```

Embeddings are NOT in git. CSVs are.

## Package Availability by Worker

| Package | firelord4 | surfer4 | terrax4 | octo4 |
|---------|-----------|---------|---------|-------|
| autogluon (Mitra) | yes | no | yes | no |
| hyperfast | yes | yes | yes | yes |
| tabpfn | yes | yes | yes | yes |
| tabicl | yes | yes | yes | yes |
| ticl (MotherNet) | no | no | no | no |

## Non-interactive SSH

Workers require explicit conda sourcing for non-interactive SSH:
```bash
ssh worker "source /home/brian/anaconda3/etc/profile.d/conda.sh && conda activate finance && ..."
```

## Blocked

- **MotherNet checkpoint**: Download host unreachable. Emailed amueller@microsoft.com (Andreas Mueller, primary author, Microsoft GSL). Would test whether hypernetwork geometric distance is architecture-class or HyperFast-specific.

## Next Steps

- TabDPT extraction (needs investigation — primary embedding was shape (20, 2) probability fallback, not internal representations)
- CARTE extraction
- MotherNet (blocked on checkpoint)
- Per-dataset CKA analysis: do regression vs classification datasets show different geometric patterns?
- SAE concept extraction and comparison
- Correlation between CKA structure and downstream task performance
