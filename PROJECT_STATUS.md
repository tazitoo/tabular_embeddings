# Project Status

Last updated: 2026-01-29

## Extraction Pipeline

Single-model extraction (`extract_embeddings.py`) runs on GPU workers via SSH, saves `.npz` per dataset. CKA computed locally (`compute_cka_from_saved.py`) from rsynced files.

```bash
# Extract on worker
ssh firelord4 "source /home/brian/anaconda3/etc/profile.d/conda.sh && conda activate finance && \
  cd /home/brian/src/tabular_embeddings && \
  python extract_embeddings.py --model mitra --suite tabarena --output-dir /data/embeddings/tabarena --device cuda"

# Rsync to local
rsync -avz firelord4:/data/embeddings/tabarena/ 3_output/embeddings/tabarena/

# Compute CKA
python compute_cka_from_saved.py --embedding-dir 3_output/embeddings/tabarena --output 3_output/geometric_sweep_full.csv
```

## Completed Extractions (TabArena, 51 datasets)

| Model | Datasets | Notes |
|-------|----------|-------|
| TabPFN | 42/51 | 9 regression datasets fail (no regressor checkpoint) |
| TabICL | 51/51 | Handles both tasks natively |
| Mitra | 50/51 | Regressor support added. 1 fail: QSAR-TID-11 (class label issue) |
| HyperFast | 39/51 | Classification only, no regressor variant exists. 12 regression skipped cleanly |

## 4-Model CKA Results

| Model Pair | CKA (mean +/- std) | n datasets |
|---|---|---|
| Mitra vs TabPFN | 0.813 +/- 0.141 | 41 |
| Mitra vs TabICL | 0.655 +/- 0.194 | 50 |
| TabICL vs TabPFN | 0.637 +/- 0.187 | 42 |
| HyperFast vs Mitra | 0.252 +/- 0.202 | 39 |
| HyperFast vs TabICL | 0.251 +/- 0.191 | 40 |
| HyperFast vs TabPFN | 0.208 +/- 0.160 | 39 |

Transformer ICL models cluster at CKA 0.64-0.81. HyperFast (hypernetwork) is geometrically distinct at ~0.2-0.25.

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
- **TabPFN regressor checkpoint**: `tabpfn-v2.5-regressor-v2.5_default.ckpt` requires accepting HuggingFace gated terms at https://huggingface.co/Prior-Labs/tabpfn_2_5 and running `huggingface-cli login` on workers. Would add 9 regression datasets to TabPFN coverage.

## Next Steps

- Download TabPFN regressor checkpoint (HF auth)
- TabDPT extraction (needs investigation — primary embedding was shape (20, 2) probability fallback, not internal representations)
- CARTE extraction
- MotherNet (blocked on checkpoint)
- Per-dataset CKA analysis: do regression vs classification datasets show different geometric patterns?
- SAE concept extraction and comparison
- Correlation between CKA structure and downstream task performance
