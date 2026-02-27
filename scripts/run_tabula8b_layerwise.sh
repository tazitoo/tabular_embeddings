#!/bin/bash
# Run Tabula-8B layerwise CKA analysis across TabArena datasets
cd /home/brian/src/tabular_embeddings
/home/brian/anaconda3/envs/tfm/bin/python scripts/4_results/layerwise_cka_analysis.py \
    --batch \
    --model tabula8b \
    --device cuda \
    --max-datasets 12 \
    --n-samples 300
