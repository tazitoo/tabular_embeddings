#!/usr/bin/env bash
# Run a sequential queue of random-baseline ablation_sweep jobs.
# Each pair argument is "ENV:MODEL_A:MODEL_B[:DATASETS_FILE]" where ENV is
# tfm or tfm2 and the optional datasets file is one dataset name per line.
#
# Usage:
#   launch_random_ablation_queue.sh <queue_file>
#
# Each line of <queue_file>: ENV MODEL_A MODEL_B [DATASETS_FILE]
# Lines starting with # are ignored.
#
# Output goes to output/ablation_sweep_random_tols/<pair>/<dataset>.npz
# Progress is written to /tmp/ablation_random_<hostname>.log

set -uo pipefail

QUEUE="${1:?Usage: $0 <queue_file>}"
REPO=/home/brian/src/tabular_embeddings
SAE_DIR=output/sae_random_baseline
IMP_DIR=output/perrow_importance_random
MATCH=output/sae_feature_matching_mnn_t0.001_random.json
OUT_DIR=output/ablation_sweep_random_tols
COMMON_ARGS=(--sae-dir "$SAE_DIR" --importance-dir "$IMP_DIR" \
             --matching-file "$MATCH" --output-dir "$OUT_DIR" \
             --gc-tolerance 0.99 --min-gap 0.01 --resume)

cd "$REPO"

while IFS= read -r line; do
    # Skip blanks and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    read -r env model_a model_b datasets_file <<< "$line"
    if [[ "$env" == "tfm2" ]]; then
        PY=/home/brian/anaconda3/envs/tfm2/bin/python
    else
        PY=/home/brian/anaconda3/envs/tfm/bin/python
    fi

    args=(--models "$model_a" "$model_b" "${COMMON_ARGS[@]}")
    if [[ -n "${datasets_file:-}" && -f "$datasets_file" ]]; then
        # Read datasets, one per line, into a bash array
        mapfile -t datasets < "$datasets_file"
        args+=(--datasets "${datasets[@]}")
    fi

    echo "=== $(date -Iseconds) $env ${model_a}_vs_${model_b} ${datasets_file:-all} ==="
    "$PY" -m scripts.intervention.ablation_sweep "${args[@]}"
    rc=$?
    echo "=== $(date -Iseconds) ${model_a}_vs_${model_b} exit=$rc ==="
done < "$QUEUE"

echo "=== $(date -Iseconds) queue complete ==="
