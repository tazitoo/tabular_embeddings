#!/usr/bin/env bash
# REBUTTAL: forward-delta transfer across morg's 5 GPUs (deployed_delta save,
# for the intervention-vs-embedding subspace analysis / ofnL Q2).
# Pairs are distributed round-robin across GPUs; each GPU runs its pairs
# sequentially, pinned via CUDA_VISIBLE_DEVICES.
#
# Usage (on morg):
#   nohup bash scripts/rebuttal/launch_forward_morg.sh <env> <a:b> [a:b ...] \
#         > /tmp/launch_fwd_morg.out 2>&1 </dev/null &
#   env = tfm  (non-v2 pairs)  |  tfm2 (tabicl_v2 pairs)
#
# Output: output/rebuttal/forward_deltas/<pair>/<dataset>.npz  (--resume skips done)
set -uo pipefail

REPO=/home/brian/src/tabular_embeddings
ENV="${1:?Usage: $0 <tfm|tfm2> <gpu_csv> <a:b> ...}"; shift
GPUS_CSV="${1:?GPU list, e.g. 1,2,3,4 (skip a faulted GPU)}"; shift
PY=/home/brian/anaconda3/envs/$ENV/bin/python
OUT=output/rebuttal/forward_deltas
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NG=${#GPUS[@]}
cd "$REPO"

pairs=("$@")
[ ${#pairs[@]} -eq 0 ] && { echo "No pairs given."; exit 1; }

# One job per GPU: round-robin pairs over the given GPU list, each GPU runs its
# pairs sequentially. expandable_segments guards against fragmentation OOM.
for idx in "${!GPUS[@]}"; do
    g=${GPUS[$idx]}
    gpairs=()
    for i in "${!pairs[@]}"; do
        [ $((i % NG)) -eq "$idx" ] && gpairs+=("${pairs[$i]}")
    done
    [ ${#gpairs[@]} -eq 0 ] && continue
    (
        log=/tmp/fwd_morg_${ENV}_gpu${g}.log
        echo "=== $(date -Iseconds) GPU $g env=$ENV pairs: ${gpairs[*]} ===" > "$log"
        for pair in "${gpairs[@]}"; do
            a=${pair%%:*}; b=${pair##*:}
            echo "=== $(date -Iseconds) GPU$g $a vs $b start ===" >> "$log"
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$PY" -m scripts.rebuttal.transfer_sweep_symmetric \
                --models "$a" "$b" --forward --device cuda --resume \
                --output-dir "$OUT" >> "$log" 2>&1
            echo "=== $(date -Iseconds) GPU$g $a vs $b exit=$? ===" >> "$log"
        done
        echo "=== $(date -Iseconds) GPU$g ALL DONE ===" >> "$log"
    ) &
done
wait
echo "=== $(date -Iseconds) launch_forward_morg complete ($ENV) ==="
