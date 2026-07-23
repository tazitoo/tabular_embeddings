#!/usr/bin/env bash
# REBUTTAL: run the reverse (above-diagonal) symmetric intervention across all
# 15 model pairs, both transfer (weak->strong) and ablation (ablate the weaker
# model's unique concepts). Sequential single-worker queue — mirrors
# launch_random_ablation_queue.sh.
#
# reverse is the DEFAULT of both vendored scripts, so no direction flag needed.
# --resume skips datasets already written (e.g. the credit-g smoke tests).
#
# tabicl_v2 pairs run under the tfm2 env (torch/tabicl-v2 deps); everything
# else under tfm.  See feedback: tabicl_v2 silently fails per-dataset in tfm.
#
# Output:
#   output/rebuttal/symmetric_transfer/<pair>/<dataset>.npz
#   output/rebuttal/symmetric_ablation/<pair>/<dataset>.npz
# Log: /tmp/reverse_symmetric_<host>.log
#
# Usage (on a worker):
#   nohup bash scripts/rebuttal/launch_reverse_queue.sh > /tmp/reverse_symmetric.out 2>&1 &

set -uo pipefail

REPO=/home/brian/src/tabular_embeddings
TFM=/home/brian/anaconda3/envs/tfm/bin/python
TFM2=/home/brian/anaconda3/envs/tfm2/bin/python
HOST=$(hostname)
LOG=/tmp/reverse_symmetric_${HOST}.log
LOCK=/tmp/reverse_symmetric.lock

# Lock file: SSH nohup can fire even when a prompt is rejected; the lock keeps
# a second launch from duplicating work.
if [[ -e "$LOCK" ]]; then
    echo "Lock $LOCK exists (pid $(cat "$LOCK" 2>/dev/null)); another run active. Exiting."
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$REPO"

MODELS=(tabpfn tabicl tabicl_v2 mitra tabdpt carte)

echo "=== $(date -Iseconds) reverse symmetric queue start on $HOST ===" | tee -a "$LOG"

for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
        a=${MODELS[i]}; b=${MODELS[j]}
        if [[ "$a" == "tabicl_v2" || "$b" == "tabicl_v2" ]]; then
            PY=$TFM2; env_name=tfm2
        else
            PY=$TFM; env_name=tfm
        fi

        for script in transfer_sweep_symmetric ablation_sweep_symmetric; do
            echo "=== $(date -Iseconds) [$env_name] $script $a vs $b ===" | tee -a "$LOG"
            "$PY" -m scripts.rebuttal.$script \
                --models "$a" "$b" --device cuda --resume >> "$LOG" 2>&1
            rc=$?
            echo "=== $(date -Iseconds) $script $a vs $b exit=$rc ===" | tee -a "$LOG"
        done
    done
done

echo "=== $(date -Iseconds) reverse symmetric queue complete on $HOST ===" | tee -a "$LOG"
