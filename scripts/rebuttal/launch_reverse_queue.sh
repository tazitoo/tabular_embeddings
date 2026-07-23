#!/usr/bin/env bash
# REBUTTAL: run ONE reverse (above-diagonal) symmetric intervention script over
# a SHARD of model pairs. Fanned out across nodes: each node gets a distinct
# pair list. Run ablation across all nodes first, let it finish, then transfer.
#
# reverse is the DEFAULT of both vendored scripts (no direction flag needed).
# --resume skips datasets already written.
# tabicl_v2 pairs auto-run under tfm2; everything else under tfm.
#
# Usage (on a worker):
#   nohup bash scripts/rebuttal/launch_reverse_queue.sh <ablation|transfer> \
#         tabpfn:mitra tabicl:carte ... > /tmp/launch.out 2>&1 </dev/null &
#
# Output:
#   output/rebuttal/symmetric_ablation/<pair>/<dataset>.npz
#   output/rebuttal/symmetric_transfer/<pair>/<dataset>.npz
# Log: /tmp/reverse_<kind>_<host>.log

set -uo pipefail

REPO=/home/brian/src/tabular_embeddings
TFM=/home/brian/anaconda3/envs/tfm/bin/python
TFM2=/home/brian/anaconda3/envs/tfm2/bin/python
HOST=$(hostname)

KIND="${1:?Usage: $0 <ablation|transfer> <a:b> [<a:b> ...]}"; shift
PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "No pairs given. Usage: $0 <ablation|transfer> <a:b> [<a:b> ...]"; exit 1
fi

case "$KIND" in
    ablation) MOD=ablation_sweep_symmetric ;;
    transfer) MOD=transfer_sweep_symmetric ;;
    *) echo "Unknown kind '$KIND' (want ablation|transfer)"; exit 1 ;;
esac

LOG=/tmp/reverse_${KIND}_${HOST}.log
LOCK=/tmp/reverse_${KIND}_${HOST}.lock

# Lock: SSH nohup can fire even when a prompt is rejected; keep a second launch
# of the same kind on the same host from duplicating work.
if [[ -e "$LOCK" ]]; then
    echo "Lock $LOCK exists (pid $(cat "$LOCK" 2>/dev/null)); $KIND already running on $HOST. Exiting."
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$REPO"

echo "=== $(date -Iseconds) reverse $KIND start on $HOST | pairs: ${PAIRS[*]} ===" | tee -a "$LOG"

for pair in "${PAIRS[@]}"; do
    a=${pair%%:*}; b=${pair##*:}
    if [[ "$a" == "tabicl_v2" || "$b" == "tabicl_v2" ]]; then
        PY=$TFM2; env_name=tfm2
    else
        PY=$TFM; env_name=tfm
    fi
    echo "=== $(date -Iseconds) [$env_name] $MOD $a vs $b ===" | tee -a "$LOG"
    "$PY" -m scripts.rebuttal.$MOD --models "$a" "$b" --device cuda --resume >> "$LOG" 2>&1
    echo "=== $(date -Iseconds) $MOD $a vs $b exit=$? ===" | tee -a "$LOG"
done

echo "=== $(date -Iseconds) reverse $KIND complete on $HOST ===" | tee -a "$LOG"
