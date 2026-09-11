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
# Output (round tree, from scripts/round_paths.py):
#   output/round{N}/symmetric_ablation[_random]/<pair>/<dataset>.npz
#   output/round{N}/symmetric_transfer[_random]/<pair>/<dataset>.npz
# Log: /tmp/reverse_<kind>_<host>_gpu<N>.log

set -uo pipefail

REPO=/home/brian/src/tabular_embeddings
TFM=/home/brian/anaconda3/envs/tfm/bin/python
TFM2=/home/brian/anaconda3/envs/tfm2/bin/python
HOST=$(hostname)

KIND="${1:?Usage: $0 <ablation|transfer|ablation_random|transfer_random> <a:b> ...}"; shift
PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "No pairs given. Usage: $0 <ablation|transfer|ablation_random|transfer_random> <a:b> ..."; exit 1
fi

# A trailing _random selects the random-SAE control: the same vendored scripts
# pointed at the random baseline SAE/importance/matching dirs and a separate
# output dir (mirrors launch_random_ablation_queue.sh). Trained mode is default.
RANDOM_MODE=0
case "$KIND" in
    *_random) RANDOM_MODE=1; KIND=${KIND%_random} ;;
esac

case "$KIND" in
    ablation) MOD=ablation_sweep_symmetric ;;
    transfer) MOD=transfer_sweep_symmetric ;;
    *) echo "Unknown kind '$KIND' (want ablation|transfer[ _random])"; exit 1 ;;
esac

if [[ $RANDOM_MODE -eq 1 ]]; then
    # The random arm's dirs live in the same round tree as the trained arm's.
    RESULTS_DIR=$(cd "$REPO" && "$TFM" -c "from scripts.round_paths import RESULTS_DIR; print(RESULTS_DIR)")
    RANDOM_SAE_DIR=$(cd "$REPO" && "$TFM" -c "from scripts.round_paths import random_sae_dir; print(random_sae_dir())")
    EXTRA=(--sae-dir "$RANDOM_SAE_DIR"
           --importance-dir "$RESULTS_DIR/perrow_importance_random"
           --matching-file "$RESULTS_DIR/sae_feature_matching_mnn_t0.001_random.json"
           --output-dir "$RESULTS_DIR/symmetric_${KIND}_random")
    tag=_random
else
    EXTRA=()
    tag=
fi

# One queue per (kind, host, GPU): CUDA_VISIBLE_DEVICES is set by the launcher, so a
# multi-GPU host (morg) can run several queues of the same kind side by side.
GPU_TAG=gpu${CUDA_VISIBLE_DEVICES:-x}
LOG=/tmp/reverse_${KIND}${tag}_${HOST}_${GPU_TAG}.log
LOCK=/tmp/reverse_${KIND}${tag}_${HOST}_${GPU_TAG}.lock

# Lock: SSH nohup can fire even when a prompt is rejected; keep a second launch
# of the same kind on the same host from duplicating work.
if [[ -e "$LOCK" ]]; then
    echo "Lock $LOCK exists (pid $(cat "$LOCK" 2>/dev/null)); $KIND already running on $HOST. Exiting."
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$REPO"

echo "=== $(date -Iseconds) reverse ${KIND}${tag} start on $HOST | pairs: ${PAIRS[*]} ===" | tee -a "$LOG"

for pair in "${PAIRS[@]}"; do
    a=${pair%%:*}; b=${pair##*:}
    if [[ "$a" == "tabicl_v2" || "$b" == "tabicl_v2" ]]; then
        PY=$TFM2; env_name=tfm2
    else
        PY=$TFM; env_name=tfm
    fi
    echo "=== $(date -Iseconds) [$env_name${tag:+ RANDOM}] $MOD $a vs $b ===" | tee -a "$LOG"
    "$PY" -m scripts.rebuttal.$MOD --models "$a" "$b" --device cuda --resume "${EXTRA[@]}" >> "$LOG" 2>&1
    echo "=== $(date -Iseconds) $MOD $a vs $b exit=$? ===" | tee -a "$LOG"
done

echo "=== $(date -Iseconds) reverse ${KIND}${tag} complete on $HOST ===" | tee -a "$LOG"
