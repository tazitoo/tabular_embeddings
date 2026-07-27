#!/usr/bin/env bash
# REBUTTAL: shared-queue forward-delta transfer launcher (random or trained).
#
# All pairs go into ONE lock-guarded queue; each GPU pops the next pair the
# instant it is free. No round-robin phase gating and no idle slots -- a GPU
# that finishes early immediately grabs more work. Env is auto-selected per
# pair (tfm2 if tabicl_v2 is involved, else tfm), so both envs interleave in a
# single launch instead of the old tfm-then-tfm2 two-phase chain.
#
# Usage (on any worker or morg), pairs are "a:b" model names (order irrelevant;
# --forward makes recipient = weaker model, recorded in each npz):
#   nohup bash scripts/rebuttal/launch_forward_queue.sh \
#         <gpu_csv> <sae_dir> <imp_dir> <matching_file> <cache_dir|-> <out_dir> a:b [a:b ...] \
#         > /tmp/fwd_queue.out 2>&1 </dev/null &
#
# Output: <out_dir>/<pair>/<dataset>.npz  (--resume skips finished datasets).
set -uo pipefail

REPO=/home/brian/src/tabular_embeddings
cd "$REPO"

GPUS_CSV="${1:?gpu csv, e.g. 0,1,2,3,4}"; shift
SAE_DIR="${1:?sae dir}"; shift
IMP_DIR="${1:?importance dir}"; shift
MATCH="${1:?matching file}"; shift
CACHE="${1:?virtual-atoms cache dir (use - for runtime/no cache)}"; shift
OUT="${1:?output dir}"; shift
pairs=("$@")
CACHE_ARG=""; [ "$CACHE" != "-" ] && CACHE_ARG="--virtual-atoms-cache-dir $CACHE"
[ ${#pairs[@]} -eq 0 ] && { echo "No pairs given."; exit 1; }

TFM=/home/brian/anaconda3/envs/tfm/bin/python
TFM2=/home/brian/anaconda3/envs/tfm2/bin/python
IFS=',' read -ra GPUS <<< "$GPUS_CSV"

# Shared work queue + lock (flock for atomic single-line pop across GPU workers).
STAMP=$(date +%s)_$$
Q=/tmp/fwd_queue_${STAMP}.txt
LOCK=/tmp/fwd_queue_${STAMP}.lock
printf '%s\n' "${pairs[@]}" > "$Q"
: > "$LOCK"
echo "queue=$Q  ${#pairs[@]} pairs over GPUs [$GPUS_CSV]: ${pairs[*]}"

pop() {  # atomically remove and echo the first queued pair (empty when drained)
    exec 9>"$LOCK"
    flock 9
    local claimed
    claimed=$(head -n1 "$Q")
    if [ -n "$claimed" ]; then
        tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    fi
    flock -u 9
    echo "$claimed"
}

for g in "${GPUS[@]}"; do
    (
        log=/tmp/fwd_queue_gpu${g}.log
        echo "=== $(date -Iseconds) GPU$g start ===" > "$log"
        while :; do
            pair=$(pop)
            [ -z "$pair" ] && break
            a=${pair%%:*}; b=${pair##*:}
            PY=$TFM; case "$pair" in *tabicl_v2*) PY=$TFM2;; esac
            env_name=$(basename "$(dirname "$(dirname "$PY")")")
            echo "=== $(date -Iseconds) GPU$g $a vs $b (env=$env_name) start ===" >> "$log"
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$PY" -m scripts.rebuttal.transfer_sweep_symmetric \
                --models "$a" "$b" --forward --device cuda --resume \
                --sae-dir "$SAE_DIR" --importance-dir "$IMP_DIR" \
                --matching-file "$MATCH" $CACHE_ARG --output-dir "$OUT" >> "$log" 2>&1
            echo "=== $(date -Iseconds) GPU$g $a vs $b exit=$? ===" >> "$log"
        done
        echo "=== $(date -Iseconds) GPU$g ALL DONE ===" >> "$log"
    ) &
done
wait
rm -f "$Q" "$Q.tmp" "$LOCK"
echo "=== $(date -Iseconds) launch_forward_queue complete ==="
