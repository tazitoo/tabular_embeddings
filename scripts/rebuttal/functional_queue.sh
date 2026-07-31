#!/usr/bin/env bash
# Round-robin functional_decomposition over a host's GPUs. Deltas must already be
# local on the host (no sync here). Per-pair env: tfm2 iff the pair involves
# tabicl_v2, else tfm.
#
# Usage (on the host):
#   nohup bash scripts/rebuttal/functional_queue.sh <gpu_csv> <delta_dir> <out_dir> a_vs_b [a_vs_b ...] \
#         > /tmp/fq.out 2>&1 </dev/null &
set -uo pipefail
REPO=/home/brian/src/tabular_embeddings; cd "$REPO"
GPUS_CSV="${1:?gpu csv}"; DELTA="${2:?delta dir}"; OUT="${3:?out dir}"; shift 3
pairs=("$@")
VT="${VAR_THRESHOLD:-0.90}"   # on-manifold cumulative-variance threshold (sweep via env)
[ ${#pairs[@]} -eq 0 ] && { echo "no pairs"; exit 1; }
IFS=',' read -ra GPUS <<< "$GPUS_CSV"; NG=${#GPUS[@]}
TFM=/home/brian/anaconda3/envs/tfm/bin/python
TFM2=/home/brian/anaconda3/envs/tfm2/bin/python

for idx in "${!GPUS[@]}"; do
  g=${GPUS[$idx]}; gp=()
  for i in "${!pairs[@]}"; do [ $((i % NG)) -eq "$idx" ] && gp+=("${pairs[$i]}"); done
  [ ${#gp[@]} -eq 0 ] && continue
  (
    for pair in "${gp[@]}"; do
      a=${pair%%_vs_*}; b=${pair##*_vs_}; PY=$TFM
      case "$pair" in *tabicl_v2*) PY=$TFM2;; esac
      echo "=== $(date -Iseconds) GPU$g $pair start ===" >> /tmp/fq_gpu${g}.log
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" -m scripts.rebuttal.functional_decomposition --models "$a" "$b" \
        --device cuda --delta-dir "$DELTA" --output-dir "$OUT" --var-threshold "$VT" >> /tmp/fq_gpu${g}.log 2>&1
      echo "=== $(date -Iseconds) GPU$g $pair exit=$? ===" >> /tmp/fq_gpu${g}.log
    done
  ) &
done
wait
echo "=== $(date -Iseconds) functional_queue complete ==="
