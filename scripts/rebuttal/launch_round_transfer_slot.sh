#!/usr/bin/env bash
# Run the whole transfer stage for a set of pairs on ONE GPU slot, both arms:
#   reverse trained -> reverse random -> forward trained -> forward random
# Same slot for every arm of a pair, so the arms are comparable (host class matters:
# docs/reproducibility.md). All directories come from scripts/round_paths.py.
#
# Usage (through gpu_launch, which pins CUDA_VISIBLE_DEVICES and thread caps):
#   gpu_launch <host> <gpu> /tmp/x.out "bash scripts/rebuttal/launch_round_transfer_slot.sh <gpu> [--arms both|trained|random] a:b [a:b ...]"
# <gpu> is the physical index: launch_forward_queue.sh sets CUDA_VISIBLE_DEVICES itself.
set -uo pipefail

GPU="${1:?physical gpu index}"; shift
ARMS=both     # both | trained | random  (split arms across slots only when both are in the same hardware class)
STAGES=both   # both | reverse | forward (forward deltas do not read the reverse outputs, so the two
              # stages of one arm can run on different slots of the same hardware class)
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
PAIRS=("$@")
[[ ${#PAIRS[@]} -eq 0 ]] && { echo "no pairs"; exit 1; }

REPO=/home/brian/src/tabular_embeddings
TFM=/home/brian/anaconda3/envs/tfm/bin/python
cd "$REPO"
eval "$("$TFM" - <<'EOF'
from scripts.round_paths import (DEFAULT_MATCHING_FILE, FORWARD_DELTAS_DIR, FORWARD_DELTAS_RANDOM_DIR,
    IMPORTANCE_DIR, IMPORTANCE_RANDOM_DIR, RESULTS_DIR, TRANSFER_CACHES_DIR, random_sae_dir, sae_sweep_dir)
print(f"SAE={sae_sweep_dir()} IMP={IMPORTANCE_DIR} MATCH={DEFAULT_MATCHING_FILE} "
      f"CACHE={TRANSFER_CACHES_DIR}/global_trained OUT={FORWARD_DELTAS_DIR} "
      f"RSAE={random_sae_dir()} RIMP={IMPORTANCE_RANDOM_DIR} "
      f"RMATCH={RESULTS_DIR}/sae_feature_matching_mnn_t0.001_random.json "
      f"RCACHE={TRANSFER_CACHES_DIR}/global_random ROUT={FORWARD_DELTAS_RANDOM_DIR}")
EOF
)"

stamp() { echo "=== $(date -Iseconds) slot gpu$GPU: $* ==="; }
if [[ "$ARMS" != random && "$STAGES" != forward ]]; then
  stamp "reverse transfer (trained) ${PAIRS[*]}"
  bash scripts/rebuttal/launch_reverse_queue.sh transfer "${PAIRS[@]}"
fi
if [[ "$ARMS" != trained && "$STAGES" != forward ]]; then
  stamp "reverse transfer (random) ${PAIRS[*]}"
  bash scripts/rebuttal/launch_reverse_queue.sh transfer_random "${PAIRS[@]}"
fi
if [[ "$ARMS" != random && "$STAGES" != reverse ]]; then
  stamp "forward transfer (trained)"
  bash scripts/rebuttal/launch_forward_queue.sh "$GPU" "$SAE" "$IMP" "$MATCH" "$CACHE" "$OUT" "${PAIRS[@]}"
fi
if [[ "$ARMS" != trained && "$STAGES" != reverse ]]; then
  stamp "forward transfer (random)"
  bash scripts/rebuttal/launch_forward_queue.sh "$GPU" "$RSAE" "$RIMP" "$RMATCH" "$RCACHE" "$ROUT" "${PAIRS[@]}"
fi
stamp "TRANSFER SLOT DONE ($ARMS, $STAGES)"
