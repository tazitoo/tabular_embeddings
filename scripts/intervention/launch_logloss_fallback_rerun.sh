#!/usr/bin/env bash
# Targeted rerun after compute_importance_metric gained a neg_logloss
# fallback. The 4 broken-holdout datasets and kddcup09 had degenerate
# stubs because AUC was undefined or Mitra structurally rejected;
# strong/weak is now determinable.
#
# Datasets:
#   - hiva_agnostic, Marketing_Campaign, seismic-bumps,
#     taiwanese_bankruptcy_prediction (single-class holdouts)
#   - kddcup09_appetency (Mitra constant-output, AUC=0.5 vs others)
# Skipped: Bioresponse (architecturally outside Mitra's 500-feature cap;
#          appendix annotation already documents this).
#
# Runs both ablation_sweep and transfer_sweep_v2 for every model pair
# in the configured env. The matching pair script for tfm2 covers
# tabicl_v2-tail cases that fail in tfm.
#
# Usage:
#   bash scripts/intervention/launch_logloss_fallback_rerun.sh <worker> <env> [device]
#   bash scripts/intervention/launch_logloss_fallback_rerun.sh firelord4 tfm cuda
#   bash scripts/intervention/launch_logloss_fallback_rerun.sh firelord4 tfm2 cuda

set -euo pipefail

WORKER="${1:?Usage: $0 <worker> <env: tfm|tfm2> [device]}"
ENV="${2:?Usage: $0 <worker> <env: tfm|tfm2> [device]}"
DEVICE="${3:-cuda}"
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/${ENV}/bin/python"
LOG="/tmp/logloss_fallback_rerun_${ENV}.log"
LOCKFILE="/tmp/logloss_fallback_rerun_${ENV}.lock"

DATASETS=(
  hiva_agnostic
  Marketing_Campaign
  seismic-bumps
  taiwanese_bankruptcy_prediction
  kddcup09_appetency
)

# All 15 classification pairs (sorted within each pair).
PAIRS=(
  "carte mitra"
  "carte tabdpt"
  "carte tabicl"
  "carte tabicl_v2"
  "carte tabpfn"
  "mitra tabdpt"
  "mitra tabicl"
  "mitra tabicl_v2"
  "mitra tabpfn"
  "tabdpt tabicl"
  "tabdpt tabicl_v2"
  "tabdpt tabpfn"
  "tabicl tabicl_v2"
  "tabicl tabpfn"
  "tabicl_v2 tabpfn"
)

echo "Launching logloss-fallback rerun (${ENV}) on ${WORKER} device=${DEVICE}"
echo "  ${#DATASETS[@]} datasets × ${#PAIRS[@]} pairs × 2 sweep types"

# Build the inner command body
INNER='
if [ -f '"${LOCKFILE}"' ]; then echo "Lock exists '"${LOCKFILE}"'; skipping"; exit 0; fi
touch '"${LOCKFILE}"'
trap "rm -f '"${LOCKFILE}"'" EXIT
export PYTHONPATH='"${REPO}"'
echo "=== logloss fallback rerun ('"${ENV}"') started $(date) ==="
'

for pair in "${PAIRS[@]}"; do
  read -r a b <<< "$pair"
  INNER+='
echo "--- ablation '"$a vs $b"' ---"
'"${PY}"' -m scripts.intervention.ablation_sweep \
  --models '"$a"' '"$b"' --device '"${DEVICE}"' --resume \
  --matching-file output/sae_feature_matching_mnn_floor_p90.json \
  --datasets '"${DATASETS[*]}"' 2>&1 || true

echo "--- transfer '"$a vs $b"' ---"
'"${PY}"' -m scripts.intervention.transfer_sweep_v2 \
  --models '"$a"' '"$b"' --device '"${DEVICE}"' --resume \
  --output-dir output/transfer_global_mnnp90_trained_tols \
  --virtual-atoms-cache-dir output/transfer_caches/global_trained \
  --matching-file output/sae_feature_matching_mnn_floor_p90.json \
  --datasets '"${DATASETS[*]}"' 2>&1 || true
'
done

INNER+='echo "=== logloss fallback rerun ('"${ENV}"') done $(date) ==="
'

ssh "${WORKER}" "cd ${REPO} && nohup bash -c '${INNER}' > ${LOG} 2>&1 &"
echo "Launched. Monitor with: ssh ${WORKER} tail -f ${LOG}"
