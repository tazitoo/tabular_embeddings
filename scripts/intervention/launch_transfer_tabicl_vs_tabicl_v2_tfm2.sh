#!/usr/bin/env bash
# Transfer sweep: tabicl vs tabicl_v2 — datasets where tabicl is strong,
# so the weak model (tabicl_v2) receives the tail. Uses tfm2 env.
#
# 13 datasets; mirrors the ablation tfm dataset list (strong-weak flipped
# for transfer because build_tail() runs on the weak model).
#
# Usage:
#   bash scripts/intervention/launch_transfer_tabicl_vs_tabicl_v2_tfm2.sh <worker> [device]
#   bash scripts/intervention/launch_transfer_tabicl_vs_tabicl_v2_tfm2.sh firelord4 cuda

set -euo pipefail

WORKER="${1:?Usage: $0 <worker> [device]}"
DEVICE="${2:-cuda}"
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm2/bin/python"
MODULE="scripts.intervention.transfer_sweep_v2"
LOG="/tmp/transfer_tabicl_vs_tabicl_v2_tfm2.log"
LOCKFILE="/tmp/transfer_tabicl_vs_tabicl_v2_tfm2.lock"

DATASETS=(
  Amazon_employee_access
  Bioresponse
  Diabetes130US
  Is-this-a-good-customer
  MIC
  SDSS17
  bank-marketing
  credit-g
  credit_card_clients_default
  hiva_agnostic
  in_vehicle_coupon_recommendation
  seismic-bumps
  taiwanese_bankruptcy_prediction
)

echo "Launching transfer tabicl_vs_tabicl_v2 (tfm2, ${#DATASETS[@]} datasets) on ${WORKER} device=${DEVICE}"

ssh "${WORKER}" "cd ${REPO} && nohup bash -c '
if [ -f ${LOCKFILE} ]; then echo \"Lock exists ${LOCKFILE}; skipping\"; exit 0; fi
touch ${LOCKFILE}
trap \"rm -f ${LOCKFILE}\" EXIT
export PYTHONPATH=${REPO}
echo \"=== transfer tabicl_vs_tabicl_v2 (tfm2) started \$(date) ===\"
${PY} -m ${MODULE} --models tabicl tabicl_v2 --device ${DEVICE} \
  --output-dir output/transfer_global_mnnp90_trained_tols \
  --virtual-atoms-cache-dir output/transfer_caches/global_trained \
  --matching-file output/sae_feature_matching_mnn_floor_p90.json \
  --resume \
  --datasets ${DATASETS[*]} 2>&1
echo \"=== transfer tabicl_vs_tabicl_v2 (tfm2) done \$(date) ===\"
' > ${LOG} 2>&1 &"

echo "Launched. Monitor with: ssh ${WORKER} tail -f ${LOG}"
