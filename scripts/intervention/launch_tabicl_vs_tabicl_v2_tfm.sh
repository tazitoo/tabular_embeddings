#!/usr/bin/env bash
# Ablation sweep: tabicl vs tabicl_v2 — datasets where tabicl is strong.
# Uses tfm env (tabicl v1 is the strong model that needs a tail).
#
# 13 datasets where tabicl outperforms tabicl_v2 on neg_logloss.
#
# Usage:
#   bash scripts/intervention/launch_tabicl_vs_tabicl_v2_tfm.sh <worker> [device]
#   bash scripts/intervention/launch_tabicl_vs_tabicl_v2_tfm.sh nova4 cuda:0

set -euo pipefail

WORKER="${1:?Usage: $0 <worker> [device]}"
DEVICE="${2:-cuda}"
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
MODULE="scripts.intervention.ablation_sweep"
LOG="/tmp/ablation_tabicl_vs_tabicl_v2_tfm.log"

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

echo "Launching tabicl_vs_tabicl_v2 (tfm, 13 datasets) on ${WORKER} device=${DEVICE}"

ssh "${WORKER}" "cd ${REPO} && nohup bash -c '
export PYTHONPATH=${REPO}
echo \"=== tabicl_vs_tabicl_v2 (tfm) started \$(date) ===\"
${PY} -m ${MODULE} --models tabicl tabicl_v2 --device ${DEVICE} --matching-file output/sae_feature_matching_mnn_floor_p90.json --datasets ${DATASETS[*]} 2>&1
echo \"=== tabicl_vs_tabicl_v2 (tfm) done \$(date) ===\"
' > ${LOG} 2>&1 &"

echo "Launched. Monitor with: ssh ${WORKER} tail -f ${LOG}"
