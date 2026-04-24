#!/usr/bin/env bash
# Transfer sweep: tabicl vs tabicl_v2 — datasets where tabicl_v2 is strong,
# so the weak model (tabicl) receives the tail. Uses tfm env.
#
# 25 datasets; mirrors the ablation tfm2 dataset list (strong-weak flipped
# for transfer because build_tail() runs on the weak model).
#
# Usage:
#   bash scripts/intervention/launch_transfer_tabicl_vs_tabicl_v2_tfm.sh <worker> [device]
#   bash scripts/intervention/launch_transfer_tabicl_vs_tabicl_v2_tfm.sh firelord4 cuda

set -euo pipefail

WORKER="${1:?Usage: $0 <worker> [device]}"
DEVICE="${2:-cuda}"
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
MODULE="scripts.intervention.transfer_sweep_v2"
LOG="/tmp/transfer_tabicl_vs_tabicl_v2_tfm.log"
LOCKFILE="/tmp/transfer_tabicl_vs_tabicl_v2_tfm.lock"

DATASETS=(
  APSFailure
  Bank_Customer_Churn
  E-CommereShippingData
  Fitness_Club
  GiveMeSomeCredit
  HR_Analytics_Job_Change_of_Data_Scientists
  Marketing_Campaign
  NATICUSdroid
  anneal
  blood-transfusion-service-center
  churn
  coil2000_insurance_policies
  customer_satisfaction_in_airline
  diabetes
  hazelnut-spread-contaminant-detection
  heloc
  jm1
  kddcup09_appetency
  maternal_health_risk
  online_shoppers_intention
  polish_companies_bankruptcy
  qsar-biodeg
  splice
  students_dropout_and_academic_success
  website_phishing
)

echo "Launching transfer tabicl_vs_tabicl_v2 (tfm, ${#DATASETS[@]} datasets) on ${WORKER} device=${DEVICE}"

ssh "${WORKER}" "cd ${REPO} && nohup bash -c '
if [ -f ${LOCKFILE} ]; then echo \"Lock exists ${LOCKFILE}; skipping\"; exit 0; fi
touch ${LOCKFILE}
trap \"rm -f ${LOCKFILE}\" EXIT
export PYTHONPATH=${REPO}
echo \"=== transfer tabicl_vs_tabicl_v2 (tfm) started \$(date) ===\"
${PY} -m ${MODULE} --models tabicl tabicl_v2 --device ${DEVICE} \
  --output-dir output/transfer_global_mnnp90_trained_tols \
  --virtual-atoms-cache-dir output/transfer_caches/global_trained \
  --matching-file output/sae_feature_matching_mnn_floor_p90.json \
  --resume \
  --datasets ${DATASETS[*]} 2>&1
echo \"=== transfer tabicl_vs_tabicl_v2 (tfm) done \$(date) ===\"
' > ${LOG} 2>&1 &"

echo "Launched. Monitor with: ssh ${WORKER} tail -f ${LOG}"
