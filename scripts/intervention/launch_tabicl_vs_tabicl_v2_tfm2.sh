#!/usr/bin/env bash
# Ablation sweep: tabicl vs tabicl_v2 — datasets where tabicl_v2 is strong.
# Uses tfm2 env (tabicl_v2 is the strong model that needs a tail).
#
# 25 datasets where tabicl_v2 outperforms tabicl on neg_logloss.
#
# Usage:
#   bash scripts/intervention/launch_tabicl_vs_tabicl_v2_tfm2.sh <worker> [device]
#   bash scripts/intervention/launch_tabicl_vs_tabicl_v2_tfm2.sh nova4 cuda:1

set -euo pipefail

WORKER="${1:?Usage: $0 <worker> [device]}"
DEVICE="${2:-cuda}"
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm2/bin/python"
MODULE="scripts.intervention.ablation_sweep"
LOG="/tmp/ablation_tabicl_vs_tabicl_v2_tfm2.log"

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

echo "Launching tabicl_vs_tabicl_v2 (tfm2, 25 datasets) on ${WORKER} device=${DEVICE}"

ssh "${WORKER}" "cd ${REPO} && nohup bash -c '
export PYTHONPATH=${REPO}
echo \"=== tabicl_vs_tabicl_v2 (tfm2) started \$(date) ===\"
${PY} -m ${MODULE} --models tabicl tabicl_v2 --device ${DEVICE} --matching-file output/sae_feature_matching_mnn_floor_p90.json --datasets ${DATASETS[*]} 2>&1
echo \"=== tabicl_vs_tabicl_v2 (tfm2) done \$(date) ===\"
' > ${LOG} 2>&1 &"

echo "Launched. Monitor with: ssh ${WORKER} tail -f ${LOG}"
