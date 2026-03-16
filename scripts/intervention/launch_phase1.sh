#!/usr/bin/env bash
# Phase 1: Per-model concept importance across all 51 TabArena datasets.
#
# Worker assignment (approach A — partition by model):
#   firelord4: tabula8b
#   terrax4:   hyperfast, tabicl_v2
#   surfer4:   mitra, carte
#   octo4:     tabpfn, tabicl, tabdpt
#
# Usage: bash scripts/intervention/launch_phase1.sh
#   Launches all 4 workers in parallel via SSH + nohup.
#   Logs: /tmp/phase1_<worker>.log on each worker.
#   Monitor: ssh <worker> tail -f /tmp/phase1_<worker>.log

set -uo pipefail

REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
PY2="/home/brian/anaconda3/envs/tfm2/bin/python"
SCRIPT="scripts/intervention/concept_importance.py"

# Regression datasets need --task regression
REGRESSION=(
  "Another-Dataset-on-used-Fiat-500"
  "Food_Delivery_Time"
  "QSAR-TID-11"
  "QSAR_fish_toxicity"
  "airfoil_self_noise"
  "concrete_compressive_strength"
  "diamonds"
  "healthcare_insurance_expenses"
  "houses"
  "miami_housing"
  "physiochemical_protein"
  "superconductivity"
  "wine_quality"
)

ALL_DATASETS=(
  "APSFailure" "Amazon_employee_access" "Another-Dataset-on-used-Fiat-500"
  "Bank_Customer_Churn" "Bioresponse" "Diabetes130US"
  "E-CommereShippingData" "Fitness_Club" "Food_Delivery_Time"
  "GiveMeSomeCredit" "HR_Analytics_Job_Change_of_Data_Scientists"
  "Is-this-a-good-customer" "MIC" "Marketing_Campaign" "NATICUSdroid"
  "QSAR-TID-11" "QSAR_fish_toxicity" "SDSS17" "airfoil_self_noise"
  "anneal" "bank-marketing" "blood-transfusion-service-center" "churn"
  "coil2000_insurance_policies" "concrete_compressive_strength" "credit-g"
  "credit_card_clients_default" "customer_satisfaction_in_airline" "diabetes"
  "diamonds" "hazelnut-spread-contaminant-detection"
  "healthcare_insurance_expenses" "heloc" "hiva_agnostic" "houses"
  "in_vehicle_coupon_recommendation" "jm1" "kddcup09_appetency"
  "maternal_health_risk" "miami_housing" "online_shoppers_intention"
  "physiochemical_protein" "polish_companies_bankruptcy" "qsar-biodeg"
  "seismic-bumps" "splice" "students_dropout_and_academic_success"
  "superconductivity" "taiwanese_bankruptcy_prediction" "website_phishing"
  "wine_quality"
)

# Build the inner loop as a shell function string
# Each worker gets: MODELS array, PYTHON command, and the dataset loop
make_worker_script() {
  local models="$1"    # space-separated model keys
  local python="$2"    # python binary to use
  local worker="$3"    # worker hostname (for log)

  cat <<SCRIPT
export PYTHONPATH="${REPO}\${PYTHONPATH:+:\$PYTHONPATH}"

# Regression dataset lookup
is_regression() {
  case "\$1" in
    Another-Dataset-on-used-Fiat-500|Food_Delivery_Time|QSAR-TID-11|\
QSAR_fish_toxicity|airfoil_self_noise|concrete_compressive_strength|\
diamonds|healthcare_insurance_expenses|houses|miami_housing|\
physiochemical_protein|superconductivity|wine_quality)
      return 0 ;;
    *) return 1 ;;
  esac
}

MODELS=(${models})
DATASETS=($(printf '%s ' "${ALL_DATASETS[@]}"))

echo "=== Phase 1 on ${worker}: \${MODELS[*]} ==="
echo "=== \${#DATASETS[@]} datasets, started \$(date) ==="

for m in "\${MODELS[@]}"; do
  echo ""
  echo ">>> Model: \$m (\$(date))"
  for d in "\${DATASETS[@]}"; do
    # Skip if output already exists
    outfile="${REPO}/output/interventions/\${d}/importance/\${m}.json"
    if [ -f "\$outfile" ]; then
      echo "  [SKIP] \$m on \$d (exists)"
      continue
    fi

    task="classification"
    if is_regression "\$d"; then
      task="regression"
    fi

    echo "  [\$(date +%H:%M)] \$m on \$d (task=\$task)"
    ${python} ${REPO}/${SCRIPT} \\
      --model "\$m" --dataset "\$d" --task "\$task" --device cuda \\
      2>&1 | tail -3
  done
done

echo ""
echo "=== Phase 1 complete on ${worker}: \$(date) ==="
SCRIPT
}

echo "Launching Phase 1 on all workers..."
echo ""

# --- firelord4: tabula8b ---
echo "firelord4: tabula8b"
ssh 192.168.10.8 "cd ${REPO} && nohup bash -c '$(make_worker_script "tabula8b" "$PY" "firelord4")' > /tmp/phase1_firelord4.log 2>&1 &"

# --- terrax4: hyperfast, tabicl_v2 ---
# tabicl_v2 needs tfm2 env; run hyperfast first (tfm), then tabicl_v2 (tfm2)
echo "terrax4: hyperfast (tfm), tabicl_v2 (tfm2)"
ssh 192.168.10.4 "cd ${REPO} && nohup bash -c '
$(make_worker_script "hyperfast" "$PY" "terrax4")
$(make_worker_script "tabicl_v2" "$PY2" "terrax4")
' > /tmp/phase1_terrax4.log 2>&1 &"

# --- surfer4: mitra, carte ---
echo "surfer4: mitra, carte"
ssh 192.168.10.6 "cd ${REPO} && nohup bash -c '$(make_worker_script "mitra carte" "$PY" "surfer4")' > /tmp/phase1_surfer4.log 2>&1 &"

# --- octo4: tabpfn, tabicl, tabdpt ---
echo "octo4: tabpfn, tabicl, tabdpt"
ssh 192.168.10.12 "cd ${REPO} && nohup bash -c '$(make_worker_script "tabpfn tabicl tabdpt" "$PY" "octo4")' > /tmp/phase1_octo4.log 2>&1 &"

echo ""
echo "All workers launched. Monitor with:"
echo "  ssh firelord4 tail -f /tmp/phase1_firelord4.log"
echo "  ssh terrax4   tail -f /tmp/phase1_terrax4.log"
echo "  ssh surfer4   tail -f /tmp/phase1_surfer4.log"
echo "  ssh octo4     tail -f /tmp/phase1_octo4.log"
