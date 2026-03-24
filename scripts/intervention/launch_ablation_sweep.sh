#!/usr/bin/env bash
# Launch ablation sweep across all 15 unordered model pairs on 2 workers.
#
# The script auto-determines strong/weak per dataset (AUC/RMSE), so each
# unordered pair only needs one run. 6 choose 2 = 15 pairs.
#
# Workers:
#   surfer4  (idle) — tfm env only
#   octo4    (idle) — tfm + tfm2 envs
#
# Constraint: tabicl_v2 pairs need tfm2 (only on octo4).
#   tabicl_v2 pairs: 5 → octo4 (tfm2)
#   remaining: 10 → split between surfer4 (5) and octo4 (5, tfm)
#
# Prerequisites:
#   - Per-row importance for all 6 models on both workers
#   - SAE checkpoints for all 6 models on both workers
#   - git pull on both workers
#
# Usage:
#   bash scripts/intervention/launch_ablation_sweep.sh

set -uo pipefail

REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
PY2="/home/brian/anaconda3/envs/tfm2/bin/python"
MODULE="scripts.intervention.ablation_sweep"

# ── surfer4: 5 pairs (tfm) ───────────────────────────────────────────────────
cat > /tmp/ablation_surfer4.sh << 'EOF'
#!/usr/bin/env bash
set -uo pipefail
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
MODULE="scripts.intervention.ablation_sweep"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

PAIRS=(
  "mitra tabpfn"
  "mitra tabdpt"
  "mitra tabicl"
  "mitra hyperfast"
  "tabpfn tabdpt"
)

echo "=== Ablation sweep on surfer4: ${#PAIRS[@]} pairs ==="
echo "=== Started $(date) ==="

i=0
for pair in "${PAIRS[@]}"; do
  i=$((i + 1))
  read -r a b <<< "$pair"
  echo ""
  echo ">>> [${i}/${#PAIRS[@]}] ${a} vs ${b} ($(date))"
  $PY -m $MODULE --models "$a" "$b" --device cuda --resume 2>&1 | tail -5
done

echo ""
echo "=== Ablation sweep complete on surfer4: $(date) ==="
EOF

# ── octo4: 5 tfm pairs + 5 tfm2 pairs ───────────────────────────────────────
cat > /tmp/ablation_octo4.sh << 'EOF'
#!/usr/bin/env bash
set -uo pipefail
REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
PY2="/home/brian/anaconda3/envs/tfm2/bin/python"
MODULE="scripts.intervention.ablation_sweep"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

# tfm pairs (no tabicl_v2)
TFM_PAIRS=(
  "tabpfn tabicl"
  "tabpfn hyperfast"
  "tabdpt tabicl"
  "tabdpt hyperfast"
  "tabicl hyperfast"
)

# tfm2 pairs (all tabicl_v2)
TFM2_PAIRS=(
  "tabicl_v2 tabpfn"
  "tabicl_v2 tabdpt"
  "tabicl_v2 mitra"
  "tabicl_v2 tabicl"
  "tabicl_v2 hyperfast"
)

total=$((${#TFM_PAIRS[@]} + ${#TFM2_PAIRS[@]}))
echo "=== Ablation sweep on octo4: ${total} pairs ==="
echo "=== Started $(date) ==="

i=0
for pair in "${TFM_PAIRS[@]}"; do
  i=$((i + 1))
  read -r a b <<< "$pair"
  echo ""
  echo ">>> [${i}/${total}] ${a} vs ${b} ($(date))"
  $PY -m $MODULE --models "$a" "$b" --device cuda --resume 2>&1 | tail -5
done

for pair in "${TFM2_PAIRS[@]}"; do
  i=$((i + 1))
  read -r a b <<< "$pair"
  echo ""
  echo ">>> [${i}/${total}] ${a} vs ${b} ($(date))"
  $PY2 -m $MODULE --models "$a" "$b" --device cuda --resume 2>&1 | tail -5
done

echo ""
echo "=== Ablation sweep complete on octo4: $(date) ==="
EOF

echo "Copying scripts to workers..."
scp /tmp/ablation_surfer4.sh brian@surfer4:/tmp/ablation_surfer4.sh
scp /tmp/ablation_octo4.sh brian@octo4:/tmp/ablation_octo4.sh

echo ""
echo "Syncing code..."
ssh brian@surfer4 "cd $REPO && git pull --ff-only"
ssh brian@octo4   "cd $REPO && git pull --ff-only"

echo ""
echo "Launching..."
ssh brian@surfer4 "cd $REPO && nohup bash /tmp/ablation_surfer4.sh > /tmp/ablation_surfer4.log 2>&1 &"
echo "surfer4: 5 pairs (tfm)"
ssh brian@octo4   "cd $REPO && nohup bash /tmp/ablation_octo4.sh > /tmp/ablation_octo4.log 2>&1 &"
echo "octo4: 5 pairs (tfm) + 5 pairs (tfm2)"

echo ""
echo "All workers launched (15 pairs across 2 workers)."
echo ""
echo "Monitor with:"
echo "  ssh surfer4 tail -f /tmp/ablation_surfer4.log"
echo "  ssh octo4   tail -f /tmp/ablation_octo4.log"
