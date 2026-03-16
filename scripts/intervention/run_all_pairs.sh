#!/usr/bin/env bash
# Run full pairwise transfer pipeline: importance → compare → transfer
# for all 28 model pairs on given datasets.
# Usage: bash run_all_pairs.sh
set -uo pipefail

PYTHON="/home/brian/anaconda3/envs/tfm/bin/python"
PYTHON2="/home/brian/anaconda3/envs/tfm2/bin/python"
export PYTHONPATH="/home/brian/src/tabular_embeddings${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(tabpfn tabicl tabicl_v2 carte tabula8b mitra tabdpt hyperfast)
DATASETS=(credit-g kddcup09_appetency)

# Select the right Python for a model
py_for_model() {
  if [[ "$1" == "tabicl_v2" ]]; then
    echo "$PYTHON2"
  else
    echo "$PYTHON"
  fi
}

echo "========================================"
echo "Phase 1: Per-model concept importance"
echo "========================================"
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    echo "--- $m on $d ---"
    PY=$(py_for_model "$m")
    $PY scripts/intervention/concept_importance.py \
      --model "$m" --dataset "$d" --device cuda \
      2>&1 | tail -20
    echo ""
  done
done

echo "========================================"
echo "Phase 2: Pairwise concept comparison"
echo "========================================"
for d in "${DATASETS[@]}"; do
  for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
      a="${MODELS[$i]}"
      b="${MODELS[$j]}"
      echo "--- $a vs $b on $d ---"
      $PYTHON scripts/intervention/concept_importance.py \
        --model "$a" --compare "$b" --dataset "$d" \
        2>&1 | tail -5
      echo ""
    done
  done
done

echo "========================================"
echo "Phase 3: Pairwise transfer + ablation (bidirectional)"
echo "========================================"
total=$((${#MODELS[@]} * (${#MODELS[@]} - 1) / 2 * ${#DATASETS[@]}))
count=0
for d in "${DATASETS[@]}"; do
  for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
      a="${MODELS[$i]}"
      b="${MODELS[$j]}"
      count=$((count+1))
      echo "=== [$count/$total] $a vs $b on $d ==="
      # Use tfm2 python if either model is tabicl_v2
      if [[ "$a" == "tabicl_v2" || "$b" == "tabicl_v2" ]]; then
        PY="$PYTHON2"
      else
        PY="$PYTHON"
      fi
      $PY scripts/intervention/transfer_concepts.py \
        --source "$a" --target "$b" --dataset "$d" \
        --bidirectional --ablation --device cuda \
        2>&1 | tail -20
      echo ""
    done
  done
done

echo "=== All phases complete ==="
