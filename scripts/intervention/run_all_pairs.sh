#!/usr/bin/env bash
# Run full pairwise transfer pipeline: importance → compare → transfer
# for all 28 model pairs on given datasets.
# Usage: bash run_all_pairs.sh
set -uo pipefail

PYTHON="/home/brian/anaconda3/envs/tfm/bin/python"
export PYTHONPATH="/home/brian/src/tabular_embeddings${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(tabpfn tabicl tabicl_v2 carte tabula8b mitra tabdpt hyperfast)
DATASETS=(credit-g kddcup09_appetency)

echo "========================================"
echo "Phase 1: Per-model concept importance"
echo "========================================"
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    out="output/concept_importance/${m}_${d}.json"
    if [[ -f "$out" ]]; then
      echo "SKIP: $m on $d (already exists)"
      continue
    fi
    echo "--- $m on $d ---"
    $PYTHON scripts/intervention/concept_importance.py \
      --model "$m" --dataset "$d" --device cuda \
      2>&1 | tail -5
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
      out="output/concept_importance/compare_${a}_vs_${b}_${d}.json"
      if [[ -f "$out" ]]; then
        echo "SKIP: $a vs $b on $d (already exists)"
        continue
      fi
      echo "--- $a vs $b on $d ---"
      $PYTHON scripts/intervention/concept_importance.py \
        --model "$a" --compare "$b" --dataset "$d" \
        2>&1 | tail -5
      echo ""
    done
  done
done

echo "========================================"
echo "Phase 3: Pairwise transfer (bidirectional)"
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
      $PYTHON scripts/intervention/transfer_concepts.py \
        --source "$a" --target "$b" --dataset "$d" \
        --bidirectional --device cuda \
        2>&1 | tail -20
      echo ""
    done
  done
done

echo "=== All phases complete ==="
