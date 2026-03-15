#!/usr/bin/env bash
# Run all 28 pairwise transfer experiments for given datasets.
# Usage: bash run_all_pairs.sh
set -euo pipefail

PYTHON="/home/brian/anaconda3/envs/tfm/bin/python"
MODELS=(tabpfn tabicl tabicl_v2 carte tabula8b mitra tabdpt hyperfast)
DATASETS=(credit-g kddcup09_appetency)

total=0
for d in "${DATASETS[@]}"; do
  for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
      total=$((total+1))
    done
  done
done

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

echo "=== All $total pairs complete ==="
