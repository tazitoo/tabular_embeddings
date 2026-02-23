#!/bin/bash
# Run all layerwise extractions for TabArena round 5.
#
# Ordering: fastest models first, slowest last.
# Already-extracted datasets are skipped automatically (per-file check in extract_all_layers.py).
#
# Usage:
#   bash scripts/run_all_extractions.sh              # Full run, all models
#   bash scripts/run_all_extractions.sh carte tabula8b  # Specific models only
#
# Expected runtimes (4090, 51 TabArena datasets):
#   HyperFast:  ~5 min  (38 cls datasets)
#   TabICL:     ~10 min (38 cls datasets)
#   TabPFN:     ~30 min (51 datasets)
#   TabDPT:     ~45 min (51 datasets)
#   CARTE:      ~90 min (51 datasets)
#   Mitra:      ~60 min (51 datasets)
#   Tabula-8B:  ~120 min (51 datasets)

set -uo pipefail

PYTHON="${PYTHON:-/home/brian/anaconda3/envs/tfm/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTRACT="$SCRIPT_DIR/extract_all_layers.py"
OUTPUT_BASE="$PROJECT_ROOT/output/embeddings/tabarena_layerwise_round5"

# All models in fastest-to-slowest order
ALL_MODELS=(hyperfast tabicl tabpfn tabdpt carte mitra tabula8b)

# Use CLI args if provided, otherwise run all
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "============================================================"
echo "TabArena Round 5 — Layerwise Extraction"
echo "============================================================"
echo "  Python:  $PYTHON"
echo "  Models:  ${MODELS[*]}"
echo "  Output:  $OUTPUT_BASE"
echo "  Started: $(date)"
echo ""

# Track results
declare -A STATUS
TOTAL_START=$(date +%s)

for model in "${MODELS[@]}"; do
    echo "============================================================"
    echo "[$model] Starting at $(date +%H:%M:%S)"
    echo "============================================================"

    MODEL_START=$(date +%s)

    if $PYTHON -u "$EXTRACT" --model "$model" --device cuda; then
        MODEL_END=$(date +%s)
        ELAPSED=$(( MODEL_END - MODEL_START ))
        MINS=$(( ELAPSED / 60 ))
        SECS=$(( ELAPSED % 60 ))
        COUNT=$(ls "$OUTPUT_BASE/$model"/tabarena_*.npz 2>/dev/null | wc -l)
        STATUS[$model]="OK (${COUNT} files, ${MINS}m${SECS}s)"
        echo ""
        echo "[$model] Done: ${COUNT} files in ${MINS}m${SECS}s"
    else
        MODEL_END=$(date +%s)
        ELAPSED=$(( MODEL_END - MODEL_START ))
        MINS=$(( ELAPSED / 60 ))
        SECS=$(( ELAPSED % 60 ))
        COUNT=$(ls "$OUTPUT_BASE/$model"/tabarena_*.npz 2>/dev/null | wc -l)
        STATUS[$model]="FAILED (${COUNT} files, ${MINS}m${SECS}s)"
        echo ""
        echo "[$model] FAILED after ${MINS}m${SECS}s"
    fi

    # Clear GPU memory between models
    $PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
printf "%-12s %s\n" "Model" "Status"
echo "------------------------------------------------------------"
for model in "${MODELS[@]}"; do
    printf "%-12s %s\n" "$model" "${STATUS[$model]}"
done
echo ""
echo "Total time: ${TOTAL_MINS}m${TOTAL_SECS}s"
echo "Finished:   $(date)"

# Exit non-zero if any model had failures
ANY_FAILED=0
for model in "${MODELS[@]}"; do
    if [[ "${STATUS[$model]}" == FAILED* ]]; then
        ANY_FAILED=1
    fi
done
exit $ANY_FAILED
