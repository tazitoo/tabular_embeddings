#!/usr/bin/env bash
# Launch ablation sweep across all 30 ordered model pairs on 2 workers.
#
# Workers:
#   surfer4  (192.168.10.6)  — tfm env only
#   octo4    (192.168.10.12) — tfm + tfm2 envs
#
# Pair assignment (30 total):
#   octo4  tfm2 (10): all tabicl_v2 pairs (constraint: tfm2 only on octo4)
#   octo4  tfm  (10): tabdpt-strong x2, tabicl-strong x4, hyperfast-strong x4
#   surfer4 tfm (10): mitra-strong x4, tabpfn-strong x4, tabdpt-strong x2
#
# Data requirements synced before launch (see prep commands below).
#
# Usage:
#   1. Run the sync commands (printed by this script or in prep section)
#   2. bash scripts/intervention/launch_ablation_sweep.sh

set -uo pipefail

REPO="/home/brian/src/tabular_embeddings"
PY="/home/brian/anaconda3/envs/tfm/bin/python"
PY2="/home/brian/anaconda3/envs/tfm2/bin/python"
PY_MODULE="scripts.intervention.ablation_sweep"

# 40GbE addresses
SURFER4="192.168.10.6"
OCTO4="192.168.10.12"
TERRAX4="192.168.10.4"

# ── Data sync commands (run these BEFORE launching) ──────────────────────────
#
# These use rsync over the 40GbE network for large binary files (.npz, .pt).
# Code sync uses git pull (never rsync for code).
#
# 1. SAE checkpoints: octo4 → surfer4 (all 6 models surfer4 will need)
#    surfer4 has NO round-10 SAE checkpoints; needs all 6 models.
#
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/sae_tabarena_sweep_round10/{tabpfn,tabdpt,mitra,tabicl,tabicl_v2,hyperfast}/ \
#      $SURFER4:$REPO/output/sae_tabarena_sweep_round10/ \
#      --include='*/' --include='sae_matryoshka_archetypal_validated.pt' --exclude='*'"
#
#    Or more explicitly, one model at a time:
#    for model in tabpfn tabdpt mitra tabicl tabicl_v2 hyperfast; do
#      ssh $OCTO4 "rsync -avz --progress \
#        $REPO/output/sae_tabarena_sweep_round10/\$model/sae_matryoshka_archetypal_validated.pt \
#        $SURFER4:$REPO/output/sae_tabarena_sweep_round10/\$model/"
#    done
#
# 2. Perrow importance: octo4 → surfer4 (tabpfn + tabdpt, needed as strong models)
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/perrow_importance/tabpfn/ \
#      $SURFER4:$REPO/output/perrow_importance/tabpfn/"
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/perrow_importance/tabdpt/ \
#      $SURFER4:$REPO/output/perrow_importance/tabdpt/"
#
# 3. Perrow importance: terrax4 → octo4 (tabicl_v2, needed as strong model)
#    ssh $TERRAX4 "rsync -avz --progress \
#      $REPO/output/perrow_importance/tabicl_v2/ \
#      $OCTO4:$REPO/output/perrow_importance/tabicl_v2/"
#
# 4. Perrow importance: surfer4 → octo4 (mitra, needed for (mitra, tabicl_v2) pair)
#    ssh $SURFER4 "rsync -avz --progress \
#      $REPO/output/perrow_importance/mitra/ \
#      $OCTO4:$REPO/output/perrow_importance/mitra/"
#
# 5. Test embeddings + norm stats: ensure surfer4 has all 6 models' data
#    (these are in output/sae_training_round10/)
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/sae_training_round10/*_taskaware_sae_test.npz \
#      $SURFER4:$REPO/output/sae_training_round10/"
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/sae_training_round10/*_taskaware_norm_stats.npz \
#      $SURFER4:$REPO/output/sae_training_round10/"
#    # tabicl_v2 test data may be on terrax4 only:
#    ssh $TERRAX4 "rsync -avz --progress \
#      $REPO/output/sae_training_round10/tabicl_v2_taskaware_sae_test.npz \
#      $OCTO4:$REPO/output/sae_training_round10/"
#    ssh $TERRAX4 "rsync -avz --progress \
#      $REPO/output/sae_training_round10/tabicl_v2_taskaware_norm_stats.npz \
#      $OCTO4:$REPO/output/sae_training_round10/"
#
# 6. Preprocessing cache: surfer4 needs cache for all 6 models
#    ssh $OCTO4 "rsync -avz --progress \
#      $REPO/output/preprocessing_cache/ \
#      $SURFER4:$REPO/output/preprocessing_cache/"
#
# 7. Code sync on both workers:
#    ssh $SURFER4 "cd $REPO && git pull --ff-only"
#    ssh $OCTO4   "cd $REPO && git pull --ff-only"
#
# ─────────────────────────────────────────────────────────────────────────────

# Helper: generate worker script for a list of (strong, weak) pairs
make_pair_script() {
  local python="$1"
  local worker="$2"
  shift 2
  # Remaining args are "strong:weak" pairs

  local pairs=("$@")

  cat <<SCRIPT
export PYTHONPATH="${REPO}\${PYTHONPATH:+:\$PYTHONPATH}"

echo "=== Ablation sweep on ${worker}: ${#pairs[@]} pairs ==="
echo "=== Started \$(date) ==="

SCRIPT

  local i=0
  for pair in "${pairs[@]}"; do
    i=$((i + 1))
    local strong="${pair%%:*}"
    local weak="${pair##*:}"
    cat <<SCRIPT
echo ""
echo ">>> [$i/${#pairs[@]}] ${strong} vs ${weak} (\$(date))"
${python} -m ${PY_MODULE} --strong ${strong} --weak ${weak} --device cuda --resume 2>&1 | tail -5

SCRIPT
  done

  cat <<SCRIPT
echo ""
echo "=== Ablation sweep complete on ${worker}: \$(date) ==="
SCRIPT
}

# ── Pair definitions ─────────────────────────────────────────────────────────

# surfer4 (tfm): 10 pairs
# mitra-strong (4) — importance data already local
# tabpfn-strong (4) — importance synced from octo4
# tabdpt-strong (2) — importance synced from octo4
SURFER4_PAIRS=(
  "mitra:tabpfn"
  "mitra:tabdpt"
  "mitra:tabicl"
  "mitra:hyperfast"
  "tabpfn:tabdpt"
  "tabpfn:mitra"
  "tabpfn:tabicl"
  "tabpfn:hyperfast"
  "tabdpt:tabpfn"
  "tabdpt:mitra"
)

# octo4 tfm: 10 pairs
# tabdpt-strong (2) — importance already local
# tabicl-strong (4) — importance already local
# hyperfast-strong (4) — importance already local
OCTO4_TFM_PAIRS=(
  "tabdpt:tabicl"
  "tabdpt:hyperfast"
  "tabicl:tabpfn"
  "tabicl:tabdpt"
  "tabicl:mitra"
  "tabicl:hyperfast"
  "hyperfast:tabpfn"
  "hyperfast:tabdpt"
  "hyperfast:mitra"
  "hyperfast:tabicl"
)

# octo4 tfm2: 10 pairs (all tabicl_v2 involvement)
# tabicl_v2-strong (5) — importance synced from terrax4
# X-strong where weak=tabicl_v2 (5) — importance already local (except mitra → synced from surfer4)
OCTO4_TFM2_PAIRS=(
  "tabicl_v2:tabpfn"
  "tabicl_v2:tabdpt"
  "tabicl_v2:mitra"
  "tabicl_v2:tabicl"
  "tabicl_v2:hyperfast"
  "tabpfn:tabicl_v2"
  "tabdpt:tabicl_v2"
  "mitra:tabicl_v2"
  "tabicl:tabicl_v2"
  "hyperfast:tabicl_v2"
)

echo "Launching ablation sweep on 2 workers (30 pairs total)..."
echo ""

# ── surfer4: 10 pairs (tfm) ─────────────────────────────────────────────────
echo "surfer4: 10 pairs (tfm) — mitra-strong x4, tabpfn-strong x4, tabdpt-strong x2"
ssh ${SURFER4} "cd ${REPO} && nohup bash -c '$(make_pair_script "$PY" "surfer4" "${SURFER4_PAIRS[@]}")' > /tmp/ablation_surfer4.log 2>&1 &"

# ── octo4: 10 tfm pairs, then 10 tfm2 pairs (sequential on same GPU) ───────
# Run tfm pairs first, then tfm2 pairs (different python binary).
echo "octo4: 10 pairs (tfm) + 10 pairs (tfm2) — sequential"
ssh ${OCTO4} "cd ${REPO} && nohup bash -c '
$(make_pair_script "$PY" "octo4" "${OCTO4_TFM_PAIRS[@]}")
$(make_pair_script "$PY2" "octo4" "${OCTO4_TFM2_PAIRS[@]}")
' > /tmp/ablation_octo4.log 2>&1 &"

echo ""
echo "All workers launched (30 pairs across 2 workers)."
echo ""
echo "Monitor with:"
echo "  ssh surfer4 tail -f /tmp/ablation_surfer4.log"
echo "  ssh octo4   tail -f /tmp/ablation_octo4.log"
echo ""
echo "Check progress:"
echo "  ssh surfer4 'ls $REPO/output/ablation_sweep/*/  | wc -l'"
echo "  ssh octo4   'ls $REPO/output/ablation_sweep/*/  | wc -l'"
echo ""
echo "Expected output: output/ablation_sweep/{strong}_vs_{weak}/{dataset}.npz"
