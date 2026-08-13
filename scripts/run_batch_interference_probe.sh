#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch-vs-batch interference probe -- sequential driver for the three
# memorization-sweep arms.
#
# WHAT THIS IS: a thin launcher, exactly like tests/run_memorization_sweep.sh.
# It owns NO measurement knobs; everything the probe does lives in
# scripts/batch_interference_probe.py and in each arm's config. It reads the
# arms' finished `last.pt` checkpoints and writes only to
# scripts/output/batch_interference/<arm>/. No training state is modified.
#
# WHAT IS BEING MEASURED: for each arm, one optimizer step is taken on each of
# the epoch's batches in turn (from the same restored checkpoint every time) and
# the loss change that step causes on EVERY batch is recorded. A step that helps
# its own batch while hurting others is the signature of a capacity-limited
# model; see the module docstring of batch_interference_probe.py.
#
# RUNTIME: ~34 batches per arm means ~34 x 34 measured losses plus 34 steps, so
# roughly 45-60 minutes per arm on a single RTX 3070. The arms run back to back.
#
# The probe holds the whole frozen epoch in VRAM when it fits; 2.5 GiB covers
# 34 batches of this profile and still leaves headroom for fp32 evaluation.
# ---------------------------------------------------------------------------

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

ARMS=(
    configs/memorization_01_no_regularization.yaml
    configs/memorization_02_t_one_oversample.yaml
    configs/memorization_03_no_regularization_plus_t_one_oversample.yaml
)

LOG_DIR="scripts/output/batch_interference"
mkdir -p "$LOG_DIR"

for arm_config in "${ARMS[@]}"; do
    arm_name="$(basename "$arm_config" .yaml)"
    echo "=== ${arm_name} started $(date -Is) ==="
    ./.venv/Scripts/python.exe scripts/batch_interference_probe.py \
        --config "$arm_config" \
        --batch-cache-gb 2.5 \
        2>&1 | tee "${LOG_DIR}/${arm_name}-console.log"
    status=${PIPESTATUS[0]}
    echo "=== ${arm_name} finished $(date -Is) exit=${status} ==="
done
