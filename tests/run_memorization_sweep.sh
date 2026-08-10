#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Memorization-probe sweep driver -- runs the three arms of the "can it TRULY
# memorize?" experiment back to back against the overfitV2 profile and the same
# 10-train / 3-dev small corpus.
#
# WHAT THIS IS: a thin sequential launcher, a direct sibling of
# tests/run_ablation_sweep.sh and built the same way. It owns NO training knobs.
# Every knob -- replay subset, model scale, LR schedule, batch size -- lives in
# the per-arm YAML profiles in configs/memorization_*.yaml, each of which extends
# configs/local_overfit_v2.yaml and differs from it ONLY by the knob(s) under
# test and by its redirected storage paths.
#
# THE THREE ARMS:
#   1  regularization off        weight_decay 0.1 -> 0.0, fog 0..0.8 -> 0..0.0
#   2  t=1 oversampled           t_one_fraction 0.0 -> 0.25
#   3  both                      all three of the above at once
#
# All three inherit `model.frozen_input_kv: true` from config/default.yaml, where
# it became a project-wide default on 2026-08-09. Their architecture identity is
# therefore `dense-multinomial-SC2-v2+frozen_input_kv`, which is the SAME
# identity ablation arm 01 was trained under -- so
# tests/output/ablations/01-frozen-input-kv-only/epoch_metrics.csv is the
# comparison run for all three, not the pre-promotion all-false baseline.
#
# RUN LENGTH: each arm runs 100 epochs = 3400 optimizer steps (34 per epoch),
# owned entirely by `train.epochs: 100` in configs/local_overfit_v2.yaml. This
# driver passes NO `--max-steps`, so each arm completes both of its
# config-derived schedules: the linear LR decay reaches its 9.0e-6 floor at the
# last step, and the EMA averaging window (10% of the run = 340 steps) turns over
# ~10 times. STEPS_PER_EPOCH x EXPECTED_EPOCHS below is used ONLY to decide skip /
# resume, never as a cap, and tests/test_config.py pins both numbers to the
# loaded config so they cannot silently desync.
#
# ---------------------------------------------------------------------------
# THIS SCRIPT IS RESTARTABLE. Re-running it resumes the sweep; it never repeats
# an arm that already finished.
#
# There is no separate state file to keep in sync, because the CHECKPOINTS ARE
# THE STATE. Before launching anything, each arm's own `last.pt` is probed for
# its `global_step`:
#
#   no last.pt              -> arm never started; run it from scratch.
#   global_step >= TOTAL    -> arm already finished its 100 epochs; SKIP it
#                              entirely (no Python launch, no GPU time).
#   0 < global_step < TOTAL -> arm was interrupted; run it. train_pipeline's
#                              own _try_resume() restores model, EMA, optimizer,
#                              scheduler, global_step, completed_epochs and the
#                              in-epoch batch offset, so training continues where
#                              it stopped. At most the steps since that arm's
#                              last per-epoch checkpoint are redone.
#   unreadable last.pt      -> SKIP with a loud warning rather than silently
#                              discarding a partly-trained arm.
#
# ONE IMPORTANT DIFFERENCE FROM THE ABLATION SWEEP: those five arms each changed
# the model's architecture_identity, so a cross-arm checkpoint mixup would abort
# on a fingerprint mismatch. THESE three arms are architecturally IDENTICAL to
# each other -- weight decay, fog, and t_one_fraction are training/data knobs,
# not architecture -- so nothing in the loader would catch one arm resuming from
# another's weights. The private `storage.*` paths in each YAML are the ONLY
# thing preventing that. Do not point two arms at one checkpoint_uri.
# ---------------------------------------------------------------------------
#
# RESILIENCE: an arm that crashes does not abort the sweep. Its exit code is
# recorded and the driver moves to the next arm, because the point of running
# overnight is to come back to usable results rather than to one crash.
#
# OUTPUTS, per arm, under tests/output/memorization/<arm-name>/:
#   console-<timestamp>.log  full stdout+stderr of the run (one per launch, so a
#                            resumed arm keeps its earlier logs)
#   epoch_metrics.csv        1 row per epoch, train + dev, every loss class
#   step_metrics.jsonl       1 line per optimizer step
# plus a sweep-level tests/output/memorization/SWEEP_STATUS.md updated as each arm
# starts and finishes, so progress is readable while the sweep is still running.
# ---------------------------------------------------------------------------
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON=".venv/Scripts/python.exe"
SWEEP_DIR="tests/output/memorization"
STATUS_FILE="$SWEEP_DIR/SWEEP_STATUS.md"

# The step count a COMPLETED arm reaches: 34 train batches/epoch x 100 epochs.
# NOT passed to the trainer -- the arms get their length from `train.epochs` in
# configs/local_overfit_v2.yaml. It exists only so probe_arm's global_step can be
# turned into a skip / resume / fresh decision, so it must be kept in step with
# that config. Both values are asserted against the loaded config in
# tests/test_config.py.
STEPS_PER_EPOCH=34
EXPECTED_EPOCHS=100
TOTAL_STEPS=$((STEPS_PER_EPOCH * EXPECTED_EPOCHS))

# arm-name : config-file. Order is the order they run in.
ARMS=(
  "01-no-regularization:configs/memorization_01_no_regularization.yaml"
  "02-t-one-oversample:configs/memorization_02_t_one_oversample.yaml"
  "03-no-regularization-plus-t-one:configs/memorization_03_no_regularization_plus_t_one_oversample.yaml"
)

mkdir -p "$SWEEP_DIR"

# ---------------------------------------------------------------------------
# probe_arm <config-path>
#
# Prints one line: "<checkpoint_dir> <global_step>".
#
# The checkpoint directory is read from the arm's own YAML via load_config rather
# than reconstructed from the arm name, so the driver can never disagree with the
# profile about where an arm's weights live.
#
# global_step is:
#     0  when the arm has no last.pt yet (never started)
#    >0  the optimizer step the arm last checkpointed at
#    -1  when last.pt exists but cannot be read (truncated / corrupt)
#
# Calls: thesis_ml.config.load_config, torch.load. Read-only -- it never writes
# or mutates a checkpoint.
# ---------------------------------------------------------------------------
probe_arm() {
  "$PYTHON" -c '
import sys
from pathlib import Path

import torch

from thesis_ml.config import load_config

config_path = Path(sys.argv[1])
checkpoint_dir = load_config(config_path).storage.checkpoint_uri
checkpoint = Path(checkpoint_dir) / "last.pt"

if not checkpoint.exists():
    global_step = 0
else:
    try:
        # weights_only=False is REQUIRED, not lazy: save_checkpoint stores the
        # pickled ProjectConfig dataclass alongside the tensors, and
        # weights_only=True rejects the whole payload when any entry is not a
        # plain tensor/primitive. This matches every torch.load call in the
        # package. The file being read was written by the training loop in this
        # same repository, to a repo-local path -- never third-party input.
        #
        # NOTE: no apostrophes anywhere inside this python block. It is passed
        # to python -c inside a SINGLE-QUOTED bash string, so one apostrophe
        # ends the string early and breaks the script.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        global_step = int(payload.get("global_step", 0))
    except Exception:
        global_step = -1

print(f"{checkpoint_dir} {global_step}")
' "$1"
}

# Rewrite the status file from scratch whenever an arm changes state. Cheap
# (3 rows) and it means the file is never left half-written.
declare -a ARM_STATE ARM_START ARM_END ARM_EXIT ARM_NOTE
for arm_index in "${!ARMS[@]}"; do
  ARM_STATE[$arm_index]="pending"
  ARM_START[$arm_index]="-"
  ARM_END[$arm_index]="-"
  ARM_EXIT[$arm_index]="-"
  ARM_NOTE[$arm_index]="-"
done

# NOTE FOR MAINTAINERS: every loop variable in this function is named
# `status_index`, NOT `index`. Bash has no function-local scope unless declared,
# so reusing `index` here would silently clobber the caller's loop counter and
# every arm's completion would be recorded into the last arm's row. (That exact
# bug happened in tests/run_ablation_sweep.sh.) Keep the names distinct.
write_status() {
  local status_index status_name
  {
    echo "# Memorization-probe sweep status"
    echo
    echo "Sweep driver last started: $SWEEP_STARTED"
    echo "Last updated:              $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "Each arm: configs/local_overfit_v2.yaml + the named knob change(s), $EXPECTED_EPOCHS epochs"
    echo "($TOTAL_STEPS optimizer steps, run to completion -- no \`--max-steps\` cap, so the"
    echo "linear LR decay and the EMA window both finish), same 10-train / 3-dev small corpus."
    echo
    echo "All three inherit \`model.frozen_input_kv: true\` from config/default.yaml, so the"
    echo "comparison run is ablation arm 01 (\`tests/output/ablations/01-frozen-input-kv-only/\`),"
    echo "which shares that architecture, subset, and schedule."
    echo
    echo "This driver is restartable: re-running \`tests/run_memorization_sweep.sh\` skips"
    echo "arms that already reached $TOTAL_STEPS steps and resumes any arm that was"
    echo "interrupted, from its own last checkpoint."
    echo
    echo "| # | arm | state | started | finished | exit | note |"
    echo "|---|-----|-------|---------|----------|------|------|"
    for status_index in "${!ARMS[@]}"; do
      status_name="${ARMS[$status_index]%%:*}"
      echo "| $((status_index + 1)) | $status_name | ${ARM_STATE[$status_index]} | ${ARM_START[$status_index]} | ${ARM_END[$status_index]} | ${ARM_EXIT[$status_index]} | ${ARM_NOTE[$status_index]} |"
    done
    echo
    echo "Per-arm artifacts live in \`$SWEEP_DIR/<arm>/\` (console log, epoch_metrics.csv, step_metrics.jsonl)."
  } > "$STATUS_FILE"
}

SWEEP_STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
write_status

# Unbuffered stdio so a traceback reaches the console log instead of dying in a
# buffer when a run is killed.
export PYTHONUNBUFFERED=1

ARM_COUNT="${#ARMS[@]}"

echo "=========================================================================="
echo "MEMORIZATION SWEEP  started $SWEEP_STARTED"
echo "Probing each arm's checkpoint to decide skip / resume / fresh..."
echo "=========================================================================="

for index in "${!ARMS[@]}"; do
  entry="${ARMS[$index]}"
  arm_name="${entry%%:*}"
  arm_config="${entry#*:}"
  arm_dir="$SWEEP_DIR/$arm_name"
  mkdir -p "$arm_dir"

  # ---- decide what to do with this arm -----------------------------------
  probe_output="$(probe_arm "$arm_config")"
  checkpoint_dir="${probe_output%% *}"
  global_step="${probe_output##* }"

  if [ "$global_step" = "-1" ]; then
    echo "!! ARM $((index + 1))/$ARM_COUNT $arm_name -- SKIPPED: $checkpoint_dir/last.pt exists but"
    echo "   cannot be read. Refusing to discard a partly-trained arm automatically."
    echo "   Inspect it and decide by hand."
    ARM_STATE[$index]="SKIPPED"
    ARM_NOTE[$index]="unreadable checkpoint"
    write_status
    continue
  fi

  if [ "$global_step" -ge "$TOTAL_STEPS" ]; then
    completed_epochs=$((global_step / STEPS_PER_EPOCH))
    echo ">> ARM $((index + 1))/$ARM_COUNT $arm_name -- already complete"
    echo "   ($global_step/$TOTAL_STEPS steps = $completed_epochs epochs). Skipping, no GPU time spent."
    ARM_STATE[$index]="done"
    ARM_NOTE[$index]="already complete ($completed_epochs ep), skipped this launch"
    write_status
    continue
  fi

  if [ "$global_step" -gt 0 ]; then
    resume_epoch=$((global_step / STEPS_PER_EPOCH))
    launch_note="resumed from epoch $resume_epoch"
    echo ">> ARM $((index + 1))/$ARM_COUNT $arm_name -- resuming from step $global_step (epoch $resume_epoch)"
  else
    launch_note="fresh start"
    echo ">> ARM $((index + 1))/$ARM_COUNT $arm_name -- fresh start"
  fi

  stamp="$(date '+%Y%m%d-%H%M%S')"
  console_log="$arm_dir/console-$stamp.log"

  ARM_STATE[$index]="RUNNING"
  ARM_START[$index]="$(date '+%m-%d %H:%M:%S')"
  ARM_NOTE[$index]="$launch_note"
  write_status

  echo "=========================================================================="
  echo "ARM $((index + 1))/$ARM_COUNT  $arm_name"
  echo "config : $arm_config"
  echo "log    : $console_log"
  echo "mode   : $launch_note"
  echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================================================="

  # No --max-steps: the arm's length, LR horizon, and EMA window all come from
  # its config (train.epochs: 100 -> 3400 steps), and passing a cap here would
  # both truncate the schedules and make train_pipeline classify the run as a
  # bounded verification that skips export_finished_model().
  "$PYTHON" -m thesis_ml.pipeline.train_pipeline \
    --config "$arm_config" \
    > "$console_log" 2>&1
  exit_code=$?

  ARM_EXIT[$index]="$exit_code"
  ARM_END[$index]="$(date '+%m-%d %H:%M:%S')"
  if [ "$exit_code" -eq 0 ]; then
    ARM_STATE[$index]="done"
  else
    ARM_STATE[$index]="FAILED"
  fi
  write_status

  echo "ARM $((index + 1))/$ARM_COUNT $arm_name finished exit=$exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
  # Last 15 lines into the driver's own stdout so a failure is visible without
  # opening the per-arm log.
  tail -n 15 "$console_log"
  echo
done

echo "=========================================================================="
echo "SWEEP COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
cat "$STATUS_FILE"
