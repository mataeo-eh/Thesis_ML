#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ablation sweep driver -- runs the five representational-toggle arms back to
# back against the overfitV2 profile and the same 10-train / 3-dev small corpus.
#
# WHAT THIS IS: a thin sequential launcher, in the same spirit as
# tests/overfit.bat. It owns NO training knobs. Every knob -- replay subset,
# model scale, LR schedule, batch size -- lives in the per-arm YAML profiles in
# configs/ablation_*.yaml, each of which extends configs/local_overfit_v2.yaml
# and differs from it ONLY by the model toggle(s) it enables and by its
# redirected storage paths.
#
# RUN LENGTH: each arm runs 100 epochs = 3400 optimizer steps (34 per epoch),
# and that length is owned entirely by `train.epochs: 100` in
# configs/local_overfit_v2.yaml, which every arm extends. This driver passes NO
# `--max-steps`, so each arm trains through all 100 epochs and completes both of
# its config-derived schedules: the linear LR decay reaches its 9.0e-6 floor at
# the last step, and the EMA averaging window (10% of the run = 340 steps) turns
# over ~10 times before the run ends. STEPS_PER_EPOCH x EXPECTED_EPOCHS below is
# used ONLY to decide skip / resume, never as a cap. See the RUN LENGTH block in
# configs/ablation_01_frozen_input_kv.yaml for the full reasoning.
#
# An earlier version of this driver capped each arm with `--max-steps 3400`
# against a 150-epoch config, deliberately truncating the LR schedule to mirror a
# baseline that had itself been cut short. That is no longer wanted: the arms are
# being rerun from scratch, so they run their schedules to the end instead.
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
#                              own _try_resume() loads last.pt and restores the
#                              model, EMA, optimizer, scheduler, global_step,
#                              completed_epochs and the in-epoch batch offset,
#                              so training continues where it stopped instead of
#                              restarting. At most the steps since the arm's last
#                              per-epoch checkpoint are redone.
#   unreadable last.pt      -> SKIP with a loud warning rather than silently
#                              discarding a partly-trained arm. A truncated
#                              checkpoint means someone was killed mid-write;
#                              that is a human decision, not the driver's.
#
# Because each arm's storage paths are private to that arm (see the YAML), a
# resumed arm can never pick up another arm's weights -- and enabling a toggle
# changes the stored architecture_identity, so a cross-arm mixup would abort on
# a fingerprint mismatch rather than train silently on the wrong thing.
# ---------------------------------------------------------------------------
#
# RESILIENCE: an arm that crashes does not abort the sweep. Its exit code is
# recorded and the driver moves to the next arm, because the point of running
# overnight is to come back to four usable results rather than to one crash.
#
# OUTPUTS, per arm, under tests/output/ablations/<arm-name>/:
#   console-<timestamp>.log  full stdout+stderr of the run (one per launch, so a
#                            resumed arm keeps its earlier logs)
#   epoch_metrics.csv        1 row per epoch, train + dev, every loss class
#   step_metrics.jsonl       1 line per optimizer step
# plus a sweep-level tests/output/ablations/SWEEP_STATUS.md updated as each arm
# starts and finishes, so progress is readable while the sweep is still running.
# ---------------------------------------------------------------------------
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON=".venv/Scripts/python.exe"
SWEEP_DIR="tests/output/ablations"
STATUS_FILE="$SWEEP_DIR/SWEEP_STATUS.md"

# The step count a COMPLETED arm reaches: 34 train batches/epoch x 100 epochs.
# This is NOT passed to the trainer -- the arms get their length from
# `train.epochs` in configs/local_overfit_v2.yaml. It exists only so probe_arm's
# global_step can be turned into a skip / resume / fresh decision, so it must be
# kept in step with that config. Both values are asserted against the loaded
# config in tests/test_config.py.
STEPS_PER_EPOCH=34
EXPECTED_EPOCHS=100
TOTAL_STEPS=$((STEPS_PER_EPOCH * EXPECTED_EPOCHS))

# arm-name : config-file. Order is the order they run in.
ARMS=(
  "01-frozen-input-kv-only:configs/ablation_01_frozen_input_kv.yaml"
  "02-segment-embeddings-only:configs/ablation_02_segment_embeddings.yaml"
  "03-per-segment-positions-only:configs/ablation_03_per_segment_positions.yaml"
  "04-segment-embeddings-plus-per-segment-positions:configs/ablation_04_segment_embeddings_plus_per_segment_positions.yaml"
  "05-frozen-input-kv-plus-segment-embeddings:configs/ablation_05_frozen_input_kv_plus_segment_embeddings.yaml"
)

mkdir -p "$SWEEP_DIR"

# ---------------------------------------------------------------------------
# probe_arm <config-path>
#
# Prints one line: "<checkpoint_dir> <global_step>".
#
# The checkpoint directory is read from the arm's own YAML via load_config
# rather than reconstructed from the arm name, so the driver can never disagree
# with the profile about where an arm's weights live.
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
        # package (train/loop.py:1219, :1279, inference/sampler.py:605,
        # viz/diagnostics.py:165). The file being read was written by the
        # training loop in this same repository, to a repo-local path, minutes
        # earlier -- never third-party input -- so the unpickling surface is
        # code we already own and run.
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
# (5 rows) and it means the file is never left half-written.
declare -a ARM_STATE ARM_START ARM_END ARM_EXIT ARM_NOTE
for arm_index in "${!ARMS[@]}"; do
  ARM_STATE[$arm_index]="pending"
  ARM_START[$arm_index]="-"
  ARM_END[$arm_index]="-"
  ARM_EXIT[$arm_index]="-"
  ARM_NOTE[$arm_index]="-"
done

# NOTE FOR MAINTAINERS: every loop variable in this function is named
# `status_index`, NOT `index`. An earlier version looped on `index` here, which
# silently clobbered the caller's loop counter (Bash has no function-local scope
# unless declared) -- so every arm's completion was recorded into the LAST arm's
# row and arms 1-3 sat at "RUNNING" forever. The training itself was unaffected,
# but the status table was pure fiction. Keep the names distinct.
write_status() {
  local status_index status_name
  {
    echo "# Ablation sweep status"
    echo
    echo "Sweep driver last started: $SWEEP_STARTED"
    echo "Last updated:              $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "Each arm: configs/local_overfit_v2.yaml + the named toggle(s), $EXPECTED_EPOCHS epochs"
    echo "($TOTAL_STEPS optimizer steps, run to completion -- no \`--max-steps\` cap, so the"
    echo "linear LR decay and the EMA window both finish), same 10-train / 3-dev small corpus."
    echo
    echo "This driver is restartable: re-running \`tests/run_ablation_sweep.sh\` skips"
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

echo "=========================================================================="
echo "ABLATION SWEEP  started $SWEEP_STARTED"
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
    echo "!! ARM $((index + 1))/5 $arm_name -- SKIPPED: $checkpoint_dir/last.pt exists but"
    echo "   cannot be read. Refusing to discard a partly-trained arm automatically."
    echo "   Inspect it (a .pausebackup copy may sit beside it) and decide by hand."
    ARM_STATE[$index]="SKIPPED"
    ARM_NOTE[$index]="unreadable checkpoint"
    write_status
    continue
  fi

  if [ "$global_step" -ge "$TOTAL_STEPS" ]; then
    completed_epochs=$((global_step / STEPS_PER_EPOCH))
    echo ">> ARM $((index + 1))/5 $arm_name -- already complete"
    echo "   ($global_step/$TOTAL_STEPS steps = $completed_epochs epochs). Skipping, no GPU time spent."
    ARM_STATE[$index]="done"
    ARM_NOTE[$index]="already complete ($completed_epochs ep), skipped this launch"
    write_status
    continue
  fi

  if [ "$global_step" -gt 0 ]; then
    resume_epoch=$((global_step / STEPS_PER_EPOCH))
    launch_note="resumed from epoch $resume_epoch"
    echo ">> ARM $((index + 1))/5 $arm_name -- resuming from step $global_step (epoch $resume_epoch)"
  else
    launch_note="fresh start"
    echo ">> ARM $((index + 1))/5 $arm_name -- fresh start"
  fi

  stamp="$(date '+%Y%m%d-%H%M%S')"
  console_log="$arm_dir/console-$stamp.log"

  ARM_STATE[$index]="RUNNING"
  ARM_START[$index]="$(date '+%m-%d %H:%M:%S')"
  ARM_NOTE[$index]="$launch_note"
  write_status

  echo "=========================================================================="
  echo "ARM $((index + 1))/5  $arm_name"
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

  echo "ARM $((index + 1))/5 $arm_name finished exit=$exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
  # Last 15 lines into the driver's own stdout so a failure is visible without
  # opening the per-arm log.
  tail -n 15 "$console_log"
  echo
done

echo "=========================================================================="
echo "SWEEP COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
cat "$STATUS_FILE"
