# eval Subpackage Contract

## Purpose

- Own Arm A's evaluation wiring: run the shared build-order extraction and scoring over diffusion-arm predictions and assemble fine-tune reports, per `EVAL.md`.

## Shared-code boundary

**Build-order extraction (`buildorder.py`) and the precision/recall/F1 metrics (`metrics.py`) are NOT owned here.** They moved to `thesis_shared.eval` because they are the instrument both arms are scored with, and are contracted in `../../../../thesis-shared/src/thesis_shared/eval/AGENTS.md`. Never add an arm-local metric intended to compare arms — add it there so both arms report it.

## Ownership

- `harness.py` owns example evaluation orchestration (`evaluate_examples`, `evaluate_example`, `EvaluationExampleResult`, `EvaluationReport`).
- `finetune_report.py` owns fog-bucketed fine-tune reporting over run CSVs (`_fog_edges`, `_fog_bucket_labels`, per-example evaluation).

## Local Contracts

- Ground truth is extracted from the project's parsed parquet rows, not an external build-order tool, so evaluation aligns with training data (`EVAL.md`).
- Both sides reduce to the ordered multiset of `(entity_type, appearance_bucket)`; positions, exact frames, coordinates, and resource values are ignored because the model cannot emit them.
- One bucket equals `sampling_interval_s`. Decoded count increases emit one event per new unit; instance IDs do not exist on the prediction side.
- Matching is entity-type exact within `timing_tolerance_buckets`; each event matches at most one counterpart. Report precision, recall, F1, and accuracy.
- Keep every decoded timestep; valid canvases cannot contain a partial final timestep. Token cross-entropy is never a reported metric.
- The harness exposes raw predicted and ground-truth canvas IDs for read-only diagnostics. Final-canvas logits are populated only when explicitly requested and are not part of normal evaluation metrics. Its default path remains iterative sampling; visualization diagnostics may explicitly select the one-pass denoising path.

## Work Guidance

- Keep the prediction-side and ground-truth-side reductions on the identical event representation; change both together.

## Verification

- Evaluation changes require `tests/test_eval.py`; fine-tune reporting changes require `tests/test_finetune_report.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
