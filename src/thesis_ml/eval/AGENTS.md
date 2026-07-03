# eval Subpackage Contract

## Purpose

- Own build-order evaluation: extract build orders from both ground-truth parquet and decoded predictions, score them, and report fine-tuning results, per `SPEC.md` §10 and `EVAL.md`.

## Ownership

- `buildorder.py` owns build-order extraction (`extract_build_order`, `extract_build_order_from_parquet`, `extract_build_order_from_frame`, `BuildOrderEvent`).
- `metrics.py` owns matching and scoring (`compare_build_orders`, `aggregate_metrics`, `BuildOrderMetrics`, `_count_matches`).
- `harness.py` owns example evaluation orchestration (`evaluate_examples`, `evaluate_example`, `EvaluationExampleResult`, `EvaluationReport`).
- `finetune_report.py` owns fog-bucketed fine-tune reporting over run CSVs (`_fog_edges`, `_fog_bucket_labels`, per-example evaluation).

## Local Contracts

- Ground truth is extracted from the project's parsed parquet rows, not an external build-order tool, so evaluation aligns with training data (`EVAL.md`).
- Both sides reduce to the ordered multiset of `(entity_type, appearance_bucket)`; positions, exact frames, coordinates, and resource values are ignored because the model cannot emit them.
- One bucket equals `sampling_interval_s`. Decoded count increases emit one event per new unit; instance IDs do not exist on the prediction side.
- Matching is entity-type exact within `timing_tolerance_buckets`; each event matches at most one counterpart. Report precision, recall, F1, and accuracy.
- Keep every decoded timestep; valid canvases cannot contain a partial final timestep. Token cross-entropy is never a reported metric.

## Work Guidance

- Keep the prediction-side and ground-truth-side reductions on the identical event representation; change both together.

## Verification

- Evaluation changes require `tests/test_eval.py`; fine-tune reporting changes require `tests/test_finetune_report.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
