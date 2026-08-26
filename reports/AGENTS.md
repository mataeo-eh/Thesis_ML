# Training Reports Contract

## Purpose

- Own compact, durable, Git-trackable evidence and chair-facing summaries for
  completed training runs.

## Ownership

- `training-runs/<date>-<run-name>/` owns one immutable run report: an AI-written
  `SUMMARY.md`, deterministic `RUN_FACTS.json`, a loss curve, and a small
  allowlist of source evidence below `evidence/`.
- `scripts/prepare_training_report.py` owns report-bundle validation and
  generation; report directories do not own independent extraction logic.

## Local Contracts

- Raw `tests/output/` state remains ignored. Never copy checkpoints, model
  weights, step-level JSONL, console logs, pipeline caches, replay data, or
  machine-absolute paths here.
- A report is publishable only when finished metadata says
  `completed_all_epochs` or `early_stopping` and its final valid metric row
  agrees with `completed_epochs`.
- Keep the chair summary short, distinguish measured facts from interpretation,
  distinguish best-dev from final raw/EMA artifacts, and link claims to the
  bundled evidence.
- Architecture claims must match the finished run's stamped
  `architecture_identity`; a mismatch blocks report generation.
- Reports must use the preparer's wall-clock diagnosis. Legacy resume resets
  make the segment sum a recorded lower bound, not an exact run duration.

## Work Guidance

- Regenerate deterministic evidence with the preparer rather than hand-editing
  `RUN_FACTS.json`, copied evidence, or `TRAINING_CURVES.png`.
- A second AI provider may audit `SUMMARY.md`, but it must use the same evidence
  bundle and should correct factual errors instead of adding a competing report.

## Verification

- Run `tests/test_training_report.py` through `.venv\Scripts\python.exe`.
- Confirm `git check-ignore` does not match published report files and no file
  exceeds the preparer's 10 MiB bundle limit.

## Child DOX Index

- `training-runs/` contains generated-but-durable report bundles and has no
  child `AGENTS.md`.
