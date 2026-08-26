# Training-run summary workflow

Use this provider-neutral workflow to turn one finished local training run into
a concise, auditable report suitable for the thesis chair.

## Input and authority

- The user must name the run directory, such as
  `tests/output/smallTrainingTestV3`.
- Read the root `AGENTS.md`, `CLAUDE.md`, `reports/AGENTS.md`,
  `Model_Architecture/AGENTS.md`, and
  `Model_Architecture/MODEL_ARCHITECTURE.md` completely before interpreting the
  run.
- Runtime artifacts are evidence, not current architecture authority. Verify
  the finished run's `architecture_identity` against the current reference and
  stop if the report preparer rejects the mismatch.
- Never read tensor payloads from `.pt` or `.safetensors` files for this task.

## Prepare deterministic evidence

Confirm `.venv/Scripts/python.exe` exists, then run:

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_training_report.py <run-directory>
```

Use the emitted `report_dir`. Read `RUN_FACTS.json`, every textual file under
its `evidence/` directory, and `REPORT_TEMPLATE.md`. Inspect
`TRAINING_CURVES.png` when image viewing is available. Do not substitute mental
arithmetic for the preparer's first/best/final calculations.

## Write the chair summary

Create or replace `<report_dir>/SUMMARY.md` using the template. Keep it between
250 and 450 words, excluding the compact metric table.

- Lead with the outcome and why the run stopped.
- Explain the tested model in plain language, using architecture details only
  when they help interpret the result.
- Separate measured facts from interpretation.
- Distinguish the strict best-dev checkpoint from the final raw/EMA export.
- State whether final dev loss improved or regressed relative to the best epoch.
- Use `epoch_metrics.wall_clock` for duration. When it reports legacy resume
  resets, state the segment-summed time as a recorded lower bound and say that
  exact end-to-end duration is unavailable; never present the final row's clock
  value as the whole run.
- Mention material data-quality warnings from `RUN_FACTS.json`, including
  excluded malformed metric rows.
- Give one restrained next step grounded in the evidence.
- Avoid hype, unsupported causal claims, and claims that the run proves
  real-game strategy quality. Token cross-entropy is not downstream gameplay
  evaluation.
- Link claims to the repository-relative evidence and architecture reference so
  the report remains auditable on GitHub.
- Do not include workstation-absolute paths, credentials, replay data, or model
  weights.

Before finishing, re-read `SUMMARY.md` and verify every number against
`RUN_FACTS.json`. Do not commit or push unless the user separately authorized
those Git operations.

## Audit mode

When the user asks a second provider to audit an existing summary, do not write
a competing report. Check every numerical and architectural claim against the
same evidence, correct `SUMMARY.md` only when authorized, and report any
remaining uncertainty.
