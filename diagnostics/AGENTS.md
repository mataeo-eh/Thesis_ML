# diagnostics Contract

## Purpose

- Own reproducible audits, investigations, failure analyses, and evidence captured while diagnosing the system.

## Ownership

- Diagnostic documents own observed symptoms, reproduction conditions, evidence, root-cause analysis, and scoped recommendations.

## Local Contracts

- Distinguish observed behavior from hypotheses and record the exact config, checkpoint, data scope, and command needed to interpret results.
- Diagnostics do not become architecture authority; accepted fixes must update source, tests, durable docs, and configs at their owning boundaries.
- Do not commit large logs, checkpoints, caches, or generated datasets here.

## Work Guidance

- Prefer the smallest reproduction that preserves the failure and link conclusions to concrete source or test evidence.

## Verification

- Re-run the focused reproduction after a fix and record whether the original failure mode is closed.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
