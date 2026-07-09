# experiments Contract

## Purpose

- Own reproducible experiment entry points and records that connect a hypothesis to versioned configuration and evaluation.

## Ownership

- Experiment definitions own the hypothesis, config/profile reference, entry point, seed/data scope, and evaluation plan.
- `runs/` owns generated checkpoints, logs, metrics, and plots and remains git-ignored.

## Local Contracts

- Every reported run must resolve to a version-controlled config in `configs/` plus a reproducible package or script entry point.
- Do not place reusable training or evaluation logic here; extend the owning package module and call it.
- Keep generated artifacts, credentials, and machine-specific paths out of version control.

## Work Guidance

- Use isolated checkpoint, log, and cache namespaces so experiments cannot overwrite baselines or one another.

## Verification

- Record the exact config, seed, data split, checkpoint source, and metric output needed to reproduce a claimed result.

## Child DOX Index

- `runs/` is generated state governed by this contract and has no child `AGENTS.md`.
