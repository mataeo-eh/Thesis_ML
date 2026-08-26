# prompts Contract

## Purpose

- Own executable agent workflows, numbered implementation prompts, and the archive of prompts completed against the repository contracts.

## Ownership

- `NNN-name.md` files describe bounded implementation tasks.
- `completed/` owns prompts whose requested work and verification have finished.
- `training-run-summary/WORKFLOW.md` is the provider-neutral finished-run reporting procedure; `REPORT_TEMPLATE.md` defines its short chair-facing deliverable.

## Local Contracts

- Prompts are task inputs, not architecture authority; `SPEC.md` and the applicable `AGENTS.md` chain control conflicts.
- Keep acceptance criteria, affected boundaries, and required verification explicit.
- Move a prompt to `completed/` only after implementation, DOX closeout, and relevant verification are complete.
- Provider adapters must point to the canonical workflow instead of copying its analytical instructions.

## Work Guidance

- Read the current on-disk prompt and contracts before acting; do not rely on a prior prompt revision from memory.

## Verification

- Confirm completed prompts have no remaining unchecked required work and that referenced paths still exist.

## Child DOX Index

- `completed/` and `training-run-summary/` are governed by this contract and have no child `AGENTS.md`.
