# CLAUDE.md

Coding conventions for the `thesis_ml` package.

## Project

- Target Python: `>=3.10,<3.13`.
- Framework: PyTorch.
- Package layout: `src/thesis_ml/`.
- Tests use pytest and live under `tests/`.
- Configuration uses one YAML file validated into dataclasses. Parameters must be read from config, not hardcoded.

## Architecture Source

Architecture decisions live in `SPEC.md`. That document is the source of truth and wins on any conflict with this file.

## Directory Layout

- `config/`: default project configuration.
- `diagnostics/`: audits, investigations, and failure analyses.
- `plans/`: implementation plans.
- `prompts/`: executable agent prompts.
- `prompts/completed/`: prompts after successful completion.
- `research/`: research outputs.
- `src/thesis_ml/`: importable package code.
- `tests/`: pytest tests.
- `tests/fixtures/`: owner-provided extractor fixtures.

## Do Not

- Do not implement anything from `SPEC.md` §14 (the hard banned list).
- Do not implement anything from `SPEC.md` §14a (discouraged) without explicit owner confirmation first. These are gated, not banned: reason through the alternative, say why the item is warranted, ask, and get a yes BEFORE writing code. When approved, it ships as a toggle defaulting to `false` and must be measured against its baseline arm before being trusted.
- Do not enable a `SPEC.md` §14b ablation toggle on your own initiative. `model.frozen_input_kv`, `model.segment_embeddings`, and `model.per_segment_positions` exist to run an experiment; `false` is the baseline and promoting one to a default is the owner's call on measured evidence.
- Do not resolve or implement open questions from `SPEC.md` section 12.
- Do not duplicate architecture decisions here.
