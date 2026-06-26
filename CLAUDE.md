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

- Do not implement anything from `SPEC.md` section 14.
- Do not resolve or implement open questions from `SPEC.md` section 12.
- Do not duplicate architecture decisions here.
