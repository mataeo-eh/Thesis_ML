# notebooks Contract

## Purpose

- Own exploratory notebooks for analysis, prototyping, and experiment write-ups.

## Ownership

- Notebook files own interactive exploration; reusable production behavior remains owned by `src/thesis_ml/`.

## Local Contracts

- Keep notebooks lightweight, reproducible, and free of secrets or machine-specific absolute paths.
- Do not duplicate stable serialization, dataset, model, training, sampling, or evaluation logic in notebooks.
- Large generated outputs and local datasets are not durable contract material.

## Work Guidance

- Promote stable logic into the appropriate package module with tests, then import it into the notebook.

## Verification

- Document required config, data scope, and kernel environment; use the project venv when executing notebook code.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
