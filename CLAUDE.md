# CLAUDE.md

Coding conventions for this repository.

## Project

This repository runs a **two-arm controlled comparison** for StarCraft II opponent-strategy prediction. The same corpus, tokenization, windows, splits, and conditioning feed two models built on two different frameworks, matched on non-embedding parameter count.

| Package | Role | Framework | Status |
|---|---|---|---|
| `thesis_shared` | Framework-agnostic core both arms import | **none** (numpy/pandas only) | Established |
| `thesis_diffusion` | Arm A — uniform discrete diffusion | **PyTorch** | Established, measured |
| `thesis_ar` | Arm B — autoregressive baseline | **TensorFlow / Keras** | Scaffold, not yet implemented |

- Target Python: `>=3.10,<3.13`.
- uv workspace; the three members live under `packages/<dist-name>/src/<import_name>/` and are independently installable.
- Tests use pytest and live under `tests/`.
- Configuration uses one YAML file validated into dataclasses. Parameters must be read from config, not hardcoded.

## Framework Rules

- **Write TensorFlow in `thesis_ar`, PyTorch in `thesis_diffusion`, and neither in `thesis_shared`.** The split is a fixed project requirement (a capstone course mandates TensorFlow), not an open design question. Do not propose consolidating the arms onto one framework.
- `import torch` under `thesis_ar/`, `import tensorflow` under `thesis_diffusion/`, or either under `thesis_shared/` is a contract violation.
- Neither arm may import the other. Everything they share comes from `thesis_shared`.
- **Shared code goes in `thesis_shared`, never duplicated into an arm.** A fork of shared logic silently breaks the comparison, because the arms stop consuming identical data.

## Environments

The two arms are **never installed together** — one Python process loads one CUDA runtime, so cu130 Torch and TensorFlow's bundled CUDA libraries conflict.

- Arm A + shared: `.venv` (`uv sync`). Use `.venv\Scripts\python.exe` for all Arm A and shared work.
- Arm B: its own venv (`uv venv .venv-tf`; `uv pip install -e packages/thesis-shared -e packages/thesis-ar`). Never `.venv`.
- **Arm B does no meaningful training locally.** TensorFlow has had no native-Windows GPU support since 2.11, so it is CPU-only here. Light verification only; all real AR training runs on Linux cloud compute. Never report a local AR timing as performance evidence.

## Architecture Source

There is no single source-of-truth architecture document. Binding rules live in the `AGENTS.md` DOX contract tree — start at the root `AGENTS.md`, then read the contract for the package you are editing. `Model_Architecture/MODEL_ARCHITECTURE.md` records Arm A's exact implemented state.

`research/SPEC-legacy.md` is a **retired** planning artifact from the project's first design phase. It is not binding, it predates the two-arm experiment, and it contradicts it in places. Do not cite it or implement from it.

## Directory Layout

- `packages/thesis-shared/src/thesis_shared/`: framework-agnostic core.
- `packages/thesis-diffusion/src/thesis_diffusion/`: Arm A package code.
- `packages/thesis-ar/src/thesis_ar/`: Arm B package code.
- `config/`: default project configuration.
- `configs/`: versioned run profiles overriding the default.
- `diagnostics/`: audits, investigations, and failure analyses.
- `plans/`: implementation plans.
- `prompts/`: executable agent prompts. `prompts/completed/`: prompts after successful completion.
- `research/`: research outputs, plus the retired `SPEC-legacy.md`.
- `Model_Architecture/`: Arm A implemented-state reference and diagrams.
- `tests/`: pytest tests. `tests/fixtures/`: owner-provided extractor fixtures.

## Do Not

- Do not change a `model:` toggle default on your own initiative. `model.segment_embeddings` and `model.per_segment_positions` default to `false` and exist to run experiments; `model.frozen_input_kv` defaults to `true`. Promotion is the owner's call on measured evidence.
- Do not duplicate architecture decisions here; they belong in the contract tree.
- Do not rewrite dated archives (`prompts/completed/`, `diagnostics/`, generated run outputs) to match current naming. They record what was true when written, including the pre-split `thesis_ml` package name.
