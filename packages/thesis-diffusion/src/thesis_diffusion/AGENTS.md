# thesis_diffusion Package Contract — Arm A, Uniform Discrete Diffusion (PyTorch)

## Purpose

- Own Arm A of the thesis comparison: the uniform-state discrete-diffusion model, its absorbing-process ablation, and the data/model/train/inference/eval/pipeline stack that supports them.
- This is the established arm. It is the reference implementation the autoregressive arm (`thesis_ar`) is compared against, and the source of the measured behavior recorded in `Model_Architecture/MODEL_ARCHITECTURE.md`.

## FRAMEWORK CONTRACT — PyTorch only

- Every learnable component in this package is PyTorch. **`import tensorflow` anywhere under `thesis_diffusion/` is a contract violation**, as is importing anything from `thesis_ar`.
- `packages/thesis-diffusion/pyproject.toml` must never declare `tensorflow` or `thesis-ar`.
- Torch is pinned through uv to the explicit official `pytorch-cu130` index. Preserve the `tool.uv.sources` mapping and regenerate the lock with uv when changing Torch.
- This package is installed in `.venv` and never alongside `thesis_ar`; a single process loads one CUDA runtime, so co-installing TensorFlow breaks both arms.

## Shared-code boundary

- **Tokenization, vocabulary, windowing, input features, standardization statistics, replay splits, canvas decoding, timing recovery, build-order extraction, and build-order metrics are NOT owned here.** They live in `thesis_shared` and are imported.
- A change to any of those lands on both arms. Make it in `thesis_shared`, and validate it for Arm B as well as Arm A in the same change. Never fork shared logic into this package to make Arm A's life easier.
- What this package legitimately owns is what is specific to diffusion: the bidirectional backbone, canvas corruption, the diffusion objective, iterative denoising, and this arm's training/reporting machinery.

## Ownership

- `__init__.py` owns the public package surface.
- `data/` owns the PyTorch-side input path: lazy per-window example construction, per-serving fog, dynamic collation with exact input/canvas masks, the resumable batch sampler, and the bounded frame cache.
- `model/` owns the dense Gemma 4-lineage bidirectional backbone, input-only feature embedding, expected-embedding self-conditioning, and the canvas clean-state loss.
- `train/` owns uniform/absorbing canvas corruption, process-compatible objectives, the training loop and its metrics, and the synthetic smoke trainer.
- `inference/` owns nonmonotonic uniform EB sampling and the absorbing EB ablation.
- `eval/` owns the evaluation harness and fine-tune reporting. The metric *definitions* it calls are shared.
- `pipeline/` owns config-only orchestration for training, fine-tuning, checkpointing, resume, and finished export.
- `viz/` owns read-only checkpoint diagnostics, static figures, and opt-in raw canvas/logit exports.

## Local Contracts

- Any package change that affects the function computed by the model or the exact data/configuration presented to learnable machinery must update all affected content in `Model_Architecture/MODEL_ARCHITECTURE.md`, update the canonical `.mmd`, and regenerate its SVG/PNG in the same change using `Model_Architecture/UPDATE_PROMPT.md`.
- Every tunable is a config field validated in `thesis_shared.config`; changing a parameter must be a YAML edit only, never a code change.
- Production model construction must load the configured train-split feature-statistics artifact and preserve its identity through checkpoints and exports. Synthetic/direct unit tests may opt into the explicit identity statistics fixture.
- Never place absolute game time, frame number, `game_loop`, or timestamp-derived values into model inputs, embeddings, attention inputs, or targets. Keep time as non-model metadata only.
- Keep the target grammar intact end to end: leading perspective-relative `[WIN]`/`[LOSS]`, bounded in-window reconstruction, whole-timestep future continuation, then `[END] [PAD]*` for game end or direct `[PAD]*` for a boundary-truncated horizon.
- Each batch row contains exactly one replay window. Do not pack sequences or add document masks.
- Fog is sampled while serving every example; persisted artifacts and manifests must remain clean.
- Batch padding is dynamic. Padding masks must exclude batch-shape padding from attention and loss.
- Uniform diffusion, dense GeGLU/sandwich-RMSNorm architecture, and process-stamped checkpoints form one compatibility boundary. Loaders must reject retired checkpoints before partial loading.
- `architecture_identity` is `ARCHITECTURE_ID + toggle_fingerprint(model_config)`. Because `frozen_input_kv` and `per_segment_positions` add ZERO parameters, that string is the only barrier against a silent cross-arm checkpoint load. Never bump `ARCHITECTURE_ID` merely for a toggle, and never let a toggle reach `manifest_config_stamp` or `vocabulary_stamp`.
- The three `model:` booleans `frozen_input_kv`, `segment_embeddings`, and `per_segment_positions` are config-owned. `frozen_input_kv` defaults to `true`; the other two default to `false` and exist to run experiments. Do not change a default on your own initiative — that is the owner's call on measured evidence.

## Work Guidance

- Extend the existing model, loss, and training loop instead of creating parallel implementations.
- Add every new parameter to the shared config dataclasses and `config/default.yaml`; wire run profiles through `configs/` overrides.
- Treat `Model_Architecture/MODEL_ARCHITECTURE.md` as the exact implemented-state reference. Resolve conflicts among source, merged config, and the architecture reference in the same task.
- When a change would also be correct and necessary for Arm B, put it in `thesis_shared`, not here.

## Verification

- Run `.venv\Scripts\python.exe -m pytest tests/ -q` for package-wide changes.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke before long runs.
- Launcher checks may use `--max-steps N`; CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- GPU claims require an environment where CUDA is visible; never infer VRAM from CPU runs.

## Child DOX Index

- `data/AGENTS.md`: lazy example construction, per-serving fog, dynamic collation, resumable sampling, bounded frame cache.
- `model/AGENTS.md`: dense Gemma 4-lineage bidirectional backbone, input-only features, expected-embedding self-conditioning, canvas clean-state loss.
- `train/AGENTS.md`: uniform/absorbing canvas corruption, process-compatible objectives, the training loop and metrics, the synthetic smoke trainer.
- `inference/AGENTS.md`: nonmonotonic uniform EB sampling and the absorbing EB ablation.
- `eval/AGENTS.md`: evaluation harness and fine-tune reporting.
- `pipeline/AGENTS.md`: config-only orchestration for training, fine-tuning, and export.
- `viz/AGENTS.md`: read-only checkpoint diagnostics, static figures, opt-in raw canvas/logit exports.

Shared concerns are contracted in `../../../thesis-shared/src/thesis_shared/AGENTS.md`.
