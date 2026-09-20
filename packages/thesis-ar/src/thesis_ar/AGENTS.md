# thesis_ar Package Contract — Arm B, Autoregressive Baseline (TensorFlow)

## Purpose

- Own Arm B of the thesis comparison: a decoder-only **autoregressive** transformer that predicts opponent strategy from partially observed StarCraft II game state, trained as the controlled counterpart to the discrete-diffusion model in `thesis_diffusion`.
- Exist so that the thesis can answer one question: **given the same corpus, the same tokenization, the same splits, the same conditioning information, and a matched non-embedding parameter budget, how does an autoregressive model compare to a discrete-diffusion model on this task?**

## STATUS

- **Scaffold. No model, training, or inference code is implemented yet.** The subpackage directories and `__init__.py` files exist to fix the layout and hold this contract. Treat every "owns" statement below as describing what the module *will* own when written.

## FRAMEWORK CONTRACT — TensorFlow only, non-negotiable

This is the defining constraint of this package. Read it before writing a single line.

- **Every learnable component, training step, optimizer, schedule, checkpoint, metric accumulator, and inference routine in this package MUST be written in TensorFlow / Keras.** There is no partial-PyTorch path, no "temporary" PyTorch prototype, and no PyTorch-to-TensorFlow conversion layer.
- **`import torch` anywhere under `thesis_ar/` is a contract violation.** So is importing anything from `thesis_diffusion`. If a task seems to require either, stop and raise it with the owner instead of doing it.
- **`thesis-ar` must never declare `torch` or `thesis-diffusion` as a dependency** in `packages/thesis-ar/pyproject.toml`.
- Prefer idiomatic TensorFlow rather than PyTorch written in TensorFlow syntax. Use `tf.keras.Model` / `tf.keras.layers.Layer` subclasses, `tf.data.Dataset` input pipelines, `tf.function`-compiled steps, Keras optimizers and LR schedules, and the TensorFlow checkpoint formats. Do not hand-port `thesis_diffusion` module-for-module; port *behavior and hyperparameters*, not code structure.
- Porting the diffusion arm's PyTorch logic is expected to be substantial work and is accepted deliberately. The owner is building this arm in TensorFlow on purpose: a capstone course requires TensorFlow as its framework, so the AR pipeline doubles as that deliverable. Framework choice here is a fixed project requirement, not an implementation detail open to optimization.

## Ownership

- `model/` will own the decoder-only autoregressive transformer: causal self-attention, token and positional representations, input-conditioning features, and the next-token output head.
- `train/` will own the next-token objective, the Keras training loop, metric accumulation, checkpointing, and resume.
- `data/` will own the TensorFlow-side input pipeline only: converting the shared, framework-agnostic window artifacts into batched `tf.data.Dataset` tensors, dynamic padding, and mask construction. It must NOT re-derive windows, features, or splits.
- `inference/` will own autoregressive decoding (sampling / greedy / beam as configured) and the AR arm's own timing recovery hookup.
- `eval/` will own AR-arm evaluation wiring. Scoring definitions themselves are shared and are NOT redefined here.
- `pipeline/` will own config-only orchestration for AR training and export, mirroring the role of `thesis_diffusion.pipeline`.

## Local Contracts

### Shared-code boundary

- **Everything both arms must hold constant comes from `thesis_shared` and is imported, never reimplemented.** That includes configuration loading, tokenization/serialization, the content vocabulary and special tokens, replay windowing, input feature construction, train-split standardization statistics, the train/dev/test replay split, canvas decoding, timing recovery, build-order extraction, and build-order metrics.
- If the AR arm appears to need a change to shared behavior, change it in `thesis_shared` so **both** arms receive it, and re-validate the diffusion arm in the same change. Never fork shared logic into this package. A shared-layer fork silently destroys the comparison, because the two arms stop consuming the same data.
- The only things that legitimately live here are things that genuinely differ between an autoregressive model and a diffusion model.

### Comparison fairness

- **Non-embedding parameter count is the matched quantity.** Vocabulary/embedding parameters are excluded from the match, because the two arms share a vocabulary and embedding tables would otherwise dominate the comparison. Record the measured non-embedding count for both arms whenever a size claim is made.
- Both arms train on the same replay corpus, the same `pipeline.seed`-derived splits, and the same perspective expansion (`p1`-as-self and `p2`-as-self), with replay-level splitting before perspective expansion so no window leaks across splits.
- The AR arm gets the same pre-training and post-training stages as the diffusion arm. Where a stage cannot be identical because the objectives differ, document the difference explicitly in the run's report rather than quietly dropping the stage.
- Report AR perplexity as standard next-token perplexity. **Do not compare it numerically to the diffusion arm's `perplexity` column**, which is computed over a sampled corruption distribution and is not a next-token quantity. The arms are compared on shared task metrics (build-order precision/recall/F1 and the other `thesis_shared.eval` metrics), not on each other's loss.

### Configuration

- Every tunable is a config field read from YAML. Nothing is hardcoded — same rule as the rest of the repository.
- AR-arm parameters live under their own top-level config key and must not collide with or silently inherit diffusion-only keys.

### Execution environment

- **No meaningful model training is ever run locally.** TensorFlow has had no native-Windows GPU support since 2.11, so on the owner's Windows workstation this package runs CPU-only.
- **Local execution is limited to lightweight verification**: unit tests, shape and graph checks, tiny smoke runs over a handful of synthetic or truncated examples, and config validation. This is expected and sufficient.
- **All real training runs on Linux cloud compute.** Do not tune, benchmark, or draw any performance, throughput, or VRAM conclusion from a local run, and never report a local timing as evidence about model speed.
- **This package is installed in its own virtual environment, never alongside `thesis_diffusion`.** A single Python process loads one CUDA runtime; co-installing the cu130 Torch wheels with TensorFlow's bundled CUDA libraries causes import errors or silent ABI mismatches. The local AR environment is created separately:

  ```
  uv venv .venv-tf
  uv pip install -e packages/thesis-shared -e packages/thesis-ar
  ```

- The Linux cloud training environment installs `packages/thesis-ar` with the `cuda` extra (`tensorflow[and-cuda]`). Do not install that extra on native Windows.
- Run Python for this arm only through the AR environment's interpreter, never `.venv`, which is the diffusion arm's environment and contains PyTorch.

## Work Guidance

- Build the arm in the order the diffusion arm was built: input pipeline, then model, then objective and loop, then inference, then evaluation wiring. Do not start with the model.
- Read `thesis_diffusion`'s corresponding contract and source for **what** a stage must accomplish and which hyperparameters it reads, then implement that in TensorFlow. Copying its structure or its PyTorch idioms is not the goal; matching its data, its configuration surface, and its reported metrics is.
- `Model_Architecture/MODEL_ARCHITECTURE.md` currently documents the diffusion arm only. When AR model code lands, the architecture reference must gain a clearly separated AR section rather than blending the two arms into one description.
- Prefer the smallest change that keeps the arms comparable. When a choice is genuinely free (a TensorFlow API detail with no behavioral consequence), pick the idiomatic TensorFlow option.

## Verification

- Run this arm's tests with the AR environment's interpreter, not `.venv`.
- AR tests are marked `tensorflow` in the root pytest configuration so they can be selected or deselected explicitly; the diffusion environment deselects them with `-m "not tensorflow"`.
- Every new module needs a test that runs on CPU within a few seconds. A test that requires a GPU or a real training run does not belong in the suite.
- Structural boundary checks (the shared package stays framework-free; neither arm depends on the other) live in `tests/test_pipeline.py::test_workspace_packages_declare_the_framework_boundary` and must keep passing.
- Before any size claim, verify the measured non-embedding parameter count against the diffusion arm rather than asserting a nominal configured size.

## Child DOX Index

- No child contracts yet. Subpackage directories (`data/`, `model/`, `train/`, `inference/`, `eval/`, `pipeline/`) are scaffolds governed by this file. Add a child `AGENTS.md` when a subpackage acquires real behavior worth contracting separately.
