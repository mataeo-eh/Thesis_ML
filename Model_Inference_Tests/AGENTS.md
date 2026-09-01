# Model_Inference_Tests Contract

## Purpose

- Own the read-only post-training suite whose primary question is how well a
  concrete checkpoint performs or behaves during model inference on recorded,
  held-out replay windows.

## Ownership

- `run_inference_tests.py` owns discovery, shared run provenance, selection,
  execution, and summary assembly.
- `inference_test_api.py` owns the common checkpoint/config/split/model/data
  plumbing used by every inference test.
- `Test_Scripts/` owns thin, ordered inference-performance measurements.
- `output/` owns generated, git-ignored per-run artifacts and may not become a
  durable source-of-truth or architecture boundary. Its one cross-run file,
  `model_comparison_bits_per_token.csv`, is an accumulating convenience index of
  measurements, not a source of truth: it is git-ignored, regenerable by
  re-scoring each checkpoint, and nothing may read it as an input.

## Local Contracts

- Admit a test here only when its primary deliverable directly measures or
  interprets how well a model checkpoint performs during inference. Loading a
  checkpoint somewhere in an investigation is not sufficient by itself.
- Iterative generation, one-pass denoising/recovery at controlled noise, output
  grammar, calibrated outcome behavior, and held-out prediction quality belong
  here because they directly characterize inference behavior.
- A model-free baseline belongs here only when it is a bounded comparator
  reported alongside and necessary to interpret an inference-performance
  measurement in the same suite. General data statistics do not qualify.
- Training-objective geometry, loss-function design, optimization behavior,
  gradient/interference/capacity studies, preprocessing audits, and causal
  training investigations do not belong here, even if they inspect inference
  artifacts or load a checkpoint. Put executable standalone analyses under
  `scripts/`, generated artifacts under `scripts/output/`, and durable findings
  under `diagnostics/`.
- In particular, delimiter-local alignment versus positional-CE investigations
  are training-objective diagnostics and must not be added to this directory.
- Every test is read-only with respect to checkpoints, configs, manifests,
  processed arrays, replay sources, and training-run state. It writes only
  inside its runner-provided subdirectory beneath `output/`, with one admitted
  exception: `model_comparison_leaderboard` also appends a single row to the
  shared `output/model_comparison_bits_per_token.csv`, because a cross-model
  comparison that cannot outlive one run does not answer the question it exists
  for. That exception is bounded to appending one row to that one filename, must
  never delete or rewrite an earlier row's values, and must not be extended to a
  second test without the same justification. A test still writes every artifact
  a reader needs into its own subdirectory, so a run directory stays
  self-contained.
- A cross-model comparison metric must carry the full evaluation condition that
  produced it and a single handle stating which rows are comparable. Ranking two
  numbers measured over different scored positions is the failure this directory
  exists to prevent, so the headline scalar is reported alongside a
  schedule-independent variant and the condition is hashed, not left implicit.
- Score the recorded held-out test split and fail closed if live split derivation
  disagrees with the run's recorded replay selection.
- Reuse production package paths rather than reimplementing model, data,
  sampler, loss, or decoding behavior inside `Test_Scripts/`.
- Generated outputs must include portable provenance and must not embed absolute
  workstation paths.

## Work Guidance

- Prefer a small, explicit inference question with an interpretable headline
  over a broad bundle of unrelated diagnostics.
- Keep per-test window budgets configurable; full iterative sampling is costly.
- Distinguish model forward-pass behavior from iterative-sampler behavior.
- A test name or location does not make an investigation an inference test; use
  the admission rule above.

## Verification

- `--list` must discover every admitted test without loading a checkpoint.
- Focused tests must verify split gating, bounded writes, deterministic selection,
  and truthful `USES_MODEL` / `REQUIRES_DEBUT_FINETUNE` declarations.
- The shared leaderboard append must be verified to preserve every earlier row
  under column-set changes and manual header reordering, and its comparability
  and redundancy keys must be verified to respond to every field they claim to
  cover. See `tests/test_model_comparison_leaderboard.py`.
- CUDA performance or VRAM claims require a visible CUDA device and the actual
  local GPU; CPU runs may verify only device-independent behavior.

## Child DOX Index

- `Test_Scripts/` and generated `output/` have no child `AGENTS.md` files.
