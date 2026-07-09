# Thesis_ML Contract

## Purpose

- Own the masked-discrete-diffusion research package, configuration, replay preprocessing, training, inference, and tests.

## Ownership

- `src/thesis_ml/data/windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests.
- `src/thesis_ml/data/dataset.py` owns lazy per-window example construction and per-serving fog (fog exists in fine-tuning only; pre-training serves canvas-only examples with no input region).
- `src/thesis_ml/data/collate.py` owns dynamic batch padding and exact input/canvas masks.
- `src/thesis_ml/pipeline/train_pipeline.py` owns config-only preprocessing, training, checkpoint, and resume orchestration.
- `config/default.yaml` owns canonical defaults; versioned overrides in `configs/` own reproducible local run profiles.
- `tests/overfit.bat` and `tests/smallTrainingTestV2.bat` are thin Windows launchers; training behavior remains owned by YAML and Python entry points.
- `tests/overfit.bat` launches `local_overfit_v2.yaml` and mirrors flushed training progress to both its visible terminal and `tests/output/overfitV2/console.log`.

## Local Contracts

- Read the root `AGENTS.md`, this file, `CLAUDE.md`, and the current task-specific prompt before editing.
- Run Python only through `.venv\Scripts\python.exe` after confirming the venv exists.
- PyTorch is pinned through uv to the explicit official `pytorch-cu130` index. Preserve the `tool.uv.sources` mapping and regenerate `uv.lock` with uv when changing Torch.
- Local replay data is consumed at its native one-second cadence. Timing recovery must use the same configured cadence.
- Pretraining windows contain contiguous whole timesteps from one replay and are bounded independently by input and enemy-reconstruction token budgets.
- Debut/outcome windows tile non-overlapping input timesteps under the input token budget only; each debut canvas starts at the input-window start and runs to replay end or the canvas budget, so output horizons may overlap. Outcome mode owns a separate stamped manifest.
- Each batch row contains exactly one replay window. Do not pack sequences or add document masks.
- Fog is sampled while serving an example (fine-tuning only). Persisted artifacts and manifests must remain clean.
- Input fog and canvas diffusion are independent. Fog samples one omission rate per served fine-tuning example and omits enemy content records of every token kind (entities and upgrades) from the clamped input only; pre-training has no input and no fog. Canvas corruption samples one diffusion level `t` per example and masks each loss-eligible canvas position independently with probability `t` (with a config-owned fraction of examples oversampled to exactly `t=1.0`).
- Pretraining and fine-tuning both expand every replay into exactly two perspective-specific sample streams: `p1` as self/`p2` as enemy and `p2` as self/`p1` as enemy. Replay splitting happens before this expansion so both perspectives remain in the same train/dev/test split.
- Batch padding is dynamic. Padding masks must exclude batch-shape padding from attention and loss.
- CUDA attention prefers fused Flash SDPA and falls back only to memory-efficient SDPA with a broadcast boolean key mask; math fallback is forbidden.
- The overfit profile fails when CUDA reserved memory reaches its configured ceiling and logs timing, throughput, PyTorch allocator memory, device-wide VRAM use, and the device-minus-reserved gap every step.
- The overfit profile enables config-gated block activation checkpointing because full-size fused-attention training was measured above the VRAM ceiling; other profiles retain the default-off path.
- The overfit profile uses batch size 10, validated for 20 real-data steps at 5.885 GiB peak reserved memory on the RTX 3070.
- The V1 overfit profile remains the baseline. V2 weights `[PAD]` loss at 0.2, disables early stopping, and runs the full 200-epoch cosine schedule unless manually stopped.
- The local-full pretraining run uses an exact 870-train/50-dev/remainder-test replay split, full reconstruction/future targets with perspective-relative `[WIN]`/`[LOSS]` at canvas index 0, and outcome-last sampling.
- The local-full run keeps workers persistent, trims unused CUDA cache after completed epochs, does not retain ignored step-log objects, and records current allocation, peak allocation, reservation, inactive-split allocator telemetry, device-wide memory use, and the device-minus-reserved gap.
- The overfit loader uses four persistent workers with four batches prefetched per worker; training batches drop raw metadata after worker-side feature construction, pin their custom batch tensors, and use non-blocking CUDA copies.
- Model scale, token budgets, paths, subset selection, epochs, and checkpoint intervals remain config-owned.
- Local runs write epoch CSV metrics and replay selections under their configured `tests/output/<run_name>/` log directory. Epoch metrics include mean and p50/p90/p95 input/future timestep counts, future-token loss bucketed at 1, 2-5, 6-10, 11-30, and 31+ prediction timesteps, cumulative attention-valid training tokens, cumulative distinct token IDs, average device-wide VRAM use, and average device-minus-PyTorch-reserved gap; batch-shape padding is excluded.
- Debut-mode full training also writes the same `finetune_report.json` metric schema as the overfitV2 fine-tune, using the true held-out test split.
- Epoch patience compares noisy resampled train loss against the best loss using the configured relative minimum improvement; a single flat epoch never stops a run.
- Absolute time and frame-derived values remain metadata only and must not enter model features.
- `thesis_ml.viz.diagnostics` always writes high-contrast, aligned ground-truth/prediction/error count figures. `--n-windows` is interpreted per selected replay. First-appearance timelines require `--first-appearance`; comparison CSV, input text, and final-canvas top-10 logit JSON exports consolidate multiple windows into one labelled artifact per export type and output-mask directory.
- `thesis_ml.viz.diagnostics --bypass-sampler` keeps those outputs unchanged while replacing iterative sampling with exactly one all-`[MASK]` denoising forward pass per example; default behavior remains iterative.

## Work Guidance

- Extend the existing serializer, model, loss, and training loop instead of creating parallel implementations.
- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- Split train/dev/test by replay before selecting local subsets to prevent window leakage.
- Preserve the full pretraining target grammar: leading perspective-relative `[WIN]`/`[LOSS]`, bounded in-window reconstruction, whole-timestep future continuation, then `[END] [PAD]*` for game end or direct `[PAD]*` for a boundary-truncated horizon.

## Verification

- Run `\.venv\Scripts\python.exe -m pytest -q` for package-wide changes.
- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke before long runs.
- Launcher checks may use `--max-steps N`; CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- GPU claims require an environment where CUDA is visible; never infer VRAM from CPU runs.

## Child DOX Index

- `src/thesis_ml/AGENTS.md`: the importable package contract; owns config loading and tokenization/serialization and indexes the `data`, `vocab`, `model`, `train`, `inference`, `eval`, and `pipeline` subpackages.
- `config/AGENTS.md`: canonical `default.yaml` base configuration validated by `src/thesis_ml/config.py`.
- `configs/AGENTS.md`: local proof-of-life run profiles that override `default.yaml`.
- `data/AGENTS.md`: on-disk `raw/`/`processed/` dataset layout (git-ignored contents) plus the master entity-list builder and token dictionary.
- `scripts/AGENTS.md`: standalone context-window analysis and GPU pre-flight utilities.
- `tests/AGENTS.md`: pytest suite, owner-provided extractor fixtures, and thin Windows launchers.
- `prompts/AGENTS.md`: executable task prompts and the completed-prompt archive.
- `plans/AGENTS.md`: implementation plans derived from accepted prompts and current contracts.
- `research/AGENTS.md`: source-attributed research outputs that inform, but do not override, project contracts.
- `diagnostics/AGENTS.md`: reproducible audits, investigations, and failure analyses.
- `notebooks/AGENTS.md`: exploratory notebooks whose reusable logic must graduate into the package.
- `experiments/AGENTS.md`: reproducible experiment definitions linked to versioned configs; generated runs remain ignored.
- `checkpoints/` and `.pipeline_cache/` hold generated run state; make no durable architecture claims from them.
- Root architecture and operating docs stay parent-owned: `SPEC.md` (architecture source of truth), `SCHEMA.md`, `RUN.md`, `EVAL.md`, `CLAUDE.md`, `README.md`, and `TODO.md`.
