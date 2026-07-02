# Thesis_ML Contract

## Purpose

- Own the masked-discrete-diffusion research package, configuration, replay preprocessing, training, inference, and tests.

## Ownership

- `src/thesis_ml/data/windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests.
- `src/thesis_ml/data/dataset.py` owns lazy per-window example construction and per-serving fog.
- `src/thesis_ml/data/collate.py` owns dynamic batch padding and exact input/canvas masks.
- `src/thesis_ml/pipeline/train_pipeline.py` owns config-only preprocessing, training, checkpoint, and resume orchestration.
- `configs/local_overfit.yaml` and `configs/local_full.yaml` own the local proof-of-life run profiles.
- `tests/overfit.bat` and `tests/smallTrainingTest.bat` are thin Windows launchers; training behavior remains owned by YAML and Python entry points.
- `tests/overfit.bat` mirrors flushed training progress to both its visible terminal and `tests/output/overfit/console.log`.

## Local Contracts

- Read the root `AGENTS.md`, this file, `CLAUDE.md`, and the current task-specific prompt before editing.
- Run Python only through `.venv\Scripts\python.exe` after confirming the venv exists.
- PyTorch is pinned through uv to the explicit official `pytorch-cu130` index. Preserve the `tool.uv.sources` mapping and regenerate `uv.lock` with uv when changing Torch.
- Local replay data is consumed at its native one-second cadence. Timing recovery must use the same configured cadence.
- Windows contain contiguous whole timesteps from one replay and are bounded independently by input and enemy-reconstruction token budgets.
- Each batch row contains exactly one replay window. Do not pack sequences or add document masks.
- Fog is sampled while serving an example. Persisted artifacts and manifests must remain clean.
- Batch padding is dynamic. Padding masks must exclude batch-shape padding from attention and loss.
- CUDA attention is restricted to fused Flash or memory-efficient SDPA with a broadcast boolean key mask; math fallback is forbidden.
- The overfit profile fails when CUDA reserved memory reaches its configured ceiling and logs timing, throughput, allocated peak, and reserved memory every step.
- The overfit profile enables config-gated block activation checkpointing because full-size fused-attention training was measured above the VRAM ceiling; other profiles retain the default-off path.
- The overfit profile uses batch size 10, validated for 20 real-data steps at 5.885 GiB peak reserved memory on the RTX 3070.
- The overfit profile weights `[PAD]` loss at 0.2 and uses 15-epoch early-stopping patience so rare and difficult target classes can continue improving after padding becomes easy.
- The overfit loader uses four persistent workers with four batches prefetched per worker; training batches drop raw metadata after worker-side feature construction, pin their custom batch tensors, and use non-blocking CUDA copies.
- Model scale, token budgets, paths, subset selection, epochs, and checkpoint intervals remain config-owned.
- Local runs write epoch CSV metrics and replay selections under their configured `tests/output/<run_name>/` log directory. Epoch metrics include per-example mean input and enemy-future timestep counts plus cumulative attention-valid training tokens and cumulative distinct token IDs; batch-shape padding is excluded.
- Epoch patience compares noisy resampled train loss against the best loss using the configured relative minimum improvement; a single flat epoch never stops a run.
- Absolute time and frame-derived values remain metadata only and must not enter model features.

## Work Guidance

- Extend the existing serializer, model, loss, and training loop instead of creating parallel implementations.
- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- Split train/dev/test by replay before selecting local subsets to prevent window leakage.
- Preserve the full target grammar: bounded in-window reconstruction first, whole-timestep future continuation second, then `[END] [PAD]*` for game end or direct `[PAD]*` for a boundary-truncated horizon.

## Verification

- Run `\.venv\Scripts\python.exe -m pytest -q` for package-wide changes.
- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke before long runs.
- Launcher checks may use `--max-steps N`; CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- GPU claims require an environment where CUDA is visible; never infer VRAM from CPU runs.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
