# configs Contract

## Purpose

- Own the local proof-of-life run profiles that override `config/default.yaml` for reproducible memorization and pipeline-validation runs.

## Ownership

- `local_overfit.yaml` owns the V1 baseline seeded 25-replay memorization run (five-epoch relative-improvement patience, default loss weights).
- `local_overfit_v2.yaml` owns the clean V2 comparison (isolated checkpoint/log/cache paths, `[PAD]` loss weight 0.2, early stopping disabled, full 200-epoch schedule).
- `local_overfit_v2_finetune.yaml` owns the debut/outcome fine-tuning variant of the V2 profile.
- `local_full.yaml` owns the eight-epoch, exact-split full-corpus pretraining profile with a leading outcome target.

## Local Contracts

- Profiles are declarative overrides on `config/default.yaml`; keep them minimal deltas, not full copies.
- Local profiles use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction and share persisted clean replay artifacts.
- The fine-tune profile shares tokenized replay artifacts but owns a separate input-tiled window manifest; its output horizons may overlap and do not use the pretraining reconstruction-fraction bound.
- Window manifests carry a semantic/config stamp and are rebuilt when these rules change.
- The V1 overfit profile is the baseline; V2 weights `[PAD]` loss at 0.2, disables early stopping, and runs the full cosine schedule unless manually stopped.
- Profiles keep experiments traceable to exact settings — version-control every profile that produced a reported run.
- CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- `local_full.yaml` assigns exactly 870 replays to train, 50 to dev, and every remainder to test; it uses batch size 9, ten persistent workers with four-batch prefetch, gradient checkpointing, 50% training self-conditioning, and a 7.5 GiB reserved-memory ceiling. Unused CUDA cache is released after each of its eight epochs.

## Work Guidance

- Add a new profile file rather than mutating an existing one when a run needs different settings; give it isolated output/checkpoint namespaces to avoid clobbering prior runs.

## Verification

- Profile selection and launch are exercised by the launcher and pipeline tests (`tests/test_windows_launchers.py`, `tests/test_pipeline.py`).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
