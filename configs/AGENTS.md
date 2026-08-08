# configs Contract

## Purpose

- Own the local proof-of-life run profiles that override `config/default.yaml` for reproducible memorization and pipeline-validation runs.

## Ownership

- `local_overfit.yaml` owns the shared overfit/learnability base: an explicitly named 10-train + 3-dev replay subset (`pipeline.train_replay_ids` / `dev_replay_ids`, chosen at the corpus median input-token count), 30 epochs, early stopping off, warmup 40, rebuilt feature statistics, and epoch-end-only dev evaluation (`train.interval_dev_evaluation: false`).
- `local_overfit_v2.yaml` owns the current learnability probe: isolated checkpoint/log/cache paths and the uniform process, inheriting every data/schedule setting from `local_overfit.yaml`. It carries no per-class loss weighting (that knob is fine-tuning-only). It is also the designated **ablation control surface** for the prompt-009 `model:` toggles (see Local Contracts).
- `local_overfit_v2_finetune.yaml` owns the debut/outcome fine-tuning variant of the V2 profile.
- `local_full.yaml` owns the eight-epoch, exact-split full-corpus uniform-diffusion pretraining profile with a leading outcome target and no positional sampling constraint.

## Local Contracts

- Profiles are declarative overrides on `config/default.yaml`; keep them minimal deltas, not full copies.
- Any profile change represented in `../Model_Architecture/MODEL_ARCHITECTURE.md`—especially `local_full.yaml` or the documented fine-tune profile—must update every affected architecture table, tensor bound, parameter/memory derivation, and training/sampling description, then update the canonical `.mmd` and regenerate its SVG/PNG in the same change using `../Model_Architecture/UPDATE_PROMPT.md`.
- Local profiles use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction and share persisted clean replay artifacts.
- The fine-tune profile shares tokenized replay artifacts but owns a separate input-tiled window manifest; its output horizons may overlap and do not use the pretraining reconstruction-fraction bound.
- Window manifests carry a semantic/config stamp and are rebuilt when these rules change.
- Each run namespace owns a feature-statistics artifact path. Preparation remains an explicit pipeline action; fine-tuning reuses and verifies the statistics learned from its selected training split.
- The overfit profiles select replays by explicit name, not by seeded subset, so `replay_subset_size` and `validation_replay_count` must stay 0 there. Their manifest config stamp equals `local_full.yaml`'s, so `overfit_window_manifest.jsonl` may be seeded by copying the built full manifest instead of re-preprocessing the corpus.
- Profiles keep experiments traceable to exact settings — version-control every profile that produced a reported run.
- CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- `local_full.yaml` assigns exactly 870 replays to train, 50 to dev, and every remainder to test; it uses batch size 9, ten persistent workers with four-batch prefetch, gradient checkpointing, 50% training self-conditioning, and a 7.5 GiB reclaim-first reserved-memory ceiling. Unused CUDA cache is also released after each of its eight epochs.
- All production profiles select uniform diffusion explicitly or inherit the uniform default, use zero terminal oversampling and zero confidence sharpening initially, and add no position-dependent sampler override. A dedicated absorbing profile may be added for the scientific ablation, with isolated checkpoint/output namespaces.
- The prompt-009 `model:` toggles (`frozen_input_kv`, `segment_embeddings`, `per_segment_positions`) are **ABLATION EXPERIMENTS, NOT DEFAULTS, and every committed profile keeps all three `false`.** `local_overfit_v2.yaml` carries them explicitly as the control surface: the owner flips ONE to `true` and runs `tests\overfit.bat` to measure that arm. An agent must not enable a toggle on its own initiative, must not add toggle-enabled profiles, and must not treat an enabled toggle in a scratch config as evidence it should become a default — that promotion is the owner's call on measured evidence (`SPEC.md` §14b). `frozen_input_kv` additionally sits behind the `SPEC.md` §14a confirmation gate.
- Enabling any toggle changes `architecture_identity`, so a non-baseline arm MUST redirect `storage.checkpoint_uri` or it will collide with the baseline `checkpoints/local-overfitV2/last.pt`. Two of the three toggles add zero parameters, so this identity string is the only thing preventing a silent cross-arm load. Toggle state does not affect `manifest_config_stamp`, so no arm triggers a manifest rebuild.

## Work Guidance

- Add a new profile file rather than mutating an existing one when a run needs different settings; give it isolated output/checkpoint namespaces to avoid clobbering prior runs.

## Verification

- Profile selection and launch are exercised by the launcher and pipeline tests (`tests/test_windows_launchers.py`, `tests/test_pipeline.py`).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
