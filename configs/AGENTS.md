# configs Contract

## Purpose

- Own the local proof-of-life run profiles that override `config/default.yaml` for reproducible memorization and pipeline-validation runs.

## Ownership

- `local_overfit.yaml` owns the shared overfit/learnability base: an explicitly named 10-train + 3-dev replay subset (`pipeline.train_replay_ids` / `dev_replay_ids`, chosen at the corpus median input-token count), 150 epochs, early stopping off, warmup 40, `max_steps: 0` so both schedule horizons stay derived, rebuilt feature statistics, and epoch-end-only dev evaluation (`train.interval_dev_evaluation: false`).
- `local_overfit_v2.yaml` owns the historical learnability probe and ablation control surface: isolated paths, 100 epochs (34 optimizer steps/epoch = 3400 steps), `lr_schedule: linear`, `lr_floor_ratio: 0.03`, uniform fog/time sampling, no terminal oversampling, and unit class weights. Everything about its subset and model scale comes from `local_overfit.yaml` unchanged.
- `ablation_01..05_*.yaml` each extend `local_overfit_v2.yaml` and differ from it ONLY by the toggle(s) they enable and by their redirected storage paths. They must not restate any schedule knob: `tests/run_ablation_sweep.sh` passes no `--max-steps`, so each arm runs the inherited 100 epochs and completes both the linear LR decay and the derived EMA window. `tests/test_config.py` asserts this inheritance arm by arm.
- `local_overfit_v2_finetune.yaml` owns the debut/outcome fine-tuning variant of the V2 profile.
- `local_full.yaml` preserves the historical eight-epoch full-corpus V2 profile. `smallTrainingTestV3.yaml` owns the current 50-epoch-cap full-corpus run, expanded 384/12/6/1536 model, high-noise power sampling, five-batch accumulation, WSD schedule, weighted PAD/END loss, dev early stopping, and three checkpoint families.

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
- Production profiles select uniform-state diffusion explicitly or inherit it and add no position-dependent sampler override. The V3/default process uses power-distributed `t` plus 5% exact terminal samples; historical profiles explicitly pin uniform `t` with no terminal mass. A dedicated absorbing profile may be added for the scientific ablation with isolated namespaces.
- `model.segment_embeddings` and `model.per_segment_positions` are **ABLATION EXPERIMENTS, NOT DEFAULTS, and every committed profile keeps both `false`.** `local_overfit_v2.yaml` carries them explicitly as the control surface: the owner flips ONE to `true` and runs `tests\overfit.bat` to measure that arm. An agent must not enable a toggle on its own initiative, must not add toggle-enabled profiles, and must not treat an enabled toggle in a scratch config as evidence it should become a default — that promotion is the owner's call on measured evidence (`SPEC.md` §14b).
- `model.frozen_input_kv` is **no longer a toggle profiles pin — it is a DEFAULT** (`true` in `config/default.yaml`, owner decision 2026-08-09 on arm-01 evidence). `local_overfit_v2.yaml` dropped its pin so it and everything extending it inherit the new value; `local_full.yaml`, `local_overfit.yaml`, and the fine-tune profile inherit it too. A new profile should say NOTHING about it. Only two committed files pin it: `ablation_00_baseline.yaml` and `ablation_02..04_*.yaml` pin it `false` to preserve the finished sweep's conditions, and `ablation_01/05_*.yaml` pin it `true` for the same reason.
- `configs/memorization_01..03_*.yaml` are the memorization-probe family (regularization off / t=1 oversampled / both), launched by `tests/run_memorization_sweep.sh`. They extend `local_overfit_v2.yaml`, inherit its schedule and subset, and change only `train.weight_decay`, `fog.rate_distribution`, and `diffusion.schedule.t_one_fraction`. Their comparison run is ablation arm 01, which shares their architecture. Arm 3 must stay the exact union of arms 1 and 2 — `tests/test_config.py` asserts that, because `extends` is single-parent and arm 3 has to restate the other two arms' values.
- Enabling any toggle changes `architecture_identity`, so a non-baseline arm MUST redirect `storage.checkpoint_uri` or it will collide with the baseline `checkpoints/local-overfitV2/last.pt`. Two of the three toggles add zero parameters, so this identity string is the only thing preventing a silent cross-arm load. Toggle state does not affect `manifest_config_stamp`, so no arm triggers a manifest rebuild.

## Work Guidance

- Add a new profile file rather than mutating an existing one when a run needs different settings; give it isolated output/checkpoint namespaces to avoid clobbering prior runs.

## Verification

- Profile selection and launch are exercised by the launcher and pipeline tests (`tests/test_windows_launchers.py`, `tests/test_pipeline.py`).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
