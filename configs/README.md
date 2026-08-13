# configs/

Configuration files (YAML/JSON/TOML) for data pipelines, model hyperparameters,
and experiment settings. Keep configs declarative and version-controlled so every
experiment is traceable to the exact settings that produced it.

- `local_overfit.yaml`: V1 base profile. Owns the named 10-train / 3-dev median-sized replay subset, the model scale, warmup 40, and a 150-epoch cosine-to-0.1 schedule.
- `local_overfit_v2.yaml`: the historical learnability probe and ablation control surface. It preserves 100 epochs (3400 optimizer steps), `lr_schedule: linear`, `lr_floor_ratio: 0.03`, uniform corruption/fog, and unit class weights.
- `ablation_00_baseline.yaml`: the finished sweep's all-toggles-false baseline arm, pinned explicitly. It exists because `model.frozen_input_kv` became a default on 2026-08-09 and `local_overfit_v2.yaml` — which used to be the baseline — now resolves to `+frozen_input_kv`. Read the recorded baseline through this file; `src/thesis_ml/viz/outcome_probe.py` does.
- `ablation_01..05_*.yaml`: the five representational-toggle arms. Each extends `local_overfit_v2.yaml` and changes only its toggle(s) and its storage paths, so all five share V2's 100-epoch budget, LR trajectory, and EMA window. Launched by `tests/run_ablation_sweep.sh`, which passes no `--max-steps` so every arm runs its schedules to completion. All five pin all three toggles explicitly, so the promotion of `frozen_input_kv` left their conditions untouched. Note arms 01 and 05 were trained before the frozen-path RoPE fix (`SPEC.md` §14b, "Known defect"), so their curves reflect the pre-fix behavior.
- `memorization_01..03_*.yaml`: the memorization-probe family, asking whether the model can truly memorize the 10-replay subset once the things preventing it are removed — arm 1 turns off weight decay and fog, arm 2 oversamples the t=1 (fully-noised canvas) regime to 25%, arm 3 does both. Each extends `local_overfit_v2.yaml` and inherits its 100-epoch budget, LR trajectory, subset, and the now-default `frozen_input_kv`, so the comparison run is ablation arm 01. Launched by `tests/run_memorization_sweep.sh`.
- `local_overfit_v2_finetune.yaml`: the debut/outcome fine-tuning variant of the V2 profile.
- `local_full.yaml`: historical eight-epoch full-corpus V2 profile, retained for reproducibility.
- `smallTrainingTestV3.yaml`: current full-corpus run. It inherits the exact split and batch size 9, scales to 384 width / 12 layers / 6 × 64 attention / FFN 1536, accumulates five batches per optimizer step, trains for up to 50 epochs with ten-epoch dev patience, uses 10/70/20 WSD from `3e-4` to `3e-6`, power-samples high corruption and omission, weights PAD 0.1 and END 24.633333333333333, and keeps run state under `tests/output/smallTrainingTestV3/`.

Both the learning-rate horizon and the EMA averaging window are DERIVED from a
profile's optimizer-step run length (`max_steps: 0` -> `ceil(len(train_loader) / accumulation_steps) × epochs`), so
changing epochs or fixed accumulation re-fits both automatically. See `train.lr_schedule` and
`train.ema_horizon_ratio` in `../config/default.yaml`.

Uniform-state diffusion is the production default. V3 power-samples continuous
time and adds 5% exact terminal mass; historical profiles pin their old uniform
schedule. Confidence sharpening remains zero, the EB sampler has a 64-pass
ceiling, and there is no outcome-last constraint. Absorbing diffusion belongs in a
dedicated ablation profile with isolated checkpoint and output namespaces.

The local profiles use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction and share persisted clean replay artifacts. Manifests carry a semantic/config stamp and are rebuilt when these rules change.
