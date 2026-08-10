# configs/

Configuration files (YAML/JSON/TOML) for data pipelines, model hyperparameters,
and experiment settings. Keep configs declarative and version-controlled so every
experiment is traceable to the exact settings that produced it.

- `local_overfit.yaml`: V1 base profile. Owns the named 10-train / 3-dev median-sized replay subset, the model scale, warmup 40, and a 150-epoch cosine-to-0.1 schedule.
- `local_overfit_v2.yaml`: the current learnability probe and the ablation control surface. Isolated checkpoint/log/cache paths, early stopping disabled, and it overrides the run length and decay shape to 100 epochs (3400 optimizer steps) with `lr_decay_shape: linear` and `lr_floor_ratio: 0.03`. No per-class loss weighting — that is fine-tuning-only.
- `ablation_00_baseline.yaml`: the finished sweep's all-toggles-false baseline arm, pinned explicitly. It exists because `model.frozen_input_kv` became a default on 2026-08-09 and `local_overfit_v2.yaml` — which used to be the baseline — now resolves to `+frozen_input_kv`. Read the recorded baseline through this file; `src/thesis_ml/viz/outcome_probe.py` does.
- `ablation_01..05_*.yaml`: the five representational-toggle arms. Each extends `local_overfit_v2.yaml` and changes only its toggle(s) and its storage paths, so all five share V2's 100-epoch budget, LR trajectory, and EMA window. Launched by `tests/run_ablation_sweep.sh`, which passes no `--max-steps` so every arm runs its schedules to completion. All five pin all three toggles explicitly, so the promotion of `frozen_input_kv` left their conditions untouched. Note arms 01 and 05 were trained before the frozen-path RoPE fix (`SPEC.md` §14b, "Known defect"), so their curves reflect the pre-fix behavior.
- `memorization_01..03_*.yaml`: the memorization-probe family, asking whether the model can truly memorize the 10-replay subset once the things preventing it are removed — arm 1 turns off weight decay and fog, arm 2 oversamples the t=1 (fully-noised canvas) regime to 25%, arm 3 does both. Each extends `local_overfit_v2.yaml` and inherits its 100-epoch budget, LR trajectory, subset, and the now-default `frozen_input_kv`, so the comparison run is ablation arm 01. Launched by `tests/run_memorization_sweep.sh`.
- `local_overfit_v2_finetune.yaml`: the debut/outcome fine-tuning variant of the V2 profile.
- `local_full.yaml`: eight-epoch uniform-diffusion pretraining run with an exact 870 train / 50 dev / remainder test replay split, batch size 9, active self-conditioning, and no early stopping.

Both the learning-rate horizon and the EMA averaging window are DERIVED from a
profile's own run length (`max_steps: 0` -> `len(train_loader) × epochs`), so
changing `epochs` re-fits both automatically. See `train.lr_decay_shape` and
`train.ema_horizon_ratio` in `../config/default.yaml`.

Uniform diffusion is the production default. Profiles begin with zero intentional
terminal-time oversampling, zero confidence sharpening, a 64-pass EB-sampler
ceiling, and no outcome-last constraint. Absorbing diffusion belongs in a
dedicated ablation profile with isolated checkpoint and output namespaces.

The local profiles use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction and share persisted clean replay artifacts. Manifests carry a semantic/config stamp and are rebuilt when these rules change.
