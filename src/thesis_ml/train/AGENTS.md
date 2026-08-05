# train Subpackage Contract

## Purpose

- Own the training-time canvas corruption, the optimization loop and its metrics, and the synthetic smoke trainer.

## Ownership

- `corruption.py` owns process-selected uniform categorical corruption and absorbing-mask ablation corruption (`corrupt_batch`, time resolution, allowed-state sampling, inverse-`t` ablation weights, `CorruptionOutput`).
- `loop.py` owns the step/epoch loop, loss reduction, and metrics (`BatchLoss`, `ValidationLog`, `auxiliary_confidence_loss`, timestep percentiles, per-class and horizon-bucketed loss logging).
- `train.py` owns the tiny synthetic smoke trainer (`run_smoke_train`, `make_synthetic_examples`).

## Local Contracts

- Canvas corruption samples one `t ~ Uniform(schedule.min, schedule.max)` per example. Default uniform mode independently retains each target with probability `1-t` or replaces it with a uniformly sampled non-`[MASK]` state; the replacement may equal the target. Absorbing ablation mode independently replaces with `[MASK]`. Input tokens are clamped and unchanged.
- Uniform-mode loss is unweighted clean-state cross-entropy over every valid target-canvas position, including semantic `[PAD]`; only batch-shape padding is excluded. Absorbing ablation loss remains restricted to corrupted positions and weighted by `1/max(t, 1e-4)`.
- Intentional exact-`t=1` oversampling is config-owned and defaults to `0.0`; ordinary training uses the continuous uniform time draw.
- Per-timestep-varying corruption does not exist on the canvas. Input-side fog is a separate per-example enemy-token omission process owned by `data/dataset.py`.
- Per-class loss logging is populated from the first run. Auxiliary confidence loss remains logits-derived and config-weighted but defaults to `0.0` because EB selection and adaptive stopping depend on calibrated entropy.
- Both modes use the stable seven-id observed/fogged/future/structural/outcome taxonomy (with mode-specific names) and emit input/future telemetry plus future-distance loss buckets.
- Maintain an EMA weight copy (decay `ema_decay`); use EMA weights for validation, the final checkpoint, sampling, and evaluation.
- Self-conditioning training uses a stopped estimate pass and a loss-bearing pass, chooses the estimate independently per example at `self_cond_prob`, and shares the expected-embedding interface with inference.
- Checkpoint metadata stamps the diffusion process and architecture revision. Resume and warm-start reject incompatible retired checkpoints before loading any weights.
- Epoch patience compares noisy resampled train loss against the best using the configured relative minimum improvement; a single flat epoch never stops a run.
- Optimizer/schedule fields (`lr`, `betas`, `weight_decay`, `warmup`, `lr_floor`, `grad_clip`, `accum`, `precision`, `epochs`, `early_stopping_*`) are config-owned.
- Checkpoints persist `feature_statistics_identity`; resume and warm-start must reject missing or mismatched identities before loading weights.
- Production pipelines stream step metrics to disk without retaining returned log objects; validation aggregates scalar metrics on CPU. CUDA runs report current/peak allocation, reservation, inactive split bytes, device-wide used memory from `cudaMemGetInfo`, and the device-minus-reserved gap. Epoch CSVs average the latter two across optimizer steps. Profiles may trim unused cache at completed epoch boundaries via config.

## Work Guidance

- Extend the existing loop rather than forking a parallel trainer; the real pipeline (`pipeline/train_pipeline.py`) drives this loop.
- Keep metric definitions aligned with the epoch CSV / `metrics.jsonl` fields documented in `RUN.md`.

## Verification

- Training and corruption changes require `tests/test_training.py`; a passing smoke run must show decreasing loss with populated per-class logging (`SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
