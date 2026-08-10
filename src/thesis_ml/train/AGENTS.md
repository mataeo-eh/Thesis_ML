# train Subpackage Contract

## Purpose

- Own the training-time canvas corruption, the optimization loop and its metrics, and the synthetic smoke trainer.

## Ownership

- `corruption.py` owns process-selected uniform categorical corruption and absorbing-mask ablation corruption (`corrupt_batch`, time resolution, allowed-state sampling, inverse-`t` ablation weights, `CorruptionOutput`).
- `loop.py` owns the step/epoch loop, loss reduction, and metrics (`BatchLoss`, `ValidationLog`, `auxiliary_confidence_loss`, timestep percentiles, per-class and horizon-bucketed loss logging).
- `train.py` owns the tiny synthetic smoke trainer (`run_smoke_train`, `make_synthetic_examples`).

## Local Contracts

- Canvas corruption samples one `t ~ Uniform(schedule.min, schedule.max)` per example. Default uniform mode independently retains each eligible target with probability `1-t` or replaces it uniformly with `[PAD]`, `[DELIMITER]`, or a content ID; `[MASK]`, `[END]`, `[WIN]`, `[LOSS]`, `[BOS]`, and `[EOS]` are never replacement draws. The position-1 outcome remains eligible. Absorbing mode independently replaces eligible positions with `[MASK]`. Input tokens and canvas position-0 BOS are clamped and unchanged.
- Uniform-mode loss is unweighted clean-state cross-entropy over every valid target-canvas position except BOS, including semantic `[PAD]`; BOS and batch-shape padding are excluded. Absorbing ablation loss remains restricted to corrupted eligible positions and weighted by `1/max(t, 1e-4)`.
- Intentional exact-`t=1` oversampling is config-owned and defaults to `0.0`; ordinary training uses the continuous uniform time draw.
- Per-timestep-varying corruption does not exist on the canvas. Input-side fog is a separate per-example enemy-token omission process owned by `data/dataset.py`.
- Per-class loss logging is populated from the first run. Auxiliary confidence loss remains logits-derived and config-weighted but defaults to `0.0` because EB selection and adaptive stopping depend on calibrated entropy.
- Both modes use the stable seven-id observed/fogged/future/structural/outcome taxonomy (with mode-specific names) and emit input/future telemetry plus future-distance loss buckets.
- Maintain an EMA weight copy; use EMA weights for validation, the final checkpoint, sampling, and evaluation. The decay is derived per run by `_resolve_ema_decay` as `1 - 1/(train.ema_horizon_ratio × total_steps)`, clamped to at most `train.ema_decay`, so the averaging window scales with the run's length instead of being pinned to a fixed step count; `fit` re-derives it from the step budget it actually resolves. `_update_ema` additionally ramps the decay in as `min(target, (1+n)/(10+n))` over the first updates so the EMA is not dominated by the random initialization it was copied from.
- Self-conditioning training uses a stopped estimate pass and a loss-bearing pass, chooses the estimate independently per example at `self_cond_prob`, and shares the expected-embedding interface with inference.
- Checkpoint metadata stamps the diffusion process and architecture revision. Resume and warm-start reject incompatible retired checkpoints before loading any weights.
- Epoch patience compares noisy resampled train loss against the best using the configured relative minimum improvement; a single flat epoch never stops a run.
- Optimizer/schedule fields (`lr`, `betas`, `weight_decay`, `warmup`, `lr_floor`, `grad_clip`, `accum`, `precision`, `epochs`, `early_stopping_*`) are config-owned.
- Checkpoints persist `feature_statistics_identity`; resume and warm-start must reject missing or mismatched identities before loading weights.
- Production pipelines stream step metrics to disk without retaining returned log objects; validation aggregates scalar metrics on CPU. Epoch/interval CSV writes retry transient file locks and, when a lock persists, switch to a timestamped continuation containing readable prior history; new loop instances resume the newest continuation. CUDA runs report current/peak allocation, reservation, inactive split bytes, device-wide used memory from `cudaMemGetInfo`, and the device-minus-reserved gap. Epoch CSVs average the latter two across optimizer steps. A configured reserved-memory ceiling first trims reclaimable allocator cache and fails only if the post-trim reservation still breaches the ceiling; profiles may also trim unused cache at completed epoch boundaries via config.

## Work Guidance

- Extend the existing loop rather than forking a parallel trainer; the real pipeline (`pipeline/train_pipeline.py`) drives this loop.
- Keep metric definitions aligned with the epoch CSV / `metrics.jsonl` fields documented in `RUN.md`.
- When corruption, scored positions, class weighting, auxiliary objectives, self-conditioning, optimizer/scheduler, accumulation, precision, gradient handling, EMA, or checkpoint state changes, update every affected training/model-flow section in `../../../Model_Architecture/MODEL_ARCHITECTURE.md`, update the canonical `.mmd`, and regenerate its SVG/PNG using `UPDATE_PROMPT.md`.

## Verification

- Training and corruption changes require `tests/test_training.py`; a passing smoke run must show decreasing loss with populated per-class logging (`SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
