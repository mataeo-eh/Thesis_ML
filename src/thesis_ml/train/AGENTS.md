# train Subpackage Contract

## Purpose

- Own the training-time canvas corruption, the optimization loop and its metrics, and the synthetic smoke trainer.

## Ownership

- `corruption.py` owns MDLM-style canvas masking (`corrupt_batch`, `inverse_t_weights`, `_resolve_t`, `CorruptionOutput`).
- `loop.py` owns the step/epoch loop, loss reduction, and metrics (`BatchLoss`, `ValidationLog`, `auxiliary_confidence_loss`, timestep percentiles, per-class and horizon-bucketed loss logging).
- `train.py` owns the tiny synthetic smoke trainer (`run_smoke_train`, `make_synthetic_examples`).

## Local Contracts

- Canvas corruption samples one `t ~ Uniform(mask_schedule.min, mask_schedule.max)` per example, then independently replaces each canvas position with `[MASK]` with probability `t`. Unmasked positions retain their ground-truth token; input tokens are clamped and unchanged.
- Loss is cross-entropy only where canvas corruption actually produced `[MASK]`, intersected with the canvas loss mask, and reweighted by `1/max(t, 1e-4)`; batch-shape padding is excluded.
- With the default 0.0-1.0 schedule, training spans nearly clean through nearly/all-masked canvases in expectation. Exact endpoints have measure zero under continuous sampling; finite canvases can still become completely unmasked or completely masked through the independent Bernoulli draws.
- Per-timestep-varying corruption does not exist on the canvas. Input-side fog is a separate per-example enemy-token omission process owned by `data/dataset.py`.
- Per-class loss logging is populated from the first run. Auxiliary confidence loss is config-weighted (`confidence_loss_weight`, `0.0` disables) and derived from logits, not a separate head (`SPEC.md` §3).
- Maintain an EMA weight copy (decay `ema_decay`); use EMA weights for validation, the final checkpoint, sampling, and evaluation.
- Self-conditioning training uses the two-pass procedure at `self_cond_prob` and shares the inference interface.
- Epoch patience compares noisy resampled train loss against the best using the configured relative minimum improvement; a single flat epoch never stops a run.
- Optimizer/schedule fields (`lr`, `betas`, `weight_decay`, `warmup`, `lr_floor`, `grad_clip`, `accum`, `precision`, `epochs`, `early_stopping_*`) are config-owned.
- Production pipelines stream step metrics to disk without retaining returned log objects; validation aggregates scalar metrics on CPU. CUDA runs report current/peak allocation, reservation, inactive split bytes, device-wide used memory from `cudaMemGetInfo`, and the device-minus-reserved gap. Epoch CSVs average the latter two across optimizer steps. Profiles may trim unused cache at completed epoch boundaries via config.

## Work Guidance

- Extend the existing loop rather than forking a parallel trainer; the real pipeline (`pipeline/train_pipeline.py`) drives this loop.
- Keep metric definitions aligned with the epoch CSV / `metrics.jsonl` fields documented in `RUN.md`.

## Verification

- Training and corruption changes require `tests/test_training.py`; a passing smoke run must show decreasing loss with populated per-class logging (`SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
