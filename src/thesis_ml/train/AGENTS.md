# train Subpackage Contract

## Purpose

- Own the training-time canvas corruption, the optimization loop and its metrics, and the synthetic smoke trainer.

## Ownership

- `corruption.py` owns MDLM-style canvas masking (`corrupt_batch`, `inverse_t_weights`, `_resolve_t`, `CorruptionOutput`).
- `loop.py` owns the step/epoch loop, loss reduction, and metrics (`BatchLoss`, `ValidationLog`, `auxiliary_confidence_loss`, timestep percentiles, per-class and horizon-bucketed loss logging).
- `train.py` owns the tiny synthetic smoke trainer (`run_smoke_train`, `make_synthetic_examples`).

## Local Contracts

- Canvas corruption is uniform i.i.d. `[MASK]` at a sampled global level *t* (MDLM-style). Per-timestep-varying corruption is input-side fog and is never applied to the canvas.
- Loss is masked-position cross-entropy over canvas positions only, reweighted by `1/t`; batch-shape padding is excluded from attention and loss.
- Per-class loss logging is populated from the first run. Auxiliary confidence loss is config-weighted (`confidence_loss_weight`, `0.0` disables) and derived from logits, not a separate head (`SPEC.md` §3).
- Maintain an EMA weight copy (decay `ema_decay`); use EMA weights for validation, the final checkpoint, sampling, and evaluation.
- Self-conditioning training uses the two-pass procedure at `self_cond_prob` and shares the inference interface.
- Epoch patience compares noisy resampled train loss against the best using the configured relative minimum improvement; a single flat epoch never stops a run.
- Optimizer/schedule fields (`lr`, `betas`, `weight_decay`, `warmup`, `lr_floor`, `grad_clip`, `accum`, `precision`, `epochs`, `early_stopping_*`) are config-owned.

## Work Guidance

- Extend the existing loop rather than forking a parallel trainer; the real pipeline (`pipeline/train_pipeline.py`) drives this loop.
- Keep metric definitions aligned with the epoch CSV / `metrics.jsonl` fields documented in `RUN.md`.

## Verification

- Training and corruption changes require `tests/test_training.py`; a passing smoke run must show decreasing loss with populated per-class logging (`SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
