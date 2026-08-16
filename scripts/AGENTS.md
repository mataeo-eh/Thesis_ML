# scripts Contract

## Purpose

- Own standalone analysis and preflight utilities run outside the training pipeline.

## Ownership

- `estimate_context_window.py` owns dataset context-window analysis: it streams parquet metadata plus the two upgrade columns and writes token-length reports to `scripts/output/`.
- `gpu_smoke_test.py` owns the pre-flight GPU fit/throughput check that fabricates a correctly-shaped random batch (no dataset required) and reports peak VRAM and per-step time.
- `batch_interference_probe.py` owns the batch-versus-batch interference diagnostic: it restores a finished checkpoint, takes one optimizer step on each batch of a frozen epoch in turn, and records the loss that step causes on every other batch. It writes only to `scripts/output/batch_interference/<arm>/` and never mutates the probed run.
- `canvas_unigram_baseline.py` owns the CPU-only constant-predictor baseline for scored canvas targets. It streams config-derived manifest splits through the production dataset/collation mask, reports unweighted and live-weighted baselines overall and by all seven loss classes, and writes JSON plus a text summary under `scripts/output/canvas_unigram_baseline/`.
- `run_batch_interference_probe.sh` owns the sequential launcher that runs that probe across the three `configs/memorization_*.yaml` arms.
- `output/` holds generated reports (not durable contract material).

## Local Contracts

- Run scripts through `.venv\Scripts\python.exe` from the submodule root.
- `estimate_context_window.py` derives the default parquet location from the repository layout and must not embed or emit a machine-specific path; prefer repository-relative `--input-dir`/`--pattern`/`--output` overrides.
- Token accounting stays consistent with the model contract in both modes: input counts self + zero-fog enemy content plus one delimiter per timestep and one terminal `[EOS]`; output counts the `[BOS]`/outcome prefix, enemy content, per-timestep delimiters, and one terminal `[END]`; padding excluded.
- `batch_interference_probe.py` must stay non-destructive with respect to the run it probes: it constructs its `TrainingLoop` with no metrics paths and no publishers, never calls `fit`, `save_checkpoint`, `scheduler.step`, or the EMA update, and restores a deep-copied model/optimizer snapshot before every probe step. Its restore is verified at the end of each run and the resulting `restore_max_abs_drift` is recorded in `batch_interference_meta.json`; a non-zero drift invalidates the deltas.
- `batch_interference_probe.py` builds its dataset, split, and batch order by calling `pipeline.train_pipeline`'s own helpers rather than reimplementing them, so the frozen batches match what the run trained on. It loads feature statistics but never recomputes them.
- Loss measurement defaults to fp32 while the probe's optimizer step keeps the run's configured precision: the step must be faithful, and bf16 cannot resolve the loss change a single late-schedule step produces.
- Probe console output and JSON provenance render paths inside the checkout relative to the submodule root so captured evidence does not embed machine-specific checkout paths.
- `gpu_smoke_test.py` requires a visible CUDA device; never infer VRAM from a CPU run. Its fabricated batch must use terminal input `[EOS]`, clamped canvas `[BOS]`, mutable position-1 `[WIN]`/`[LOSS]`, and the same loss mask as production. It derives the live vocabulary width from the configured dictionary unless `--vocab-size` explicitly overrides it.

## Work Guidance

- Keep these utilities read-only against source data and side-effect-bounded to `scripts/output/`.
- Canvas baseline class membership uses the dataset's deterministic per-serving fog draw for the selected epoch; overall target counts do not depend on fog, while the observed/fogged class split does. Its weighted baseline must derive weights from `CanvasCrossEntropyLoss` and normalize by their scored-position sum.

## Verification

- `tests/test_context_window_estimator.py` covers the context-window estimator; `tests/test_gpu_smoke_script.py` covers the fabricated benchmark batch grammar without requiring CUDA.
- `tests/test_canvas_unigram_baseline.py` pins the closed-form weighted optimum and proves that semantic `[PAD]`, clamped BOS, and batch-shape padding match the live loss mask/reduction.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
