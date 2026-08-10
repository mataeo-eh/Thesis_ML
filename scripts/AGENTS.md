# scripts Contract

## Purpose

- Own standalone analysis and preflight utilities run outside the training pipeline.

## Ownership

- `estimate_context_window.py` owns dataset context-window analysis: it streams parquet metadata plus the two upgrade columns and writes token-length reports to `scripts/output/`.
- `gpu_smoke_test.py` owns the pre-flight GPU fit/throughput check that fabricates a correctly-shaped random batch (no dataset required) and reports peak VRAM and per-step time.
- `output/` holds generated reports (not durable contract material).

## Local Contracts

- Run scripts through `.venv\Scripts\python.exe` from the submodule root.
- `estimate_context_window.py` derives the default parquet location from the repository layout and must not embed or emit a machine-specific path; prefer repository-relative `--input-dir`/`--pattern`/`--output` overrides.
- Token accounting stays consistent with the model contract in both modes: input counts self + zero-fog enemy content plus one delimiter per timestep and one terminal `[EOS]`; output counts the `[BOS]`/outcome prefix, enemy content, per-timestep delimiters, and one terminal `[END]`; padding excluded.
- `gpu_smoke_test.py` requires a visible CUDA device; never infer VRAM from a CPU run. Its fabricated batch must use terminal input `[EOS]`, clamped canvas `[BOS]`, mutable position-1 `[WIN]`/`[LOSS]`, and the same loss mask as production. It derives the live vocabulary width from the configured dictionary unless `--vocab-size` explicitly overrides it.

## Work Guidance

- Keep these utilities read-only against source data and side-effect-bounded to `scripts/output/`.

## Verification

- `tests/test_context_window_estimator.py` covers the context-window estimator; `tests/test_gpu_smoke_script.py` covers the fabricated benchmark batch grammar without requiring CUDA.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
