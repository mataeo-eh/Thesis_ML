# scripts Contract

## Purpose

- Own standalone analysis and preflight utilities run outside the training pipeline.

## Ownership

- `estimate_context_window.py` owns dataset context-window analysis: it streams parquet metadata plus the two upgrade columns and writes token-length reports to `scripts/output/`.
- `gpu_smoke_test.py` owns the pre-flight GPU fit/throughput check that fabricates a correctly-shaped random batch (no dataset required) and reports peak VRAM and per-step time.
- `batch_interference_probe.py` owns the batch-versus-batch interference diagnostic: it restores a finished checkpoint, takes one optimizer step on each batch of a frozen epoch in turn, and records the loss that step causes on every other batch. It writes only to `scripts/output/batch_interference/<arm>/` and never mutates the probed run.
- `canvas_unigram_baseline.py` owns the CPU-only constant-predictor baseline for scored canvas targets. It streams config-derived manifest splits through the production dataset/collation mask, reports unweighted and live-weighted baselines overall and by all seven loss classes, and writes JSON plus a text summary under `scripts/output/canvas_unigram_baseline/`.
- `timestep_alignment_probe.py` owns the training-objective geometry diagnostic that measures how fixed-index positional cross entropy prices small delimiter-local SC2 semantic edits. It runs a model-independent pseudo-logit perturbation arm, an observational one-denoiser-pass arm over a named checkpoint at configurable corruption levels, and input-conditioning/model-free controls. It writes only to `scripts/output/timestep_alignment_probe/`.
- `rare_token_signal_probe.py` owns the per-token-type follow-up to that geometry diagnostic: it disaggregates the alignment overcount by token type. A model-independent arm computes each type's closed-form exposure to a one-position bounded shift (run-boundary fraction, expected hits, run length, in-timestep prefix count); an observational arm scores each type at three increasingly forgiving levels (exact coordinate, order-invariant inside the ground-truth timestep span, soft probability mass) plus a present-versus-absent span base-rate control and a per-type share of the weighted objective. It writes only to `scripts/output/rare_token_signal_probe/`.
- `prepare_training_report.py` owns read-only validation and extraction of a finished local run into a compact tracked bundle below `reports/training-runs/`. It reads metadata and epoch metrics but never checkpoint tensor payloads.
- `run_size_ablation.py` owns smallest-to-largest capacity-suite order, validated-finish skipping, resume detection, per-launch console logs, and sweep status. It delegates every training action to the canonical pipeline and stops on the first unresolved arm.
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
- Training-report preparation accepts only `completed_all_epochs` or `early_stopping`, requires the final valid epoch to match finished metadata, blocks architecture-identity drift, copies only the documented textual allowlist, rejects files above 10 MiB, and verifies that every produced report file is visible to Git.
- The report preparer detects legacy non-monotonic wall-clock resets across resumed processes. It sums segment terminal values as completed recorded fit time while explicitly marking that value as a lower bound; monotonic post-fix runs retain an exact cumulative total.

## Work Guidance

- Keep these utilities read-only against source data and side-effect-bounded to `scripts/output/`.
- The training-report preparer is the exception to the `scripts/output/` boundary: it writes only its deliberately durable allowlisted bundle beneath `reports/training-runs/` and leaves the source run untouched.
- Canvas baseline class membership uses the dataset's deterministic per-serving fog draw for the selected epoch; overall target counts do not depend on fog, while the observed/fogged class split does. Its weighted baseline must derive weights from `CanvasCrossEntropyLoss` and normalize by their scored-position sum.
- Canvas baseline serving reuses the production sequential DataLoader and worker-side collation. `--num-workers` overrides the profile worker count, and `0` is the bounded-memory single-process fallback; worker shutdown must remain explicit on completion, interruption, or failure.
- `timestep_alignment_probe.py` stays read-only against the checkpoint, run metrics, manifests, processed arrays, and replay sources. It belongs here and NOT in `Model_Inference_Tests/`: its primary deliverable is loss-function geometry, not checkpoint inference performance. It reuses the production config, vocabulary, split helpers, dataset, collation, `corrupt_batch`, `CanvasCrossEntropyLoss` weights, and `load_diagnostic_model` rather than building a parallel pipeline, defaults to EMA weights with an explicit `--raw` opt-out, and fails closed when the config-derived split disagrees with the run's recorded `replay_selection.json`.
- Its corruption sweep is COUPLED across noise levels: an identically seeded generator is rebuilt before every `corrupt_batch` call, so the Bernoulli draw and the replacement tokens are shared and the corrupted sets are nested. Changing that seeding would turn the sweep into unrelated random canvases and invalidate every by-`t` comparison.
- The model arm takes exactly ONE denoiser forward pass with `canvas_self_conditioning=None`, matching the iterative sampler's first step; it must never be described as sampler behavior.
- The oracle aligned content score is an order-invariant minimum-cost assignment INSIDE one ground-truth timestep content span, with duplicate token occurrences treated as distinct columns. Structural targets (outcome, `[DELIMITER]`, `[END]`, semantic `[PAD]`) are outside every span and can never be consumed or moved by the assignment. It is a deliberately optimistic diagnostic, not a likelihood, not the production loss, and not a proposal to train a matching algorithm.
- Every ratio it reports is formed once from pooled numerators and denominators; per-window means are never averaged.
- `rare_token_signal_probe.py` imports canvas segmentation, split resolution/verification, window selection, batching, coupled corruption seeding, and portable-path rendering from `timestep_alignment_probe.py` rather than reimplementing them, so the two diagnostics can never disagree about what a timestep is. It keeps the same read-only, EMA-default, fail-closed-on-split -mismatch, no-CPU-fallback contract.
- Its three recall levels are ordered by construction and must stay that way: positional recall <= timestep (order-invariant multiset overlap) recall, so the gap between them IS the alignment component of a type's error. Reporting either alone is uninterpretable.
- Soft-mass evidence is only admissible alongside the present-versus-absent span control. Expected count inside spans that contain a type proves timestep-level knowledge ONLY when it exceeds the rate inside spans that do not; a ratio of 1.0 is a sprayed corpus base rate. Never cite `exp/target` without `pres/abs`.
- Its per-token frequency buckets are roughly logarithmic on purpose. The population spans three orders of magnitude (probe ~7.6 occurrences per timestep, twilightcouncil ~0.018); linear buckets collapse the tail this probe exists to show.
- `TECH_BUILDING_NAMES` is a stated editorial choice, not a frequency cut, and is validated against the configured dictionary at run start. The frequency buckets are the objective restatement of the same claim and must be reported beside it.
- Its conclusions must respect the causal limit that the probed checkpoint was itself trained under positional CE: the model arm is observational, and only the model-independent geometry arm supports an objective-geometry claim on its own.

## Verification

- `tests/test_context_window_estimator.py` covers the context-window estimator; `tests/test_gpu_smoke_script.py` covers the fabricated benchmark batch grammar without requiring CUDA.
- `tests/test_canvas_unigram_baseline.py` pins the closed-form weighted optimum and proves that semantic `[PAD]`, clamped BOS, and batch-shape padding match the live loss mask/reduction.
- `tests/test_training_report.py` covers finish validation, first/best/final extraction, anomalous-row reporting, architecture matching, and exclusion of weights/checkpoints.
- `tests/test_rare_token_signal_probe.py` pins the run-boundary/run-length closed forms, the singleton-versus-long-run exposure asymmetry that is the probe's central claim, the uniform deletion-offset marginalization, in-timestep prefix counting, the three-level scoring summary and its ordering invariant, the present-versus-absent base-rate control, pooled-before-ratio bucketing, and that the named tech set exists in the shipped dictionary and sorts after the Protoss economy structures. It requires no CUDA, checkpoint, or replay data.
- `tests/test_timestep_alignment_probe.py` pins canvas segmentation (terminal `[END]` versus boundary-truncated `[PAD]`, empty groups), the exclusion of batch-shape padding while semantic `[PAD]` stays scored, every controlled perturbation's closed-form mismatch count and cross entropy, the assignment solver against brute force including duplicate occurrences, pooled-before-ratio aggregation, coupled-corruption nesting, and the write boundary. It requires no CUDA, checkpoint, or replay data.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
