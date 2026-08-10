# viz Subpackage Contract

## Purpose

- Own read-only, file-based diagnostics for trained checkpoints and held-out replay windows.

## Ownership

- `diagnostics.py` owns replay ingestion into scratch artifacts, EMA evaluation, static count/build-order figures, raw canvas text exports, and final-canvas top-k logit JSON.
- `outcome_probe.py` owns the canvas-position-1 outcome-token probe (`probe_arm`, `probe_batch`, `build_probe_dataloader`, `select_probe_indices`, `summarize`, `OutcomePositionRecord`, `ARMS`): per-ablation-arm JSON reporting the model's output distribution at the `[WIN]`/`[LOSS]` slot under high canvas noise.
- `__init__.py` describes the public visualization boundary.

## Local Contracts

- Reuse the existing preprocessing, dataset, sampler, decode, and evaluation harness; do not implement parallel inference or tokenization paths.
- Load the configured feature-statistics artifact and require it to match the checkpoint identity before constructing the diagnostic model.
- Write only under `--out-dir`. Checkpoints, replay sources, configs, and shared tokenized artifacts remain read-only.
- PNG/SVG/PDF count comparisons are the default output. Each window figure aligns ground-truth counts, predicted counts, and high-contrast under/exact/over error cells on the same entity/timestep axes.
- `--n-windows` limits windows per selected replay; there is no separate overall window cap.
- First-appearance timelines are emitted only with `--first-appearance` and are intended for models fine-tuned to emit debut/build-order targets.
- `--csv` and `--json` are independent opt-in exports and produce no files when omitted.
- `--bypass-sampler` preserves the same figures and optional exports but replaces iterative sampling with one forward pass from the configured process's `t=1` prior: uniform random non-`[MASK]` states by default, all `[MASK]` only for the absorbing ablation.
- Non-image exports preserve labelled filenames for one window. With multiple windows, `--show-input` writes labelled sections to `input_canvas.txt`, `--csv` writes all rows to `canvas_comparison.csv` with a leading `window` column, and `--json` writes all labelled examples to the existing `canvas_logits.json` array. Aggregate files are rewritten per run rather than appended across reruns.
- `canvas_logits.json` records each position's final output token, ground-truth token, and the top 10 vocabulary items with raw logits and softmax confidence values.
- `outcome_probe.py` defaults to the **TRAINED (raw) weights**, deliberately opposite to `diagnostics.py`'s EMA default: it measures what training produced, not what the sampler serves. `--ema` opts in to EMA, and every output JSON names the weight set under `weights`.
- The probe runs exactly ONE denoising forward pass per example at an explicit `t`. It never invokes the sampler, because iterative sampling would let later steps overwrite position 1 and would measure the sampler's schedule rather than the weights' distribution at a known noise level.
- Noise levels and corruption draws come from a fixed seed and the probed windows are selected deterministically, so every arm is measured on the same windows at the same per-example `t`; cross-arm comparison is paired, not merely similar.
- Window selection defaults to `--sample-mode strided` because dataloader order is grouped by replay and perspective, so a head sample can be single-outcome-class. Summaries report the `[WIN]`/`[LOSS]` balance actually sampled.
- Summaries separate genuinely-noised position-1 rows from rows the corruption left clean, and bucket the noised rows by `t`. Mass on the shown token only measures a copy prior on noised rows, and the measured quantities move by orders of magnitude across the `[0.9, 1.0]` band.
- `p_true_given_pair` (`true_mass / pair_mass`) divides "did the model find the slot" out of "did it call the game", separating two failure modes `pair_mass` alone conflates. Report the mass-weighted `pooled` value, not the per-example mean: rows with negligible `pair_mass` carry an essentially random ratio.
- `by_perspective` exists because the outcome label is perspective-relative — the same game is `[WIN]` from one side and `[LOSS]` from the other. Read `prefers_win_fraction` ACROSS p1/p2, never against 0.5: a standing class preference holds it constant, reading the game moves it. Both breakdowns also appear inside each `t` bucket so they can be read where the canvas is pure noise.
- `load_diagnostic_model` reconciles a checkpoint's pickled `ModelConfig` against the current schema (`_reconcile_model_config`), so checkpoints written before a field was added still load. Backfilled values only decide how the model is built; `validate_checkpoint_compatibility` still rejects any architecture-identity disagreement, so this can never load a checkpoint into the wrong architecture.

## Work Guidance

- Keep optional logit collection off the normal evaluation path because it requires one extra model forward pass over the completed canvas.
- Resolve token names through the shared content vocabulary and reserved special-token table.
- `outcome_probe.py` reuses `train_pipeline`'s own replay selection and checkpoint-directory resolution rather than re-deriving either, so it can never disagree with a profile about which examples the arm trained on or where its weights live. Keep `ARMS` in sync with `tests/run_ablation_sweep.sh`.

## Verification

- Visualization changes require `tests/test_viz.py`; sampler-backed logit changes also require `tests/test_sampler.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
