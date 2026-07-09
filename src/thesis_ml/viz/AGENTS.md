# viz Subpackage Contract

## Purpose

- Own read-only, file-based diagnostics for trained checkpoints and held-out replay windows.

## Ownership

- `diagnostics.py` owns replay ingestion into scratch artifacts, EMA evaluation, static count/build-order figures, raw canvas text exports, and final-canvas top-k logit JSON.
- `__init__.py` describes the public visualization boundary.

## Local Contracts

- Reuse the existing preprocessing, dataset, sampler, decode, and evaluation harness; do not implement parallel inference or tokenization paths.
- Write only under `--out-dir`. Checkpoints, replay sources, configs, and shared tokenized artifacts remain read-only.
- PNG/SVG/PDF count comparisons are the default output. Each window figure aligns ground-truth counts, predicted counts, and high-contrast under/exact/over error cells on the same entity/timestep axes.
- `--n-windows` limits windows per selected replay; there is no separate overall window cap.
- First-appearance timelines are emitted only with `--first-appearance` and are intended for models fine-tuned to emit debut/build-order targets.
- `--csv` and `--json` are independent opt-in exports and produce no files when omitted.
- `--bypass-sampler` preserves the same figures and optional exports but replaces iterative sampling with one all-`[MASK]` denoising forward pass per example; it is off by default.
- Non-image exports preserve labelled filenames for one window. With multiple windows, `--show-input` writes labelled sections to `input_canvas.txt`, `--csv` writes all rows to `canvas_comparison.csv` with a leading `window` column, and `--json` writes all labelled examples to the existing `canvas_logits.json` array. Aggregate files are rewritten per run rather than appended across reruns.
- `canvas_logits.json` records each position's final output token, ground-truth token, and the top 10 vocabulary items with raw logits and softmax confidence values.

## Work Guidance

- Keep optional logit collection off the normal evaluation path because it requires one extra model forward pass over the completed canvas.
- Resolve token names through the shared content vocabulary and reserved special-token table.

## Verification

- Visualization changes require `tests/test_viz.py`; sampler-backed logit changes also require `tests/test_sampler.py` and `tests/test_sampler_outcome_last.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
