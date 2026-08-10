# Thesis_ML Contract

## Purpose

- Own the uniform-state discrete-diffusion research package, its absorbing-process ablation, configuration, replay preprocessing, training, inference, and tests.

## Ownership

- `Model_Architecture/` owns the exact current implementation/configuration reference for every learnable component and model-facing pipeline stage, plus the reusable update prompt that keeps the reference synchronized with source.
- `src/thesis_ml/data/windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests.
- `src/thesis_ml/data/dataset.py` owns lazy per-window example construction and per-serving fog for both training modes.
- `src/thesis_ml/data/feature_stats.py` owns train-split-only normalization statistics, deterministic artifact identity, and strict loading.
- `src/thesis_ml/data/collate.py` owns dynamic batch padding and exact input/canvas masks.
- `src/thesis_ml/pipeline/train_pipeline.py` owns config-only preprocessing, training, checkpoint, and resume orchestration.
- `config/default.yaml` owns canonical defaults; versioned overrides in `configs/` own reproducible local run profiles.
- `tests/overfit.bat` and `tests/smallTrainingTestV2.bat` are thin Windows launchers; training behavior remains owned by YAML and Python entry points.
- `tests/overfit.bat` launches `local_overfit_v2.yaml` and mirrors flushed training progress to both its visible terminal and `tests/output/overfitV2/console.log`.

## Local Contracts

- Read the root `AGENTS.md`, this file, `CLAUDE.md`, and the current task-specific prompt before editing.
- Any change to model-facing data, vocabulary, features, sequence grammar, configuration, learnable modules, parameterization, corruption/loss, optimization/EMA, checkpoint compatibility, or sampling must update every affected section in `Model_Architecture/MODEL_ARCHITECTURE.md`, update `MODEL_ARCHITECTURE_DIAGRAM.mmd`, and regenerate its SVG/PNG in the same change. Use `Model_Architecture/UPDATE_PROMPT.md`, recompute derived values from live source, and remove superseded text; Git owns historical versions.
- Run Python only through `.venv\Scripts\python.exe` after confirming the venv exists.
- PyTorch is pinned through uv to the explicit official `pytorch-cu130` index. Preserve the `tool.uv.sources` mapping and regenerate `uv.lock` with uv when changing Torch.
- Local replay data is consumed at its native one-second cadence. Timing recovery must use the same configured cadence.
- Pretraining windows contain contiguous whole timesteps from one replay and are bounded independently by input and enemy-reconstruction token budgets.
- Debut/outcome windows tile non-overlapping input timesteps under the input token budget only; each debut canvas starts at the input-window start and runs to replay end or the canvas budget, so output horizons may overlap. Outcome mode owns a separate stamped manifest.
- Each batch row contains exactly one replay window. Do not pack sequences or add document masks.
- Fog is sampled while serving every example. Persisted artifacts and manifests must remain clean.
- Both training modes serve a clamped input region with per-timestep `[self records][fog-filtered enemy records][ONE DELIMITER]`, terminated by one `[EOS]`. Every canvas starts with clamped, unscored `[BOS]` at position 0 and the mutable/scored outcome at position 1. Input fog and canvas diffusion are independent. Default uniform diffusion draws replacement noise only from `[PAD]`, `[DELIMITER]`, and content IDs; absorbing mode uses `[MASK]`. Intentional exact-`t=1` oversampling defaults to zero.
- Static conditioning is learned jointly with token identity from allowlisted standardized continuous values, explicit continuous-validity bits, categorical cloak/buff values, and numeric allegiance. Statistics use valid observations only, are computed from selected training replays, persisted with a content identity, and must match resumes, warm starts, diagnostics, exports, and sampling checkpoints.
- Pretraining and fine-tuning both expand every replay into exactly two perspective-specific sample streams: `p1` as self/`p2` as enemy and `p2` as self/`p1` as enemy. Replay splitting happens before this expansion so both perspectives remain in the same train/dev/test split.
- Batch padding is dynamic. Padding masks must exclude batch-shape padding from attention and loss.
- CUDA attention prefers fused Flash SDPA and falls back only to memory-efficient SDPA with a broadcast boolean key mask; math fallback is forbidden. The order is a performance preference — both fused backends are correct. Measured 2026-08-07: the local Windows torch wheel ships NO Flash kernel, so memory-efficient attention is what actually runs on the RTX 3070 and that is fine, not a defect. `torch.backends.cuda.flash_sdp_enabled()` reports the preference, not availability, and must not be used to check whether Flash will run.
- The overfit profile treats its configured CUDA reserved-memory ceiling as a reclaim trigger: it empties unused allocator cache when the ceiling is reached and fails only if the post-trim reservation remains at or above the ceiling. It logs timing, throughput, PyTorch allocator memory, device-wide VRAM use, and the device-minus-reserved gap every step.
- The overfit profile enables config-gated block activation checkpointing because full-size fused-attention training was measured above the VRAM ceiling; other profiles retain the default-off path.
- The overfit profile uses batch size 10, validated for 20 real-data steps at 5.885 GiB peak reserved memory on the RTX 3070.
- The overfit profiles train on an explicitly named 10-train/3-dev replay subset selected at the corpus median input-token count, with early stopping disabled: 150 epochs for V1 (`local_overfit.yaml`), 100 for V2 (`local_overfit_v2.yaml`) and every `configs/ablation_*.yaml` arm that extends it, so V2 and the arms share one 3400-step budget and are comparable epoch for epoch. Named selection (`pipeline.train_replay_ids`/`dev_replay_ids`) replaces the seeded split entirely; every unnamed replay becomes test.
- BOTH the learning-rate schedule and the EMA averaging window are fitted to a run's real length, never to a fixed step count. `train.max_steps: 0` means the horizon is derived: `train_pipeline` injects `len(train_loader) * train.epochs` as `train.max_steps`; `TrainingLoop._lr_multiplier` spreads the decay over exactly that, and `TrainingLoop._resolve_ema_decay` sizes the EMA window to `train.ema_horizon_ratio ×` that (with `train.ema_decay` as a ceiling), so the EMA always completes inside the run instead of trailing a window sized for some other run's length. Changing `epochs` re-fits both automatically; a non-zero `max_steps` pins the LR horizon and must be updated by hand.
- `train.lr_decay_shape` selects the post-warmup decay curve: `cosine` (default; lingers at the peak, steep middle, flat tail) or `linear` (leaves the peak at once on a constant, shallower slope). V2 and the ablation arms use `linear` with `lr_floor_ratio: 0.03` to stop late-training loss hovering around a minimum; V1 and local-full keep `cosine` / `0.1`.
- The local-full pretraining run uses an exact 870-train/50-dev/remainder-test replay split and full reconstruction/future targets with clamped `[BOS]` at canvas index 0 and perspective-relative `[WIN]`/`[LOSS]` at index 1. Only BOS is position-clamped; the outcome is corrupted and sampled normally.
- The local-full run keeps workers persistent, trims unused CUDA cache after completed epochs, does not retain ignored step-log objects, and records current allocation, peak allocation, reservation, inactive-split allocator telemetry, device-wide memory use, and the device-minus-reserved gap.
- The overfit loader uses four persistent workers with four batches prefetched per worker; training batches drop raw metadata after worker-side feature construction, pin their custom batch tensors, and use non-blocking CUDA copies.
- Model scale, token budgets, paths, subset selection, epochs, and checkpoint intervals remain config-owned.
- Local runs write epoch CSV metrics, ten-per-epoch interval CSV metrics, and replay selections under their configured `tests/output/<run_name>/` log directory. A transient CSV writer lock is retried; a persistent lock redirects logging to a timestamped `*-continued-*.csv` containing readable prior history, and later resumes keep using the newest continuation instead of terminating training. In both modes, epoch metrics include mean and p50/p90/p95 input/future timestep counts, future-token loss bucketed at 1, 2-5, 6-10, 11-30, and 31+ prediction timesteps, cumulative attention-valid training tokens, cumulative distinct token IDs, average device-wide VRAM use, and average device-minus-PyTorch-reserved gap; batch-shape padding is excluded.
- Canvas loss reports six read-only decompositions in both pipelines: seven per-class losses, four corruption buckets (`t_eq_1`, `[0.75,1)`, `[0.25,0.75)`, `[0,0.25)`), a ground-truth-preserved vs noised split keyed on token inequality rather than the corruption flag, p1/p2 perspective, future-distance buckets, and a rare-class-by-corruption cross of {win/loss, `[END]`, `[DELIMITER]`} x the four corruption buckets. `interval_metrics.csv` emits all of them ten times per epoch, each row scoped to its own slice; `epoch_metrics.csv` emits the same loss columns once per epoch.
- The rare-class cross emits 12 loss columns and 12 scored-position count columns per split. It is returned as an unreduced sum/count pair and pooled by total scored positions, never as a mean of per-microbatch means. Its count columns always carry every cell including explicit zeros, because a bucket containing no `[END]` token is the observation; only its loss columns follow the blank-when-empty convention.
- `train.interval_dev_evaluation` and `train.interval_train_evaluation` independently gate the dev and train halves of the interval rows, and both default to true. A disabled half leaves its columns blank and is reported once per epoch instead. Both disabled writes no interval row at all and creates no `interval_metrics.csv`, with the accumulation wiring left intact. The overfit profiles set both false; the overfitV2 fine-tune pins `interval_train_evaluation` back to true.
- Debut-mode full training also writes the same `finetune_report.json` metric schema as the overfitV2 fine-tune, using the true held-out test split.
- Epoch patience compares noisy resampled train loss against the best loss using the configured relative minimum improvement; a single flat epoch never stops a run.
- Absolute time and frame-derived values remain metadata only and must not enter model features.
- Entity instance IDs are variable-width digit strings used only for deterministic ordering. Slash-form current/maximum unit stats are encoded as current/max fractions before train-split standardization.
- Entity presence requires a finite `pos_(X,Y,Z)` tuple; lifecycle/non-position sentinels are treated as null and emit no entity token, while valid `(0,0,Z)` positions remain present. Individual nonnumeric feature sentinels are missing, never coerced into valid numeric zero.
- Tokenized artifacts and manifests are versioned and bound to source-file and vocabulary identities. Source or vocabulary drift must force preprocessing instead of silently reusing stale arrays.
- `thesis_ml.viz.diagnostics` always writes high-contrast, aligned ground-truth/prediction/error count figures. `--n-windows` is interpreted per selected replay. First-appearance timelines require `--first-appearance`; comparison CSV, input text, and final-canvas top-10 logit JSON exports consolidate multiple windows into one labelled artifact per export type and output-noise directory.
- `thesis_ml.viz.diagnostics --bypass-sampler` keeps those outputs unchanged while replacing iterative sampling with exactly one denoising forward pass from the selected process's terminal prior (uniform random non-`[MASK]` canvas by default; all-`[MASK]` only for the absorbing ablation).
- Uniform diffusion, dense GeGLU/sandwich-RMSNorm architecture, and process-stamped checkpoints form one compatibility boundary. Loaders must reject retired checkpoints before partial loading; repository-local retired checkpoints are removed only after the migration and verification complete.
- The three `model:` booleans `frozen_input_kv`, `segment_embeddings`, and `per_segment_positions` are **prompt-009 ABLATION TOGGLES, NOT DEFAULTS AND NOT STAGED FEATURES.** All three default to `false`, every committed profile keeps them `false`, and all-off adds nothing to the current vocabulary-v2 baseline. Their presence is not an invitation to enable them: an agent must never turn one on unprompted, and promotion of any toggle to a default is the owner's decision on measured evidence (`SPEC.md` §14b). `architecture_identity` is `ARCHITECTURE_ID + toggle_fingerprint(model_config)`; since `frozen_input_kv` and `per_segment_positions` add ZERO parameters, that string is the only barrier against a silent cross-arm checkpoint load that would corrupt the ablation. Never bump `ARCHITECTURE_ID` merely for a toggle, and never let a toggle reach `manifest_config_stamp` or `vocabulary_stamp`.
- `SPEC.md` §14a lists changes that are discouraged but not banned — input/prompt KV caching, a separate input encoder, encoder-decoder or cross-attention conditioning. These require explicit owner confirmation BEFORE any code is written, must ship toggled off, and must be measured before being trusted. Reason through the alternative first and say so; do not implement provisionally pending review.

## Work Guidance

- Extend the existing serializer, model, loss, and training loop instead of creating parallel implementations.
- Treat `Model_Architecture/MODEL_ARCHITECTURE.md` as the exact implemented-state companion to normative `SPEC.md`. Resolve conflicts among source, merged config, `SPEC.md`, and the architecture reference in the same task.
- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- Split train/dev/test by replay before selecting local subsets to prevent window leakage.
- Preserve the full pretraining target grammar: leading perspective-relative `[WIN]`/`[LOSS]`, bounded in-window reconstruction, whole-timestep future continuation, then `[END] [PAD]*` for game end or direct `[PAD]*` for a boundary-truncated horizon.

## Verification

- Run `\.venv\Scripts\python.exe -m pytest -q` for package-wide changes.
- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke before long runs.
- Launcher checks may use `--max-steps N`; CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- GPU claims require an environment where CUDA is visible; never infer VRAM from CPU runs.

## Child DOX Index

- `Model_Architecture/AGENTS.md`: Exact current model reference, canonical Mermaid source, rendered SVG/PNG visual, deterministic renderer, source/impact map, freshness contract, and reusable GPT-5.6-sol update prompt.
- `src/thesis_ml/AGENTS.md`: the importable package contract; owns config loading and tokenization/serialization and indexes the `data`, `vocab`, `model`, `train`, `inference`, `eval`, and `pipeline` subpackages.
- `config/AGENTS.md`: canonical `default.yaml` base configuration validated by `src/thesis_ml/config.py`.
- `configs/AGENTS.md`: local proof-of-life run profiles that override `default.yaml`.
- `data/AGENTS.md`: on-disk `raw/`/`processed/` dataset layout (git-ignored contents) plus the master entity-list builder and token dictionary.
- `scripts/AGENTS.md`: standalone context-window analysis and GPU pre-flight utilities.
- `tests/AGENTS.md`: pytest suite, owner-provided extractor fixtures, and thin Windows launchers.
- `prompts/AGENTS.md`: executable task prompts and the completed-prompt archive.
- `plans/AGENTS.md`: implementation plans derived from accepted prompts and current contracts.
- `research/AGENTS.md`: source-attributed research outputs, including the dated DiffusionGemma uniform-migration evidence, that inform but do not override project contracts.
- `diagnostics/AGENTS.md`: reproducible audits, investigations, and failure analyses.
- `notebooks/AGENTS.md`: exploratory notebooks whose reusable logic must graduate into the package.
- `experiments/AGENTS.md`: reproducible experiment definitions linked to versioned configs; generated runs remain ignored.
- `checkpoints/` and `.pipeline_cache/` hold generated run state; make no durable architecture claims from them.
- Root architecture and operating docs stay parent-owned: `SPEC.md` (architecture source of truth), `SCHEMA.md`, `RUN.md`, `EVAL.md`, `CLAUDE.md`, `README.md`, and `TODO.md`.
