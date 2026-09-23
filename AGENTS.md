# Thesis_ML Contract

## Purpose

- Own a **two-arm controlled comparison** for StarCraft II opponent-strategy prediction: a uniform-state discrete-diffusion model against a similarly sized autoregressive model, plus the shared preprocessing, configuration, and tests both depend on.
- The experimental question is whether discrete diffusion or autoregression performs better on this task **when everything else is held constant**: same corpus, same tokenization, same windows, same splits, same conditioning, and a matched non-embedding parameter budget.

## The Two Arms

| | Arm A | Arm B |
|---|---|---|
| Package | `thesis_diffusion` | `thesis_ar` |
| Method | Uniform-state discrete diffusion | Decoder-only autoregressive |
| **Framework** | **PyTorch** | **TensorFlow / Keras** |
| Environment | `.venv` (Windows, cu130 Torch, RTX 3070) | separate venv; real training on Linux cloud |
| Status | Established, measured | Scaffold — not yet implemented |

`thesis_shared` is the third package and the experiment's controlled constant. Both arms import it; neither arm may import the other.

**The framework split is deliberate and fixed.** Arm B is written in TensorFlow because a capstone course requires TensorFlow, and building the AR pipeline from scratch was the natural opportunity to satisfy that requirement. Do not propose unifying the arms on one framework, and do not write PyTorch in `thesis_ar` or TensorFlow in `thesis_diffusion`.

**The arms are never installed in the same environment.** A single Python process loads one CUDA runtime, so the cu130 Torch wheels and TensorFlow's bundled CUDA libraries conflict. See `packages/thesis-ar/src/thesis_ar/AGENTS.md`.

**No meaningful training runs locally for Arm B.** TensorFlow has had no native-Windows GPU support since 2.11, so locally it is CPU-only. Light verification (unit tests, shape checks, tiny smoke runs) is fine; all real AR training happens on Linux cloud compute.

## Ownership

- `packages/thesis-shared/src/thesis_shared/sequence_formats/` and `data/window_policies/` own the additive unconditioned joint sequence preview. `config/sequence_preview.yaml` explicitly selects its new default; `scripts/preview_sequence.py` inspects one replay without rebuilding manifests. Existing training entry points remain conditioned until joint training integration is implemented.

- `Log.md` and `Timelog.xlsx` hold the research journal and time records. `TACCS_Proposal.docx` and `TACCS_Proposal_V1.docx` hold the original and revised cluster functional-testing proposals; these describe proposed work, not implemented architecture or measured cluster capabilities.
- `Model_Architecture/` owns the exact current implementation/configuration reference for every learnable component and model-facing pipeline stage, plus the reusable update prompt that keeps the reference synchronized with source.
- `packages/thesis-shared/src/thesis_shared/data/windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests.
- `packages/thesis-diffusion/src/thesis_diffusion/data/dataset.py` owns lazy per-window example construction and per-serving fog for both training modes.
- `packages/thesis-shared/src/thesis_shared/data/feature_stats.py` owns train-split-only normalization statistics, deterministic artifact identity, and strict loading.
- `packages/thesis-diffusion/src/thesis_diffusion/data/collate.py` owns dynamic batch padding and exact input/canvas masks.
- `packages/thesis-diffusion/src/thesis_diffusion/pipeline/train_pipeline.py` owns config-only preprocessing, training, checkpoint, and resume orchestration.
- `config/default.yaml` owns canonical defaults; versioned overrides in `configs/` own reproducible local run profiles.
- `tests/overfit.bat`, `tests/smallTrainingTestV2.bat`, and `tests/smallTrainingTestV3.bat` are thin Windows launchers; training behavior remains owned by YAML and Python entry points.
- `tests/overfit.bat` launches `local_overfit_v2.yaml` and mirrors flushed training progress to both its visible terminal and `tests/output/overfitV2/console.log`.
- `scripts/prepare_training_report.py` owns deterministic, size-bounded evidence bundles for finished runs; `reports/training-runs/` owns their tracked chair-facing summaries and curated evidence.
- `prompts/training-run-summary/` owns the provider-neutral reporting workflow; `.agents/skills/training-run-summary/` and `.claude/skills/training-run-summary/` are thin discovery adapters to that one workflow.

## Local Contracts

- Full tokenization (including BPE vocabulary fitting), windowing, and manifest building are cloud-compute work. Local single-replay previews and bounded tests are debugging exceptions. The optional `config/sequence_preview_bpe.yaml` fits only its named replay, preserves legacy IDs and training defaults, and must not be treated as a production tokenizer or held-out evaluation.

- The additive joint preview uses `[BOS] [DELIMITER] ([SELF] self-content [ENEMY] enemy-content [DELIMITER])+ [SELF] self-outcome [ENEMY] enemy-outcome`, with END only at replay end. It has no fog or metadata features, and appends ownership IDs without altering legacy IDs. Its intended diffusion replacement support is the entire vocabulary; only position zero is clamped. Production training/corruption contracts below still describe the conditioned path. See `packages/thesis-shared/src/thesis_shared/sequence_formats/AGENTS.md` for the preview/integration boundary.

- Read the root `AGENTS.md`, this file, `CLAUDE.md`, and the current task-specific prompt before editing.
- Any change to model-facing data, vocabulary, features, sequence grammar, configuration, learnable modules, parameterization, corruption/loss, optimization/EMA, checkpoint compatibility, or sampling must update every affected section in `Model_Architecture/MODEL_ARCHITECTURE.md`, update `MODEL_ARCHITECTURE_DIAGRAM.mmd`, and regenerate its SVG/PNG in the same change. Use `Model_Architecture/UPDATE_PROMPT.md`, recompute derived values from live source, and remove superseded text; Git owns historical versions.
- Run Python for Arm A and the shared package only through `.venv\Scripts\python.exe` after confirming the venv exists. Arm B runs through its own separate interpreter and never through `.venv`.
- PyTorch is pinned through uv to the explicit official `pytorch-cu130` index in `packages/thesis-diffusion/pyproject.toml`. Preserve the `tool.uv.sources` mapping and regenerate the lock with uv when changing Torch.
- The repository is a uv workspace. Its three members under `packages/` are independently installable, and dependencies flow one way: each arm depends on `thesis-shared`, and neither arm depends on the other. Never add `torch` or `tensorflow` to `thesis-shared`.
- Local replay data is consumed at its native one-second cadence. Timing recovery must use the same configured cadence.
- Pretraining windows contain contiguous whole timesteps from one replay and are bounded independently by input and enemy-reconstruction token budgets.
- Debut/outcome windows tile non-overlapping input timesteps under the input token budget only; each debut canvas starts at the input-window start and runs to replay end or the canvas budget, so output horizons may overlap. Outcome mode owns a separate stamped manifest.
- Each batch row contains exactly one replay window. Do not pack sequences or add document masks.
- Fog is sampled while serving every example. Persisted artifacts and manifests must remain clean.
- Both training modes serve a clamped input region with per-timestep `[self records][fog-filtered enemy records][ONE DELIMITER]`, terminated by one `[EOS]`. Every canvas starts with clamped, unscored `[BOS]` at position 0 and the mutable/scored outcome at position 1. Input fog and canvas diffusion are independent. Default fog and diffusion-time draws use `Beta(2,1)` power sampling, with fog scaled to 0–0.8 and diffusion assigning 5% additional exact `t=1` mass. Uniform-state replacement noise remains restricted to `[PAD]`, `[DELIMITER]`, and content IDs; absorbing mode uses `[MASK]`.
- Static conditioning is learned jointly with token identity from allowlisted standardized continuous values, explicit continuous-validity bits, categorical cloak/buff values, and numeric allegiance. Statistics use valid observations only, are computed from selected training replays, persisted with a content identity, and must match resumes, warm starts, diagnostics, exports, and sampling checkpoints.
- Pretraining and fine-tuning both expand every replay into exactly two perspective-specific sample streams: `p1` as self/`p2` as enemy and `p2` as self/`p1` as enemy. Replay splitting happens before this expansion so both perspectives remain in the same train/dev/test split.
- Batch padding is dynamic. Padding masks must exclude batch-shape padding from attention and loss.
- CUDA attention prefers fused Flash SDPA and falls back only to memory-efficient SDPA with a broadcast boolean key mask; math fallback is forbidden. The order is a performance preference — both fused backends are correct. Measured 2026-08-07: the local Windows torch wheel ships NO Flash kernel, so memory-efficient attention is what actually runs on the RTX 3070 and that is fine, not a defect. `torch.backends.cuda.flash_sdp_enabled()` reports the preference, not availability, and must not be used to check whether Flash will run.
- The overfit profile treats its configured CUDA reserved-memory ceiling as a reclaim trigger: it empties unused allocator cache when the ceiling is reached and fails only if the post-trim reservation remains at or above the ceiling. It logs timing, throughput, PyTorch allocator memory, device-wide VRAM use, and the device-minus-reserved gap every step.
- The overfit and smallTrainingTestV3 profiles enable config-gated block activation checkpointing because full-size training was measured above their practical VRAM ceilings; profiles that do not override the canonical default retain the off path.
- The overfit profile uses batch size 10, validated for 20 real-data steps at 5.885 GiB peak reserved memory on the RTX 3070.
- The overfit profiles train on an explicitly named 10-train/3-dev replay subset selected at the corpus median input-token count, with early stopping disabled: 150 epochs for V1 (`local_overfit.yaml`), 100 for V2 (`local_overfit_v2.yaml`) and every `configs/ablation_*.yaml` arm that extends it, so V2 and the arms share one 3400-step budget and are comparable epoch for epoch. Named selection (`pipeline.train_replay_ids`/`dev_replay_ids`) replaces the seeded split entirely; every unnamed replay becomes test.
- BOTH the learning-rate schedule and the EMA averaging window are fitted to a run's real optimizer-step length, never to dataloader-batch count. `train.max_steps: 0` derives `ceil(len(train_loader) / accumulation_steps) × epochs`; changing epochs or fixed accumulation re-fits both automatically.
- `train.lr_schedule` selects `wsd`, `cosine`, or `linear`. WSD is the default with 500 fixed warmup optimizer steps, a stable peak phase filling the middle, and a final 20% linear decay to 0.01× peak. Historical V1/V2/local-full profiles explicitly pin their previous cosine/linear schedules.
- `smallTrainingTestV3.yaml` is the current full-corpus run: 384 width, 12 blocks, six 64-dimensional heads, FFN 1536, 50-epoch cap, six-row microbatches, seven-batch accumulation (42 windows and about 275k valid tokens per update), a 6.5 GiB reclaim-first CUDA reservation ceiling, and ten-epoch dev-loss patience. It writes all run-owned state below `tests/output/smallTrainingTestV3/`.
- `tests/SizeAblationTest.bat` launches the restartable capacity suite in ascending size order: 4.75M, 15.08M, 29.32M baseline, 30.21M deeper control, 59.36M, and 121.47M parameters. Every arm stops after three epochs while `train.schedule_horizon_epochs: 50` keeps LR and EMA on the reference V3 schedule. The driver skips only validated finished exports and stops at the first failed/blocked arm so larger models never leapfrog an unresolved smaller one.
- Production stochastic training is paired by `(pipeline.seed, epoch, manifest index)`: batch order, fog, diffusion time, corruption mask/replacements, and self-conditioning remain random but repeat exactly across size arms and resumes, independent of DataLoader worker assignment or microbatch size.
- The local-full pretraining run uses an exact 870-train/50-dev/remainder-test replay split and full reconstruction/future targets with clamped `[BOS]` at canvas index 0 and perspective-relative `[WIN]`/`[LOSS]` at index 1. Only BOS is position-clamped; the outcome is corrupted and sampled normally.
- The local-full run keeps workers persistent, trims unused CUDA cache after completed epochs, does not retain ignored step-log objects, and records current allocation, peak allocation, reservation, inactive-split allocator telemetry, device-wide memory use, and the device-minus-reserved gap.
- The overfit loader uses four persistent workers with four batches prefetched per worker; training batches drop raw metadata after worker-side feature construction, pin their custom batch tensors, and use non-blocking CUDA copies.
- Model scale, token budgets, paths, subset selection, epochs, schedule, accumulation, class weights, early stopping, and checkpoint intervals/subdirectories remain config-owned.
- Local runs write epoch CSV metrics, ten-per-epoch interval CSV metrics, and replay selections under their configured `tests/output/<run_name>/` log directory. A transient CSV writer lock is retried; a persistent lock redirects logging to a timestamped `*-continued-*.csv` containing readable prior history, and later resumes keep using the newest continuation instead of terminating training. In both modes, epoch metrics include mean and p50/p90/p95 input/future timestep counts, future-token loss bucketed at 1, 2-5, 6-10, 11-30, and 31+ prediction timesteps, cumulative attention-valid training tokens, cumulative distinct token IDs, average device-wide VRAM use, and average device-minus-PyTorch-reserved gap; batch-shape padding is excluded.
- Raw `tests/output/` state remains ignored and may contain multi-gigabyte checkpoints. Publish completed-run results only through the allowlisted report preparer: tracked bundles contain the finished config/metadata, epoch CSV, replay selection, compact derived facts, and loss curve, never tensor weights, step JSONL, console logs, caches, replay data, or absolute workstation paths.
- Canvas loss reports six read-only decompositions in both pipelines: seven per-class losses, four corruption buckets (`t_eq_1`, `[0.75,1)`, `[0.25,0.75)`, `[0,0.25)`), a ground-truth-preserved vs noised split keyed on token inequality rather than the corruption flag, p1/p2 perspective, future-distance buckets, and a rare-class-by-corruption cross of {win/loss, `[END]`, `[DELIMITER]`} x the four corruption buckets. `interval_metrics.csv` emits all of them ten times per epoch, each row scoped to its own slice; `epoch_metrics.csv` emits the same loss columns once per epoch.
- Both CSVs also carry token-level reporting per split, placed directly after that split's headline loss column: argmax accuracy and macro-F1 over the token vocabulary, each split by canvas state (`ground_truth_preserved`, `noised`), plus `bits_per_token` and `perplexity`. Accuracy and F1 are reported only per canvas state and never pooled into one number, because a pooled figure would mostly track the sampled corruption level. Micro-F1 is deliberately absent: with one prediction per scored position it equals accuracy exactly. Counts are pooled across microbatches and only then turned into ratios, never averaged as per-microbatch scores.
- `bits_per_token` and `perplexity` come from a SEPARATE unweighted cross-entropy accumulator, not from `train_loss`/`dev_loss`. Those are the class-weighted objective (`loss.class_loss_weights` boosts `[END]` and damps `[PAD]`), so exponentiating them would not be a perplexity of any distribution and would not be comparable between runs with different weights. `perplexity` is exactly `2 ** bits_per_token`. Both average over every scored position at the run's sampled corruption distribution, so they compare models trained on the same splits under the same corruption schedule, and are NOT comparable to an autoregressive language model's next-token perplexity.
- The rare-class cross emits 12 loss columns and 12 scored-position count columns per split. It is returned as an unreduced sum/count pair and pooled by total scored positions, never as a mean of per-microbatch means. Its count columns always carry every cell including explicit zeros, because a bucket containing no `[END]` token is the observation; only its loss columns follow the blank-when-empty convention.
- `train.interval_dev_evaluation` and `train.interval_train_evaluation` independently gate the dev and train halves of the interval rows, and both default to true. A disabled half leaves its columns blank and is reported once per epoch instead. Both disabled writes no interval row at all and creates no `interval_metrics.csv`, with the accumulation wiring left intact. The overfit profiles set both false; the overfitV2 fine-tune pins `interval_train_evaluation` back to true.
- Debut-mode full training also writes the same `finetune_report.json` metric schema as the overfitV2 fine-tune, using the true held-out test split.
- Epoch patience compares dev loss against its thresholded best using the configured relative minimum improvement. Best-checkpoint replacement is separately based on any strict dev-loss improvement.
- Absolute time and frame-derived values remain metadata only and must not enter model features.
- Entity instance IDs are variable-width digit strings used only for deterministic ordering. Slash-form current/maximum unit stats are encoded as current/max fractions before train-split standardization.
- Entity presence requires a finite `pos_(X,Y,Z)` tuple; lifecycle/non-position sentinels are treated as null and emit no entity token, while valid `(0,0,Z)` positions remain present. Individual nonnumeric feature sentinels are missing, never coerced into valid numeric zero.
- Tokenized artifacts and manifests are versioned and bound to source-file and vocabulary identities. Source or vocabulary drift must force preprocessing instead of silently reusing stale arrays.
- `thesis_diffusion.viz.diagnostics` always writes high-contrast, aligned ground-truth/prediction/error count figures. `--n-windows` is interpreted per selected replay. First-appearance timelines require `--first-appearance`; comparison CSV, input text, and final-canvas top-10 logit JSON exports consolidate multiple windows into one labelled artifact per export type and output-noise directory.
- `thesis_diffusion.viz.diagnostics --bypass-sampler` keeps those outputs unchanged while replacing iterative sampling with exactly one denoising forward pass from the selected process's terminal prior (uniform random non-`[MASK]` canvas by default; all-`[MASK]` only for the absorbing ablation).
- Uniform diffusion, dense GeGLU/sandwich-RMSNorm architecture, and process-stamped checkpoints form one compatibility boundary. Loaders must reject retired checkpoints before partial loading; repository-local retired checkpoints are removed only after the migration and verification complete.
- The three `model:` booleans `frozen_input_kv`, `segment_embeddings`, and `per_segment_positions` originated as prompt-009 ablation toggles. `frozen_input_kv` is the owner-promoted default (`true`); the other two remain experiments and default to `false`. An agent must never promote or enable the remaining experiments unprompted. `architecture_identity` is `ARCHITECTURE_ID + toggle_fingerprint(model_config)`; since `frozen_input_kv` and `per_segment_positions` add ZERO parameters, that string is the only barrier against a silent cross-arm checkpoint load. Never bump `ARCHITECTURE_ID` merely for a toggle, and never let a toggle reach `manifest_config_stamp` or `vocabulary_stamp`.

## Work Guidance

- Within an arm, extend the existing serializer, model, loss, and training loop instead of creating parallel implementations. This rule is about avoiding redundant code inside one arm; it does NOT forbid the second arm, which is a deliberate parallel implementation in a different framework and is the point of the experiment.
- Code shared by both arms belongs in `thesis_shared`, never duplicated into an arm. A fork of shared logic silently breaks the comparison, because the arms stop consuming identical data.
- Treat `Model_Architecture/MODEL_ARCHITECTURE.md` as the exact implemented-state reference for Arm A. Resolve conflicts among source, merged config, and the architecture reference in the same task.
- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- Split train/dev/test by replay before selecting local subsets to prevent window leakage.
- Preserve the full pretraining target grammar: leading perspective-relative `[WIN]`/`[LOSS]`, bounded in-window reconstruction, whole-timestep future continuation, then `[END] [PAD]*` for game end or direct `[PAD]*` for a boundary-truncated horizon.

## Verification

- Run `.venv\Scripts\python.exe -m pytest tests/ -q` for changes to the shared package or Arm A. Scope to `tests/` explicitly: `Model_Inference_Tests/` has its own runner (`run_inference_tests.py`) that prepares `sys.path` itself, and its scripts are not collectable by a bare root pytest invocation.
- Arm B tests are marked `tensorflow` and run from the AR environment, not `.venv`.
- Structural boundary checks (shared package stays framework-free; neither arm depends on the other) live in `tests/test_pipeline.py::test_workspace_packages_declare_the_framework_boundary`.
- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke before long runs.
- Launcher checks may use `--max-steps N`; CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- GPU claims require an environment where CUDA is visible; never infer VRAM from CPU runs.

## Child DOX Index

- `packages/thesis-shared/src/thesis_shared/AGENTS.md`: **the experiment's controlled constant.** Framework-agnostic core — config loading, tokenization/serialization, vocabulary, windowing, input features, standardization statistics, replay splits, canvas decode, timing recovery, build-order extraction and metrics. Imports no deep-learning framework; both arms depend on it.
- `packages/thesis-diffusion/src/thesis_diffusion/AGENTS.md`: **Arm A — discrete diffusion, PyTorch.** Indexes the `data`, `model`, `train`, `inference`, `eval`, `pipeline`, and `viz` subpackages.
- `packages/thesis-ar/src/thesis_ar/AGENTS.md`: **Arm B — autoregressive baseline, TensorFlow.** Currently a scaffold. Read this before writing any AR code; it is explicit that the arm is TensorFlow-only, that `torch` is forbidden in it, and that no meaningful training runs locally.
- `Model_Architecture/AGENTS.md`: Exact current model reference for Arm A, canonical Mermaid source, rendered SVG/PNG visual, deterministic renderer, source/impact map, freshness contract, and reusable update prompt.
- `config/AGENTS.md`: canonical `default.yaml` base configuration validated by `packages/thesis-shared/src/thesis_shared/config.py`.
- `configs/AGENTS.md`: local proof-of-life run profiles that override `default.yaml`.
- `data/AGENTS.md`: on-disk `raw/`/`processed/` dataset layout (git-ignored contents) plus the master entity-list builder and token dictionary.
- `scripts/AGENTS.md`: standalone context-window analysis and GPU pre-flight utilities.
- `tests/AGENTS.md`: pytest suite, owner-provided extractor fixtures, and thin Windows launchers.
- `Model_Inference_Tests/AGENTS.md`: strict admission and operating contract for read-only tests that directly measure or interpret held-out checkpoint inference performance.
- `prompts/AGENTS.md`: executable task prompts and the completed-prompt archive.
- `reports/AGENTS.md`: durable, size-bounded finished-run evidence bundles and thesis-chair summaries.
- `plans/AGENTS.md`: implementation plans derived from accepted prompts and current contracts.
- `research/AGENTS.md`: source-attributed research outputs, including the dated DiffusionGemma uniform-migration evidence, that inform but do not override project contracts.
- `diagnostics/AGENTS.md`: reproducible audits, investigations, and failure analyses.
- `notebooks/AGENTS.md`: exploratory notebooks whose reusable logic must graduate into the package.
- `experiments/AGENTS.md`: reproducible experiment definitions linked to versioned configs; generated runs remain ignored.
- `checkpoints/` and `.pipeline_cache/` hold generated run state; make no durable architecture claims from them.
- Root operating docs stay parent-owned: `SCHEMA.md`, `RUN.md`, `EVAL.md`, `CLAUDE.md`, `README.md`, and `TODO.md`. There is no single architecture source-of-truth document; binding rules live in this DOX contract tree, and Arm A implemented state lives in `Model_Architecture/MODEL_ARCHITECTURE.md`.
