# SPEC.md — SC2 Strategy Prediction via Uniform Discrete Diffusion

Single source of truth for all architecture decisions in this repository.

**Agent instructions:** Read this document in full before beginning any task. Decisions marked SETTLED are final — do not revisit, "improve," extend, or suggest alternatives to them. §11 parameters are PROVISIONAL config defaults — implement them as config fields, never hardcode them. §12 items are OPEN — do not resolve them and do not implement anything for them. §14 is a hard ban list. On any conflict between this document and CLAUDE.md, this document wins.

---

## 1. Project summary

Masters thesis system: predict opponent strategy in StarCraft II from partially observed game state, via self-supervised pretraining on replay data.

Core claim: strategy structure emerges from pretraining alone — no label supervision during representation learning.

Framing: joint bidirectional denoising plausibly suits strategy prediction because discriminative signal lives in joint correlations among hidden and future tokens. This is a suitability investigation, not a superiority claim over causal transformers. No autoregressive head-to-head comparison exists in scope.

Data extraction is complete and lives in a separate repository (`SC2-gamestate-extractor`, pysc2 + s2protocol engine-simulation parsing). This repository consumes its output. Replays sourced from aiarena.net; dataset at Kaggle `mataeoanderson/sc2-replay-data`.

## 2. Model family — SETTLED

- **Uniform-state multinomial discrete diffusion is the default process.** The forward process, clean-state prediction target, self-conditioning path, and sampler are adapted from DiffusionGemma and its released Gemma/Hackable Diffusion implementations. Absorbing `[MASK]` diffusion remains available only as the config-selected scientific ablation `diffusion.process: absorbing`; it must retain process-compatible corruption, loss, prior, and sampling semantics rather than mixing masked and uniform components. See `research/diffusiongemma-uniform-migration.md` for dated provenance and the adopt/adapt/reject rationale.
- Backbone: a dense bidirectional **Gemma 4-lineage** transformer with sandwich RMSNorm around each residual branch (pre-attention RMSNorm, post-attention RMSNorm, pre-FFN RMSNorm, post-FFN RMSNorm), dense **GeGLU** feed forwarding (`GELU(gate, approximate="tanh") * up`, then down projection), **Llama 3.1-style frequency-scaled RoPE** for sequence position, **vanilla multi-head attention (MHA), NOT grouped-query attention**, and **QK-norm** (per-head RMSNorm on queries/keys before RoPE, config-gated and default on). The stack remains dense and single-path: no MoE. The RoPE base and scaling factors are config fields so pretraining can use shorter sequences while inference can evaluate longer sequences without learned position tables or an architecture change. MHA is retained because GQA's KV-cache savings target autoregressive decoding; full-canvas diffusion has no autoregressive KV cache. Attention uses FlashAttention kernels via PyTorch SDPA with no causal mask and a padding mask only.
- DiffusionGemma is a mechanics and modern-backbone reference, not a wholesale topology template. This project adopts uniform corruption, clean-state prediction, self-conditioning, GeGLU/sandwich normalization, and uniform EB sampling while rejecting MoE routing, local/sliding attention, prompt KV caching, encoder-decoder separation, fixed 256-token canvases, and multi-canvas block autoregression.
- Each training example is one flat sequence: `[input region][canvas region]`. Full bidirectional attention over the entire sequence. The input region is clamped — never noised, never receives loss. The canvas region is noised; loss is computed on canvas positions only.
- Conditioning is clamping. There is no encoder-decoder split, no cross-attention conditioning, no separate prompt encoder.
- No SSM layers in v1 (see §12 for the shelved fallback).

## 3. Training objective — SETTLED

One unified SSL denoising task family. Both stages use the same clamped-input corruption task; they differ in the canvas body semantics and fine-tuning loss weighting.

**Input (clamped; both modes):**
- Interleaved per timestep: each timestep contributes `[self records][enemy records][ONE DELIMITER]` (exact grammar in §6).
- Fog mechanism: **entity omission**. Fogged tokens are removed from the input entirely. No placeholder tokens, no mask tokens, no count signal of any kind for omitted tokens. Fog applies uniformly to enemy content tokens of EVERY token kind — entities AND cumulative upgrades (no entity-only special case).
- Fog rate: a parameter of the corruption distribution, sampled once per example from the required `config.fog` distribution. Zero fog degenerates to clean-past-predict-future. Fog applies to the enemy sequence only.

**Target canvas (noised, receives loss):**
- Leading outcome token: ground truth always begins at position 0 with the perspective player's `[WIN]`/`[LOSS]` token. It is noised, predicted, sampled, and scored exactly like every other eligible canvas position; there is no positional sampler rule or `outcome_last` mechanism. The network learns the position-zero convention through gradient descent. Outcome/debut fine-tuning (§8) reuses the same layout.
- The enemy sequence only: full reconstruction of the enemy past/present (both observed and fogged portions) plus the enemy future continuation, regenerated jointly, following the outcome token.
- Canvas corruption samples one global `t ~ Uniform(0,1)` per example. In default uniform mode, each canvas position independently retains its clean token with probability `1-t`; otherwise it is replaced by a uniformly sampled allowed canvas state. A corruption draw may coincidentally equal the clean token. Input-side fog remains independent and never applies to the output canvas.
- Uniform-mode random states are sampled from every vocabulary ID except `[MASK]`. `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`, and content tokens may appear at any noised position. No grammar- or position-dependent restriction is applied. `[MASK]` remains reserved for the absorbing ablation and is invalid in a completed uniform-mode canvas.
- **No intentional terminal oversampling by default:** `diffusion.schedule.t_one_fraction` remains a config-owned experimental knob but defaults to `0.0`. The ordinary continuous uniform time draw is the training distribution unless an explicit experiment changes the field.
- **Per-epoch generator reseeding:** the training loop reseeds its corruption/self-conditioning generator to `base_seed + epoch_index` at every epoch boundary. Each epoch's masking stream is therefore a deterministic function of (seed, epoch), which keeps a resumed run's corruption draws aligned with the stream an uninterrupted run would have produced.
- At inference, uniform mode initializes every eligible canvas position independently from the same uniform state distribution. Absorbing ablation mode initializes eligible positions as `[MASK]`.
- `[PAD]` is a semantic canvas token for surplus output positions. Batch-shape padding remains separate and excluded by attention/loss masks.

**Loss:**
- Position-wise cross-entropy against canonically ordered targets (§5), canvas positions only.
- Per-token-class loss logging is mandatory from the first training run. Both modes retain stable dense ids 0–6 with mode-specific names:
  - **Pre-training (7 classes — `PRETRAIN_CLASS_ID_TO_NAME`):** enemy-observed, enemy-fogged, enemy-future, delimiter, end, pad, win-loss.
  - **Debut fine-tuning (7 classes — `DEBUT_CLASS_ID_TO_NAME`):** visible-debut, fogged-debut, future-debut, delimiter, end, pad, win-loss.
- **Uniform-mode pretraining objective:** unweighted clean-state `x0` cross-entropy over every valid target-canvas position, including semantic `[PAD]` targets. It does not use the Bernoulli corruption branch as a score mask and does not apply inverse-`t` weighting. Batch-shape padding alone is excluded. Pretraining never reads `config.loss.class_loss_weights`; that section must be absent from a pretraining config.
- **Absorbing ablation objective:** preserve masked-position-only cross-entropy with inverse-`t` weighting so the ablation remains a coherent MDLM/LLaDA-style process. The process selector owns this pairing; invalid cross-process objective combinations are not configurable.
- **Fine-tuning per-class loss weights** remain a config knob (`loss.class_loss_weights`, REQUIRED when `data.debut_mode=true`), default 1.0 (PROVISIONAL).
- **Loss-breakdown metrics:** both training modes log clean-state CE broken down by the example's sampled *t*-bucket (`t_eq_1`, `[0.7,1.0)`, `[0.5,0.7)`, `[0.3,0.5)`, `[0.0,0.3)`) and by player perspective (`p1`/`p2`). Uniform mode uses all valid canvas positions in these breakdowns; absorbing mode uses its scored masked positions. Both modes retain input/future telemetry and future-distance buckets.
- **Auxiliary confidence loss is an ablation only:** keep the logits-derived, config-weighted term, but default `confidence_loss_weight` to `0.0`. Overconfidence can distort both entropy-bounded selection and entropy-based stopping, so the uniform baseline relies on the clean-state objective unless an explicit experiment enables sharpening.
- **EMA (SOTA diffusion practice):** maintain an exponential-moving-average copy of the weights during training (decay ~0.9999); use EMA weights for validation, the final checkpoint, sampling, and evaluation. EMA is standard for diffusion training and one of the practices distinguishing it from AR pretraining.
- **Self-conditioning (config-gated, default on):** training uses a no-grad clean-state estimate pass followed by the loss-bearing pass; independently per example, the stopped estimate is used with probability `self_cond_prob=0.5`, otherwise a zero signal is used. Convert probabilities to an expected embedding with the shared token-embedding table, then apply RMSNorm -> dense GeGLU -> residual addition to the current canvas token embedding -> scale-less RMSNorm. The input region is untouched. Inference reuses the preceding denoising pass and adds no extra forward pass. It derives the expected embedding from the same temperature-shaped distribution used for that step's entropy and candidate sampling. Train and inference interfaces must remain aligned.

There is no copy mechanism of any kind. Input-to-output copying is a learned behavior produced by the loss.

## 4. Tokenization and vocabulary — SETTLED

- Raw atomic entity-level tokens only. One token per entity instance per timestep snapshot — unit counts emerge from token repetition. No BPE, no merges, no compound tokens, no learned tokenizer.
- Single shared vocabulary for input and output. Content tokens are **raw entity-type tokens — entirely location-agnostic**. The token identity carries NO spatial information of any kind. Position is input-only and lives entirely in the contextual encodings (§6): the exact (X,Y) coordinate from the extractor parquet is added to the input token's embedding. The output canvas is location-agnostic — it predicts entity-type presence, timing, and counts (by repetition), never position.
- Special tokens: `[MASK]` (noise-only state for the absorbing ablation), `[PAD]`, `[END]`, `[DELIMITER]`, and outcome tokens `[WIN]` / `[LOSS]` (used in both training modes). Uniform corruption/sampling excludes `[MASK]` and otherwise permits every state at every eligible canvas position; target grammar is learned rather than imposed through positional logit masks.
- Outputs never contain raw coordinates, frame numbers, or absolute times. The vocabulary contains no tokens for them.
- Concrete vocabulary contents are derived from the extractor schema (§13).

## 5. Serialization order — SETTLED

- Within a timestep, entities serialize in canonical order: primary sort by entity type ID; within-type tiebreak by unit ID (tiebreak key PROVISIONAL — the binding requirement is stable and deterministic).
- The same canonical ordering applies to input serialization and target construction.
- Targets are canonically ordered and the model learns to emit canonical order via position-wise CE. No permutation-invariant losses, no Hungarian matching, no set losses.

## 6. Input representation — SETTLED

Two distinct kinds of "position" exist in this system and must never be conflated:
- **Sequence position** — a token's index in the flat sequence. Encoded only with **Llama 3.1-style frequency-scaled RoPE**, applied to queries/keys in attention. Chosen specifically so that entity-counts-per-timestep and timestep-counts not seen during training do not break the model at inference. This is the only numerical sequence-position encoding the model receives: no learned absolute position table, absolute game clock, frame number, `game_loop`, or timestamp-derived feature.
- **Map position** — where a unit sits on the game map: the exact (X,Y) coordinate from the extractor parquet. A standardized *feature* of the entity, encoded through the joint **input-only** conditioning path below, never part of token identity. Unrelated to RoPE.

**Input grammar — interleaved per timestep in both modes.** Walking the window's timesteps in order, each timestep contributes `[self records][enemy records][ONE DELIMITER]`: all self records first, then the fog-filtered enemy records, closed by exactly ONE `[DELIMITER]`. The total input delimiter count therefore equals the window's timestep count.

Input embedding pipeline — lives in the MODEL, not the tokenizer (these are learned parameters trained by backprop), applied to input tokens in both modes:
1. Token embedding lookup (learned).
2. Build the allowlisted static feature vector from standardized `(map_x, map_y, unit stats)` and numeric allegiance (`+1` self, `-1` enemy, `0` structural/missing). The statistics artifact is computed from the selected training split only with float64 population moments; zero-variance features use unit scale. Absolute game time, frame number, `game_loop`, token type, and timestamp-derived values are prohibited from this path.
3. Project static features with `Linear(F,32) -> ReLU -> Linear(32,32) -> ReLU`, concatenate the result with the token embedding, and apply `Linear(d_model+32,d_model) -> GELU -> Linear(d_model,d_model)`. Add the residual only where static features exist. The final projection is exactly zero-initialized after generic model initialization, making the initial forward identical to token lookup while allowing the residual path to unlock through optimization.
4. RoPE applied in attention for sequence position over the combined input-plus-canvas sequence.
5. Timestep boundaries: `[DELIMITER]` tokens, present in both input and canvas.

The tokenizer (§4–5) emits one sequence of token records carrying token identity and source metadata. A model-facing allowlisted feature structure carries only map position, unit stats, and allegiance into the embedding stack. Feature-statistics artifacts carry a deterministic identity and source-split metadata; production training, resume, warm start, diagnostics, sampling, and exports must reject missing or mismatched identities. Absolute clock metadata may remain available outside the model for dataset ordering and post-sampling evaluation, but must never be copied into model inputs, embeddings, attention inputs, or targets. The tokenizer never computes embeddings or positional encodings.

Windows may begin mid-game; the model must infer game phase from observed game state and sequence structure rather than an absolute clock.

Pretraining windows are greedy contiguous runs of whole timesteps from one replay.
A timestep is added only while the zero-fog serialized input remains within its
budget and the full in-window enemy reconstruction remains within
`canvas_recon_fraction × canvas_budget_tokens`.
Successive default windows tile each replay without overlap. One window is one
batch sequence; sequence packing and cross-document masks are not used.

v1 uses delimiters only for timestep structure; a separate timestep-membership encoding is OPEN (§12) — do not implement one.

## 7. Output canvas semantics — SETTLED

- Flat token canvas with one fixed overall budget (config). Model-placed `[DELIMITER]`s partition it into contiguous timesteps. No per-timestep slot budgets.
- Each timestep's tokens are followed by one `[DELIMITER]`. After the final timestep of a replay: `[END]`, then semantic `[PAD]` targets. Collation may add further batch-shape `[PAD]` values, which are excluded from attention and loss.
- Absolute timing of canvas timesteps is recovered externally by arithmetic: clock of last input frame + fixed sampling interval × timestep index. The model never emits time.
- **Target truncation rule — whole timesteps only:** reconstruction contains exactly the window timesteps and future continuation admits a timestep only when all enemy tokens plus its `[DELIMITER]` fit. No partial timestep is ever emitted. If the game ends within budget, append `[END]` then `[PAD]` to budget. Otherwise stop at the last complete boundary and append `[PAD]` directly to budget.
- Grammar invariant (enforced in tests): a valid canvas is a leading `[WIN]`/`[LOSS]` outcome token, then `(timestep-tokens [DELIMITER])+`, followed by either `[END] [PAD]*` or `[PAD]*`. (The leading outcome token is present in both pre-training and debut fine-tuning; see §3 and §8.)
- In pre-training the canvas follows the clamped input and retains the observed/fogged/future content-class split (§3).

## 8. Outcome/debut training — SETTLED mechanism

- The `[WIN]`/`[LOSS]` outcome token is emitted in BOTH pre-training (§3) and debut fine-tuning at the same leading canvas position. `debut_mode` selects only the canvas body and loss-weighting contract: full enemy reconstruction plus future roll-out in pre-training versus sparse first-appearance debut events in fine-tuning. Both modes serve the same interleaved fogged input grammar.
- `debut_mode` gates `config.loss.class_loss_weights`; `config.fog`, future-distance loss decomposition, and input/future telemetry apply to both modes.
- Debut detection is UNIFIED across token kinds: an event debuts when its per-timestep count exceeds the running maximum seen so far in the window's scan (count increase). For cumulative upgrade tokens — whose per-timestep count is always 0 or 1 — this fires exactly once, at first appearance, reproducing the previous upgrade special case, which is deleted.
- Task: game outcome prediction. No classification head exists anywhere in this project.
- Mechanism: the outcome token (`[WIN]`/`[LOSS]`) occupies the leading ground-truth canvas position; the model denoises it and the normal continuation jointly. Sampling applies no special position-zero ordering or token restriction.
- Input: the same observed-gamestate input as pretraining.
- Outcome-mode inputs are separate, contiguous whole-timestep windows bounded by
  `input_budget_tokens` only. They tile each replay without overlapping input
  timesteps; pretraining's reconstruction-fraction bound does not shorten them.
- Each outcome-mode canvas starts at its input window's first timestep and emits
  debut events through replay end or `canvas_budget_tokens`, on whole-timestep
  boundaries. Adjacent input windows may therefore have overlapping output
  horizons, which is intentional.
- Outcome mode uses its own stamped window manifest; it must not overwrite or
  silently reuse the pretraining manifest.
- The outcome task may be warm-started as fine-tuning or trained directly by a
  dedicated profile; both paths use the same target, sampler, loss, and report
  contracts.

## 9. Inference and sampling — SETTLED mechanism, PROVISIONAL hyperparameters

- **Uniform mode uses DiffusionGemma-style nonmonotonic entropy-bounded sampling.** Initialize eligible positions from the uniform non-`[MASK]` state distribution. At each pass, temperature-shape clean-state logits, sample one categorical candidate per eligible position, compute full-vocabulary entropy, sort positions by ascending entropy, and accept the prefix satisfying `cumsum(sorted_entropy) - sorted_entropy <= entropy_bound`. Replace every nonaccepted eligible position with a fresh independent uniform state. Recompute acceptance from scratch every pass: previously accepted tokens may be renoised and revised, and no persistent committed mask exists.
- Uniform-mode defaults are a 64-pass hard ceiling, linear temperature `0.8 -> 0.4` (`exponent=1.0`), and `entropy_bound=0.1`. There is no minimum denoising-step count, confidence threshold, or forced minimum acceptance count.
- Adaptive stopping is evaluated per batch row over eligible positions and fires only when BOTH conditions hold: mean model entropy is below `0.005`, and argmax clean-token predictions are identical across two consecutive denoiser passes. Completed rows freeze while unfinished rows continue up to the 64-pass ceiling.
- Uniform candidate/noise sampling excludes `[MASK]` but applies no position-dependent grammar mask. Infill diagnostics may clamp revealed ground-truth positions; clamped positions are excluded from candidate sampling, renoising, entropy aggregation, and stability checks.
- **Absorbing ablation mode uses process-compatible EB unmasking:** initialize as `[MASK]`, sample categorical candidates, apply the same correct EB prefix formula among still-masked eligible positions, leave nonaccepted positions masked, and never remask accepted positions. It stops when all eligible positions are unmasked. It does not use uniform full renoising.
- Sampler traces record transient acceptance, renoising/unaccept events, entropy, argmax stability, temperature, canvas state, per-row stop reason, and actual step count. “Committed” terminology is reserved for the monotonic absorbing ablation and must not describe uniform-mode acceptance.
- Normal sampling performs no post-sampling model call. Optional final-logit diagnostics may perform one explicit extra pass and must remain off the normal evaluation path.

## 10. Evaluation — SETTLED

- Headline metrics: accuracy and F1 of predicted build orders against a deterministic build-order extraction tool run on ground-truth replays.
- Token cross-entropy is for training curves and model selection only. It is never a reported result.
- Evaluation keeps every decoded timestep; valid canvases cannot contain a partial final timestep (§7).
- Baselines (later phase, not v1): naive Bayes and SVM on naive features. Literature reference point: Synnaeve & Bessière, ~63–68% accuracy at 5 minutes.

## 11. PROVISIONAL config parameters

All of the following are config fields in one YAML file, validated by a dataclass. Changing any of them must require a config edit only — never a code change. Defaults below are placeholders pending fixture inspection and first runs; treat none of them as load-bearing.

| Parameter | Default | Notes |
|---|---|---|
| `sampling_interval_s` | 1 | Must equal the native cadence of the tokens consumed by the model; the current dataset is one-second cadence. |
| `input_budget_tokens` | 4096 | Hard per-window input bound. Windows grow only at whole-timestep boundaries. |
| `canvas_budget_tokens` | 4096 | Output canvas length for reconstruction plus future continuation. |
| `canvas_recon_fraction` | 0.5 | Maximum canvas fraction consumed by in-window enemy reconstruction; reserves the remainder for future prediction. |
| `fog_rate_distribution` | uniform(0.0, 0.8) | Sampled once per example in both modes; the `fog` section is always required (§3). |
| `data.feature_statistics_path` | `data/processed/feature_statistics.json` | Deterministic train-split normalization artifact; required by production model construction and checked against checkpoints. |
| `pipeline.prepare_feature_statistics` | false | Explicit permission to compute or replace the statistics artifact from the selected training replay artifacts only. |
| `within_type_tiebreak` | unit ID | §5 |
| `class_loss_weights` | all 1.0 | Keyed by §3's debut classes. Fine-tuning-only: required when `data.debut_mode=true`, rejected when false; uniform pretraining uses unweighted all-valid-position CE (§3). |
| `diffusion.process` | `uniform` | `uniform` is the production default; `absorbing` is the coherent masked-diffusion ablation (§2–3, §9). |
| `diffusion.schedule.t_one_fraction` | 0.0 | Experimental exact-terminal oversampling fraction. The baseline intentionally oversamples nothing (§3). |
| local `model.*` (d_model / layers / heads / ffn) | 256 / 10 / 4 / 1024 | ~10.7M proof-of-life shape with head dimension 64. Cloud scale remains config-only. |
| `diffusion.schedule` | linear, t ~ U(0,1) | Uniform mode uses retain-with-`1-t`, otherwise uniform replacement; absorbing ablation uses `[MASK]` replacement and inverse-`t` masked loss. |
| `train.*` (lr / betas / weight_decay / warmup / lr_floor / grad_clip / accum / precision) | 3e-4 / (0.9,0.95) / 0.1 / 2000 / 0.1×peak / 1.0 / as-needed / bf16 | Cosine decay to `lr_floor`; accumulation derives from the target effective batch size. |
| `train.ema_decay / confidence_loss_weight / val_interval` | 0.9999 / 0.0 / periodic | EMA on by default; confidence sharpening is an opt-in ablation (§3). |
| `train.epochs / early_stopping_*` | profile-owned / disabled by default | Epoch CSV metrics are always available; local overfit uses 0.1% relative improvement with five-epoch patience and a 200-epoch cap. |
| `model.qk_norm / model.self_conditioning / train.self_cond_prob` | true / true / 0.5 | QK-norm and expected-embedding GeGLU self-conditioning (§2–3). |
| `model.rope_theta / model.rope_scaling.*` | 500000 / llama3, factor 8, low/high 1/4, original context 8192 | Llama 3.1 frequency-scaled RoPE; all constants are config-owned and sequence length is not hard-capped in the rotary implementation |
| `sampler.max_steps / temperature / entropy_bound` | 64 / 0.8→0.4 / 0.1 | Hard ceiling with no minimum; temperature exponent `1.0` gives the linear schedule (§9). |
| `sampler.adaptive_stop / entropy_threshold / stability_steps` | true / 0.005 / 2 | Both confidence and stability conditions are required (§9). |

**Model-sizing note.** All `model.*` values are config; size changes require no code changes. The local ~10M shape validates the pipeline and is not a capability target. The intended cloud-scale endpoint remains approximately 450M parameters and is restored by configuration alone.

## 12. OPEN questions — do not resolve, do not implement

- Fog-rate curriculum (ablation candidate).
- Separate timestep-membership encoding alongside delimiter tokens.
- SSM+transformer hybridization: shelved fallback, used only if context length becomes binding. Constraint recorded for that contingency: attention on the input→output copy pathway must be full/global, not sliding-window.
- Loss-weight values for trivially-copyable token classes.
- Real-fog (in-game observed) data iteration; death-disambiguation tokens belong to that horizon, not this one.
- Sequence packing for throughput (multiple windows in one sequence). NOT used in v1 (one example per sequence). If adopted later, a document-level attention mask restricting attention within each packed example is required, since full bidirectional attention across packed examples forms spurious cross-example dependencies (LLaDA2.0). Deferred with packing.

## 13. Extractor output schema — PLACEHOLDER

- Sample extractor outputs live in `./tests/fixtures/` (provided by the project owner before prompt 002 runs).
- Prompt 002's first task: derive the schema from fixtures, document it in `./SCHEMA.md`, and pause for owner approval before implementing tokenization against it.
- Until `SCHEMA.md` exists and is approved, no code may assume field names or structure of extractor output.

## 14. Banned list — DO NOT IMPLEMENT, DO NOT SUGGEST

Each item below was explicitly evaluated and cut. Do not introduce them in any form, including "lightweight," "optional," or "configurable" versions:

- Set aggregators, set encoders, or pooling-over-entities modules of any kind
- Encoder-decoder architecture, including cross-attention conditioning
- Semi-autoregressive or block-autoregressive generation
- Prompt KV caching or a separate prompt encoder
- Mixture-of-experts routing, grouped-query attention, or cache-oriented local/sliding attention
- Copy mechanisms: pointer networks, copy gates, copy losses
- Classification heads (outcome prediction is generative, §8)
- Learned or compound tokenizers: BPE, merges, hierarchical clustering
- Strategy-label supervision anywhere in training
- Per-timestep output slot budgets
- Coordinates in the output vocabulary; frame numbers, `game_loop`, absolute times, or timestamp-derived values anywhere in model inputs, embeddings, or the output vocabulary
- Placeholder tokens for fogged entities (fog is omission)
- Death-signal tokens
- Permutation-invariant losses, Hungarian matching, set losses
- DBNs; JEPA-style objectives

## 15. Repository conventions

- `./prompts/` holds executable agent prompts (`NNN-name.md`); completed prompts move to `./prompts/completed/`.
- `./research/` research outputs, including the dated DiffusionGemma migration evidence in `research/diffusiongemma-uniform-migration.md`; `./plans/` plans; `./diagnostics/` diagnostics; `./tests/fixtures/` owner-provided sample extractor outputs.
- `CLAUDE.md` (created by prompt 001) carries coding conventions. This SPEC.md carries architecture truth and wins on conflict.
- Python + PyTorch. Tests via pytest. Configuration via one YAML file validated by a dataclass.

## 16. Global acceptance criteria

| Criterion | Owning prompt |
|---|---|
| Round-trip serialization fidelity tests pass | 002 |
| Smoke-train on tiny synthetic dataset: loss decreases; per-class loss logging present and populated | 005 |
| Sampler output grammar validity: §7 invariant holds on generated canvases | 006 |
| Evaluation harness computes accuracy/F1 vs build-order tool on held-out replays | 007 |

## 17. Cloud/runtime conventions — SETTLED

Training runs on cloud GPU compute; inference runs locally (RTX 3070). This split is intentional from day one and constrains the whole pipeline:

- **No hardcoded local paths anywhere.** Every path — data source, checkpoint output, logs — is config-driven. A path may resolve to a local directory (dev) or a remote/bucket location (cloud) without code changes.
- **Storage is abstracted to a configurable location.** Checkpoints and outputs write there; training must be **resumable** from there (cheap cloud compute is often preemptible/spot — checkpoint frequently to persistent storage so preemption loses little).
- **Data is not bundled.** It is fetched from a configured remote source (Kaggle `mataeoanderson/sc2-replay-data` and/or aiarena.net) and produced by the `SC2-gamestate-extractor` (separate repo). Data-acquisition is a **decoupled stage**, runnable independently of training (extraction is CPU-bound; training is GPU-bound — they need not share an instance or environment).
- **Reproducible env, single entry command.** The pipeline installs and runs from a clean checkout via uv (locked) and one entry command. Secrets (data-source creds, bucket creds) come from environment/config, never hardcoded, never committed.
- **Provider-agnostic interface, AWS as the initial target.** The pipeline runs on any Linux GPU instance via plain `git clone` + `uv sync` + run — **no Docker** (an unneeded complication until a concrete need forces it). The storage interface is generic (local path or remote bucket); the initial concrete backends are local filesystem and **AWS S3**, with EC2 GPU instances (Deep Learning AMI, CUDA/drivers preinstalled) for training and a cheap CPU instance for data-acquisition. No managed-service or orchestration lock-in (no Airflow/Prefect/K8s/SageMaker-specific glue).
- The existing master training and data-acquisition entry points follow these conventions; every prompt that touches paths, storage, data, or checkpoints must preserve them.
