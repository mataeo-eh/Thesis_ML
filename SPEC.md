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
- Backbone: a dense bidirectional **Gemma 4-lineage** transformer with sandwich RMSNorm around each residual branch (pre-attention RMSNorm, post-attention RMSNorm, pre-FFN RMSNorm, post-FFN RMSNorm), dense **GeGLU** feed forwarding (`GELU(gate, approximate="tanh") * up`, then down projection), **Llama 3.1-style frequency-scaled RoPE** for sequence position, **vanilla multi-head attention (MHA), NOT grouped-query attention**, and **QK-norm** (per-head RMSNorm on queries/keys before RoPE, config-gated and default on). The stack remains dense and single-path: no MoE. The RoPE base and scaling factors are config fields so pretraining can use shorter sequences while inference can evaluate longer sequences without learned position tables or an architecture change. MHA is retained because GQA's KV-cache savings target autoregressive decoding; full-canvas diffusion has no autoregressive KV cache. Attention goes through PyTorch SDPA with no causal mask and a padding mask only. FlashAttention is the preferred kernel and is requested first; memory-efficient attention is the accepted fallback, and MATH is deliberately excluded so an unsupported shape errors instead of silently allocating O(seq^2). Either fused backend is correct — this is a performance preference. On the local Windows target the installed torch wheel ships no FlashAttention kernel at all, so memory-efficient attention is what actually runs there.
- DiffusionGemma is a mechanics and modern-backbone reference, not a wholesale topology template. This project adopts uniform corruption, clean-state prediction, self-conditioning, GeGLU/sandwich normalization, and uniform EB sampling while rejecting MoE routing, local/sliding attention, prompt KV caching, encoder-decoder separation, fixed 256-token canvases, and multi-canvas block autoregression.
- Each training example is one flat sequence: `[input region][canvas region]`. Full bidirectional attention over the entire sequence. The input region is clamped — never noised, never receives loss. The canvas region is noised; loss is computed on canvas positions only.
- Conditioning is clamping. There is no encoder-decoder split, no cross-attention conditioning, no separate prompt encoder. Those remain the strong preference; see §14a for the confirmation gate that applies before any of them may be built, and note that `model.frozen_input_kv` (§14b, promoted to default `true` on 2026-08-09) makes attention one-directional across the input/canvas seam. Clamping is still the conditioning mechanism — the input is one region of one stack, not a separate encoder — but it is no longer bidirectional across the seam by default.
- No SSM layers in v1 (see §12 for the shelved fallback).

## 3. Training objective — SETTLED

One unified SSL denoising task family. Both stages use the same clamped-input corruption task; they differ in the canvas body semantics and fine-tuning loss weighting.

**Input (clamped; both modes):**
- Interleaved per timestep: each timestep contributes `[self records][enemy records][ONE DELIMITER]` (exact grammar in §6).
- Every input sequence ends with exactly one `[EOS]` after its final timestep delimiter. `[EOS]` is part of the clamped input region and never appears in a target canvas.
- Fog mechanism: **entity omission**. Fogged tokens are removed from the input entirely. No placeholder tokens, no mask tokens, no count signal of any kind for omitted tokens. Fog applies uniformly to enemy content tokens of EVERY token kind — entities AND cumulative upgrades (no entity-only special case).
- Fog rate: sampled once per example from the required `config.fog` distribution. The default is a scaled `Beta(2,1)` power draw (`0.8 * U**0.5`), which retains support over `0.0–0.8` while sampling high enemy-omission rates more often than low ones. Zero fog still degenerates to clean-past-predict-future. Fog applies to the enemy sequence only.

**Target canvas (one clamped anchor; remaining positions noised and scored):**
- Ground truth always begins with `[BOS]` at position 0, followed by the perspective player's `[WIN]`/`[LOSS]` token at position 1. `[BOS]` is attended as the first canvas landmark but is permanently excluded from corruption, sampling, renoising, and loss; it is not part of the input region, so the frozen-input-KV ablation does not absorb it. This is a stable relative-RoPE landmark, not an absolute positional encoding.
- The position-1 outcome is noised, predicted, sampled, re-noised, and scored exactly like every other eligible canvas position. Excluding `[WIN]`/`[LOSS]` from the *replacement support* does not exempt the outcome position from the forward corruption branch.
- The enemy sequence only: full reconstruction of the enemy past/present (both observed and fogged portions) plus the enemy future continuation, regenerated jointly, following the two-token prefix.
- Canvas corruption samples one global continuous `t` per example from a configurable distribution. The default is `t = U**0.5`, equivalently `Beta(2,1)`, so the continuous component assigns 43.75% of its mass to `t >= 0.75` and 6.25% below `0.25`. In default uniform-state mode, each eligible canvas position independently retains its clean token with probability `1-t`; otherwise it is replaced by a uniformly sampled allowed noise state. A corruption draw may coincidentally equal the clean token. Input-side fog remains independent and never applies to the output canvas.
- Uniform-mode random noise states are sampled exactly from `[PAD]`, `[DELIMITER]`, and all content-token IDs. They never inject `[MASK]`, `[END]`, `[WIN]`, `[LOSS]`, `[BOS]`, or `[EOS]`. This restriction applies to the initial prior and every uniform re-noising draw, not to the clean-token output head: the model can still predict every legitimate target token. `[MASK]` remains reserved for the absorbing ablation and is invalid in a completed uniform-mode canvas.
- **Terminal conditioning coverage:** after the continuous draw, an independent Bernoulli branch forces exactly `t=1` for 5% of examples by default. This explicitly trains the no-clean-canvas regime encountered at inference while leaving the remaining 95% on the continuous power distribution.
- **Per-epoch generator reseeding:** the training loop reseeds its corruption/self-conditioning generator to `base_seed + epoch_index` at every epoch boundary. Each epoch's masking stream is therefore a deterministic function of (seed, epoch), which keeps a resumed run's corruption draws aligned with the stream an uninterrupted run would have produced.
- At inference, position-0 `[BOS]` is installed and clamped before sampling. Uniform mode initializes every other eligible canvas position independently from the same `[PAD]`/`[DELIMITER]`/content noise distribution. Absorbing ablation mode initializes eligible positions as `[MASK]`.
- `[PAD]` is a semantic canvas token for surplus output positions. Batch-shape padding remains separate and excluded by attention/loss masks.

**Loss:**
- Position-wise cross-entropy against canonically ordered targets (§5), canvas positions only.
- Per-token-class loss logging is mandatory from the first training run. Both modes retain stable dense ids 0–6 with mode-specific names:
  - **Pre-training (7 classes — `PRETRAIN_CLASS_ID_TO_NAME`):** enemy-observed, enemy-fogged, enemy-future, delimiter, end, pad, win-loss.
  - **Debut fine-tuning (7 classes — `DEBUT_CLASS_ID_TO_NAME`):** visible-debut, fogged-debut, future-debut, delimiter, end, pad, win-loss.
- **Uniform-mode pretraining objective:** class-weighted clean-state `x0` cross-entropy over every valid target-canvas position except clamped `[BOS]`, including semantic `[PAD]` targets. It does not use the Bernoulli corruption branch as a score mask and does not apply inverse-`t` weighting. Clamped `[BOS]` and batch-shape padding are excluded. Both modes require `config.loss.class_loss_weights`; the full-corpus V3 profile uses `[PAD]=0.1`, `[END]=24.633333333333333`, and 1.0 for the other classes. The END multiplier is the measured train-window ratio `42,862 / 1,740`, equalizing aggregate END exposure with the once-per-window outcome class.
- **Absorbing ablation objective:** preserve masked-position-only cross-entropy with inverse-`t` weighting so the ablation remains a coherent MDLM/LLaDA-style process. The process selector owns this pairing; invalid cross-process objective combinations are not configurable.
- **Per-class loss weights** are a required config knob in both modes; profiles may pin unit weights to preserve historical experiment semantics.
- **Loss-breakdown metrics:** both training modes log clean-state CE broken down by the example's sampled *t*-bucket (`t_eq_1`, `[0.75,1.0)`, `[0.25,0.75)`, `[0.0,0.25)`) and by player perspective (`p1`/`p2`). Uniform mode uses all valid canvas positions in these breakdowns; absorbing mode uses its scored masked positions. Both modes retain input/future telemetry and future-distance buckets.
- **Auxiliary confidence loss is an ablation only:** keep the logits-derived, config-weighted term, but default `confidence_loss_weight` to `0.0`. Overconfidence can distort both entropy-bounded selection and entropy-based stopping, so the uniform baseline relies on the clean-state objective unless an explicit experiment enables sharpening.
- **EMA (SOTA diffusion practice):** maintain an exponential-moving-average copy of the weights during training; use EMA weights for validation, the final checkpoint, sampling, and evaluation. EMA is standard for diffusion training and one of the practices distinguishing it from AR pretraining. The decay is not a fixed constant: it is DERIVED from the run's own optimizer-step horizon so the averaging window always completes inside the run. `train.ema_horizon_ratio` (default `0.1`) sets the window as a fraction of total steps and `train.ema_decay` (`0.9999`, a ~10,000-step window) is the ceiling, so a long run behaves as the constant always did while a short run gets a proportionally shorter window instead of an EMA still dominated by early-training weights at its final step.
- **Self-conditioning (config-gated, default on):** training uses a no-grad clean-state estimate pass followed by the loss-bearing pass; independently per example, the stopped estimate is used with probability `self_cond_prob=0.5`, otherwise a zero signal is used. Convert probabilities to an expected embedding with the shared token-embedding table, then apply RMSNorm -> dense GeGLU -> residual addition to the current canvas token embedding -> scale-less RMSNorm. The input region is untouched. Inference reuses the preceding denoising pass and adds no extra forward pass. It derives the expected embedding from the same temperature-shaped distribution used for that step's entropy and candidate sampling. Train and inference interfaces must remain aligned.

There is no copy mechanism of any kind. Input-to-output copying is a learned behavior produced by the loss.

## 4. Tokenization and vocabulary — SETTLED

- Raw atomic entity-level tokens only. One token per entity instance per timestep snapshot — unit counts emerge from token repetition. No BPE, no merges, no compound tokens, no learned tokenizer.
- An entity emits a token only when its parquet `pos_(X,Y,Z)` value is a finite coordinate tuple. Nonnumeric lifecycle/sentinel text is treated as null regardless of non-null storage; valid `(0,0,Z)` remains present. Individual nonnumeric feature sentinels are missing values, not numeric zero.
- Single shared vocabulary for input and output. Content tokens are **raw entity-type tokens — entirely location-agnostic**. The token identity carries NO spatial information of any kind. Position is input-only and lives entirely in the contextual encodings (§6): the exact (X,Y) coordinate from the extractor parquet is added to the input token's embedding. The output canvas is location-agnostic — it predicts entity-type presence, timing, and counts (by repetition), never position.
- Special tokens are the contiguous IDs 0–7: `[MASK]`, `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`, `[BOS]`, `[EOS]`. Content IDs begin immediately at 8; there is no reserved-ID hole. Uniform corruption/prior/renoising draws only `[PAD]`, `[DELIMITER]`, and content tokens. The clean-state output head remains vocabulary-wide so it can predict `[END]` and the position-1 outcome.
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

**Input grammar — interleaved per timestep in both modes.** Walking the window's timesteps in order, each timestep contributes `[self records][enemy records][ONE DELIMITER]`: all self records first, then the fog-filtered enemy records, closed by exactly ONE `[DELIMITER]`. The total input delimiter count therefore equals the window's timestep count, and one `[EOS]` then terminates the complete input sequence.

Input embedding pipeline — lives in the MODEL, not the tokenizer (these are learned parameters trained by backprop), applied to input tokens in both modes:
1. Token embedding lookup (learned).
2. Build the allowlisted static feature vector from X/Y map position; health/energy/shield fractions; lossless facing sine/cosine; radius, build progress, cooldown, per-entity upgrades, cargo, order count, approved boolean flags, ideal harvesters, buff durations, and detect range; complete categorical `CloakState`; sparse categorical raw-protocol buff IDs through the corpus-audited maximum `302`; per-continuous-field validity; and numeric allegiance (`+1` self, `-1` enemy, `0` structural/missing). Z, assigned harvesters, radar range, rally coordinates, display type, and raw relationship tags are excluded. Statistics use only valid continuous observations from the selected training split, with float64 population moments and unit scale for zero variance. Invalid continuous values become neutral zero only after standardization while their validity bits remain explicit. Absolute game time, frame number, `game_loop`, token type, and timestamp-derived values are prohibited from this path.
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
- Grammar invariant (enforced in tests): a valid canvas is `[BOS]`, then exactly one `[WIN]`/`[LOSS]` outcome token, then `(timestep-tokens [DELIMITER])+`, followed by either `[END] [PAD]*` or `[PAD]*`. `[BOS]` and the outcome prefix are present in both pre-training and debut fine-tuning; `[EOS]` is forbidden on the canvas.
- In pre-training the canvas follows the clamped input and retains the observed/fogged/future content-class split (§3).

## 8. Outcome/debut training — SETTLED mechanism

- `[BOS]` at position 0 and the `[WIN]`/`[LOSS]` outcome at position 1 are emitted in BOTH pre-training (§3) and debut fine-tuning. `debut_mode` selects only the canvas body and loss-weighting contract: full enemy reconstruction plus future roll-out in pre-training versus sparse first-appearance debut events in fine-tuning. Both modes serve the same EOS-terminated interleaved fogged input grammar.
- `debut_mode` gates `config.loss.class_loss_weights`; `config.fog`, future-distance loss decomposition, and input/future telemetry apply to both modes.
- Debut detection is UNIFIED across token kinds: an event debuts when its per-timestep count exceeds the running maximum seen so far in the window's scan (count increase). For cumulative upgrade tokens — whose per-timestep count is always 0 or 1 — this fires exactly once, at first appearance, reproducing the previous upgrade special case, which is deleted.
- Task: game outcome prediction. No classification head exists anywhere in this project.
- Mechanism: clamped `[BOS]` occupies position 0; the outcome token occupies position 1 and is denoised jointly with the normal continuation. Sampling protects only the BOS anchor; the outcome remains fully mutable.
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

- **Uniform mode uses DiffusionGemma-style nonmonotonic entropy-bounded sampling.** Clamp position-0 `[BOS]` and initialize the remaining eligible positions from the uniform `[PAD]`/`[DELIMITER]`/content noise distribution. At each pass, temperature-shape clean-state logits, sample one categorical candidate per eligible position, compute full-vocabulary entropy, sort positions by ascending entropy, and accept the prefix satisfying `cumsum(sorted_entropy) - sorted_entropy <= entropy_bound`. Replace every nonaccepted eligible position with a fresh independent allowed noise state. Recompute acceptance from scratch every pass: previously accepted tokens may be renoised and revised, and no persistent committed mask exists.
- Uniform-mode defaults are a 64-pass hard ceiling, linear temperature `0.8 -> 0.4` (`exponent=1.0`), and `entropy_bound=0.1`. There is no minimum denoising-step count, confidence threshold, or forced minimum acceptance count.
- Adaptive stopping is evaluated per batch row over eligible positions and fires only when BOTH conditions hold on the same pass: mean model entropy is below `0.005`, and the pass is a **denoiser fixed point** — the clean-state argmax equals, at every mutable position, the canvas that pass read. The stability half is a predicate on the STATE, not on a history of predictions; an argmax-versus-previous-argmax test never inspects the canvas and can certify a row whose renoised positions all disagree with the model. Completed rows freeze while unfinished rows continue up to the 64-pass ceiling.
- **Terminal passes and the returned canvas.** A pass is terminal for a row when its adaptive stop fires there or when the 64-pass ceiling ends the row there. On a terminal pass the entropy budget is not applied: every eligible position takes its categorical candidate and nothing is renoised. The budget bounds the error of committing many positions while a later pass can still revise them, so it is meaningless when no later pass exists, and renoising there would place noise the process can never remove into the returned result. This uses the distribution already computed for that pass and adds no model call and no hyperparameter.
- **Result contract.** The returned canvas is the state the stop decision refers to. Every mutable position holds a draw from that row's terminal-pass temperature-shaped clean-state distribution, and no position ever holds a uniform renoise draw. The sampler reports the 1-based pass that finalized each row, and each row's stop reason states what its canvas is certified to be: `adaptive_entropy_stability` (a certified fixed point), `max_steps` (fully committed, not certified), `absorbing_complete`, or `no_eligible`.
- Uniform clean-token candidate sampling masks only `[MASK]`; uniform prior/renoising draws use `[PAD]`, `[DELIMITER]`, or content tokens. Position-0 `[BOS]` and any infill-revealed positions are clamped and excluded from candidate sampling, renoising, entropy aggregation, and stability checks.
- **Absorbing ablation mode uses process-compatible EB unmasking:** initialize as `[MASK]`, sample categorical candidates, apply the same correct EB prefix formula among still-masked eligible positions, leave nonaccepted positions masked, and never remask accepted positions. It stops when all eligible positions are unmasked. It does not use uniform full renoising. It shares only the terminal-pass rule, which commits any positions still masked when the ceiling arrives so `[MASK]` never survives into a returned canvas; that remains monotone unmasking.
- Sampler traces record transient acceptance, renoising/unaccept events, entropy, the per-row fixed-point certificate, per-row terminal-pass flags, temperature, canvas state, per-row stop reason, and actual step count. “Committed” terminology is reserved for the monotonic absorbing ablation and must not describe uniform-mode acceptance.
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
| `fog_rate_distribution` | scaled Beta(2,1) over 0.0–0.8 | Sampled once per example in both modes; the `fog` section is always required (§3). |
| `data.feature_statistics_path` | `data/processed/feature_statistics.json` | Deterministic train-split normalization artifact; required by production model construction and checked against checkpoints. |
| `pipeline.prepare_feature_statistics` | false | Explicit permission to compute or replace the statistics artifact from the selected training replay artifacts only. |
| `within_type_tiebreak` | unit ID | §5 |
| `class_loss_weights` | PAD 0.1 / END 24.633333333333333 / others 1.0 | Required in both modes and keyed by §3's stable seven classes. Historical profiles pin unit weights where needed. |
| `diffusion.process` | `uniform` | `uniform` is the production default; `absorbing` is the coherent masked-diffusion ablation (§2–3, §9). |
| `diffusion.schedule.t_one_fraction` | 0.05 | Exact-terminal training fraction, mixed over the configurable continuous time distribution (§3). |
| default/full-V3 `model.*` (d_model / layers / heads / ffn) | 384 / 12 / 6 / 1536 | 29,318,720 parameters with head dimension 64. Legacy profiles may explicitly retain their historical shape. |
| `diffusion.schedule` | linear corruption rate, continuous t ~ Beta(2,1) | Uniform-state mode uses retain-with-`1-t`, otherwise uniform replacement; absorbing ablation uses `[MASK]` replacement and inverse-`t` masked loss. |
| full-V3 `pipeline.batch_size / train.accumulation_steps / train.max_cuda_reserved_gb` | 6 / 7 / 6.5 | Forty-two windows and about 275k valid tokens per optimizer update on the measured corpus slice; step planning uses `ceil(batches/7)` so scheduler/EMA horizons remain optimizer-step based. The reclaim-first ceiling leaves room for Windows/display use on the 8 GiB RTX 3070. |
| `train.*` (lr / betas / weight_decay / lr_floor / grad_clip / precision) | 3e-4 / (0.9,0.95) / 0.1 / 0.01×peak / 1.0 / bf16 | Optimizer and precision settings remain config-owned; full-V3 accumulation is specified separately above. |
| `train.lr_schedule` | `wsd` | Config-selectable `wsd`, `cosine`, or `linear`. WSD uses 500 fixed linear-warmup optimizer steps, holds `3e-4` through the remaining middle phase, and linearly decays over the final 20% to `3e-6`. Historical profiles pin their prior schedule explicitly. |
| `train.ema_decay / ema_horizon_ratio / confidence_loss_weight / val_interval` | 0.9999 (ceiling) / 0.1 / 0.0 / periodic | EMA on by default, with the decay derived from the run's step horizon and capped by `ema_decay` (§3); confidence sharpening is an opt-in ablation (§3). |
| full-V3 `train.epochs / early_stopping_*` | 50 / 10-epoch patience / 0.1% relative threshold | Best checkpoint replacement uses any strict dev-loss improvement; the threshold applies only to patience resets. |
| full-V3 checkpoint cadence | resume every 100 optimizer steps; best each improved epoch; durable every 5 epochs | Stored under separate `resume/`, `best/`, and `durable/` subdirectories with epoch-numbered best/durable filenames. |
| `model.qk_norm / model.self_conditioning / train.self_cond_prob` | true / true / 0.5 | QK-norm and expected-embedding GeGLU self-conditioning (§2–3). |
| `model.rope_theta / model.rope_scaling.*` | 500000 / llama3, factor 8, low/high 1/4, original context 8192 | Llama 3.1 frequency-scaled RoPE; all constants are config-owned and sequence length is not hard-capped in the rotary implementation |
| `sampler.max_steps / temperature / entropy_bound` | 64 / 0.8→0.4 / 0.1 | Hard ceiling with no minimum; temperature exponent `1.0` gives the linear schedule (§9). |
| `sampler.adaptive_stop / entropy_threshold` | true / 0.005 | Both the confidence and the fixed-point condition are required (§9). `stability_steps` is retired: a fixed-point certificate already compares a prediction against the state that produced it, so a consecutive-pass count is unsatisfiable whenever the entropy budget leaves a nonempty renoised tail. |

**Model-sizing note.** All `model.*` values are config; size changes require no code changes. V3 is the current 29.3M full-corpus shape. Historical ~11M profiles remain reproducible through explicit overrides.

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
- Semi-autoregressive or block-autoregressive generation
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

### 14a. Discouraged — REQUIRES EXPLICIT OWNER CONFIRMATION BEFORE ANY CODE IS WRITTEN

These are not banned, but the project's clear preference is to stay away from them. They were moved out of §14 because the ablation work in prompt 009 showed the hard ban was overbroad: some of these are legitimate to *measure* even where they are undesirable to *adopt*.

The rules for anything in this list:

1. **Think first, and say the thinking out loud.** Justify why the item is warranted here, and what cheaper alternative you rejected. "It would be faster" is not sufficient on its own.
2. **Ask the owner and get an explicit yes before writing code.** This is a hard gate. Do not begin an implementation, and do not implement it "provisionally" pending review.
3. **It ships as an ablation toggle, defaulting to `false` — never as the default path.** With the toggle off, the code must be behaviorally identical to the baseline. Promotion to default is a separate decision the owner makes on evidence, not something an implementing agent may do.
4. **It must be measured before it is trusted.** A toggle that has not been run against its baseline arm is not evidence of anything.

The items:

- **Prompt/input KV caching.** Allowable and worth measuring, since the input region is static across denoising steps and recomputing it is pure waste. Implemented as the `model.frozen_input_kv` toggle and **PROMOTED to `true` by default on 2026-08-09 after measurement** (§14b has the evidence and the consequences). Note the real semantic cost, which the owner accepted rather than eliminated: caching the input's K/V means the input no longer attends to the canvas, which makes attention one-directional across the seam. That is a genuine architecture change, not just an optimization. This promotion is specific to this item — the two entries below remain undesirable and ungranted.
- **A separate prompt/input encoder.** Undesirable. The conditioning model is clamping within one bidirectional stack (§4), and splitting the input into its own encoder walks toward the encoder-decoder split this project deliberately rejected. Do not propose it as a performance fix.
- **Encoder-decoder architecture, including cross-attention conditioning.** Undesirable, for the same reason. If some future need seems to require it, that is a signal to re-examine the conditioning design with the owner, not to build it.

### 14b. Model toggles — two active ablations, one promoted default

Three `model:` toggles were introduced to run the prompt-009 representational ablation. **One has since been promoted to a default; the other two remain experiments and default to `false`.**

| Toggle | Default | What it changes | §14a gated |
|---|---|---|---|
| `model.frozen_input_kv` | **`true` — PROMOTED, see below** | Two-pass forward; input K/V cached per layer and reused across denoising steps | Yes — input KV caching |
| `model.segment_embeddings` | `false` | Learned `nn.Embedding(2, d_model)` marking input vs canvas | No |
| `model.per_segment_positions` | `false` | RoPE position ids computed per segment instead of over the concatenation | No |

`segment_embeddings` and `per_segment_positions` are NOT on a path to becoming defaults; enabling one is an experiment, and promotion is the owner's call on measured evidence.

**`frozen_input_kv` promotion — owner decision, 2026-08-09, SETTLED.** Measured on ablation arm 01 (`configs/ablation_01_frozen_input_kv.yaml`, 100 epochs against the all-false baseline): no statistically meaningful loss difference, materially faster inference. The input region is static across every denoising step, so recomputing its K/V per step was pure waste. The accepted cost is the §14a one — the input no longer attends to the canvas, making attention one-directional across the seam. This is the promotion path §14a rule 3 describes working as intended: shipped as a toggle, measured against its baseline, then promoted by the owner. It does NOT loosen the rule for anything else in §14a, and it does not make the other two toggles promotable by an agent.

Three consequences of the promotion:

- `config/default.yaml` sets `frozen_input_kv: true`, so every profile inherits it unless it explicitly opts out.
- **`toggle_fingerprint` is deliberately NOT rebased.** A default-derived model stamps `architecture_identity` as `"dense-multinomial-SC2-v2+frozen_input_kv"`. Pre-promotion all-off checkpoints therefore fail closed on load instead of silently entering a two-pass model, and ablation arms 01/05 keep the exact identity they were trained under. The unsuffixed `"dense-multinomial-SC2-v2"` still means all three off, and `configs/ablation_00_baseline.yaml` pins all three false so the completed sweep's baseline arm stays reproducible.
- Vocabulary-v1 checkpoints remain deliberately incompatible.

**Known defect in the promoted path, fixed 2026-08-09 in the same change.** The frozen path supplied no explicit RoPE position ids to its second pass, so `RotaryEmbedding` fell back to `arange(canvas_len)` and the canvas restarted at position 0 while the cached input keys carried `0..input_len-1`. That (a) applied per-segment canvas positions unconditionally, conflating the toggle with `per_segment_positions`, and (b) made the canvas-to-input relative offset vary with the batch's left padding, so a window's logits depended on which other windows shared its batch. `BidirectionalTransformer.forward` now derives absolute positions over the concatenation for both passes. **Ablation arms 01 and 05 were trained with the defect present**, so their recorded curves are not a clean measurement of the toggle as it now behaves.

See `diagnostics/009-rare-class-position-blindness.md` for the motivating failure and what each toggle is meant to test.

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
