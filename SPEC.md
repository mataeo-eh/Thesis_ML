# SPEC.md — SC2 Strategy Prediction via Masked Discrete Diffusion

Single source of truth for all architecture decisions in this repository.

**Agent instructions:** Read this document in full before beginning any task. Decisions marked SETTLED are final — do not revisit, "improve," extend, or suggest alternatives to them. §11 parameters are PROVISIONAL config defaults — implement them as config fields, never hardcode them. §12 items are OPEN — do not resolve them and do not implement anything for them. §14 is a hard ban list. On any conflict between this document and CLAUDE.md, this document wins.

---

## 1. Project summary

Masters thesis system: predict opponent strategy in StarCraft II from partially observed game state, via self-supervised pretraining on replay data.

Core claim: strategy structure emerges from pretraining alone — no label supervision during representation learning.

Framing: joint bidirectional denoising plausibly suits strategy prediction because discriminative signal lives in joint correlations among hidden and future tokens. This is a suitability investigation, not a superiority claim over causal transformers. No autoregressive head-to-head comparison exists in scope.

Data extraction is complete and lives in a separate repository (`SC2-gamestate-extractor`, pysc2 + s2protocol engine-simulation parsing). This repository consumes its output. Replays sourced from aiarena.net; dataset at Kaggle `mataeoanderson/sc2-replay-data`.

## 2. Model family — SETTLED

- Masked (absorbing-state) discrete diffusion, MDLM/LLaDA family. Methodology citations: SEDD, MDLM, LLaDA.
- Backbone: a dense bidirectional transformer in the **LLaDA / LLaMA lineage** — RMSNorm (pre-norm), SwiGLU FFN, **Llama 3.1-style frequency-scaled RoPE** for sequence position, **vanilla multi-head attention (MHA), NOT grouped-query attention**, with **QK-norm** (per-head RMSNorm on queries/keys before RoPE, for attention stability; config-gated, default on). Single stack. The RoPE base and Llama 3 scaling factors are config fields so pretraining can use shorter sequences while inference can evaluate much longer sequences without learned position tables or an architecture change. This is the Llama 3 scaled-RoPE variant, not the separately defined YaRN algorithm. Rationale for MHA over GQA: GQA exists to shrink the KV cache during autoregressive decoding; full-canvas diffusion does not decode autoregressively, so (following LLaDA) MHA is used and FFN width is the parameter-budget knob. QK-norm note: it can flatten attention and slightly weaken precise retrieval, which matters for the input→canvas copy pathway — watch the copy-class loss when toggling. Attention uses FlashAttention kernels via PyTorch SDPA (no causal mask — full bidirectional with a padding mask only).
- Reference architecture: LLaDA (dense, from-scratch masked diffusion LM) is the closest published anchor and is downscaled to fit this project. DiffusionGemma is NOT a structural template — it is MoE, block/semi-autoregressive, and encoder-decoder, all of which this project cuts; it serves only as a family existence proof and a source of sampler defaults (§9).
- Each training example is one flat sequence: `[input region][canvas region]`. Full bidirectional attention over the entire sequence. The input region is clamped — never noised, never receives loss. The canvas region is noised; loss is computed on canvas positions only.
- Conditioning is clamping. There is no encoder-decoder split, no cross-attention conditioning, no separate prompt encoder.
- No SSM layers in v1 (see §12 for the shelved fallback).

## 3. Training objective — SETTLED

One unified SSL denoising task family. The two stages differ in whether a clamped input region exists at all.

**Pre-training input: NONE.** Pre-training is the published-MDLM-style pure reconstruction objective: the model sequence is 100% output canvas. There is no clamped input region, no self sequence, no fogged enemy sequence, and NO fog paradigm of any kind in pre-training. The `config.fog` section must be ABSENT from a pre-training config (`data.debut_mode=false`) — config validation rejects its presence rather than tolerating a dead knob.

**Fine-tuning input (clamped; debut fine-tuning only, §8):**
- Interleaved per timestep: each timestep contributes `[self records][enemy records][ONE DELIMITER]` (exact grammar in §6).
- Fog mechanism: **entity omission**. Fogged tokens are removed from the input entirely. No placeholder tokens, no mask tokens, no count signal of any kind for omitted tokens. Fog applies uniformly to enemy content tokens of EVERY token kind — entities AND cumulative upgrades (no entity-only special case).
- Fog rate: a parameter of the corruption distribution, sampled per example from the config distribution (`config.fog`, REQUIRED when `data.debut_mode=true`). Zero fog degenerates to clean-past-predict-future. Fog applies to the enemy sequence only.

**Target canvas (noised, receives loss):**
- Leading outcome token: the canvas begins at position 0 with the `[WIN]`/`[LOSS]` outcome token for the perspective player, denoised LAST (see §9's `outcome_last` constraint). This token is part of pre-training itself — folded in so the model's frozen embedding prior includes it, rather than being introduced only at fine-tune time when the reduced learning rate makes a brand-new token hard to learn. Outcome/debut fine-tuning (§8) reuses the same leading-outcome-token layout.
- The enemy sequence only: full reconstruction of the enemy past/present (both observed and fogged portions) plus the enemy future continuation, regenerated jointly, following the outcome token.
- Canvas corruption during training: uniform i.i.d. masking with `[MASK]` at a sampled global corruption level *t* (MDLM-style). Per-timestep-varying corruption is an input-side data property (fog, fine-tuning only) and is never applied to the output canvas.
- **t=1.0 oversampling (`diffusion.mask_schedule.t_one_fraction`, default 0.1):** when *t* is sampled (the real training path, not explicit-*t* eval/validation calls), each example independently draws a Bernoulli(`t_one_fraction`) coin; winners get *t* forced to EXACTLY 1.0 (fully masked canvas — the inference condition), while the rest keep the uniform draw over the schedule range. This is an expected per-epoch fraction (per-example Bernoulli), not an exact per-batch quota. Applies to both training modes.
- **Per-epoch generator reseeding:** the training loop reseeds its corruption/self-conditioning generator to `base_seed + epoch_index` at every epoch boundary. Each epoch's masking stream is therefore a deterministic function of (seed, epoch), which keeps a resumed run's corruption draws aligned with the stream an uninterrupted run would have produced.
- Canvas position 0 (the `[WIN]`/`[LOSS]` outcome token) receives NO training exemption: it is masked i.i.d. at rate *t* and weighted in the loss like any other canvas position. `outcome_last` (§9) is a sampler-only inference constraint.
- At inference, the canvas initializes as all `[MASK]`.
- `[MASK]` is the noise state. `[PAD]` is a content token in the vocabulary that surplus canvas positions denoise into.

**Loss:**
- Position-wise cross-entropy against canonically ordered targets (§5), canvas positions only.
- Per-token-class loss logging is mandatory from the first training run. The class taxonomy is now MODE-DEPENDENT:
  - **Pre-training (collapsed, 5 classes — `PRETRAIN_CLASS_ID_TO_NAME`):** content (every enemy content token — with no input and no fog there is no observed/fogged/future split to make), `[DELIMITER]`, `[END]`, `[PAD]`, and win-loss (the leading `[WIN]`/`[LOSS]` outcome token). Class ids 1 and 2 (the old fogged/future ids) are UNUSED in this mode and are NEVER renumbered — the map is sparse (ids 0/3/4/5/6), so id-indexed buffers must be sized `max(id) + 1`, not `len(map)`.
  - **Debut fine-tuning (7 classes — `DEBUT_CLASS_ID_TO_NAME`):** visible-debut, fogged-debut, future-debut, delimiter, end, pad, win-loss.
- **Pre-training loss weighting is fully uniform (published MDLM style):** every class weight is 1.0 except `[PAD]`, which is 0.0 so padding positions never contribute. Pre-training NEVER reads `config.loss.class_loss_weights` — that section must be ABSENT from a pre-training config (validation rejects it).
- **Fine-tuning per-class loss weights** remain a config knob (`loss.class_loss_weights`, REQUIRED when `data.debut_mode=true`), default 1.0 (PROVISIONAL).
- **Loss-breakdown metrics:** BOTH modes additionally log masked-CE broken down by the example's sampled *t*-bucket (`t_eq_1`, `[0.7,1.0)`, `[0.5,0.7)`, `[0.3,0.5)`, `[0.0,0.3)` — contiguous and exhaustive over [0,1]) and by player perspective (`p1`/`p2`). The future-distance decomposition (loss bucketed by prediction distance) is FINE-TUNING-ONLY: pre-training has no future class and must emit no future-distance keys or columns at all.
- **Auxiliary confidence loss (config-weighted):** an auxiliary term that sharpens per-position predictive confidence so confidence-based unmasking commits reliably and aggressively (fewer denoising steps → faster inference). Purpose per LLaDA2.0 (arXiv:2512.15745); implement by verifying that source's formulation, preferring a logits-derived form over a separate head. Config weight `confidence_loss_weight` (default small; 0.0 disables). Distinct from the §14-banned outcome classification head.
- **EMA (SOTA diffusion practice):** maintain an exponential-moving-average copy of the weights during training (decay ~0.9999); use EMA weights for validation, the final checkpoint, sampling, and evaluation. EMA is standard for diffusion training and one of the practices distinguishing it from AR pretraining.
- **Self-conditioning (config-gated, default on):** each denoising step may condition on the model's own previous clean-state canvas prediction (the prior step's softmax distribution), projected and added to canvas embeddings (canvas-only; input untouched). Training uses a two-pass procedure (prob `self_cond_prob`, default 0.5: a no-grad estimate pass, then a conditioned grad pass on which the loss is computed; null tensor otherwise). Inference reuses the previous step's prediction and adds NO extra forward passes. Train and inference interfaces must be identical. Per SCMDM (arXiv:2604.26985). This is prediction REUSE, not committed-token revision (remasking is deferred, §12).

There is no copy mechanism of any kind. Input-to-output copying is a learned behavior produced by the loss.

## 4. Tokenization and vocabulary — SETTLED

- Raw atomic entity-level tokens only. One token per entity instance per timestep snapshot — unit counts emerge from token repetition. No BPE, no merges, no compound tokens, no learned tokenizer.
- Single shared vocabulary for input and output. Content tokens are **raw entity-type tokens — entirely location-agnostic**. The token identity carries NO spatial information of any kind. Position is input-only and lives entirely in the contextual encodings (§6): the exact (X,Y) coordinate from the extractor parquet is added to the input token's embedding. The output canvas is location-agnostic — it predicts entity-type presence, timing, and counts (by repetition), never position.
- Special tokens: `[MASK]` (absorbing noise state), `[PAD]`, `[END]`, `[DELIMITER]`, and reserved outcome tokens `[WIN]` / `[LOSS]` (used only in outcome fine-tuning, §8, but reserved in the vocabulary from day one so embeddings exist).
- Outputs never contain raw coordinates, frame numbers, or absolute times. The vocabulary contains no tokens for them.
- Concrete vocabulary contents are derived from the extractor schema (§13).

## 5. Serialization order — SETTLED

- Within a timestep, entities serialize in canonical order: primary sort by entity type ID; within-type tiebreak by unit ID (tiebreak key PROVISIONAL — the binding requirement is stable and deterministic).
- The same canonical ordering applies to input serialization and target construction.
- Targets are canonically ordered and the model learns to emit canonical order via position-wise CE. No permutation-invariant losses, no Hungarian matching, no set losses.

## 6. Input representation — SETTLED

Two distinct kinds of "position" exist in this system and must never be conflated:
- **Sequence position** — a token's index in the flat sequence. Encoded only with **Llama 3.1-style frequency-scaled RoPE**, applied to queries/keys in attention. Chosen specifically so that entity-counts-per-timestep and timestep-counts not seen during training do not break the model at inference. This is the only numerical sequence-position encoding the model receives: no learned absolute position table, absolute game clock, frame number, `game_loop`, or timestamp-derived feature.
- **Map position** — where a unit sits on the game map: the exact (X,Y) coordinate from the extractor parquet. A *feature* of the entity, encoded as an additive **input-only** contextual encoding (below), never part of any token identity. Unrelated to RoPE.

**Pre-training has NO input region.** The model sequence is exactly the output canvas: the backbone runs over canvas embeddings alone, the attention mask has zero input columns, there is no separator/BOS/segment token, and RoPE position 0 is the FIRST CANVAS TOKEN. Everything below about input token embedding applies to fine-tuning only.

**Fine-tuning input grammar — interleaved per timestep.** Walking the window's timesteps in order, each timestep contributes `[self records][enemy records][ONE DELIMITER]`: all self records first, then the (fog-filtered) enemy records, closed by exactly ONE `[DELIMITER]`. The total input delimiter count therefore equals the window's timestep count. (This replaces the earlier `[all self timesteps][all enemy timesteps]` layout with one delimiter per player per timestep.)

Input embedding pipeline — lives in the MODEL, not the tokenizer (these are learned parameters trained by backprop), applied to input tokens (fine-tuning only):
1. Token embedding lookup (learned).
2. Additive contextual encodings, **input-only**: exact (X,Y) map position and unit stats. Canvas tokens never receive these. Map position uses extrapolation-friendly Fourier/sinusoidal features rather than a learned lookup over fixed bins. Absolute game time, frame number, `game_loop`, and timestamp-derived values are prohibited from this feature path.
3. Team flag: a learned embedding component distinguishing self tokens from enemy tokens.
4. RoPE applied in attention for sequence position (see above; in pre-training RoPE runs over the canvas-only sequence).
5. Timestep boundaries: `[DELIMITER]` tokens, present in both input and canvas.

The tokenizer (§4–5) emits one sequence of token records carrying token identity and source metadata. A model-facing allowlisted feature structure carries only map position, unit stats, and allegiance into the embedding stack. Absolute clock metadata may remain available outside the model for dataset ordering and post-sampling evaluation, but must never be copied into model inputs, embeddings, attention inputs, or targets. The tokenizer never computes embeddings or positional encodings.

Windows may begin mid-game; the model must infer game phase from observed game state and sequence structure rather than an absolute clock.

Pretraining windows are greedy contiguous runs of whole timesteps from one replay.
A timestep is added only while the zero-fog serialized input remains within its
budget and the full in-window enemy reconstruction remains within
`canvas_recon_fraction × canvas_budget_tokens`. (The input budget still governs
pre-training window TILING even though no input region is served in
pre-training; the served sequence is canvas-only per the paragraph above.)
Successive default windows tile each replay without overlap. One window is one
batch sequence; sequence packing and cross-document masks are not used.

v1 uses delimiters only for timestep structure; a separate timestep-membership encoding is OPEN (§12) — do not implement one.

## 7. Output canvas semantics — SETTLED

- Flat token canvas with one fixed overall budget (config). Model-placed `[DELIMITER]`s partition it into contiguous timesteps. No per-timestep slot budgets.
- Each timestep's tokens are followed by one `[DELIMITER]`. After the final timestep of a replay: `[END]`, then semantic `[PAD]` targets. Collation may add further batch-shape `[PAD]` values, which are excluded from attention and loss.
- Absolute timing of canvas timesteps is recovered externally by arithmetic: clock of last input frame + fixed sampling interval × timestep index. The model never emits time.
- **Target truncation rule — whole timesteps only:** reconstruction contains exactly the window timesteps and future continuation admits a timestep only when all enemy tokens plus its `[DELIMITER]` fit. No partial timestep is ever emitted. If the game ends within budget, append `[END]` then `[PAD]` to budget. Otherwise stop at the last complete boundary and append `[PAD]` directly to budget.
- Grammar invariant (enforced in tests): a valid canvas is a leading `[WIN]`/`[LOSS]` outcome token, then `(timestep-tokens [DELIMITER])+`, followed by either `[END] [PAD]*` or `[PAD]*`. (The leading outcome token is present in both pre-training and debut fine-tuning; see §3 and §8.)
- In pre-training the canvas is the ENTIRE model sequence (§6: no input region), and its class labeling is collapsed: every content token is the single "content" class — the observed/fogged/future split exists only in debut fine-tuning (§3).

## 8. Outcome/debut training — SETTLED mechanism

- Superseded framing (was: "the outcome token exists only in fine-tuning"). The `[WIN]`/`[LOSS]` outcome token is now emitted in BOTH pre-training (§3) and debut fine-tuning, at the same leading canvas position. `debut_mode` selects the canvas BODY that follows the outcome token — full enemy reconstruction + future roll-out (pre-training) vs sparse first-appearance debut events (fine-tuning) — AND whether an input region exists at all: pre-training serves no input (§3, §6), while debut fine-tuning serves the interleaved fogged input.
- `debut_mode` also gates the fine-tuning-only config sections and metrics: `config.fog` and `config.loss.class_loss_weights` are REQUIRED when `debut_mode=true` and REJECTED when false (§3, §11), and the future-distance loss decomposition plus input-timestep/fog telemetry exist only in this mode.
- Debut detection is UNIFIED across token kinds: an event debuts when its per-timestep count exceeds the running maximum seen so far in the window's scan (count increase). For cumulative upgrade tokens — whose per-timestep count is always 0 or 1 — this fires exactly once, at first appearance, reproducing the previous upgrade special case, which is deleted.
- Task: game outcome prediction. No classification head exists anywhere in this project.
- Mechanism: the outcome token (`[WIN]`/`[LOSS]`) occupies the leading canvas position; the model denoises the outcome token and the normal continuation jointly in one pass. Pre-training uses the identical outcome-token placement and `outcome_last` denoising order.
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

- Iterative denoising with confidence-based unmasking: each step commits the highest-confidence positions, re-masks nothing committed, and repeats until the canvas is fully committed or early stopping triggers.
- Starting hyperparameters (PROVISIONAL, anchored on DiffusionGemma's published serving defaults): max ~48 denoising steps; adaptive early stopping; falling temperature schedule; entropy-bound selection of tokens to commit per step.
- Remasking-style samplers (inference-time revision of committed tokens) are a permitted future extension. Not v1. Do not implement.

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
| `fog_rate_distribution` | uniform(0.0, 0.8) | Sampled per example. FINE-TUNING-ONLY: the `fog` section is required when `data.debut_mode=true` and rejected when false (§3) |
| `within_type_tiebreak` | unit ID | §5 |
| `class_loss_weights` | all 1.0 | Keyed by §3's debut classes. FINE-TUNING-ONLY: required when `data.debut_mode=true`, rejected when false; pre-training uses fixed uniform weighting with `[PAD]`=0 (§3) |
| `diffusion.mask_schedule.t_one_fraction` | 0.1 | Expected per-epoch fraction of examples oversampled to exactly t=1.0 (per-example Bernoulli); 0.0 disables (§3). Stated explicitly in every profile YAML |
| local `model.*` (d_model / layers / heads / ffn) | 256 / 10 / 4 / 1024 | ~10.7M proof-of-life shape with head dimension 64. Cloud scale remains config-only. |
| `mask_schedule` | linear, t ~ U(0,1) | MDLM/LLaDA default; loss reweighted by 1/t over masked positions |
| `train.*` (lr / betas / weight_decay / warmup / lr_floor / grad_clip / accum / precision) | 3e-4 / (0.9,0.95) / 0.1 / 2000 / 0.1×peak / 1.0 / as-needed / bf16 | Cosine decay to lr_floor; accumulation derived from a target effective batch size (§005) |
| `train.ema_decay / confidence_loss_weight / val_interval` | 0.9999 / 0.1 / periodic | EMA on by default; confidence loss off-able via 0.0 (§3, §005) |
| `train.epochs / early_stopping_*` | profile-owned / disabled by default | Epoch CSV metrics are always available; local overfit uses 0.1% relative improvement with five-epoch patience and a 200-epoch cap. |
| `model.qk_norm / model.self_conditioning / train.self_cond_prob` | true / true / 0.5 | QK-norm and self-conditioning (§2, §3, §009); both gated, OFF reproduces pre-009 behavior |
| `model.rope_theta / model.rope_scaling.*` | 500000 / llama3, factor 8, low/high 1/4, original context 8192 | Llama 3.1 frequency-scaled RoPE; all constants are config-owned and sequence length is not hard-capped in the rotary implementation |
| `sampler.max_steps / temperature / entropy_bound` | 48 / 0.8→0.4 / 0.1 | §9 |

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
- `./research/` research outputs; `./plans/` plans; `./diagnostics/` diagnostics; `./tests/fixtures/` owner-provided sample extractor outputs.
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
- The master training script and data-acquisition script are built in prompt 008; these conventions apply to every prompt that touches paths, storage, data, or checkpoints.
