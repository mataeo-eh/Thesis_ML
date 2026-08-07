# Thesis_ML Current Model Architecture

## Scope and authority

This document is the exact implementation-and-configuration companion to `SPEC.md`. `SPEC.md` remains the normative design authority; this file records the current learnable machinery, model-facing plumbing, tensor sizes, parameter counts, training objective, and inference procedure implemented by the repository.

Unless a section explicitly says otherwise, exact run-profile values refer to the merged `configs/local_full.yaml` profile. Values are derived from the loaded configuration, the live vocabulary, the configured feature-statistics artifact, and direct `SC2StrategyDiffusionModel` construction—not from an older checkpoint or console log.

## Architecture at a glance

- Model family: dense, bidirectional uniform-state discrete-diffusion transformer.
- Trainable parameters: **11,042,880**.
- Model width: **256**.
- Transformer blocks: **10**.
- Attention: **4** heads × **64** dimensions per head.
- Dense GeGLU width: **1,024**.
- Vocabulary tensor width: **383** IDs.
- Local-full batch size: **9**.
- Input budget: at most **4,096** served tokens per row.
- Canvas size: exactly **4,096** target tokens per row in the production builders.
- Maximum concatenated model sequence: **8,192** positions.
- Output: one untied, bias-free **256 → 383** token head at every concatenated position; training slices and scores only the canvas region.
- Position encoding: parameter-free Llama 3.1-style frequency-scaled RoPE.
- No encoder-decoder split, cross-attention, causal mask, learned positional embedding, region/segment embedding, copy head, classification head, or explicit diffusion-time embedding.

## End-to-end data flow

![Thesis_ML model architecture data-flow diagram](MODEL_ARCHITECTURE_DIAGRAM.png)

The directly viewable artifacts are [`MODEL_ARCHITECTURE_DIAGRAM.png`](MODEL_ARCHITECTURE_DIAGRAM.png) and [`MODEL_ARCHITECTURE_DIAGRAM.svg`](MODEL_ARCHITECTURE_DIAGRAM.svg). The single canonical graph definition is [`MODEL_ARCHITECTURE_DIAGRAM.mmd`](MODEL_ARCHITECTURE_DIAGRAM.mmd); `render_diagram.py` reads it and regenerates both images without requiring Mermaid CLI or a browser.

## Exact local-full configuration

| Property | Current value |
|---|---:|
| `model.d_model` | 256 |
| `model.layers` | 10 |
| `model.heads` | 4 |
| Derived head dimension | 64 |
| `model.ffn` | 1,024 |
| Model dropout used by the production constructor | 0.0 |
| `model.qk_norm` | true |
| `model.self_conditioning` | true |
| `model.gradient_checkpointing` | true |
| `model.rope_theta` | 500,000 |
| RoPE type | `llama3` |
| RoPE factor | 8.0 |
| RoPE low/high frequency factors | 1.0 / 4.0 |
| RoPE original context | 8,192 |
| `data.input_budget_tokens` | 4,096 |
| `data.canvas_budget_tokens` | 4,096 |
| `data.canvas_recon_fraction` | 0.5 |
| Derived reconstruction limit | 2,048 tokens |
| `pipeline.batch_size` | 9 |
| `diffusion.process` | `uniform` |
| Diffusion schedule | linear, `t ~ Uniform(0,1)` |
| Exact-`t=1` oversampling | 0.0 |
| Peak learning rate | 3e-4 |
| AdamW betas | 0.9, 0.95 |
| AdamW epsilon | 1e-8 |
| Weight decay | 0.1 |
| Warmup | 200 optimizer steps |
| LR floor | 0.1 × peak |
| Gradient clipping | 1.0 |
| Accumulation steps | 1 |
| Effective-token accumulation target | disabled (`0`) |
| Epochs | 8 |
| EMA decay | 0.9999 |
| Training self-conditioning probability | 0.5 per row |
| Compute precision | BF16 autocast |
| Stored parameter dtype | FP32 |
| Sampler maximum passes | 64 |
| Sampler temperature | linear 0.8 → 0.4 |
| Entropy bound | 0.1 |
| Adaptive entropy threshold | 0.005 |
| Required stable passes | 2 |

`train.max_steps: 0` is resolved by the pipeline to `len(train_loader) × train.epochs` before the scheduler is constructed. The learning-rate multiplier rises linearly for 200 steps and then follows cosine decay to 10% of the peak.

## Vocabulary and learned state space

### Token identities

| ID range | Meaning | Count |
|---|---|---:|
| 0 | `[MASK]` | 1 |
| 1 | `[PAD]` | 1 |
| 2 | `[END]` | 1 |
| 3 | `[DELIMITER]` | 1 |
| 4 | `[WIN]` | 1 |
| 5 | `[LOSS]` | 1 |
| 6–99 | unnamed/reserved tensor IDs | 94 |
| 100–382 | contiguous content-token IDs | 283 |

The vocabulary width is the largest assigned token ID plus one:

\[
V = 382 + 1 = 383.
\]

The input embedding and output head therefore each allocate 383 rows, although only 289 IDs are named special or content tokens. Current uniform corruption samples integer IDs from `[1, 383)`, and inference masks only ID 0 before softmax. Consequently, IDs 6–99 are part of the current noising and inference state space and own trainable embedding/output rows. Ground-truth targets do not use them.

### Shared token embedding

\[
E \in \mathbb{R}^{383 \times 256}
\]

The same table embeds clamped input IDs, noised canvas IDs, and the expected token distribution used by self-conditioning. It is not tied to the output head.

## Clamped input and feature conditioning

### Input token sequence

For each one-second replay timestep, the served input grammar is:

```text
[self content records]
[fog-filtered enemy content records]
[one DELIMITER]
```

One fog rate is sampled per served example from `Uniform(0.0, 0.8)`. Each enemy content token is independently omitted with that probability. Fog does not insert `[MASK]`; self records and the delimiter remain. Persisted artifacts remain clean.

The collator left-pads each row to the longest served input in its batch:

| Tensor | Shape | Dtype |
|---|---|---|
| `input_token_ids` | `[B, Li]`, `Li ≤ 4096` | int64 |
| `input_attention_mask` | `[B, Li]` | bool |
| `input_lengths` | `[B]` | int64 |

Because fog is applied after clean-window selection, the served `Li` can be below the clean 4,096-token window budget.

### Model-facing feature tensors

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `continuous_values` | `[B, Li, 25]` | float32 | approved scalar features |
| `continuous_validity` | `[B, Li, 25]` | bool | valid zero versus missing |
| `categorical_values` | `[B, Li, 309]` | float32 | cloak and buff encodings |
| `allegiance_values` | `[B, Li, 1]` | float32 | self `+1`, enemy `-1`, structural `0` |
| `feature_mask` | `[B, Li]` | bool | true only for content records |

The 25 continuous channels are:

```text
map_x, map_y, health, energy, shields,
facing_sin, facing_cos, radius, build_progress,
weapon_cooldown, attack_upgrade_level, armor_upgrade_level,
shield_upgrade_level, cargo_space_taken, cargo_space_max,
order_count, is_flying, is_burrowed, is_hallucination,
is_active, is_powered, ideal_harvesters,
buff_duration_remain, buff_duration_max, detect_range
```

The categorical width is:

- 5 cloak-state one-hot channels;
- 1 buff-validity channel;
- 303 buff-ID multi-hot channels for raw IDs 0–302.

The learnable branch input width is therefore:

\[
25_{values} + 25_{validity} + 309_{categorical} + 1_{allegiance} = 360.
\]

Valid continuous features are standardized using 25 non-trainable train-split means and standard deviations. Invalid standardized values are replaced by zero. Absolute time, frame number, `game_loop`, and timestamp-derived values do not enter any model-facing feature tensor.

### Feature MLP and joint mixer

The feature network is:

\[
360 \xrightarrow{Linear+ReLU} 32 \xrightarrow{Linear+ReLU} 32.
\]

The result is concatenated with the 256-dimensional token embedding and mixed by:

\[
(256 + 32) = 288 \xrightarrow{Linear+GELU} 256 \xrightarrow{Linear} 256.
\]

The final joint-mixer projection is reset to exactly zero after generic model initialization. Initially, the input path therefore equals token lookup exactly. The learned residual is multiplied by `feature_mask`, so delimiters and batch padding receive no feature residual.

## Canvas construction and corruption

### Clean target canvas

The production pretraining builder emits exactly 4,096 tokens:

```text
[perspective-relative WIN or LOSS]
(enemy timestep content [DELIMITER])*
[END] [PAD]*                  # when game end fits
or
[PAD]*                        # when horizon is boundary-truncated
```

Position 0 is always the outcome token. The body first reconstructs the enemy sequence overlapping the input window, then continues into enemy-future timesteps. In-window enemy reconstruction is bounded by `4096 × 0.5 = 2048` tokens; the remaining budget can carry the outcome, future continuation, delimiters, end token, and semantic padding.

| Tensor | Shape | Dtype |
|---|---|---|
| `target_canvas` | `[B, 4096]` | int64 |
| `canvas_attention_mask` | `[B, 4096]` | bool |
| `class_labels` | `[B, 4096]` | int64 |
| `canvas_loss_mask` | `[B, 4096]` | bool |
| `canvas_prediction_distances` | `[B, 4096]` | int64 |

Semantic `[PAD]` is a real attended and scored target. Only extra padding introduced to batch differently sized examples is excluded. The current production builders already fill their canvas budget, so every produced canvas row is 4,096 positions.

### Uniform corruption

For every batch row, training samples one scalar:

\[
t_b \sim Uniform(0,1).
\]

Every canvas position independently enters the corruption branch with probability `t_b`. Selected positions are replaced by an independent uniform integer in `[1, 383)`; the replacement can equal the clean target. The clamped input is returned unchanged.

The model does not receive `t_b` as a scalar, embedding, token, or feature. It must infer the current noise level from the noised canvas itself.

The absorbing scientific ablation replaces selected positions with `[MASK]` and changes the scored mask/position weighting; it does not change the model parameterization.

## Expected-embedding self-conditioning

Given prior canvas logits

\[
Z \in \mathbb{R}^{B \times 4096 \times 383},
\]

the stopped expected token embedding is:

\[
P = softmax(Z), \qquad S = stopgrad(P)\;stopgrad(E),
\]

with

\[
S \in \mathbb{R}^{B \times 4096 \times 256}.
\]

The canvas conditioning branch is:

```text
S
→ learned RMSNorm(256)
→ GeGLU gate/up/down, all width 256
→ add to ordinary canvas token embedding
→ scale-less RMSNorm
```

When no signal is supplied, the branch uses zeros, but the ordinary canvas embedding still passes through the scale-less post-add RMSNorm.

Training performs a no-gradient estimate pass for the entire batch, samples an independent Bernoulli mask with probability 0.5 per row, zeros the conditioning signal for unselected rows, and then performs the loss-bearing pass. Validation through the explicit evaluation-model path does not sample this training branch.

Inference reuses the preceding denoising pass's temperature-shaped probability distribution, so normal self-conditioned inference does not add an extra forward pass.

## Concatenated transformer input

Input and canvas embeddings are concatenated rather than passed through separate encoder and decoder stacks:

\[
X_0 = [X_{input}; X_{canvas}]
\in \mathbb{R}^{B \times (L_i + 4096) \times 256}.
\]

For a maximum-size local-full batch:

\[
X_0 \in \mathbb{R}^{9 \times 8192 \times 256}.
\]

The combined boolean mask is `[input_attention_mask; canvas_attention_mask]`. Attention is fully bidirectional and non-causal. The mask is a broadcast key mask shaped internally as `[B, 1, 1, L]`; it masks invalid keys, not query computation. Input logits and padded-query hidden states may be computed, but only canvas positions enter the loss.

There is no learned input/canvas region embedding. Region identity is expressed indirectly through feature residual availability, self-conditioning on the canvas, token grammar, and concatenated position.

## Transformer backbone

### RoPE

Each attention layer owns a parameter-free Llama 3.1-style frequency-scaled rotary module configured with:

- head dimension 64;
- 32 inverse frequencies;
- base `theta = 500,000`;
- scaling factor 8;
- low/high frequency factors 1 and 4;
- original context 8,192.

The 32-element inverse-frequency tensor is a non-persistent FP32 buffer in each of the 10 layers. RoPE can evaluate arbitrary sequence lengths and contains no learned position parameters; 8,192 is the configured scaling reference and current maximum budget-derived length, not a hard implementation cap.

### One transformer block

Each of the 10 blocks applies:

```text
x
→ learned pre-attention RMSNorm
→ dense multi-head self-attention
→ learned post-attention RMSNorm
→ residual add
→ learned pre-FFN RMSNorm
→ dense GeGLU FFN
→ learned post-FFN RMSNorm
→ residual add
```

All RMSNorms use epsilon `1e-6`. This is sandwich normalization: each residual branch is normalized both before its transformation and after the branch transformation, before addition.

### Attention shapes

For hidden state `[B, L, 256]`:

1. One bias-free `256 → 768` projection produces QKV.
2. It reshapes to `[B, L, 3, 4, 64]`.
3. Q, K, and V become `[B, 4, L, 64]`.
4. Q and K each pass through a learned 64-element RMSNorm shared across the four heads.
5. RoPE rotates Q and K.
6. PyTorch scaled-dot-product attention uses the default scale `1/sqrt(64) = 1/8`, the broadcast key mask, and `is_causal=False`.
7. Heads merge back to `[B, L, 256]` and pass through a bias-free `256 → 256` output projection.

This is vanilla multi-head attention, not grouped-query or multi-query attention. CUDA execution permits Flash Attention and memory-efficient SDPA only; the quadratic math fallback is forbidden.

### Dense GeGLU

The bias-free FFN is:

\[
g = GELU_{tanh}(W_gx), \qquad u = W_ux,
\]

\[
FFN(x) = W_d(g \odot u),
\]

where gate and up project `256 → 1024` and down projects `1024 → 256`.

### Gradient checkpointing

The local-full profile checkpoints each transformer block while training. This changes activation storage and recomputation, but does not change model outputs, parameter shapes, or parameter counts.

## Output head and loss

After 10 blocks, a final learned RMSNorm produces:

\[
H \in \mathbb{R}^{B \times (L_i + 4096) \times 256}.
\]

An untied, bias-free linear head maps every position to 383 logits:

\[
W_{out} \in \mathbb{R}^{383 \times 256},
\]

\[
logits \in \mathbb{R}^{B \times (L_i + 4096) \times 383}.
\]

At the maximum local-full shape this is `[9, 8192, 383]`. Training discards the input-region logits and scores:

\[
canvas\_logits \in \mathbb{R}^{9 \times 4096 \times 383}.
\]

Uniform-mode training uses clean-target cross-entropy over every position selected by `canvas_loss_mask`, including unchanged positions, changed positions, replacements equal to the target, the outcome token, delimiters, `[END]`, and semantic `[PAD]`. It applies neither inverse-time weighting nor corruption-mask restriction.

The seven class IDs are loss-decomposition and optional weighting labels, not separate learned output heads:

| Class ID | Pretraining meaning |
|---:|---|
| 0 | enemy-observed reconstruction |
| 1 | enemy-fogged reconstruction |
| 2 | enemy-future prediction |
| 3 | delimiter |
| 4 | end |
| 5 | pad |
| 6 | win/loss |

All pretraining class weights are 1.0. The optional entropy-based auxiliary confidence loss is implemented but multiplied by the current weight 0.0, so it contributes no training gradient.

### Reported loss decompositions

`CanvasCrossEntropyLoss.forward` returns the scalar training loss plus five read-only decompositions of the same per-position cross-entropy. None of them changes the optimized objective; they exist so a run's failure mode can be localized. Every decomposition follows one emptiness convention: a key with zero scored positions is absent from the dict, and the CSV writers render it as a blank cell rather than a sentinel.

| Decomposition | Keys | Partitions by |
|---|---|---|
| `per_class` | the seven class IDs above | semantic role of the target token |
| `t_bucket` | `t_eq_1`, `t_0_75_to_1_0`, `t_0_25_to_0_75`, `t_0_0_to_0_25` | the example's sampled corruption level `t` |
| `canvas_state` | `ground_truth_preserved`, `noised` | whether the shown canvas token already equalled the target |
| `perspective` | `p1`, `p2` | which player the example was built from |
| `future_distance` | `1`, `2_5`, `6_10`, `11_30`, `31_plus` | prediction distance in timesteps, over `enemy-future` positions only |

Two boundaries carry meaning beyond bookkeeping:

- **`t_eq_1` is separate from `t_0_75_to_1_0`.** At exactly `t = 1` no ground-truth canvas token survives and the sequence must be generated from the clamped input alone. At `t = 0.99` a few true tokens survive and can anchor the rest. Merging the two would hide performance in the only regime that matches unconditional sampling. With `schedule.t_one_fraction = 0.0` the `t_eq_1` bucket is populated only by an exact 1.0 draw and is usually empty; raising `t_one_fraction` densifies it.
- **`canvas_state` keys on token inequality (`CorruptionOutput.changed_positions`), not on the Bernoulli corruption flag (`corrupted_positions`).** Under uniform diffusion a corrupted position can be re-drawn as its own target token; the model cannot distinguish that from an untouched position, so scoring it as noised would blur the exact distinction the split measures. `ground_truth_preserved` therefore reads as "does the model recognize an already-valid token and leave it alone", which is a direct probe of how well it has learned the shape of a legal sequence. Under the absorbing ablation only corrupted positions are scored at all, so that key is legitimately absent there.

`TrainingLoop` writes these decompositions to three artifacts under the profile's log directory:

| Artifact | Cadence | Contents |
|---|---|---|
| `step_metrics.jsonl` | every optimizer step | last microbatch's loss and all decompositions |
| `interval_metrics.csv` | `INTERVAL_REPORTS_PER_EPOCH` = 10 times per epoch | train values for every decomposition, each row scoped to that ~10% slice of the epoch, plus dev values when `train.interval_dev_evaluation` is true |
| `epoch_metrics.csv` | once per epoch | the same loss columns plus timestep percentiles, future-distance buckets, and memory/throughput telemetry |

The intra-epoch cadence exists because a corpus large enough that pretraining converges in one epoch would otherwise yield exactly one per-epoch observation and no visible trend. Report boundaries are `ceil(batches_per_epoch × k / 10)` for `k = 1..10`, so the tenth boundary always coincides with the epoch end and the two CSVs describe the same point in training.

`train.interval_dev_evaluation` controls the dev half of those rows, and defaults to true. When true, each interval row's dev values come from a full pass over the dev loader with EMA weights, identical to epoch-end validation. When false the train-side breakdown is still reported ten times per epoch and only the dev columns are left blank, with dev evaluated once per epoch into `epoch_metrics.csv`. The knob exists because dev cost does not scale with training cost: on the overfit profile a dev pass is ~18 s against ~105 s of training per epoch, so ten of them would more than double a 30-epoch run for resolution that 30 per-epoch dev points already supply.

## Exact trainable parameter inventory

### Embedding and conditioning subsystem

| Parameter group | Shape/count derivation | Parameters |
|---|---:|---:|
| Shared token embedding | `383 × 256` | 98,048 |
| Feature MLP first weight | `32 × 360` | 11,520 |
| Feature MLP first bias | `32` | 32 |
| Feature MLP second weight | `32 × 32` | 1,024 |
| Feature MLP second bias | `32` | 32 |
| Joint mixer first weight | `256 × 288` | 73,728 |
| Joint mixer first bias | `256` | 256 |
| Joint mixer second weight | `256 × 256` | 65,536 |
| Joint mixer second bias | `256` | 256 |
| Self-conditioning RMSNorm | `256` | 256 |
| Self-conditioning GeGLU gate | `256 × 256` | 65,536 |
| Self-conditioning GeGLU up | `256 × 256` | 65,536 |
| Self-conditioning GeGLU down | `256 × 256` | 65,536 |
| Scale-less post RMSNorm | no learned scale | 0 |
| **Embedding subsystem total** |  | **447,296** |

### One transformer block

| Parameter group | Shape/count derivation | Parameters |
|---|---:|---:|
| Four sandwich RMSNorm scales | `4 × 256` | 1,024 |
| QKV projection | `768 × 256` | 196,608 |
| Attention output projection | `256 × 256` | 65,536 |
| Q RMSNorm | `64` | 64 |
| K RMSNorm | `64` | 64 |
| GeGLU gate | `1024 × 256` | 262,144 |
| GeGLU up | `1024 × 256` | 262,144 |
| GeGLU down | `256 × 1024` | 262,144 |
| **One-block total** |  | **1,049,728** |

Ten blocks contain `10 × 1,049,728 = 10,497,280` parameters. The final RMSNorm adds 256.

### Whole model

| Subsystem | Parameters |
|---|---:|
| Embedding and conditioning | 447,296 |
| Ten transformer blocks | 10,497,280 |
| Final RMSNorm | 256 |
| Complete backbone | 10,497,536 |
| Untied output head | 98,048 |
| **Total trainable parameters** | **11,042,880** |

Every model parameter currently has `requires_grad=True`.

### Non-trainable model buffers

| Buffer group | Elements |
|---|---:|
| Feature means | 25 |
| Feature standard deviations | 25 |
| Ten RoPE inverse-frequency buffers | `10 × 32 = 320` |
| **Total buffer elements** | **370 FP32 values** |

The RoPE buffers are non-persistent; feature-statistics buffers are carried in model state and bound to a content identity.

## Initialization

All embeddings and linear weights normally initialize from `Normal(0, 0.02)`. Attention-output and FFN-down residual projections use:

\[
\sigma_{residual} = \frac{0.02}{\sqrt{2 \times 10}} \approx 0.0044721.
\]

Linear biases initialize to zero and learned RMSNorm scales initialize to one. After generic initialization, the joint mixer's final weight and bias are reset to zero.

## Optimizer, scheduler, EMA, and precision

### AdamW

One AdamW parameter group receives every model parameter:

- learning rate `3e-4`;
- betas `(0.9, 0.95)`;
- epsilon `1e-8`;
- weight decay `0.1`.

There is no special no-decay group, so embeddings, RMSNorm scales, and biases receive the same configured AdamW decay behavior as the other parameters.

### Schedule and optimization

- Linear warmup for 200 optimizer steps.
- Cosine decay over the resolved planned-step count.
- Final learning-rate multiplier 0.1.
- Gradient norm clipping at 1.0.
- One batch per optimizer step in local-full (`accumulation_steps=1`, token target disabled).
- BF16 autocast compute on CPU/CUDA; parameters are not converted to BF16 and remain FP32.

### EMA

Training owns a complete non-trainable model copy updated after optimizer steps:

\[
\theta_{EMA} \leftarrow 0.9999\theta_{EMA} + 0.0001\theta.
\]

EMA weights are used for validation, final checkpointing, sampling, and evaluation.

### Persistent parameter-state memory

For 11,042,880 FP32 parameters:

| State | Size |
|---|---:|
| Raw FP32 model | 44,171,520 bytes = 42.125 MiB |
| Raw model + FP32 EMA | 84.250 MiB |
| Two FP32 Adam moment tensors | 84.250 MiB |
| FP32 gradients | 42.125 MiB |
| **Model + EMA + Adam moments + gradients** | **210.626 MiB** |

These figures deliberately exclude activations, saved tensors, SDPA workspaces, input batches, allocator fragmentation/reservation, and checkpoint serialization. They are not a GPU peak-memory prediction.

## Sampling machinery

Uniform inference starts every eligible canvas position from an independent integer sampled from `[1, 383)`. Each of at most 64 passes:

1. runs the same model over the clamped input and current canvas;
2. temperature-shapes the logits using the current linear 0.8 → 0.4 schedule;
3. sets only the `[MASK]` logit to negative infinity;
4. samples categorical clean-state candidates;
5. sorts eligible positions by entropy;
6. accepts the prefix satisfying `cumsum(sorted_entropy) - sorted_entropy <= 0.1`;
7. replaces every nonaccepted eligible position with fresh uniform non-`[MASK]` noise.

Acceptance is transient and recomputed on every pass; a previously plausible position can be renoised. Completed rows freeze. Adaptive stopping requires mean eligible-position entropy below 0.005 and unchanged argmax predictions across two consecutive passes.

Normal sampling returns a canvas `[B, 4096]`. Optional diagnostic final logits require one explicitly requested extra forward pass and have shape `[B, 4096, 383]`. The diagnostics-only one-pass path starts from the selected process's terminal prior and performs exactly one denoiser call.

## Pretraining versus debut/outcome fine-tuning

Fine-tuning reuses the same model, embeddings, backbone, 383-way token head, and parameter count. It does not add a classification head.

The current overfit-V2 fine-tune profile changes the following pipeline behavior:

- uses debut/first-appearance canvas targets plus the same position-zero outcome token;
- warm-starts model weights from the overfit-V2 pretraining checkpoint;
- uses learning rate `1e-6` for 150 epochs;
- assigns semantic `[PAD]` class weight 0.2 while other seven-class weights remain 1.0;
- uses a separate window manifest and checkpoint namespace;
- caps the configured evaluation report at 40 examples.

The stable numeric class IDs remain the same, but IDs 0–2 are interpreted as visible-debut, fogged-debut, and future-debut for fine-tuning metrics.

## Explicit architecture observations

These are high-consequence implementation facts that must remain explicit in future updates:

1. **No explicit diffusion-time input.** Training samples `t`, but the model forward signature does not receive it.
2. **One joint bidirectional stack.** Input and canvas are concatenated; there is no encoder-decoder split or cross-attention.
3. **Input logits are computed but unsupervised.** The output head runs over the whole sequence, then training slices away `Li` positions.
4. **Output weights are untied.** Token embedding and output head have equal shapes but are separate parameter tensors.
5. **No region embedding.** There is no learned input-versus-canvas segment ID.
6. **Semantic padding is learned.** `[PAD]` inside the canvas is an attended target; batch-shape padding is excluded.
7. **Tensor IDs 6–99 participate in uniform noise and inference.** Only `[MASK]` is explicitly excluded, even though 6–99 are unnamed.
8. **Null self-conditioning still normalizes canvas embeddings.** With a zero conditioning signal, canvas token embeddings pass through the scale-less post RMSNorm.
9. **Q/K norm scales are shared across heads within a layer.** Each Q or K norm owns 64 values, not `4 × 64` independent values.
10. **Gradient checkpointing is not architecture capacity.** It changes compute/memory tradeoffs only.

## Provenance boundary

The last recorded `local_full` console log predates the current feature-conditioning implementation and reported 10,803,200 parameters. That number is historical and must not be used as the current source architecture. No checkpoint should be assumed to embody the 11,042,880-parameter architecture unless its stamped metadata and state dictionary are verified against current construction.

## Source-of-truth map

| Concern | Owning source |
|---|---|
| Canonical defaults and merged profiles | `config/default.yaml`, `configs/local_full.yaml`, `src/thesis_ml/config.py` |
| Special/content vocabulary | `src/thesis_ml/vocab/special_tokens.py`, `src/thesis_ml/vocab/content_vocab.py`, `data/Token_Dictionary.json` |
| Raw feature codec and widths | `src/thesis_ml/data/features.py` |
| Feature statistics | `src/thesis_ml/data/feature_stats.py` and configured statistics artifact |
| Input/target grammar | `src/thesis_ml/data/dataset.py`, `src/thesis_ml/data/windowing.py` |
| Batch shapes and masks | `src/thesis_ml/data/collate.py` |
| Embedding and conditioning | `src/thesis_ml/model/embedding.py` |
| Transformer/RoPE | `src/thesis_ml/model/backbone.py` |
| Model assembly/output | `src/thesis_ml/model/model.py` |
| Canvas loss | `src/thesis_ml/model/loss.py` |
| Corruption | `src/thesis_ml/train/corruption.py` |
| Optimizer, scheduler, self-conditioning, EMA | `src/thesis_ml/train/loop.py` |
| Training-profile construction | `src/thesis_ml/pipeline/train_pipeline.py` |
| Fine-tuning-profile construction | `src/thesis_ml/pipeline/finetune_pipeline.py` |
| Sampling | `src/thesis_ml/inference/sampler.py` |

## Required freshness check

When any source above or any model-facing config changes, use `UPDATE_PROMPT.md`, recompute the live values, update every affected section here, edit the canonical `.mmd`, regenerate the SVG/PNG, run the owning focused tests, and refresh the semantic indexes. Do not append a new architecture version; replace stale content in place.
