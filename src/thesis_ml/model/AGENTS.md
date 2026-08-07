# model Subpackage Contract

## Purpose

- Own the dense bidirectional uniform-diffusion network, absorbing ablation compatibility, learned input embeddings, expected-embedding self-conditioning, and clean-state canvas loss, per `SPEC.md` §2, §3, §6.

## Ownership

- `backbone.py` owns the dense Gemma 4-lineage layers: sandwich `RMSNorm`, dense GeGLU, `RotaryEmbedding`, and `apply_rope` (Llama 3.1 frequency-scaled RoPE).
- `embedding.py` owns joint token-feature conditioning (`InputFeatures`, `build_input_features`, `InputContextEmbedding`): standardized valid continuous fields, validity bits, categorical cloak/buff fields, numeric allegiance, and feature-masked residual mixing. The strict raw codec is owned by `data/features.py`.
- `model.py` owns the assembled network (`SC2StrategyDiffusionModel`, `ModelOutput`, `_combine_attention_masks`, `canvas_self_conditioning_from_logits`).
- `loss.py` owns position-wise canvas cross-entropy with per-class logging (`CanvasCrossEntropyLoss`, `LossOutput`, `active_class_id_to_name`).

## Local Contracts

- Single dense bidirectional stack: pre/post-sublayer RMSNorm, dense GeGLU using tanh-approximate GELU, vanilla multi-head attention (never grouped-query), config-gated QK-norm (default on), and Llama 3.1 scaled RoPE for sequence position only.
- Attention is full bidirectional with a padding mask only — no causal mask. CUDA attention explicitly prioritizes fused Flash SDPA, falls back only to memory-efficient SDPA with a broadcast boolean key mask, and forbids math fallback.
- The input region is clamped: never noised, never receives loss. Loss is computed on canvas positions only.
- Joint static conditioning is input-only; canvas tokens never receive map position, unit stats, or allegiance. Slash-form current/maximum stats become ratios and facing becomes lossless sine/cosine. Valid continuous fields are standardized with valid-only train-split statistics; invalid fields are zeroed after standardization. Values, validity bits, categorical cloak/buff encodings, and allegiance are concatenated, projected by `Linear(F,32) -> ReLU -> Linear(32,32) -> ReLU`, then mixed jointly with token embeddings by `Linear(d_model+32,d_model) -> GELU -> Linear(d_model,d_model)`.
- The joint mixer's final projection is exactly zero-initialized after generic model initialization. The initial forward therefore equals token lookup exactly; the first optimizer step unlocks the residual path without altering delimiter or missing-feature positions.
- Conditioning is clamping only: no encoder-decoder split, no cross-attention, no copy mechanism, no classification head, no set/pooling module (`SPEC.md` §14).
- Uniform-mode loss predicts clean `x0` at every valid canvas position without inverse-time or corruption-mask weighting. The absorbing ablation retains its masked inverse-time objective. Per-token-class loss logging remains mandatory.
- `rope_theta`, `rope_scaling.*`, `d_model`/`layers`/`heads`/`ffn`, `qk_norm`, `self_conditioning`, and `gradient_checkpointing` are config-owned; sequence length is not hard-capped in the rotary implementation.
- Self-conditioning converts the stopped prior distribution through the shared token-embedding table, applies RMSNorm -> dense GeGLU, adds it to canvas token embeddings only, and applies scale-less RMSNorm. Training and inference share this interface.
- The GeGLU/sandwich-norm revision is intentionally incompatible with retired checkpoints; architecture/process metadata must fail closed before weight loading.

## Work Guidance

- FFN width is the parameter-budget knob; keep MHA and adjust `ffn` for scale rather than switching attention variants.
- Self-conditioning must present identical train and inference interfaces and add no extra inference forward pass; inference uses the same temperature-shaped distribution that drives sampling.
- Watch the observed/copy-class loss when toggling QK-norm or conditioning behavior (either can weaken the input→canvas copy pathway).
- Every model change requires a synchronized update of all affected prose, equations, shapes, per-module counts, whole-model totals, initialization details, and explicit absences in `../../../Model_Architecture/MODEL_ARCHITECTURE.md`, followed by canonical `.mmd` updates and SVG/PNG regeneration. Use `UPDATE_PROMPT.md` and verify by live model construction.

## Verification

- Model changes require `tests/test_model.py` (shapes, masking, parameter count, class-loss wiring).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
