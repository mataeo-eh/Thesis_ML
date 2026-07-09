# model Subpackage Contract

## Purpose

- Own the dense bidirectional masked-diffusion network and its learned input embeddings and canvas loss, per `SPEC.md` §2, §3, §6.

## Ownership

- `backbone.py` owns the LLaDA/LLaMA-lineage layers: `RMSNorm` (pre-norm), `RotaryEmbedding`, and `apply_rope` (Llama 3.1 frequency-scaled RoPE).
- `embedding.py` owns the input-only additive contextual encodings (`InputFeatures`, `FourierFeatures`, `build_input_features`): map position and unit stats, team flag, delimiters.
- `model.py` owns the assembled network (`SC2StrategyDiffusionModel`, `ModelOutput`, `_combine_attention_masks`, `canvas_self_conditioning_from_logits`).
- `loss.py` owns position-wise canvas cross-entropy with per-class logging (`CanvasCrossEntropyLoss`, `LossOutput`, `active_class_id_to_name`).

## Local Contracts

- Single dense bidirectional stack: RMSNorm, SwiGLU FFN, vanilla multi-head attention (never grouped-query), config-gated QK-norm (default on), Llama 3.1 scaled RoPE for sequence position only.
- Attention is full bidirectional with a padding mask only — no causal mask. CUDA attention explicitly prioritizes fused Flash SDPA, falls back only to memory-efficient SDPA with a broadcast boolean key mask, and forbids math fallback.
- The input region is clamped: never noised, never receives loss. Loss is computed on canvas positions only.
- Additive contextual encodings are input-only; canvas tokens never receive map position or unit stats. Map position uses extrapolation-friendly Fourier/sinusoidal features, never a learned bin lookup.
- Conditioning is clamping only: no encoder-decoder split, no cross-attention, no copy mechanism, no classification head, no set/pooling module (`SPEC.md` §14).
- Per-token-class loss logging is mandatory from the first run; class weights are config (`class_loss_weights`, default 1.0).
- `rope_theta`, `rope_scaling.*`, `d_model`/`layers`/`heads`/`ffn`, `qk_norm`, `self_conditioning`, and `gradient_checkpointing` are config-owned; sequence length is not hard-capped in the rotary implementation.

## Work Guidance

- FFN width is the parameter-budget knob; keep MHA and adjust `ffn` for scale rather than switching attention variants.
- Self-conditioning must present identical train and inference interfaces and add no extra inference forward pass.
- Watch the copy-class loss when toggling QK-norm (it can weaken the input→canvas copy pathway).

## Verification

- Model changes require `tests/test_model.py` (shapes, masking, parameter count, class-loss wiring).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
