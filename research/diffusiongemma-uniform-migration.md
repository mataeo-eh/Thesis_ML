# DiffusionGemma uniform-state migration research

Research snapshot: 2026-08-05. Sampler-stopping section re-verified and
corrected against primary source on **2026-08-26** (see "2026-08-26 correction:
what the reference stopping rule actually compares").

This note records the evidence used to migrate `Thesis_ML` from absorbing-state
masked diffusion to uniform-state multinomial diffusion. It separates published
or released behavior from project decisions. `SPEC.md` remains authoritative.

## Primary sources and inspected versions

- Google AI DiffusionGemma model card and sampling defaults:
  <https://ai.google.dev/gemma/docs/diffusiongemma/model_card?hl=en>
- Google AI uniform-state and multi-canvas explanation:
  <https://ai.google.dev/gemma/docs/diffusiongemma/explained>
- Google developer guide and official Hackable Diffusion recipe announcement:
  <https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/>
- `How Transparent is DiffusionGemma?`, arXiv:2606.20560:
  <https://arxiv.org/abs/2606.20560>
- Gemma JAX implementation at commit
  `7b785991bd78626c73b317eb43fdbb6c292f7b9c` (2026-08-04), especially
  `gemma/diffusion/_sampler.py`, `gemma/diffusion/_transformer.py`, and
  `gemma/diffusion/hackable_diffusion_adapter/hd/sft_model.py`:
  <https://github.com/google-deepmind/gemma/tree/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/diffusion>
- Hackable Diffusion at commit
  `c221af9ef131f8e266846f287a2f4446aa75cb69` (2026-07-29), especially the
  categorical corruption and discrete-loss implementations:
  <https://github.com/google/hackable_diffusion/tree/c221af9ef131f8e266846f287a2f4446aa75cb69/hackable_diffusion/lib>
- Entropy-Bounded Sampler paper, arXiv:2505.24857:
  <https://arxiv.org/abs/2505.24857>
- D3PM foundation, arXiv:2107.03006:
  <https://arxiv.org/abs/2107.03006>
- Gemma 4 technical report, arXiv:2607.02770:
  <https://arxiv.org/abs/2607.02770>
- Post-release observational study of DiffusionGemma commitment behavior,
  arXiv:2606.14620:
  <https://arxiv.org/abs/2606.14620>

## Evidence

### Forward process and training recipe

- DiffusionGemma uses a linear uniform categorical process. At noise level `t`,
  each position retains its clean token with probability `1 - t`; otherwise it
  is replaced by a uniformly drawn vocabulary state. A random replacement may
  equal the clean token.
- Its terminal sampling prior is an independent uniform random-token canvas.
- The released Hackable Diffusion SFT recipe samples time uniformly, predicts
  clean-state `x0` logits, and uses unweighted cross-entropy over the selected
  valid target canvas rather than only the Bernoulli-corrupted positions.
- The released wrapper does not condition the denoiser explicitly on time.
- The released recipe is warm-started supervised fine-tuning, not a published
  from-scratch DiffusionGemma pretraining recipe. Its loss is therefore direct
  evidence for model compatibility, not proof of an optimal scratch objective.

### Self-conditioning

- Training runs a first denoiser pass, stops gradients through its logits, and
  performs a second loss-bearing pass with self-conditioning enabled per example
  with probability `0.5`; disabled examples receive a zero signal.
- Inference reuses the preceding denoising pass and adds no extra forward pass.
- Soft logits are converted through the shared token embedding table. The
  resulting expected embedding passes through RMSNorm and a gated feed-forward
  block, is added to the current canvas embedding, and receives a scale-less
  post RMSNorm.

### Entropy-bounded uniform sampler

- Each step temperature-shapes logits and samples one categorical clean-token
  candidate at every eligible canvas position.
- Candidate entropies are sorted ascending. With sorted entropies `h_i`, the
  accepted prefix is exactly the positions satisfying
  `cumsum(h)_i - h_i <= gamma`; DiffusionGemma uses `gamma = 0.1`.
- Every nonaccepted position is replaced with a fresh independent uniform token.
  Acceptance is recomputed from scratch on every step, so a position accepted on
  one step may be renoised later. There is no persistent committed mask.
- Temperature anneals linearly from `0.8` to `0.4` by default.
- Adaptive stopping combines a mean-entropy threshold of `0.005` with a token
  stability condition. See the 2026-08-26 correction below for exactly what that
  stability condition compares; the 2026-08-05 snapshot recorded it incorrectly.
- The official ceiling is 48 passes. This project changes only that ceiling to
  64 and otherwise begins from the released uniform EB behavior.

### Backbone

- DiffusionGemma inherits Gemma 4's RMSNorm, QK normalization, gated GELU feed
  forwarding, and post-attention/post-FFN normalization, but also inherits MoE,
  cache, local-attention, and block-generation choices tied to a different
  deployment goal.
- Gemma 4's dense feed-forward form is GeGLU: `GELU(gate) * up`, followed by the
  down projection.

## 2026-08-26 correction: what the reference stopping rule actually compares

Accessed 2026-08-26. Sources re-read in full at pinned revisions:

- `gemma/diffusion/_early_stopping.py` and `gemma/diffusion/_sampler.py` in
  google-deepmind/gemma at commit `7b785991bd78626c73b317eb43fdbb6c292f7b9c`
  (repository HEAD on this date; `_sampler.py` last changed in
  `0360c6629acf31361ab68924d5ed4a15c6097dc0`, 2026-07-03):
  <https://github.com/google-deepmind/gemma/tree/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/diffusion>
- Entropy-Bounded Sampler, arXiv:2505.24857v1, "Accelerated Sampling from Masked
  Diffusion Models via Entropy Bounded Unmasking", Algorithm 1.

### Released behavior (read from source, not inferred)

`TokenStabilityEarlyStop.should_stop` is, verbatim:

```python
most_likely_tokens = jnp.argmax(logits, axis=-1)
return jnp.all(most_likely_tokens == previous_canvas, axis=-1)
```

The comparison is **this pass's argmax against the canvas that produced those
logits** — a fixed-point certificate on the STATE. It is NOT a comparison of two
consecutive argmax tensors. The 2026-08-05 snapshot in the section above recorded
it as the latter, and that error was transcribed into this project's sampler and
into `SPEC.md`. The two rules are not equivalent: an argmax-versus-argmax test
never inspects the canvas, so it is satisfied while every renoised position
disagrees with the model.

`EntropyEarlyStop` is a separate mean-per-token-entropy threshold defaulting to
`0.005`, and `ChainedEarlyStop` ANDs stoppers together. There is no
consecutive-pass counter anywhere in the released early-stopping module; the
stability predicate is evaluated on a single pass.

`SampleFromPredictions` does renoise every nonselected position with a uniform
random token, and `sample_next_canvas` returns `final_carry.canvas`, i.e. the
post-renoising state. Inference from source: that is safe in the released setting
only because the stop condition it is paired with guarantees the renoise is a
no-op. `entropy_bound` is a budget in **total nats**, not per token, so on a
256-token canvas from a converged model the accepted prefix covers the whole
canvas and nothing is renoised.

### EB-Sampler paper

Algorithm 1 is a masked (absorbing) procedure: `while I_m != {} and not C(x)`,
annotated "Stop if all tokens unmasked". It never renoises, and its stopping
criterion is typed `C: S -> {True, False}` — a function of the state `x`. The
guarantee that no noise survives into the result comes from the loop condition,
not from the entropy rule. The entropy rule itself,
`sum(h_U) - max(h_U) <= gamma`, is what this project already implements.

### Project decision (2026-08-26)

Two changes, adopted together because neither is sufficient alone:

1. **Stopping-state validation.** The uniform adaptive stop now requires a
   denoiser fixed point over mutable positions, matching
   `TokenStabilityEarlyStop`. `sampler.stability_steps` is retired: a fixed-point
   certificate already compares a prediction against the state that produced it,
   and requiring it on consecutive passes is unsatisfiable whenever the entropy
   budget leaves a nonempty renoised tail.

2. **Terminal-pass full acceptance** (an intentional, documented deviation from
   the released code). On the last pass a row will ever execute — its stop firing
   or the ceiling — the budget is not applied and nothing is renoised. Rationale:
   this project's canvas is 4,096 positions rather than 256, and its model is far
   from the released model's convergence, so a fixed **total**-nats budget of
   `0.1` accepts a much smaller fraction and leaves a real renoised tail on every
   pass, including the last one. The released code's implicit assumption — that
   the terminal renoise is a no-op — does not hold here. Committing on the
   terminal pass restores the invariant the released sampler relies on, uses the
   distribution already computed for that pass, and adds no model call and no
   hyperparameter. Where the released assumption does hold, the two behaviors are
   identical.

Measured effect on the real held-out reproduction (checkpoint
`smallTrainingTestV3/epoch-0033`, EMA, CUDA, seed `20260826`, three windows at
`t=1.0`): grammar-valid canvases 1/3 -> 3/3, pooled build-order F1 0.296 -> 0.498,
returned-versus-final-argmax disagreements 7/4/0 -> 0/0/0, at a cost of 154 -> 164
total forward passes (+6.5%). See `diagnostics/010-uniform-sampler-final-state.md`.

## Accepted project decisions

- Uniform-state diffusion is the default; absorbing `[MASK]` diffusion remains a
  configuration-selectable ablation with process-compatible corruption, loss,
  prior, and sampling behavior.
- Uniform noise and categorical candidate sampling use every canvas state except
  `[MASK]`. `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`, and content tokens
  may occur at any position during noising and sampling. No position-dependent
  logit mask hardcodes the target grammar.
- Uniform training uses unweighted clean-state cross-entropy over every valid
  target-canvas position. It does not use inverse-`t` weighting or a
  corruption-branch score mask.
- Time remains uniformly sampled. Intentional exact-`t=1` oversampling is
  disabled by default (`0.0`).
- The clamped input, one full output canvas, dense MHA, full bidirectional
  attention, QK norm, and Llama 3.1 frequency-scaled RoPE remain.
- The backbone moves from SwiGLU/pre-norm-only blocks to dense GeGLU with
  pre- and post-sublayer RMSNorm. Self-conditioning moves to the released
  expected-embedding and gated-FFN path.
- The uniform sampler follows DiffusionGemma's categorical EB selection,
  mid-process full renoising, linear `0.8 -> 0.4` temperature, and
  dual-condition adaptive stop, with the stability half implemented as the
  released fixed-point certificate. The intentional default deviations are
  `max_steps = 64` and terminal-pass full acceptance (see the 2026-08-26
  correction); there is no minimum step count.
- `outcome_last`, confidence-threshold commits, minimum-commit fallbacks, and
  positional token restrictions are removed. Position zero is `[WIN]` or
  `[LOSS]` in ground truth, and the model learns that convention through loss.
- Auxiliary confidence sharpening remains an ablation knob but defaults to
  `0.0` because miscalibrated overconfidence can directly distort EB selection
  and adaptive stopping.
- The migration is checkpoint-incompatible. Checkpoint metadata must identify
  the diffusion process and architecture revision; incompatible weights fail
  before partial loading. Repository-local checkpoints from the retired
  architecture are deleted after the migration is verified.

## Explicitly rejected transfers

- Autoregressive warm-start assumptions or skipping scratch pretraining.
- Multi-canvas, block-autoregressive, or semi-autoregressive generation.
- Prompt KV-cache encoding, encoder-decoder separation, or cross-attention.
- MoE routing, grouped-query attention, and cache-oriented local/sliding
  attention.
- DiffusionGemma's fixed 256-token canvas, multimodal path, serving stack, and
  latency-first early-commit objective.
- Treating the released SFT objective as conclusive pretraining theory without
  project validation.
