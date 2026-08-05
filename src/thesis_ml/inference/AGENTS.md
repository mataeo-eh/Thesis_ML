# inference Subpackage Contract

## Purpose

- Own iterative denoising sampling, canvas grammar validation and decoding, and external absolute-time recovery, per `SPEC.md` §7, §9.

## Ownership

- `sampler.py` owns DiffusionGemma-style nonmonotonic uniform EB sampling (`sample_canvas`), coherent absorbing EB ablation sampling, diagnostics-only one-pass denoising (`denoise_canvas_once`), temperature shaping, adaptive stopping, checkpoint loading, and sampler result types.
- `decode.py` owns grammar validation and canvas decoding (`validate_canvas`, `validate_debut_canvas`, `decode_canvas`, `CanvasValidation`, `DecodedCanvas`).
- `timing.py` owns post-sampling absolute-time recovery (`attach_absolute_times`, `TimedTimestep`).

## Local Contracts

- Default uniform sampling initializes eligible positions independently from every vocabulary state except `[MASK]`. Each pass samples categorical clean-state candidates, accepts the exact entropy-bounded prefix, and replaces every nonaccepted eligible position with fresh uniform noise. Acceptance is transient and recomputed from scratch, so earlier positions may be renoised and revised.
- Load EMA weights for sampling and evaluation.
- Sampling validates the checkpoint's feature-statistics identity against the model before loading EMA weights; missing or mismatched identities are incompatible rather than silently accepted.
- A valid canvas starts with perspective-relative `[WIN]`/`[LOSS]`, followed by `(timestep-tokens [DELIMITER])+`, then either `[END] [PAD]*` or `[PAD]*`. Reject partial final timesteps; debut mode also permits empty timestep groups represented by a bare delimiter.
- The model never emits time. Absolute timing is recovered externally by arithmetic (last input-frame clock + `sampling_interval_s × timestep index`) and stays metadata only.
- The uniform EB rule sorts eligible entropies ascending and accepts positions where `cumsum(sorted_entropy) - sorted_entropy <= entropy_bound`. Candidate tokens are categorical samples from temperature-shaped logits; every nonaccepted position is fully renoised from the uniform non-`[MASK]` state distribution.
- Uniform defaults are `max_steps=64`, linear temperature `0.8 -> 0.4`, and `entropy_bound=0.1`. There is no minimum step count, commit-gating path, persistent acceptance mask, or position-dependent token restriction.
- Adaptive stopping requires both mean entropy below `0.005` over eligible positions and unchanged argmax predictions across two consecutive passes. Done rows freeze; unfinished rows continue to the hard ceiling.
- The absorbing ablation initializes `[MASK]`, applies the same correct EB prefix among still-masked positions, leaves nonaccepted positions masked, never remasks accepted positions, and stops only when eligible positions are filled.
- Uniform sampling excludes `[MASK]` but otherwise permits every state at every position. Ground-truth grammar and the position-zero outcome convention are learned; validation remains the downstream contract boundary.
- Normal sampling performs no post-sampling model call. The diagnostics-only `return_final_logits` option performs one final forward pass conditioned on the completed canvas and returns those raw canvas logits on CPU.
- `denoise_canvas_once` is the diagnostics-only `t=1` endpoint: it uses a uniform random non-`[MASK]` canvas in uniform mode and all `[MASK]` in absorbing mode, performs exactly one forward call, and returns categorical/argmax diagnostic output as explicitly defined by the call without iterative refinement or an estimate pass.

## Work Guidance

- Keep the sampler's self-conditioning reuse identical to the training interface. Extra forward work must remain explicit and diagnostics-only.
- Grammar validation is the contract boundary between sampling and evaluation; decode only validated canvases.

## Verification

- Sampler changes require `tests/test_sampler.py`; position-unconstrained and nonmonotonic-renoising coverage replaces the retired positional sequencing tests. Grammar validity remains required by `SPEC.md` §16.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
