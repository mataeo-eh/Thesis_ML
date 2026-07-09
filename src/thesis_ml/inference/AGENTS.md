# inference Subpackage Contract

## Purpose

- Own iterative denoising sampling, canvas grammar validation and decoding, and external absolute-time recovery, per `SPEC.md` §7, §9.

## Ownership

- `sampler.py` owns confidence-based iterative unmasking (`sample_canvas`), diagnostics-only one-pass denoising (`denoise_canvas_once`), sampling temperature, checkpoint loading, and sampler result types.
- `decode.py` owns grammar validation and canvas decoding (`validate_canvas`, `validate_debut_canvas`, `decode_canvas`, `CanvasValidation`, `DecodedCanvas`).
- `timing.py` owns post-sampling absolute-time recovery (`attach_absolute_times`, `TimedTimestep`).

## Local Contracts

- The canvas initializes as all `[MASK]`. Each step commits the highest-confidence positions and never re-masks committed tokens; remasking-style revision is out of scope (`SPEC.md` §12).
- Load EMA weights for sampling and evaluation.
- A valid canvas starts with perspective-relative `[WIN]`/`[LOSS]`, followed by `(timestep-tokens [DELIMITER])+`, then either `[END] [PAD]*` or `[PAD]*`. Reject partial final timesteps; debut mode also permits empty timestep groups represented by a bare delimiter.
- The model never emits time. Absolute timing is recovered externally by arithmetic (last input-frame clock + `sampling_interval_s × timestep index`) and stays metadata only.
- Normal inference starts from an all-`[MASK]` canvas regardless of the training `t` distribution. Sampler hyperparameters (`max_steps`, falling `temperature`, `entropy_bound`, confidence threshold, minimum commits, and outcome-last behavior) are config-owned and provisional.
- Each iteration sorts still-masked positions by predictive entropy and commits qualifying low-entropy positions subject to the cumulative entropy budget, confidence threshold, and minimum-commit floor. Committed tokens are never revised.
- Token choice is argmax, not categorical sampling; temperature changes probability sharpness, entropy, and therefore commit timing. If `max_steps` expires before all positions qualify, the sampler returns the remaining `[MASK]` positions rather than force-filling them.
- Normal sampling performs no post-sampling model call. The diagnostics-only `return_final_logits` option performs one final forward pass conditioned on the completed canvas and returns those raw canvas logits on CPU.
- `denoise_canvas_once` is the diagnostics-only `t=1` training-corruption endpoint: it predicts an all-`[MASK]` canvas in exactly one forward call, commits every argmax position, and performs no iterative refinement or self-conditioning estimate pass.

## Work Guidance

- Keep the sampler's self-conditioning reuse identical to the training interface. Extra forward work must remain explicit and diagnostics-only.
- Grammar validation is the contract boundary between sampling and evaluation; decode only validated canvases.

## Verification

- Sampler changes require `tests/test_sampler.py` and `tests/test_sampler_outcome_last.py` (grammar validity of generated canvases, `SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
