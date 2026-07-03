# inference Subpackage Contract

## Purpose

- Own iterative denoising sampling, canvas grammar validation and decoding, and external absolute-time recovery, per `SPEC.md` §7, §9.

## Ownership

- `sampler.py` owns confidence-based iterative unmasking (`sample_canvas`, `sampler_temperature`, `load_sampling_checkpoint`, `SamplerStep`, `SamplerOutput`).
- `decode.py` owns grammar validation and canvas decoding (`validate_canvas`, `validate_debut_canvas`, `decode_canvas`, `CanvasValidation`, `DecodedCanvas`).
- `timing.py` owns post-sampling absolute-time recovery (`attach_absolute_times`, `TimedTimestep`).

## Local Contracts

- The canvas initializes as all `[MASK]`. Each step commits the highest-confidence positions and never re-masks committed tokens; remasking-style revision is out of scope (`SPEC.md` §12).
- Load EMA weights for sampling and evaluation.
- A valid canvas is `(timestep-tokens [DELIMITER])+` followed by either `[END] [PAD]*` or `[PAD]*`. Reject partial final timesteps.
- The model never emits time. Absolute timing is recovered externally by arithmetic (last input-frame clock + `sampling_interval_s × timestep index`) and stays metadata only.
- Sampler hyperparameters (`max_steps`, falling `temperature`, `entropy_bound`) are config-owned and provisional.

## Work Guidance

- Keep the sampler's self-conditioning reuse identical to the training interface with no extra forward pass.
- Grammar validation is the contract boundary between sampling and evaluation; decode only validated canvases.

## Verification

- Sampler changes require `tests/test_sampler.py` and `tests/test_sampler_outcome_last.py` (grammar validity of generated canvases, `SPEC.md` §16).

## Child DOX Index

- No child `AGENTS.md` files currently exist.
