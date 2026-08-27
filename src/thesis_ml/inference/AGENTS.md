# inference Subpackage Contract

## Purpose

- Own iterative denoising sampling, canvas grammar validation and decoding, and external absolute-time recovery, per `SPEC.md` §7, §9.

## Ownership

- `sampler.py` owns DiffusionGemma-style nonmonotonic uniform EB sampling (`sample_canvas`), coherent absorbing EB ablation sampling, diagnostics-only one-pass denoising (`denoise_canvas_once`), temperature shaping, stopping-state validation, terminal-pass finalization, checkpoint loading, and sampler result types.
- `decode.py` owns grammar validation and canvas decoding (`validate_canvas`, `validate_debut_canvas`, `decode_canvas`, `CanvasValidation`, `DecodedCanvas`).
- `timing.py` owns post-sampling absolute-time recovery (`attach_absolute_times`, `TimedTimestep`).

## Local Contracts

- Default uniform sampling clamps canvas position-0 `[BOS]` and initializes every other eligible position independently from `[PAD]`, `[DELIMITER]`, or a content ID. Each non-terminal pass samples categorical clean-state candidates, accepts the exact entropy-bounded prefix, and replaces every nonaccepted eligible position with fresh allowed noise. Acceptance is transient and recomputed from scratch, so earlier eligible positions may be renoised and revised. Canvas position 1 (`[WIN]`/`[LOSS]`) is denoised jointly with every other mutable position; only `[BOS]` is clamped and no position-ordering rule exists.
- Load EMA weights for sampling and evaluation.
- Sampling validates the checkpoint's feature-statistics identity against the model before loading EMA weights; missing or mismatched identities are incompatible rather than silently accepted.
- A valid canvas starts `[BOS] [WIN|LOSS]`, followed by `(timestep-tokens [DELIMITER])+`, then either `[END] [PAD]*` or `[PAD]*`. Reject `[EOS]`, extra BOS/outcome tokens, and partial final timesteps; debut mode also permits empty timestep groups represented by a bare delimiter.
- The model never emits time. Absolute timing is recovered externally by arithmetic (last input-frame clock + `sampling_interval_s × timestep index`) and stays metadata only.
- The uniform EB rule sorts eligible entropies ascending and accepts positions where `cumsum(sorted_entropy) - sorted_entropy <= entropy_bound`. Candidate tokens are categorical samples from temperature-shaped logits with `[MASK]` suppressed; every nonaccepted position is fully renoised from the `[PAD]`/`[DELIMITER]`/content distribution.
- Uniform defaults are `max_steps=64`, linear temperature `0.8 -> 0.4`, and `entropy_bound=0.1`. There is no minimum step count, commit-gating path, persistent acceptance mask, or position-dependent token restriction.
- Adaptive stopping requires both mean entropy below `0.005` over eligible positions and a **denoiser fixed point** on the same pass: the clean-state argmax equals the canvas that pass read at every mutable position. The stability half is a predicate on the STATE, matching the reference `TokenStabilityEarlyStop`; an argmax-versus-argmax test never inspects the canvas and can certify a row whose renoised positions all disagree with the model. Done rows freeze; unfinished rows continue to the hard ceiling.
- **Terminal-pass finalization.** A pass is terminal for a row when its adaptive stop fires there or the 64-pass ceiling ends it there. A terminal pass does not apply the entropy budget: every eligible position takes its categorical candidate and nothing is renoised. Renoising is a mid-process transition, never a result. This uses the distribution already computed for that pass, so it adds no model call and no hyperparameter.
- **Result contract.** The returned canvas is the state the stop decision validated. `SamplerOutput.finalized_steps[row]` names the pass that produced each row's canvas, and `stop_reasons[row]` states what it is certified to be: `adaptive_entropy_stability` (a certified fixed point), `max_steps` (fully committed, not certified), `absorbing_complete`, or `no_eligible`.
- The absorbing ablation shares only the terminal-pass rule, which commits still-masked positions when the ceiling arrives so `[MASK]` never survives into a returned canvas. Nothing accepted is ever remasked, so it stays monotone.
- The absorbing ablation initializes `[MASK]`, applies the same correct EB prefix among still-masked positions, leaves nonaccepted positions masked, never remasks accepted positions, and stops only when eligible positions are filled.
- Uniform replacement noise excludes `[MASK]`, `[END]`, `[WIN]`, `[LOSS]`, `[BOS]`, and `[EOS]`; clean-token candidates may still predict legitimate target specials. BOS alone is position-clamped; outcome position 1 remains mutable. Validation remains the downstream grammar boundary.
- Normal sampling performs no post-sampling model call. The diagnostics-only `return_final_logits` option performs one final forward pass conditioned on the completed canvas and returns those raw canvas logits on CPU.
- `denoise_canvas_once` is the diagnostics-only `t=1` endpoint: it clamps BOS and uses the uniform allowed-noise support elsewhere in uniform mode, or BOS plus `[MASK]` elsewhere in absorbing mode, performs exactly one forward call, and returns categorical/argmax diagnostic output as explicitly defined by the call without iterative refinement or an estimate pass.

### Frozen input-KV reuse — AN ABLATION TOGGLE, NOT THE DEFAULT PATH

Sampling participates in `model.frozen_input_kv`, which was **promoted to default `true` on 2026-08-09** (`SPEC.md` §14b) after ablation arm 01 measured no meaningful loss difference and materially faster inference. Sampling is where that speedup is realized: the input region is identical across every denoising step, so its K/V is computed once and reused for all of them. With the toggle off the sampler is still byte-for-byte the historical implementation — no cache objects, no extra kwargs — so the opt-out path is intact and unchanged.

- When on, the FIRST denoising step passes `return_cached_input_kv=True` and captures a `FrozenInputKV`; every later step passes `cached_input_kv=cache`, skipping the L backbone blocks over the input region. Measured 7.1x per-step speedup at `input_len=1536`.
- **There is deliberately NO separate pre-loop priming forward.** One would add a whole extra full-length model call — exactly the work the toggle exists to avoid — and would break the one-model-call-per-pass contract above, which `tests/test_sampler.py` asserts as `model.calls == output.steps`. This is a contract, not an oversight; do not "optimize" a priming pass in.
- The sampler's enable predicate mirrors the backbone's (`frozen_input_kv and input_len > 0`) and is read off the MODEL, not off `ProjectConfig`, so the documented `ValueError` for an illegal cache request is unreachable rather than merely unlikely.
- A cache is valid only for the same input region, batch size, and layer count. The input region is bound once before the loop and never rebound; if that ever stops being true, the cache must be invalidated.
- `input_lengths=` is passed only when the model declares `per_segment_positions` truthy AND the batch actually exposes an `input_lengths` tensor. Both conditions are load-bearing: sampler helpers are called with duck-typed stub models and `SimpleNamespace` batches from `tests/test_eval.py`, `tests/test_viz.py`, and `tests/test_finetune_report.py`, and passing the kwarg unconditionally broke eight tests. Do not collapse this guard. When omitted, the model derives the value from the attention mask, which is identically equal by construction.
- `SamplerStep` carries `forward_wall_seconds: float = 0.0` and `used_cached_input_kv: bool = False`. Both are defaulted and appended, so nothing constructing or reading `SamplerStep` changes. Timing uses `time.perf_counter` on CPU and `torch.cuda.Event` on GPU — a bare `perf_counter` around an async CUDA launch would report a fake speedup. This is unrelated to `timing.py`, which owns in-game absolute-clock recovery (`SPEC.md` §7), not performance measurement.

## Work Guidance

- Keep the sampler's self-conditioning reuse identical to the training interface. Extra forward work must remain explicit and diagnostics-only.
- Grammar validation is the contract boundary between sampling and evaluation; decode only validated canvases.
- When priors, allowed states, temperature shaping, entropy selection, adaptive stopping, self-conditioning reuse, forward-call count, returned tensors, or grammar constraints change, update every affected inference section in `../../../Model_Architecture/MODEL_ARCHITECTURE.md`, update canonical `.mmd` edges, and regenerate its SVG/PNG using `UPDATE_PROMPT.md`.

## Verification

- Sampler changes require `tests/test_sampler.py`; position-unconstrained, nonmonotonic-renoising, stopping-state-certificate, and terminal-pass finalization coverage replaces the retired positional sequencing tests. Grammar validity remains required by `SPEC.md` §16.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
