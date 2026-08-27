# 010 — Uniform sampler returned a mid-process latent, not a result

Investigated and repaired 2026-08-26. Scope: `src/thesis_ml/inference/sampler.py`.

## Symptom

Held-out inference returned isolated nonsensical tokens and malformed output
grammar. Two of three rendered canvases carried an ordinary content token at the
perspective outcome slot (canvas position 1) where the grammar admits only
`[WIN]`/`[LOSS]`, while a post-sampling denoiser pass over that same canvas
preferred `[WIN]` with high probability. The returned token was outside the
exported final top 10 in both cases.

## Reproduction

Read-only, real pipeline, visible CUDA (RTX 3070):

- config `configs/smallTrainingTestV3.yaml`
- checkpoint `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt`, EMA
- seed `20260826`, output noise exact `t = 1.0`, fog `0.0`
- three held-out test windows (`--n-replays 1 --n-windows-per-replay 3`)
- sampler `thesis_ml.inference.sampler.sample_canvas`

## Root cause — two coupled defects in one control flow

### A. The returned canvas was a mid-process latent

The uniform transition accepts an entropy-bounded prefix and replaces every
**nonaccepted** eligible position with a fresh uniform draw. The loop then exited
immediately after that mutation and returned the mutated canvas. So the returned
result contained, verbatim, the uniform renoise draws of the last executed pass —
tokens the model never proposed and had never seen.

Traced at canvas position 1, window `..._p1_t0`, last three passes:

| pass | canvas read | argmax | entropy | accepted | renoised | canvas written |
|---|---|---|---|---|---|---|
| 47 | `supplydepot` | `[WIN]` | 0.165 | False | True | `ghostnova` |
| 48 | `ghostnova` | `[WIN]` | 0.272 | False | True | `lockon` |
| 49 | `lockon` | `[WIN]` | 0.239 | False | True | `nuke` |

The row was marked done on pass 49 and `nuke` was returned. All 7 of that
window's returned-versus-final-argmax disagreements were exactly the 7 positions
renoised on the final pass, and all 7 had final probability `< 1e-4`.

The third window is the natural control: its final pass happened to accept all
4,095 mutable positions, so nothing was renoised, and it had 0 disagreements,
0 implausible tokens, and valid grammar.

The outcome slot is the *most* affected position precisely because it is the
canvas's genuinely uncertain one. The entropy-bounded prefix accepts in ascending
entropy order, so the highest-entropy positions are the ones that fall outside
the budget and get renoised — every pass.

### B. The stop rule never inspected the canvas

The adaptive stop required mean entropy below `0.005` **and** that the argmax be
unchanged across two consecutive passes. That second condition compares one
prediction tensor against another prediction tensor. It is blind to the canvas,
so it was satisfied — see `argmax` column above, a stable `[WIN]` across passes
44–49 — while the canvas at that position churned through uniform noise on every
one of those passes. The sampler certified "converged" about a state it had never
checked.

## Primary-source comparison

Pinned at google-deepmind/gemma commit
`7b785991bd78626c73b317eb43fdbb6c292f7b9c`, accessed 2026-08-26.

`gemma/diffusion/_early_stopping.py::TokenStabilityEarlyStop.should_stop` is:

```python
most_likely_tokens = jnp.argmax(logits, axis=-1)
return jnp.all(most_likely_tokens == previous_canvas, axis=-1)
```

The reference compares this pass's argmax against **the canvas that produced
it** — a fixed-point certificate on the state, evaluated on a single pass with no
consecutive-pass counter. This project had transcribed it as an
argmax-versus-argmax test. That transcription error is the origin of defect B and
is corrected in `research/diffusiongemma-uniform-migration.md`.

The reference's `_sampler.py` does return the post-renoising canvas, and that is
safe *there* because the stop condition it is paired with makes the renoise a
no-op: `entropy_bound` is a budget in **total nats**, so on a 256-token canvas
from a converged model the accepted prefix covers everything. This project's
canvas is 4,096 positions and its model is far from that convergence, so the same
fixed `0.1` budget leaves a real renoised tail on every pass, including the last.
The released code's implicit precondition does not hold here.

The EB-Sampler paper (arXiv:2505.24857v1, Algorithm 1) is a masked procedure that
never renoises and loops `while I_m != {} and not C(x)` — "stop if all tokens
unmasked" — with the stopping criterion typed `C: S -> {True, False}`, again a
function of the state.

## Repair

Both defects are fixed together, because neither fix alone is sufficient: without
A, a legitimate ceiling exit still returns noise; without B, the stop can fire on
a mostly-noised canvas.

1. **Stopping-state validation.** The uniform adaptive stop now requires a
   denoiser fixed point over mutable positions, matching the reference.
   `sampler.stability_steps` is retired: a fixed-point certificate already
   compares a prediction against the state that produced it, and requiring it on
   consecutive passes is unsatisfiable whenever the budget leaves a nonempty
   renoised tail.

2. **Terminal-pass full acceptance.** On the last pass a row will ever execute —
   its stop firing, or the ceiling — the entropy budget is not applied: every
   eligible position takes its categorical candidate and nothing is renoised. The
   budget exists to bound the error of committing many positions while a later
   pass can still revise them; when no later pass exists it is meaningless, and
   renoising there injects noise the process can never remove. This uses the
   distribution already computed for that pass, so it adds no model call, no
   hyperparameter, and no grammar knowledge.

The process stays nonmonotonic: acceptance is still recomputed from scratch every
pass, there is still no persistent commitment mask, and any mutable position is
still revisable until its row's terminal pass. `[BOS]` alone remains clamped and
position 1 keeps no special treatment. The 64-pass ceiling is unchanged.

## Measured effect

Same checkpoint, windows, seed, device.

| window | passes (before → after) | disagreements | outcome slot | grammar |
|---|---|---|---|---|
| `..._p1_t0` | 49 → 55 | 7 → 0 | `nuke` → `[WIN]` | invalid → valid |
| `..._p1_t99` | 51 → 54 | 4 → 0 | `interceptor` → `[LOSS]` | invalid → valid |
| `..._p1_t161` | 54 → 55 | 0 → 0 | `[LOSS]` → `[LOSS]` | valid → valid |

Ground truth is `[LOSS]` for all three. Returned tokens with final probability
`< 1e-4` fell from 7/4/0 to 0/0/0. Every window now stops via
`adaptive_entropy_stability` on a certified fixed point.

Suite-level, three windows: grammar-valid canvases 1/3 → 3/3; pooled build-order
F1 0.2963 → 0.4984 (precision 0.7059 → 0.6320, recall 0.1875 → 0.4115); mean
per-window F1 0.2087 → 0.4909.

Cost: 154 → 164 total forward passes (+6.5%). Per-pass cost is unchanged — the
repair adds no model call anywhere, and the optional diagnostic final-logit pass
remains the only extra call the sampler can make.

## Not fixed by this repair

The remaining errors are model quality, not sampler mechanics. Window `t0` now
returns a well-formed `[WIN]` where ground truth is `[LOSS]` — a wrong but
legitimate prediction. Build-order recall remains low. Precision fell slightly
because previously-invalid canvases contributed no predictions at all.
