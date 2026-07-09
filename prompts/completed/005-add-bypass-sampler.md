<objective>
Add a `--bypass-sampler` option to the visualization workflow exercised by `./tests/test_viz.py`.

When enabled, preserve the existing replay selection, fogging, decoding, labels, optional exports, and every visualization/output format, but replace iterative canvas sampling with exactly one model denoising forward pass. This mode is intended to expose a training-like one-pass prediction for diagnostic comparison without changing the default sampler-backed behavior.
</objective>

<context>
Work entirely within the `Thesis_ML` submodule. Read `./AGENTS.md`, `./tests/AGENTS.md`, `./CLAUDE.md`, and the applicable child `AGENTS.md` files before editing. Review `./tests/test_viz.py`, the visualization implementation it exercises, the evaluation harness, the sampler, the model forward contract, and the training loss/noising path.

The existing visualization pipeline is read-only and already owns figure generation, canvas text exports, and final-canvas logit JSON. Users need bypass mode to produce those same diagnostics from one denoising model call, while the normal CLI path must remain unchanged.
</context>

<research>
1. Trace the current CLI argument flow through visualization evaluation to `sample_canvas` and its returned final logits.
2. Trace how training constructs a noised canvas, denoising step/noise level, attention inputs, masks, and model call.
3. Identify the narrowest owning boundary for a reusable one-pass prediction helper. Do not put production behavior in a pytest test module merely because `./tests/test_viz.py` is the requested regression location.
4. Verify the predicted canvas grammar and final logits expected by the existing decode/export/render pipeline.

Thoroughly analyze the semantic differences between a single training-style denoising pass and the sampler's iterative loop before implementing. Reuse existing masking, model-input, and canvas-update helpers where available.
</research>

<requirements>
1. Add a boolean CLI flag named `--bypass-sampler`, defaulting to false.
2. Thread the flag explicitly through the existing visualization entry points without changing existing callers or defaults.
3. When false, retain the current iterative sampler behavior exactly.
4. When true, perform exactly one forward denoising model call for each selected example instead of invoking the iterative sampler.
5. Make the one-pass input training-like: use the repository's established discrete-diffusion/noising and mask semantics rather than inventing a parallel corruption convention. Do not expose ground-truth canvas content to the model except as allowed by the established training corruption process.
6. Convert the one-pass logits into a predicted canvas using the established masked-position update semantics, while preserving clamped/non-predicted positions and producing logits in the same shape/meaning expected by optional JSON export.
7. Preserve all downstream behavior: decoding, count-difference heatmaps, first-appearance timelines, PNG/SVG/PDF output, `--txt`, `--json`, selection limits, seeds/determinism, device handling, and read-only artifact confinement.
8. Do not add absolute game time, frame numbers, `game_loop`, timestamps, or timestamp-derived values to model inputs.
9. Keep the implementation architecture-aware: reuse existing helpers and keep sampler bypass selection at the visualization/evaluation orchestration boundary. Avoid duplicating the visualization pipeline.
10. Add focused regression coverage in `./tests/test_viz.py` (and another focused test module only if ownership requires it) proving CLI threading, default compatibility, sampler bypass, and exactly one model forward call.
</requirements>

<implementation>
For maximum efficiency, inspect independent relevant symbols in parallel when possible. After receiving retrieval or test results, assess whether they prove the required behavior before proceeding.

Keep public signatures backward compatible by using defaulted keyword arguments. If the existing evaluation harness is shared by non-visual evaluation, avoid changing its default behavior. Prefer dependency injection or a small owned prediction strategy/helper over branching throughout rendering code.

Explain important implementation choices in concise code comments only where the one-pass diffusion semantics would otherwise be unclear. Do not weaken canvas validation or existing read-only safeguards.
</implementation>

<output>
Modify the minimum cohesive set of files under the submodule, expected to include:
- `./src/thesis_ml/viz/diagnostics.py` - CLI wiring and visualization orchestration.
- The appropriate existing inference/evaluation module if a reusable one-pass helper belongs there.
- `./tests/test_viz.py` - focused regression tests for bypass mode.
- The nearest owning `AGENTS.md` only if the durable behavior or ownership contract changes.

Keep this prompt under `./prompts/005-add-bypass-sampler.md`; after successful implementation and verification, move it to `./prompts/completed/005-add-bypass-sampler.md` if that matches the repository's established completed-prompt workflow.
</output>

<verification>
Before declaring complete:
1. Confirm `./.venv/Scripts/python.exe` exists and run all Python commands through it.
2. Run focused visualization and affected inference/evaluation tests.
3. Add an assertion or instrumented test proving bypass mode makes exactly one model forward call and does not call the iterative sampler.
4. Verify the same rendered/exported artifact categories are produced in bypass mode.
5. Run the broader relevant test set if shared inference/evaluation behavior changed.
6. Run `git diff --check` and review the final diff for accidental changes.
7. Perform the required DOX pass; update durable docs only if contracts changed, and state why unchanged docs did not need edits.
</verification>

<success_criteria>
- `--bypass-sampler` is accepted by the visualization CLI and defaults off.
- Default visualization behavior is unchanged.
- Bypass mode uses one and only one model denoising forward pass per example, with training-aligned corruption/masking semantics.
- Bypass mode never invokes iterative canvas sampling.
- Existing figures and optional text/JSON diagnostics work in both modes.
- Focused tests pass through the submodule virtual environment.
- The implementation preserves model-input and read-only architecture contracts.
</success_criteria>
