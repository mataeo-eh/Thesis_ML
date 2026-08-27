<objective>
Diagnose and repair a known defect in the production uniform-state sampler.

The defect is established at the behavioral boundary: the sampler can return a
final canvas containing tokens that a subsequent denoiser pass over that same
canvas strongly prefers to revise. This violates the intended practical promise
of nonmonotonic uniform-state diffusion—that every mutable position remains
revisable during denoising—and has produced isolated nonsensical tokens and
malformed output grammar in real held-out inference.

Do not assume the root cause supplied by prior discussion. Reproduce it, trace
the complete state transition and stopping behavior, compare the implementation
with current primary literature and released reference code, identify the actual
defect, and implement the best evidence-supported repair. The repair must remain
scientifically coherent with modern uniform-state discrete diffusion and
entropy-bounded sampling.
</objective>

<execution_gate>
This prompt is an implementation task when explicitly executed. Its presence in
`prompts/` does not authorize implementation by the agent that merely creates
or reviews it.
</execution_gate>

<context>
Work from the `Thesis_ML` repository root.

Read these authorities in full before editing:

- `./AGENTS.md`, `./CLAUDE.md`, and every child `AGENTS.md` governing files
  inspected or changed
- `./SPEC.md`, especially the target grammar and sampling sections
- `./Model_Architecture/UPDATE_PROMPT.md`
- `./Model_Architecture/MODEL_ARCHITECTURE.md`
- `./research/diffusiongemma-uniform-migration.md`
- `./src/thesis_ml/inference/sampler.py`
- `./src/thesis_ml/inference/decode.py`
- `./src/thesis_ml/eval/harness.py`
- `./src/thesis_ml/viz/diagnostics.py`
- `./tests/test_sampler.py`
- `./tests/test_diffusion_integration.py`
- `./Model_Inference_Tests/AGENTS.md`
- `./Model_Inference_Tests/README.md`

The current production process is uniform-state diffusion with nonmonotonic
entropy-bounded sampling. Canvas position 0 is the only clamped output position
and contains `[BOS]`. The perspective-relative `[WIN]`/`[LOSS]` target is at
position 1 and must be treated jointly with every other mutable canvas position.

The retired `sampler.outcome_last` rule must not exist in configuration, source,
tests, or active documentation. Do not restore it, emulate it, or introduce any
other position-1 denoising-order exception.
</context>

<known_evidence>
The following observations are evidence to reproduce and explain, not a supplied
root-cause verdict.

The affected held-out run is:

- config: `configs/smallTrainingTestV3.yaml`
- checkpoint: `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt`
- weights: EMA
- device used by the recorded run: CUDA
- seed: `20260826`
- output noise: exact `t=1.0`
- sampler: iterative `thesis_ml.inference.sampler.sample_canvas`
- artifact root:
  `Model_Inference_Tests/output/smallTrainingTestV3-epoch-0033__2026-Aug-26_03-07PM/`

In `canvas_reconstruction_figures/canvas_comparison.csv`, two held-out canvases
returned ordinary content tokens at the outcome slot:

- window `match_4745748_game_state-6721e9982e13_p1_t0`: returned `nuke` at
  sequence index 1 while ground truth was `[LOSS]`
- window `match_4745748_game_state-6721e9982e13_p1_t99`: returned `interceptor`
  at sequence index 1 while ground truth was `[LOSS]`

The separate post-sampling logits recorded in `canvas_logits.json` preferred:

- `[WIN]` with probability about `0.7783` on the final canvas containing `nuke`
- `[WIN]` with probability about `0.5569` on the final canvas containing
  `interceptor`

The returned token was outside the exported final top 10 in both cases. Similar
isolated returned/final-preference discrepancies include `liberator`,
`swarmhost`, `swarmhostburrowed`, `extractorrich`, and `nyduscanal`. Across the
three 4096-token canvases, the returned canvas and post-sampling argmax differed
at only a small number of mutable positions, so this is a localized but real
sampler-finalization defect rather than an explanation for every semantic model
error.

The active code is expected to have no persistent uniform-mode commitment mask:
acceptance is recomputed every pass and any mutable position may be renoised or
revised. Nevertheless, the observable returned canvas is not always consistent
with what the denoiser says after sampling concludes.
</known_evidence>

<research_requirement>
Before choosing a repair, refresh the literature and implementation comparison
against current primary sources. Record access date and an immutable commit or
release identifier for mutable repositories.

At minimum inspect:

- DiffusionGemma Technical Report, arXiv:2608.00146:
  <https://arxiv.org/abs/2608.00146>
- Google DeepMind's released Gemma diffusion implementation, especially the
  sampler, early-stopping, temperature-shaping, and selection code:
  <https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion>
- Google DiffusionGemma model documentation and model card:
  <https://ai.google.dev/gemma/docs/diffusiongemma>
  <https://ai.google.dev/gemma/docs/diffusiongemma/model_card>
- Entropy-Bounded Sampler paper, arXiv:2505.24857:
  <https://arxiv.org/abs/2505.24857>
- Hackable Diffusion released implementation where relevant:
  <https://github.com/google/hackable_diffusion>

Use primary paper/code as authority over summaries. Reconcile apparent
differences among the technical report, Google repository, public documentation,
and this project rather than copying one snippet mechanically. DiffusionGemma's
256-token block-autoregressive deployment setting differs from this project's
single long structured SC2 canvas; adopt the sampling principle, not unrelated
architecture.

Update `research/diffusiongemma-uniform-migration.md` if its dated snapshot is
now incomplete or inaccurate in a way material to the repair. Clearly separate
published/released behavior, inference from source, and project decisions.
</research_requirement>

<investigation>
Establish the root cause with trace evidence before editing the sampler.

1. Reproduce the discrepancy on the named real checkpoint and recorded held-out
   windows on an actually visible CUDA device. Do not substitute a CPU or
   synthetic-only result for the real-pipeline reproduction.
2. Preserve and inspect the complete per-step trace for the affected positions:
   current token, temperature-shaped distribution, argmax, categorical
   candidate, candidate probability/rank, entropy, acceptance, renoising,
   self-conditioning signal ownership, row done state, and stop reason.
3. Determine whether each failure exits through adaptive stopping, the hard
   step ceiling, or another path.
4. Trace the exact ordering of model evaluation, categorical draw, acceptance,
   canvas mutation, self-conditioning update, stability bookkeeping, stopping,
   row freezing, optional final-logit evaluation, and returned result.
5. Confirm from live source and tests that position 1 receives no special
   denoising order and that `outcome_last` is absent.
6. Verify whether the behavior is caused by the sampler, the diagnostics
   exporter, stale cached tensors, self-conditioning state, probability masking,
   batch-row freezing, or some interaction among them. Do not infer the answer
   merely from the existence of a final extra forward pass.
7. Compare the project's exact control flow with the pinned primary reference
   implementation, including what state its stopping rule compares and what
   canvas it returns.

Keep the reproduction read-only with respect to the checkpoint, config,
manifests, and replay artifacts. Any temporary traces must be size-bounded and
must not expose workstation-absolute paths.
</investigation>

<implementation_requirements>
After proving the defect, implement the repair that best matches the primary
literature and this project's structured single-canvas setting.

The solution is deliberately not prescribed. It may change transition ordering,
stopping-state validation, candidate acceptance, finalization, or another
mechanism if the evidence supports that choice. It must satisfy all of the
following:

- Uniform-mode sampling remains nonmonotonic: no persistent commitment mask and
  no irreversible accepted-token state.
- Every non-clamped eligible position remains revisable under the uniform
  process until that row legitimately finishes.
- `[BOS]` alone remains clamped. Position 1 is denoised jointly and never delayed
  or protected by an outcome-specific rule.
- The fix must address the sampler's returned result, not cosmetically rewrite
  CSV/logit diagnostics after sampling.
- The result contract must make the relationship among the returned canvas,
  the last denoiser distribution, self-conditioning state, and stop decision
  explicit and internally consistent.
- Both adaptive-stop and hard-ceiling exits must have deliberate, tested
  semantics.
- Done rows must remain frozen while unfinished rows continue.
- Infill-revealed positions and batch-invalid positions remain clamped/excluded.
- Seeded categorical sampling and uniform renoising remain reproducible.
- Absorbing-mode behavior remains process-compatible and isolated.
- Do not add a grammar-specific patch, forced argmax projection, extra normal
  model call, or new hyperparameter merely because it makes the example pass.
  Any such choice must be justified against primary literature, measured for
  quality/cost, and integrated as the coherent sampler contract.
- Preserve the configured 64-pass ceiling unless research and measured evidence
  demonstrate that changing it is part of the actual correction. Do not hide a
  defect by merely increasing the limit.

If the research-supported repair intentionally changes a previously settled
sampler semantic, update `SPEC.md`, configuration validation/defaults, all
affected source/tests, and the exact architecture reference together. Do not
leave source, tests, and documentation describing different algorithms.
</implementation_requirements>

<verification>
Add focused regression coverage that would fail on the known defect and prove
the chosen repaired contract.

At minimum cover:

- a mutable position can be revised after appearing accepted on an earlier
  uniform pass
- no outcome-position special case exists
- a constructed final-step disagreement between a sampled candidate and the
  denoiser's relevant clean-state preference is resolved according to the newly
  documented sampler contract
- adaptive-stop exit semantics
- hard-step-ceiling exit semantics
- mixed done/unfinished batch rows
- categorical RNG reproducibility
- self-conditioning call count and state alignment
- frozen-input-KV parity/reuse
- revealed-position clamping
- absorbing-mode non-regression
- normal inference versus optional diagnostic-final-logit call counts

Then rerun the named real held-out reproduction and report, per window:

- stop reason and executed passes
- number of returned-canvas versus relevant final/verification argmax
  disagreements
- outcome-slot returned token and model probability mass on `[WIN]+[LOSS]`
- grammar validity
- isolated out-of-race/nonsensical token count
- build-order and timestep-count metrics before versus after the repair using
  the same checkpoint, examples, and seed
- runtime and added/removed forward-pass cost

Because this directly tests model inference behavior, sampler regression
evidence may extend the existing `Model_Inference_Tests` suite, provided it obeys
that directory's contract and writes only beneath its ignored `output/` tree.

Run focused sampler/model/eval/viz tests, the full package suite, and
`git diff --check`. Run Python only through the confirmed project virtual
environment. CUDA claims require visible CUDA and the actual local GPU.
</verification>

<dox_closeout>
This is a model-facing sampling change. Use
`Model_Architecture/UPDATE_PROMPT.md`, update every affected section of
`Model_Architecture/MODEL_ARCHITECTURE.md`, update
`MODEL_ARCHITECTURE_DIAGRAM.mmd`, and regenerate the SVG and PNG in the same
change. Update `SPEC.md`, `RUN.md`, `EVAL.md`, relevant `AGENTS.md` files,
research notes, and tests wherever the implemented contract changed.

Remove superseded statements rather than retaining contradictory historical
descriptions. Git owns history.
</dox_closeout>

<deliverables>
- Root-cause explanation backed by a minimal trace and primary-source comparison.
- Implemented sampler repair with no `outcome_last` behavior.
- Focused regressions plus full-suite verification.
- Same-checkpoint, same-window, same-seed before/after real-GPU evidence.
- Synchronized source, configuration, tests, research, SPEC, architecture DOX,
  Mermaid, SVG, and PNG.
- A concise final report distinguishing the proven defect, the implemented
  correction, remaining model-quality errors, and any intentional deviation
  from DiffusionGemma with rationale.
</deliverables>
