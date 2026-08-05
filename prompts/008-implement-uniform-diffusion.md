<objective>
Migrate the complete `Thesis_ML` model, training, and inference stack from
absorbing `[MASK]` diffusion to uniform-state multinomial diffusion, and make
uniform diffusion the production default.

Implement the accepted DiffusionGemma-derived mechanics end to end: linear
uniform corruption, clean-state prediction over all valid canvas positions,
expected-embedding self-conditioning, a dense Gemma 4-style GeGLU/sandwich-
RMSNorm backbone, and nonmonotonic entropy-bounded sampling with full uniform
renoising. Preserve absorbing diffusion as a coherent config-selected ablation.

This is intentionally checkpoint-incompatible. The end goal is a scientifically
coherent from-scratch pretraining system whose full output canvas can revise any
eligible position during denoising, while retaining the project's clamped input,
single-canvas generation, dense full-bidirectional attention, and structured SC2
target grammar.
</objective>

<context>
Work from the `Thesis_ML` repository root.

Read these authorities in full before editing:
- `./CLAUDE.md`
- `./SPEC.md`, especially sections 2, 3, 4, 7, 8, 9, 11, and 14
- `./research/diffusiongemma-uniform-migration.md`
- `./AGENTS.md` and every child `AGENTS.md` governing files you inspect or edit

The source and tests still implement the retired absorbing-only architecture.
The documents above have already been updated to the accepted target and win on
conflict. Do not revert or dilute those decisions to preserve old behavior.

The task matters because this model is trained from scratch for SC2 strategy
pretraining, not warm-started from an autoregressive language model. We adopt
DiffusionGemma's released diffusion mechanics while deliberately rejecting its
multi-canvas block autoregression, KV-cache prompt encoding, encoder/decoder
split, MoE, GQA, local/sliding attention, fixed 256-token canvas, multimodal
path, and latency-first design assumptions.

Relevant existing boundaries include:
- `./src/thesis_ml/config.py`
- `./config/default.yaml`
- `./configs/*.yaml`
- `./src/thesis_ml/train/corruption.py`
- `./src/thesis_ml/train/loop.py`
- `./src/thesis_ml/model/backbone.py`
- `./src/thesis_ml/model/embedding.py`
- `./src/thesis_ml/model/model.py`
- `./src/thesis_ml/model/loss.py`
- `./src/thesis_ml/inference/sampler.py`
- `./src/thesis_ml/pipeline/`
- `./src/thesis_ml/eval/`
- `./src/thesis_ml/viz/diagnostics.py`
- `./tests/`

Preserve the user's unrelated existing modification to `./TODO.md`.
</context>

<research>
Thoroughly inspect the existing ownership, callers, checkpoint paths, result
types, metric consumers, and tests before modifying code. Use semantic/AST-aware
repository retrieval when available, then verify consequential behavior against
the source itself.

For maximum efficiency, invoke independent read-only searches and file reads in
parallel. After receiving tool results, carefully reflect on their quality and
the complete blast radius before editing.

The dated research note contains pinned primary sources and exact released
behavior. If a subtle implementation detail is missing, consult the pinned
official Gemma or Hackable Diffusion source rather than inventing a new sampler
or importing an unrelated third-party implementation.
</research>

<settled_architecture>
These decisions are final:

1. `diffusion.process: uniform` is the production default.
2. `diffusion.process: absorbing` remains the only masked-diffusion ablation.
3. The input region remains clamped, is never noised, and never receives loss.
4. One full canvas is denoised jointly with full bidirectional dense MHA.
5. Uniform noising, its terminal prior, and uniform renoising sample from every
   vocabulary ID except `[MASK]`. There are no position-dependent token masks.
6. Ground truth always places `[WIN]` or `[LOSS]` at canvas position zero, but
   sampling gives that position no special treatment. Delete `outcome_last`.
7. Uniform training uses unweighted `x0` cross-entropy over every valid target
   position, including semantic `[PAD]`; only batch-shape padding is excluded.
8. Uniform training does not score only corruption-branch positions and does not
   apply inverse-`t` weights.
9. Absorbing ablation training retains corrupted-position-only inverse-`t` loss.
10. Sample one global `t ~ Uniform(0,1)` per example. Intentional exact-`t=1`
    oversampling remains configurable but defaults to `0.0` in every baseline.
11. The backbone uses dense GeGLU and sandwich RMSNorm while retaining current
    QK norm, dense MHA, Flash/memory-efficient SDPA policy, and Llama 3.1-style
    frequency-scaled RoPE.
12. Self-conditioning uses the shared token embedding table and the released
    RMSNorm -> dense GeGLU -> residual add -> scale-less RMSNorm path.
13. Uniform sampling follows DiffusionGemma's categorical entropy-bounded
    sampler with full renoising. Do not preserve the current monotonic commit
    state machine.
14. `sampler.max_steps=64` is a hard ceiling, not a required number of steps.
    There is no minimum step count.
15. Default sampler values remain temperature `0.8 -> 0.4`, exponent `1.0`,
    entropy bound `0.1`, adaptive entropy threshold `0.005`, and two-pass argmax
    stability.
16. Confidence sharpening remains an optional logits-derived ablation but
    defaults to weight `0.0`.
17. Old checkpoints are incompatible and must fail closed before partial load.
</settled_architecture>

<implementation>
Implement the migration sequentially so each layer consumes a stable preceding
contract.

1. Configuration and identity
   - Replace the absorbing-specific `MaskScheduleConfig`/`mask_schedule` naming
     with a process-neutral schedule owned by `diffusion.schedule`.
   - `DiffusionConfig` must contain `process` and `schedule`.
   - Accept exactly `uniform` and `absorbing`; reject unknown values clearly.
   - Preserve the linear schedule fields, uniform time distribution, min/max,
     and experimental `t_one_fraction`, but set its default to `0.0`.
   - Update `SamplerConfig` to contain the required uniform EB settings:
     `max_steps`, temperature start/end/exponent, `entropy_bound`,
     `adaptive_stop`, `entropy_threshold`, and `stability_steps`.
   - Remove `confidence_threshold`, `min_commit_per_step`, and `outcome_last`
     from dataclasses, YAML, validation, profiles, consumers, and tests.
   - Set canonical defaults to uniform diffusion, enabled self-conditioning,
     confidence-loss weight `0.0`, max 64 passes, linear `0.8 -> 0.4`
     temperature, entropy bound `0.1`, and dual-condition adaptive stopping.
   - Update every existing production profile to the new schema. Do not leave a
     partial old `mask_schedule` block that silently inherits conflicting data.
   - Introduce one stable architecture identity, for example
     `uniform-gemma4-dense-v1`, and persist both it and the configured diffusion
     process in checkpoints and finished exports. Resume, warm-start, sampling,
     diagnostics, and evaluation must reject missing or mismatched identities
     before calling `load_state_dict`.

2. Process-compatible corruption
   - Refactor `corrupt_batch` rather than creating a parallel trainer path.
   - Sample one `t` per example using the existing reproducible generator and
     epoch reseeding contract.
   - Uniform mode: independently select a corruption branch with probability
     `t`, draw replacement IDs uniformly from `[1, vocab_size)` because
     `[MASK]` is ID zero, and select target vs replacement with `torch.where`.
     A replacement equal to the target is valid; distinguish the Bernoulli
     corruption branch from actual token inequality in result fields/metrics.
   - Absorbing mode: preserve independent `[MASK]` replacement at probability
     `t` and inverse-`t` weights.
   - Give corruption the authoritative vocabulary size through an existing
     model/vocabulary ownership boundary; do not infer it from IDs observed in a
     batch.
   - Rename misleading mask-only result fields and metric names to
     process-neutral corruption/noise terminology, updating every consumer.
   - Keep explicit-`t` diagnostic and validation calls deterministic and free
     from terminal oversampling.

3. Loss and training loop
   - Uniform mode sets the scored mask to `batch.canvas_loss_mask` and uses unit
     position weights. Semantic `[PAD]` positions inside that mask contribute to
     pretraining loss with weight 1.0.
   - Absorbing mode uses `corruption.corrupted_positions &
     batch.canvas_loss_mask` and inverse-`t` weights.
   - Fine-tuning class weights remain supported on top of the process-compatible
     base mask/weights. Pretraining must still reject fine-tuning-only class
     weights.
   - Keep per-class, future-distance, time-bucket, perspective, and epoch
     reporting, but make descriptions and calculations truthful for clean-state
     CE rather than universally calling it masked CE.
   - Apply optional confidence loss to the same process-owned scored positions;
     leave it disabled by default.
   - Preserve EMA, generator reseeding, checkpoint/resume behavior, accumulation,
     memory logging, and pipeline orchestration.

4. Dense Gemma 4-style backbone
   - Replace `SwiGLU` with dense GeGLU:
     `down(F.gelu(gate(x), approximate="tanh") * up(x))`.
   - Give every transformer block pre-attention and post-attention RMSNorm plus
     pre-FFN and post-FFN RMSNorm. Normalize each branch output before its
     residual addition.
   - Preserve dense vanilla MHA, existing QK norm placement, full bidirectional
     attention, scaled RoPE, fused SDPA constraints, gradient checkpointing, and
     config-owned widths/depths.
   - Do not introduce MoE, GQA, local/sliding attention, causal masks, KV cache,
     prompt encoding, cross-attention, or a second stack.
   - Update initialization and parameter-count tests for the new mandatory
     architecture rather than adding a compatibility mode for retired weights.

5. Exact self-conditioning path
   - Replace the independent `vocab_size -> d_model` projection with probabilities
     multiplied by the shared token embedding weight, producing `[B,L,D]`
     expected embeddings.
   - Stop gradients through the estimate pass and expected embedding.
   - Apply self-conditioning RMSNorm -> dense GeGLU -> add to current canvas
     token embedding -> scale-less RMSNorm. Never apply it to the clamped input.
   - When self-conditioning is enabled but unavailable for a row, use a zero
     `[L,D]` signal through the same interface. Training chooses estimate vs zero
     independently per example with probability `self_cond_prob=0.5`.
   - Inference's first pass uses zero conditioning. Each later pass reuses the
     preceding pass's expected embedding without another model call.
   - At inference, derive the expected embedding from the same temperature-shaped
     probability distribution used for candidate sampling and entropy.
   - Keep `model.self_conditioning=false` as an explicit architecture ablation,
     but do not preserve old checkpoint compatibility.

6. Uniform entropy-bounded sampler
   - Replace the current monotonic implementation. The exact per-row uniform
     step is:
       1. Run one denoiser pass over the current canvas.
       2. Divide logits by the scheduled temperature.
       3. Exclude `[MASK]` from probabilities/candidates by assigning it zero
          probability; otherwise do not mask logits by position or grammar.
       4. Sample one categorical candidate independently at every eligible
          position using an explicit `torch.Generator`-compatible operation.
       5. Compute entropy over the same allowed state distribution.
       6. Sort eligible positions by ascending entropy.
       7. Accept exactly the sorted prefix satisfying
          `cumsum(sorted_entropy) - sorted_entropy <= entropy_bound`.
       8. Draw fresh independent uniform non-`[MASK]` states for all eligible
          positions and use the categorical candidate only where accepted.
   - Recompute acceptance from scratch every pass. Do not OR it with a previous
     mask. A previously accepted position that is not accepted now is renoised.
   - The scheduled noise proportions may be retained for trace/temperature
     calculation, but do not invent a reverse posterior: the released sampler's
     acceptance/renoising step ignores current/target noise proportions.
   - Adaptive stopping is per row and requires BOTH mean eligible-position
     entropy below `entropy_threshold` and identical argmax predictions across
     `stability_steps=2` consecutive passes. Do not compare sampled canvases in
     place of argmax predictions.
   - Freeze done rows while unfinished rows continue. Stop all work when every
     row is done or the 64-pass ceiling is reached.
   - If the ceiling is reached, return the last sampled canvas. Do not silently
     force argmax, apply grammar repair, or run an unreported extra pass.
   - Normal sampling performs exactly one model call per executed denoising pass.
     `return_final_logits` may retain its explicit diagnostics-only extra pass.
   - Infill/reveal diagnostics clamp revealed target positions. Exclude clamped
     and batch-invalid positions from candidate sampling, renoising, entropy
     means, EB sorting, and stability checks.
   - Replace persistent `committed` result/trace semantics with accurate fields
     for transient acceptance, renoising/unaccept events, entropy, argmax
     stability, temperature, done rows, stop reasons, and actual step count.

7. Absorbing EB ablation
   - Initialize eligible positions as `[MASK]`.
   - Temperature-shape logits, sample categorical candidates, and apply the same
     correct EB prefix formula only among positions still masked.
   - Accepted candidates become tokens permanently; nonaccepted positions remain
     `[MASK]`. Never uniformly renoise this process.
   - Stop when every eligible position is unmasked. Keep this path isolated
     behind `diffusion.process=absorbing` so it cannot contaminate uniform mode.

8. Outcome, grammar, and downstream consumers
   - Delete all outcome-last branching and its dedicated test module if it has
     no remaining purpose.
   - Do not constrain `[WIN]`/`[LOSS]` logits to position zero or prohibit them at
     other positions. The only sampler-level excluded state in uniform mode is
     `[MASK]`.
   - Keep target construction placing the perspective-relative outcome at index
     zero and keep grammar validation rejecting malformed final canvases.
   - Update evaluation, fine-tuning reports, diagnostics, visualization, result
     serialization, and CLI/API names from mask/commit semantics to process-
     neutral noise/acceptance semantics where behavior changed.
   - `--bypass-sampler` must perform one pass from the configured `t=1` prior:
     uniform random non-`[MASK]` states by default, all `[MASK]` for absorbing.
   - Preserve clamped input, feature-statistics identity, dynamic masks, timing
     recovery, and all target serialization rules.

9. Retired checkpoints
   - First make all loaders fail clearly on old metadata/architecture/process.
   - After implementation and the full verification suite pass, enumerate and
     verify that the resolved deletion target is exactly the repository-local
     `./checkpoints/` directory.
   - Delete the retired contents under `./checkpoints/` as explicitly authorized
     by the owner. Keep the directory itself if runtime code expects it.
   - Do not delete remote/S3 objects, dataset artifacts, `./tests/output/`, or
     anything outside the verified repository-local checkpoint directory.
   - Report what was removed and that it is not recoverable from this worktree.

10. DOX closeout
   - Re-read `./SPEC.md` and the complete applicable `AGENTS.md` chain after the
     code works.
   - Update only documentation details made inaccurate by concrete symbol/config
     naming during implementation. Do not reopen settled architecture choices.
   - Remove stale code comments, docstrings, and active documentation that still
     describe uniform mode as masking, monotonic commitment, outcome-last,
     confidence thresholds, minimum commits, SwiGLU, or a 48-step ceiling.
   - Historical completed prompts and dated diagnostics remain historical; do
     not rewrite them merely to hide prior behavior.
</implementation>

<constraints>
- Never modify the clamped input during canvas corruption or sampling.
- Never add explicit timestep conditioning unless `SPEC.md` is changed by the
  owner; the accepted model predicts `x0` without a time embedding.
- Never introduce autoregressive, block-autoregressive, semi-autoregressive,
  encoder-decoder, cross-attention, MoE, GQA, local/sliding-attention, copy-head,
  or classification-head machinery.
- Never hardcode grammar by token position. Ground-truth ordering and CE teach
  the position-zero outcome convention.
- `[MASK]` must never be drawn by uniform corruption, uniform prior sampling,
  uniform renoising, or uniform categorical candidates.
- Use the repository virtual environment for every Python command. First verify
  `./.venv/Scripts/python.exe` exists, then invoke Python only through it.
- Do not run expensive model training, replay processing, or broad data rebuilds.
  Use unit tests and bounded synthetic smoke checks.
- Preserve unrelated user changes, especially `./TODO.md`.
- Extend existing helpers and ownership boundaries; do not create parallel
  training, model, sampler, config, or checkpoint stacks.
</constraints>

<validation>
Add or rewrite focused tests that prove, at minimum:

1. Config defaults and validation for `uniform`/`absorbing`, zero terminal
   oversampling, enabled self-conditioning, zero confidence weight, exact EB
   defaults, max 64, and removal of retired sampler fields.
2. Uniform `t=0` is clean; uniform `t=1` is sampled only from non-`[MASK]`
   states; replacement may equal the target; empirical branch frequency tracks
   `t`; fixed seeds reproduce outputs.
3. Absorbing corruption retains the expected `[MASK]` and inverse-`t` behavior.
4. Uniform loss scores every valid canvas position including semantic `[PAD]`,
   excludes only batch padding, and does not depend on the corruption branch.
5. Absorbing loss scores only corrupted eligible positions with inverse-`t`
   weights.
6. GeGLU uses tanh-approximate GELU, sandwich RMSNorm is applied on both
   residual branches, shapes remain correct, parameter-count expectations are
   updated, and forbidden architecture variants are absent.
7. Self-conditioning uses the shared embedding table, supplies zero signal on
   the first pass, applies the per-example 0.5 training gate with stopped
   gradients, touches canvas only, and adds no inference model call.
8. A hand-computed entropy example proves the exact
   `cumsum(entropy)-entropy <= gamma` prefix. Do not accept the retired plain
   cumulative-sum implementation.
9. Seeded categorical candidate sampling and uniform renoising are reproducible.
10. Uniform acceptance is nonmonotonic: a previously accepted position can be
    unaccepted and renoised on a later step.
11. Uniform candidates/noise never contain `[MASK]`, while `[WIN]`/`[LOSS]` and
    other allowed states are not position-restricted.
12. Adaptive stopping does not fire on entropy alone or stability alone, fires
    when both hold, freezes completed batch rows, and never exceeds 64 passes.
13. Absorbing EB remains monotonic and process-isolated.
14. Infill revealed positions remain unchanged and are excluded from EB/stopping.
15. Old or cross-process checkpoints fail before partial weight loading.
16. Bypass diagnostics use the selected process's terminal prior.

Run focused tests while iterating, including the real affected modules under
`./tests/`. Then run:

`./.venv/Scripts/python.exe -m pytest -q`

Also run `git diff --check` and inspect `git status --short`. Distinguish focused
test results from final-suite status. If the full suite fails, report exact
failures and do not call the migration complete.
</validation>

<output>
Modify the existing files under these relative boundaries as required:
- `./src/thesis_ml/config.py`
- `./config/default.yaml`
- `./configs/*.yaml`
- `./src/thesis_ml/train/`
- `./src/thesis_ml/model/`
- `./src/thesis_ml/inference/`
- `./src/thesis_ml/pipeline/`
- `./src/thesis_ml/eval/`
- `./src/thesis_ml/viz/`
- `./tests/`
- Applicable active `./AGENTS.md` and operating documentation only when concrete
  implementation names require reconciliation

Do not create a second implementation tree. Delete obsolete outcome-last tests
and retired code paths instead of leaving unreachable compatibility branches.
</output>

<success_criteria>
- Uniform multinomial diffusion is the default and works end to end from
  corruption through training, self-conditioning, sampling, diagnostics, and
  evaluation.
- Absorbing diffusion remains a coherent, tested config-selected ablation.
- Uniform training uses all-valid-position unweighted `x0` CE with no deliberate
  terminal oversampling and no default confidence sharpening.
- The model uses dense GeGLU, sandwich RMSNorm, full dense bidirectional MHA,
  QK norm, and existing scaled RoPE without forbidden DiffusionGemma topology.
- Self-conditioning is tied to the shared embedding table and matches the
  released two-pass/reuse mechanics.
- Uniform sampling implements categorical candidates, exact EB prefix math,
  full nonaccepted-position renoising, genuine revision, dual-condition adaptive
  stopping, and a 64-pass hard ceiling with no minimum.
- No outcome-last or other position-dependent output-token constraint remains.
- Checkpoint identity fails closed across retired architecture or process
  boundaries, and verified repository-local retired checkpoint contents are
  removed only after tests pass.
- Focused tests, the complete pytest suite, and `git diff --check` pass.
- Active source, tests, configs, and DOX agree with `SPEC.md`; unrelated user
  changes remain intact.
</success_criteria>
