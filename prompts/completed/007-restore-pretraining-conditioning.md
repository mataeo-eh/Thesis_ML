---
prompt_number: "007"
status: completed
completed_at: "2026-08-05T11:52:50-05:00"
execution_strategy: single-delegated
verification:
  focused_tests: "61 passed"
  full_suite: "169 passed"
  smoke: "8-step CPU synthetic smoke passed"
---

<objective>
Audit two architecture assumptions against the live Thesis_ML implementation, then implement only the corrections whose assumptions are confirmed:

1. Determine whether pretraining currently omits the clamped input region. If it does, restore a clamped pretraining input containing the full self sequence plus the enemy sequence corrupted by per-example percentage-based token omission.
2. Determine whether static per-entity information is currently injected through independent additive projections/embeddings. If it is, replace that path with the exact learned residual joint embedding specified below.

The end goal is to let pretraining condition on both self/enemy identity and static entity state while preserving an unobstructed entity-type representation. This is an owner-directed architecture correction: it intentionally supersedes only the current contracts that say pretraining is canvas-only or that contextual fields are added independently. Preserve all unrelated settled architecture.
</objective>

<context>
This repository trains a masked discrete-diffusion model to predict StarCraft II opponent strategy. A canvas-only pretraining sequence cannot learn the intended conditional distinction between the player's state and the corrupted enemy state. Likewise, independently adding type, position, stats, and team embeddings cannot express the required joint type-feature interactions as directly as the requested concatenative residual branch.

Work from `./Thesis_ML` as the repository root. Before editing, read the current files rather than relying on prior prompts or memory:

- @AGENTS.md
- @CLAUDE.md
- @SPEC.md
- @src/thesis_ml/AGENTS.md
- @src/thesis_ml/data/AGENTS.md
- @src/thesis_ml/model/AGENTS.md
- @src/thesis_ml/pipeline/AGENTS.md
- @src/thesis_ml/train/AGENTS.md
- @tests/AGENTS.md
- @src/thesis_ml/data/dataset.py
- @src/thesis_ml/data/collate.py
- @src/thesis_ml/data/windowing.py
- @src/thesis_ml/model/embedding.py
- @src/thesis_ml/model/model.py
- @src/thesis_ml/config.py
- @src/thesis_ml/pipeline/train_pipeline.py
- @config/default.yaml
- Relevant focused tests under @tests/

Start with `!git status --short`. Preserve unrelated worktree changes, including any pre-existing change to `./TODO.md`.
</context>

<research>
Thoroughly analyze the live data flow before changing it. For maximum efficiency, perform independent searches and file inspections in parallel where practical. Reuse existing serializers, fog logic, split ownership, artifact formats, feature containers, model APIs, and tests instead of creating parallel implementations.

1. Trace pretraining examples from replay splitting and preprocessing through dataset serving, collation, model embedding, attention masks, corruption, loss masks, training, validation, sampling, and diagnostics.
2. Trace every static feature from `TokenRecord` into `InputFeatures` and block 0. Enumerate the continuous fields and prove whether entity type and static features are currently combined additively or jointly.
3. Record an evidence table for both assumptions with exact file paths, symbols, and observed behavior.
4. Apply this decision matrix:
   1. If an assumption is confirmed, implement its correction and all required downstream integration.
   2. If an assumption is false because the exact requested behavior already exists, do not rewrite that subsystem; add or strengthen only the missing regression proof.
   3. If implementation and durable contracts disagree, treat source and tests as evidence of current behavior, then update both implementation and contracts to the owner-directed target in this prompt.
5. After receiving retrieval or test results, carefully reflect on their quality and use source-level evidence before proceeding.
</research>

<requirements>
<clamped_pretraining_input>
When assumption 1 is confirmed, restore the pretraining input using the existing input grammar and ownership boundaries unless a source-backed incompatibility requires a cohesive shared helper:

1. Each pretraining example must contain a clamped input region and a noised canvas region in one flat bidirectional sequence.
2. The clamped input must contain every self content token and an enemy sequence subjected to one sampled omission rate per served example. Apply independent omission draws to enemy content tokens under that percentage, using the project's existing fog distribution and reproducibility conventions.
3. Omission means physical token omission. Never insert `[MASK]`, placeholders, count signals, or another representation for omitted enemy entities/upgrades.
4. Never omit self content because of fog. Preserve timestep delimiters and the current canonical per-timestep serialization grammar.
5. Keep persisted replay artifacts and manifests clean. Sample corruption while serving examples; do not bake a fog realization into preprocessing output.
6. The input is clamped: never diffusion-noise it and never include it in the denoising loss. The canvas remains the full enemy reconstruction/future target governed by the existing target grammar.
7. Ensure both perspective streams remain supported: p1-as-self/p2-as-enemy and p2-as-self/p1-as-enemy.
8. Restore all semantically necessary pretraining configuration, validation, class-label, metric, checkpoint/resume, inference, and diagnostics behavior. Do not leave stale claims that pretraining has zero input, no fog, zero input attention columns, or canvas position 0 at overall sequence position 0.
9. Reuse or generalize the existing input builder so pretraining and fine-tuning cannot silently drift into two incompatible fog/serialization implementations.
</clamped_pretraining_input>

<joint_static_embedding>
When assumption 2 is confirmed, replace the naive static-feature path with this exact strictly per-position computation for input entity tokens. Let `E` be the type embedding and `C` the continuous feature matrix:

1. Type path:
   1. Entity type ID enters only `nn.Embedding(vocab_size, d_model)` and yields `E` with shape `[B, L, d_model]`.
   2. Preserve `E` as an unobstructed identity residual. Do not project, gate, or normalize this residual path.
2. Static-feature path:
   1. Build a stable allowlisted continuous vector containing map position, health, energy, and the available unit stats owned by the current schema. Absolute game time, frame number, `game_loop`, timestamps, token/type ID, and timestamp-derived values are prohibited.
   2. Standardize each continuous field with a frozen mean and standard deviation computed exactly once from the training split only. Never compute statistics per batch and never use validation/test examples.
   3. Encode allegiance categorically as a raw scalar: self `+1`, enemy `-1`. Do not standardize it. Concatenate this scalar with the standardized continuous values as the feature-branch input; it must not enter the type embedding table.
   4. Apply `Linear(F, 32) -> ReLU -> Linear(32, 32) -> ReLU` independently at every sequence position to produce `H` with shape `[B, L, 32]`.
3. Joint residual mixer:
   1. Concatenate `[E, H]` along the final dimension. Concatenation is load-bearing because the mixer must represent interactions between entity type and static features; do not replace it with addition of independent embeddings.
   2. Apply `Linear(d_model + 32, d_model) -> GELU -> Linear(d_model, d_model)`.
   3. Zero-initialize both weight and bias of the final `Linear(d_model, d_model)`.
   4. Return `E + mixer(concat(E, H))`, shape `[B, L, d_model]`.
4. Strict separation and locality:
   1. Entity type IDs must never enter the feature MLP.
   2. Static values and allegiance must never enter the embedding lookup.
   3. Every operation in this layer must be position-wise. No attention, convolution, pooling, sequence reduction, or other cross-token mixing is allowed.
   4. Do not add dropout or trailing normalization. Block 0's pre-norm RMSNorm owns normalization.
   5. RoPE remains applied only to Q and K inside attention and is outside this embedding layer.
5. Preserve canvas behavior: canvas tokens have no static entity features. Keep the shared type embedding and existing canvas-only self-conditioning semantics without leaking input features into the canvas.
</joint_static_embedding>

<feature_statistics>
Implement a durable, reproducible statistics lifecycle at the existing data/pipeline ownership boundary:

1. Compute per-feature count, mean, and standard deviation from the selected training replay split only, after replay-level splitting and before model training.
2. Persist a versioned statistics artifact with an explicit ordered feature schema. Use config/storage abstractions rather than absolute paths so local and cloud runs behave consistently.
3. Generate or refresh the artifact only through an explicit preprocessing/statistics step. Normal train, resume, validation, evaluation, sampling, and diagnostics paths must load the frozen artifact and fail loudly with a clear actionable error if it is missing, malformed, non-finite, schema-incompatible, or inconsistent with the checkpoint/config. Never silently recompute it.
4. Define and test a deterministic zero-variance policy without weakening the missing-artifact failure rule.
5. Store enough statistics identity/schema metadata in checkpoints or their config metadata to prevent loading incompatible feature statistics on resume or inference.
</feature_statistics>
</requirements>

<implementation>
1. Add the smallest cohesive shared abstractions at their existing ownership boundaries; do not duplicate fog construction, feature ordering, standardization, or statistics loading.
2. Update configuration dataclasses, `./config/default.yaml`, and affected local profiles so behavior is config-owned where paths or preprocessing controls are involved. The requested architecture itself is not an optional alternative path unless an existing compatibility requirement demonstrably requires a migration switch.
3. Update dataset/collation types so feature tensors, masks, and allegiance values stay aligned with dynamically padded input tokens and move to the active device together.
4. Update attention-position and mask assumptions for the restored non-empty pretraining input while retaining full bidirectional attention and canvas-only loss.
5. Update manifests/cache stamps when changed semantics would otherwise allow stale processed artifacts to load as valid.
6. Maintain serialization and target grammar, shared vocabulary, QK-norm, self-conditioning, EMA, outcome-last sampling, and all unrelated settled behavior.
7. Keep all paths relative/configured and all Python verification routed through the confirmed `./.venv/Scripts/python.exe` shim.
8. Do not modify unrelated files or overwrite the existing `./TODO.md` change.
</implementation>

<mathematical_consistency>
Honor both the exact zero-output initialization and ordinary backpropagation. Under the specified architecture, zero-initializing the final mixer projection makes the initial joint-branch output exactly zero, which also makes gradients to upstream mixer layers and the feature MLP exactly zero on the first backward pass by the chain rule.

Do not hide this fact with a straight-through estimator, fake gradient hook, nonzero initialization, or altered residual design. Interpret the requested gradient/sensitivity checks as follows:

1. At initialization, output must equal the pure type embedding exactly; feature perturbations must therefore produce no output change.
2. On the first backward pass, verify nonzero gradients for the type embedding and final mixer projection, and verify the expected zero gradients for upstream feature/mixer layers.
3. After applying the first optimizer update, run a second forward/backward pass and verify nonzero gradients reach the feature MLP and upstream mixing MLP.
4. After the branch has been unlocked by training (or in a controlled test with a deliberately nonzero final projection), perturb one continuous feature with type IDs fixed and verify a measurable output change at only that sequence position.

Document this test interpretation concisely because it is the only mathematically consistent way to retain the load-bearing zero initialization and exact initial identity.
</mathematical_consistency>

<output>
Create or modify only the files proven necessary by the audit, centered on these relative paths:

- `./SPEC.md` — replace the superseded canvas-only pretraining and naive additive-input contracts with the restored clamped-pretraining and joint-embedding contracts; remove contradictions.
- `./AGENTS.md`, `./src/thesis_ml/AGENTS.md`, and the affected child `AGENTS.md` files — complete the required DOX pass and record stable ownership/workflow changes.
- `./src/thesis_ml/data/dataset.py` and `./src/thesis_ml/data/collate.py` — serve/collate aligned clamped pretraining input and static features.
- `./src/thesis_ml/data/windowing.py` and, if no existing cohesive owner exists, `./src/thesis_ml/data/feature_stats.py` — own training-split statistics generation, schema, persistence, and validation.
- `./src/thesis_ml/model/embedding.py` — implement the exact feature MLP and zero-initialized joint residual mixer.
- `./src/thesis_ml/model/model.py` — integrate the embedding without changing the transformer contract unnecessarily.
- `./src/thesis_ml/config.py`, `./config/default.yaml`, and affected `./configs/*.yaml` — configure and validate artifact paths/lifecycle and restored pretraining fog requirements.
- `./src/thesis_ml/pipeline/train_pipeline.py` and affected inference/evaluation/diagnostic consumers — wire frozen statistics and non-empty pretraining inputs end to end.
- `./tests/test_dataset.py`, `./tests/test_model.py`, `./tests/test_config.py`, `./tests/test_windowing.py`, and focused pipeline/inference tests — add regression coverage at the owning boundary.

Do not create files that duplicate an adequate existing owner. If investigation shows a listed path does not require modification, leave it unchanged and explain why in the completion report.
</output>

<validation>
Add focused tests that prove at least the following:

1. Pretraining input is non-empty and contains full self records plus omission-corrupted enemy records for both perspectives.
2. Fog omission affects enemy content only, uses one sampled per-example rate, inserts no placeholders, preserves delimiters, and leaves clean persisted artifacts unchanged.
3. Input tokens are clamped and excluded from diffusion corruption and loss; canvas tokens retain existing corruption/loss behavior.
4. Frozen statistics are derived from training data only, serialize deterministically with stable feature order, are reused unchanged by dev/test/resume/inference, and missing or incompatible artifacts fail loudly.
5. Allegiance is exactly `+1` for self and `-1` for enemy and is not standardized.
6. Type IDs and feature tensors remain disjoint at their module inputs.
7. Joint embedding output has shape `[B, L, d_model]` and equals the type embedding exactly at initialization, including zero final-projection weight and bias.
8. Gradient flow follows the mathematically consistent two-backward sequence in `<mathematical_consistency>`.
9. After the branch is unlocked, perturbing one continuous feature with fixed type IDs measurably changes only the corresponding position's output.
10. Canvas embedding and self-conditioning behavior remain unchanged and receive no static features.
11. Dynamic padding, device transfer, attention masks, both perspective streams, checkpoint/resume, and existing fine-tuning behavior remain valid.

Before running Python, confirm `./.venv/Scripts/python.exe` exists. Then run the smallest relevant tests first, followed by the package-wide suite because this change crosses core data, model, config, and pipeline contracts:

- `!.venv/Scripts/python.exe -m pytest -q tests/test_model.py tests/test_dataset.py tests/test_config.py tests/test_windowing.py`
- `!.venv/Scripts/python.exe -m pytest -q`
- `!git diff --check`

Run a bounded smoke train through the existing synthetic or smallest supported profile if it does not require unavailable GPU/data. Do not run expensive replay processing or a long cloud/GPU job. Clearly separate tests actually run from checks blocked by unavailable local data or CUDA.
</validation>

<success_criteria>
The task is complete only when:

1. Both assumptions have source-backed verdicts.
2. Every confirmed assumption is corrected end to end with no stale parallel behavior.
3. Pretraining again conditions on clamped full-self plus omission-corrupted-enemy input while loss remains canvas-only.
4. Input entity representations implement exactly `E + Linear(GELU(Linear(concat(E, feature_mlp(features)))))`, with the requested layer sizes, activations, zero initialization, strict input separation, and per-position locality.
5. Continuous statistics are frozen training-split artifacts and all consuming paths fail loudly rather than silently recomputing them.
6. Tests prove shape, exact initialization identity, mathematically valid staged gradient flow, post-unlock feature sensitivity, locality, fog semantics, and input clamping.
7. Relevant focused tests, the full pytest suite, and `git diff --check` pass, or any environment-only limitation is reported with exact evidence.
8. The DOX chain and `SPEC.md` describe the implemented architecture without contradictions.
9. The relevant jcodemunch/jdocmunch indexes are refreshed incrementally when practical; report any skipped refresh and why.
10. The final report lists assumption verdicts, changed files, architecture/data-flow decisions, tests run with results, and any remaining risk.
</success_criteria>
