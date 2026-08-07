<objective>
Add three independently-togglable architecture changes to the `thesis_ml` model so an
ablation can isolate why the overfit run cannot memorize its rare-class tokens.

The motivating failure: on `configs/local_overfit_v2.yaml` (150 epochs, 10 replays, a
subset the model should be able to memorize outright), the win/loss token — which is
ALWAYS at canvas index 0 — never gets its cross-entropy down to ~0.69, the value a
model would reach by learning nothing beyond "only two tokens ever appear here". The
`[END]` token behaves similarly. The model has ample capacity for this, so the working
hypothesis is a REPRESENTATIONAL defect, not a capacity or optimization one.

The specific diagnosis this ablation tests (already verified against the code — do not
re-derive it, but do respect it):

  - RoPE is the model's ONLY positional signal. There is no absolute positional
    embedding anywhere in `src/thesis_ml/model/`. RoPE's `q_i · k_j` depends solely on
    `i - j`, so the model's positional information is purely RELATIVE.
  - Positions are `torch.arange(seq_len)` over the CONCATENATED `[input | canvas]`
    sequence (`backbone.py` `RotaryEmbedding.forward`, called from
    `MultiHeadSelfAttention.forward`).
  - The input region is LEFT-padded (`collate.py`: `input_token_ids[row, max_input_len - length:]`,
    mirrored by `build_input_features(..., left_pad=True)`). The canvas is RIGHT-padded
    (`target_canvas[row, :length]`).
  - Left padding is LOAD-BEARING and must NOT be changed: it guarantees the last real
    input token sits at relative offset -1 from canvas index 0. That adjacency is the
    only crisp landmark the canvas has. (Measured on the overfit train set: input
    lengths span 174-4094, stdev 873; per-batch `max_input_len` is 4073 +/- 27 across
    100 distinct values. Right-padding the input would insert a masked gap of that size
    between input content and the canvas and destroy the landmark.)
  - The defect: NOTHING marks that offset -1 key as "the last input token". There is no
    segment signal and no boundary token, so a canvas-index-0 query cannot distinguish
    the input/output seam from any other neighbor. To identify canvas index 0 it would
    have to COUNT valid keys to its left, and that count varies from 174 to 4094.

So: segment embeddings and per-segment positions are the two changes most likely to fix
the failure, and they are designed to COMPOSE (a per-segment position reset aliases
input and canvas onto the same relative offsets, which requires segment embeddings to
disambiguate). The frozen-KV change is a separate, orthogonal efficiency question.

DO NOT change the padding layout in `collate.py`. That decision has been made.
</objective>

<architecture>
This task spans config parsing, five YAML profiles, three model-package modules, the
sampler, the training loop's forward call sites, and tests. That is spread-out work, so
you MUST execute it using the sub-agent team pattern:

  - You (Opus) are the ORCHESTRATOR. You do NOT write implementation code directly.
  - You delegate each change to a worker agent via the Task tool. Each worker gets fresh
    context, preventing pollution between changes.
  - You handle the interface contract, coordination, and final validation.

AGENT TEAM:

Every worker except the test agent runs on OPUS. This task changes attention math,
positional encoding, and checkpoint-compatibility gating on a model whose failure mode is
already subtle — a plausible-looking but subtly wrong implementation would silently
invalidate the entire ablation rather than fail loudly. The cost of a wrong result here
is a wasted 150-epoch run and a false conclusion about the root cause, so correctness is
worth far more than worker cost. Only Worker D (tests) runs on Sonnet, because writing
tests against an already-specified criteria list is mechanical.

  - **Investigation Agent** (OPUS, runs FIRST): produces the exact interface contract
    for threading per-example position ids and the KV-cache split through the backbone,
    plus the gradient-checkpointing and SDPA-mask implications.
  - **Worker A — Config** (OPUS, runs SECOND, blocking): the three boolean fields,
    validation, YAML defaults, and the `toggle_fingerprint` helper. Everyone downstream
    depends on these names.
  - **Worker B1 — Backbone + Model** (OPUS, parallel with B2/E): per-segment
    content-anchored RoPE, frozen input KV cache, architecture-fingerprint wiring, and
    the `loop.py` forward call sites. This is the hardest reasoning in the task.
  - **Worker B2 — Segment embeddings** (OPUS, parallel with B1/E): `embedding.py`
    only. Fully self-contained.
  - **Worker E — Documentation** (OPUS, parallel with B1/B2): the diagnostics writeup
    and the deferred BOS/EOS proposal.
  - **Worker C — Sampler** (OPUS, runs after B1): reuse the frozen input KV across
    denoising steps, plus the new forward kwarg at `sampler.py`'s three call sites.
  - **Worker D — Tests** (SONNET, runs after B1/B2/C).

EXECUTION ORDER:
  1. Investigation agent (alone).
  2. Worker A (alone — everything downstream imports its field names).
  3. Workers B1, B2, E in PARALLEL (spawn all three Task calls in ONE message).
  4. Worker C (needs B1's cache API).
  5. Worker D (needs everything).
  6. Orchestrator final validation.
</architecture>

<context>
**EXECUTION LOCATION — read this first.** This prompt is executed from inside the
`Thesis_ML` submodule, and `Thesis_ML` is its working directory root. Every path in this
prompt is relative to that root: `src/thesis_ml/...`, `configs/...`, `tests/...`,
`.venv/Scripts/python.exe`. Do NOT prefix paths with `Thesis_ML/` — that would resolve to
a non-existent `Thesis_ML/Thesis_ML/`. No path in this prompt reaches outside the
submodule. If you find yourself at the outer `local-play-bootstrap-main` directory
instead, `cd Thesis_ML` once, and only once, before doing anything else.

`Thesis_ML` is a self-contained uv project with its own `pyproject.toml`, `uv.lock`, and
`.venv`. Read `CLAUDE.md`, `AGENTS.md`, and `SPEC.md` (all three at this root) before
delegating — `SPEC.md` is the architecture source of truth and wins on any conflict.

Every worker must also follow the global commenting conventions, which are auto-loaded
into each agent's context and need no path: every function gets a docstring naming its
purpose, parameters, return value, and what it calls; every non-obvious transformation
gets an inline "why" comment.

Python interpreter: `.venv/Scripts/python.exe` (this is the interpreter
`tests/overfit.bat` requires). Never use bare `python`.

Files that matter:
  - `src/thesis_ml/config.py` — frozen dataclasses; `ModelConfig` at line ~57;
    unknown YAML keys are rejected (line ~382), so every new field needs a default in
    `config/default.yaml`.
  - `src/thesis_ml/model/backbone.py` — `RotaryEmbedding`, `apply_rope`,
    `MultiHeadSelfAttention`, `TransformerBlock`, `BidirectionalTransformer`.
  - `src/thesis_ml/model/model.py` — `SC2StrategyDiffusionModel.forward`,
    `ARCHITECTURE_ID`, `validate_checkpoint_compatibility`.
  - `src/thesis_ml/model/embedding.py` — `InputContextEmbedding` with its
    already-separate `embed_input` / `embed_canvas` methods.
  - `src/thesis_ml/data/collate.py` — DO NOT MODIFY. Read only, to confirm
    `input_lengths` already exists on the batch.
  - `src/thesis_ml/train/loop.py` — forward call sites at ~1115 and ~1138;
    `batch.input_lengths` already reaches the device at ~2139.
  - `src/thesis_ml/inference/sampler.py` — forward call sites at ~124, ~217,
    ~353; the denoising loop that re-runs the full forward every step.
  - `config/default.yaml`, `configs/local_overfit.yaml`,
    `local_overfit_v2.yaml`, `local_overfit_v2_finetune.yaml`, `local_full.yaml`.
</context>

<toggle_specifications>

All three toggles live under the `model:` block in YAML and default to **false**. With
all three false the code path must be BIT-IDENTICAL to today's, and the checkpoint
architecture identity must be the bare, unchanged `"uniform-gemma4-dense-v1"` string so
existing checkpoints still load.

---

**TOGGLE 1 — `model.frozen_input_kv: bool`**

Restructures the forward into two passes instead of one joint bidirectional pass:

  Pass 1: the input region alone goes through all L transformer blocks, attending only
          to itself (masked by `input_attention_mask`). At each layer, capture that
          layer's input K and V.
  Pass 2: the canvas region goes through all L blocks. At layer l, canvas queries attend
          to `concat(cached_input_K[l], canvas_K)` and the matching V. Canvas keys use
          `canvas_attention_mask`; cached input keys use `input_attention_mask`.

Consequences to honor:
  - The input hidden states no longer depend on the canvas. This is a REAL semantic
    change from the baseline (today the input attends to the canvas). Do NOT write a
    test asserting logit parity with the toggle on — that would be wrong. The intended
    equivalence test is different (see `<success_criteria>`).
  - `forward()` must still return logits over the FULL `[input | canvas]` length, so the
    existing `logits[:, input_len:, :]` slices in `loop.py` and `sampler.py` keep
    working unchanged. Concatenate the pass-1 and pass-2 hidden states before the head.
  - The per-layer cache must come from the input-only pass at that same layer.
  - RoPE rotations applied to the cached input keys must be consistent with what the
    pass-2 canvas queries expect. This is exactly why Toggle 1 and Toggle 3 are owned by
    the same worker.
  - Gradient checkpointing must still work with the two-pass structure (the current
    `checkpoint()` call closes over `attention_mask` via a lambda default arg).
  - The payoff being measured: at inference the input KV is computed ONCE and reused
    across every denoising step, roughly halving attention work per step. Worker C
    delivers that reuse.

---

**TOGGLE 2 — `model.segment_embeddings: bool`**

A learned `nn.Embedding(2, d_model)`: index 0 = input segment, index 1 = canvas segment.
The corresponding vector is added to every position's embedding in that region, inside
`InputContextEmbedding`.

  - Add it to the FINAL per-region embedding — i.e. to what `embed_input` and
    `embed_canvas` return — so it lands after the joint feature residual and after the
    self-conditioning post-norm rather than getting renormalized away.
  - Purpose: the same token id appearing in the input and in the canvas becomes
    genuinely different, uniformly across all positions, which is what lets a canvas
    query identify the input/output seam.
  - Initialize to ZERO. Note `SC2StrategyDiffusionModel._init_weights` walks every
    `nn.Embedding` and applies `std=0.02`, so re-zero it afterward the same way
    `embedding.reset_joint_output()` is already re-applied at `model.py` line ~70.
    Rationale: day-0 behavior then matches the baseline exactly and any divergence is
    attributable to learning, not to initialization noise.
  - When the toggle is OFF the module must NOT be constructed at all, so no extra keys
    enter the `state_dict` and old checkpoints keep loading.

---

**TOGGLE 3 — `model.per_segment_positions: bool`**

Position ids are computed PER SEGMENT, from the first real token of that segment:

  - Input: real input content gets positions `0 .. L_i - 1`, where `L_i` is that
    example's `input_lengths[i]`, assigned to its left-padded slots. The leading pad
    slots are masked out as keys, so whatever position they receive is inert — but be
    explicit and deterministic about it rather than leaving it accidental.
  - Canvas: positions `0 .. C - 1` in its own frame, restarting at 0 at canvas index 0.

This gives both coordinate systems stability that is invariant to how much left padding
a given batch happened to need, and it requires NO change to the collator.

Implementation consequences:
  - Positions become PER-EXAMPLE, so RoPE `cos`/`sin` gain a batch dimension: `[B, S, D]`
    instead of `[S, D]`. `apply_rope` currently broadcasts with `cos[None, None, :, :]`
    and must become `cos[:, None, :, :]` on this path. Keep the existing `[S, D]` path
    intact and exactly as-is when the toggle is off.
  - `input_lengths` must reach `SC2StrategyDiffusionModel.forward`. It is already on the
    batch and already moved to the device (`loop.py` ~2139), so add an optional
    `input_lengths: torch.Tensor | None = None` kwarg to `forward()` and pass
    `batch.input_lengths` at the `loop.py` call sites (~1115, ~1138) and the
    `sampler.py` call sites (~124, ~217, ~353).
  - The kwarg must be optional and unused when the toggle is off, so any call site that
    does not pass it keeps working.
  - When OFF: positions are `torch.arange(total_seq_len)` exactly as today.

---

**CHECKPOINT / RESUME GATING (not a toggle — a requirement across all three)**

Runs with different toggle sets must NOT be able to resume from or warm-start off each
other. Runs with the SAME toggle set must resume normally.

This is necessary, not merely defensive: Toggle 1 and Toggle 3 add ZERO parameters, so
`load_state_dict` would silently succeed across mismatched arms and quietly corrupt the
ablation.

Implementation: `validate_checkpoint_compatibility` is already called on all three load
paths — full resume (`loop.py` ~1212), warm start (`loop.py` ~1272), and inference
(`sampler.py` ~400) — and it already compares `architecture_identity`. So make
`architecture_identity` DERIVED rather than the bare constant:

  - Add `toggle_fingerprint(model_config) -> str` to `config.py`. It returns the empty
    string when all toggles are false, otherwise a deterministic, sorted, `+`-joined
    suffix of the enabled toggle names, e.g. `"+frozen_input_kv+per_segment_positions"`.
  - In `model.py`, set `self.architecture_identity = ARCHITECTURE_ID + toggle_fingerprint(model_config)`.
  - With all toggles off this MUST equal `"uniform-gemma4-dense-v1"` character for
    character, so `checkpoints/local-overfitV2/last.pt` still loads.
  - `pipeline/finished_export.py` (~208) reads `architecture_identity` off the model and
    therefore inherits this for free — verify, do not duplicate the logic.

</toggle_specifications>

<orchestrator_process>

You are the orchestrator. Follow these phases exactly. Do not write implementation code
yourself.

---

**PHASE 0 — INVESTIGATE**

Spawn ONE investigation agent (`subagent_type: "general-purpose"`, `model: "opus"`).
Its prompt must ask it to READ ONLY (make no edits) and report:

  1. The exact current signature chain from `SC2StrategyDiffusionModel.forward` down
     through `BidirectionalTransformer.forward` -> `TransformerBlock.forward` ->
     `MultiHeadSelfAttention.forward`, and the minimal set of new optional parameters
     needed to thread (a) per-example position ids and (b) a per-layer input K/V cache
     through that chain without changing any existing call site's behavior.
  2. How `gradient_checkpointing` currently wraps each block (the lambda with the
     `block=layer` default arg in `BidirectionalTransformer.forward`) and what breaks if
     extra tensor arguments are added — specifically whether they must become explicit
     `checkpoint()` positional args for autograd to track them.
  3. The SDPA masking situation: today `attn_mask = attention_mask[:, None, None, :]`, a
     key-only boolean mask. Report the exact mask shape needed for pass 2 of the frozen
     KV path, where the key axis is `input_len + canvas_len` but the query axis is only
     `canvas_len`. Flag whether the `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])`
     restriction (which deliberately excludes MATH so unsupported shapes error rather
     than silently allocating O(seq^2)) tolerates that non-square mask on CUDA, and if
     not, what the fallback must be.
  4. Every consumer of `ModelOutput.hidden_states` and `ModelOutput.logits` across
     `train/`, `eval/`, `inference/`, and `viz/`, confirming which ones assume the full
     `[input | canvas]` length. This determines whether the frozen-KV path can safely
     return a shorter tensor (it cannot — but confirm the blast radius).
  5. Confirmation that none of the three toggles feed `manifest_config_stamp` or
     `vocabulary_stamp` in `data/windowing.py` (~324-370), i.e. that flipping a toggle
     cannot invalidate `data/processed/local/overfit_window_manifest.jsonl` and trigger a
     rebuild across all 943 replays.

Have it write the report to `diagnostics/009-ablation-toggle-interface-map.md`
and also return the key findings in its final message.

When it returns, extract the interface contract. You will paste the relevant parts into
Workers B1, B2, and C — they cannot see the investigation results otherwise.

---

**PHASE 1 — CONFIG (blocking, runs alone)**

Spawn Worker A (`subagent_type: "general-purpose"`, `model: "opus"`). Scope:

  - `config.py`: add `frozen_input_kv: bool`, `segment_embeddings: bool`, and
    `per_segment_positions: bool` to `ModelConfig`, matching the existing frozen-dataclass
    and parsing style exactly. Add the `toggle_fingerprint(model_config) -> str` helper
    specified above.
  - `config/default.yaml`: add all three under `model:`, all `false`, each with a comment
    explaining what it does and that false is the baseline.
  - `configs/local_overfit_v2.yaml`: add all three under a `model:` block, all `false`,
    with a comment block explaining that this is the ablation control surface — flip a
    flag here and run `tests\overfit.bat`.
  - `configs/local_overfit.yaml`, `local_overfit_v2_finetune.yaml`, `local_full.yaml`:
    only touch these if the inheritance chain requires it. Explain the decision either
    way in the worker's report.
  - `tests/test_config.py`: assert the three fields parse, default to false, and that
    `toggle_fingerprint` returns `""` for all-off and a stable sorted string otherwise.

DO NOT let Worker A touch anything in `model/`, `train/`, or `inference/`.

Wait for it to finish. Read `config.py` yourself to confirm the exact field and helper
names before writing the downstream worker prompts — the downstream workers must use the
real names, not names you assumed.

---

**PHASE 2 — CORE IMPLEMENTATION (spawn all three Task calls in ONE message)**

Worker B1 — `model: "opus"`. The hardest task in the set: it requires attention-math
reasoning about RoPE consistency between cached keys and live queries, plus a non-trivial
restructure of the block loop.
  - FILES TO MODIFY: `model/backbone.py`, `model/model.py`, and the two forward call
    sites in `train/loop.py` (~1115, ~1138).
  - DO NOT TOUCH: `model/embedding.py`, `inference/sampler.py`, `data/collate.py`,
    anything else in `train/loop.py`.
  - Implements Toggle 1 and Toggle 3 and wires `toggle_fingerprint` into
    `architecture_identity`.
  - Paste in the Phase-0 interface contract verbatim.
  - Emphasize: when both toggles are off, `backbone.py` must execute the identical code
    path it does today, including the `[S, D]` RoPE broadcast.

Worker B2 — `model: "opus"`.
  - FILES TO MODIFY: `model/embedding.py` only.
  - DO NOT TOUCH: `model/model.py`, `model/backbone.py`, or anything else. If it believes
    a change outside `embedding.py` is required, it must STOP and report rather than make
    it — B1 owns `model.py` concurrently and a conflicting edit will be lost.
  - Implements Toggle 2 per the spec above. Note the one cross-file dependency it must
    NOT act on itself: the zero-re-initialization after `_init_weights` belongs in
    `model.py`. Have B2 expose a `reset_segment_embeddings()` method mirroring the
    existing `reset_joint_output()`, and report its name back so YOU can hand it to B1.

    Because of that ordering, put the `reset_segment_embeddings()` method name in B1's
    prompt as a CONTRACT it must call at `model.py` line ~70 alongside
    `reset_joint_output()`, guarded on the toggle. Specify the exact method name to both
    workers up front so they cannot diverge.

Worker E — `model: "opus"`.
  - FILES TO CREATE: `diagnostics/009-rare-class-position-blindness.md`.
  - FILES TO MODIFY: `Model_Architecture/MODEL_ARCHITECTURE.md` (document the
    three toggles and their off-by-default status; follow the process described in that
    directory's `UPDATE_PROMPT.md`).
  - The diagnostics document must record, in full: the verified padding behavior (input
    left-padded, canvas right-padded, with the exact `collate.py` line references); the
    finding that RoPE is the only positional signal and is purely relative; the measured
    input-length statistics (174-4094, stdev 873; per-batch `max_input_len` 4073 +/- 27
    across 100 distinct values on the overfit train set); the reasoning for why left
    padding must be preserved; and the conclusion that the missing seam marker — not the
    padding layout — is the likely cause of the rare-class failure.
  - It must ALSO record the DEFERRED future work, clearly labeled as not implemented:
    learned structural BOS/EOS seam markers on the canvas (BOS at canvas index 0 pushing
    win/loss to index 1; a structural EOS after the last real content token, distinct
    from the existing semantic `[END]` token which only appears when a game actually
    ends). Record why it was deferred: it touches the vocabulary, both canvas builders in
    `data/dataset.py`, corruption clamping, loss masking, and canvas budget accounting —
    a materially larger and data-touching change than the three model-side toggles. Note
    that if implemented, the markers would need to be clamped (never corrupted, or they
    are not landmarks) and excluded from the loss mask, and that the
    `manifest_config_stamp` / `vocabulary_stamp` impact must be checked first or flipping
    it would force a manifest rebuild across all 943 replays.
  - Worker E writes documentation only. It must modify NO code.

---

**PHASE 3 — SAMPLER (after B1 returns)**

Worker C — `model: "opus"`.
  - FILES TO MODIFY: `inference/sampler.py`.
  - Pass the new `input_lengths` kwarg at all three forward call sites (~124, ~217, ~353).
  - When `frozen_input_kv` is on, compute the input KV cache ONCE before the denoising
    loop and reuse it across every step, instead of re-running the full forward. This is
    the efficiency payoff the toggle exists to measure.
  - Add timing telemetry consistent with whatever `inference/timing.py` already does, so
    the per-step speedup is actually observable rather than merely asserted.
  - Paste in B1's cache API surface verbatim.

---

**PHASE 4 — TESTS (after B1, B2, C)**

Worker D — `model: "sonnet"`. Writes the tests enumerated in `<success_criteria>`. Give
it the full criteria list. It must add tests to the existing files under
`tests/` following their established style, not create a parallel structure.

---

**PHASE 5 — ORCHESTRATOR VALIDATION**

Do this yourself. Do not delegate it.

  1. Read every modified file end to end. Verify coherence: no broken imports, no
     signature mismatches between `model.py` and `backbone.py`, no divergence between
     B2's `reset_segment_embeddings()` name and B1's call to it, no leftover references
     to config field names that Worker A named differently.
  2. Run every verification action in `<verification>` below, in order.
  3. For each: state the exact command run, the actual result, and PASS or FAIL.
  4. If ANY check fails, spawn a targeted fix agent scoped to just that failure, then
     re-run ALL checks from the top.
  5. Report a summary: what each worker changed, the final toggle names as they appear in
     YAML, and the exact edit a user makes to `configs/local_overfit_v2.yaml` to run each
     ablation arm.

</orchestrator_process>

<agent_prompt_templates>

Every worker prompt you construct must follow this structure:

```
You are a focused implementation agent. Your job is to make ONE specific change.

CRITICAL SHELL RULES:
- You are ALREADY at the `Thesis_ML` project root. Do NOT `cd` into it, and do NOT
  prefix paths with `Thesis_ML/` — every path below is relative to this root.
- Package code lives at `src/thesis_ml/`. Use relative paths from this root throughout.
- Python interpreter is `.venv/Scripts/python.exe`. NEVER use bare `python`.

READ FIRST: `CLAUDE.md` and `AGENTS.md` at this root. Follow the commenting conventions
exactly — every function needs a docstring stating purpose, parameters, return value, and
what it calls; every non-obvious transformation needs a "why" comment. `SPEC.md` is the
architecture source of truth and wins on any conflict.

TASK: [exactly what to change]
CONTEXT FROM INVESTIGATION: [paste the relevant Phase-0 findings verbatim]
INTERFACE CONTRACT: [exact names/signatures this worker must produce or consume]
FILES TO READ: [specific files]
FILES TO MODIFY: [specific files]
DO NOT TOUCH: [everything else, named explicitly]

REQUIREMENTS:
- [specific requirements]
- With the toggle(s) off, the code path must be byte-for-byte behaviorally identical to
  the current implementation. This is a hard requirement, not a preference.

VERIFICATION:
- [how to verify this specific change]

IMPORTANT: Only make the change described above. Do not refactor surrounding code, do
not add features, do not make "improvements" beyond your assigned task. If you believe a
change outside your assigned files is required, STOP and report it instead of making it —
another agent may be editing that file concurrently.
```

</agent_prompt_templates>

<success_criteria>

These are the user's confirmed definition of done. Every one must pass.

1. **All-off parity.** With all three toggles false, `SC2StrategyDiffusionModel.forward`
   produces logits matching a fixed-seed reference within floating-point tolerance, and
   the entire existing test suite passes unchanged. `architecture_identity` is exactly
   `"uniform-gemma4-dense-v1"`, and `checkpoints/local-overfitV2/last.pt` still loads.

2. **Toggles compose.** Each toggle runs alone and all three run together, end to end,
   through the real training pipeline.

3. **Per-toggle behavioural tests.**
   - `frozen_input_kv`: with the toggle on, perturbing the canvas tokens leaves the input
     region's hidden states bitwise unchanged (proving the input genuinely does not
     attend to the canvas); and the sampler's cached-KV result equals a recomputed-KV
     result for the same inputs.
   - `per_segment_positions`: the SAME example collated into two batches with different
     `max_input_len` (i.e. batched against partners of different lengths) yields identical
     canvas logits. With the toggle off, this test must FAIL — assert both directions, so
     the test proves the toggle does something rather than proving nothing.
   - `segment_embeddings`: the same token id at an input position and at a canvas position
     produces different embeddings when on, and identical embeddings when off.

4. **No manifest rebuild.** `manifest_config_stamp` and `vocabulary_stamp` are unchanged
   by every toggle combination, and a run with all toggles on does not regenerate
   `data/processed/local/overfit_window_manifest.jsonl`.

5. **Resume gating.** A checkpoint written with toggle set X can be resumed AND
   warm-started by a run configured with the same toggle set X, and is REJECTED with a
   clear error by a run configured with any different toggle set. Test both directions
   explicitly, covering full resume (`load_checkpoint`), warm start
   (`load_model_weights`), and inference load (`sampler.load_*`). Include at minimum the
   user's stated cases: a `{frozen_input_kv, segment_embeddings, per_segment_positions}`
   run must NOT resume a `{frozen_input_kv}` run's checkpoint; a `{segment_embeddings}`
   run MUST resume a `{segment_embeddings}` run's checkpoint.

6. **Full suite green.** `pytest` passes with zero failures.

7. **Documentation.** `diagnostics/009-rare-class-position-blindness.md` exists
   and contains both the positional analysis and the explicitly-labeled deferred BOS/EOS
   proposal. `MODEL_ARCHITECTURE.md` documents the three toggles.

Structural criteria:
  - All workers completed; orchestrator verified coherence across all modified files.
  - No broken imports, signature mismatches, or cross-worker name divergence.

</success_criteria>

<verification>

Run these yourself in PHASE 5, in this order. For each, report the command, the actual
output, and PASS/FAIL.

1. **Full test suite**
   `.venv/Scripts/python.exe -m pytest tests -q`
   Expect zero failures. Covers criteria 1, 3, 5, 6.

2. **All-off architecture identity**
   Run a short Python snippet with the venv interpreter that builds
   `SC2StrategyDiffusionModel` from `configs/local_overfit_v2.yaml` with all toggles
   false and asserts `model.architecture_identity == "uniform-gemma4-dense-v1"`.
   Covers criterion 1.

3. **Existing checkpoint still loads**
   Load `checkpoints/local-overfitV2/last.pt` through
   `validate_checkpoint_compatibility` against an all-off model. Must not raise.
   Covers criterion 1.

4. **Toggle smoke runs — five arms**
   Write five temporary configs under
   `tests/output/ablation_smoke/` that each `extends:` the ABSOLUTE-or-correctly-relative
   path to `configs/local_overfit_v2.yaml` (note: `extends:` resolves relative to the
   config file's own directory, so get the relative path right or the run will fail at
   parse time), overriding only the `model:` toggles:
     a. all three false
     b. `frozen_input_kv: true` only
     c. `segment_embeddings: true` only
     d. `per_segment_positions: true` only
     e. all three true
   Run each with:
   `.venv/Scripts/python.exe -m thesis_ml.pipeline.train_pipeline --config <path> --max-steps 3`
   from the `Thesis_ML` root. Each must exit 0 and write `step_metrics.jsonl`.
   Delete the temporary configs afterward — the user chose "toggles only", so no new
   config files are to be committed.
   Covers criterion 2.

5. **No manifest rebuild**
   In the console output of run (e) above, confirm the window manifest was REUSED, not
   regenerated (look for the absence of manifest-building output and the presence of a
   stamp match). Additionally assert in a test that `manifest_config_stamp(config)` is
   equal across all five toggle combinations.
   Covers criterion 4.

6. **Documentation present**
   Read `diagnostics/009-rare-class-position-blindness.md` and confirm it
   contains the padding findings, the relative-RoPE analysis, and a clearly-labeled
   deferred BOS/EOS section. Confirm `MODEL_ARCHITECTURE.md` mentions all three toggles.
   Covers criterion 7.

If any check fails: spawn a targeted fix agent for that failure only, then re-run ALL six
checks from the top. Do not declare the task complete until every check passes. Do not
skip a check because it "should" pass.

</verification>
