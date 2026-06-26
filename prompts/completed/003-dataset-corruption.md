<objective>
Build the PyTorch dataset that turns serialized replays into training examples for the masked-diffusion model. This is where the two halves of every example come from: the clamped INPUT (full self + fogged enemy) and the clean TARGET CANVAS (full enemy reconstruction + future, to budget). It also computes the per-canvas-position class labels that make the mandatory per-class loss logging possible.

This is the meatiest data-side prompt. Get the input/target asymmetry, the fog mechanism, and the canvas grammar exactly right — everything downstream depends on it.
</objective>

<context>
- Python + PyTorch, `src/` layout, uv-managed. Read `./CLAUDE.md` for conventions.
- Architectural source of truth is `./SPEC.md`. READ IT IN FULL. Directly relevant: §3 (training objective), §5 (serialization order), §6 (input representation), §7 (output canvas semantics), §11 (config params), §14 (ban list). SPEC wins on conflict.
- This consumes prompt 002's output: the serializer (snapshot/sequence → token records carrying token ID + raw input-only fields) and the approved `./SCHEMA.md`. USE that serializer — do not reimplement serialization. If `./SCHEMA.md` or the serializer is missing, STOP and say so.
- Localized task in one module (the dataset). Single agent, no orchestration — the windowing/fog/target/labeling logic shares one example structure and belongs in one context.
</context>

<two_corruptions_do_not_confuse>
There are TWO distinct corruption processes in this system. Confusing them is a failed task.

1. **Fog (input-side, THIS prompt).** Entity omission applied to the ENEMY input only. Realistic-fog modeling is out of scope (§12); v1 fog is synthetic random omission. Sample one fog rate per example from `fog_rate_distribution` (§11); each enemy entity in each input timestep is omitted independently with that probability (Bernoulli). Fog rate 0 is a valid draw and degenerates to clean-past-predict-future. Fog is applied to ENTITY RECORDS before serialization — omit entities, then serialize the survivors. Self entities are NEVER fogged.

2. **Canvas corruption (output-side, NOT this prompt).** MDLM-style uniform i.i.d. `[MASK]` at a global level t. This is the diffusion forward process and lives in the TRAINING LOOP (prompt 005), applied fresh each step. THIS prompt produces the clean target canvas x0 only. Do NOT apply `[MASK]` noising here. Do NOT sample t here.

The canvas never receives per-timestep-varying corruption; fog never touches the canvas.
</two_corruptions_do_not_confuse>

<constraints_from_spec>
Non-negotiable (§3, §7, §14):
- INPUT region = full self sequence + fogged enemy sequence, both carrying their raw input-only fields (from 002's records), with `[DELIMITER]` between timesteps. Allegiance is a raw field on each record (the team-flag embedding is applied later in the model).
- TARGET CANVAS = ENEMY ONLY. It reconstructs the FULL (UNFOGGED) enemy past/present for every input timestep PLUS the enemy future continuation. The target is the clean ground truth — fog is on the input, never on the target. That asymmetry (fogged enemy in, full enemy out) IS the learning signal.
- Canvas is a flat token sequence of fixed length `canvas_budget_tokens` (§11). Model-placed `[DELIMITER]`s partition timesteps; here you CONSTRUCT targets that already contain the correct delimiters.
- **Grammar invariant (§7):** a valid canvas is `(timestep-tokens [DELIMITER])*` followed by EITHER exact-fill termination (truncated horizon, no `[END]`) OR `[END] [PAD]*` to budget. `[PAD]` appears ONLY after `[END]`.
- **Mid-timestep-fill truncation (§7):** fill the canvas exactly to budget. If the remaining game exceeds budget, truncate mid-timestep so the canvas is exactly full and NO `[END]` appears. If the game ends within budget, emit `[END]` then `[PAD]` to budget. A truncated example's final timestep is partial and is dropped at eval (mark it; see below).
- No coordinates/frames/times in the canvas (they are not even in the vocabulary). No placeholder tokens for omitted entities. No death tokens. (§14)

Per-canvas-position class labels (§3) — THIS prompt computes them, because only the dataset knows the fog decisions and the past/future split. For every canvas position, emit a class label from: `enemy-observed` (entity present in the fogged input — a copy), `enemy-fogged` (entity omitted from input — an inference), `enemy-future` (a future-timestep entity), `[DELIMITER]`, `[END]`, `[PAD]`. These drive per-class loss logging in training; they are not a training input to the model.
</constraints_from_spec>

<requirements>
1. **Windowing.** Slice replays into examples. An example's INPUT spans `input_window_timesteps` (§11) consecutive snapshots; windows may begin mid-game (sample a start). The TARGET covers the enemy state of those same input timesteps (full/unfogged) followed by the enemy future continuation (subsequent snapshots) until the canvas budget is reached. Document how window starts are sampled and how short games (few or no future snapshots) are handled.

2. **Fog application.** Per the two-corruptions section: sample fog rate per example, Bernoulli-omit enemy entity records per input timestep, serialize survivors via 002's serializer for the input enemy sequence. Serialize ALL enemy entities (unfogged) for the corresponding target timesteps.

3. **Input assembly.** Concatenate the full self sequence and the fogged enemy sequence into the input region, with delimiters between timesteps, preserving each record's raw fields and allegiance. Define and document the input-region layout (e.g. self block then enemy block, or interleaved per timestep — pick one, justify it briefly, keep it deterministic).

4. **Target canvas construction.** Build the clean enemy-only canvas: full enemy reconstruction for input timesteps + future enemy timesteps, canonical order (§5) within each timestep, one `[DELIMITER]` after each timestep, then `[END] [PAD]*` (game ended within budget) or mid-timestep-fill exact truncation (horizon exceeds budget). Enforce the grammar invariant. Tag each example as terminated or truncated so eval can drop the partial final timestep on truncated examples.

5. **Class labels.** Produce a per-canvas-position label tensor (classes above). For repeated identical tokens within a (timestep, entity-type) group that were partially fogged — e.g. 5 marines with 2 omitted — assign labels BY COUNT within the group (2 `enemy-fogged`, 3 `enemy-observed`); the specific position assignment among identical tokens is arbitrary and only the aggregate count matters for logging. Document this.

6. **Dataset + collate.** A `Dataset` yielding per-example: input-region records, target canvas x0 (token IDs, fixed budget length), per-position class labels, and the terminated/truncated flag. A `collate_fn` batching these: input region padded to batch-max with a padding/attention mask; canvas is already fixed-length. The collate output must clearly delineate the input region vs canvas region (e.g. an input-length or region mask) so the model can clamp input (no noise, no loss) and compute loss on the canvas only. Do NOT assemble embeddings and do NOT noise the canvas — those are the model and loop respectively.

7. **Config-driven.** Read `input_window_timesteps`, `canvas_budget_tokens`, `fog_rate_distribution`, `sampling_interval_s` (and tiebreak indirectly via 002) from config. Never hardcode.
</requirements>

<implementation>
- The input/target asymmetry is the whole point: fog the input, never the target. Re-read your code to confirm the target enemy sequence is the UNFOGGED full state.
- Seeding: training fog should be random per example per epoch, but tests must be reproducible — provide a seedable RNG so the same seed yields identical fog draws. Don't bake a fixed seed into the training path.
- Mid-timestep-fill is the subtle bit: when the future overflows the budget, the canvas ends mid-timestep with no trailing `[DELIMITER]` for that partial timestep and no `[END]`. Verify `[PAD]` never appears without a preceding `[END]`.
- Keep it simple: a Dataset, a collate_fn, and a few helpers (window, fog, build_target, label). No abstract loader hierarchy, no caching layer, no augmentation framework. Build the straight path; if data loading is slow later, optimize then.
- Do not reimplement 002's serialization, do not compute embeddings (model/004), do not apply `[MASK]` or sample t (loop/005), do not build the build-order evaluation (007).
</implementation>

<output>
Create, using relative paths:
- `./src/<pkg>/data/dataset.py` — the Dataset (windowing, fog, target construction, class labeling)
- `./src/<pkg>/data/collate.py` — the collate_fn and batch structure (or co-locate in dataset.py if cleaner; document the choice)
- `./tests/test_dataset.py` — tests covering the verification below
- Use small synthetic or fixture-derived snapshots for tests; do not require the full Kaggle dataset to run tests.
</output>

<verification>
Run these and report each PASS/FAIL with the command and result:

1. **Input/target asymmetry:** A test with a fixed seed and nonzero fog: assert the input enemy token count is ≤ the target enemy count for the overlapping past timesteps, and that with fog rate 0 they match exactly (per-timestep, per token-type). PASS only if the asymmetry holds and the zero-fog degenerate case is exact.

2. **Canvas grammar invariant (§7):** A test asserting every constructed target canvas matches `(timestep-tokens [DELIMITER])*` followed by either exact-fill termination or `[END] [PAD]*`, and that `[PAD]` NEVER appears before an `[END]`. Test BOTH a game-ends-within-budget example and a horizon-exceeds-budget (mid-timestep-fill) example. PASS only if the invariant holds in both.

3. **Exact budget length:** Assert every target canvas is exactly `canvas_budget_tokens` long. PASS only if all canvases are exactly full.

4. **Class-label coverage and counts:** Assert the class-label tensor is the same length as the canvas, covers every position, and that the `enemy-fogged` count for a partially-fogged group equals the number of omitted entities in that group. PASS only if counts are correct.

5. **Truncated flag + eval drop:** Assert truncated examples are flagged and that the documented eval-time "drop final partial timestep" behavior is exposed (a function or flag downstream eval can use). PASS only if truncated examples are correctly identified.

6. **Determinism under seed:** Same seed → identical input tokens, target canvas, and labels across two dataset instantiations. PASS only if identical.

For each: state what was run, the result, PASS/FAIL. If any fails, fix and re-run ALL. Test on real fixtures and/or realistic synthetic snapshots — not degenerate one-entity stubs only.
</verification>

<success_criteria>
- SPEC was read; nothing from §14 implemented; the two corruption processes are kept strictly separate (fog here, `[MASK]`/t in the loop).
- Input region = full self + fogged enemy with raw fields and allegiance preserved; target canvas = unfogged enemy reconstruction + future, enemy-only.
- Target canvases are exactly `canvas_budget_tokens` long, satisfy the §7 grammar invariant, use mid-timestep-fill truncation, and place `[PAD]` only after `[END]`.
- Per-canvas-position class labels are produced (observed/fogged/future/delimiter/end/pad), with partially-fogged repeated-token groups labeled by count.
- Dataset + collate cleanly delineate input vs canvas regions without embedding or noising anything.
- Zero-fog degenerates exactly to clean-past-predict-future; nonzero fog produces the correct input/target asymmetry.
- All six verification checks PASS on realistic data.
</success_criteria>
