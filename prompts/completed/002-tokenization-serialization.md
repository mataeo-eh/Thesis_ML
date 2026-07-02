<objective>
Build the tokenization and serialization layer: derive the extractor output schema from sample fixtures, construct the content vocabulary on top of the reserved special tokens, and implement deterministic serialization of a game-state snapshot into the canonical token sequence (and back, for testing). This is the layer that turns extractor output into the flat token sequences the model will consume.

This prompt has a HARD GATE in the middle: you derive and document the schema, then STOP for owner approval before building anything against it.
</objective>

<context>
- Python + PyTorch, `src/` layout, uv-managed. Read `./CLAUDE.md` for coding conventions.
- The architectural source of truth is `./SPEC.md`. READ IT IN FULL first. The directly relevant sections are §4 (tokenization and vocabulary), §5 (serialization order), §6 (input representation), §13 (extractor schema is a placeholder until you derive it), and §14 (ban list). SPEC.md wins on any conflict.
- Sample extractor outputs are in `./tests/fixtures/`, provided by the project owner. If that directory is empty or missing, STOP immediately and tell the owner you need fixtures before proceeding — do not fabricate a schema.
- Prior prompt 001 created the special-token registry (`[MASK]`, `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`) with reserved IDs and a documented content-token offset. Content tokens you create here begin at that offset.
- This is a localized task in one area (the vocab/serialization module). Single agent, no orchestration.
</context>

<constraints_from_spec>
These are non-negotiable (SPEC.md §4, §5, §14). Violating any of them is a failed task:
- Raw atomic entity-level tokens ONLY. One token per entity instance per timestep snapshot. Unit counts emerge from token repetition — never emit a count field or a "5×marine" compound.
- NO learned or compound tokenizer: no BPE, no merges, no clustering, no hierarchical schemes.
- Content tokens are **raw entity-type tokens — entirely location-agnostic**. The token identity carries NO spatial information. There is no spatial grid, no region quantization, no `spatial_grid` parameter. Position is input-only: the exact (X,Y) coordinate from the parquet is carried as a raw field on the input token record and becomes an additive contextual embedding later in the model (§6) — it never enters a token's identity.
- Single shared vocabulary for input and output.
- The OUTPUT vocabulary contains NO coordinates, frame numbers, or absolute times. High-fidelity map position and unit stats are NEVER tokens; they become allowlisted input-side contextual encodings in the model. Absolute clock metadata may be retained on source records for ordering and evaluation but must never enter model-facing features, embeddings, attention inputs, or targets. The tokenizer itself computes no embeddings and no positional encodings.
- Sequence position is handled by RoPE (or a RoPE-equivalent relative scheme) in the model's attention — NOT here. The tokenizer's only obligation toward this: produce flat, variable-length sequences with NO fixed per-timestep slot count and NO fixed entity-count-per-timestep assumptions, so sequence position stays well-defined and unbounded and unseen lengths do not break inference. Never conflate map position (an entity feature) with sequence position (a token index).
- Canonical serialization order: primary sort by entity type ID, tiebreak by the `within_type_tiebreak` config field (§11 default unit ID). Same ordering for input serialization and target construction.
- NO placeholder tokens for fogged/omitted entities, NO death-signal tokens. (These are later/out-of-scope per §12/§14 — not your concern here.)
</constraints_from_spec>

<phase_1_schema_derivation>
Do this FIRST, then STOP.

1. Read every fixture in `./tests/fixtures/`. Inspect the actual structure — field names, nesting, types, what an entity record contains, how timesteps/snapshots are delimited, what spatial/position information is present, what distinguishes self from enemy entities.

2. Write `./SCHEMA.md` documenting the derived schema precisely:
   - The top-level structure of an extractor output (a replay? a list of snapshots?).
   - The fields on an entity record, with types and example values.
   - Which field is the entity type, which identifies the unit instance, which carries position, which carries the self/enemy allegiance, which carries stats.
   - How snapshots/timesteps are represented and ordered.
   - Any fields present that the model will NOT use (note them explicitly so later prompts don't accidentally pull coordinates into the output vocabulary).
   - Explicitly map each derived field to the SPEC.md concept it serves (entity type → token identity; position → input-only contextual encoding; etc.).

3. STOP. Output a message to the owner: "SCHEMA.md derived from fixtures — please review and approve before I build tokenization against it." Do not proceed to Phase 2 until the owner approves. If the owner has pre-authorized continuing, you may proceed, but SCHEMA.md must still be written and committed first.
</phase_1_schema_derivation>

<phase_2_implementation>
Only after SCHEMA.md exists and is approved.

1. **Content vocabulary construction** (`./src/<pkg>/vocab/`):
   - Enumerate the content vocabulary as **raw entity-type tokens** — one token ID per distinct entity type from the schema. No spatial variants, no region quantization, no `spatial_grid`. The vocabulary is location-agnostic.
   - Assign content-token IDs starting at the documented offset from prompt 001's special-token registry. Special-token IDs are untouched.
   - Provide a vocabulary object that maps entity-type ↔ token ID, and exposes vocab size. The full token ID space is special tokens + content tokens.
   - The vocabulary must be reconstructible deterministically from the schema + config (same inputs → identical ID assignments). Document how it is built and persisted.

2. **Serialization** (`./src/<pkg>/serialize.py` or similar):
   - `serialize_snapshot(snapshot, config) -> list[token_id]`: take one timestep snapshot, emit its entities as content tokens in canonical order (§5), terminated by `[DELIMITER]`. Counts emerge from repetition — five marines produce five identical marine tokens (identity is entity-type only; their distinct positions live in the per-record raw fields, not the token).
   - A function to serialize a sequence of snapshots into a flat token sequence with delimiters between timesteps, matching the structure the dataset prompt will consume.
   - Produce ONE sequence of token records. Each record carries its token ID plus raw source fields for that entity. The later model-facing feature builder must explicitly allowlist precise map position, stats, and allegiance while excluding absolute clock, frame number, `game_loop`, and every timestamp-derived value. Source time metadata may remain on records for non-model dataset/evaluation use. Document this boundary clearly — it is the contract prompts 003 and 004 depend on.
   - A `deserialize` / decode function: token sequence → entity-type counts per timestep (location-agnostic — entity type and count are all the token identity carries; no position is recoverable, by design). This exists primarily so round-trip fidelity can be tested and so evaluation can later read generated canvases.

3. The self/enemy distinction is NOT encoded in the token identity (it is an input-side learned team-flag embedding per §6, built later). The token identity is allegiance-agnostic; allegiance is carried as a raw field on each token record so the later prompt can apply the team-flag embedding. Document this.
</phase_2_implementation>

<implementation>
- Determinism is the core property. Identical input snapshot + identical config must produce byte-identical token sequences across runs and machines. Sort keys must be total orders — no ties left to dict/set iteration order.
- The tokenizer carries raw field VALUES; it does NOT compute embeddings, positional encodings, or RoPE. Those are learned parameters and live in the model (prompt 004). Carrying a value is not the same as embedding it — keep that boundary crisp.
- Build for variable length from the start: never pad timesteps to a fixed entity count, never assume fixed slots. This is precisely what keeps the model RoPE-compatible and robust to unseen entity-counts and timestep-counts at inference.
- Do not build dataset windowing, fog/omission, or target-canvas construction — those are prompt 003. Stay in your lane: schema, vocab, single-snapshot and sequence serialization, decode.
- Read `within_type_tiebreak` from config; never hardcode it.
- Keep it simple: a couple of functions and a vocabulary object. No abstract base classes, no plugin tokenizer interface, no registry beyond the vocab map.
</implementation>

<output>
Phase 1:
- `./SCHEMA.md` — derived, documented extractor schema (then STOP for approval)

Phase 2 (after approval):
- `./src/<pkg>/vocab/` additions — content vocabulary construction and the vocab object
- `./src/<pkg>/serialize.py` — snapshot/sequence serialization + decode, producing one sequence of token records (token ID + raw input-only field values per record)
- `./tests/test_serialization.py` — round-trip and determinism tests (see verification)
</output>

<verification>
Before declaring complete, run these checks and report each as PASS/FAIL with the command and result:

1. **Schema gate respected:** Confirm `./SCHEMA.md` exists and was written before any Phase 2 code. PASS only if SCHEMA.md is present and Phase 2 did not begin before it existed.

2. **Round-trip serialization fidelity:** Run `pytest ./tests/test_serialization.py`. Tests must include: serialize a fixture snapshot, decode it, and assert the decoded entity-type counts per timestep exactly match the counts computed directly from the fixture. This is SPEC.md §16's "round-trip serialization fidelity" acceptance criterion — it is owned by this prompt. PASS only if round-trip tests pass on real fixtures.

3. **Determinism:** A test that serializes the same fixture twice (and, if feasible, re-builds the vocabulary from scratch) and asserts byte-identical token sequences and identical vocab ID assignments. PASS only if outputs are identical.

4. **Canonical order:** A test asserting that within a serialized timestep, tokens are ordered by entity type then by the configured tiebreak, and that each timestep ends with exactly one `[DELIMITER]`. PASS only if ordering and delimiter placement hold.

5. **No banned content in vocab:** A test/assertion that the vocabulary contains no coordinate, frame-number, absolute-time, or spatial-region tokens, and no compound/count tokens — only special tokens plus raw entity-type content tokens. PASS only if the vocabulary is clean.

For each check: state what you ran, the result, and PASS/FAIL. If any check fails, fix it and re-run ALL checks. Do not declare success without running every check on REAL fixtures (not synthetic stand-ins).
</verification>

<success_criteria>
- SPEC.md was read; nothing from §14 was implemented; the schema gate in Phase 1 was respected (SCHEMA.md written and approved before Phase 2).
- `./SCHEMA.md` accurately documents the real fixture structure and maps each field to its SPEC.md role.
- Content vocabulary is raw entity-type tokens only (location-agnostic), built deterministically on top of the reserved special-token IDs, with no coordinates/frames/times/regions/compounds.
- Serialization produces canonical-order token sequences with correct delimiter placement; decode recovers entity-type counts per timestep.
- Model-approved input-only fields (map position, stats, self/enemy allegiance) are carried as raw values for the later input-representation prompt, NOT in the vocabulary. Absolute time remains non-model metadata only, and the tokenizer computes no embeddings or positional encodings.
- Round-trip fidelity, determinism, canonical-order, and clean-vocab checks all PASS on real fixtures.
</success_criteria>
