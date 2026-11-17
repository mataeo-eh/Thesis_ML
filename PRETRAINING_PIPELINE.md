# Unconditioned Pretraining Sequence Pipeline

This document defines how a replay becomes an unconditioned pretraining sequence
for **both** arms (diffusion and autoregressive). Everything described here lives in
`thesis_shared` and is framework-agnostic: it outputs plain Python integers and
booleans. Arm-specific code starts only when a window is converted to a tensor.

Status: the grammar, vocabulary, windowing and packing are implemented and
exercised by `scripts/preview_sequence.py` and `tests/test_sequence_preview.py`.
Neither training loop consumes this format yet.

## 1. Pipeline stages

| Stage | Function | Module |
|---|---|---|
| Read replay rows, sorted by `game_loop` | `pd.read_parquet` | `sequence_formats/preview.py` (preview driver) |
| Serialize one snapshot into canonical records | `serialize_snapshot` | `serialize.py` |
| Build one joint timestep | `iter_joint_timesteps` | `sequence_formats/unconditioned_joint.py` |
| Resolve races and outcomes | `resolve_replay_race`, `resolve_replay_outcome` | `data/replay_outcomes.py` |
| Replay header and outcome footer | `build_replay_header`, `build_outcome_footer` | `sequence_formats/unconditioned_joint.py` |
| Optional content BPE | `fit_content_bpe`, `ContentBPE.encode` | `sequence_formats/bpe.py` |
| Window and pack | `iter_joint_windows` | `data/window_policies/unconditioned_joint.py` |
| Arm-specific masks | `SequenceWindow.diffusion_clamp_mask`, `.ar_loss_mask` | `sequence_formats/types.py` |

## 2. Token order within a timestep (binding rule)

Each timestep is:

```
[SELF] <self content tokens> [ENEMY] <enemy content tokens> [DELIMITER]
```

**Inside each player's block, content tokens are in ascending content token ID.**
This rule is deterministic. It depends only on the frozen token dictionary
(`data/Token_Dictionary.json`), never on replay contents, row order, or hashing.

How the IDs arise, and why this equals the serializer's order:

1. The token dictionary maps each **source ID** to a name. For entities (units and
   structures), the source ID is the StarCraft II game-data unit type ID (for example
   `commandcenter` = 18, `supplydepot` = 19, `scv` = 45, `marine` = 48). Upgrades and
   research use source IDs offset by 100,000, so they always sort after every entity.
2. `build_content_vocabulary` assigns content token IDs in ascending source-ID order,
   starting right after the special tokens. When two source IDs normalize to the same
   name, the lowest one is kept.
3. `serialize_snapshot` sorts records by (entities before upgrades, source ID,
   instance ID). Because token IDs follow source-ID order, this is exactly
   "ascending content token ID". The instance-ID tiebreak orders only copies of the
   same token, so it never changes the token sequence.
4. `iter_joint_timesteps` checks the rule on every player block and raises
   `ValueError` if it is ever broken.

Consequences:

- Repeated units are contiguous: `scv scv scv …`.
- The order is **not** alphabetical and **not** production order. It follows the
  game's unit-type numbering. A Terran block therefore reads
  `commandcenter supplydepot refinery barracks … scv … marine … mule`, with
  upgrades/research at the end.
- The player without content still emits their marker (`[SELF] [ENEMY] [DELIMITER]`
  is a valid empty timestep).
- Self always precedes enemy. "Self" is the perspective player, and each replay is
  intended to be served from both perspectives.

## 3. Vocabulary (joint format `unconditioned_joint_v1`)

```
[0–7]      special tokens: MASK PAD END DELIMITER WIN LOSE BOS EOS
[8–290]    content tokens (current dictionary), in the order above
[291–292]  [SELF] [ENEMY]
[293–296]  [PROTOSS] [RANDOM] [TERRAN] [ZERG]
```

Size: **297 IDs** with the current dictionary (291 legacy + 2 ownership + 4 race).
Optional BPE appends one ID per learned merge. All four race tokens are always
present, including `[RANDOM]`, even if a given corpus sample never uses one. Race
strings come from `players.<p>.race` in the replay's metadata JSON. An unknown value
raises rather than defaulting. `[END]` serves as the end-of-sequence marker. The
legacy `[EOS]` (ID 7) is unused by this format but remains in the vocabulary.

## 4. Sequence grammar

One replay **segment** (a contiguous span of one replay inside a window):

```
[BOS]
[SELF] <self race>
[ENEMY] <enemy race>
[DELIMITER]                         <- opens the first timestep of the segment
([SELF] … [ENEMY] … [DELIMITER])+   <- whole timesteps only
[SELF] <self outcome>
[ENEMY] <enemy outcome>
[END]                               <- only if the segment reaches the replay's real end
```

- A timestep is never split across windows. A timestep too large for an empty
  window raises an error.
- Every segment closes with the four-token outcome footer. A replay that spans
  several windows repeats its header and footer in each window.
- `[END]` appears only at a replay's true end, never at a window cut. At a cut,
  the next window continues the same replay with a fresh `[BOS]` + race header.

## 5. Replay packing (`pack_replays: true`)

After a replay's `[END]`, if the window still has room, the next replay's segment
starts immediately:

```
[DELIMITER | replay end after timestep 1136]
[SELF] [LOSE]
[ENEMY] [WIN]
[END]
[BOS]
[SELF] [TERRAN]
[ENEMY] [PROTOSS]
[DELIMITER 1]
…
```

- The next replay is packed only if its header, opening delimiter, at least one whole
  timestep, and its footer all fit (`5 + 1 + timestep + 4` tokens, plus `[END]` if that
  timestep is its last). Otherwise the window closes and the replay starts a new one.
- A replay that is short, or short after BPE, can therefore sit entirely inside a
  window, followed by the next replay.
- Replay order must be deterministic (fixed, split-respecting ordering). All replays
  in one packed stream must come from the same split.
- Each window records its `segments` (replay ID, start/end timestep, window position of
  each `[BOS]`). An arm can use them for document masking or per-segment positions.
  Neither choice is made here.

## 6. What each arm treats as input vs. target

| Positions | Diffusion (Arm A) | Autoregressive (Arm B) |
|---|---|---|
| Window position 0 `[BOS]` | clamped, unscored | context only (no predecessor) |
| Positions 1–4: first `[SELF] race [ENEMY] race` | **clamped, unscored** | ordinary targets |
| Everything else, including packed `[BOS]`/race headers, `[END]`, outcomes | mutable, scored | ordinary targets |

Clamping is by **position** (`clamped_prefix_length = 5`), never by token ID.
Later packed headers stay mutable for diffusion. This matches unconditional sampling,
where only the leading prefix is supplied and any later replay boundary must be
generated.

## 7. Framework boundary

`thesis_shared` delivers `SequenceWindow` objects: token IDs, segments, and both
arms' masks as tuples of Python ints and bools. It never imports torch, TensorFlow,
Keras or JAX, and a test imports every `thesis_shared` module to enforce that. Arm A
and Arm B each convert windows to their own tensors, add batch padding, and apply
their own loss. Nothing before that point depends on a framework.

## 8. Reproducing the preview

```
.venv\Scripts\python.exe scripts\preview_sequence.py
.venv\Scripts\python.exe scripts\preview_sequence.py --config config\sequence_preview_bpe.yaml
```

Outputs land in `scripts/output/sequence_preview/`: `sequence.txt`, `tokens.csv`,
`summary.json`, and `packed_boundary/` (the first window that crosses a replay
boundary). In `sequence.txt`, `[BOS | replay]` and numbered `[DELIMITER n]` labels
are display-only. The CSV holds the real token IDs.
