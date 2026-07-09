# 006 — Pretraining rework investigation

Read-only investigation for the upcoming refactor:
- Pre-training (`config.data.debut_mode=false`): input becomes literally absent (zero-length `input_token_ids`), no fog, class labels collapse (observed/fogged/future → one content class, ids not renumbered).
- Fine-tuning (`debut_mode=true`): keeps input + fog + 7-class debut taxonomy, but input grammar changes from `[all self timesteps][all enemy timesteps]` (one delimiter per player per timestep) to per-timestep interleaving `[t1 self][t1 enemy][DELIM][t2 self][t2 enemy][DELIM]...` (one delimiter per timestep).
- Both pipelines gain t=1.0 oversampling (10%/epoch), per-epoch generator reseeding, t-bucketed loss metrics, perspective-split (p1/p2) loss metrics.
- Config schema: fog + `class_loss_weights` become fine-tuning-only; `MaskScheduleConfig` gains `t_one_fraction`.
- `token_kind` branches removed except position-feature handling at embedding time.

All paths below are relative to `Thesis_ML/` unless stated otherwise.

---

## (a) Input-sequence consumers

### `src/thesis_ml/data/collate.py` — `collate_diffusion_examples` (lines 61–128)
- Line 68: `max_input_len = max(example.input_token_ids.numel() for example in examples)`. With `numel()==0` for every example (pretraining-absent-input) this evaluates to `0` — the loop at lines 73–77 (`input_token_ids[row, max_input_len - length :] = example.input_token_ids`) degrades to a `[:, 0:0]` no-op assignment, which is safe. **No crash**, but produces an `input_token_ids` tensor of shape `[B, 0]` and `input_attention_mask` of shape `[B, 0]`, all `input_lengths == 0`.
- Padding scheme: input is **left-padded** (`input_token_ids[row, max_input_len - length :] = ...`, line 75) so real tokens are right-aligned; canvas is **right-padded** (`target_canvas[row, :length] = ...`, line 90) so real tokens are left-aligned. This means the input→canvas boundary sits at the exact same absolute sequence position (`max_input_len`) for every row in a batch regardless of each row's real input length — this is what makes RoPE position of canvasconsistent across batch rows today (see backbone note below). If pretraining always yields `numel()==0`, `max_input_len` is uniformly `0` across the whole dataset (not just per-batch), so this degrades cleanly with no special-casing needed in collate itself.
- `build_input_features(input_records, max_input_len, left_pad=True)` (line 100) — called unconditionally; with `input_records == []` per example and `max_input_len == 0` it returns zero-sized `InputFeatures` tensors (see (a) embedding section) — no crash.
- `_input_timestep_count` (line 131) and `_enemy_future_timestep_count` (line 143) read `example.window_end`/`window_start`, not `input_token_ids`, so they are **unaffected** by empty input.
- **Verdict:** collate.py has no explicit "assume non-empty input" logic and no assumption about `[all self][all enemy]` block order or per-player delimiter counts — it treats `input_token_ids` as an opaque 1-D tensor. Zero-length input degrades cleanly today; no code change is structurally required here, only verification.

### `src/thesis_ml/model/embedding.py` — `InputContextEmbedding` (lines 110–178), `build_input_features` (60–92), `_records_to_tensors` (181–209)
- `forward` (124–137): `torch.cat([input_embeddings, canvas_embeddings], dim=1)` — concatenates unconditionally. With `input_token_ids` shape `[B,0]`, `self.token_embedding(input_token_ids)` in `embed_input` (line 145) returns `[B,0,D]`; `torch.cat` with an empty tensor along dim 1 is a no-op and returns exactly `canvas_embeddings`. **No crash, no special-casing needed.**
- `embed_input` (139–160): additive contextual encodings (map Fourier, unit stats, team embedding) are computed unconditionally over `input_features`; with 0-length feature tensors these ops are all no-ops (empty tensor arithmetic). `embed_input`/`build_input_features` **can be skipped cleanly** when input is absent — currently they are not skipped, but running them on empty tensors is harmless (near-zero cost, correctly shaped `[B,0,D]` output). A future optimization could special-case `input_token_ids.shape[1] == 0` to skip these calls, but it is not required for correctness.
- **No separator/boundary/BOS token and no segment embedding of any kind exists between the input region and the canvas region.** The only "segment" signal is `team_embedding` (`nn.Embedding(3, d_model, padding_idx=0)`, line 121), applied **only inside `embed_input`** (self=1/enemy=2/pad=0) — canvas tokens receive **no** team embedding (`embed_canvas`, line 162, only adds `self_cond_projection` output, never touches `team_embedding`). There is no explicit `[SEP]`/`[BOS]` special token in `vocab/special_tokens.py` (only `MASK/PAD/END/DELIMITER/WIN/LOSS`). If input becomes literally absent, this "team-embedding as pseudo-segment marker" signal simply disappears for that mode — nothing currently substitutes for it, which is consistent with the design ("full output canvas, no fog paradigm at all").

### `src/thesis_ml/model/model.py` — `SC2StrategyDiffusionModel.forward` (53–77), `_combine_attention_masks` (93–103)
- `forward` calls `self.embedding(...)` (concat as above) then `self.backbone(embeddings, attention_mask=attention_mask)` (line 75) over the **full concatenated sequence** — RoPE positions run `0..(input_len+canvas_len-1)` continuously across input+canvas (see backbone note). Because input is **left-padded**, real input content always occupies positions `[max_input_len - length, max_input_len)` and canvas always starts at position `max_input_len` exactly, independent of any individual row's real input length — i.e. canvas position 0 is **always** RoPE sequence-position `max_input_len`, batch-wide. When `max_input_len == 0` (pretraining-absent-input), canvas position 0 becomes RoPE sequence-position 0 uniformly. **This is a semantically significant behavior change**: today, RoPE positions assigned to canvas tokens shift with `max_input_len` per *batch* (not per example — that's masked by left-padding), so canvas position 0 does not always map to the same absolute RoPE angle across different batches/runs with different input lengths. Once input is always empty, canvas RoPE positions become perfectly stable batch-to-batch — worth confirming this is the intended, simpler behavior (it should be, and is strictly simpler).
- `_combine_attention_masks` (93–103): builds `input_attention_mask` via `torch.ones_like(input_token_ids, dtype=torch.bool)` if `None`, then `torch.cat([input_attention_mask, canvas_attention_mask], dim=1)`. With `input_token_ids` shape `[B,0]` this yields an empty mask segment concatenated with the canvas mask — no crash.
- Downstream `train/loop.py` slices canvas logits as `output.logits[:, input_len:, :]` where `input_len = batch.input_token_ids.shape[1]` (loop.py line 647, 660, 672) — with `input_len == 0` this is `output.logits[:, 0:, :]`, i.e. the entire output, which is correct.
- **Verdict:** `forward`, the embedding concat, and the attention-mask assembly all already degrade correctly to zero-length input with no code changes required; the substantive design question is only the RoPE-position/team-embedding points above, not a crash risk.

### `src/thesis_ml/model/backbone.py` — `RotaryEmbedding.forward` (68–75), `MultiHeadSelfAttention.forward` (122–163)
- `RotaryEmbedding.forward(seq_len, ...)` (line 68) is called with `seq_len = x.shape[1]` (the *full* concatenated input+canvas length) inside `MultiHeadSelfAttention.forward` (line 133: `cos, sin = self.rope(seq_len, ...)`), i.e. **one shared position axis across input and canvas**, positions `torch.arange(seq_len)`. There is no separate position axis or offset reset for the canvas region. This is the mechanism behind the "canvas position 0 == RoPE position `max_input_len`" behavior noted above.
- Attention mask: `attn_mask = attention_mask[:, None, None, :].to(torch.bool)` (line 140) is a **pure padding/key mask** — `True` = key participates. No causal mask (`is_causal=False`, line 160) — full bidirectional attention, consistent with SPEC §2/§6. Zero-length input contributes zero keys/queries to this dimension; nothing special required.

### Both pipelines under `src/thesis_ml/pipeline/`
- `train_pipeline.py` (pretraining): does not read `input_token_ids` directly; it flows entirely through `SC2DiffusionDataset` → `collate_diffusion_examples` → `TrainingLoop.fit`/`compute_batch_loss`. No pretraining-specific input-shape assumption found beyond `batch.input_token_ids.shape[1]` (`loop.py` line 647) which degrades to 0 cleanly.
- `finetune_pipeline.py`: identical data flow; unaffected since fine-tuning keeps input non-empty. It does *not* read `input_token_ids` directly either.

### `src/thesis_ml/eval/harness.py` (`evaluate_example`, lines 95–151; `_last_input_clock`, 167–173)
- `_last_input_clock` (167–173) reads `example.input_records` timestamps and falls back to `float(example.window_start)` when the list is empty (`if clocks else float(example.window_start)`, line 173) — **already degrades cleanly** to an empty input-records list (pretraining-absent-input). Used by `evaluate_example`'s `_timed` helper (line 154) for both predicted and ground-truth timestep timing. **No change required.**
- `evaluate_example`/`evaluate_examples` call `sample_canvas`/`denoise_canvas_once` with the collated batch; these are the sampler entry points (see `inference/sampler.py` below) which already tolerate zero-length input via `input_token_ids.shape[1]` slicing.
- **Verdict:** `harness.py` is input-length-agnostic; used for BOTH pretraining (headline build-order metric, SPEC §10) and, incidentally, fine-tuning debut evaluation goes through `eval/finetune_report.py` instead (see below) — but nothing here assumes non-empty input or `[all self][all enemy]` layout.

### `src/thesis_ml/eval/finetune_report.py` — fine-tuning only
- `_last_input_clock_seconds` (208–234) has the identical empty-fallback pattern as harness.py (falls back to `window_end`/`window_start` × `sampling_interval_s`). Since fine-tuning **keeps** non-empty input, this code path is not directly affected by the pretraining change, but the interleaved-grammar change (fine-tuning input becomes per-timestep `[self][enemy][DELIM]`) does **not** touch this module — it only reads `timestamp_seconds` off `TokenRecord`s, not sequence layout. **No change required here for either sub-change**, but note this module is fine-tuning-only so it never sees empty input at all.

### `src/thesis_ml/inference/sampler.py` — `denoise_canvas_once` (94–188), `sample_canvas` (191–360), `_partial_mask_canvas` (46–90)
- Both entry points slice canvas logits with `output.logits[:, input_token_ids.shape[1]:, :]` (lines 160, 270, 349) — **already correct** for `input_token_ids.shape[1] == 0` (slices the whole tensor).
- `_partial_mask_canvas` calls `corrupt_batch(input_token_ids=batch.input_token_ids.to(device), ...)` (line 84) purely to reuse `corrupt_batch`'s `target_canvas`-shaped RNG; `input_token_ids` is passed straight through unmodified (`corrupt_batch` never touches it) — zero-length input is a no-op pass-through.
- **Verdict:** the sampler makes **no** assumption about non-empty input or `[all self][all enemy]` layout; it is purely shape-driven (`.shape[1]`). No changes structurally required for zero-length input, though the fine-tuning interleaved-grammar change is likewise invisible to the sampler (it never inspects input token *content*, only `.shape`).

### `src/thesis_ml/inference/decode.py`
- Operates entirely on the **canvas** token sequence (`validate_canvas`, `validate_debut_canvas`, `decode_canvas`) — never reads `input_token_ids`. **Not an input-sequence consumer; unaffected by either change.**

### `src/thesis_ml/viz/diagnostics.py`
- `write_input_canvas_text_files` (937–983) iterates `item.example.input_records` (line 979) to dump a human-readable input-canvas text file, tagging each record `SELF`/`ENEMY`/`-----` via `_allegiance_marker`. With an empty `input_records` list (pretraining) this produces a header-only file — no crash, but the diagnostic becomes vacuous for pretraining checkpoints (nothing to show). Worth an explicit skip/guard once the refactor lands (currently it will silently emit an "empty" file with just the 3-line header) — not currently broken, just not very useful for pretraining.
- The rest of the module (`plot_count_comparison`, `plot_first_appearance_timeline`, `evaluate_selected`) goes through `eval.harness.evaluate_example`, which is input-length-agnostic as established above.

### `src/thesis_ml/scripts/estimate_context_window.py` — **hard-codes the current grammar; will need updating**
- Lines 65–70 (comment + code):
  ```python
  # Input grammar is [all self timesteps][all zero-fog enemy timesteps], so
  # each timestep contributes one delimiter to each block.  The output canvas
  # contains the enemy block, one delimiter per timestep, then [END].
  input_tokens = p1_content + p2_content + (2 * timesteps)
  p1_output = p2_content + timesteps + END_TOKEN_COUNT
  p2_output = p1_content + timesteps + END_TOKEN_COUNT
  ```
  This **directly encodes** the `[all self][all enemy]`, 2-delimiters-per-timestep grammar being replaced. Under the refactor:
  - Pretraining: `input_tokens` should become `0` (input is literally absent) — this script's `input_tokens` computation would need to either report 0 unconditionally or be parameterized by mode.
  - Fine-tuning: the interleaved grammar (`[t1 self][t1 enemy][DELIM]...`) means only **one** delimiter per timestep, not two (`self` and `enemy` share one trailing delimiter) — the `2 * timesteps` term must become `1 * timesteps` for the fine-tuning input estimate.
  - `token_accounting.input` docstring string at line 113 (`"all self tokens plus all zero-fog enemy tokens, with one delimiter per player per timestep"`) also needs updating.
  - This script's own test (`tests/test_context_window_estimator.py::test_estimate_replay_counts_unique_entities_upgrades_and_sequence_grammar`) hard-asserts `estimate.input_tokens == 17` (`= 7 + 4 + 2*3` delimiters) and will need its expected numbers recomputed for the new grammar (see §f).
  - **This is the single clearest "needs a code change, not just verification" hit for the input-grammar change.**

### Summary table (a)
| Consumer | Assumes non-empty input? | Assumes `[self][enemy]` block order / 2 delimiters-per-timestep? | Structural risk under zero-length / interleaved input |
|---|---|---|---|
| `data/collate.py` | No | No | None — degrades cleanly |
| `model/embedding.py` | No | No | None — concat/empty-tensor no-ops |
| `model/model.py` | No | No | RoPE canvas-position semantics shift when `max_input_len` becomes always-0 (design point, not a bug) |
| `model/backbone.py` | No | No | None |
| `eval/harness.py` | No (has explicit fallback) | No | None |
| `eval/finetune_report.py` | Fine-tuning only; N/A for empty input | No | None |
| `inference/sampler.py` | No | No | None |
| `inference/decode.py` | N/A (canvas-only) | N/A | None |
| `viz/diagnostics.py` | No (degrades to empty file) | No | Cosmetic only |
| `scripts/estimate_context_window.py` | **Yes, implicitly (`2*timesteps`, non-zero `input_tokens`)** | **Yes, hard-coded** | **Needs a real code change** |

---

## (b) Class-taxonomy consumers

### Class ids and names — `src/thesis_ml/data/dataset.py` (lines 38–71)
```python
CLASS_ENEMY_OBSERVED = 0
CLASS_ENEMY_FOGGED = 1
CLASS_ENEMY_FUTURE = 2
CLASS_DELIMITER = 3
CLASS_END = 4
CLASS_PAD = 5
CLASS_WINLOSS = 6   # fine-tuning-only outcome-token id, added ALONGSIDE, not renumbering the above
```
- `CLASS_LABELS` (49–56): a `dict[str,int]` (`"enemy-observed"→0`, … `"[PAD]"→5`) — **only referenced inside dataset.py's own module scope**; grep found no external importer. It predates the pretraining/debut split and does not include `CLASS_WINLOSS`. Candidate for cleanup but not a functional consumer.
- `DEBUT_CLASS_ID_TO_NAME` (63–71): the 7-class **fine-tuning** id→name map (`0→"visible-debut"`, `1→"fogged-debut"`, `2→"future-debut"`, `3→"delimiter"`, `4→"end"`, `5→"pad"`, `6→"win-loss"`). Imported by: `model/loss.py` (`active_class_id_to_name`), `eval/finetune_report.py` (`FOG_CLASS_NAMES`, line 76–80), `tests/test_debut_target.py`, `tests/test_training.py`.

### `src/thesis_ml/model/loss.py`
- `CLASS_ID_TO_NAME` (30–38): the **7-class pretraining** id→name map (`0→"enemy-observed"`, `1→"enemy-fogged"`, `2→"enemy-future"`, `3→"[DELIMITER]"`, `4→"[END]"`, `5→"[PAD]"`, `6→"win-loss"`). **This whole map needs to collapse under the refactor**: ids 0/1/2 currently carry three distinct pretraining names (observed/fogged/future) that must become **one** shared "content" name while the ids stay 0/1/2 as raw label values (per refactor context: "class labels collapsed... ids not renumbered"). Concretely, `_build_artifact_target`'s labeling call (`_canvas_label`, dataset.py 794–809) currently produces distinct ids 0/1/2 for pretraining canvases too — since pretraining loses fog and always predicts the full sequence, that call site's branching (fogged vs. observed vs. future) needs to collapse to a single id for pretraining while dataset.py's id **constants** stay the same (0/1/2 still exist for fine-tuning/debut use).
- `active_class_id_to_name(config)` (41–74): **the single seam already built for exactly this kind of mode-conditional map selection** — returns `DEBUT_CLASS_ID_TO_NAME` when `config.data.debut_mode` else `CLASS_ID_TO_NAME`. Both `CanvasCrossEntropyLoss.__init__` (loss.py line 112) and `TrainingLoop._write_epoch_metrics` (loop.py line 924) call this **same** helper, which is explicitly documented (loss.py 41–70) as the reason pretraining's per-class keys / CSV columns "stay byte-for-byte unchanged" whenever debut_mode is False — **any pretraining-side class collapse must happen by changing what `CLASS_ID_TO_NAME` itself contains, not by adding a third branch**, or the two call sites will disagree if only one is updated.
- `CanvasCrossEntropyLoss.__init__` (94–128): the class-weight buffer is sized as `torch.ones(len(self.class_id_to_name), dtype=torch.float32)` (line 114) — **i.e. `len(map)`, not `max(id)+1` and not hardcoded**. Since both `CLASS_ID_TO_NAME` and `DEBUT_CLASS_ID_TO_NAME` currently have exactly 7 entries, the buffer is length 7 in both modes today. If pretraining's map collapses to fewer *names* while still emitting label ids 0/1/2 for the single content class, **the buffer-sizing formula `len(map)` would break** unless the collapsed map still has one entry per *id* (e.g. `{0: "content", 1: "content", 2: "content", ...}` — a dict cannot have duplicate keys mapped differently, so this needs care: either (a) keep `CLASS_ID_TO_NAME` as `{0: "content", 3: "[DELIMITER]", 4: "[END]", 5: "[PAD]", 6: "win-loss"}` with ids 1/2 never appearing as separate dict entries — meaning `len(map)==5`, weight buffer length 5, and dataset.py's `_canvas_label` for pretraining must never emit ids 1 or 2 — or (b) keep the same 7-id space with the loss weight array indexed by id (needs `max(id)+1` sizing instead of `len(map)` if any id/name pairing becomes non-injective). This buffer-sizing formula is a concrete implementation decision the refactor must make explicit.
- `forward()` (130–178): per-class decomposition loop (153–157) is `for class_id, name in self.class_id_to_name.items(): class_mask = active & (class_labels == class_id); ...` — **iterates the class MAP** (not a fixed range), so it already tolerates a smaller/collapsed map automatically once `class_id_to_name` reflects the new pretraining taxonomy — no loop-structure change needed, only the map contents.
- `FUTURE_DISTANCE_BUCKETS` (77–83): `{"1": (1,1), "2_5": (2,5), "6_10": (6,10), "11_30": (11,30), "31_plus": (31,None)}`. Consumed by `forward()` (159–171, only when `prediction_distances is not None`, gated on `class_labels == CLASS_ENEMY_FUTURE` i.e. id 2) and by `train/loop.py`'s `_accumulate_future_distance`/`_finalize_future_distance` (1180–1199) and CSV column generation (`loop.py` 940–947). **This is pretraining-specific machinery today** (future-prediction distance only makes sense when there is a genuine future continuation — both pretraining full-reconstruction and fine-tuning debut both have a "future" class currently, ids/name differ only cosmetically: `enemy-future` vs `future-debut`). If pretraining's collapse merges id 2 into the single "content" class, `FUTURE_DISTANCE_BUCKETS` bucketing (keyed on `class_labels == CLASS_ENEMY_FUTURE`, i.e. `== 2`) **would silently stop firing for pretraining** unless the collapsed content class still uses id 2 for genuinely-future positions, or the distance-bucketing gate is redefined. This is a load-bearing consequence of "ids not renumbered" that needs explicit design attention — worth flagging to the implementer.

### Per-class metric keys — **exact patterns**
- **step-metrics JSONL** (`train/loop.py` `_write_metrics_line`, 889–902, writing `asdict(TrainStepLog)`): `per_class` is a `dict[str, float]` keyed by the **raw class name string** exactly as it appears in `active_class_id_to_name(config)` values, e.g.:
  ```json
  {"step": 1, "loss": 2.1, ..., "per_class": {"[DELIMITER]": 0.5, "[END]": 0.1, "[PAD]": 0.05, "enemy-fogged": 1.2, "enemy-future": 1.8, "enemy-observed": 0.9, "win-loss": 0.7}, "future_distance": {"1": 1.1, "2_5": 1.4, ...}, ...}
  ```
  (Debut mode swaps `"enemy-observed"/"enemy-fogged"/"enemy-future"` for `"visible-debut"/"fogged-debut"/"future-debut"`.) Sorted alphabetically per `_make_log` (loop.py 843: `dict(sorted(per_class.items()))`).
- **epoch-CSV columns** (`train/loop.py` `_write_epoch_metrics`, 914–989): column name = `f"train_{name}_loss"` / `f"dev_{name}_loss"` where `name = _metric_class_name(source_name)` (line 1166–1167: `source_name.strip("[]").replace("-", "_").lower()`). Concrete examples: `"[DELIMITER]"` → `train_delimiter_loss`; `"enemy-observed"` → `train_enemy_observed_loss`; `"win-loss"` → `dev_win_loss_loss`; `"future-debut"` → `train_future_debut_loss`. Future-distance CSV columns: `f"train_enemy_future_loss_distance_{name}"` for `name` in `FUTURE_DISTANCE_BUCKETS` (e.g. `train_enemy_future_loss_distance_2_5`, `dev_enemy_future_loss_distance_31_plus`).
- Both patterns are **derived programmatically from `active_class_id_to_name(config)`**, so a collapsed pretraining map (fewer names) will automatically shrink both the JSONL keys and the CSV columns for pretraining runs, and — because `_prepare_epoch_metrics_file` (991–1009) auto-migrates a CSV whose header doesn't match the current `fieldnames` list into a `.schema-migration` file and starts fresh — **a mid-run taxonomy change will silently fork the epoch-metrics CSV file**, worth flagging as an operational note (not a bug, existing safety mechanism, but relevant to the refactor's rollout).

### `enemy_future_timestep_counts` / `_enemy_future_prediction_distances` — `src/thesis_ml/data/collate.py`
- `_enemy_future_timestep_count` (143–167) and `_enemy_future_prediction_distances` (170–216) both gate on `label == CLASS_ENEMY_FUTURE` (id 2) to decide "is this position part of the future continuation." Both are **pretraining-shaped concepts today** (used unconditionally in `collate_diffusion_examples`, lines 93–96, 112–119 — i.e. **both** modes populate `canvas_prediction_distances` today, keyed the same way). If pretraining collapses id 2 out of active use (folded into "content"), these two functions must either (a) be redefined to detect "future" a different way (e.g. via `timestep_index >= input_count`, which `_input_timestep_count`/window boundaries already compute — note `_canvas_label`, dataset.py line 802: `if timestep_index >= input_timestep_count: return CLASS_ENEMY_FUTURE`, so the "future" *concept* is really a timestep-boundary comparison that currently happens to be recorded via a dedicated class id), or (b) be explicitly scoped to fine-tuning only. **This is the second load-bearing consequence of the label-collapse plan** (alongside `FUTURE_DISTANCE_BUCKETS` above) that needs explicit design attention — both currently derive "is this a future position" purely from `class_labels == CLASS_ENEMY_FUTURE`, which stops being a reliable signal once pretraining stops assigning that id distinctly.

### Split by pretraining / fine-tuning / both
| Consumer | Pretraining | Fine-tuning | Both |
|---|---|---|---|
| `CLASS_ID_TO_NAME` (loss.py) | ✓ (needs collapse) | | |
| `DEBUT_CLASS_ID_TO_NAME` (dataset.py) | | ✓ (unaffected by pretraining collapse) | |
| `active_class_id_to_name` (loss.py) | ✓ (mode switch already exists) | ✓ | ✓ (shared seam) |
| `CanvasCrossEntropyLoss` weight buffer sizing | ✓ (sizing formula needs a decision) | ✓ (unaffected) | |
| `FUTURE_DISTANCE_BUCKETS` / `_accumulate_future_distance` | ✓ (currently relies on id 2 staying distinct — **at risk**) | ✓ (unaffected, debut mode's "future-debut" stays id 2) | shared bucket dict/CSV columns |
| `enemy_future_timestep_counts` / `_enemy_future_prediction_distances` (collate.py) | ✓ (**at risk**, same reason) | ✓ (unaffected) | populated unconditionally today |
| epoch-CSV / step-JSONL key generation | ✓ (auto-shrinks with map) | ✓ | shared code path (`active_class_id_to_name`) |

---

## (c) Fog + class-weight config

### Reads of `config.fog.*`
- **Only one read in the whole `src/` tree**: `src/thesis_ml/data/dataset.py` line 218, inside `SC2DiffusionDataset._sample_fog_rate` (215–221): `distribution = self.config.fog.rate_distribution`. This is called from `SC2DiffusionDataset.__getitem__` (line 141: `fog_rate = self._sample_fog_rate(rng)`) — **called unconditionally for every example, regardless of `debut_mode`**. Since fog is applied to the enemy input block in `_build_artifact_input` (dataset.py 224–261) which itself runs unconditionally (used by both `_build_artifact_target` and `_build_debut_target` callers in `__getitem__`, lines 143–149), **fog is currently applied identically in both pretraining and fine-tuning** — this is exactly the coupling the refactor needs to break (fog should become fine-tuning-only, since pretraining has no input at all to fog).

### Reads of `config.loss.class_loss_weights.*`
- **Only one read**: `src/thesis_ml/model/loss.py` line 113, inside `CanvasCrossEntropyLoss.__init__` (94–128): `weights = config.loss.class_loss_weights`, then per-field reads at lines 118–127 (`weights.enemy_observed_reconstruction`, `.enemy_fogged_reconstruction`, `.enemy_future_prediction`, `.delimiter`, `.end`, `.pad`, `.win_loss`). This runs for **every** `CanvasCrossEntropyLoss` instantiation (both modes; `TrainingLoop.__init__`, loop.py line 128) — currently unconditional.

### Config YAML inventory (`Thesis_ML/configs/*.yaml` + base `config/default.yaml`)
| File | `extends` | `data.debut_mode` | fog / class-weight / mask-schedule keys present |
|---|---|---|---|
| `config/default.yaml` | — (base) | `false` | Full `fog.rate_distribution` block (13–17); full `loss.class_loss_weights` block (131–140, all 7 fields incl. `win_loss`); full `diffusion.mask_schedule` block (35–41: `name/t_distribution/min/max/loss_reweight`, **no `t_one_fraction` yet**) |
| `configs/local_full.yaml` | `../config/default.yaml` | `false` (explicit, line 15) — **pre-training profile** | Inherits `fog`/`loss.class_loss_weights`/`diffusion` unchanged from default.yaml (does not override any of them) |
| `configs/local_overfit.yaml` | `../config/default.yaml` | not set here → inherits `false` | Inherits `fog`/`loss.class_loss_weights`/`diffusion` unchanged |
| `configs/local_overfit_v2.yaml` | `local_overfit.yaml` | inherited `false` | Overrides only `loss.class_loss_weights.pad: 0.2`; `fog`/`diffusion` unchanged |
| `configs/local_overfit_v2_finetune.yaml` | `local_overfit_v2.yaml` | **`true`** (override, line ~34) — **fine-tuning profile** | Overrides `data.debut_mode`, `data.window_manifest_path`, `train.*`, `sampler.outcome_last`, `eval.debut_max_examples`, `storage.checkpoint_uri`; **does NOT override `fog` or `loss.class_loss_weights`** — inherits the exact same fog/weight values as the pretraining chain it extends |

**Key finding:** every profile — pretraining and fine-tuning alike — currently supplies (via inheritance) the *same* `fog.rate_distribution` and `loss.class_loss_weights` blocks, sourced from `config/default.yaml`. There is no existing per-mode config split at all.

### Config dataclass validation style — `src/thesis_ml/config.py`
- `_build_dataclass` (300–321): for **every** dataclass field, `if field.name not in raw: raise ConfigError(f"{field_path} is required")` (line 315–316) — **all fields are unconditionally required, no defaults, no optionality**, regardless of any other field's value (e.g. `debut_mode`). `_validate_value` (325–349) type-checks `int`/`float`/`str`/`bool`/nested-dataclass only; there is **no `Optional[...]` support, no default-value mechanism, and no cross-field conditional logic anywhere in this module**.
- `_build_dataclass` also rejects unknown keys unconditionally (`unknown = sorted(set(raw) - field_names); if unknown: raise ConfigError(...)`, lines 307–309) — so a YAML that *omits* `fog:` entirely when `debut_mode: false` would fail today with `"config.fog is required"`, and a YAML that includes an *extra* mode-conditional key would fail with `"config has unknown key: ..."`.
- **There is zero existing mode-conditional validation of any kind** in `config.py` — no code path reads one field's value to decide whether another field is required/forbidden. Making `fog`/`class_loss_weights` fine-tuning-only (per the refactor context) will require **new** validation logic that does not exist in any form today; it cannot be bolted onto the existing per-field-required loop without a structural change (e.g. making `FogConfig`/`ClassLossWeightsConfig` `Optional[...]` fields on `ProjectConfig` plus a new post-construction check keyed on `data.debut_mode`, or splitting `ProjectConfig` construction into a two-stage validate).
- `MaskScheduleConfig` (config.py 67–73) currently has exactly 5 fields: `name, t_distribution, min, max, loss_reweight`. Adding `t_one_fraction` is a straightforward new required field, but per the required-field rule above, **every existing YAML that supplies a `diffusion.mask_schedule` block will need the new key added** (or the loader will raise `"config.diffusion.mask_schedule.t_one_fraction is required"`) — currently only `config/default.yaml` defines this block (all four local configs inherit it via `extends`), so only that one file needs the new key added in practice, but this is worth flagging since the `extends`/`_deep_merge` mechanism (259–297) merges dicts recursively, not per-dataclass, so a partial override of `diffusion.mask_schedule` in a leaf config would still need to satisfy the full-field-required check after merging.

---

## (d) Corruption + generator

### Callers of `corrupt_batch` (`src/thesis_ml/train/corruption.py`, defined lines 24–65)
| Call site | `t=` argument | Bypasses t-oversampling? |
|---|---|---|
| `train/loop.py` `compute_batch_loss` (line 632–638) | `t=fixed_t` where `fixed_t` is a parameter threaded from `fit()`/`validate()`, default `None` | **No** when `fixed_t is None` (real training/eval runs) — samples via `_resolve_t`'s `t is None` branch. **Yes** when a caller passes a concrete `fixed_t` (e.g. smoke pipelines, most unit tests) |
| `inference/sampler.py` `_partial_mask_canvas` (line 83–89) | `t=mask_rate` (always a concrete float, default caller value `1.0`, or an explicit diagnostic rate) | **Yes, always** — this is a sampling-time diagnostic reproducing one fixed point on the schedule, never a training corruption draw |
| `pipeline/train_pipeline.py` `_run_smoke_pipeline` (line 139: `loop.fit(dataloader, max_steps=..., fixed_t=1.0)`) | `fixed_t=1.0` | **Yes** |
| `train/train.py` `run_smoke_train` (line 53: `loop.fit(dataloader, max_steps=max_steps, fixed_t=1.0)`) | `fixed_t=1.0` | **Yes** |
| `pipeline/train_pipeline.py` `_run_real_pipeline` (line 277–283: `loop.fit(train_loader, val_dataloader=val_loader, max_steps=..., epochs=..., retain_logs=False)`) | **no `fixed_t` passed → defaults to `None`** | **No** — this is the real (non-smoke) pretraining run; `t` is sampled fresh each batch via `_resolve_t`'s `t is None` branch, i.e. **this is exactly the call site the t=1.0-oversampling feature must patch** |
| `pipeline/finetune_pipeline.py` `run_finetune_pipeline` (line 228–234: `loop.fit(train_loader, val_dataloader=val_loader, max_steps=..., epochs=..., retain_logs=False)`) | no `fixed_t` → `None` | **No** — same as above, the real fine-tune run; also needs the oversampling patch |
| Most of `tests/test_training.py` (`_loop_and_batch`, individual tests) | explicit `fixed_t=1.0` or `t` in `[0.0, 0.25, 0.75, 1.0]` (e.g. `test_corruption_never_masks_input_region`, corruption.py-level call, lines 44–63) | **Yes, always** — unit tests deliberately pin `t` for determinism, so none of them currently exercise the `t is None` sampling path except indirectly through `_resolve_t`'s own logic |

### `_resolve_t` (corruption.py 74–95)
- Single implementation (no `_resolveT` variant exists in the repo — the prompt's `_resolveT` name does not match any symbol; the actual function is `_resolve_t`). Called only from `corrupt_batch` (line 48) — no other call sites found anywhere in `src/` or `tests/`.
- When `t is None` (line 83–85): `sampled = torch.rand(batch_size, ...); sampled = schedule.min + sampled * (schedule.max - schedule.min)` — **uniform sampling only**; this is the exact spot where t=1.0 oversampling (10%/epoch) needs to be injected (e.g. a per-example Bernoulli draw that forces `t=1.0` for ~10% of the batch before/after the uniform draw).

### `TrainingLoop.generator` — creation, seeding, checkpoint persistence
- Created in `TrainingLoop.__init__` (`train/loop.py` lines 164–169):
  ```python
  generator_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
  self.generator = torch.Generator(device=generator_device)
  if seed is not None:
      self.generator.manual_seed(seed)
  else:
      self.generator.seed()
  ```
  Seeded **once**, at construction time, from the `seed` constructor kwarg (itself sourced from `config.pipeline.seed` at every call site — see `pipeline/train_pipeline.py` lines 137, 258 and `pipeline/finetune_pipeline.py` line 209 and `train/train.py` line 52). **Never reseeded anywhere else in the class** — no per-epoch reseed call exists today.
- Used by: `compute_batch_loss` → `corrupt_batch(..., generator=self.generator, ...)` (loop.py line 636) for both the `t` draw and the mask-position `torch.rand` draw (corruption.py line 50–54); and `_use_self_conditioning` (loop.py 1044–1053) for the self-conditioning coin-flip (`torch.rand((), ..., generator=self.generator)`, line 1053).
- **Checkpoint persistence: the generator's RNG state is NOT saved or restored.** `save_checkpoint` (loop.py 700–726) writes `model/ema_model/optimizer/scheduler/global_step/completed_epochs/batches_completed_in_epoch/best_train_loss/epochs_without_improvement/elapsed_wall_seconds/total_tokens_ingested/unique_token_ids_seen/config` — **no `generator` key**. `load_checkpoint` (728–747) restores exactly that same key set — again no generator state. Consequence: **on every resume, `self.generator` is re-seeded from the constructor's `seed` argument again** (since `TrainingLoop.__init__` always runs on resume too, via the same `seed=config.pipeline.seed` call sites), meaning the corruption/self-conditioning RNG stream **restarts from the same seed** on every process restart rather than continuing — i.e. resumed runs replay the same `t`/mask/self-cond draws they would have drawn immediately after the *original* construction, not a continuation of wherever the pre-preemption stream had gotten to. This is a pre-existing property, not something the refactor introduces, but the refactor's "per-epoch generator reseeding" plan should account for it (if the plan is `generator.manual_seed(base_seed + epoch)` at the top of each epoch, that actually **improves** reproducibility on resume, since the seed becomes a deterministic function of `epoch_index` rather than of "how many draws happened since construction").

### `fit()` epoch loop — where a per-epoch reseed would go
- `train/loop.py` `fit()`, the `for epoch_index in range(self.completed_epochs, epoch_limit):` loop starts at line 230. Immediately inside it (lines 233–235):
  ```python
  dataset = getattr(dataloader, "dataset", None)
  if dataset is not None and hasattr(dataset, "set_epoch"):
      dataset.set_epoch(epoch_index)
  ```
  This already reseeds the **dataset's** per-example fog RNG for the new epoch (`SC2DiffusionDataset.set_epoch`, dataset.py line 196–197, consumed by `_rng_for_index`, lines 205–213, which folds `self._epoch` into the `SeedSequence`). A **per-epoch reseed of `self.generator`** (the corruption/self-conditioning generator) would naturally go right here, alongside this existing `dataset.set_epoch(epoch_index)` call and the `batch_sampler.set_epoch(epoch_index)` call two lines later (243–246) — the loop already has an established "reseed per-epoch-scoped RNGs here" pattern to extend.
- Precedent for the exact seeding formula already exists in `data/resumable_sampler.py`'s `ResumableBatchSampler` (`__init__` docstring line 40–41: `"base_seed: Base RNG seed; the per-epoch seed is base_seed + epoch"`; implemented at `__iter__` line 103–104: `generator.manual_seed(self._base_seed + self._epoch)`). This is constructed with `base_seed=config.pipeline.seed` at `pipeline/train_pipeline.py` line 478. **The `base_seed + epoch` pattern is the established idiom in this codebase** and is the natural template for `TrainingLoop.generator`'s per-epoch reseed.

### Is a `base_seed` field already in config?
- **No field literally named `base_seed` exists in `config.py`.** The parameter name `base_seed` belongs only to `ResumableBatchSampler.__init__` (a local constructor kwarg, not a config field) and is populated from `config.pipeline.seed` (`PipelineConfig.seed`, config.py line 106) at its one call site (`train_pipeline.py` line 478). `TrainingLoop.__init__`'s `seed` kwarg is likewise populated from `config.pipeline.seed` at every call site. **The refactor's per-epoch generator reseed should almost certainly reuse `config.pipeline.seed`** (the same value already threaded everywhere else as the run's base seed) rather than introduce a new config field, unless independence from the batch-order seed is explicitly desired (cf. how `split_seed` is deliberately kept separate from `seed` today, per the `PipelineConfig.split_seed` docstring at config.py 122–124: "split_seed is independent of the training seed so re-seeding a run does not reshuffle which replays are held out" — the same independence argument could apply to a corruption-specific seed, but no such field exists today).

---

## (e) `token_kind` branches

Full inventory of every `token_kind`-conditioned branch found via `grep -rn "token_kind"` across `src/`:

### `src/thesis_ml/serialize.py`
- `TokenRecord.token_kind: str` (line 34) — the field itself; values are `"entity"` / `"upgrade"` / `"delimiter"` (set at construction: `_entity_record` line 211, `_upgrade_records` line 235, `_delimiter_record` line 251).
- `_record_sort_key` (259–272) — **the canonical-serialization-order branch** (SPEC §5): 
  ```python
  if record.token_kind == "delimiter":
      return (2, 0, 0, "", "")
  if record.token_kind == "upgrade":
      return (1, vocabulary.source_id_for(record.token_name), 0, record.owner or "", "")
  # else: entity
  return (0, vocabulary.source_id_for(record.token_name), int(record.instance_id or 0), record.owner or "", record.entity_type or "")
  ```
  Entities sort before upgrades before delimiters (tuple leading `0/1/2`); only entities use `instance_id`/`entity_type` as tiebreakers. **This branch is a SPEC §5 serialization-order requirement, not fog/embedding special-casing** — it determines canonical token order, which the refactor context says should stay (only fog-application and embedding-position-feature special-casing are targeted for removal). Note this function is used by `serialize_snapshot` (the **fallback, non-artifact** input path — `build_input_records`/`build_target_canvas`), not by the primary `_artifact_*` path in dataset.py, which sources order directly from the pre-sorted memory-mapped `TokenizedReplay` arrays instead.

### `src/thesis_ml/data/dataset.py`
- Line 250 (`_build_artifact_input`): `if record.token_kind == "entity" and rng.random() < fog_rate: ...` — **the fog-application gate**. Only entity records are eligible for omission; upgrade records are never fogged. This is the primary target for removal from the pretraining path (fog goes away entirely for pretraining) and stays for fine-tuning.
- Line 732 (`build_input_records`, the fallback/legacy path): identical `if record.token_kind == "entity" and rng.random() < fog_rate:` gate — same concern, duplicate logic in the non-artifact path.
- Lines 311/349/355/511/592/607/613: `"token_kind": "outcome"/"end"/"pad"/"delimiter"` — these are **canvas metadata tags**, not conditional branches (they're just labels written into the `metadata` dict for downstream consumers like `eval/finetune_report.py`'s `_ground_truth_debut_events`, which reads `meta.get("token_name")`/`meta.get("timestep_index")` but never branches on `token_kind` itself). Not part of the "entity vs upgrade special-casing" being removed.
- Lines 633/664 (`_artifact_canvas_record`, `_artifact_timestep_records`): `token_kind="entity" if int(replay.kinds[position]) == ENTITY_CODE else "upgrade"` — this is where `token_kind` is **derived** from the memory-mapped `replay.kinds` array (see `data/windowing.py`'s `ENTITY_CODE` constant), not a behavioral branch itself, but the entity/upgrade dichotomy propagates into `TokenRecord.token_kind` from here for the artifact path.
- `_build_debut_target` (439–614): entity vs. upgrade debut-detection logic (lines 536–556) branches on `int(replay.kinds[position]) == ENTITY_CODE` (the raw kind, not `record.token_kind`, since this operates on the memory-mapped arrays before any `TokenRecord` is built) — **this is the "seen_upgrades special case" referenced in the task**. Exact current logic (536–556):
  ```python
  entity_counts_this_step: dict[int, int] = {}
  for position in enemy_positions:
      if int(replay.kinds[position]) == ENTITY_CODE:
          token_id = int(replay.token_ids[position])
          entity_counts_this_step[token_id] = entity_counts_this_step.get(token_id, 0) + 1

  debut_positions: list[int] = []
  emitted_per_entity: dict[int, int] = {}
  for position in enemy_positions:
      token_id = int(replay.token_ids[position])
      if int(replay.kinds[position]) == ENTITY_CODE:
          new_instances = entity_counts_this_step[token_id] - entity_running_max.get(token_id, 0)
          already_emitted = emitted_per_entity.get(token_id, 0)
          if already_emitted < new_instances:
              debut_positions.append(position)
              emitted_per_entity[token_id] = already_emitted + 1
      elif token_id not in seen_upgrades:
          seen_upgrades.add(token_id)
          debut_positions.append(position)
  ```
  Entities debut by **count increase** (running-max tracking, since instance ids aren't stored in the memory-mapped artifact); upgrades debut by **first-ever appearance** (`seen_upgrades` set membership, since upgrades are booleans/cumulative flags with no notion of "count"). This is **fine-tuning-only** code (`_build_debut_target` is only called when `debut_mode=True`, dataset.py line 158) — not something the pretraining-absent-input change touches, and the refactor context says `token_kind` branches should be removed "except position-feature handling at embedding time," which this arguably is *not* (it's debut-target construction, not embedding) — **this branch appears to be intentionally out of scope for removal**, worth confirming with the implementer since the instruction is ambiguous about whether entity-vs-upgrade debut-detection semantics themselves are being removed or just simplified/kept.

### `src/thesis_ml/model/embedding.py` — position-feature handling (**the branch the refactor explicitly keeps**)
- **No literal `token_kind` string appears in this file** — `embedding.py` never imports or reads `TokenRecord.token_kind` directly. Instead, position/stat features are computed uniformly for **every** input record via `_records_to_tensors` (181–209): `position = _parse_position(record.raw_position)` (line 202) and `raw = record.raw_attributes or {}` (line 205), with no entity/upgrade distinction in the code path itself.
- **What actually differs between entity and upgrade records feeding into this path** is data, not code: `_entity_record` (serialize.py 201–220) sets `raw_position=raw_attributes.get("pos_(X,Y,Z)")` (line 218, populated from the extractor's per-entity `pos_(X,Y,Z)` column) and a full `raw_attributes` dict (health/energy/shields/etc., from `_non_null_attributes`); `_upgrade_records` (serialize.py 223–244) **never sets `raw_position`** (defaults to `None`, `TokenRecord.raw_position: Any | None = None`, dataclass default) and sets `raw_attributes={"upgrade": upgrade}` — a **single key that is not in `STAT_KEYS`** (embedding.py 16–35), so `_records_to_tensors`'s stat loop (`for stat_index, key in enumerate(STAT_KEYS): stat_values[...] = _numeric_feature(raw.get(key))`, lines 206–207) reads `raw.get(key)` for a key that's never present in an upgrade's `raw_attributes` → `_numeric_feature(None)` → `0.0` for every stat slot (line 231–232: `if value is None: return 0.0`). And `_parse_position(None)` (line 212–213: `if value is None: return None`) leaves `map_values` at its `torch.zeros(...)` default (line 190) for upgrade tokens.
- **Would any spurious position signal be injected for upgrade records?** **No** — upgrade tokens get `map_values = (0.0, 0.0)` (the same as PAD/no-position default), which is indistinguishable from "at the origin" only in the sense that it's the same all-zero encoding every non-positioned token gets; there is no separate "unknown position" sentinel, so a genuine entity legitimately positioned near map-origin `(0,0)` would be embedding-indistinguishable from an upgrade token's absent-position default. **This is a pre-existing minor ambiguity, not something the refactor is introducing** — worth a one-line note to the implementer but not a blocking finding, since `team_embedding`/token-identity embedding still disambiguate upgrade vs. entity regardless of the shared zero-position encoding. This IS the "position-feature handling at embedding time" the refactor explicitly says to keep — confirmed there is no `token_kind`-branching *code* here to remove, only this data-shape observation to be aware of.

### `src/thesis_ml/eval/buildorder.py` — ground-truth build-order extraction
- `extract_build_order_from_frame` (66–106), lines 95–105:
  ```python
  if record.token_kind == "entity":
      key = (record.token_name, record.instance_id or "")
      if key in seen_entities: continue
      seen_entities.add(key)
      events.append(BuildOrderEvent(...))
  elif record.token_kind == "upgrade":
      if record.token_name in seen_upgrades: continue
      seen_upgrades.add(record.token_name)
      events.append(BuildOrderEvent(...))
  ```
  Same entity-by-instance-id vs. upgrade-by-name-only first-appearance dichotomy as `_build_debut_target` above, but this is the **ground-truth oracle** used by `eval/harness.py` (SPEC §10 headline metric, both modes) — reads directly from the source parquet via `serialize_snapshot`, entirely independent of the model/embedding/fog pipeline. **Not fog- or embedding-related**; this is evaluation-only logic that determines what counts as a "build order event" for accuracy/F1 scoring, unrelated to pretraining/fine-tuning input structure. Should almost certainly stay as-is (it doesn't touch input sequence structure or class labels at all) — flagged here only because it matched the `token_kind` grep, not because it's implicated by the refactor's stated goals.

### `src/thesis_ml/viz/diagnostics.py`
- **No `token_kind` branches** — the module reads `record.allegiance` (`_allegiance_marker`, 912–934) and `record.token_name`/`token_id` (`_token_name`, 813–821), never `record.token_kind`. Not implicated.

### `src/thesis_ml/train/train.py`
- Line 160 (`_synthetic_input_records`): `token_kind="entity" if index % 3 else "upgrade"` — synthetic smoke-test fixture data generation only (alternates kind every 3rd record for test coverage variety). Not production logic; would only need updating if the smoke test itself needs to model the new interleaved-grammar/absent-input behavior (see §f).

### Summary (e)
| Location | Branch purpose | In scope for removal per refactor wording? |
|---|---|---|
| `serialize.py:_record_sort_key` (259–272) | Canonical serialization order (SPEC §5) | **No** — order requirement, not fog/embedding special-casing |
| `data/dataset.py:250, 732` | Fog-application gate (entity-only) | **Yes** — this is exactly what pretraining removing fog entails; fine-tuning keeps it |
| `data/dataset.py:_build_debut_target` (536–556, on raw `kinds` not `TokenRecord.token_kind`) | Debut first-appearance detection (count-based vs. seen-set-based) | **Ambiguous** — fine-tuning-only, not fog/embedding; likely out of scope but worth confirming |
| `model/embedding.py` | Position-feature handling | **N/A — no branch exists to remove; this is the one path refactor explicitly keeps** (confirmed: differs only via data shape, not `token_kind` code) |
| `eval/buildorder.py:95-105` | Ground-truth build-order oracle first-appearance dichotomy | **No** — evaluation-only, independent of embedding/fog |
| `train/train.py:160` | Synthetic test-fixture generation | Test-only; update if smoke fixtures need to model new grammar |

---

## (f) Tests

One line each: file + test function, what it asserts, whether the refactor breaks it.

| File :: test | Asserts | Refactor impact |
|---|---|---|
| `test_dataset.py::test_input_target_asymmetry_and_zero_fog_degenerate_case` | fog_rate=0.0 yields observed_counts == full enemy counts; fog_rate=0.5 yields subset | Uses `build_input_records` (legacy fallback path, not `_artifact_*`); fog concept stays in fine-tuning so **not broken**, but if fog becomes fine-tuning-only via config validation this legacy-path helper's direct call (bypassing config) may need a debut_mode context — **low risk, verify** |
| `test_dataset.py::test_canvas_grammar_exact_budget_for_terminated_and_truncated_examples` | `build_target_canvas` produces exact-budget, grammar-valid canvases | Canvas grammar (§7) unaffected by input-side changes — **not broken** |
| `test_dataset.py::test_class_label_coverage_and_partially_fogged_group_counts` | fogged/observed counts and `CLASS_ENEMY_FOGGED` labels appear correctly | Legacy-path pretraining-shaped fog test — **breaks in spirit** once pretraining loses fog/class-collapse (this legacy path may need to be re-scoped to fine-tuning-only or removed) |
| `test_dataset.py::test_truncated_target_ends_at_boundary_and_pads_without_end` | Truncated canvas grammar | **Not broken** |
| `test_dataset.py::test_dataset_and_collate_determinism_under_seed` | Determinism of `input_token_ids`/`target_canvas`/`class_labels` under repeated construction; collate shape/mask correctness; `canvas_prediction_distances` positive iff `CLASS_ENEMY_FUTURE` | Uses default (pretraining, `debut_mode=False`) config. `canvas_prediction_distances[future_mask] > 0` assumes `CLASS_ENEMY_FUTURE` is still assigned distinctly in pretraining — **breaks under label collapse** unless the future-distance definition is redefined (see §b) |
| `test_debut_target.py::test_outcome_token_at_position_zero_with_winloss_class` | Position-0 outcome token + `CLASS_WINLOSS`, exactly once | Fine-tuning target builder, unaffected by pretraining-only changes and independent of input grammar — **not broken** |
| `test_debut_target.py::test_debut_event_timestep_is_first_appearance` | First-appearance timestep bucketing for entities/upgrades | Fine-tuning-only, `token_kind` debut-detection logic — **not broken unless that branch is also refactored** (see §e ambiguity) |
| `test_debut_target.py::test_debut_builder_materializes_only_emitted_records` | Vocabulary lookup count == emitted debut count | Fine-tuning-only — **not broken** |
| `test_debut_target.py::test_empty_timestep_produces_back_to_back_delimiters` | Back-to-back `[DELIMITER]` legal in debut canvas | Canvas-side (output), not input-grammar — **not broken** |
| `test_debut_target.py::test_fog_class_labels_visible_fogged_and_future` | `CLASS_ENEMY_OBSERVED/FOGGED/FUTURE` assigned correctly for debut events | Fine-tuning keeps 3-way fog-state distinction — **not broken** (fine-tuning retains this taxonomy per refactor context) |
| `test_debut_target.py::test_terminates_with_end_then_pads` | `[END]`/`[PAD]` grammar | **Not broken** |
| `test_debut_target.py::test_whole_timestep_truncation_when_budget_overflows` | Truncation grammar | **Not broken** |
| `test_debut_target.py::test_pretraining_artifact_path_leads_with_winloss_token` | `_build_artifact_target` (pretraining) leads with outcome token, has `[END]`/`[DELIMITER]` | Structural-only assertions (no per-class-id checks) — **likely survives**, but this test directly exercises the pretraining target builder that the refactor is reworking; must be re-verified against the new class-collapse output |
| `test_debut_target.py::test_debut_class_id_to_name_map_is_complete` | `DEBUT_CLASS_ID_TO_NAME` exact 7-entry dict | Fine-tuning map is explicitly unaffected by pretraining-only changes — **not broken** |
| `test_debut_target.py::test_default_config_debut_mode_off` | `default.yaml` has `debut_mode: false` | **Not broken** |
| `test_model.py::test_contextual_encodings_are_input_only` | Canvas tokens never receive map/stat/team contextual encodings | Still true when input is empty (vacuously true for pretraining) — **not broken**, though may want a new explicit zero-length-input case |
| `test_model.py::test_absolute_game_time_cannot_enter_model_features` | Timestamp never leaks into `InputFeatures` | **Not broken** |
| `test_model.py::test_padding_mask_is_boolean_key_mask_broadcast_over_heads_and_queries` | Attention mask semantics | Should still hold for zero-length input segment — **not broken, but worth a zero-length regression test** |
| `test_model.py::test_attention_is_bidirectional` | No causal masking | **Not broken** |
| `test_model.py::test_loss_is_canvas_only` / `test_per_class_logging_populated_and_consistent` / `test_future_loss_is_bucketed_by_prediction_distance` | Loss only scores canvas; per-class dict populated; future-distance bucketing works | The per-class/future-distance tests assume the current 7-name pretraining taxonomy — **at risk under class collapse** (same root cause as §b's `FUTURE_DISTANCE_BUCKETS` finding) |
| `test_training.py::test_smoke_train_loss_decreases_and_first_step_per_class_logs` | `first.per_class` keys == set of `CLASS_ID_TO_NAME[label]` for labels actually present in the synthetic canvas | Derives expected keys from `CLASS_ID_TO_NAME` dynamically, so it self-adjusts IF the map is updated — but `make_synthetic_examples` (`train/train.py` 56–103) hand-builds a canvas using `CLASS_ENEMY_OBSERVED/FOGGED/FUTURE` as three distinct labels **and** non-empty synthetic `input_records` (`_synthetic_input_records`, 8 fixed records) for a `debut_mode=False` (pretraining-shaped) smoke config — **breaks in spirit**: the synthetic fixture itself encodes the *old* taxonomy/input-shape and needs updating to match pretraining's new collapsed-class / absent-input contract |
| `test_training.py::test_corruption_never_masks_input_region` | `corrupt_batch` never alters `input_token_ids`, for `t ∈ {0,0.25,0.75,1.0}` | Directly calls `corrupt_batch` at the corruption-module level with fabricated non-empty tensors, independent of dataset/config — **not broken**, and is exactly the kind of explicit-`t` call that bypasses t-oversampling (see §d) |
| `test_training.py::test_schedule_weighting_uses_inverse_t_not_flat` | `inverse_t_weights` math | **Not broken** |
| `test_training.py::test_seeded_smoke_runs_are_deterministic` | Two identically-seeded `run_smoke_train` calls produce identical loss/masked-fraction/per-class sequences | Relies on `TrainingLoop.generator` being seeded once and deterministically from `seed` — **at risk if per-epoch reseeding changes the exact draw sequence** (should still be deterministic given the same seed, just a different sequence than today — the test itself should keep passing since it only compares two runs against each other, not against a fixed golden sequence) |
| `test_windowing.py::test_short_smoke_logs_all_seven_classes_from_first_step` | `set(first.per_class) == set(CLASS_ID_TO_NAME.values())` (all 7 pretraining names present from step 1) | **Directly breaks** under class collapse — pretraining's per-class set will have fewer than 7 entries once observed/fogged/future collapse to one name |
| `test_windowing.py::test_fog_is_resampled_per_serving_while_clean_tokens_stay_fixed` | `dataset.set_epoch(1)` changes `input_token_ids` but not `clean_input_token_ids` (pretraining config, `debut_mode=False` default in `_prepared`) | **Directly breaks** — with input literally absent in pretraining, `input_token_ids` is always an empty tensor regardless of epoch/fog-rate resampling, so `not torch.equal(first.input_token_ids, second.input_token_ids)` (both empty) fails. This test's *concept* (fog resampled per epoch) only makes sense in fine-tuning now — needs re-scoping to `debut_mode=True` |
| `test_windowing.py::test_dynamic_padding_masks_loss_and_preserves_real_position_outputs` | Variable-length input padding/masking + numerically-consistent batched-vs-alone forward pass (pretraining config) | **Breaks or becomes vacuous** — relies on picking a "short" vs "long" input example by `input_token_ids.numel()`; with pretraining input always zero-length there's no variation to test. Needs re-scoping to fine-tuning (where variable-length input still exists) to keep testing real padding behavior |
| `test_windowing.py::test_local_cadence_matches_timing_recovery` | Timestamp-recovery arithmetic for all three local pretraining profiles | **Not broken** |
| `test_windowing.py::test_local_model_parameter_count_is_near_ten_million` | Model parameter count budget | **Not broken** (architecture unaffected) |
| `test_context_window_estimator.py::test_estimate_replay_counts_unique_entities_upgrades_and_sequence_grammar` | Exact hard-coded token counts (`input_tokens == 17` = `p1_content + p2_content + 2*timesteps`) | **Directly breaks** — hard-codes the `[self][enemy]`/2-delimiters grammar this refactor replaces (see §a); numbers must be recomputed for both the pretraining-absent-input case and the fine-tuning interleaved-grammar case |
| `test_context_window_estimator.py::test_report_statistics_include_both_perspectives` | Report statistics shape, also depends on `input_tokens` formula | **Directly breaks** for the same reason (asserts `input_tokens.minimum/maximum == 4`, derived from the old formula) |
| `test_config.py::test_valid_config_loads` | Every `default.yaml` field, including `fog.rate_distribution`, `loss.class_loss_weights.*` unconditionally present | **Breaks if fog/class_loss_weights become optional/conditionally-absent fields** — this test currently asserts they're always readable off any loaded config; if the dataclass shape changes (e.g. `Optional[FogConfig]`), assertions like line 95-100 need updating |
| `test_config.py::test_local_profiles_extend_default_with_profile_specific_self_conditioning` | Cross-checks all 4 local YAML profiles' inherited/overridden fields, including `debut_mode`, `class_loss_weights.pad` | **At risk** — if `fog`/`class_loss_weights` become conditionally required/forbidden by `debut_mode`, this test's profile-loading assumptions (all 4 profiles currently load successfully with the same fog/weights blocks) need re-verification, especially since `local_full.yaml`/`local_overfit*.yaml` (`debut_mode=False`) would need those blocks *removed* rather than inherited if fog becomes fine-tuning-only and validation forbids it on pretraining configs |
| `test_config.py::test_unknown_key_is_rejected` | Extra key anywhere raises `ConfigError` | **Not broken** structurally, but validates the exact validation style that must be extended for mode-conditional fields (§c) |
| `test_finetune_report.py::*` (all) | Fine-tuning-only debut-report metrics (win/loss accuracy, fog-class F1, timing MAE, grammar validity, structural checks) | **Not broken** — fine-tuning keeps fog/7-class taxonomy/input entirely; only the input *grammar* (interleaving) changes, and none of these tests inspect input token layout directly |
| `test_finetune_pipeline.py::test_finetune_config_extends_overfit_v2_with_warm_start_and_debut_settings` | Fine-tune config inheritance chain, incl. implicit inheritance of fog/loss blocks from pretraining ancestors | **At risk** if fog/class-weights become split by mode in the config chain — the fine-tune profile currently gets its fog/weights purely via inheritance from `local_overfit_v2.yaml` → `local_overfit.yaml` → `default.yaml`; if `default.yaml`'s `fog`/`loss.class_loss_weights` move to living only in a fine-tuning-specific location, this inheritance chain needs rework |
| `test_sampler_outcome_last.py::*` (all 4) | `sampler.outcome_last` denoise-order constraint on canvas position 0 | Canvas-side only, sampler doesn't inspect input content — **not broken** |
| `test_sampler.py::test_sampler_generated_canvas_validates_and_input_is_clamped` | Sampler leaves input untouched, canvas passes `validate_canvas` | Should still hold with empty input (`torch.equal` on two empty tensors is trivially True) — **not broken**, though doesn't explicitly test the zero-length case |
| `test_pipeline.py::test_master_pipeline_smoke_run_writes_checkpoint_and_resumes` | Smoke pipeline (uses `make_synthetic_examples`) writes/resumes checkpoints | Depends on `make_synthetic_examples` (see `test_training.py` finding above) staying internally consistent — **needs re-verification once synthetic fixtures are updated** for the new pretraining grammar/taxonomy |
| `test_windows_launchers.py::test_windows_launchers_are_thin_config_driven_wrappers` | `.bat` launcher scripts are thin config-driven wrappers | **Not broken** — unrelated to data/model internals |

**General note:** many tests derive expected values dynamically from the production maps/functions they're testing against (e.g. `CLASS_ID_TO_NAME[label]`, `active_class_id_to_name(config)`), which means they will "self-adjust" once those maps are edited for the refactor — but the ones listed above as "directly breaks" either (a) hard-code the *old* taxonomy/grammar as literal expected values, or (b) construct hand-built synthetic fixtures that encode the old contract, or (c) test a *behavior* (variable-length pretraining input, fog resampling on pretraining input) that stops existing once pretraining input becomes literally absent.

---

## (g) SPEC.md sections

| Section | Heading | One-line summary |
|---|---|---|
| §2 | Model family — SETTLED | Masked discrete diffusion (LLaDA/LLaMA-lineage backbone: RMSNorm, SwiGLU, Llama3 RoPE, MHA+QK-norm); one flat `[input][canvas]` sequence, input clamped/never noised, canvas noised/loss-scored |
| §3 | Training objective — SETTLED | **Class taxonomy** and **loss weighting**: input=clamped self+fogged-enemy, canvas=leading outcome token + enemy reconstruction/future, 7-class per-token loss (`enemy-observed/fogged/future`, `[DELIMITER]/[END]/[PAD]`, `win-loss`), self-conditioning, EMA, confidence loss |
| §4 | Tokenization and vocabulary — SETTLED | Raw atomic entity tokens, single shared vocab, location-agnostic content tokens, special tokens incl. reserved `[WIN]/[LOSS]` |
| §5 | Serialization order — SETTLED | Canonical entity-type-then-instance-id ordering; same order for input and target |
| §6 | Input representation — SETTLED | **Input grammar / RoPE / positional encoding**: sequence-position via RoPE only (no learned table), map-position via input-only Fourier features + team-flag; input pipeline lives in the model, not the tokenizer; `[all self][all enemy]` per-timestep block structure implied by "windows are greedy contiguous runs of whole timesteps" |
| §7 | Output canvas semantics — SETTLED | Canvas grammar: leading outcome token, `(timestep-tokens [DELIMITER])+`, then `[END][PAD]*` or `[PAD]*`; whole-timestep-only truncation |
| §8 | Outcome/debut training — SETTLED mechanism | Debut fine-tuning canvas body (sparse first-appearance events) vs. pretraining's full reconstruction; same leading outcome-token placement in both |
| §9 | Inference and sampling — SETTLED mechanism, PROVISIONAL hyperparameters | Confidence-based iterative denoising, no remasking |
| §10 | Evaluation — SETTLED | Headline metric = build-order accuracy/F1 vs. deterministic oracle; CE is training-curve-only, never reported |
| §11 | PROVISIONAL config parameters | **Fog / mask-schedule / loss-weight config defaults**: `fog_rate_distribution`, `class_loss_weights`, `mask_schedule` table (this is the section the refactor's `MaskScheduleConfig.t_one_fraction` addition and fog/class-weight mode-splitting would update) |
| §12 | **OPEN questions — do not resolve, do not implement** | Fog-curriculum, timestep-membership encoding, SSM hybridization, copy-class loss weights, real-fog iteration, sequence packing — **confirmed: must NOT be touched by this refactor** |
| §14 | **Banned list — DO NOT IMPLEMENT, DO NOT SUGGEST** | Set aggregators, encoder-decoder/cross-attention, semi-AR generation, copy mechanisms, classification heads, learned tokenizers, strategy-label supervision, per-timestep slot budgets, coordinates/frame-numbers in output vocab, fog placeholder tokens, death-signal tokens, permutation-invariant losses, DBNs/JEPA — **confirmed: must NOT be touched by this refactor**

**Confirmation:** §12 and §14 are exactly what `CLAUDE.md` also flags ("Do not implement anything from SPEC.md section 14"; "Do not resolve or implement open questions from SPEC.md section 12") — both independently confirm these two sections are off-limits for any work stemming from this investigation.

---

## Cross-cutting risks worth flagging to the implementer (not asked for explicitly, but load-bearing)

1. **Class-id collapse vs. `FUTURE_DISTANCE_BUCKETS`/`enemy_future_timestep_counts`**: both currently gate on `class_labels == CLASS_ENEMY_FUTURE` (id 2) to detect "future" positions. If pretraining stops assigning that id distinctly, these two pieces of machinery silently stop firing for pretraining unless redefined against timestep-boundary comparison instead of class id (see §b).
2. **Loss-weight buffer sizing (`len(self.class_id_to_name)`)**: currently correct only because both taxonomies happen to have exactly 7 *names*. A collapsed pretraining map with fewer names (dict can't have 3 ids sharing one name as 3 separate entries) needs an explicit sizing decision — `len(map)` breaks unless the map is redefined to still enumerate one entry per surviving *name*, with ids 0/1/2 pointing at a *shared* name conceptually but the weight buffer needs to know it's still indexed 0-6 by id, not 0-4 by distinct name count.
3. **Config validation has zero existing conditional-field machinery** (§c) — making `fog`/`class_loss_weights` fine-tuning-only is a new capability for `config.py`, not an extension of an existing pattern.
4. **`scripts/estimate_context_window.py`** is the one concrete "will not just degrade gracefully" hit for the input-grammar change — it hard-codes both the block-order assumption and the `2 * timesteps` delimiter count.
