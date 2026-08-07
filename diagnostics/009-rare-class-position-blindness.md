# 009 — Rare-class position blindness

Read-only analysis. **No source was modified by this document.** It records the representational
hypothesis behind the three `model.*` ablation toggles: why a model with ample capacity cannot
memorize the two rare, position-pinned canvas tokens on a subset it should be able to overfit
outright.

Sibling document: [`009-ablation-toggle-interface-map.md`](009-ablation-toggle-interface-map.md)
covers the *mechanics* of implementing those toggles — signatures, gradient-checkpointing hazards,
SDPA mask shapes, logits-consumer blast radius, manifest-stamp safety. This document covers the
*motivation*. Where the two overlap, the interface map is authoritative on how the code must be
shaped and this document is authoritative on why the change is being made at all. Neither is
architecture authority; `SPEC.md` and `Model_Architecture/MODEL_ARCHITECTURE.md` own that.

---

## Scope and conditions

- Repository: `Thesis_ML` (submodule), branch `main`.
- Profile: `configs/local_overfit_v2.yaml` (extends `configs/local_overfit.yaml` →
  `config/default.yaml`). Pre-training semantics (`data.debut_mode: false`, inherited).
- Data scope: the explicitly named 10-train / 3-dev replay subset
  (`pipeline.train_replay_ids`, `configs/local_overfit.yaml:50`), selected at the corpus median
  input-token count.
- Run shape: `pipeline.batch_size: 10`, `train.epochs: 150`, early stopping disabled
  (`train.early_stopping_patience_epochs: 0`), ~34 optimizer steps per epoch.
- Model scale: `d_model 256`, `layers 10`, `heads 4`, `ffn 1024`
  (`configs/local_overfit.yaml:59-65`) — the same ~11.0M-parameter model documented in
  `Model_Architecture/MODEL_ARCHITECTURE.md`.
- Command: `tests\overfit.bat`. Metrics read from `tests/output/overfitV2/epoch_metrics.csv`.
- Baseline for every claim below is **all three toggles false**, which is the state currently
  committed in `config/default.yaml:37,41,45` and `configs/local_overfit_v2.yaml:50-52`.

**Status of the two categories of statement in this document.** Sections 2 and 3 are *verified
against live source* and cite it. Section 4 carries *measured statistics* forward from the
investigation that produced them. Sections 6 and 7 are *hypothesis*. Section 8 is *unimplemented
future work*.

---

## 1. The observed failure

On the overfit-V2 profile the model is asked to memorize 10 replays over 150 epochs. It largely
does. Two canvas classes do not follow:

- **The outcome token.** Ground truth always begins at canvas index 0 with the perspective
  player's `[WIN]` or `[LOSS]` token (`SPEC.md` §3, §7; built by `_build_artifact_target` at
  `src/thesis_ml/data/dataset.py:349` and `_build_debut_target` at `:528`, both documented as
  "position 0: the `outcome_id` token"). Only two token IDs — `WIN_ID = 4` and `LOSS_ID = 5`
  (`src/thesis_ml/vocab/special_tokens.py`) — ever occupy that slot.
  Its cross-entropy never comes down to **≈0.69**.
- **`[END]`.** The same behavior. `[END]` marks a genuinely-ended game and is appended once, after
  the last real content token, in every canvas builder
  (`dataset.py:436`, `:702`, `:903`).

`ln(2) ≈ 0.693` is the loss a model reaches by learning *nothing whatsoever* about the game and
only the fact that exactly two tokens ever appear at that slot — a uniform guess between `[WIN]`
and `[LOSS]`. Failing to reach it means the model has not even learned the *slot*, let alone which
of the two belongs there. It has 11M parameters, 150 epochs, and 10 fixed replays; ten memorized
constants is not a capacity problem and it is not an optimization problem.

**Working hypothesis: this is a representational defect.** The model cannot reliably tell which
position *is* canvas index 0, so it cannot attach a constant to it.

---

## 2. Verified padding behavior

The two regions are padded in **opposite directions**. Both are built in
`src/thesis_ml/data/collate.py::collate_diffusion_examples`.

**Input region — LEFT-padded** (`collate.py:117-121`):

```python
    for row, example in enumerate(examples):
        length = example.input_token_ids.numel()
        input_token_ids[row, max_input_len - length :] = example.input_token_ids
        input_attention_mask[row, max_input_len - length :] = True
        input_lengths[row] = length
```

Real input tokens are flush **right**. The model-facing feature tensors are padded the same way,
via the explicit `left_pad=True` at `collate.py:159`:

```python
        input_features = build_input_features(input_records, max_input_len, left_pad=True)
```

which resolves to the offset computation in
`src/thesis_ml/model/embedding.py:397` —
`offset = max(0, seq_len - len(row_records)) if left_pad else 0`.

**Canvas region — RIGHT-padded** (`collate.py:132-136`):

```python
    for row, example in enumerate(examples):
        length = example.target_canvas.numel()
        target_canvas[row, :length] = example.target_canvas
        class_labels[row, :length] = example.class_labels
        canvas_attention_mask[row, :length] = True
```

Real canvas tokens are flush **left**. `canvas_loss_mask` is a clone of `canvas_attention_mask`
(`collate.py:141`).

**Consequence.** The two regions are concatenated in `[input | canvas]` order by
`_combine_attention_masks` (`src/thesis_ml/model/model.py:249-259`) and by
`InputContextEmbedding.forward`. Because the input is right-flush and the canvas is left-flush,
the last real input token and canvas index 0 are **always immediately adjacent** in the
concatenated sequence, with no masked gap between them, for every row of every batch.

---

## 3. RoPE is the only positional signal, and it is purely relative

**There is no absolute positional embedding anywhere in `src/thesis_ml/model/`.** A sweep for
`nn.Embedding` in that package returns exactly two learnable tables: `token_embedding`
(`embedding.py:147`) and the new, off-by-default `segment_embedding` (`embedding.py:179`). There is
no position table, no learned absolute index, no clock feature — and `SPEC.md` §6 forbids adding
one ("no learned absolute position table, absolute game clock, frame number, `game_loop`, or
timestamp-derived feature") and §14 bans timestamp-derived values outright.

Position enters the model in exactly one place: rotary embeddings applied to queries and keys
inside attention.

- `RotaryEmbedding.forward` — `src/thesis_ml/model/backbone.py` (class declared at `:55`). In the
  baseline (`position_ids is None`) branch it computes, verbatim (`backbone.py:131-135`):

  ```python
        if position_ids is None:
            # Untouched baseline path. Kept verbatim (torch.arange + torch.outer)
            # so a toggles-off model executes exactly the tensor ops it always has.
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device=device))
  ```

- Called from `MultiHeadSelfAttention.forward` at `backbone.py:263`:

  ```python
        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype, position_ids=position_ids)
  ```

  where `seq_len` is unpacked from `x.shape` — the width of the **concatenated**
  `[input | canvas]` hidden state, not of either region alone.

> **Note on line numbers.** The three toggles have since landed across `backbone.py`, `model.py`,
> and `embedding.py`, which moved a lot of line numbers in those files. Every citation in this
> document was re-verified against the settled source. The sibling interface map still cites
> `RotaryEmbedding.forward` at `backbone.py:69` and the rope call at `:134`, both of which are now
> stale; the values here are current. Prefer the symbol names — `RotaryEmbedding.forward`,
> `MultiHeadSelfAttention.forward`, the `position_ids is None` baseline branch — which are stable.

The load-bearing property is the mathematics, not the line number. RoPE rotates query `i` and key
`j` by phases `θ·i` and `θ·j`; their dot product depends only on the difference:

```
q_i · k_j  =  f(i - j)
```

**RoPE carries no absolute index.** A token learns nothing about "where am I in the sequence"; it
learns only "how far away is that other token". A canvas-index-0 query and a canvas-index-0 query
from a different example with a different input length are, to attention, the same query in a
different neighborhood — there is no signal that says "you are the first output slot".

---

## 4. Measured input-length statistics

Measured on the overfit-V2 train subset (10 replays, both perspectives, all windows) and carried
forward from the investigation:

| Quantity | Value |
|---|---|
| Per-example input length, minimum | **174** tokens |
| Per-example input length, maximum | **4094** tokens |
| Per-example input length, standard deviation | **873** tokens |
| Per-batch `max_input_len` | **4073 ± 27**, across **100 distinct values** |

Two separate facts live in that table and both matter:

1. **Within a batch**, input lengths vary enormously — a spread of roughly 3,900 tokens with a
   standard deviation of 873. The amount of left padding a given row carries is essentially
   arbitrary.
2. **Across batches**, `max_input_len` is nearly constant (4073 ± 27) but never exactly constant:
   100 distinct values were observed. So the absolute concatenated index at which the canvas
   begins wobbles slightly from batch to batch as well.

Fact 1 is the important one for §6. Fact 2 only means the wobble is not even stable across
batches, so a model could not memorize a single absolute offset even if RoPE gave it one.

---

## 5. Why left padding must be preserved — SETTLED

Left-padding the input is **load-bearing and is not to be changed.** `src/thesis_ml/data/collate.py`
is not to be modified as part of this line of work.

The reason is §2's consequence. Left padding guarantees that the last real input token sits at
relative offset **−1** from canvas index 0, in every row, in every batch, regardless of that row's
input length. Since RoPE only sees `i - j`, an offset of exactly −1 is the *only* crisp, invariant
positional landmark the canvas has. Everything else the canvas can perceive positionally is a
relative distance to a neighbor it cannot identify.

Right-padding the input instead would replace that adjacency with a masked gap whose width is
`max_input_len - input_length` — i.e. distributed exactly as §4's statistics, up to roughly 3,900
tokens wide and varying per row. Canvas index 0 would find its nearest real key at a relative
offset drawn from that distribution. The one landmark the canvas currently has would be destroyed,
and the failure this document describes would get strictly worse, not better.

**The padding layout is therefore a settled decision.** The hypothesis in §6 is not that the
padding is wrong; it is that the padding by itself is not enough.

---

## 6. Conclusion: the missing seam marker

Left padding guarantees the seam is *there*. Nothing marks it as a seam.

Walk it from the model's point of view. A query at canvas index 0 wants to know that it is canvas
index 0. Its available evidence:

- **Its own embedding.** In the baseline this is `token_embedding[corrupted_token_id]` plus the
  self-conditioning residual (`InputContextEmbedding.embed_canvas`). Under uniform diffusion the
  token at that slot is, with probability `t`, a uniformly drawn non-`[MASK]` state
  (`SPEC.md` §3), so its own identity is noise. No region term is added in the baseline — see
  `Model_Architecture/MODEL_ARCHITECTURE.md`, "Explicit architecture observations" #5,
  "No region embedding".
- **RoPE offsets to its neighbors.** Purely relative (§3).
- **Its neighbors' embeddings.** The key at offset −1 is an ordinary content token embedding. It
  looks exactly like the key at offset −2, or −900. **Nothing distinguishes the last input token
  from any other input token.** There is no boundary token in the vocabulary
  (`special_tokens.py` defines only `[MASK] [PAD] [END] [DELIMITER] [WIN] [LOSS]`; `[DELIMITER]`
  is a per-timestep separator emitted many times inside the input, not a region marker), and there
  is no segment signal on either side.

So a canvas-index-0 query cannot recognize the input/output seam from its local neighborhood. The
only remaining route to "I am canvas index 0" is to **count valid keys to its left** — to
establish that the number of unmasked keys at negative offsets is exactly its own input length.
Per §4, that count ranges from **174 to 4094** and changes every example. Counting an unbounded,
per-example number of positions is not something a 10-layer bidirectional stack with relative-only
position information does reliably, and there is no reason it should.

**That is the hypothesized cause of the rare-class failure.** The outcome token and `[END]` are
precisely the two classes whose entire predictability is positional: `[WIN]`/`[LOSS]` is a constant
at a fixed index, and `[END]` is a constant at "just past the last real content token". Every other
class carries content the model can predict from the input semantically. The two classes that
depend on *knowing where you are* are exactly the two that fail. The correlation is the evidence.

Restated as a falsifiable claim: **the defect is the missing seam marker, not the padding layout.**
The three toggles below are the experiment that tests it.

---

## 7. How the three toggles address it

All three default to `false`; all-off is bit-identical to the pre-toggle model. Flip exactly one at
a time in `configs/local_overfit_v2.yaml` and run `tests\overfit.bat`. See the interface map for
implementation shape, and `Model_Architecture/MODEL_ARCHITECTURE.md` for the documented
architecture contract.

### 7.1 `model.segment_embeddings` — gives the seam a side

A learned `nn.Embedding(2, d_model)` (`0 = input`, `1 = canvas`) added to the **final** per-region
embedding, after the joint feature residual and after the self-conditioning post-norm so it is not
renormalized away. Zero-initialized (`InputContextEmbedding.reset_segment_embeddings`), so day-0
behavior matches the baseline exactly and any divergence is attributable to learning rather than to
initialization noise.

Effect on §6: the key at offset −1 now carries `segment = input` and the query at canvas index 0
carries `segment = canvas`. The seam becomes locally visible — a canvas query can detect a sign
change in its neighborhood instead of counting. It does **not**, on its own, tell canvas index 0
that it is index 0 rather than index 1; it tells it which side of the boundary each neighbor is on,
which combined with the −1 adjacency is enough to locate the boundary.

### 7.2 `model.per_segment_positions` — makes canvas index 0 a fixed RoPE phase

RoPE position ids computed **per segment** by `_build_per_segment_position_ids`
(`src/thesis_ml/model/model.py:200`): input real content gets `0..L_i-1` in its left-padded slots
(derived from that example's `input_lengths[i]`, itself recoverable as
`input_attention_mask.sum(dim=1)` by the construction in §2), and the canvas **restarts at 0** at
canvas index 0. Positions become **per-example**, so `cos`/`sin` gain a batch dimension:
`[B, S, D]` instead of `[S, D]`. `RotaryEmbedding.forward` carries the `position_ids` branch and
`apply_rope` dispatches on `cos.dim()`; the baseline branch is kept verbatim so a toggles-off model
runs the identical tensor ops.

Effect on §6: canvas index 0 stops being "somewhere around absolute index 4073, ±27, minus however
much padding this row had" and becomes RoPE position 0, exactly, always. Relative offsets *within*
the canvas also stop drifting with the input length.

### 7.3 They are designed to COMPOSE, and here is why

A per-segment position reset **aliases** the two regions onto the same relative offsets. Input
position 5 and canvas position 5 become indistinguishable to RoPE, because RoPE sees only
`i - j` and the two now produce the same phase. Position reset alone therefore *creates* an
ambiguity while removing the drift.

**Segment embeddings are what disambiguate the aliased positions.** With both on, a key is
identified by the pair `(segment, position-within-segment)`, which is unique. With only
`per_segment_positions` on, the pair collapses to `position-within-segment`, which is not.

This is the reason the ablation is run as a lattice rather than a race: `per_segment_positions`
alone is expected to be the *weakest* arm and may well be worse than baseline, and that would not
falsify the hypothesis. The arm that tests the hypothesis is both together.

### 7.4 `model.frozen_input_kv` is NOT part of this hypothesis

`frozen_input_kv` splits the single joint bidirectional forward into two passes: pass 1 runs the
input region alone through all L blocks attending only to itself, capturing each layer's input K
and V; pass 2 runs the canvas region through all L blocks, where at layer `l` canvas queries attend
to `concat(cached_input_K[l], canvas_K[l])` and the matching V.

This *is* a real semantic change — the input hidden states no longer depend on the canvas, which
they do in the baseline — so it must be ablated rather than assumed neutral. But its **payoff is
efficiency, not representation**: at inference the input KV is captured on the first denoising step
and reused across every later one (up to 64 passes per `SPEC.md` §9). Measured on CPU at
`input_len = 1536`, the cache-building step takes 0.0693 s against a 0.0097 s mean for
cache-reusing steps — a **7.1× per-step speedup**. The sampler deliberately captures the cache from
the first real step rather than from a dedicated priming forward, so it still makes exactly one
model call per step; see `Model_Architecture/MODEL_ARCHITECTURE.md`, "Sampling machinery".

It is included in the same toggle set because it is togglable at the same seam and shares the
identity-gating machinery, not because it is expected to fix rare-class loss. Do not read a
`frozen_input_kv` result as evidence for or against the seam-marker hypothesis.

### 7.5 Why the toggles are identity-gated

`toggle_fingerprint(model_config)` (`src/thesis_ml/config.py:363`) returns `""` when all three
toggles are false, otherwise a sorted `+`-joined suffix such as
`"+frozen_input_kv+per_segment_positions"`. The model stamps
`architecture_identity = ARCHITECTURE_ID + toggle_fingerprint(model_config)`, and
`validate_checkpoint_compatibility` (`src/thesis_ml/model/model.py:274`) rejects any checkpoint
whose stamp disagrees, so runs from different arms cannot resume from or warm-start off each other.

**This gating is necessary, not defensive.** `frozen_input_kv` and `per_segment_positions` add
**zero parameters**. Without the identity suffix, a strict `load_state_dict` would silently
*succeed* across mismatched arms — the key sets are identical — and the ablation would be quietly
corrupted with no error anywhere. (`segment_embeddings` does add a table and would be caught by
strict loading, but relying on that would leave two of the three arms unprotected.) With all
toggles off the identity is the unchanged `"uniform-gemma4-dense-v1"`, so every existing checkpoint
still loads.

---

## 8. DEFERRED FUTURE WORK — NOT IMPLEMENTED

> **Nothing in this section exists in the codebase.** It is a recorded proposal, deliberately cut
> from the current change. Do not treat any of it as implemented, in progress, or agreed. There is
> no config flag for it, no vocabulary entry, no code path.

### 8.1 The proposal: learned structural BOS/EOS seam markers on the canvas

Instead of signalling the seam indirectly through a segment embedding, mark it with real tokens:

- A structural **BOS** at canvas index 0, pushing the `[WIN]`/`[LOSS]` outcome token to canvas
  index **1**. Canvas index 0 then stops being a slot the model must *locate* and becomes a slot
  the model can *see*, and the outcome token acquires an unmistakable left neighbor.
- A structural **EOS** immediately after the last real content token. This is **distinct from the
  existing semantic `[END]` token**, and the distinction is the entire point: `[END]` is emitted
  only when a game actually ends within the window (`dataset.py:436`, `:702`, `:903`), so a
  boundary-truncated horizon has no `[END]` at all and instead runs straight into `[PAD]`
  (`SPEC.md` §7; `AGENTS.md` "Work Guidance"). A structural EOS would be emitted unconditionally
  and would carry no semantics about the game. The model would then be able to identify the
  content/padding boundary directly instead of inferring it — which is plausibly the second half
  of the `[END]` failure in §1.

### 8.2 Why it was deferred

It is a materially larger, **data-touching** change. It reaches:

- **The vocabulary** — two new reserved IDs in `src/thesis_ml/vocab/special_tokens.py`, which
  currently defines exactly six (`MASK_ID` 0 … `LOSS_ID` 5) with content tokens starting at
  `CONTENT_TOKEN_OFFSET = 100`.
- **Every canvas builder in `src/thesis_ml/data/dataset.py`** — `_build_artifact_target` (`:349`),
  `_build_debut_target` (`:528`), and the legacy/fallback `build_target_canvas` (`:866`). Note
  there are **three**, not two: the fallback path is easy to miss and would silently produce
  canvases in a different grammar from the artifact path.
- **Corruption clamping** — `src/thesis_ml/train/corruption.py` currently clamps only the input
  region; every canvas position is eligible for uniform replacement (`SPEC.md` §3).
- **Loss masking** — `canvas_loss_mask` is today an unconditional clone of
  `canvas_attention_mask` (`collate.py:141`); markers would need to be carved out of it.
- **Canvas budget accounting** — `data.canvas_budget_tokens` and the derived reconstruction limit
  (`windowing.py:_reconstruction_limit`, and its use at `dataset.py:203`, `:213`, `:875`) would
  each lose slots to the markers, shifting how much real content fits in a window.

By contrast, the three model-side toggles touch only `src/thesis_ml/model/` and `config.py`. That
asymmetry is the whole reason for the split.

### 8.3 Implementation notes for whoever picks it up

1. **The markers must be CLAMPED — never corrupted.** A landmark that is replaced by uniform noise
   with probability `t` is not a landmark. They must be excluded from the corruption draw the same
   way the input region already is.
2. **The markers must be EXCLUDED from the loss mask.** Scoring a token that is present by
   construction inflates the apparent accuracy of exactly the metric this work is trying to move
   (the rare-class cross in `epoch_metrics.csv` / `interval_metrics.csv`), and would make the
   before/after comparison meaningless.
3. Both properties must hold in the pre-training and debut/fine-tuning canvas builders, and in the
   sampler's eligible-position set (`src/thesis_ml/inference/sampler.py`) — a clamped position must
   be excluded from candidate sampling, renoising, entropy aggregation, and the stability check,
   the same way infill-diagnostic clamped positions already are (`SPEC.md` §9).

### 8.4 CHECK THE MANIFEST STAMPS FIRST — and the contrast that makes this the deferred item

**Before writing any code**, confirm the stamp impact. Adding vocabulary IDs and changing the
canvas grammar plausibly changes **both** stamps:

- `vocabulary_stamp` (`src/thesis_ml/data/windowing.py:351`) hashes
  `(token.name, token.token_id, token.source_id, token.kind)` for **every** token in the
  vocabulary. Two new reserved tokens change it by construction.
- `manifest_config_stamp` (`windowing.py:324`) hashes `MANIFEST_VERSION`,
  `TOKENIZED_ARTIFACT_VERSION`, `_target_semantics(config)`, and five `config.data.*` fields
  including `canvas_budget_tokens` and `canvas_recon_fraction` — so any budget adjustment made to
  accommodate the markers changes it too.

A change to either stamp forces a **manifest rebuild across all 943 replays**
(`pipeline/train_pipeline.py` rebuild trigger; `windowing.py` load-time mismatch raise).

**The explicit contrast — this is the reason for the split.** The investigation agent verified
empirically, against the real on-disk `data/processed/local/overfit_window_manifest.jsonl`, that
mutating **every** `model.*` field leaves `manifest_config_stamp` byte-identical
(`009-ablation-toggle-interface-map.md` §5.3). Neither stamp function reads `config.model` at all —
one hashes only module constants and `config.data.*`, the other only vocabulary tokens.

So:

| Change | Touches `config.model` only? | Manifest rebuild? |
|---|---|---|
| The three model-side toggles | **Yes** | **No** — free at the data layer |
| BOS/EOS seam markers | **No** — vocabulary + `config.data.*` + all three canvas builders | **Yes** — all 943 replays |

That is exactly why the toggles ship now and BOS/EOS does not.

---

## 9. Verification

The failure is closed when, on an ablation arm run through `tests\overfit.bat` with an otherwise
unchanged `configs/local_overfit_v2.yaml`:

- `tests/output/overfitV2/epoch_metrics.csv` shows the win/loss and `[END]` per-class losses
  falling **below 0.69** and continuing down, rather than plateauing above it; and
- the rare-class-by-corruption cross ({win/loss, `[END]`, `[DELIMITER]`} × the four corruption
  buckets) shows the improvement concentrated in the high-corruption buckets, where positional
  identification is the only available signal.

Record the arm's `architecture_identity` suffix with the result. Point `storage.checkpoint_uri`
somewhere new before running an enabled arm so the all-false baseline
`checkpoints/local-overfitV2/last.pt` survives for comparison.

---

## 10. Open flag for the owner — NOT resolved here

`SPEC.md` §14 (Banned list — DO NOT IMPLEMENT, DO NOT SUGGEST) includes **"Prompt KV caching or a
separate prompt encoder"** and **"Encoder-decoder architecture, including cross-attention
conditioning"**. `model.frozen_input_kv` (§7.4) reuses the input region's per-layer K/V across
denoising steps and runs the input region through all L blocks before the canvas sees it, which is
close enough to both entries to need an explicit owner ruling.

This document does not resolve it, and `CLAUDE.md` forbids implementing anything from `SPEC.md`
§14. Two things are worth noting for whoever does rule on it: the toggle defaults to `false`, so
nothing is enabled by its presence; and the seam-marker hypothesis in §6 rests entirely on
`segment_embeddings` and `per_segment_positions`, neither of which is implicated by §14. If
`frozen_input_kv` is ruled out, §7.1–§7.3 stand unchanged.
