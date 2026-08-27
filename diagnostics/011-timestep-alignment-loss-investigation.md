# 011 — Positional cross entropy versus delimiter-local SC2 semantics

Date: 2026-08-26
Status: investigation complete; no production change made.
Executable: `scripts/timestep_alignment_probe.py`
Generated artifacts: `scripts/output/timestep_alignment_probe/` (git-ignored)
Tests: `tests/test_timestep_alignment_probe.py` (48 tests)

## 1. Question

Production uniform-mode training uses class-weighted clean-state positionwise
cross entropy over every valid canvas position except clamped `[BOS]`. The canvas
grammar is

```
[BOS] [WIN|LOSS] (content* [DELIMITER])+ ([END] [PAD]* | [PAD]*)
```

and each delimiter-bounded group is one one-second SC2 timestep whose decoded
semantic state is the multiset of its content token types — `decode_canvas` does
not retain entity order inside a group.

The hypothesis under test: because the objective scores fixed serialized
coordinates, a single missing or extra content token shifts the expected index of
the remaining content and of that timestep's delimiter, so a prediction whose
delimiter-local semantic edit distance is small can still collect many
wrong-coordinate penalties. Those penalties would then push the model toward
frequent-token predictions and weak delimiter / `[END]` / semantic `[PAD]`
behaviour, especially at exact `t=1` where no truthful canvas token survives as
an alignment landmark.

Note on wording used throughout: the penalties are **additive across positions**,
never exponential. What one semantic insertion or deletion can do is create
*many* additive positionwise penalties before the sequence re-aligns. That
multiplicity is reported below as a counted amplification ratio.

## 2. Headline

**The hypothesis is only partly supported, and the specific mechanism it names is
not the dominant failure.**

1. The objective geometry does overcount, and by an unbounded amount when the
   model fails to re-align: one content deletion whose left shift runs to the end
   of the canvas costs **640 positional mismatches per semantic edit**
   (mean penalty span 2497 positions).
2. But when the shift is **bounded by the timestep's own delimiter** — the case
   the hypothesis is really about — the overcount is modest: **3.15 positional
   mismatches per semantic edit** in the maximal construction and **1.50** when
   the deletion offset is swept uniformly across the group. Canonical
   sort-by-entity-type serialization produces long runs of identical ids, and a
   one-position shift across such a run changes nothing at those coordinates.
3. On the real V3 EMA checkpoint the alignment component of the trained model's
   content error is **small at exactly `t=1` (6.1% of content CE)** and largest
   in the partially-anchored regime **`t=0.99` (22.1%)**. At `t=1` the
   model's content errors are mostly *wrong-token* errors, not misplaced-token
   errors.
4. What the same run *does* show, decisively, is a **structural generation
   failure that is not primarily an alignment artefact**: at `t=1` the model
   emits **19.1 delimiters where the target needs 168.3** (11.4%), delimiter CE
   is 3.15 nats against a content CE of 1.06, delimiter argmax accuracy on noised
   positions is **0.053**, and macro-F1 over the vocabulary collapses to **0.113**
   while plain accuracy stays at 0.695 — the signature of collapse onto frequent
   tokens. On terminal windows `[END]` collapses in exactly the same shape
   (accuracy 1.000 → 0.837 → 0.239 → **0.109** across the sweep), so this is one
   structural failure and not two rare-class problems.
5. The weak `[WIN]`/`[LOSS]` behaviour is **not** part of that failure. On
   terminal windows the same checkpoint scores outcome accuracy **1.000 at every
   corruption level**; the 0.560 seen on mid-game windows is a data-horizon
   effect, not a loss-geometry one.
6. A trivial **previous-ground-truth-timestep persistence baseline reaches
   multiset F1 0.9916** against the model's **0.7326 at `t=1`**. The model is far
   from the easy ceiling on exactly the delimiter-local semantics the alignment
   hypothesis cares about.
7. The model is **not** running on a learned marginal at `t=1`: replacing the
   clamped input with another window's input moves unweighted CE from 1.141 to
   **8.410**, and stripping enemy records while keeping input structure moves it
   to **7.844**. Conditioning is used heavily.

Conclusion: a narrowly specified training ablation is warranted, but it should
target **delimiter/termination structure and count calibration**, not
order-invariant content matching alone. See §8.

## 3. What was run

Primary run (bounded, real GPU, real recorded replay windows):

```
.venv\Scripts\python.exe scripts/timestep_alignment_probe.py \
  --config configs/smallTrainingTestV3.yaml \
  --checkpoint tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt \
  --split test --device cuda --num-workers 0 --seed 20260826 \
  --max-examples 48 --windows-per-replay 1 --geometry-canvases 24
```

Supplementary terminal-window run, taken because every mid-game window is
boundary-truncated and therefore contains no `[END]` target at all:

```
.venv\Scripts\python.exe scripts/timestep_alignment_probe.py \
  --config configs/smallTrainingTestV3.yaml \
  --checkpoint tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt \
  --split test --device cuda --num-workers 0 --seed 20260826 \
  --max-examples 46 --windows-per-replay 1 --window-position last \
  --skip-geometry --baseline-max-windows 0 \
  --output scripts/output/timestep_alignment_probe/smallTrainingTestV3-test-terminal.json
```

| item | value |
| --- | --- |
| config | `configs/smallTrainingTestV3.yaml` |
| checkpoint | `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt` |
| weights | EMA (the sampler's weight set; `--raw` is the explicit opt-out) |
| split | recorded `test`, **verified** against `tests/output/smallTrainingTestV3/metrics/replay_selection.json` (23 replays) |
| windows scored | 46 (23 replays × both perspectives), 753 480 scored canvas positions and 30 788 pooled target-timestep comparisons (15 640 p1 / 15 148 p2) |
| corruption levels | 0.75, 0.90, 0.99, 1.00, coupled across `t` |
| device | NVIDIA GeForce RTX 3070 (CUDA visible; no CPU fallback was used) |
| forward pass | one denoiser pass, `canvas_self_conditioning=None` — the sampler's first step, **not** iterative sampling |
| wall clock | 176 s primary run, 102 s terminal-window run |
| artifacts | `scripts/output/timestep_alignment_probe/smallTrainingTestV3-test.{json,summary.txt,per_window.csv}` and `...-test-terminal.{json,summary.txt,per_window.csv}` |

The sweep over `t` is **coupled**: an identically seeded `torch.Generator` is
rebuilt before every `corrupt_batch` call, so the Bernoulli draw and the
replacement tokens are shared across levels and the corrupted sets are nested.
The sweep therefore changes which truthful anchors survive rather than swapping
in unrelated random canvases.

Every ratio below is formed once from pooled numerators and denominators. No
per-window mean is ever averaged with another.

## 4. Experiment A — model-independent objective geometry

24 real clean target canvases from the test split. No model, no checkpoint, no
GPU. Deterministic pseudo-logits put probability 0.9 on the predicted token and
spread the rest uniformly over the remaining 290 ids, so every position pays
exactly `-ln 0.9 = 0.1054` nats when right and `7.9725` nats when wrong.

Edits are applied to the median-length qualifying timestep group (mean content
length 23.4) at its first content position. "Focus" columns scope the semantic
comparison to the edited group plus the group it can spill into; whole-canvas
multiset F1 is not shown because 24 untouched groups drown a one-group edit.

| case | intended semantic edits | positional mismatches | mismatches per edit | excess CE (nats) | penalty span | focus multiset F1 | focus edit distance | stops at intended delimiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact target | 0 | 0 | — | −0.01 | 0.0 | 1.0000 | 0 | 24/24 |
| content substitution | 24 | 24 | **1.00** | 188.80 | 1.0 | 0.9787 | 24 | 24/24 |
| content deletion, delimiter-bounded left shift | 48 | 151 | **3.15** | 1187.93 | 24.0 | 0.9787 | 48 | 24/24 |
| content insertion, delimiter-bounded right shift | 48 | 151 | **3.15** | 1187.93 | 24.0 | 0.9787 | 48 | 24/24 |
| delimiter displacement | 48 | 48 | **1.00** | 377.61 | 2.0 | 0.9787 | 48 | 24/24 |
| *control:* deletion, unbounded shift | 24 | 15 371 | **640.46** | 120 925.27 | 2497.2 | 0.9892 | 24 | n/a |

Sweeping one bounded deletion across every offset of the focus group:
group length 23.4, positional mismatches min 2 / mean 3.0 / max 8, i.e. a mean
amplification of **1.50 mismatches per semantic edit**.

### What this arm establishes

* **The overcount is real but bounded when the model re-aligns.** Three of the
  four perturbed cases have identical delimiter-local damage (focus multiset F1
  0.9787, focus edit distance 48 for the three two-edit cases) while their
  positional penalties differ by 3.15×. So the objective genuinely prices two
  semantically equivalent errors very differently.
* **Canonical serialization damps it.** SPEC §5 sorts a timestep's entities by
  type, so a timestep holding many `probe` tokens has long identical runs. A
  one-position shift across such a run costs nothing. This is why the bounded
  amplification is 1.5–3.2× rather than the ~23× a naive "whole group shifts"
  reading would predict. This is a property of the representation, not of the
  measurement, and it materially weakens the original hypothesis.
* **The catastrophic case is failure to re-align, not the shift itself.** The
  unbounded control costs 640 mismatches per edit over a 2497-position span. Any
  mechanism that makes a model re-establish the delimiter grid quickly is worth
  far more than any refinement of the within-group cost.

## 5. Experiment B — real V3 EMA checkpoint logits (observational)

### 5.1 Positional and alignment metrics, pooled over 46 windows

| slice | weighted objective | unweighted CE | noised argmax acc | content positional CE | oracle aligned CE | gap (nats) | gap fraction | multiset F1 | exact-multiset rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.6101 | 0.6129 | 0.8096 | 0.5663 | 0.4870 | 0.0792 | **13.99%** | 0.8618 | 0.3534 |
| t = 0.75 | 0.0370 | 0.0372 | 0.9823 | 0.0338 | 0.0318 | 0.0020 | 6.05% | 0.9927 | 0.8372 |
| t = 0.90 | 0.1860 | 0.1876 | 0.9223 | 0.1668 | 0.1392 | 0.0277 | 16.58% | 0.9670 | 0.4346 |
| t = 0.99 | 1.0807 | 1.0859 | 0.6925 | 1.0059 | 0.7834 | 0.2225 | **22.12%** | 0.7550 | 0.0712 |
| t = 1.00 | 1.1368 | 1.1409 | 0.6945 | 1.0585 | 0.9938 | 0.0646 | **6.11%** | 0.7326 | 0.0704 |

Macro-F1 on genuinely noised positions, by `t`: 0.8908 (0.75), 0.6679 (0.90),
**0.1255 (0.99), 0.1128 (1.00)**. Accuracy barely moves between 0.99 and 1.00
while macro-F1 halves relative to 0.90 — the frequent-token collapse the
hypothesis predicted is present, and it is present at both 0.99 and exactly 1.0.

Excluding the two dominant tokens (`probe`, `nexus`) changes the multiset picture
very little: F1 0.8405 overall, 0.6851 at `t=1`; exact-multiset rate excluding
dominants 0.0839 at `t=1`. Per-token count MAE at `t=1` is 2.29 occurrences per
(timestep, token-type) cell, and delimiter-local edit distance is 6.41 per
timestep.

By perspective (pooled, equal support): p1 gap fraction 13.41%, p2 14.60%. By
future distance the CE is flat: 0.579 (d=1), 0.567 (2–5), 0.593 (6–10), 0.649
(11–30), 0.753 (31+). There is no cliff at the input horizon.

**The alignment gap is largest where alignment is actually possible.** At
`t=0.99` a handful of truthful tokens survive, the model half-commits to a grid,
and 22% of its content CE is recoverable by reordering within a ground-truth
timestep span. At exactly `t=1` the gap falls to 6% — because the prediction is
not a misaligned version of the right content, it is largely the wrong content.

### 5.2 Structural metrics

| slice | outcome CE | `[WIN]`+`[LOSS]` mass | delimiter-count exact rate | mean abs delimiter drift | mean predicted / target delimiters |
| --- | --- | --- | --- | --- | --- |
| overall | 0.6370 | 0.9965 | 0.0598 | 106.4 | — |
| t = 0.75 | 0.5273 | 0.9981 | 0.2391 | 16.5 | 168.3 / 168.3 |
| t = 0.90 | 0.6113 | 0.9986 | 0.0000 | 173.8 | 157.0 / 168.3 |
| t = 0.99 | 0.7050 | 0.9958 | 0.0000 | 372.8 | 23.1 / 168.3 |
| t = 1.00 | 0.7041 | 0.9934 | 0.0000 | 19.3 | **19.1 / 168.3** |

Per class, pooled over every `t` (46 mid-game windows, 753 480 scored positions):

| class | scored positions | unweighted CE | noised argmax accuracy |
| --- | --- | --- | --- |
| enemy-observed | 177 536 | 0.4253 | 0.8678 |
| enemy-fogged | 175 632 | 0.4264 | 0.8669 |
| enemy-future | 365 632 | 0.7019 | 0.7894 |
| `[DELIMITER]` | 30 972 | 1.6201 | 0.4248 |
| semantic `[PAD]` | 3 524 | 1.2753 | 0.5173 |
| win/loss | 184 | 0.6370 | 0.5602 |

By `t`, the delimiter class degrades far faster than content: CE 0.094 → 0.591 →
2.942 → **3.152**, accuracy 0.952 → 0.743 → 0.059 → **0.053**. Semantic `[PAD]`
follows the same shape (CE 0.122 → 0.380 → 2.267 → 2.406). Outcome mass on the
`[WIN]`/`[LOSS]` pair stays at 0.993–0.998 at every level — the model always
knows an outcome token belongs at position 1 — but on these mid-game windows
outcome accuracy is 0.56, near chance. §5.4 shows that is a horizon effect, not a
loss effect: on terminal windows the same checkpoint is at 1.000.

The `19.1 / 168.3` row is the single most important number in this arm. At `t=1`
the model does not merely misplace the delimiter grid; it **does not emit one**.
The low mean absolute drift at `t=1` (19.3) is an artefact of pairing only the
first `min(count)` delimiters — with 19 predicted delimiters, only 19 pairs
exist, and they are the early ones that are easiest to place.

### 5.3 Windows whose delimiter count is right versus wrong

| delimiter count | scored positions | unweighted CE | noised acc | content CE | oracle CE | gap fraction | multiset F1 | exact-multiset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | 45 045 | 0.0404 | 0.9808 | 0.0367 | 0.0349 | **4.90%** | 0.9920 | 0.8151 |
| wrong | 708 435 | 0.6493 | 0.8007 | 0.5999 | 0.5158 | **14.02%** | 0.8535 | 0.3241 |

Getting the delimiter grid right and being nearly perfect are the same event in
this data — every window with a correct delimiter count came from `t = 0.75`.
This is a correlation, not a mechanism: it says the delimiter grid survives
exactly when enough truthful canvas tokens survive to carry it, not that fixing
delimiters would fix content.

### 5.4 Terminal windows — `[END]`, semantic `[PAD]`, and the outcome

All 46 mid-game windows are boundary-truncated (canvas budget exhausted before
game end), so the primary run scored **zero** `[END]` positions. The
supplementary `--window-position last` run (46 terminal windows, same 23 replays,
102 s) exists solely to reach them.

| class | t = 0.75 | t = 0.90 | t = 0.99 | t = 1.00 |
| --- | --- | --- | --- | --- |
| `[END]` CE / acc (46 positions per level) | 0.038 / 1.000 | 0.457 / 0.837 | 2.537 / 0.239 | **3.969 / 0.109** |
| `[DELIMITER]` CE / acc (525) | 0.470 / 0.776 | 1.355 / 0.421 | 2.972 / 0.057 | **3.560 / 0.029** |
| semantic `[PAD]` CE / acc (160 517) | 0.003 / 0.999 | 0.011 / 0.997 | 0.243 / 0.934 | 1.295 / 0.658 |
| win/loss CE / acc (46) | 0.007 / **1.000** | 0.015 / **1.000** | 0.042 / **1.000** | 0.040 / **1.000** |

Two things stand out.

* **`[END]` fails exactly like `[DELIMITER]`.** Both are near-perfect while
  truthful anchors survive and both collapse to ~0.03–0.11 accuracy at `t≥0.99`.
  This is one structural-generation failure, not two independent rare-class
  problems, and it is the same failure as the 19/168 delimiter deficit in §5.2.
* **The outcome is not a rare-class failure at all when the endgame is visible.**
  Terminal windows give win/loss accuracy 1.000 and CE 0.026 at every corruption
  level, against 0.560 / 0.637 on mid-game windows. The model reads the outcome
  off the input when the input contains the end of the game and is near chance
  when it does not. That is a data-horizon property, not a loss-geometry one.

Caveat on the terminal-window controls: those canvases are ~96% semantic `[PAD]`,
so the enemy-stripped arm scores *better* than the correct-input arm there
(CE 1.260 versus 1.363) simply because predicting `[PAD]` everywhere is nearly
free. Only the mid-game control numbers in §6 are interpretable as conditioning
sensitivity.

## 6. Experiment C — controls

All three conditioning arms see a **byte-identical** noised canvas, target, and
corruption draw at `t = 1.00`. Only the clamped input differs.

| arm | unweighted CE | noised argmax acc | multiset F1 | exact-multiset |
| --- | --- | --- | --- | --- |
| correct input | 1.1409 | 0.6945 | 0.7326 | 0.0704 |
| input rows shuffled between examples | 8.4101 | 0.1180 | 0.1245 | 0.0092 |
| enemy content removed, structure retained | 7.8441 | 0.0085 | 0.0002 | 0.0000 |

Model-free comparators on the same target timesteps:

| baseline | multiset F1 | exact-multiset | count MAE |
| --- | --- | --- | --- |
| previous ground-truth timestep (persistence) | **0.9916** | 0.7771 | 0.0715 |
| train-split constant most-frequent content token (`scv`) | 0.2366 | 0.0000 | 5.8384 |
| model at `t = 1.00` | 0.7326 | 0.0704 | 2.2932 |

Reading these honestly:

* Input-shuffle sensitivity is **enormous** (1.14 → 8.41 nats). Whatever the
  model is doing at `t=1`, it is not emitting a learned marginal. This measures
  conditioning use only; it is not by itself evidence of a loss defect.
* Enemy stripping is **off-distribution**: training fog tops out at 0.8, so a
  zero-enemy input is a condition the model never saw. Its 7.84 nats therefore
  bounds "the enemy evidence matters" but should not be read as a calibrated
  degradation curve.
* The persistence baseline reads one step of ground truth, so it is
  oracle-flavoured and not a legitimate model. Its value is as a ceiling: SC2
  state is slow-moving, delimiter-local multisets are highly predictable, and the
  model captures far less of that structure than the representation allows.

## 7. Interpretation contract — what is and is not established

* **Established (model-independent):** the serialized coordinate objective can
  overcount a small semantic edit. Bounded within a timestep it overcounts by
  1.5–3.2× per edit; unbounded it overcounts by ~640× per edit over ~2500
  positions. Substitutions and delimiter displacements are priced correctly at
  1.0×.
* **Established (observational):** the trained model's current content errors
  have an alignment component — 14% overall, 22% at `t=0.99`, 6% at exactly
  `t=1`.
* **NOT established:** that the alignment objective *caused* the observed
  frequent-token behaviour. The probed checkpoint was itself trained under
  positional CE, so a low aligned score on its logits cannot separate "the loss
  shaped this" from "this model is simply weak at high corruption". Neither
  result, alone or together, supports a causal claim.
* **NOT established:** that an alignment-aware objective would improve the
  learned model. The oracle aligned score is deliberately optimistic — it is
  order-invariant inside a ground-truth timestep span, is handed the true span
  boundaries, never has to generate a delimiter sequence, and is not a
  likelihood, not the production loss, and not a proposal to train a matching
  algorithm.
* **NOT established by the shuffle control:** input-shuffle sensitivity measures
  how much the prediction depends on its particular conditioning. High
  sensitivity is evidence against the "collapsed to a marginal" reading; it says
  nothing about whether the loss is well specified.
* **Only a matched training ablation** — same data, same splits, same schedule,
  same step budget, one objective change, measured against the positional-CE
  baseline arm — can establish whether an alignment-aware objective improves the
  learned model.

## 8. Recommendation

A training ablation is warranted, but the evidence redirects it. The dominant
measured failure at high corruption is **structural**: the model emits 11% of the
required delimiters at `t=1`, `[END]` accuracy falls to 0.109, its delimiter class
CE is 3× its content CE, and it sits far below a trivial persistence ceiling on
delimiter-local multisets. The within-timestep alignment overcount, which the
original hypothesis named, is real but modest (1.5–3.2×) because canonical
sorting already absorbs most of it.

There is also a plain arithmetic reason the delimiter and `[END]` classes are
starved that has nothing to do with alignment: a mid-game window carries ~168
delimiters and one `[END]` against ~4 000 content positions, and the V3 profile
weights `[DELIMITER]` at 1.0 while weighting `[END]` at 24.63. Any ablation in
this area should measure a re-weighting arm alongside the structural-loss arm, or
it will not be able to say which knob moved the result.

Candidate families, stated without preference. None is implemented here.

### (a) Keep positional CE; add a delimiter-local count/multiset auxiliary

* **Compatibility:** high. The fixed-coordinate uniform-diffusion objective is
  untouched; the auxiliary is an additional weighted term over ground-truth
  timestep spans, which are already parsed by this diagnostic.
* **Cost:** low. One extra reduction per span; no assignment solve.
* **Preserved clean anchors:** unaffected — the auxiliary can be masked to
  genuinely noised positions the same way the canvas-state split already is.
* **Delimiter generation:** **not addressed.** A count auxiliary over
  ground-truth spans does not teach the model to *emit* a delimiter grid, which
  is the measured failure. Would need pairing with (d) or an explicit delimiter
  term.
* **Duplicate tokens:** handled naturally — counts are the representation.
* **Checkpoint/architecture:** no change; ablation arms remain loadable.

### (b) Strict structural CE + assignment/optimal-transport content loss on noised positions

* **Compatibility:** medium. Structural targets (outcome, `[DELIMITER]`,
  `[END]`, semantic `[PAD]`) keep ordinary positionwise CE; content spans get an
  order-invariant transport cost. Requires span boundaries, which at training
  time are the ground-truth delimiters — so the *loss* knows the grid the *model*
  must still generate. That asymmetry is the main scientific risk.
* **Cost:** the exact assignment used in this diagnostic is O(n²) NumPy calls per
  span at n≈23–200 content tokens; a differentiable Sinkhorn/entropic transport
  is the practical training form and adds several iterations per span per step.
* **Preserved clean anchors:** must be handled deliberately. A surviving truthful
  token at a coordinate is information the transport cost would discard;
  restricting the transport to genuinely noised positions inside each span is the
  obvious guard and matches the existing `changed_positions` split.
* **Delimiter generation:** not addressed by the content term; the structural CE
  half must carry it.
* **Duplicate tokens:** must treat duplicate occurrences as distinct demand, as
  this diagnostic's assignment does, or a single confident slot can pay for an
  arbitrary count.
* **Checkpoint/architecture:** no architecture change; loss-only, so
  `architecture_identity` is unaffected and arms stay comparable.

### (c) CTC or another differentiable alignment marginalization

* **Compatibility:** **low with the current process.** CTC marginalizes over
  monotonic alignments of a *variable-length* output to a target label sequence.
  Uniform discrete diffusion predicts a clean state at every fixed coordinate
  simultaneously, and the sampler renoises those same coordinates; there is no
  emission-time axis to marginalize over, and a CTC-style blank would collide
  with the semantic `[PAD]` token. Adopting it would mean changing the process,
  not the loss.
* **Cost:** O(T·L) forward-backward per example.
* **Preserved clean anchors:** poorly served — CTC has no natural way to say
  "this coordinate is already correct, keep it".
* **Delimiter generation:** would be addressed, since delimiters become ordinary
  labels in the target sequence.
* **Duplicate tokens:** the classic CTC weakness (repeated labels need blanks
  between them), and SC2 timesteps are full of repeats.
* **Checkpoint/architecture:** likely a new `ARCHITECTURE_ID` boundary. This is
  the most invasive family and is listed for completeness, not as a near-term arm.

### (d) Representation change: put semantically stable quantities at stable coordinates

* **Compatibility:** this is a §12-adjacent change to the canvas grammar and
  therefore **owner territory, not an agent decision.** Examples would be a
  fixed-width per-timestep count vector, or an explicit timestep-membership
  encoding (which SPEC §12 lists as OPEN and explicitly forbids implementing).
* **Cost:** preprocessing and serializer rewrite; possibly a different canvas
  budget.
* **Preserved clean anchors:** improved — a count at a fixed coordinate cannot be
  shifted by a neighbouring insertion at all, which removes the overcount by
  construction rather than by re-weighting it.
* **Delimiter generation:** removed as a problem, because the grid stops being
  something the model has to emit.
* **Duplicate tokens:** subsumed into counts.
* **Checkpoint/architecture:** breaks vocabulary/manifest stamps and every
  existing checkpoint. Highest cost, cleanest fix.

**Suggested narrow first arm, if one is authorised:** family (a) plus an explicit
delimiter/termination term, measured against the unmodified V3 profile on the
same 50-epoch budget, with the primary readouts being predicted-versus-target
delimiter count at `t=1`, delimiter class CE by `t`, macro-F1 on noised positions
at `t≥0.99`, and delimiter-local multiset F1 against the 0.9916 persistence
ceiling, plus `[END]` accuracy at `t≥0.99` on terminal windows. That arm tests
the failure this diagnostic actually measured. It is **not** implemented here and
requires owner approval; it must ship as a config toggle defaulting to the
current behaviour and be measured against its baseline arm before being trusted.

## 9. Limitations

* 46 mid-game windows across 23 test replays. Bounded by design; `--max-examples`
  reproduces a smaller version and larger sweeps are available.
* Zero terminal `[END]` targets in the primary run (every mid-game window is
  boundary-truncated). The supplementary `--window-position last` run is the only
  `[END]` evidence and carries 46 `[END]` positions per corruption level — enough
  to see the collapse shape, not enough for a precise number.
* Terminal-window canvases are ~96% semantic `[PAD]`, which makes their aggregate
  CE and their control arms non-comparable with the mid-game run. They are used
  only for the per-class rows in §5.4.
* The persistence baseline reads one step of ground truth and is therefore a
  ceiling, not a competitor.
* The enemy-stripped control is off-distribution relative to the ≤0.8 training
  fog.
* One denoiser forward pass per level, not iterative sampling. Nothing here
  characterizes the sampler; prompt 010's `diagnostics/010-uniform-sampler-final-state.md`
  owns that.
* The oracle aligned score is capped at `--oracle-max-span` (1024) content tokens
  per span; no span in this corpus reached the cap, so coverage was complete.
* Observational with respect to the checkpoint, as stated in §7.

## 10. Reproduction

Fast smoke (no checkpoint, no GPU):

```
.venv\Scripts\python.exe scripts/timestep_alignment_probe.py --geometry-only --max-examples 12 --geometry-canvases 12
```

Focused tests:

```
.venv\Scripts\python.exe -m pytest tests/test_timestep_alignment_probe.py -q
```

Full commands and provenance for the GPU runs are in §3; every artifact carries
its own repository-relative provenance block including the config SHA-256, the
manifest metadata, the verified split, and the exact selected window IDs.
