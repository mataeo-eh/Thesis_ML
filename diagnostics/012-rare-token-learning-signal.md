# 012 — Where the alignment overcount actually lands: the rare-token learning signal

Date: 2026-08-27
Status: investigation complete; no production change made.
Executable: `scripts/rare_token_signal_probe.py`
Generated artifacts: `scripts/output/rare_token_signal_probe/` (git-ignored)
Tests: `tests/test_rare_token_signal_probe.py` (20 tests)
Predecessor: `diagnostics/011-timestep-alignment-loss-investigation.md`

## 1. Why this exists

011 measured the positional-CE alignment overcount **pooled over every content
position** and concluded it was modest — 1.5–3.2 wrong coordinates per semantic
edit — with this reasoning:

> Canonical serialization damps it. SPEC §5 sorts a timestep's entities by type,
> so a timestep holding many `probe` tokens has long identical runs. A
> one-position shift across such a run costs nothing.

That reasoning is correct and its conclusion does not follow. A one-position
shift mis-scores a coordinate **exactly when the token there differs from its
neighbour** — that is, exactly at a **run boundary**. So the damping is not the
objective being gentle. It is the objective concentrating the entire cost of a
shift onto the boundaries, and a token type that occurs once in a timestep is
*all* boundary while a type that occurs 18 times is 17/18 interior.

The pooled amplification is small **because** the damage is concentrated on the
rare types, and a pooled metric is structurally incapable of showing that. This
diagnostic disaggregates it by token type.

The rare types are the semantically pivotal ones — tech-unlock structures, whose
presence is what tells you what an opponent can build next. `SPEC.md` §5 sorts a
timestep by SC2 source (unit-type) ID, and for Protoss the numerous economy
structures nexus(59), pylon(60), assimilator(61), gateway(62) sort **before**
every tech building: fleetbeacon(64), twilightcouncil(65), photoncannon(66),
stargate(67), templararchive(68), darkshrine(69), roboticsbay(70),
roboticsfacility(71), cyberneticscore(72). One pylon miscount displaces every
tech building in that timestep. (`tests/test_rare_token_signal_probe.py::
test_tech_buildings_sort_after_protoss_economy_structures` pins this ordering.)

## 2. Headline

**The original concern is confirmed, and the mechanism is sharper than the
concern stated it.** The model has strong per-timestep knowledge of the rare tech
tokens and the positional objective never converts that knowledge into an
emitted token.

1. **The overcount is concentrated on exactly the tokens that matter.** Pooled
   boundary exposure — the probability that a type pays a wrong-coordinate
   penalty when an upstream count error reaches it — is **0.0582 for the three
   dominant worker tokens and 0.8475 for tech buildings, a 14.6× differential**.
   Every singleton tech building (`cyberneticscore`, `spawningpool`, `factory`,
   `forge`, `roachwarren`, `roboticsfacility`, `twilightcouncil`, `stargate`,
   `armory`) is at **1.0000** — total exposure. This is model-independent.
2. **The trained model's rare-token recall is a clean frequency-ordered
   collapse.** At `t=1`, positional recall by occurrences-per-timestep:
   dominant (≥4/step) **0.9934**, uncommon (0.25–1.5) **0.0650**, rare
   (0.05–0.25) **0.0158**, ultra-rare (<0.05) **0.0000**, tech buildings
   **0.0000**. 011's pooled "noised argmax accuracy 0.6945" is carried almost
   entirely by `probe`/`drone`/`scv`, which are 71% of content positions.
3. **It is not a placement problem — the token is absent, not misplaced.**
   Forgiving alignment entirely (does argmax emit the type *anywhere* in its
   ground-truth timestep span?) moves tech-building recall from 0.0000 to
   **0.0003**. Order-invariant scoring recovers nothing.
4. **But the knowledge is unambiguously present in the distribution.** The model
   puts **110.8× more soft probability mass** on a tech building inside timesteps
   that actually contain it than inside timesteps that do not, and its expected
   count over the span is **0.864 of the true count**. It knows which second a
   cybernetics core exists in, with 38× discrimination for that token
   specifically, and never once writes it down.
5. **The failure shape is dilution, not washout.** ~0.86 units of belief spread
   over a ~23-slot span is ~0.037 per slot; argmax goes to a worker every time.
   The model cannot resolve *which coordinate*, because the coordinate is a
   function of the exact count of the ~5.4 preceding tokens it also cannot pin
   down.
6. **The owner's "numerically small" intuition is half right and worth
   correcting.** Tech buildings are 3.8% of content positions but carry
   **10.6% of the weighted training objective** — the loss on them is *not*
   small. It is large, persistent, and irreducible under this parameterization,
   because no amount of gradient on "put `cyberneticscore` at index 2417" is
   learnable when index 2417 depends on an upstream count. The signal is not
   diluted away; it is spent on an unanswerable question.

## 3. What was run

```
.venv\Scripts\python.exe scripts/rare_token_signal_probe.py \
  --config configs/smallTrainingTestV3.yaml \
  --checkpoint tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt \
  --split test --device cuda --num-workers 0 --seed 20260827 \
  --max-examples 48 --windows-per-replay 1 --exposure-canvases 24
```

| item | value |
| --- | --- |
| config | `configs/smallTrainingTestV3.yaml` |
| checkpoint | `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt` |
| weights | EMA (`--raw` is the explicit opt-out) |
| split | recorded `test`, **verified** against `tests/output/smallTrainingTestV3/metrics/replay_selection.json` (23 replays) |
| windows scored | 46 (23 replays × both perspectives), same selection as 011 |
| Arm A scope | 24 canvases, 3 925 timesteps, 93 754 content occurrences |
| Arm B scope | 179 669 scored content target positions per corruption level |
| corruption levels | 0.75, 0.90, 0.99, 1.00, coupled across `t` exactly as in 011 |
| device | NVIDIA GeForce RTX 3070 (CUDA visible; no CPU fallback) |
| forward pass | one denoiser pass, `canvas_self_conditioning=None` — the sampler's first step, **not** iterative sampling |
| wall clock | 65 s |

Every ratio is formed once from pooled numerators and denominators; no per-window
mean is averaged with another. The corruption sweep is coupled — an identically
seeded generator is rebuilt before every `corrupt_batch` call — so the sweep
changes which truthful anchors survive rather than swapping in unrelated
canvases.

## 4. Arm A — model-independent: where a one-position shift lands

No model, no checkpoint, no GPU. Closed form over real clean target canvases.

A deletion at offset `k` in a timestep makes the prediction read
`pred[j] = content[j+1]` for every `j ≥ k`, with the group's `[DELIMITER]`
shifting into the final content slot. Coordinate `j` is mis-scored **iff**
`content[j+1] != content[j]`. Two exposure measures follow:

* `boundary_fraction` — the fraction of a type's occurrences that sit at a run
  boundary, i.e. **P(this type pays a wrong-coordinate penalty | a shift reaches
  it)**.
* `expected_hits` — the same event marginalized over a deletion offset drawn
  uniformly over the timestep's slots (equivalently: deleting an occurrence
  chosen in proportion to token frequency, which is the realistic error). This
  accounts for the fact that tech buildings sort early and so fewer deletions
  reach them at all.

| group | types | occurrences | occ/timestep | mean run length | **boundary_fraction** | expected_hits | mean prefix count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dominant (≥4/step: `probe`, `drone`, `scv`) | 3 | 67 414 | 17.18 | 17.86 | **0.0582** | 0.0543 | 13.9 |
| all content | 40 | 93 754 | 23.89 | 13.43 | 0.2264 | 0.0926 | 12.3 |
| tech buildings | 11 | 3 469 | 0.88 | 1.41 | **0.8475** | 0.1891 | 5.4 |
| ultra-rare (<0.05/step) | 17 | 1 513 | 0.39 | 1.35 | **0.8843** | 0.4452 | 15.5 |

**Boundary-exposure ratio, tech / dominant: 14.6×. Unconditional: 3.5×.**

Selected individual types:

| token | occurrences | mean run length | boundary_fraction | expected_hits | mean prefix count |
| --- | --- | --- | --- | --- | --- |
| `probe` | 29 686 | 18.91 | 0.0550 | 0.0549 | 15.3 |
| `drone` | 20 865 | 17.09 | 0.0610 | 0.0495 | 11.7 |
| `scv` | 16 863 | 16.97 | 0.0605 | 0.0590 | 14.0 |
| `pylon` | 2 466 | 2.05 | 0.5880 | 0.0656 | 1.7 |
| `spawningpool` | 719 | 1.00 | **1.0000** | 0.1299 | 2.6 |
| `cyberneticscore` | 670 | 1.00 | **1.0000** | 0.2711 | 7.3 |
| `factory` | 215 | 1.00 | **1.0000** | 0.2668 | 7.1 |
| `forge` | 117 | 1.00 | **1.0000** | 0.2204 | 5.1 |
| `roboticsfacility` | 72 | 1.00 | **1.0000** | 0.2550 | 7.8 |
| `twilightcouncil` | 70 | 1.00 | **1.0000** | 0.2731 | 7.5 |
| `stargate` | 14 | 1.00 | **1.0000** | 0.2440 | 7.9 |

### What this arm establishes

* 011's damping claim is true and its reading was wrong. Canonical
  sort-by-type does not reduce the cost of a shift; it **relocates** the entire
  cost onto run boundaries. Every singleton type is 100% boundary.
* The `mean prefix count` column is the credit-assignment statement.
  `cyberneticscore` sits on average 7.3 content tokens into its timestep. For it
  to land on its trained coordinate, the model must reproduce that prefix count
  **exactly** — every preceding nexus, pylon, assimilator and gateway. The
  learning signal for "a cybernetics core exists here" is therefore conditioned
  on an arithmetic problem that has nothing to do with the semantics.
* This is a property of the serialization plus the objective. It cannot be
  confounded by the checkpoint having been trained under that objective.

## 5. Arm B — observational: the trained model's rare-token signal

Same logits and targets scored at three increasingly forgiving levels, which
discriminate the explanations a single accuracy number confounds:

| observed pattern | interpretation |
| --- | --- |
| positional low, **timestep high** | misplacement — an alignment problem |
| positional low, timestep low, **soft mass elevated** | dilution — knows it, loses argmax |
| positional low, timestep low, soft mass ≈ 0 | washed out — no signal survived |

Columns: `position` = argmax right at the exact coordinate (the production view);
`timestep` = argmax emits the type anywhere in its ground-truth span (alignment
forgiven entirely); `presence` = fraction of spans containing the type where it
is emitted at least once; `p@target` = mean probability on the type at its true
coordinate; `exp/target` = the model's expected count over the span / true count;
`pres/abs` = soft rate inside spans that contain the type / spans that do not
(1.0 would mean a sprayed base rate with no timestep-level knowledge).

### 5.1 By frequency, at `t = 1.00`

| bucket | types | targets | position | timestep | presence | p@target | exp/target | **pres/abs** | CE | loss share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dominant ≥4/step | 3 | 127 879 | **0.9934** | 0.9990 | 1.0000 | 0.7182 | 0.967 | 11 339 | 0.348 | 0.209 |
| uncommon 0.25–1.5 | 12 | 40 528 | 0.0650 | 0.0912 | 0.1070 | 0.1205 | 0.918 | 1 086 | 2.577 | 0.489 |
| rare 0.05–0.25 | 8 | 8 278 | 0.0158 | 0.0161 | 0.0198 | 0.0498 | 0.772 | 30.9 | 3.428 | 0.133 |
| ultra-rare <0.05 | 19 | 2 984 | **0.0000** | 0.0000 | 0.0000 | 0.0219 | 0.445 | 104.6 | 4.292 | 0.060 |
| **tech buildings** | 11 | 6 845 | **0.0000** | **0.0003** | 0.0003 | 0.0470 | **0.864** | **110.8** | 3.317 | **0.106** |
| all content | 42 | 179 669 | 0.7224 | 0.7324 | 0.7368 | 0.5410 | 0.938 | 8 318 | 1.058 | 0.891 |

### 5.2 The same buckets across the corruption sweep (positional recall)

| bucket | t=0.75 | t=0.90 | t=0.99 | t=1.00 |
| --- | --- | --- | --- | --- |
| dominant ≥4/step | 0.9964 | 0.9843 | 0.9838 | **0.9934** |
| uncommon 0.25–1.5 | 0.9740 | 0.8436 | 0.1046 | 0.0650 |
| rare 0.05–0.25 | 0.9500 | 0.7588 | 0.0194 | 0.0158 |
| ultra-rare <0.05 | 0.9514 | 0.7292 | 0.0013 | **0.0000** |
| tech buildings | 0.9652 | 0.7832 | 0.0137 | **0.0000** |

The dominant tokens are **essentially unaffected** by the corruption sweep —
0.9964 at `t=0.75` and 0.9934 at `t=1.00`. Everything else falls off a cliff
between 0.90 and 0.99, precisely where the surviving truthful anchors that carry
the coordinate grid disappear. The pooled 011 number sits on top of two
completely different behaviours.

### 5.3 Tech buildings individually, at `t = 1.00`

| token | targets | position | timestep | presence | p@target | exp/target | pres/abs | CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `barracks` | 2 759 | 0.0000 | 0.0000 | 0.0000 | 0.0698 | 0.980 | 172.0 | 2.809 |
| `spawningpool` | 1 660 | 0.0000 | 0.0012 | 0.0012 | 0.0405 | 0.933 | 43.8 | 3.306 |
| `cyberneticscore` | 933 | 0.0000 | 0.0000 | 0.0000 | 0.0285 | 0.853 | 38.4 | 3.621 |
| `factory` | 571 | 0.0000 | 0.0000 | 0.0000 | 0.0204 | 0.609 | 27.7 | 3.965 |
| `barrackstechlab` | 258 | 0.0000 | 0.0000 | 0.0000 | 0.0338 | 0.515 | 24.0 | 3.630 |
| `roachwarren` | 211 | 0.0000 | 0.0000 | 0.0000 | 0.0180 | 0.499 | 28.3 | 4.344 |
| `evolutionchamber` | 154 | 0.0000 | 0.0000 | 0.0000 | 0.0419 | 0.929 | 780.6 | 3.231 |
| `forge` | 117 | 0.0000 | 0.0000 | 0.0000 | 0.0349 | 0.990 | 214.4 | 3.363 |
| `roboticsfacility` | 72 | 0.0000 | 0.0000 | 0.0000 | 0.0043 | 0.146 | 11.9 | 5.547 |
| `twilightcouncil` | 70 | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 0.039 | 9.0 | 6.772 |
| `starport` | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0066 | 0.217 | 15.5 | 5.094 |

Read the `barracks` row carefully. 2 759 target occurrences. The model's expected
count over the span is **0.980 of the true count** and it discriminates
containing from non-containing timesteps by **172×**. It emits the token
**zero times**. That is not a model that has failed to learn what a barracks is
or when one exists. It is a model whose belief cannot be expressed at the
resolution the objective demands.

The two rows where `exp/target` is genuinely low — `roboticsfacility` (0.146)
and `twilightcouncil` (0.039) — still discriminate at 11.9× and 9.0×. Those are
the types with both the least data and the deepest prefix (7.8 and 7.5 preceding
tokens); they are the tail of the same effect, not a different one.

## 6. Interpretation contract — what is and is not established

* **Established (model-independent):** canonical sort-by-type serialization does
  not reduce the cost of a one-position shift, it relocates that cost entirely
  onto run boundaries. Pooled boundary exposure is 14.6× higher for tech
  buildings than for the dominant worker tokens, and is exactly 1.0 for every
  singleton type. 011's pooled 1.5–3.2× figure is an average over a population
  whose members differ by more than an order of magnitude.
* **Established (observational):** the trained checkpoint's rare-token failure at
  `t ≥ 0.99` is a **placement** failure, not a knowledge failure. Per-timestep
  discrimination is 110.8× for tech buildings while emission is 0.0000. Order
  -invariant scoring inside the true span recovers essentially nothing, so the
  token is absent rather than misplaced *within* the span.
* **Established (accounting):** tech buildings carry 10.6% of the weighted
  objective from 3.8% of content positions. The rare-token loss is not small.
* **NOT established:** that positional CE *caused* this. The probed checkpoint
  was trained under positional CE, so this arm identifies **which failure shape**
  the model is in — which is what determines what an ablation should change —
  but not that the objective produced it. A model that simply lacks capacity for
  rare types would look similar on recall; it would **not** be expected to show
  110× per-timestep discrimination while emitting nothing, which is why the
  soft-mass control was added, but that is an argument, not a proof.
* **NOT established:** that any specific alternative objective fixes it. The
  soft-mass and order-invariant scores are deliberately optimistic diagnostics —
  they are handed the true span boundaries and never have to generate a
  delimiter grid.
* **Only a matched training ablation** — same data, splits, schedule and step
  budget, one objective change, measured against the positional-CE baseline arm —
  can establish causation.

### Relationship to 011's conclusion

011 §8 recommended an ablation targeting **delimiter/termination structure**,
having found the model emits 19.1 of 168.3 required delimiters at `t=1`. That
finding stands and is not contradicted here. The two are the same failure viewed
at two scales: the model cannot place a delimiter for the same reason it cannot
place a cybernetics core — both coordinates are determined by an upstream running
count it cannot reproduce. 011 was right that the delimiter grid is the
prerequisite. It was wrong to conclude that the within-timestep overcount is
"modest ... because canonical sorting already absorbs most of it"; that
absorption is the harm, and it falls on the tokens the thesis is about.

## 7. Recommendation

A training ablation is warranted. Nothing here changes 011's judgement that
the delimiter grid must be fixed first — a content-side fix that assumes
ground-truth span boundaries would be measuring a condition the sampler never
reaches. What this diagnostic changes is the **content-side arm's target and its
primary readout**.

The dilution finding rules one candidate family in and one out:

* **Ruled out as a first arm: order-invariant / assignment / optimal-transport
  content losses** (011 family (b)). Forgiving order inside the true span
  recovers 0.0003 recall for tech buildings. There is nothing for an assignment
  to match, because the token is not being emitted anywhere. A transport cost
  would change the price of an error the model is not making.
* **Ruled in: anything that lets a per-timestep belief be expressed without
  resolving a coordinate.** The model already holds the belief at 110×
  discrimination. Both 011 family (a) — a delimiter-local count/multiset
  auxiliary — and 011 family (d) — putting semantically stable quantities at
  stable coordinates — act on exactly that. Family (d) removes the problem by
  construction and is `SPEC.md` §12-adjacent, therefore owner territory and not
  an agent decision.

**Suggested narrow first arm, if one is authorised:** 011's recommended
delimiter/termination term plus a delimiter-local **count** auxiliary over
ground-truth timestep spans, shipped as a config toggle defaulting to current
behaviour. Add to 011 §8's readouts:

* positional recall by frequency bucket at `t ≥ 0.99` — the primary readout, and
  the one that must move; the pooled accuracy will not show it
* tech-building emission rate at `t = 1` (currently 0.0000)
* `pres/abs` discrimination (currently 110.8×) — this should *stay high*; a drop
  would mean the auxiliary bought placement by destroying knowledge
* `exp/target` calibration (currently 0.864)

A per-token or inverse-frequency **weighting** arm should be measured alongside
it, as 011 §8 already argued for the structural classes. The evidence here is
that weighting alone is unlikely to be sufficient — the objective is already
spending 10.6% of its mass on tech buildings and getting 0.0000 recall for it —
but without that arm an improvement cannot be attributed to the auxiliary rather
than to the reweighting it implies.

## 8. Limitations

* 46 windows across 23 test replays, same selection as 011. Bounded by design;
  `--max-examples` scales it either way.
* Mid-game windows only. Every one is boundary-truncated, so this run scores no
  terminal `[END]`; 011 §5.4 owns that measurement.
* One denoiser forward pass per level, not iterative sampling. Nothing here
  characterizes the sampler; `diagnostics/010-uniform-sampler-final-state.md`
  owns that.
* `exp/target` and `pres/abs` are computed over ground-truth spans, which the
  model does not have at inference. They measure the belief the model holds, not
  a quantity it could act on unaided.
* Arm A models a single one-position deletion. Real errors compound, and the
  compound case is 011's unbounded control (640 mismatches per edit).
* The tech-building set is a stated editorial choice, not a frequency cut. The
  frequency buckets are the objective version of the same claim and show the same
  gradient.
* Observational with respect to the checkpoint, as stated in §6.

## 9. Reproduction

Model-independent arm only (no checkpoint, no GPU, ~1 s):

```
.venv\Scripts\python.exe scripts/rare_token_signal_probe.py --exposure-only \
  --max-examples 24 --exposure-canvases 24
```

Focused tests:

```
.venv\Scripts\python.exe -m pytest tests/test_rare_token_signal_probe.py -q
```

The full GPU command and provenance are in §3; the JSON artifact carries the
verified split and the exact selected window IDs.
