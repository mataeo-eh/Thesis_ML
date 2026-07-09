<objective>
Rework the training pipelines in six coordinated ways, then version the smoke-test harness from `smallTrainingTest` to `smallTrainingTestV2`:

1. **Pre-training gets NO input — literally absent, not "empty" — and drops the fogged/observed paradigm entirely.** The pre-training pipeline (`debut_mode=false`) currently conditions on a clamped input sequence (`self_block + fogged_enemy_block`). Remove it: during pre-training the model's transformer sequence must be 100% output canvas and nothing else. No input positions, no separator/boundary/BOS tokens, no reserved or padded input segment, no input segment embeddings, no input attention-mask columns — the sequence the backbone sees IS the noised canvas, full stop, with RoPE/positional treatment starting at the first canvas token exactly as if the concept of an input never existed. (Zero-length input tensors are acceptable purely as internal plumbing through `DatasetExample`/collate IF they contribute zero sequence positions and zero computation artifacts to the model — the current `forward` concatenates input and canvas embeddings with no separator, so a `[B, 0]` input yields exactly a canvas-only sequence; prefer skipping `embed_input` entirely when the input is absent.) With no input, the fogged/observed/future distinction is meaningless in pre-training, so remove it from pre-training completely: no fog sampling, no fogged/observed bookkeeping, no input-side metrics recorded, and a collapsed class taxonomy (see change 2). Pre-training becomes the published MDLM/LLaDA objective: learn the prior distribution of the data by pure reconstruction. Input/prompt alignment is deferred to fine-tuning. Fine-tuning KEEPS the fog paradigm and its debut class taxonomy.
2. **Pre-training loss goes fully uniform, the published way — reflected in the config, not overridden.** In pre-training, every non-PAD canvas token (content, delimiter, [END], win/loss) gets class weight 1.0; PAD stays excluded. Collapse the three enemy classes (observed/fogged/future) into a single content class for pre-training (keep class ids stable — do not renumber; rename rather than restructure where possible so fine-tuning's 7-class debut taxonomy is untouched). Restructure the config to REFLECT this reversion: pre-training YAML configs must no longer carry the fogged/observed-era class-weight knobs (`enemy_observed_reconstruction`, `enemy_fogged_reconstruction`, `enemy_future_prediction`) or fog-rate settings — those become fine-tuning-only concerns in the schema. Do not leave dead knobs in pre-training configs that silently do nothing.
3. **Oversample t=1.0 in BOTH pipelines, with per-epoch reseeding.** At inference the canvas is essentially always 100% masked, but uniform t sampling almost never trains that region. Make 10% of examples per epoch (per-example Bernoulli, in expectation) use t=1.0 exactly; the remaining 90% keep the existing uniform sampling over `[schedule.min, schedule.max]`. The per-token iid Bernoulli masking at rate t already matches the literature — do not change the masking mechanics. Reproducibility requirement: the training loop's corruption `torch.Generator` must be re-seeded to `base_seed + epoch_index` at each epoch boundary (when a seed is configured), so each epoch's masking pattern differs but any epoch is exactly reproducible; checkpoint resume must preserve this alignment.
4. **New loss metrics in BOTH pipelines.** (a) A t-bucketed masked-CE loss breakdown with contiguous buckets: `t == 1.0` exactly (the oversampled region), `[0.7, 1.0)`, `[0.5, 0.7)`, `[0.3, 0.5)`, `[0.0, 0.3)` — every sampled t lands in exactly one bucket. (b) A perspective-split loss: loss over examples run from the p1 perspective (p2 is the reconstructed enemy) vs from the p2 perspective (p1 is the reconstructed enemy). Both breakdowns go to the step-metrics JSONL and the epoch-metrics CSV using the same plumbing pattern as the existing per-class metrics. The pre-training-only input-window-anchored metrics (future-distance buckets, fogged/observed counts) are REMOVED from pre-training; fine-tuning keeps its existing per-class debut and future-distance metrics.
5. **Interleave the fine-tuning input per timestep.** Fine-tuning (`debut_mode=true`) is now the only pipeline with an input. Change its input grammar from `[all self timesteps][all enemy timesteps]` (one delimiter per player per timestep) to per-timestep interleaving with ONE delimiter per timestep: `[t1 self tokens][t1 enemy tokens][DELIM][t2 self tokens][t2 enemy tokens][DELIM]...`
6. **Uniform token treatment across the vocabulary.** Every content token — unit, building, research/upgrade — must be treated identically in every code path. Concretely: (a) input fog in fine-tuning must apply to ALL enemy content tokens, not just `token_kind == "entity"` (upgrades are currently never fogged); (b) unify debut detection in `_build_debut_target` to the single running-max count-increase rule for all tokens (this mathematically subsumes the current first-appearance upgrade rule, since a cumulative upgrade token's per-timestep count is always 1 — verify with a test, then delete the `seen_upgrades` special case); (c) audit and remove any other `token_kind`-based branching in dataset/eval/viz that treats research tokens differently. The ONE permitted exception: at input embedding time, research tokens have no X,Y position to add to the token embedding — verify how `model/embedding.py` handles position features for upgrade records (their position feature columns are likely zeros) and ensure no spurious position signal is injected; document whatever the verified behavior is.

Also confirmed design intent to preserve: the win/loss outcome token STAYS at canvas position 0 in both modes. Its "denoise last" behavior is an inference-time sampler constraint (`sampler.outcome_last`) ONLY — during training, position 0 must be masked iid at rate t and scored like every other token. Add a regression test asserting this so it can never silently become a training-time handicap.

Finally, rename `tests/smallTrainingTest.bat` to `tests/smallTrainingTestV2.bat` (the old file must not remain — V1 tested behavior that no longer exists) with its output directory updated to `tests/output/smallTrainingTestV2`.
</objective>

<architecture>
This task requires many distinct code changes across different modules. You MUST execute this using the sub-agent team pattern:

- You (Opus) are the ORCHESTRATOR. You do NOT write implementation code directly.
- You delegate each change group to a worker agent via the Task tool.
- Each worker gets fresh context, preventing pollution between changes.
- You handle investigation, cross-worker contracts, coordination, and final validation.

AGENT TEAM:

- **Investigation Agent** (Sonnet, runs FIRST): maps every consumer of the input sequence, the class-label taxonomy, fog config, sampled t, and token_kind branching, so workers know exactly what breaks.
- **Worker 1 — Dataset layer** (Sonnet): empty pre-training input + no pre-training fog, interleaved fine-tuning input, fog-all-kinds, unified debut rule, collapsed pre-training class labels — `src/thesis_ml/data/dataset.py` + `scripts/estimate_context_window.py`.
- **Worker 2 — Corruption + config** (Sonnet): t=1.0 oversampling in `src/thesis_ml/train/corruption.py`, config schema restructure in `src/thesis_ml/config.py`, all YAMLs under `configs/`.
- **Worker 3 — Loss + metrics + loop** (Opus — this is the most cross-cutting reasoning): uniform pre-training loss weighting in `src/thesis_ml/model/loss.py`, t-bucket + perspective metrics and per-epoch generator reseed in `src/thesis_ml/train/loop.py`, batch plumbing in `src/thesis_ml/data/collate.py`, removal of input-side metrics from pre-training, plus fixes to downstream input/class consumers (eval harness, sampler, viz) per the investigation.
- **Worker 4 — Tests, harness versioning, docs** (Sonnet): rewrite/add pytest tests, rename the .bat harness, update SPEC.md and docstrings.

EXECUTION ORDER:
1. Investigation agent runs first.
2. Workers 1 and 2 run in PARALLEL (spawn both Task calls in ONE message — disjoint files).
3. Worker 3 runs AFTER 1 and 2 complete (it consumes Worker 1's taxonomy constants and Worker 2's config fields).
4. Worker 4 runs AFTER Worker 3.
5. Orchestrator performs final validation and runs the verification suite.

CRITICAL CROSS-WORKER CONTRACT: the class-taxonomy constants live in `dataset.py` (Worker 1) but are consumed by `loss.py`/`loop.py` (Worker 3) via `CLASS_ID_TO_NAME` / `DEBUT_CLASS_ID_TO_NAME` / `active_class_id_to_name`. YOU (the orchestrator) must define the exact new pre-training class map (ids, names, which ids are unused in pre-training) BEFORE spawning Worker 1, and paste that identical contract into both Worker 1's and Worker 3's prompts. Recommended shape: keep all 7 ids; in pre-training, label every content token with id 0 renamed to a single content-class name; ids 1–2 become unused in pre-training; the debut map is untouched.
</architecture>

<context>
This prompt executes from the root of the `Thesis_ML` repository. Read `CLAUDE.md` and consult `SPEC.md` (SPEC.md is the architecture source of truth) before delegating. Honor the CLAUDE.md "Do Not" rules: do not implement anything from SPEC.md section 14 and do not resolve open questions from SPEC.md section 12.

The project is uv-managed; run Python via `uv run ...` (e.g. `uv run pytest`). The .bat harness drives `.venv\Scripts\python.exe` directly — leave that invocation style intact.

Key code facts (verified against the current tree):

- `src/thesis_ml/data/dataset.py`:
  - `SC2DiffusionDataset.__getitem__` calls `_build_artifact_input(...)` (~line 224) returning `(self_block + fogged_enemy_block, self_block + clean_enemy_block, fogged_counts, observed_counts)`; delimiters are appended per player per timestep. `debut_mode` currently selects only the TARGET builder; the INPUT is built identically for both modes. Fog is applied only when `record.token_kind == "entity"` (~line 252) — upgrades bypass fog.
  - `DatasetExample` carries `input_token_ids` AND `clean_input_token_ids`; both must follow the new mode-dependent grammar (pre-training: both empty; fine-tuning: both interleaved, fogged vs zero-fog enemy).
  - Class constants `CLASS_ENEMY_OBSERVED/FOGGED/FUTURE/DELIMITER/END/PAD/WINLOSS` and `DEBUT_CLASS_ID_TO_NAME` live here. `_canvas_label` (~line 794) assigns observed/fogged/future by window boundary + fogged counts.
  - `_build_debut_target` (~line 505–555) detects entity debuts by running-max count increase and upgrade debuts by a `seen_upgrades` first-appearance set — the special case to unify.
  - `_sample_fog_rate` reads `config.fog.rate_distribution`; pre-training must stop calling into fog entirely.
- `src/thesis_ml/train/corruption.py`: `corrupt_batch` masks canvas positions iid at per-example rate t (KEEP this mechanism); `_resolve_t` (~line 74) samples `t ~ U[schedule.min, schedule.max]` when `t is None`, clamped to `MIN_T`. Explicit-t callers (eval/validation pass `fixed_t`) must bypass oversampling.
- `src/thesis_ml/train/loop.py`: `TrainingLoop.__init__` (~line 165) creates ONE `torch.Generator`, seeded once for the whole run — this is where per-epoch reseeding (`base_seed + epoch_index` at each epoch boundary in `fit()`) goes. Epoch/step metric emission: `_make_log`, `_write_metrics_line` (step JSONL), `_write_epoch_metrics` / `EpochMetrics` (epoch CSV). Future-distance buckets (`FUTURE_DISTANCE_BUCKETS` in loss.py, `_accumulate_future_distance` in loop.py) are anchored to the input-window boundary — remove from pre-training, keep for fine-tuning.
- `src/thesis_ml/model/loss.py`: `CanvasCrossEntropyLoss` applies per-class weights from `config.loss.class_loss_weights` (fields include `enemy_observed_reconstruction`, `enemy_fogged_reconstruction`, `enemy_future_prediction`, `delimiter`, `end`, `pad`, `win_loss`); `active_class_id_to_name(config)` is the single choke point both loss.py and loop.py use for the class map — route the new pre-training taxonomy through it.
- `src/thesis_ml/data/collate.py`: computes `enemy_future_timestep_counts` and `_enemy_future_prediction_distances` from `CLASS_ENEMY_FUTURE` labels — pre-training must not compute/emit these; `DiffusionBatch` must gain a per-example perspective field (`DatasetExample.perspective_player` already exists) for the perspective-split loss.
- `src/thesis_ml/config.py`: `MaskScheduleConfig` (add `t_one_fraction`), `class_loss_weights` dataclass, fog config — restructure so fog + class weights are fine-tuning-only; every YAML under `configs/` must validate under the new schema.
- `scripts/estimate_context_window.py` (~lines 65, 113–114) documents the OLD input grammar and must be updated for both new grammars (pre-training context = canvas only).
- Likely input/class consumers to investigate and fix: `src/thesis_ml/eval/harness.py`, `src/thesis_ml/eval/finetune_report.py`, `src/thesis_ml/inference/sampler.py` (incl. `outcome_last` — inference-only, keep), `src/thesis_ml/inference/decode.py`, `src/thesis_ml/viz/diagnostics.py`, `src/thesis_ml/model/embedding.py` (position features for upgrade tokens), `src/thesis_ml/eval/buildorder.py` (entity/upgrade branching), both pipelines under `src/thesis_ml/pipeline/`.
- `tests/smallTrainingTest.bat` runs `thesis_ml.pipeline.train_pipeline --config configs\local_full.yaml` (PRE-TRAINING) and tees console output to `tests\output\smallTrainingTest\console.log`.

SCOPE GUARDS:
- The pre-training TARGET canvas layout (win/loss at position 0, full enemy roll-out per timestep + delimiter, [END], PAD to budget) is unchanged — only its class labels collapse and its fog dependence disappears.
- Windowing/window manifests are unchanged: with no input, windows still choose the canvas start offset (data augmentation). Do not touch `windowing.py` semantics.
- The fine-tuning debut TARGET stays fog-aware and keeps its 7-class debut taxonomy.
- Do not change masking mechanics, the linear schedule, or the 1/t loss reweighting.
</context>

<orchestrator_process>
You are the orchestrator. Follow these steps exactly:

PHASE 1 - INVESTIGATE:
1. Spawn an investigation agent (Task tool, subagent_type "general-purpose", model "sonnet") directed to produce a structured report answering:
   a. Every code path reading `DatasetExample.input_token_ids` / `clean_input_token_ids` (collate, model forward/embedding, eval harness, finetune_report, sampler, decode, viz, pipelines, scripts) and whether each assumes non-empty input, the `[all self][all enemy]` block order, or per-player delimiter counting — including how batch collation pads variable-length (now possibly zero-length) inputs.
   b. Every consumer of the class-label taxonomy (`CLASS_*` constants, `CLASS_ID_TO_NAME`, `DEBUT_CLASS_ID_TO_NAME`, `active_class_id_to_name`, per-class metric keys, epoch-CSV columns) and of `FUTURE_DISTANCE_BUCKETS` / prediction distances, split by which behavior belongs to pre-training vs fine-tuning.
   c. Every read of fog config (`config.fog.*`) and of `config.loss.class_loss_weights.*`, and every YAML under `configs/` with which mode each config file serves (pre-training vs fine-tuning), so the schema split is exact.
   d. Every caller of `corrupt_batch` / `_resolve_t` and which pass explicit `t` (those bypass oversampling); where the TrainingLoop generator is used and whether its state is checkpointed/restored (needed for the per-epoch reseed + resume-alignment design).
   e. Every `token_kind` branch in the repo (input fog, debut detection, embedding position features, eval/buildorder, viz) — the uniformity audit for change 6, including what position features upgrade records actually carry through `model/embedding.py`.
   f. Every pytest test asserting on input layout, delimiter counts, fog behavior, class names/weights, metric keys/columns, or t sampling — by file and test name.
   g. Which SPEC.md sections describe the input grammar, class taxonomy, fog model, mask schedule, and metrics.
2. Wait for results. Decide the exact new pre-training class map (the CROSS-WORKER CONTRACT in <architecture>) and the exact new metric key/column names (t-bucket keys per the contiguous edges in the objective; perspective keys for p1/p2). Paste these contracts verbatim into every relevant worker prompt — workers cannot see the investigation report unless you paste the relevant parts.

PHASE 2 - IMPLEMENT:
Spawn Workers 1 and 2 in a SINGLE message (parallel). Spawn Worker 3 after both complete, then Worker 4 after Worker 3.

**Worker 1 — Dataset layer** (subagent_type "general-purpose", model "sonnet"). Include investigation findings (a), (e) for dataset.py, and the class-map contract. TASK:
- Make input construction mode-aware via `config.data.debut_mode`:
  - `debut_mode=false`: the input is ABSENT. `input_token_ids`, `clean_input_token_ids`, and the record lists carry zero elements (length-0 long tensors / empty lists) purely as plumbing — they must contribute ZERO sequence positions downstream: no separator/boundary tokens, no minimum-length padding of the input segment anywhere. No fog sampling occurs at all in this mode (no `_sample_fog_rate` call, empty `fogged_counts`/`observed_counts`).
  - `debut_mode=true`: interleave per timestep — `[self records][enemy records][one delimiter]`, exactly ONE delimiter per timestep, after the enemy records. Fogged variant fogs enemy content tokens of EVERY kind (drop the `token_kind == "entity"` guard); clean variant uses zero-fog enemy records. Keep `fogged_counts`/`observed_counts` bookkeeping semantics for fine-tuning (now including upgrade tokens).
- Collapse pre-training canvas labels per the class-map contract (`_canvas_label` behavior when building the pre-training target; the debut path is untouched).
- Unify debut detection to the running-max rule for all token kinds; delete the `seen_upgrades` special case (verify equivalence in a quick local check; Worker 4 adds the durable test).
- Update grammar math + description strings in `scripts/estimate_context_window.py` for both modes.
- DO NOT TOUCH: corruption.py, config.py, YAMLs, loss.py, loop.py, collate.py, tests, .bat files, SPEC.md.

**Worker 2 — Corruption + config** (subagent_type "general-purpose", model "sonnet"). Include investigation findings (c), (d). TASK:
- `config.py`: add validated `t_one_fraction: float` in [0,1] to `MaskScheduleConfig`; restructure the schema so fog settings and `class_loss_weights` are fine-tuning-only (pre-training configs must not carry them; validation should reject or not require them for `debut_mode=false` — pick the cleanest mechanism consistent with the existing dataclass validation style and state your choice).
- All YAMLs under `configs/`: set `t_one_fraction: 0.1` everywhere; strip fog + class-weight knobs from pre-training configs; keep them in fine-tuning configs.
- `corruption.py` `_resolve_t`: when `t is None`, draw a per-example Bernoulli(`t_one_fraction`) — selected examples get `t = 1.0` exactly, the rest keep the existing uniform path. ALL randomness through the provided `torch.Generator`. Explicit-t callers unchanged. Do not alter masking mechanics or `inverse_t_weights`. Update docstrings.
- DO NOT TOUCH: dataset.py, loss.py, loop.py, collate.py, tests, .bat files, SPEC.md.

**Worker 3 — Loss + metrics + loop** (subagent_type "general-purpose", model "opus" — cross-cutting reasoning). AFTER 1 & 2. Include the class-map + metric-name contracts, findings (a), (b), (d), and diffs/summaries of what Workers 1 and 2 actually changed. TASK:
- `loss.py`: route the pre-training taxonomy through `active_class_id_to_name`; in pre-training the weight buffer is fully uniform — 1.0 for every class except PAD = 0.0, and it must NOT read the (now absent) class-weight config in that mode; fine-tuning weighting unchanged. Future-distance decomposition becomes fine-tuning-only.
- `loop.py`: per-epoch generator reseed (`base_seed + epoch_index` at each epoch boundary in `fit()` when a seed is configured; document interaction with checkpoint resume so a resumed run continues the correct epoch stream). Add t-bucket loss breakdown (masked-CE aggregated per bucket using each example's sampled t; contiguous edges: t==1.0 | [0.7,1.0) | [0.5,0.7) | [0.3,0.5) | [0.0,0.3)) and perspective-split loss (p1 vs p2 perspective) to BOTH the step JSONL and epoch CSV in both pipelines, following the existing per-class plumbing pattern. Remove input-side/fog-derived metrics and future-distance columns from the pre-training path; fine-tuning keeps them.
- `collate.py`: add the per-example perspective field to `DiffusionBatch`; make `enemy_future_timestep_counts` / prediction-distance computation fine-tuning-only; collate absent inputs to zero sequence positions (`[B, 0]` — never pad the input segment to a minimum length).
- Enforce the LITERALLY-ABSENT-INPUT requirement end to end in pre-training: the backbone's sequence length must equal the canvas length exactly (the current `forward` concat of input+canvas embeddings gives this for a `[B, 0]` input since there is no separator token or segment embedding — prefer skipping `embed_input`/`build_input_features` entirely when the input is absent), the combined attention mask must have no input columns, and positional treatment must start at the first canvas token.
- Fix every downstream consumer flagged by investigation (a)/(b) that breaks under absent pre-training inputs or the collapsed taxonomy (eval harness, sampler, decode, viz, pipelines). `sampler.outcome_last` stays inference-only.
- DO NOT TOUCH: dataset.py input/target builders, corruption.py, YAML values Worker 2 set, tests, .bat files, SPEC.md.

**Worker 4 — Tests, harness, docs** (subagent_type "general-purpose", model "sonnet"). AFTER Worker 3. Include summaries of all prior changes, the contracts, and finding (f)/(g). TASK:
- Update every existing test asserting old behavior (rewrite assertions, do not delete coverage).
- Add new tests:
  (1) pre-training `DatasetExample` has zero-length `input_token_ids` and `clean_input_token_ids`, no fog applied, and all content canvas labels equal the collapsed content class id; AND the input is literally absent from the model: for a collated pre-training batch, the backbone hidden states / logits have sequence length exactly equal to the canvas length (no extra positions), and the combined attention mask has no input columns;
  (2) fine-tuning input is interleaved per timestep — self before enemy within each timestep, exactly one delimiter per timestep, total delimiters == window timestep count;
  (3) fine-tuning fog with `fog_rate=1.0` fogs ALL enemy content tokens including upgrades;
  (4) debut unification: the running-max rule reproduces the old upgrade first-appearance behavior on a replay containing upgrades;
  (5) with a fixed generator over ≥10,000 draws, the fraction of exact `t == 1.0` lies in [0.07, 0.13], and `t_one_fraction=0.0` yields none;
  (6) t-bucket edges are contiguous (every t in [0,1] maps to exactly one bucket; t=0.995 → the [0.7,1.0) bucket) and perspective metrics emit p1 and p2 keys;
  (7) per-epoch reseed: same run seed reproduces identical per-epoch masks; epoch 0 and epoch 1 masks differ;
  (8) REGRESSION — canvas position 0 (win/loss) is masked iid at rate t and contributes to the training loss like any other position (training never exempts it; `outcome_last` is sampler-only);
  (9) pre-training loss weights are fully uniform (all 1.0, PAD 0.0) and pre-training config with fog/class-weight knobs is rejected (or ignored, matching Worker 2's stated mechanism).
- `git mv tests/smallTrainingTest.bat tests/smallTrainingTestV2.bat`; change `OUTPUT_DIR` to `%~dp0output\smallTrainingTestV2`; no other .bat logic changes; grep the repo for stale references to the old name.
- Update SPEC.md sections from finding (g) and stale docstrings (dataset.py header, `SC2DiffusionDataset`, loss.py class-map comments) for: no-input pre-training, uniform published-style pre-training loss, interleaved fine-tune grammar, t=1.0 oversampling + per-epoch reseed, new metrics, uniform token treatment. Do not touch SPEC.md sections 12 or 14.
- DO NOT TOUCH: implementation logic beyond docstrings.

Every worker prompt must also state: "You are ALREADY at the Thesis_ML project root. Do NOT `cd` into it. Read CLAUDE.md for conventions (novice-readable comments, config-driven parameters). Only make the change described; no refactors or improvements beyond your assigned task. Run targeted tests for your change with `uv run pytest <files>` before reporting done."

PHASE 3 - VALIDATE:
After ALL workers complete:
1. Read every modified file and check coherence: imports resolve; the class-map contract is identical in dataset.py and loss.py/loop.py; every YAML validates under the new config schema; collate/model/eval/sampler agree on empty inputs; metric keys in step JSONL match epoch CSV columns; no stale references to `smallTrainingTest` (V1), the old grammar ("all self", "per player per timestep"), or fogged/observed in pre-training paths (grep for these).
2. Run the full verification suite in <verification>.
3. If any check fails, spawn a targeted fix agent (Sonnet) with the failure output, then re-run ALL checks.
4. Produce a summary of every change made, the chosen config-schema mechanism, the final metric key/column names, and any residual concerns.
</orchestrator_process>

<success_criteria>
Confirmed with the owner — all must hold:

1. Full pytest suite passes (`uv run pytest` from the Thesis_ML root, zero failures).
2. New unit tests (1)–(9) from Worker 4's list exist and pass — covering empty pre-training input with no fog and collapsed labels, interleaved fine-tune input, fog over all token kinds, unified debut rule, ~10% exact-t=1.0 oversampling, contiguous t-buckets + perspective metrics, per-epoch reseed reproducibility, the position-0 training regression, and uniform pre-training weights with the reformed config schema.
3. `tests/smallTrainingTestV2.bat` runs end-to-end, exits 0, writes `tests/output/smallTrainingTestV2/console.log` plus training metrics, and the emitted step JSONL / epoch CSV contain the new t-bucket and perspective keys while containing NO fogged/observed/future-distance keys (it is a pre-training run); `tests/smallTrainingTest.bat` no longer exists.
4. SPEC.md and affected docstrings describe: no-input pre-training with uniform published-style loss, the interleaved fine-tuning grammar, t=1.0 oversampling with per-epoch reseeding, the new metrics, and uniform token treatment (inspect the diffs to confirm).

Structural criteria: all workers completed, orchestrator validated cross-file coherence (especially the dataset↔loss class-map contract and config schema split), no broken imports or interfaces.

MANDATORY FINAL VERIFICATION: run every action in <verification>; for each, state what was run, the result, and PASS/FAIL. If ANY check fails, fix (via a targeted agent) and re-run ALL checks. Do NOT declare the task complete until every criterion passes.
</success_criteria>

<verification>
Run these from the Thesis_ML root AFTER the Phase-3 coherence read:

1. `uv run pytest` — expect exit 0, no failures. (Criteria 1 and 2.)
2. Confirm each of Worker 4's new tests (1)–(9) actually ran and passed (target them with `-k` by their real names) — not merely collected. (Criterion 2.)
3. Execute `tests\smallTrainingTestV2.bat` (via cmd), expect exit code 0; read `tests/output/smallTrainingTestV2/console.log` to confirm training completed; open the emitted step-metrics JSONL and epoch-metrics CSV and confirm the new t-bucket and perspective keys are present and no fogged/observed/future-distance keys appear. (Criterion 3.)
4. Confirm `tests/smallTrainingTest.bat` does not exist and grep the repo for references to it — expect none. (Criterion 3.)
5. Validate every YAML under `configs/` loads through the new config schema (`uv run python -c` snippet importing the config loader for each file) — pre-training configs carry no fog/class-weight knobs, fine-tuning configs still do, all carry `t_one_fraction`. (Criteria 2 and 4.)
6. Read the SPEC.md and docstring diffs (`git diff SPEC.md src/`) and confirm all five documented behaviors from Criterion 4. (Criterion 4.)
</verification>
