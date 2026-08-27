<objective>
Build and run a read-only diagnostic that tests the hypothesis that fixed-index
token cross-entropy substantially over-penalizes small semantic count/alignment
errors in delimiter-structured SC2 canvas sequences, especially at exact
`t=1`, and may thereby encourage frequent-token predictions and weak delimiter,
`[END]`, and semantic `[PAD]` behavior.

This task measures the hypothesis; it does not change the production loss,
serializer, model, corruption process, or sampler. Its result should provide the
evidence needed to decide whether a later training-loss ablation is warranted.
</objective>

<execution_gate>
This prompt is an implementation-and-diagnostic task only when explicitly
executed. Creating or reviewing this prompt does not authorize running it or
changing the training objective.
</execution_gate>

<placement_boundary>
Do not add this diagnostic to `Model_Inference_Tests/`, its runner, its test
registry, or its output tree.

This is not primarily a test of how well a model performs at inference. It is an
investigation of training-objective geometry and the relationship between
positionwise CE and delimiter-local SC2 semantics.

Use these ownership boundaries instead:

- executable: `scripts/timestep_alignment_probe.py`
- generated, git-ignored artifacts: `scripts/output/timestep_alignment_probe/`
- durable evidence and interpretation:
  `diagnostics/011-timestep-alignment-loss-investigation.md`
- unit/regression coverage for the diagnostic implementation:
  `tests/test_timestep_alignment_probe.py`

Update `scripts/AGENTS.md` and `diagnostics/AGENTS.md` when implementing their
new owned surfaces. Do not write generated artifacts into `diagnostics/` except
for a deliberately curated, size-bounded textual/CSV summary needed to support
the durable investigation.
</placement_boundary>

<context>
Work from the `Thesis_ML` repository root. Read in full before editing:

- `./AGENTS.md`, `./CLAUDE.md`, and every applicable child `AGENTS.md`
- `./SPEC.md`, especially canvas grammar, corruption, loss, and sampling
- `./SCHEMA.md`
- `./src/thesis_ml/data/dataset.py`
- `./src/thesis_ml/data/collate.py`
- `./src/thesis_ml/model/loss.py`
- `./src/thesis_ml/train/corruption.py`
- `./src/thesis_ml/train/loop.py`
- `./src/thesis_ml/inference/decode.py`
- `./scripts/AGENTS.md`
- `./diagnostics/AGENTS.md`
- `./Model_Inference_Tests/AGENTS.md` for the boundary this task must not cross

The production canvas grammar is:

`[BOS] [WIN|LOSS] (content* [DELIMITER])+ ([END] [PAD]* | [PAD]*)`

Each delimiter-bounded group corresponds to one one-second SC2 timestep. The
decoded semantic state of a timestep is the multiset/count of its content token
types; entity-token order inside that group is deterministic serialization
bookkeeping and is not retained by `decode_canvas`.

Production uniform-mode training currently uses class-weighted clean-state
positionwise CE over every valid target position except clamped BOS. Semantic
`[PAD]` is scored; batch-shape padding is not.
</context>

<hypothesis_and_caveat>
Hypothesis:

A single missing or extra content token can shift the expected sequence indices
of the remaining content and delimiter. Positionwise CE may therefore assign
many wrong-coordinate penalties to a prediction whose delimiter-local semantic
edit distance is small. At exact `t=1`, where no truthful canvas token survives
as an alignment landmark, the coordinatewise conditional marginal may favor
very common tokens such as `probe` and may make delimiter/termination positions
especially difficult.

Important causal caveat supplied by the owner:

The trained checkpoint may already have been shaped by this objective. A low
oracle-aligned score on its current logits would not cleanly prove whether
alignment caused the learned frequent-token behavior, and a high aligned score
would not prove that an alignment-aware training objective would fail. Treat the
checkpoint measurement as observational. The diagnostic must include a
model-independent objective-geometry experiment and must state that a matched
training ablation is ultimately required for causal attribution.

Do not describe CE penalties as exponential. They are additive across positions,
although one semantic insertion/deletion/count error may create many additive
positionwise penalties before re-alignment.
</hypothesis_and_caveat>

<implementation>
Create one bounded CLI diagnostic that reuses production configuration,
vocabulary, manifest split, dataset, collation, corruption, checkpoint loading,
and model-forward paths. Do not create a parallel dataset or model pipeline.

The CLI must:

- require an explicit config and checkpoint or provide clearly documented V3
  defaults
- select recorded train/dev/test replay splits without leakage and stamp the
  selected replay/window IDs
- default to EMA weights while allowing an explicit raw-weight option
- accept `--device`, `--num-workers`, `--seed`, `--max-examples`, and configurable
  noise levels
- use `--num-workers 0` as the safe in-process fallback
- print configuration immediately, then progress, throughput, elapsed time, and
  ETA during long work
- write portable repository-relative provenance and never absolute workstation
  paths
- remain read-only with respect to checkpoint, run metrics, manifests, processed
  arrays, and replay sources

Run Python only through `.venv\Scripts\python.exe` after confirming that virtual
environment exists.
</implementation>

<experiment_a_model_independent_geometry>
First demonstrate the objective geometry without using trained model weights.

From real clean target canvases, construct deterministic high-confidence
pseudo-logits/predictions for at least these controlled cases:

1. exact target
2. one content-token substitution within one timestep
3. one content-token deletion followed by a one-position left shift until that
   timestep's delimiter, with later timesteps restored exactly
4. one content-token insertion/duplication followed by a one-position right
   shift until that timestep's delimiter, with later timesteps restored exactly
5. one displaced delimiter with otherwise correct delimiter-local content

For every perturbation report:

- number of intended semantic edits
- ordinary positional mismatches
- ordinary unweighted CE under the same pseudo-logit confidence
- CE divided into outcome, content, delimiter, `[END]`, and semantic `[PAD]`
- delimiter-local Levenshtein/edit distance
- timestep multiset precision, recall, F1, and count error
- a positional-penalty amplification ratio relative to the minimal semantic
  edits
- how far the penalty propagates and whether it stops at the intended delimiter

This arm establishes what the current objective rewards independently of what
the trained model learned. Unit tests must pin its closed-form/small-example
behavior.
</experiment_a_model_independent_geometry>

<experiment_b_real_checkpoint_logits>
Use the actual CUDA-visible GPU, real recorded replay windows, and the named V3
checkpoint as the primary model arm:

- config: `configs/smallTrainingTestV3.yaml`
- checkpoint: `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt`
- weights: EMA

Use one denoiser forward pass at controlled corruption levels rather than the
iterative sampler. At minimum include `t=0.75`, `0.90`, `0.99`, and exact
`1.00`. Couple the corruption draws across t wherever possible: reuse replacement
tokens and nested corruption masks so the sweep changes surviving truthful
anchors rather than replacing the entire experiment with unrelated random
canvases.

For the same logits and targets compute:

1. Current positional metrics
   - weighted production objective
   - unweighted positional CE
   - argmax accuracy and macro-F1 on genuinely noised positions
   - per-class CE, especially content, `[DELIMITER]`, `[END]`, semantic `[PAD]`,
     and outcome

2. Structural metrics
   - outcome-position CE and pair mass on `[WIN]+[LOSS]`
   - delimiter-position CE/accuracy
   - `[END]` and semantic `[PAD]` CE/accuracy
   - predicted versus target active length and delimiter count
   - delimiter-index drift

3. Delimiter-local semantic metrics
   - parse target timestep spans from ground-truth delimiters
   - for each target timestep, compare hard argmax content as an unordered
     multiset/count vector
   - report exact-multiset rate, multiset precision/recall/F1, per-token count
     MAE, and delimiter-local edit distance
   - report with all content and again excluding dominant `probe` and `nexus`
     tokens

4. Oracle aligned content score
   - within each ground-truth timestep content span, build the negative
     log-probability cost between output slots and target token occurrences
   - compute a minimum-cost one-to-one assignment, treating duplicate token
     occurrences distinctly
   - sum the selected negative log probabilities as an order-invariant
     timestep-content score
   - keep outcome/delimiter/end/pad structural CE separate; never allow the
     assignment to consume or move those structural targets
   - report the gap between ordinary positional content CE and the oracle
     aligned content score
   - label this score explicitly as an optimistic diagnostic, not a likelihood,
     production loss, or proof that the matching algorithm should be trained

Report results overall, by t, by perspective, by future-distance bucket, by
replay/window, and by token type. Pool sums/counts before forming ratios; do not
average per-window means when support sizes differ.
</experiment_b_real_checkpoint_logits>

<experiment_c_controls>
Add controls that make the observational result harder to misread:

- Run the same fixed noised canvases with the correct clamped input, input rows
  shuffled between examples, and enemy content removed while retaining legal
  input structure. This assesses how much the t=1 prediction depends on its
  particular input rather than only a learned marginal prior.
- Compare against a simple last-observed-state/persistence baseline and a
  train-split unigram or per-position marginal baseline where those baselines
  can be computed without leakage.
- Report the alignment gap separately for windows where hard argmax delimiter
  count is correct and where it is wrong.
- Include enough real replays/windows to avoid drawing a conclusion from the
  single three-window artifact, while keeping `--max-examples` available for a
  fast reproduction.

Do not claim GPU behavior from a CPU fallback and do not use synthetic data for
the model-quality conclusions. Synthetic/pseudo-logit cases belong only to the
model-independent geometry arm.
</experiment_c_controls>

<interpretation_contract>
The durable diagnostic report must distinguish these statements:

- A large model-independent amplification ratio proves that the serialized
  coordinate objective can overcount a small semantic edit.
- A large real-logit positional-versus-aligned gap shows that the trained model's
  current errors have an alignment component.
- Neither result alone proves that the alignment objective caused the trained
  model's frequent-token behavior.
- Input-shuffle sensitivity measures conditioning use; it does not by itself
  establish a loss defect.
- The oracle assignment score is deliberately optimistic and does not enforce a
  generatable delimiter sequence.
- Only a later matched training ablation can establish whether an alignment-aware
  objective improves the learned model.

If the evidence supports a training ablation, recommend a narrowly specified
follow-up but do not implement it here. Discuss at least these candidate families
without selecting one by preference alone:

- ordinary positional CE retained with a delimiter-local count/multiset
  auxiliary
- strict structural CE plus assignment/optimal-transport content loss on
  genuinely noised positions
- CTC or another differentiable alignment marginalization
- a representation change that places semantically stable quantities at stable
  coordinates

For each, state compatibility with the fixed-coordinate uniform-diffusion
objective, computational cost, treatment of preserved clean anchors, delimiter
generation, duplicate tokens, and checkpoint/architecture consequences.
</interpretation_contract>

<verification>
- Unit-test segmentation, terminal `[END]` versus boundary-truncated `[PAD]`,
  duplicate-token assignment, empty/debut timesteps where applicable, pooled
  aggregation, and every controlled perturbation.
- Test that batch-shape padding is excluded while semantic `[PAD]` remains.
- Test deterministic results for fixed seeds and `--num-workers 0`.
- Test that the diagnostic never writes into `Model_Inference_Tests/` or the
  probed training run.
- Run CLI `--help`, focused pytest, and `git diff --check`.
- Run the bounded real-GPU diagnostic on the named V3 EMA checkpoint and real
  replay data. Record the exact command and output location.

This prompt changes no model-facing contract. Therefore it does not update the
architecture diagram unless implementation unexpectedly crosses that boundary;
if it does, stop and obtain owner direction rather than silently expanding scope.
</verification>

<deliverables>
- `scripts/timestep_alignment_probe.py`
- focused unit tests under `tests/`
- size-bounded generated artifacts under
  `scripts/output/timestep_alignment_probe/`
- `diagnostics/011-timestep-alignment-loss-investigation.md`
- updated `scripts/AGENTS.md` and `diagnostics/AGENTS.md`
- an evidence-backed conclusion that explicitly respects the causal limitation
  that the current checkpoint was itself trained under positional CE
- no files added to or modified under `Model_Inference_Tests/`
</deliverables>
