# Current Model Architecture Update Prompt

Use this prompt when a change may affect any part of the model-facing pipeline. It is intentionally written for a high-intelligence coding agent working from the `Thesis_ML` submodule root.

```text
<objective>
Update Model_Architecture/MODEL_ARCHITECTURE.md so every diagram, number,
shape, formula, table, behavior, caveat, and source pointer exactly describes
the current live Thesis_ML implementation after the architecture-impacting
change in this task. Modify the existing reference in place. Do not create a
dated copy or retain old behavior for history; Git owns history.
</objective>

<authority_and_scope>
1. Read the workspace-root AGENTS.md, Thesis_ML/AGENTS.md, this directory's
   AGENTS.md, and every AGENTS.md governing the changed files.
2. Read current SPEC.md. It remains the normative design authority.
3. Treat MODEL_ARCHITECTURE.md as the exact implementation/configuration
   companion. If source, config, SPEC.md, and this reference disagree, resolve
   the inconsistency in this task; do not merely describe competing states.
4. Preserve unrelated user changes in the dirty worktree.
</authority_and_scope>

<retrieval>
1. Resolve the existing Thesis_ML jcodemunch and jdocmunch indexes live.
2. Refresh existing indexes incrementally before relying on them for a
   non-trivial audit. Never silently begin a first-time broad index.
3. Use semantic retrieval to locate the changed symbols and all upstream and
   downstream model-facing consumers.
4. Verify every high-impact claim against live source, merged YAML, tests, and
   model construction. Retrieval summaries are navigation, not authority.
</retrieval>

<impact_trace>
Audit all relevant paths, not only the edited module:
- config/default.yaml, configs/local_full.yaml, and config dataclasses/merge;
- vocabulary IDs, maximum ID, valid/noise/sampling state space, embedding and
  output widths;
- raw-to-model feature codec, normalization statistics, validity and category
  widths, allegiance, feature masking, and input-only boundaries;
- dataset input grammar, target grammar, fog, token budgets, dynamic padding,
  attention masks, loss masks, and batch size;
- embeddings, feature MLP, joint mixer, self-conditioning, backbone, norms,
  attention projections, RoPE, FFN, output head, initialization, sharing, and
  checkpoint compatibility;
- corruption, diffusion-time handling, scored positions, class weighting,
  auxiliary objectives, optimizer, scheduler, accumulation, precision, EMA,
  gradient checkpointing, and memory-accounting assumptions;
- iterative and one-pass sampling, allowed token states, temperature, entropy
  rule, adaptive stopping, self-conditioning reuse, and returned shapes;
- pretraining versus debut/outcome fine-tuning differences.
</impact_trace>

<required_live_measurement>
1. Confirm .venv/Scripts/python.exe exists and use it for every Python command.
2. Load configs/local_full.yaml through thesis_ml.config.load_config.
3. Load the configured content vocabulary and feature-statistics artifact.
4. Instantiate SC2StrategyDiffusionModel from the current source.
5. Enumerate every named parameter and buffer with shape, dtype, numel, and
   requires_grad; calculate subsystem, per-block, and whole-model totals.
6. Recalculate maximum batch/input/canvas/combined hidden/logit shapes from the
   merged profile. Do not assume that a budget is an observed fixed length.
7. Distinguish FP32 parameter storage from BF16 autocast compute and keep
   activation/runtime-memory estimates separate from persistent state.
8. If an expected artifact or checkpoint is absent, say so and do not replace
   current-source facts with an older log.
</required_live_measurement>

<document_update>
Update all affected parts of MODEL_ARCHITECTURE.md together:
- scope and provenance;
- references to the rendered architecture visual;
- exact configuration table;
- token, feature, canvas, attention, hidden, and output tensor contracts;
- vocabulary state-space accounting;
- learnable module definitions and parameter inventory;
- initialization, loss, optimizer, EMA, precision, and memory sections;
- sampling and fine-tuning sections;
- explicit absences and high-consequence implementation observations;
- source-of-truth map and verification instructions.
Search the whole Model_Architecture directory for superseded symbols, values,
and descriptions. Remove stale text rather than preserving compatibility prose.
Edit diagram labels and edges only in MODEL_ARCHITECTURE_DIAGRAM.mmd, update the
renderer POSITIONS only if graph membership or layout changes, then regenerate
MODEL_ARCHITECTURE_DIAGRAM.svg and MODEL_ARCHITECTURE_DIAGRAM.png with:
  .venv\Scripts\python.exe Model_Architecture\render_diagram.py
Never hand-edit the generated images or duplicate the graph definition in the
Markdown document.
</document_update>

<dox_update>
Update the nearest owning AGENTS.md and any affected parents or sibling-domain
contracts when the change alters a durable contract. Keep the architecture-doc
update trigger present in every architecture-owning subsystem AGENTS.md.
</dox_update>

<verification>
At minimum run:
  .venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_model.py tests/test_windowing.py::test_local_model_parameter_count_is_near_ten_million
Add the focused tests required by each changed subsystem AGENTS.md. Then run
git diff --check, inspect the rendered PNG/SVG, verify all arithmetic,
and refresh the affected jcodemunch/jdocmunch indexes when practical.
</verification>

<closeout>
Report the current parameter total and maximum model-facing shapes, the exact
sections updated, tests run, index refresh status, and any claim that could not
be verified live. Never call an older checkpoint or run log current.
</closeout>
```
