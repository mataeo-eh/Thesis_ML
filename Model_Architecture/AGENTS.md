# Model_Architecture Contract

## Purpose

- Own the exact, current-state description of the implemented `Thesis_ML` model pipeline so a reader can understand its learnable machinery, tensor shapes, parameter counts, training objective, and inference plumbing without reconstructing the system from scratch.
- Keep one evolving architecture reference. Git history owns historical versions; this directory must not accumulate dated copies, legacy diagrams, or stale snapshots.

## Ownership

- `MODEL_ARCHITECTURE.md` owns the human-readable current architecture, embedded rendered data-flow image, exact active-profile dimensions, parameter inventory, tensor contracts, optimizer/EMA behavior, and explicit architectural absences or caveats.
- `MODEL_ARCHITECTURE_DIAGRAM.mmd` is the single canonical graph definition. `MODEL_ARCHITECTURE_DIAGRAM.svg` and `MODEL_ARCHITECTURE_DIAGRAM.png` are committed, directly viewable renderings generated from it.
- `render_diagram.py` owns deterministic local rendering from the supported Mermaid flowchart subset to SVG and PNG without a Mermaid CLI or browser.
- `UPDATE_PROMPT.md` owns the reusable GPT-5.6-sol update procedure for auditing an architecture-impacting change and synchronizing this directory with live source, configuration, tests, and DOX.
- `SPEC.md` remains the normative architecture/design authority. `MODEL_ARCHITECTURE.md` is its exact implementation-and-configuration companion. A conflict is a defect: resolve it in the same task rather than documenting both states as alternatives.

## Local Contracts

- Every statement in this directory must describe the current repository state. Do not preserve an old number or behavior for historical context; Git already preserves history.
- Any change that affects model-facing data, token identities, vocabulary width, feature channels, sequence grammar, token budgets, tensor shapes, learnable modules, parameter sharing, initialization, attention, position encoding, self-conditioning, corruption, loss, optimizer, scheduler, precision, EMA, checkpoint compatibility, sampling, or the profile identified here as current requires an architecture-documentation pass in the same change.
- Update every affected occurrence together: prose, canonical Mermaid nodes and edges, rendered SVG/PNG, equations, tables, totals, caveats, source map, and verification notes. Search this directory for old names and values before closeout.
- Derived values must be recomputed from live code and merged YAML, never copied forward on trust. At minimum, instantiate the current `local_full` model with the live vocabulary and configured feature-statistics artifact when available, enumerate parameter tensors, and check the input/canvas/output formulas.
- Separate three kinds of facts explicitly: current code/configuration, a concrete checkpoint's stamped architecture, and historical run logs. Never use an older log or checkpoint count as the current source count.
- Preserve the distinction between learnable parameters, non-trainable buffers/state, training-only machinery, and inference-only procedures.
- Preserve the distinction between semantic `[PAD]` targets and batch-shape padding, input fog and canvas diffusion, and pretraining versus debut/outcome fine-tuning.
- Do not introduce machine-specific absolute paths, generated logs, checkpoints, or bulky artifacts into this directory.
- Treat the committed SVG and PNG as required derived documentation, not disposable runtime output. They must be regenerated in the same change as `MODEL_ARCHITECTURE_DIAGRAM.mmd` and must never be hand-edited.
- Use `UPDATE_PROMPT.md` as the default task prompt for architecture-impacting maintenance. It is guidance, not a substitute for the parent DOX chain or live-source verification.

## Work Guidance

- Start with semantic retrieval over `SPEC.md`, the changed subsystem, and the symbols/source map listed in `MODEL_ARCHITECTURE.md`; then verify high-impact claims directly against source and merged configuration.
- Treat a change as architecture-impacting when it changes either the function computed by the model or the exact data/configuration presented to learnable machinery, even if no `nn.Module` file changed.
- Recompute dependent quantities transitively. For example, a vocabulary change affects embedding and head shapes, parameter totals, logits, corruption/sampler state space, memory estimates, and the Mermaid diagram.
- Edit graph labels and edges only in `MODEL_ARCHITECTURE_DIAGRAM.mmd`, then run `.venv\Scripts\python.exe Model_Architecture\render_diagram.py`. Update `POSITIONS` only when graph membership/layout changes; do not duplicate labels or edges in the renderer.
- Prefer modifying the existing sections over adding appendices. Delete superseded wording immediately.
- Keep the document useful to both researchers and implementers: name the semantic role, exact tensor shape, learned map, parameter count, and train/inference behavior of each component.

## Verification

- Confirm `Thesis_ML/.venv/Scripts/python.exe` exists before Python commands and run all Python through that shim.
- For architecture/model/config changes, run at least `tests/test_config.py`, `tests/test_model.py`, and `tests/test_windowing.py::test_local_model_parameter_count_is_near_ten_million`; add the owning subsystem tests named by its `AGENTS.md`.
- Load `configs/local_full.yaml` through `thesis_ml.config.load_config`; do not manually merge YAML overrides.
- Instantiate `SC2StrategyDiffusionModel` with `load_content_vocabulary(...)` and the configured feature-statistics artifact when present. Record `sum(p.numel() for p in model.parameters())`, subsystem totals, parameter shapes, buffer shapes, and dtypes.
- Run `.venv\Scripts\python.exe Model_Architecture\render_diagram.py`, check that its Mermaid parser fails on no line or layout mismatch, visually inspect the PNG/SVG, and confirm the Markdown image/link targets resolve.
- Check internal arithmetic and `git diff --check`.
- Refresh the `Thesis_ML` jcodemunch and jdocmunch indexes after meaningful source or documentation changes when practical.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
