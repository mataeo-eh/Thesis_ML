# diagnostics Contract

## Purpose

- Own reproducible audits, investigations, failure analyses, and evidence captured while diagnosing the system.

## Ownership

- Diagnostic documents own observed symptoms, reproduction conditions, evidence, root-cause analysis, and scoped recommendations.
- `011-timestep-alignment-loss-investigation.md` owns the durable finding on positional cross entropy versus delimiter-local SC2 semantics: the model-independent objective-geometry result, the observational V3 EMA checkpoint measurement, the conditioning and model-free controls, and the candidate follow-up families. Its executable lives at `scripts/timestep_alignment_probe.py` and its generated artifacts at `scripts/output/timestep_alignment_probe/`.
- `012-rare-token-learning-signal.md` owns the durable finding that the alignment overcount 011 measured as pooled-and-modest is concentrated on rare, semantically pivotal tokens: the model-independent per-token shift-exposure result, the observational three-level per-token recall measurement with its base-rate control, the per-type share of the weighted objective, and the redirection of 011's recommended content-side ablation arm. Its executable lives at `scripts/rare_token_signal_probe.py` and its generated artifacts at `scripts/output/rare_token_signal_probe/`. It supersedes 011's reading of canonical serialization as damping the overcount; it does not supersede 011's delimiter/termination finding, which stands.

## Local Contracts

- Distinguish observed behavior from hypotheses and record the exact config, checkpoint, data scope, and command needed to interpret results.
- Diagnostics do not become architecture authority; accepted fixes must update source, tests, durable docs, and configs at their owning boundaries.
- Do not commit large logs, checkpoints, caches, or generated datasets here. A curated, size-bounded table transcribed into a diagnostic document is the exception; the raw artifact stays in the owning generated tree.
- A diagnostic that measures a trained checkpoint must state whether the measurement is observational. When the checkpoint was itself shaped by the mechanism under investigation, the document must say so explicitly, must not present the measurement as causal evidence, and must name the matched ablation that would be required for causal attribution.
- A pooled per-position metric may not be used to dismiss a concern about a rare subpopulation. Canonical sort-by-type serialization RELOCATES a shift's cost onto run boundaries rather than reducing it, so a pooled amplification ratio averages over types whose true exposure differs by more than an order of magnitude. Disaggregate by token type before concluding that an overcount is modest.
- Distinguish a placement failure from a knowledge failure before recommending an objective change. Low exact-coordinate recall alone is ambiguous; it must be read against order-invariant recall inside the true span and against soft probability mass with a present-versus-absent base-rate control.
- Do not describe additive positionwise cross-entropy penalties as exponential. One semantic edit can create MANY additive penalties before re-alignment; that multiplicity is the claim, and it must be reported as a counted amplification ratio rather than as growth in the loss's functional form.

## Work Guidance

- Prefer the smallest reproduction that preserves the failure and link conclusions to concrete source or test evidence.

## Verification

- Re-run the focused reproduction after a fix and record whether the original failure mode is closed.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
