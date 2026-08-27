# diagnostics Contract

## Purpose

- Own reproducible audits, investigations, failure analyses, and evidence captured while diagnosing the system.

## Ownership

- Diagnostic documents own observed symptoms, reproduction conditions, evidence, root-cause analysis, and scoped recommendations.
- `011-timestep-alignment-loss-investigation.md` owns the durable finding on positional cross entropy versus delimiter-local SC2 semantics: the model-independent objective-geometry result, the observational V3 EMA checkpoint measurement, the conditioning and model-free controls, and the candidate follow-up families. Its executable lives at `scripts/timestep_alignment_probe.py` and its generated artifacts at `scripts/output/timestep_alignment_probe/`.

## Local Contracts

- Distinguish observed behavior from hypotheses and record the exact config, checkpoint, data scope, and command needed to interpret results.
- Diagnostics do not become architecture authority; accepted fixes must update source, tests, durable docs, and configs at their owning boundaries.
- Do not commit large logs, checkpoints, caches, or generated datasets here. A curated, size-bounded table transcribed into a diagnostic document is the exception; the raw artifact stays in the owning generated tree.
- A diagnostic that measures a trained checkpoint must state whether the measurement is observational. When the checkpoint was itself shaped by the mechanism under investigation, the document must say so explicitly, must not present the measurement as causal evidence, and must name the matched ablation that would be required for causal attribution.
- Do not describe additive positionwise cross-entropy penalties as exponential. One semantic edit can create MANY additive penalties before re-alignment; that multiplicity is the claim, and it must be reported as a counted amplification ratio rather than as growth in the loss's functional form.

## Work Guidance

- Prefer the smallest reproduction that preserves the failure and link conclusions to concrete source or test evidence.

## Verification

- Re-run the focused reproduction after a fix and record whether the original failure mode is closed.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
