# thesis_shared.data Contract — Framework-Agnostic Replay Data

## Purpose

- Own the replay-data stages that BOTH experiment arms must see identically: tokenized replay artifacts, timestep-aligned window manifests, input feature construction, train-split normalization statistics, and the train/dev/test replay split.
- These modules are the reason the two-arm comparison is controlled. If Arm A and Arm B ever disagree about what a window is or how a feature is standardized, the comparison is measuring the pipeline rather than the architecture.

## FRAMEWORK CONTRACT

- No `torch`, no `tensorflow`, at module level or lazily inside a function. Arrays crossing out of here are numpy; each arm converts to its own tensor type at its own boundary.

## Ownership

- `windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests, including budget enforcement and boundary handling.
- `features.py` owns input feature construction from allowlisted continuous values, continuous-validity bits, categorical cloak/buff values, and numeric allegiance.
- `feature_stats.py` owns train-split-only normalization statistics, deterministic artifact identity, and strict loading.
- `split.py` owns the replay-level train/dev/test split.

## Local Contracts

- Pretraining windows contain contiguous whole timesteps from one replay and are bounded independently by input and enemy-reconstruction token budgets.
- Debut/outcome windows tile non-overlapping input timesteps under the input token budget only; each debut canvas starts at the input-window start and runs to replay end or the canvas budget, so output horizons may overlap. Outcome mode owns a separate stamped manifest.
- Replay data is consumed at its native one-second cadence.
- Replay splitting happens BEFORE perspective expansion, so both perspectives of a replay land in the same split and no window leaks across splits.
- Statistics are computed from selected training replays only, use valid observations only, and are persisted with a content identity that must match across resumes, warm starts, diagnostics, exports, and sampling checkpoints — in both arms.
- Tokenized artifacts and manifests are versioned and bound to source-file and vocabulary identities. Source or vocabulary drift must force preprocessing rather than silently reusing stale arrays.
- Entity presence requires a finite `pos_(X,Y,Z)` tuple; lifecycle/non-position sentinels are treated as null and emit no entity token, while valid `(0,0,Z)` positions remain present. Individual nonnumeric feature sentinels are missing, never coerced into valid numeric zero.
- Entity instance IDs are variable-width digit strings used only for deterministic ordering. Slash-form current/maximum unit stats are encoded as current/max fractions before train-split standardization.
- Absolute time and frame-derived values remain metadata only and must never enter model features.

## Work Guidance

- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- A change here reaches both arms. Validate Arm A in the same change, and Arm B too once it exists.
- Per-serving fog and batch collation are NOT owned here — they are arm-side concerns, because they operate on framework tensors. See `thesis_diffusion/data/AGENTS.md`.

## Verification

- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Confirm that importing these modules pulls neither `torch` nor `tensorflow` into `sys.modules`.
