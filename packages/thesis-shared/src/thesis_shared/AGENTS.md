# thesis_shared Package Contract — Framework-Agnostic Core

## Purpose

- Own everything that MUST be identical between the two experiment arms, so that the comparison between them is controlled rather than incidental.
- Both `thesis_diffusion` (Arm A, PyTorch) and `thesis_ar` (Arm B, TensorFlow) import this package. That shared dependency is the structural guarantee that both arms consume byte-identical data and are scored by identical metrics.

## FRAMEWORK CONTRACT — no deep-learning framework, ever

- **Nothing in this package may import `torch` or `tensorflow`**, at module level or lazily inside a function. Permitted dependencies are the scientific-Python stack: numpy, pandas, pyarrow, PyYAML, matplotlib, boto3, psutil.
- **`packages/thesis-shared/pyproject.toml` must never declare `torch` or `tensorflow`.** This is enforced by `tests/test_pipeline.py::test_workspace_packages_declare_the_framework_boundary`.
- The reason is operational, not stylistic: this package is installed into a TensorFlow-only environment that has no PyTorch present at all. A single torch import here would break Arm B outright.
- Tensors crossing into this package are plain numpy arrays. Framework tensor types are converted at the arm's boundary, never here.

## Ownership

- `config.py` owns the single-YAML-to-dataclass configuration loader (`load_config`, `ConfigError`, deep-merge over `config/default.yaml`). All runtime parameters are read from here; nothing is hardcoded.
- `serialize.py` owns tokenization and serialization (`serialize_snapshot`, `serialize_sequence`, `parse_entity_columns`, `TokenRecord`): raw atomic entity-level tokens in the canonical order.
- `vocab/` owns the shared content vocabulary and the reserved special tokens.
- `data/windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests.
- `data/features.py` owns input feature construction from allowlisted continuous, validity-bit, categorical, and allegiance values.
- `data/feature_stats.py` owns train-split-only normalization statistics, deterministic artifact identity, and strict loading.
- `data/split.py` owns the train/dev/test replay split.
- `eval/buildorder.py` owns build-order event extraction; `eval/metrics.py` owns precision/recall/F1 aggregation. **These are the metrics both arms are scored on.**
- `inference/decode.py` owns canvas grammar validation and decoding; `inference/timing.py` owns external time recovery.
- `pipeline/acquire_data.py` owns data acquisition; `pipeline/storage.py` owns the storage abstraction.

## Local Contracts

- **A change here lands on both arms simultaneously.** Before changing anything in this package, confirm the change is correct for Arm A *and* Arm B. Re-validate the diffusion arm in the same change; if AR code exists by then, re-validate it too.
- **Never fork this code into an arm to make one arm's life easier.** If an arm needs different behavior, that is a signal to discuss the experiment design with the owner, not to duplicate the module. A silent fork destroys the comparison, because the arms stop seeing the same data.
- Every tunable is a config field validated in `config.py`; changing a parameter must be a YAML edit only, never a code change.
- Tokens are location-agnostic entity-type tokens. Standardized valid continuous features, continuous-validity bits, categorical cloak/buff values, and numeric allegiance are input-only joint-conditioning features; they never enter token identity or the output vocabulary.
- Never place absolute game time, frame number, `game_loop`, or timestamp-derived values into model inputs, embeddings, attention inputs, or targets. Keep time as non-model metadata only.
- Preserve the canonical serialization order (primary: entity type ID; tiebreak: config `within_type_tiebreak`) across input serialization and target construction.
- Accept every digit width emitted for entity instance IDs; three-digit zero padding is not a schema limit. Slash-form current/maximum stats are converted to ratios at the model-feature boundary.
- Require a finite position tuple for entity presence. Treat lifecycle/non-position sentinels as null entity rows and individual nonnumeric feature sentinels as invalid fields; never conflate them with valid `(0,0)` or numeric/boolean zero.
- Local replay data is consumed at its native one-second cadence. Timing recovery must use the same configured cadence.
- Replay splitting happens BEFORE perspective expansion, so both perspectives of a replay remain in the same split and no window leaks across splits.
- Tokenized artifacts and manifests are versioned and bound to source-file and vocabulary identities. Source or vocabulary drift must force preprocessing instead of silently reusing stale arrays.
- Feature statistics are computed from selected training replays only, persisted with a content identity, and that identity must match across resumes, warm starts, diagnostics, exports, and sampling checkpoints — in both arms.

## Known Gaps

- `frame_cache.py` currently lives in `thesis_diffusion/data/` rather than here, because its RAM-budget sharding is written against PyTorch's DataLoader worker model. It is a candidate for promotion into this package once Arm B needs frame caching, which would require abstracting the worker-count lookup out of the module. Do not copy it into `thesis_ar`.

## Work Guidance

- Extend the existing serializer, config schema, and windowing rather than adding parallel implementations.
- Add every new parameter to the config dataclasses and `config/default.yaml`; wire run profiles through `configs/` overrides.
- Keep preprocessing incremental and bounded to one replay per worker; memory-map persisted arrays during training.
- When adding an evaluation metric, add it here so both arms report it, not in an arm.

## Verification

- Run the shared suite from the diffusion environment (`.venv`), which installs this package: `.venv\Scripts\python.exe -m pytest tests/ -q`.
- Serialization changes require `tests/test_serialization.py` (round-trip fidelity).
- Config changes require `tests/test_config.py`.
- Windowing changes require `tests/test_windowing.py`, including budget, boundary, fog, padding, cadence, and parameter-count checks.
- Any change here must additionally be checked for framework neutrality: importing `thesis_shared` must not pull `torch` or `tensorflow` into `sys.modules`.

## Child DOX Index

- `vocab/AGENTS.md`: shared content vocabulary and reserved special tokens.
- `data/`, `eval/`, `inference/`, `pipeline/`: governed by this file; no separate child contracts.
