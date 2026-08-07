# thesis_ml Package Contract

## Purpose

- Own the importable `thesis_ml` package: the source of truth for tokenization, config, and the diffusion data/model/train/inference/eval/pipeline stack described in `SPEC.md`.

## Ownership

- `config.py` owns the single-YAML-to-dataclass configuration loader (`load_config`, `ConfigError`, deep-merge over `config/default.yaml`). All runtime parameters are read from here; nothing is hardcoded.
- `serialize.py` owns tokenization and serialization (`serialize_snapshot`, `serialize_sequence`, `parse_entity_columns`, `TokenRecord`): raw atomic entity-level tokens in the canonical order of `SPEC.md` §4–5.
- `__init__.py` owns the public package surface.
- Subpackages own their domains: `data/`, `vocab/`, `model/`, `train/`, `inference/`, `eval/`, `pipeline/` (see Child DOX Index).

## Local Contracts

- `SPEC.md` is the architecture source of truth and wins on any conflict. Do not implement §14 (banned list) or §12 (open questions) in any form.
- Any package change that affects the function computed by the model or the exact data/configuration presented to learnable machinery must update all affected content in `../../Model_Architecture/MODEL_ARCHITECTURE.md`, update the canonical `.mmd`, and regenerate its SVG/PNG in the same change using `../../Model_Architecture/UPDATE_PROMPT.md`.
- Every tunable is a config field validated in `config.py`; changing a parameter must be a YAML edit only, never a code change.
- Tokens are location-agnostic entity-type tokens. Standardized valid continuous features, continuous-validity bits, categorical cloak/buff values, and numeric allegiance are input-only joint-conditioning features (owned by `data/features.py` and `model/embedding.py`); they never enter token identity or the output vocabulary.
- Production model construction must load the configured train-split feature-statistics artifact and preserve its identity through checkpoints and exports. Synthetic/direct unit tests may opt into the explicit identity statistics fixture.
- Never place absolute game time, frame number, `game_loop`, or timestamp-derived values into model inputs, embeddings, attention inputs, or targets. Keep time as non-model metadata only.
- Preserve the canonical serialization order (primary: entity type ID; tiebreak: config `within_type_tiebreak`) across input serialization and target construction.
- Accept every digit width emitted for entity instance IDs; three-digit zero padding is not a schema limit. Slash-form current/maximum stats are converted to ratios at the model-feature boundary.
- Require a finite position tuple for entity presence. Treat lifecycle/non-position sentinels as null entity rows and individual nonnumeric feature sentinels as invalid fields; never conflate them with valid `(0,0)` or numeric/boolean zero.

## Work Guidance

- Extend the existing serializer, config schema, model, loss, and loop instead of adding parallel implementations.
- Add every new parameter to the config dataclasses and `config/default.yaml`; wire local profiles through `configs/` overrides.
- Keep the target grammar intact end to end: bounded in-window reconstruction, then whole-timestep future continuation, then `[END] [PAD]*` or a boundary-truncated `[PAD]*` (`SPEC.md` §7).

## Verification

- Run `.venv\Scripts\python.exe -m pytest -q` for package-wide changes.
- Serialization changes require `tests/test_serialization.py` (round-trip fidelity, `SPEC.md` §16).
- Config changes require `tests/test_config.py`.

## Child DOX Index

- `data/AGENTS.md`: tokenized replay artifacts, budget-driven windows, lazy example construction, per-serving fog, dynamic collation, replay split, bounded frame cache.
- `vocab/AGENTS.md`: shared content vocabulary and reserved special tokens.
- `model/AGENTS.md`: dense Gemma 4-lineage bidirectional backbone, input-only features, expected-embedding self-conditioning, and canvas clean-state loss.
- `train/AGENTS.md`: uniform/absorbing canvas corruption, process-compatible objectives, the training loop and metrics, and the synthetic smoke trainer.
- `inference/AGENTS.md`: nonmonotonic uniform EB sampling, absorbing EB ablation, canvas grammar validation/decoding, and external time recovery.
- `eval/AGENTS.md`: build-order extraction, evaluation harness, precision/recall/F1 metrics, fine-tune reporting.
- `pipeline/AGENTS.md`: config-only orchestration for data acquisition, training, fine-tuning, and storage abstraction.
- `viz/AGENTS.md`: read-only checkpoint diagnostics, static figures, and opt-in raw canvas/logit exports.
