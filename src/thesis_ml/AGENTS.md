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
- Every tunable is a config field validated in `config.py`; changing a parameter must be a YAML edit only, never a code change.
- Tokens are location-agnostic entity-type tokens. Map position and unit stats are input-only additive features (owned by `model/embedding.py`); they never enter token identity or the output vocabulary.
- Never place absolute game time, frame number, `game_loop`, or timestamp-derived values into model inputs, embeddings, attention inputs, or targets. Keep time as non-model metadata only.
- Preserve the canonical serialization order (primary: entity type ID; tiebreak: config `within_type_tiebreak`) across input serialization and target construction.

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
- `model/AGENTS.md`: LLaDA/LLaMA-lineage bidirectional diffusion backbone, input-only embeddings, canvas cross-entropy loss.
- `train/AGENTS.md`: canvas corruption, the training loop and metrics, and the synthetic smoke trainer.
- `inference/AGENTS.md`: iterative confidence-based sampler, canvas grammar validation/decoding, external time recovery.
- `eval/AGENTS.md`: build-order extraction, evaluation harness, precision/recall/F1 metrics, fine-tune reporting.
- `pipeline/AGENTS.md`: config-only orchestration for data acquisition, training, fine-tuning, and storage abstraction.
