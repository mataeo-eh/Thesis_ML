# config Contract

## Purpose

- Own `default.yaml`, the canonical base configuration that every entry point and run profile merges over.

## Ownership

- `default.yaml` owns the full default set of `data.*`, `fog.*`, `model.*`, `train.*`, `pipeline.*`, `storage.*`, `data_source.*`, and evaluation parameters validated by `src/thesis_ml/config.py`.

## Local Contracts

- This is the single source of default values. Run profiles in `configs/` are overrides layered on top of it, not replacements.
- Every parameter here is validated into a dataclass by `config.py`; adding a field requires updating both together.
- Values follow `SPEC.md` §11 (provisional defaults) — treat none as load-bearing, and never hardcode a value that belongs here into code.
- No secrets or machine-specific absolute paths: storage locations are URIs (local or `s3://`) and credentials come from the environment.
- Absolute time, frame number, `game_loop`, and timestamp-derived values must never be introduced as model-feature config.

## Work Guidance

- Change behavior by editing config, not code; keep `default.yaml` complete so profiles only override deltas.

## Verification

- Config loading and validation are covered by `tests/test_config.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
