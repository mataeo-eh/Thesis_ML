# config Contract

## Purpose

- Own `default.yaml`, the canonical base configuration that every entry point and run profile merges over.

## Ownership

- `default.yaml` owns the full default set of `data.*`, `fog.*`, `diffusion.*`, `model.*`, `train.*`, `sampler.*`, `pipeline.*`, `storage.*`, `data_source.*`, and evaluation parameters validated by `src/thesis_ml/config.py`, including uniform diffusion, adaptive EB stopping, feature-statistics identity, and explicit preparation switches.

## Local Contracts

- This is the single source of default values. Run profiles in `configs/` are overrides layered on top of it, not replacements.
- Any default that affects model scale, feature/input/canvas widths, sequence budgets, diffusion, loss, optimizer, precision, EMA, checkpoint compatibility, or sampling must update every affected section in `../Model_Architecture/MODEL_ARCHITECTURE.md`, update its canonical `.mmd`, and regenerate its SVG/PNG in the same change using `../Model_Architecture/UPDATE_PROMPT.md`.
- Every parameter here is validated into a dataclass by `config.py`; adding a field requires updating both together.
- Values follow `SPEC.md` §11 (provisional defaults) — treat none as load-bearing, and never hardcode a value that belongs here into code.
- No secrets or machine-specific absolute paths: storage locations are URIs (local or `s3://`) and credentials come from the environment.
- Absolute time, frame number, `game_loop`, and timestamp-derived values must never be introduced as model-feature config.
- `fog` is required in both modes. `data.feature_statistics_path` names the deterministic training-statistics artifact, while `pipeline.prepare_feature_statistics` must be enabled explicitly when that artifact should be computed or replaced.
- `model.frozen_input_kv`, `model.segment_embeddings`, and `model.per_segment_positions` are prompt-009 ABLATION TOGGLES, not staged features. **All three must stay `false` here.** `config/default.yaml` defines the baseline, and the baseline is all-off; an agent must never flip one on in this file, nor treat their presence as an invitation to enable them. Promotion of any toggle to a default is an owner decision on measured evidence (`SPEC.md` §14b). Note `config.py` rejects unknown YAML keys AND ignores dataclass defaults, so each of the three needs its explicit `false` entry here or config loading fails.
- `diffusion.process` defaults to `uniform`; `absorbing` is the only supported ablation. The process determines compatible corruption, loss, prior, and sampler behavior. Default terminal oversampling and confidence sharpening are `0.0`; sampler maximum steps is `64` with the published DiffusionGemma temperature, entropy-bound, and adaptive-stop defaults.

## Work Guidance

- Change behavior by editing config, not code; keep `default.yaml` complete so profiles only override deltas.

## Verification

- Config loading and validation are covered by `tests/test_config.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
