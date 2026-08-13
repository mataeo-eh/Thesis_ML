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
- `model.segment_embeddings` and `model.per_segment_positions` are prompt-009 ABLATION TOGGLES, not staged features. **Both must stay `false` here.** An agent must never flip one on in this file, nor treat their presence as an invitation to enable them. Promotion of a toggle to a default is an owner decision on measured evidence (`SPEC.md` §14b).
- `model.frozen_input_kv` is **`true` here — that promotion already happened** (owner decision 2026-08-09, on ablation arm 01's evidence: no meaningful loss difference, much faster inference). Do not "restore" it to `false` as a supposed baseline; the baseline moved. `toggle_fingerprint` is intentionally left un-rebased, so the default-derived `architecture_identity` is `dense-multinomial-SC2-v2+frozen_input_kv` and pre-promotion checkpoints fail closed rather than loading silently. `configs/ablation_00_baseline.yaml` is the one profile that opts back out, and it exists to keep the finished sweep's baseline arm loadable.
- Note `config.py` rejects unknown YAML keys AND ignores dataclass defaults, so each of the three toggles needs its explicit entry here or config loading fails.
- `diffusion.process` defaults to `uniform`; `absorbing` is the only supported ablation. The default time distribution is `Beta(2,1)` power sampling plus 5% exact `t=1`, and fog mirrors that power law over 0–0.8. Confidence sharpening remains `0.0`; sampler maximum steps is `64` with the published temperature, entropy-bound, and adaptive-stop defaults.
- The default optimizer uses five-batch accumulation and a WSD schedule: 10% warmup, 70% stable at `3e-4`, then 20% linear decay to `3e-6`. Checkpoint family paths and retention are config-owned.

## Work Guidance

- Change behavior by editing config, not code; keep `default.yaml` complete so profiles only override deltas.

## Verification

- Config loading and validation are covered by `tests/test_config.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
