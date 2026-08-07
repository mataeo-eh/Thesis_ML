# data Subpackage Contract

## Purpose

- Turn parsed extractor parquet into budget-bounded, timestep-aligned training examples: persisted tokenized replay artifacts, window manifests, lazy per-window datasets, and dynamically padded batches.

## Ownership

- `windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests (`preprocess_replays`, `build_replay_windows`, `WindowManifestEntry`, `PreprocessingResult`).
- `dataset.py` owns lazy per-window example construction and per-serving fog (`ReplayWindow`, `CanvasBuild`, `_build_artifact_input`/`_build_artifact_target`, `resolve_replay_outcome`).
- `collate.py` owns dynamic batch padding and exact input/canvas attention and loss masks (`DiffusionBatch`, `collate_diffusion_examples`).
- `features.py` owns the single strict parquet-to-feature codec, the approved continuous/categorical schema, sentinel handling, validity packing, facing sine/cosine, cloak states, and buff IDs. `feature_stats.py` owns valid-only train-split statistics, deterministic JSON identity, and strict loading.
- `split.py` owns the reproducible train/dev/test split over whole replays (`ReplaySplit`, `split_replays`).
- `frame_cache.py` owns the RAM-bounded frame cache (`BoundedFrameCache`, `detect_total_ram_bytes`, `resolve_cache_budget_bytes`, `estimate_frame_bytes`).

## Local Contracts

- Windows are greedy contiguous runs of whole timesteps from one replay, bounded independently by `input_budget_tokens` and by `canvas_recon_fraction × canvas_budget_tokens`. No partial timestep is ever emitted.
- Successive default windows tile each replay without overlap. Each batch row is exactly one window; do not pack sequences or add document masks.
- In debut/outcome mode, input windows still tile without overlap but are bounded only by `input_budget_tokens`; each target starts at its input-window start and may overlap adjacent targets while extending to replay end or `canvas_budget_tokens`.
- In both modes, each input timestep serializes self records, fog-filtered enemy records, and exactly one delimiter. Fog samples one rate per served example from `fog.rate_distribution` (uniform 0.0-0.8 by default), then independently omits each enemy content record, including upgrades, from the clamped input. Self records and delimiters remain; the clean enemy sequence still owns target construction. Persisted artifacts and manifests stay clean — never bake fog into them.
- Omitted in-window enemy records remain explicit reconstruction targets and are labeled separately from enemy records that stayed visible; input fog never inserts placeholder or `[MASK]` tokens.
- Padding is dynamic to batch maxima; padding masks must exclude batch-shape padding from attention and loss.
- Split replays before selecting any local subset so windows never leak across train/dev/test.
- Exact-count split mode assigns the configured train/dev replay counts after one seeded shuffle and preserves every remainder replay as test.
- Preprocessing is incremental and bounded to one replay per worker; persisted arrays are memory-mapped during training.
- Artifact reuse requires matching artifact version, source size/mtime, vocabulary identity, and every required array. Manifests carry source-corpus and vocabulary stamps so source or token-ID drift forces a rebuild.
- Debut-mode targets operate on memory-mapped token ids and materialize records only for emitted debut events; replay outcome metadata is cached per worker so overlapping fine-tune windows do not repeat full object decoding or JSON reads.
- Pretraining and fine-tuning own separate manifests. Manifests carry a mode-specific semantic/config stamp and are rebuilt when windowing rules or relevant config change.
- Pipeline manifests record both `p1` and `p2` perspectives. Each replay is expanded into both perspective streams only after replay-level splitting, so perspective windows cannot cross train/dev/test boundaries.
- Feature statistics use float64 population moments over valid observations in selected training replay artifacts only; missing values and upgrade placeholders do not affect moments. Zero-variance features use unit scale, and malformed, non-finite, schema-incompatible, split-mismatched, or identity-mismatched artifacts fail before training or inference.
- Entity rows with non-parseable position sentinels are null for tokenization and emit no entity token. Valid `(0,0,Z)` remains present. Individual nonnumeric allowlisted values receive validity `0` and a neutral placeholder; persisted continuous validity is bit-packed and buff lists remain sparse. Buff categories use raw protocol IDs through the audited corpus maximum `302`, not PySC2's incomplete enum, and unseen larger IDs fail rather than being dropped.
- Consume replay data at its native one-second cadence; timing recovery uses the same configured cadence.

## Work Guidance

- Extend the existing serializer and manifest schema rather than adding a parallel windowing path.
- Keep artifact writers and the dataset reader on the same on-disk shape; update both together.
- When a change affects model-facing feature channels, normalization, input/target grammar, token budgets, fog semantics, padding/masks, batch shapes, or perspective/outcome targets, update every affected section in `../../../Model_Architecture/MODEL_ARCHITECTURE.md`, update the canonical `.mmd`, and regenerate its SVG/PNG using `UPDATE_PROMPT.md`.

## Verification

- Windowing changes require `tests/test_windowing.py` (budget, boundary, fog, padding, cadence, and parameter-count checks).
- Dataset/collation changes require `tests/test_dataset.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
