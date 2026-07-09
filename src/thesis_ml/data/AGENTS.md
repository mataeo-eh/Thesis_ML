# data Subpackage Contract

## Purpose

- Turn parsed extractor parquet into budget-bounded, timestep-aligned training examples: persisted tokenized replay artifacts, window manifests, lazy per-window datasets, and dynamically padded batches.

## Ownership

- `windowing.py` owns tokenized replay artifacts and timestep-aligned window manifests (`preprocess_replays`, `build_replay_windows`, `WindowManifestEntry`, `PreprocessingResult`).
- `dataset.py` owns lazy per-window example construction and per-serving fog (`ReplayWindow`, `CanvasBuild`, `_build_artifact_input`/`_build_artifact_target`, `resolve_replay_outcome`).
- `collate.py` owns dynamic batch padding and exact input/canvas attention and loss masks (`DiffusionBatch`, `collate_diffusion_examples`).
- `split.py` owns the reproducible train/dev/test split over whole replays (`ReplaySplit`, `split_replays`).
- `frame_cache.py` owns the RAM-bounded frame cache (`BoundedFrameCache`, `detect_total_ram_bytes`, `resolve_cache_budget_bytes`, `estimate_frame_bytes`).

## Local Contracts

- Windows are greedy contiguous runs of whole timesteps from one replay, bounded independently by `input_budget_tokens` and by `canvas_recon_fraction × canvas_budget_tokens`. No partial timestep is ever emitted.
- Successive default windows tile each replay without overlap. Each batch row is exactly one window; do not pack sequences or add document masks.
- In debut/outcome mode, input windows still tile without overlap but are bounded only by `input_budget_tokens`; each target starts at its input-window start and may overlap adjacent targets while extending to replay end or `canvas_budget_tokens`.
- Fog samples one rate per served example from `fog.rate_distribution` (uniform 0.0-0.8 by default), then independently omits each enemy entity record from the clamped input with that probability. Self records, delimiters, and non-entity enemy records remain; the clean enemy sequence still owns target construction. Persisted artifacts and manifests stay clean — never bake fog into them.
- Omitted in-window enemy records remain explicit reconstruction targets and are labeled separately from enemy records that stayed visible; input fog never inserts placeholder or `[MASK]` tokens.
- Padding is dynamic to batch maxima; padding masks must exclude batch-shape padding from attention and loss.
- Split replays before selecting any local subset so windows never leak across train/dev/test.
- Exact-count split mode assigns the configured train/dev replay counts after one seeded shuffle and preserves every remainder replay as test.
- Preprocessing is incremental and bounded to one replay per worker; persisted arrays are memory-mapped during training.
- Debut-mode targets operate on memory-mapped token ids and materialize records only for emitted debut events; replay outcome metadata is cached per worker so overlapping fine-tune windows do not repeat full object decoding or JSON reads.
- Pretraining and fine-tuning own separate manifests. Manifests carry a mode-specific semantic/config stamp and are rebuilt when windowing rules or relevant config change.
- Pipeline manifests record both `p1` and `p2` perspectives. Each replay is expanded into both perspective streams only after replay-level splitting, so perspective windows cannot cross train/dev/test boundaries.
- Consume replay data at its native one-second cadence; timing recovery uses the same configured cadence.

## Work Guidance

- Extend the existing serializer and manifest schema rather than adding a parallel windowing path.
- Keep artifact writers and the dataset reader on the same on-disk shape; update both together.

## Verification

- Windowing changes require `tests/test_windowing.py` (budget, boundary, fog, padding, cadence, and parameter-count checks).
- Dataset/collation changes require `tests/test_dataset.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
