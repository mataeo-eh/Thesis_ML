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
- Fog is sampled while serving an example (entity omission on the enemy sequence only). Persisted artifacts and manifests stay clean — never bake fog into them.
- Padding is dynamic to batch maxima; padding masks must exclude batch-shape padding from attention and loss.
- Split replays before selecting any local subset so windows never leak across train/dev/test.
- Preprocessing is incremental and bounded to one replay per worker; persisted arrays are memory-mapped during training.
- Manifests carry a semantic/config stamp and are rebuilt when windowing rules or relevant config change.
- Consume replay data at its native one-second cadence; timing recovery uses the same configured cadence.

## Work Guidance

- Extend the existing serializer and manifest schema rather than adding a parallel windowing path.
- Keep artifact writers and the dataset reader on the same on-disk shape; update both together.

## Verification

- Windowing changes require `tests/test_windowing.py` (budget, boundary, fog, padding, cadence, and parameter-count checks).
- Dataset/collation changes require `tests/test_dataset.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
