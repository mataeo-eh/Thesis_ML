# pipeline Subpackage Contract

## Purpose

- Own config-only orchestration: decoupled data acquisition, end-to-end training with checkpoint/resume, fine-tuning, and the local/remote storage abstraction, per `SPEC.md` §17 and `RUN.md`.

## Ownership

- `train_pipeline.py` owns config-only preprocessing, exact/fractional replay splitting, training, checkpoint/resume orchestration, and debut-mode held-out reporting (`run_training_pipeline`, `main` → `thesis-ml-train`, smoke path).
- `acquire_data.py` owns decoupled replay acquisition wrapping `SC2-gamestate-extractor` (`run_acquisition`, `main` → `thesis-ml-acquire`, `CredentialError`, `AcquisitionResult`).
- `finetune_pipeline.py` owns the debut/outcome fine-tuning pipeline (`run_finetune_pipeline`, `_select_eval_examples`, `main`).
- `storage.py` owns the generic path resolver over local filesystem and S3 (`StorageResolver`, `S3Uri`, `parse_s3_uri`).

## Local Contracts

- No hardcoded local paths anywhere. Every data, checkpoint, and log location resolves through `storage.*` config and may be a local dir or an `s3://` URI without code changes.
- Training is resumable: roll `last.pt` to `storage.checkpoint_uri` every `checkpoint_interval` and pull it back on startup before any local fallback, so a replacement spot instance resumes with at most one interval lost.
- Acquisition is a separate stage runnable without a GPU; it must not be coupled into the training instance. `pipeline.auto_acquire` may invoke it when processed data is absent.
- CUDA-required profiles must fail before preprocessing when CUDA is unavailable; never infer GPU/VRAM behavior from a CPU run.
- The overfit profile fails when CUDA reserved memory reaches `max_cuda_reserved_gb` and logs timing, throughput, allocated peak, and reserved memory every step.
- Secrets (Kaggle, AWS) come from environment/config only, never hardcoded or committed.
- Model scale, token budgets, paths, subset selection, epochs, and checkpoint intervals stay config-owned.
- Debut-mode full training evaluates the EMA model with the fine-tune report schema against the true test split; uncapped evaluation remains lazy instead of materializing every window in host RAM.
- Both real pipelines require `pipeline.perspectives` to contain exactly `p1,p2`, canonicalize that order, and reject stale manifests that do not record both perspectives.
- `pipeline.prepare_feature_statistics` explicitly controls artifact generation. Training computes statistics from the selected train replay artifacts only, then all production train/fine-tune/diagnostic paths strictly load `data.feature_statistics_path` and require its source split and identity to match.
- Resume, warm-start, and sampler checkpoint loads fail when feature-statistics identity is absent or mismatched. Finished exports carry the same identity in model metadata and bundles.

## Work Guidance

- Extend this orchestration rather than adding a parallel entry point; it drives the `data`, `train`, and (for fine-tuning) `eval` subpackages.
- When model construction, active profile resolution, vocabulary/statistics loading, DataLoader batch shape, planned-step calculation, warm start/resume behavior, or pretraining/fine-tuning routing changes, update every affected current-profile/provenance section in `../../../Model_Architecture/MODEL_ARCHITECTURE.md`, update the canonical `.mmd`, and regenerate its SVG/PNG using `UPDATE_PROMPT.md`.
- Keep the real DataLoader path memory-mapped and bounded to one replay per worker; drop raw metadata before worker IPC and use pinned, non-blocking CUDA transfers.
- Keep worker persistence config-owned. Both local overfit and full-corpus profiles retain workers to avoid repeated Windows process startup; worker-exit messages after manually terminating a run are expected teardown fallout.
- Keep the `metrics.jsonl` / epoch-CSV fields and CLI flags aligned with `RUN.md`.

## Verification

- Pipeline changes require `tests/test_pipeline.py` and `tests/test_pipeline_hardening.py`; fine-tuning changes require `tests/test_finetune_pipeline.py`.
- Real-pipeline changes require a bounded multi-worker checkpoint/resume smoke (`--max-steps N`) before any long run.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
