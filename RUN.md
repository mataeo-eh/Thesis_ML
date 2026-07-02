# Running Thesis_ML

This project is intentionally packaged as `git clone` + `uv sync` + one command. No Dockerfile is used.

## Local Setup

```bash
git clone <repo-url>
cd local-play-bootstrap-main/Thesis_ML
uv sync --extra dev
```

The project pins `torch` to PyTorch's official CUDA 13.0 wheel index through
`tool.uv.sources`; `uv sync` therefore installs a CUDA-enabled build rather
than silently selecting the CPU wheel. Verify a local GPU environment with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Run a tiny smoke training pipeline:

```bash
uv run thesis-ml-train --config config/default.yaml --smoke
```

Acquire replay data independently:

```bash
uv run thesis-ml-acquire --config config/default.yaml
```

Run training against configured parquet data:

```bash
uv run thesis-ml-train --config config/default.yaml
```

Run the local proof-of-life profiles:

```bash
uv run thesis-ml-train --config configs/local_overfit.yaml
uv run thesis-ml-train --config configs/local_full.yaml
```

On Windows, equivalent thin launchers write console output and run artifacts to
`tests\output\overfit\` and `tests\output\smallTrainingTest\`:

```bat
tests\overfit.bat
tests\smallTrainingTest.bat
```

Both launchers forward extra CLI arguments, so `--max-steps 1` provides a
bounded launch check. The local profiles require a CUDA-enabled PyTorch build.

For a bounded full-pipeline smoke that still exercises preprocessing, loading,
checkpoint save, and resume, add `--max-steps N`. The first non-smoke launch
builds or incrementally refreshes the config-owned tokenized replay artifacts
and window manifest before training.

## Configuration

All input/output locations are in `config/default.yaml`:

- `storage.data_uri`: processed parquet dataset location.
- `storage.raw_uri`: raw replay staging location.
- `storage.checkpoint_uri`: checkpoint output and resume location.
- `storage.log_uri`: pipeline log output.
- `storage.local_cache_dir`: local staging directory used when a URI is remote.
- `data_source.*`: replay source and extractor wrapper settings.
- `pipeline.auto_acquire`: if true, training invokes acquisition when processed data is absent.
- `data.input_budget_tokens` / `data.canvas_budget_tokens`: hard per-window bounds.
- `data.canvas_recon_fraction`: maximum canvas share used by in-window enemy reconstruction; the remainder is reserved for whole future timesteps.
- `data.tokenized_replay_dir` / `data.window_manifest_path`: preprocessing outputs.
- `pipeline.replay_subset_size`: seeded training-replay subset (`0` means all).
- `train.epochs`: used when `train.max_steps` is `0`.
- `train.early_stopping_patience_epochs`: consecutive sub-threshold epochs before stopping (`0` disables).
- `train.early_stopping_min_relative_improvement`: relative improvement required to reset patience.
- `train.max_cuda_reserved_gb`: hard CUDA allocator reservation ceiling (`0` disables); the overfit profile uses 7 GiB.
- `model.gradient_checkpointing`: recompute transformer blocks during backward to bound saved-activation memory; enabled for the local overfit profile after measured RTX 3070 spillover.

`s3://bucket/prefix` is supported for data, checkpoints, and logs through the same resolver as local paths. AWS credentials must come from the normal AWS environment/instance profile chain.

Kaggle credentials must come from environment variables named by config:

```bash
export KAGGLE_USERNAME=<username>
export KAGGLE_KEY=<api-key>
```

## AWS Recipe

Data acquisition is CPU-bound and training is GPU-bound, so run them separately.

1. Launch a small EC2 CPU instance for acquisition.
2. Install `uv`, clone the repo, and run `uv sync`.
3. Set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or point config at an alternate source.
4. Set `storage.raw_uri` and `storage.data_uri` to persistent S3 prefixes.
5. Run:

```bash
uv run thesis-ml-acquire --config config/default.yaml
```

For training:

1. Launch an EC2 GPU instance with an AWS Deep Learning AMI so CUDA/drivers are already present.
2. Install `uv`, clone the repo, and run `uv sync`.
3. Point `storage.data_uri`, `storage.checkpoint_uri`, and `storage.log_uri` at S3 prefixes.
4. Run:

```bash
uv run thesis-ml-train --config config/default.yaml
```

Spot instances are safe to use: every `train.checkpoint_interval` steps the run
overwrites and uploads `last.pt` to `storage.checkpoint_uri`, and on startup it
pulls `last.pt` back from that URI before falling back to a local checkpoint. A
fresh replacement instance pointed at the same S3 prefix therefore resumes where
the preempted one left off, losing at most one checkpoint interval. Set
`train.keep_step_checkpoints: true` to also retain timestamped `step-N.pt`
snapshots (otherwise only the rolling `last.pt` is kept, so disk/S3 stays flat).

### Monitoring a long run

Per-step metrics (loss, per-class losses, learning rate, masked fraction, step
wall time, tokens/sec, CUDA peak allocated bytes, CUDA reserved bytes, and
periodic held-out validation) are appended to `metrics.jsonl` and uploaded to
`storage.log_uri` on the checkpoint cadence. CUDA attention is restricted to
Flash or memory-efficient SDPA, so an incompatible mask fails instead of
falling back to quadratic-memory math attention. Tail or parse the file to
track a multi-day run and abort early if the loss curves go wrong. A reproducible
train/dev/test split (config `pipeline.split_seed` / `test_fraction` /
`dev_fraction`, split over whole replays to avoid leakage) drives the in-training
validation; the test split is held out for final evaluation.

Local profiles also write `epoch_metrics.csv` with epoch train/dev loss,
train/dev per-class losses, tokens/sec, and cumulative wall-clock elapsed time.
The overfit profile records its seeded 25 train and three disjoint dev replay
IDs in `replay_selection.json` beside that CSV.

### Throughput knobs

The real training DataLoader uses `pipeline.num_workers` background loader
processes with `pipeline.prefetch_factor` batches prefetched each, plus pinned
memory on CUDA. Each worker memory-maps only the tokenized replay artifact needed
for its current window; the corpus is never loaded into RAM. Inputs and canvases
are padded only to their batch maxima and carry exact attention/loss masks. The
overfit profile uses four persistent workers with four batches prefetched per
worker. Workers build model features, then omit raw record/metadata object graphs
from training batches before IPC; the custom batch pins its tensors and transfers
them to CUDA non-blockingly.

### Pre-flight GPU smoke test

Before a long run, confirm the full-size model fits and trains on the target GPU
and measure peak VRAM + per-step time (no dataset required — it fabricates a
correctly-shaped random batch):

```bash
uv run python scripts/gpu_smoke_test.py --batch-size 1 --input-len 2048 --steps 5
```

Pass `--vocab-size` matching your real vocabulary for an accurate parameter
count, and raise `--batch-size` until VRAM headroom runs out to find the largest
micro-batch your GPU supports.

## Extractor Wrapper

The acquisition command wraps the separate `SC2-gamestate-extractor` repository. By default it runs the configured command from `data_source.extractor_path`, passing:

```text
--process-replay-directory <raw_uri-or-cache> --output <data_uri-or-cache> --workers <workers>
```

The extractor itself remains the source of truth for replay parsing and parquet production.
