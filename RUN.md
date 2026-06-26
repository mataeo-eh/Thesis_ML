# Running Thesis_ML

This project is intentionally packaged as `git clone` + `uv sync` + one command. No Dockerfile is used.

## Local Setup

```bash
git clone <repo-url>
cd local-play-bootstrap-main/Thesis_ML
uv sync --extra dev
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

## Configuration

All input/output locations are in `config/default.yaml`:

- `storage.data_uri`: processed parquet dataset location.
- `storage.raw_uri`: raw replay staging location.
- `storage.checkpoint_uri`: checkpoint output and resume location.
- `storage.log_uri`: pipeline log output.
- `storage.local_cache_dir`: local staging directory used when a URI is remote.
- `data_source.*`: replay source and extractor wrapper settings.
- `pipeline.auto_acquire`: if true, training invokes acquisition when processed data is absent.

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

Spot instances are safe to use because checkpoints are written to `storage.checkpoint_uri`; rerunning the same command resumes from `last.pt` when present.

## Extractor Wrapper

The acquisition command wraps the separate `SC2-gamestate-extractor` repository. By default it runs the configured command from `data_source.extractor_path`, passing:

```text
--process-replay-directory <raw_uri-or-cache> --output <data_uri-or-cache> --workers <workers>
```

The extractor itself remains the source of truth for replay parsing and parquet production.
