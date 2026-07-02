"""Master cloud-runnable training pipeline entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import (
    MANIFEST_VERSION,
    load_window_manifest,
    manifest_config_stamp,
    preprocess_replays,
    read_manifest_metadata,
)
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.storage import StorageResolver
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.train.train import SMOKE_VOCAB_SIZE, make_synthetic_examples
from thesis_ml.vocab.content_vocab import load_content_vocabulary


@dataclass(frozen=True)
class TrainingPipelineResult:
    checkpoint_uri: str
    resumed: bool
    steps: int
    smoke: bool
    parameter_count: int
    peak_vram_bytes: int


def run_training_pipeline(
    config_path: str | Path,
    *,
    smoke: bool | None = None,
    storage: StorageResolver | None = None,
    max_steps_override: int | None = None,
) -> TrainingPipelineResult:
    config = load_config(config_path)
    resolver = storage or StorageResolver()
    use_smoke = config.pipeline.smoke if smoke is None else smoke
    if not use_smoke and not _dataset_available(config, resolver) and config.pipeline.auto_acquire:
        from thesis_ml.pipeline.acquire_data import run_acquisition

        run_acquisition(config_path, storage=resolver)
    elif not use_smoke and not _dataset_available(config, resolver):
        raise FileNotFoundError(f"dataset not found at configured URI: {config.storage.data_uri}")

    if use_smoke:
        result = _run_smoke_pipeline(config, resolver)
    else:
        result = _run_real_pipeline(config, resolver, max_steps_override=max_steps_override)
    _write_log(config, resolver, f"resumed={result.resumed} steps={result.steps} smoke={result.smoke}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SC2 diffusion training from config")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--smoke", action="store_true", help="force the tiny synthetic smoke pipeline")
    parser.add_argument("--max-steps", type=int, default=None, help="bounded verification override")
    args = parser.parse_args()
    try:
        result = run_training_pipeline(
            args.config,
            smoke=True if args.smoke else None,
            max_steps_override=args.max_steps,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(
        f"checkpoint_uri={result.checkpoint_uri} resumed={result.resumed} steps={result.steps} "
        f"parameters={result.parameter_count} peak_vram_bytes={result.peak_vram_bytes}"
    )


def _run_smoke_pipeline(config: ProjectConfig, resolver: StorageResolver) -> TrainingPipelineResult:
    torch.manual_seed(config.pipeline.seed)
    checkpoint_dir = _local_checkpoint_dir(config, resolver)
    smoke_config = replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64),
        train=replace(
            config.train,
            lr=0.01,
            warmup=1,
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=config.pipeline.smoke_steps,
            val_interval=0,
            checkpoint_interval=max(1, config.pipeline.smoke_steps + 1),
            checkpoint_dir=str(checkpoint_dir),
            ema_decay=0.9,
            precision="fp32",
        ),
    )
    examples = make_synthetic_examples(smoke_config, count=4)
    dataloader = DataLoader(
        examples,
        batch_size=smoke_config.pipeline.batch_size,
        shuffle=False,
        collate_fn=collate_diffusion_examples,
    )
    model = SC2StrategyDiffusionModel(smoke_config, vocab_size=SMOKE_VOCAB_SIZE)
    loop = TrainingLoop(model=model, config=smoke_config, seed=smoke_config.pipeline.seed)
    resumed = _try_resume(loop, checkpoint_dir)
    loop.fit(dataloader, max_steps=smoke_config.pipeline.smoke_steps, fixed_t=1.0)
    checkpoint_uri = _publish_checkpoint(config, resolver, checkpoint_dir / "last.pt")
    return TrainingPipelineResult(
        checkpoint_uri=checkpoint_uri,
        resumed=resumed,
        steps=loop.global_step,
        smoke=True,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        peak_vram_bytes=0,
    )


def _run_real_pipeline(
    config: ProjectConfig,
    resolver: StorageResolver,
    *,
    max_steps_override: int | None = None,
) -> TrainingPipelineResult:
    torch.manual_seed(config.pipeline.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.train.require_cuda and device != "cuda":
        raise RuntimeError(
            "this training profile requires CUDA, but torch.cuda.is_available() is false"
        )
    checkpoint_dir = _local_checkpoint_dir(config, resolver)
    token_dictionary = _materialize_file(config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver)
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)

    # Reproducible split over replays (not windows) to avoid leakage. The test
    # set is held out here for later evaluation; dev drives in-training
    # validation / early-abort decisions.
    split = split_replays(
        replay_paths,
        seed=config.pipeline.split_seed,
        test_fraction=config.pipeline.test_fraction,
        dev_fraction=config.pipeline.dev_fraction,
    )
    train_replays, dev_replays = _select_replays(
        list(split.train),
        list(split.dev),
        config,
    )
    _record_replay_selection(config, resolver, train_replays, dev_replays)
    train_windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=train_replays,
    )
    dev_windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=dev_replays,
    )
    train_dataset = SC2DiffusionDataset(
        train_windows,
        config,
        vocabulary,
        seed=config.pipeline.seed,
        fog_rate_override=None,
    )
    train_loader = _make_dataloader(train_dataset, config, shuffle=True, device=device)
    val_loader = None
    if dev_windows:
        dev_dataset = SC2DiffusionDataset(
            dev_windows,
            config,
            vocabulary,
            seed=config.pipeline.seed,
            fog_rate_override=None,
        )
        val_loader = _make_dataloader(dev_dataset, config, shuffle=False, device=device)

    planned_steps = config.train.max_steps
    if planned_steps <= 0:
        planned_steps = len(train_loader) * config.train.epochs
    training_config = replace(
        config,
        train=replace(config.train, checkpoint_dir=str(checkpoint_dir), max_steps=planned_steps),
    )
    model = SC2StrategyDiffusionModel(training_config, vocab_size=vocabulary.vocab_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics_dir = _local_metrics_dir(config, resolver)
    metrics_path = metrics_dir / "step_metrics.jsonl"
    epoch_metrics_path = metrics_dir / "epoch_metrics.csv"
    loop = TrainingLoop(
        model=model,
        config=training_config,
        device=device,
        seed=config.pipeline.seed,
        metrics_path=metrics_path,
        epoch_metrics_path=epoch_metrics_path,
        checkpoint_publisher=_checkpoint_publisher(config, resolver),
        metrics_publisher=_metrics_publisher(config, resolver),
    )
    # Resume from durable storage first (fresh spot instance has no local
    # checkpoint), then fall back to a local checkpoint.
    resumed = _resume_from_remote(loop, config, resolver, checkpoint_dir) or _try_resume(loop, checkpoint_dir)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    requested_steps = training_config.train.max_steps if max_steps_override is None else max_steps_override
    loop.fit(
        train_loader,
        val_dataloader=val_loader,
        max_steps=requested_steps,
        epochs=training_config.train.epochs,
    )
    peak_vram_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    checkpoint_uri = _publish_checkpoint(config, resolver, checkpoint_dir / "last.pt")
    return TrainingPipelineResult(
        checkpoint_uri=checkpoint_uri,
        resumed=resumed,
        steps=loop.global_step,
        smoke=False,
        parameter_count=parameter_count,
        peak_vram_bytes=peak_vram_bytes,
    )


def _select_replays(
    train_candidates: list[str],
    dev_candidates: list[str],
    config: ProjectConfig,
) -> tuple[list[str], list[str]]:
    train_replays = list(train_candidates)
    dev_replays = list(dev_candidates)
    if config.pipeline.replay_subset_size > 0:
        if len(train_replays) < config.pipeline.replay_subset_size:
            raise ValueError(
                f"requested {config.pipeline.replay_subset_size} training replays, "
                f"but only {len(train_replays)} are available"
            )
        generator = torch.Generator().manual_seed(config.pipeline.seed)
        order = torch.randperm(len(train_replays), generator=generator).tolist()
        train_replays = [train_replays[index] for index in order[: config.pipeline.replay_subset_size]]
    if config.pipeline.validation_replay_count > 0:
        if len(dev_replays) < config.pipeline.validation_replay_count:
            raise ValueError(
                f"requested {config.pipeline.validation_replay_count} dev replays, "
                f"but only {len(dev_replays)} are available"
            )
        generator = torch.Generator().manual_seed(config.pipeline.seed + 1)
        order = torch.randperm(len(dev_replays), generator=generator).tolist()
        dev_replays = [dev_replays[index] for index in order[: config.pipeline.validation_replay_count]]
    if set(train_replays) & set(dev_replays):
        raise ValueError("training and dev replay selections must be disjoint")
    return train_replays, dev_replays


def _ensure_window_manifest(
    replay_paths: list[str],
    config: ProjectConfig,
    vocabulary,
) -> None:
    manifest_path = Path(config.data.window_manifest_path)
    rebuild = config.pipeline.rebuild_manifest or not manifest_path.exists()
    if not rebuild:
        metadata = read_manifest_metadata(manifest_path)
        rebuild = any(
            (
                metadata.get("manifest_version") != MANIFEST_VERSION,
                metadata.get("config_stamp") != manifest_config_stamp(config),
                metadata.get("replay_count") != len(replay_paths),
            )
        )
    if rebuild:
        if not config.pipeline.preprocess_if_missing:
            raise FileNotFoundError(f"window manifest is absent or stale: {manifest_path}")
        preprocess_replays(
            replay_paths,
            config,
            vocabulary,
            perspectives=_perspectives(config.pipeline.perspectives),
            force=config.pipeline.rebuild_manifest,
        )


def _record_replay_selection(
    config: ProjectConfig,
    resolver: StorageResolver,
    train: list[str],
    validation: list[str],
) -> None:
    output = _local_metrics_dir(config, resolver) / "replay_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "seed": config.pipeline.seed,
                "train_replay_ids": [Path(path).stem for path in train],
                "dev_replay_ids": [Path(path).stem for path in validation],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _local_metrics_dir(config: ProjectConfig, resolver: StorageResolver) -> Path:
    if resolver.is_s3(config.storage.log_uri):
        path = Path(config.storage.local_cache_dir) / "logs"
    else:
        path = Path(config.storage.log_uri)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_dataloader(
    dataset: SC2DiffusionDataset,
    config: ProjectConfig,
    *,
    shuffle: bool,
    device: str,
) -> DataLoader:
    """Build a throughput-tuned DataLoader for the real training pipeline.

    Uses multiple worker processes (so CPU-bound serialization runs off the
    GPU thread), per-worker prefetching, persistent workers (avoid re-spawn
    each epoch), and pinned memory on CUDA (faster host->device copies).
    Worker/prefetch knobs come from config. Falls back to a simple in-process
    loader when num_workers is 0.
    """

    num_workers = max(0, config.pipeline.num_workers)
    kwargs: dict = {
        "batch_size": config.pipeline.batch_size,
        "shuffle": shuffle,
        # InputFeatures are built in the worker before raw TokenRecord metadata
        # is dropped. Training does not consume those Python object graphs, so
        # excluding them avoids expensive worker-to-main-process serialization.
        "collate_fn": partial(collate_diffusion_examples, retain_metadata=False),
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = config.pipeline.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _checkpoint_publisher(config: ProjectConfig, resolver: StorageResolver) -> Callable[[Path], None] | None:
    """Return a callback that mirrors a local checkpoint to remote storage.

    None when checkpoints are local-only (nothing to publish). Otherwise the
    callback uploads each saved file under its basename to the checkpoint URI,
    so `last.pt` lands at `<checkpoint_uri>/last.pt` for the resume path.
    """

    checkpoint_uri = config.storage.checkpoint_uri
    if not resolver.is_s3(checkpoint_uri):
        return None

    def publish(local_path: Path) -> None:
        resolver.put_file(local_path, f"{checkpoint_uri.rstrip('/')}/{local_path.name}")

    return publish


def _metrics_publisher(config: ProjectConfig, resolver: StorageResolver) -> Callable[[Path], None] | None:
    """Return a callback that mirrors the local metrics JSONL to remote logs."""

    log_uri = config.storage.log_uri
    if not resolver.is_s3(log_uri):
        return None

    def publish(local_path: Path) -> None:
        resolver.put_file(local_path, f"{log_uri.rstrip('/')}/{local_path.name}")

    return publish


def _resume_from_remote(
    loop: TrainingLoop,
    config: ProjectConfig,
    resolver: StorageResolver,
    checkpoint_dir: Path,
) -> bool:
    """Pull `last.pt` from remote storage and resume, if one exists.

    Enables a fresh (replacement) spot instance to continue a preempted run:
    point storage.checkpoint_uri at the same S3 prefix and training picks up
    where it left off. Returns True if a remote checkpoint was loaded.
    """

    checkpoint_uri = config.storage.checkpoint_uri
    if not resolver.is_s3(checkpoint_uri):
        return False
    remote = f"{checkpoint_uri.rstrip('/')}/last.pt"
    if not resolver.exists(remote):
        return False
    local_checkpoint = checkpoint_dir / "last.pt"
    resolver.get_file(remote, local_checkpoint)
    loop.load_checkpoint(local_checkpoint)
    return True


def _dataset_available(config: ProjectConfig, resolver: StorageResolver) -> bool:
    if config.pipeline.smoke:
        return True
    return bool(resolver.list_files(config.storage.data_uri, config.pipeline.replay_glob))


def _try_resume(loop: TrainingLoop, checkpoint_dir: Path) -> bool:
    checkpoint = checkpoint_dir / "last.pt"
    if not checkpoint.exists():
        return False
    loop.load_checkpoint(checkpoint)
    return True


def _local_checkpoint_dir(config: ProjectConfig, resolver: StorageResolver) -> Path:
    if resolver.is_s3(config.storage.checkpoint_uri):
        path = Path(config.storage.local_cache_dir) / "checkpoints"
    else:
        path = Path(config.storage.checkpoint_uri)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _publish_checkpoint(config: ProjectConfig, resolver: StorageResolver, local_checkpoint: Path) -> str:
    if not resolver.is_s3(config.storage.checkpoint_uri):
        return str(local_checkpoint)
    target = f"{config.storage.checkpoint_uri.rstrip('/')}/last.pt"
    resolver.put_file(local_checkpoint, target)
    return target


def _write_log(config: ProjectConfig, resolver: StorageResolver, text: str) -> None:
    if resolver.is_s3(config.storage.log_uri):
        log_path = Path(config.storage.local_cache_dir) / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
        resolver.put_file(log_path, f"{config.storage.log_uri.rstrip('/')}/pipeline.log")
        return
    path = Path(config.storage.log_uri) / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _materialize_file(uri: str, cache_dir: str, resolver: StorageResolver) -> Path:
    if not resolver.is_s3(uri):
        return Path(uri)
    target = Path(cache_dir) / "inputs" / Path(uri).name
    return resolver.get_file(uri, target)


def _materialize_replay_paths(config: ProjectConfig, resolver: StorageResolver) -> list[str]:
    uris = resolver.list_files(config.storage.data_uri, config.pipeline.replay_glob)
    if not resolver.is_s3(config.storage.data_uri):
        return uris
    local_paths: list[str] = []
    target_dir = Path(config.storage.local_cache_dir) / "data"
    for uri in uris:
        local_paths.append(str(resolver.get_file(uri, target_dir / Path(uri).name)))
    return local_paths


def _perspectives(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("pipeline.perspectives must contain at least one value")
    return values


if __name__ == "__main__":
    main()
