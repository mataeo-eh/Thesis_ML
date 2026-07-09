"""Master cloud-runnable training pipeline entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
import gc
import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.resumable_sampler import ResumableBatchSampler
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import (
    MANIFEST_VERSION,
    load_window_manifest,
    manifest_config_stamp,
    preprocess_replays,
    read_manifest_metadata,
)
from thesis_ml.eval.finetune_report import build_and_write_finetune_report
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.finished_export import export_finished_model
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
    report_path: str | None = None


def run_training_pipeline(
    config_path: str | Path,
    *,
    smoke: bool | None = None,
    storage: StorageResolver | None = None,
    max_steps_override: int | None = None,
    lr_override: float | None = None,
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
        result = _run_real_pipeline(
            config, resolver, max_steps_override=max_steps_override, lr_override=lr_override
        )
    _write_log(config, resolver, f"resumed={result.resumed} steps={result.steps} smoke={result.smoke}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SC2 diffusion training from config")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--smoke", action="store_true", help="force the tiny synthetic smoke pipeline")
    parser.add_argument("--max-steps", type=int, default=None, help="bounded verification override")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "override train.lr for this run without editing the config -- the "
            "quickest way to toggle the base learning rate between runs (e.g. "
            "--lr 3e-4 for from-scratch pre-training vs --lr 1e-6 for fine-tuning)"
        ),
    )
    args = parser.parse_args()
    try:
        result = run_training_pipeline(
            args.config,
            smoke=True if args.smoke else None,
            max_steps_override=args.max_steps,
            lr_override=args.lr,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(
        f"checkpoint_uri={result.checkpoint_uri} resumed={result.resumed} steps={result.steps} "
        f"parameters={result.parameter_count} peak_vram_bytes={result.peak_vram_bytes} "
        f"report_path={result.report_path or ''}"
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
        # Smoke fixtures are pre-training-shaped (absent input, collapsed
        # taxonomy), so collate in pre-training mode.
        collate_fn=partial(collate_diffusion_examples, debut_mode=False),
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
    lr_override: float | None = None,
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
        train_count=config.pipeline.train_replay_count,
        dev_count=config.pipeline.validation_replay_count,
    )
    train_replays, dev_replays = _select_replays(
        list(split.train),
        list(split.dev),
        config,
    )
    test_replays = list(split.test)
    _record_replay_selection(config, resolver, train_replays, dev_replays, test_replays)
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
    test_windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=test_replays,
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
    test_dataset = None
    if config.data.debut_mode and test_windows:
        test_dataset = SC2DiffusionDataset(
            test_windows,
            config,
            vocabulary,
            seed=config.pipeline.seed,
            fog_rate_override=None,
        )

    planned_steps = config.train.max_steps
    if planned_steps <= 0:
        planned_steps = len(train_loader) * config.train.epochs
    # --lr overrides the base learning rate for this run without editing YAML.
    # Applied here (before the optimizer is built inside TrainingLoop) so the
    # override actually takes effect; None leaves the config value untouched.
    effective_lr = config.train.lr if lr_override is None else lr_override
    training_config = replace(
        config,
        train=replace(
            config.train,
            checkpoint_dir=str(checkpoint_dir),
            max_steps=planned_steps,
            lr=effective_lr,
        ),
    )
    if lr_override is not None:
        print(f"lr_override: train.lr set to {effective_lr:.3e} via --lr", flush=True)
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
    # Guarantee DataLoader worker teardown on EVERY exit path (normal finish,
    # exception, or Ctrl+C). With persistent_workers=True the loaders keep their
    # worker processes and prefetch threads alive in loader._iterator until that
    # iterator is garbage-collected; on Windows an interrupted run may not GC it
    # promptly, orphaning python worker processes that keep pinning CPU. The
    # finally block shuts them down deterministically rather than relying on GC.
    try:
        loop.fit(
            train_loader,
            val_dataloader=val_loader,
            max_steps=requested_steps,
            epochs=training_config.train.epochs,
            retain_logs=False,
        )
    finally:
        _shutdown_dataloader(train_loader)
        _shutdown_dataloader(val_loader)
    peak_vram_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    checkpoint_uri = _publish_checkpoint(config, resolver, checkpoint_dir / "last.pt")

    # A "proper finish" is a real run that returned normally from loop.fit() --
    # it trained through every configured epoch or stopped via early stopping.
    # Reaching this line already guarantees the loop did not crash or get
    # interrupted (those propagate out of the try/finally above). We ADDITIONALLY
    # require this not to be a bounded --max-steps verification run
    # (max_steps_override is None), matching the finetune-report guard below: a
    # smoke/verification run is not a finished model. On a proper finish, write
    # the durable, separately-tagged raw+EMA finished model alongside last.pt.
    if max_steps_override is None:
        stop_reason = (
            "completed_all_epochs"
            if loop.completed_epochs >= training_config.train.epochs
            else "early_stopping"
        )
        export_finished_model(
            checkpoint_dir=checkpoint_dir,
            model=loop.model,
            ema_model=loop.ema_model,
            config=training_config,
            vocab_size=vocabulary.vocab_size,
            global_step=loop.global_step,
            completed_epochs=loop.completed_epochs,
            configured_epochs=training_config.train.epochs,
            stop_reason=stop_reason,
            publisher=_finished_publisher(config, resolver),
        )

    report_path: Path | None = None
    if config.data.debut_mode and max_steps_override is None:
        report_path = metrics_dir / "finetune_report.json"
        build_and_write_finetune_report(
            memorized_examples=_select_eval_examples(
                train_dataset, training_config.eval.debut_max_examples
            ),
            test_examples=(
                _select_eval_examples(test_dataset, training_config.eval.debut_max_examples)
                if test_dataset is not None
                else []
            ),
            model=loop.ema_model,
            vocabulary=vocabulary,
            config=training_config,
            path=report_path,
            device=device,
        )
    return TrainingPipelineResult(
        checkpoint_uri=checkpoint_uri,
        resumed=resumed,
        steps=loop.global_step,
        smoke=False,
        parameter_count=parameter_count,
        peak_vram_bytes=peak_vram_bytes,
        report_path=str(report_path) if report_path is not None else None,
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
    perspectives = _perspectives(config.pipeline.perspectives)
    manifest_path = Path(config.data.window_manifest_path)
    rebuild = config.pipeline.rebuild_manifest or not manifest_path.exists()
    if not rebuild:
        metadata = read_manifest_metadata(manifest_path)
        rebuild = any(
            (
                metadata.get("manifest_version") != MANIFEST_VERSION,
                metadata.get("config_stamp") != manifest_config_stamp(config),
                metadata.get("replay_count") != len(replay_paths),
                metadata.get("perspectives") != list(perspectives),
            )
        )
    if rebuild:
        if not config.pipeline.preprocess_if_missing:
            raise FileNotFoundError(f"window manifest is absent or stale: {manifest_path}")
        preprocess_replays(
            replay_paths,
            config,
            vocabulary,
            perspectives=perspectives,
            force=config.pipeline.rebuild_manifest,
        )


def _record_replay_selection(
    config: ProjectConfig,
    resolver: StorageResolver,
    train: list[str],
    validation: list[str],
    test: list[str],
) -> None:
    output = _local_metrics_dir(config, resolver) / "replay_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split_seed": config.pipeline.split_seed,
                "train_replay_ids": [Path(path).stem for path in train],
                "dev_replay_ids": [Path(path).stem for path in validation],
                "test_replay_ids": [Path(path).stem for path in test],
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
        # InputFeatures are built in the worker before raw TokenRecord metadata
        # is dropped. Training does not consume those Python object graphs, so
        # excluding them avoids expensive worker-to-main-process serialization.
        # debut_mode threads the pipeline mode EXPLICITLY into collate so the
        # future telemetry is scoped correctly (fine-tuning-only). Both the
        # pre-training and fine-tuning pipelines share this builder.
        "collate_fn": partial(
            collate_diffusion_examples,
            retain_metadata=False,
            debut_mode=config.data.debut_mode,
        ),
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = config.pipeline.prefetch_factor
        kwargs["persistent_workers"] = config.pipeline.persistent_workers
    if shuffle:
        # Training loader: replace the default shuffling RandomSampler with a
        # resumable, per-epoch-seeded batch sampler. This makes the batch order
        # reproducible across process restarts AND lets a run interrupted
        # mid-epoch skip the batches it already trained on, so training actually
        # advances through the epoch instead of replaying it from batch 1 on
        # every resume. The batch sampler subsumes batch_size/shuffle/drop_last,
        # which is why those keys must NOT also be passed to the DataLoader.
        batch_sampler = ResumableBatchSampler(
            dataset_size=len(dataset),
            batch_size=config.pipeline.batch_size,
            base_seed=config.pipeline.seed,
            drop_last=False,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    # Validation/eval loader: deterministic sequential order, and it is re-run
    # from scratch each time, so no resumption machinery is needed.
    return DataLoader(
        dataset,
        batch_size=config.pipeline.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )


def _shutdown_dataloader(loader: DataLoader | None) -> None:
    """Deterministically stop a DataLoader's worker processes and threads.

    A DataLoader created with ``persistent_workers=True`` caches its live
    iterator (with its worker subprocesses, prefetch threads, and the optional
    pin-memory thread) on ``loader._iterator`` and keeps it running between
    epochs -- and after the final epoch -- until that iterator object is
    garbage-collected. If a run is interrupted (Ctrl+C, an exception, or spot
    preemption) that GC can be delayed on Windows, leaving orphaned ``python``
    worker processes that continue to consume CPU. Calling this from a
    ``finally`` block frees them immediately regardless of how training ended.

    The public DataLoader API has no ``close()``, so we invoke the same
    ``_shutdown_workers`` routine the iterator's finalizer would eventually run,
    then drop the reference so a fresh iterator is built on any later reuse.

    Parameters:
        loader: the DataLoader to tear down, or ``None`` (a no-op) so callers
            can pass an optional validation loader without a guard.
    """

    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader._iterator = None
    # Reclaim the iterator (and any single-process resources) now instead of at
    # an unpredictable later GC, so worker teardown is fully synchronous here.
    gc.collect()


def _select_eval_examples(dataset, cap: int):
    """Return the lazy full dataset or a bounded evenly-strided sample."""

    total = len(dataset)
    if cap <= 0:
        return dataset
    if total <= cap:
        return [dataset[index] for index in range(total)]
    selected_indices = [(position * total) // cap for position in range(cap)]
    return [dataset[index] for index in selected_indices]


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


def _finished_publisher(config: ProjectConfig, resolver: StorageResolver) -> Callable[[Path], None] | None:
    """Return a callback that mirrors a finished-model file to remote storage.

    None when checkpoints are local-only. Otherwise each finished artifact is
    uploaded under the `finished/` prefix of the checkpoint URI (mirroring the
    local `<checkpoint_dir>/finished/` layout), so `model.ema.safetensors` lands
    at `<checkpoint_uri>/finished/model.ema.safetensors`. All finished files live
    flat in one directory, so uploading by basename preserves the layout.
    """

    checkpoint_uri = config.storage.checkpoint_uri
    if not resolver.is_s3(checkpoint_uri):
        return None

    def publish(local_path: Path) -> None:
        resolver.put_file(local_path, f"{checkpoint_uri.rstrip('/')}/finished/{local_path.name}")

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
    if len(values) != 2 or set(values) != {"p1", "p2"}:
        raise ValueError(
            "pipeline.perspectives must contain exactly p1,p2 so every replay is "
            "represented once from each player's perspective"
        )
    return ("p1", "p2")


if __name__ == "__main__":
    main()
