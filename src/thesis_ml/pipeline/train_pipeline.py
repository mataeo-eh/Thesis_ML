"""Master cloud-runnable training pipeline entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import SC2DiffusionDataset
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


def run_training_pipeline(
    config_path: str | Path,
    *,
    smoke: bool | None = None,
    storage: StorageResolver | None = None,
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
        result = _run_real_pipeline(config, resolver)
    _write_log(config, resolver, f"resumed={result.resumed} steps={result.steps} smoke={result.smoke}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SC2 diffusion training from config")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--smoke", action="store_true", help="force the tiny synthetic smoke pipeline")
    args = parser.parse_args()
    result = run_training_pipeline(args.config, smoke=True if args.smoke else None)
    print(f"checkpoint_uri={result.checkpoint_uri} resumed={result.resumed} steps={result.steps}")


def _run_smoke_pipeline(config: ProjectConfig, resolver: StorageResolver) -> TrainingPipelineResult:
    torch.manual_seed(config.pipeline.seed)
    checkpoint_dir = _local_checkpoint_dir(config, resolver)
    smoke_config = replace(
        config,
        data=replace(config.data, input_window_timesteps=4, canvas_budget_tokens=12),
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
    )


def _run_real_pipeline(config: ProjectConfig, resolver: StorageResolver) -> TrainingPipelineResult:
    torch.manual_seed(config.pipeline.seed)
    checkpoint_dir = _local_checkpoint_dir(config, resolver)
    token_dictionary = _materialize_file(config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver)
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    dataset = SC2DiffusionDataset(
        replay_paths,
        config,
        vocabulary,
        seed=config.pipeline.seed,
        examples_per_replay=config.pipeline.examples_per_replay,
        perspectives=_perspectives(config.pipeline.perspectives),
        fog_rate_override=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.pipeline.batch_size,
        shuffle=False,
        collate_fn=collate_diffusion_examples,
    )
    training_config = replace(config, train=replace(config.train, checkpoint_dir=str(checkpoint_dir)))
    model = SC2StrategyDiffusionModel(training_config, vocab_size=vocabulary.vocab_size)
    loop = TrainingLoop(model=model, config=training_config, seed=config.pipeline.seed)
    resumed = _try_resume(loop, checkpoint_dir)
    loop.fit(dataloader, max_steps=training_config.train.max_steps)
    checkpoint_uri = _publish_checkpoint(config, resolver, checkpoint_dir / "last.pt")
    return TrainingPipelineResult(
        checkpoint_uri=checkpoint_uri,
        resumed=resumed,
        steps=loop.global_step,
        smoke=False,
    )


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
