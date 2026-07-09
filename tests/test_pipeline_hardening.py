"""Tests for the cloud-readiness hardening: split, RAM cache, checkpoint sync, metrics."""

from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.frame_cache import BoundedFrameCache, resolve_cache_budget_bytes
from thesis_ml.data.split import split_replays
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.storage import StorageResolver
from thesis_ml.pipeline.train_pipeline import _resume_from_remote
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.train.train import make_synthetic_examples


def test_split_replays_is_deterministic_disjoint_and_covers_all() -> None:
    paths = [f"replay_{index}.parquet" for index in range(100)]

    first = split_replays(paths, seed=1234, test_fraction=0.15, dev_fraction=0.10)
    second = split_replays(paths, seed=1234, test_fraction=0.15, dev_fraction=0.10)

    # Reproducible: same seed -> identical partition.
    assert first == second
    # Disjoint and total coverage (no leakage, nothing dropped).
    assert set(first.train) | set(first.dev) | set(first.test) == set(paths)
    assert len(first.train) + len(first.dev) + len(first.test) == len(paths)
    assert set(first.train).isdisjoint(first.dev)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.dev).isdisjoint(first.test)
    # Roughly 15% test, then 10% of the remaining 85% as dev
    # (round(8.5) == 8 under banker's rounding).
    assert len(first.test) == 15
    assert len(first.dev) == 8
    assert len(first.train) == 77

    # A different seed yields a different partition.
    other = split_replays(paths, seed=9999, test_fraction=0.15, dev_fraction=0.10)
    assert other != first


def test_split_replays_supports_exact_counts_and_holds_out_every_remainder() -> None:
    paths = [f"replay_{index}.parquet" for index in range(943)]

    split = split_replays(
        paths,
        seed=2718,
        test_fraction=0.15,
        dev_fraction=0.10,
        train_count=870,
        dev_count=50,
    )

    assert (len(split.train), len(split.dev), len(split.test)) == (870, 50, 23)
    assert set(split.train) | set(split.dev) | set(split.test) == set(paths)
    assert set(split.train).isdisjoint(split.dev)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.dev).isdisjoint(split.test)


def test_cache_budget_divides_across_workers() -> None:
    single = resolve_cache_budget_bytes(0.5, num_workers=1)
    four = resolve_cache_budget_bytes(0.5, num_workers=4)
    # Aggregate footprint stays bounded: per-worker budget shrinks with workers.
    assert four <= single
    # Never below the 256 MB floor so one large frame always fits.
    assert four >= 256 * 1024**2


def test_bounded_frame_cache_evicts_least_recently_used() -> None:
    cache = BoundedFrameCache(ram_fraction=0.5)
    # Force a tiny budget so two frames fit but a third forces an eviction.
    frame = pd.DataFrame({"game_loop": range(1000), "value": range(1000)})
    one_frame_bytes = int(frame.memory_usage(deep=True).sum())
    cache._budget_bytes = int(one_frame_bytes * 2.5)

    loads: list[Path] = []

    def loader(path: Path) -> pd.DataFrame:
        loads.append(path)
        return frame.copy()

    a, b, c = Path("a"), Path("b"), Path("c")
    cache.get(a, loader)
    cache.get(b, loader)
    cache.get(a, loader)  # hit -> makes 'a' most-recently-used
    cache.get(c, loader)  # insert 'c' -> evicts LRU, which is 'b' (not 'a')

    assert loads == [a, b, c]  # only misses triggered a load; 'a' second time was a hit
    assert len(cache) == 2
    # 'a' survived (recently used), 'b' was evicted -> re-loads.
    cache.get(a, loader)
    assert loads == [a, b, c]  # still cached, no new load
    cache.get(b, loader)
    assert loads == [a, b, c, b]  # 'b' had to be re-loaded


def test_training_loop_writes_metrics_jsonl_and_publishes(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    config = replace(config, train=replace(config.train, checkpoint_interval=1))
    metrics_path = tmp_path / "metrics.jsonl"
    published_checkpoints: list[str] = []
    published_metrics: list[str] = []

    torch.manual_seed(5)
    examples = make_synthetic_examples(config, count=2)
    # Pre-training fixtures (make_synthetic_examples) -> pre-training collate.
    batch = next(
        iter(
            DataLoader(
                examples,
                batch_size=2,
                shuffle=False,
                collate_fn=partial(collate_diffusion_examples, debut_mode=False),
            )
        )
    )
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model,
        config=config,
        seed=5,
        metrics_path=metrics_path,
        checkpoint_publisher=lambda path: published_checkpoints.append(path.name),
        metrics_publisher=lambda path: published_metrics.append(path.name),
    )

    logs = loop.fit([batch], max_steps=2, fixed_t=1.0)

    # One JSONL line per step, each valid JSON carrying loss + per-class.
    import json

    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "loss" in first and "per_class" in first and first["step"] == 1
    # Pre-training JSONL carries the t-bucket / perspective breakdowns but must
    # be grep-clean of the fine-tuning-only "future_distance" key: the writer
    # strips it from the serialized record entirely (not even an empty {}).
    assert "t_bucket_loss" in first
    assert "perspective_loss" in first
    assert "future_distance" not in lines[0]
    assert "future_distance" not in lines[1]
    assert first["step_wall_seconds"] > 0
    assert first["tokens_per_second"] > 0
    assert first["cuda_max_memory_allocated_bytes"] == 0
    assert first["cuda_memory_reserved_bytes"] == 0
    # Periodic checkpoint (interval=1) published last.pt and the metrics file.
    assert "last.pt" in published_checkpoints
    assert "metrics.jsonl" in published_metrics
    assert logs[-1].step == 2


def test_resume_from_remote_pulls_last_checkpoint(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    config = replace(
        config,
        storage=replace(config.storage, checkpoint_uri="s3://bucket/checkpoints"),
    )
    # Produce a real checkpoint, then serve it through a fake S3 resolver.
    torch.manual_seed(3)
    source_model = SC2StrategyDiffusionModel(config, vocab_size=128)
    source_loop = TrainingLoop(model=source_model, config=config, seed=3)
    source_loop.global_step = 7
    saved = source_loop.save_checkpoint(tmp_path / "remote_last.pt")

    resolver = StorageResolver(s3_client=_FakeS3(saved))
    target_loop = TrainingLoop(model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=3)
    local_dir = tmp_path / "fresh_instance"
    local_dir.mkdir()

    resumed = _resume_from_remote(target_loop, config, resolver, local_dir)

    assert resumed is True
    assert target_loop.global_step == 7


def test_collated_model_features_exclude_absolute_game_time(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    torch.manual_seed(13)
    examples = make_synthetic_examples(config, count=2)
    # Pre-training fixtures -> pre-training collate mode.
    batch = collate_diffusion_examples(examples, debut_mode=False)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    model.eval()

    with torch.no_grad():
        output = model(
            input_token_ids=batch.input_token_ids,
            canvas_token_ids=batch.target_canvas,
            input_attention_mask=batch.input_attention_mask,
            input_features=batch.input_features,
        ).logits

    assert not hasattr(batch.input_features, "clock_values")
    assert output.shape[:2] == (
        batch.input_token_ids.shape[0],
        batch.input_token_ids.shape[1] + batch.target_canvas.shape[1],
    )


class _FakeS3:
    """Minimal S3 stand-in: head/list/download backed by one local file."""

    def __init__(self, local_checkpoint: Path) -> None:
        self._local = local_checkpoint

    def head_object(self, *, Bucket: str, Key: str):
        return {}

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_bytes(self._local.read_bytes())


def _small_config(tmp_path: Path) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64),
        train=replace(
            config.train,
            lr=0.01,
            warmup=1,
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=2,
            val_interval=0,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            ema_decay=0.9,
            confidence_loss_weight=0.1,
            precision="fp32",
        ),
    )
