from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch
from torch import nn

from scripts.run_size_ablation import ARMS, _inspect_arm
from thesis_ml.config import load_config
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.feature_stats import FeatureStatistics
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.train_pipeline import _resolve_step_budgets
from thesis_ml.train.corruption import corrupt_batch
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.vocab.content_vocab import load_content_vocabulary


ROOT = Path(__file__).resolve().parents[1]


def test_size_ablation_profiles_are_ordered_scaled_and_exactly_counted() -> None:
    expected_shapes = {
        "005m": (192, 12, 3, 384),
        "015m": (320, 14, 5, 640),
        "030m-baseline": (384, 12, 6, 1536),
        "030m-deep": (384, 18, 6, 896),
        "060m": (512, 22, 8, 1024),
        "120m": (768, 20, 12, 1536),
    }
    assert [arm.name for arm in ARMS] == list(expected_shapes)
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    for arm in ARMS:
        config = load_config(ROOT / arm.config_path)
        observed_shape = (
            config.model.d_model,
            config.model.layers,
            config.model.heads,
            config.model.ffn,
        )
        assert observed_shape == expected_shapes[arm.name]
        assert config.model.d_model // config.model.heads == 64
        assert config.train.epochs == 3
        assert config.train.schedule_horizon_epochs == 50
        assert config.train.max_steps == 0
        assert config.pipeline.batch_size * config.train.accumulation_steps >= 42
        # Meta construction exercises the live module graph and exact parameter
        # inventory without allocating model weights on CPU or touching CUDA.
        with torch.device("meta"):
            model = SC2StrategyDiffusionModel(
                config,
                vocab_size=vocabulary.vocab_size,
                feature_statistics=FeatureStatistics.identity_for_tests(),
            )
        assert sum(parameter.numel() for parameter in model.parameters()) == arm.expected_parameters


def test_size_ablation_run_and_schedule_step_budgets_are_separate() -> None:
    for arm in ARMS:
        config = load_config(ROOT / arm.config_path)
        train_batches = 101
        steps_per_epoch = math.ceil(train_batches / config.train.accumulation_steps)
        run_steps, schedule_steps = _resolve_step_budgets(config, train_batches)
        assert run_steps == steps_per_epoch * 3
        assert schedule_steps == steps_per_epoch * 50


def test_finished_arm_is_skipped_only_when_its_export_is_complete(tmp_path: Path) -> None:
    arm = ARMS[0]
    config = load_config(ROOT / arm.config_path)
    config = replace(
        config,
        storage=replace(config.storage, checkpoint_uri=str(tmp_path / "checkpoint")),
    )
    finished = Path(config.storage.checkpoint_uri) / "finished"
    finished.mkdir(parents=True)
    (finished / "config.json").write_text(
        json.dumps(asdict(config)),
        encoding="utf-8",
    )
    metadata = {
        "stop_reason": "completed_all_epochs",
        "completed_epochs": 3,
        "configured_epochs": 3,
        "weights": {"raw": "model.raw.safetensors", "ema": "model.ema.safetensors"},
        "torch_bundle": "finished.pt",
        "config_file": "config.json",
    }
    (finished / "finished_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert _inspect_arm(arm, config)["state"] == "blocked"
    for name in ("model.raw.safetensors", "model.ema.safetensors", "finished.pt"):
        (finished / name).write_bytes(b"test")
    assert _inspect_arm(arm, config)["state"] == "done"

    saved = json.loads((finished / "config.json").read_text(encoding="utf-8"))
    saved["model"]["layers"] += 1
    (finished / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    assert _inspect_arm(arm, config)["state"] == "blocked"


def test_paired_corruption_is_invariant_to_microbatch_boundaries() -> None:
    config = load_config(ROOT / "configs" / "size_ablation_005m.yaml")
    target = torch.arange(3 * 32, dtype=torch.long).reshape(3, 32) % 20 + 8
    target[:, 0] = 6  # clamped [BOS]
    inputs = torch.ones((3, 4), dtype=torch.long)
    seeds = torch.tensor([101, 202, 303], dtype=torch.int64)
    together = corrupt_batch(
        input_token_ids=inputs,
        target_canvas=target,
        process=config.diffusion.process,
        schedule=config.diffusion.schedule,
        vocab_size=291,
        canvas_noise_mask=target.ne(6),
        row_seeds=seeds,
    )
    separate = [
        corrupt_batch(
            input_token_ids=inputs[row : row + 1],
            target_canvas=target[row : row + 1],
            process=config.diffusion.process,
            schedule=config.diffusion.schedule,
            vocab_size=291,
            canvas_noise_mask=target[row : row + 1].ne(6),
            row_seeds=seeds[row : row + 1],
        )
        for row in range(3)
    ]
    assert torch.equal(together.t, torch.cat([item.t for item in separate]))
    assert torch.equal(
        together.corrupted_positions,
        torch.cat([item.corrupted_positions for item in separate]),
    )
    assert torch.equal(
        together.noised_canvas,
        torch.cat([item.noised_canvas for item in separate]),
    )


def test_fog_and_self_conditioning_are_epoch_keyed_per_example() -> None:
    config = load_config(ROOT / "configs" / "size_ablation_005m.yaml")
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    first = SC2DiffusionDataset([], config, vocabulary, seed=123)
    second = SC2DiffusionDataset([], config, vocabulary, seed=123)
    assert first._epoch.is_shared()
    first.set_epoch(2)
    second.set_epoch(2)
    assert first._rng_for_index(17).random(20).tolist() == second._rng_for_index(17).random(20).tolist()
    assert first._stochastic_seed_for_index(17) == second._stochastic_seed_for_index(17)
    previous_seed = first._stochastic_seed_for_index(17)
    first.set_epoch(3)
    assert first._stochastic_seed_for_index(17) != previous_seed

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.vocab_size = 291
            self.architecture_identity = "test"
            self.diffusion_process = "uniform"
            self.feature_statistics_identity = "identity-tests"

    tiny_config = replace(
        config,
        train=replace(config.train, checkpoint_dir="tests/output/SizeAblationTest/test-checkpoint"),
    )
    loop = TrainingLoop(model=TinyModel(), config=tiny_config, seed=123)
    row_seeds = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
    together = loop._sample_self_conditioning_rows(4, row_seeds=row_seeds)
    separate = torch.cat(
        [
            loop._sample_self_conditioning_rows(1, row_seeds=row_seeds[index : index + 1])
            for index in range(4)
        ]
    )
    assert torch.equal(together, separate)


def test_fallback_training_generator_continues_from_checkpoint(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "default.yaml")

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.vocab_size = 291
            self.architecture_identity = "test-generator-resume"
            self.diffusion_process = "uniform"
            self.feature_statistics_identity = "identity-tests"

    config = replace(
        config,
        train=replace(config.train, checkpoint_dir=str(tmp_path), max_steps=10),
    )
    original = TrainingLoop(model=TinyModel(), config=config, seed=987)
    torch.rand(37, generator=original.generator)
    checkpoint = original.save_checkpoint(tmp_path / "resume" / "last.pt")
    expected = torch.rand(19, generator=original.generator)

    restored = TrainingLoop(model=TinyModel(), config=config, seed=987)
    restored.load_checkpoint(checkpoint)
    observed = torch.rand(19, generator=restored.generator)
    assert torch.equal(observed, expected)
    assert restored._generator_state_restored is True
