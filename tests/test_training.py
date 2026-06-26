from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.model.loss import CLASS_ID_TO_NAME
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.corruption import corrupt_batch, inverse_t_weights
from thesis_ml.train.loop import TrainingLoop, auxiliary_confidence_loss
from thesis_ml.train.train import make_synthetic_examples, run_smoke_train


def test_smoke_train_loss_decreases_and_first_step_per_class_logs(tmp_path: Path) -> None:
    logs = run_smoke_train(max_steps=40, seed=17, checkpoint_dir=tmp_path / "smoke")

    first = logs[0]
    last = logs[-1]
    assert last.loss < first.loss

    examples = make_synthetic_examples(_small_config(tmp_path), count=1)
    expected_classes = {CLASS_ID_TO_NAME[int(label)] for label in examples[0].class_labels.unique()}
    assert set(first.per_class) == expected_classes
    assert all(value > 0 for value in first.per_class.values())


def test_corruption_never_masks_input_region() -> None:
    config = _small_config()
    input_token_ids = torch.tensor([[100, 101, 102], [103, 104, 105]])
    target_canvas = torch.tensor([[100, 101, 102, 103], [104, 105, 106, 107]])
    generator = torch.Generator(device="cpu").manual_seed(1)

    for t in (0.0, 0.25, 0.75, 1.0):
        corrupted = corrupt_batch(
            input_token_ids=input_token_ids,
            target_canvas=target_canvas,
            schedule=config.diffusion.mask_schedule,
            generator=generator,
            t=t,
        )
        assert torch.equal(corrupted.input_token_ids, input_token_ids)
        assert not (corrupted.input_token_ids == 0).any()
        if t == 1.0:
            assert corrupted.masked_positions.all()
            assert (corrupted.noised_canvas == 0).all()


def test_training_scores_exactly_masked_canvas_positions(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=9)

    result = loop.compute_batch_loss(batch, fixed_t=0.5)

    assert torch.equal(result.scored_mask, result.corruption.masked_positions)
    assert not result.scored_mask[result.corruption.masked_positions.logical_not()].any()
    assert result.canvas_logits.shape[1] == batch.target_canvas.shape[1]


def test_self_conditioning_training_uses_no_grad_estimate_then_grad_pass(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        train=replace(_small_config(tmp_path).train, self_cond_prob=1.0),
    )
    torch.manual_seed(61)
    examples = make_synthetic_examples(config, count=2)
    batch = next(iter(DataLoader(examples, batch_size=2, shuffle=False, collate_fn=collate_diffusion_examples)))
    model = CountingDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=61)

    result = loop.compute_batch_loss(batch, fixed_t=1.0)

    assert result.self_conditioning_used is True
    assert model.forward_records == [(False, False), (True, True)]
    assert loop.global_step == 0
    assert loop.optimizer.state_dict()["state"] == {}


def test_self_conditioning_off_uses_single_legacy_training_forward(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        model=replace(_small_config(tmp_path).model, self_conditioning=False),
        train=replace(_small_config(tmp_path).train, self_cond_prob=1.0),
    )
    torch.manual_seed(62)
    examples = make_synthetic_examples(config, count=2)
    batch = next(iter(DataLoader(examples, batch_size=2, shuffle=False, collate_fn=collate_diffusion_examples)))
    model = CountingDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=62)

    result = loop.compute_batch_loss(batch, fixed_t=1.0)

    assert result.self_conditioning_used is False
    assert model.forward_records == [(True, False)]


def test_schedule_weighting_uses_inverse_t_not_flat() -> None:
    weights = inverse_t_weights(torch.tensor([0.25, 0.75]), canvas_len=3)

    assert weights[0, 0].item() == pytest.approx(4.0)
    assert weights[1, 0].item() == pytest.approx(4.0 / 3.0)
    assert not torch.allclose(weights, torch.ones_like(weights))
    assert weights[0, 0] > weights[1, 0]


def test_checkpoint_roundtrip_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=21)
    loop.fit([batch], max_steps=1, fixed_t=1.0)
    checkpoint = loop.save_checkpoint(tmp_path / "manual.pt")

    restored_model = SC2StrategyDiffusionModel(config, vocab_size=128)
    restored = TrainingLoop(model=restored_model, config=config, seed=21)
    restored.load_checkpoint(checkpoint)

    assert restored.global_step == loop.global_step
    for saved, loaded in zip(loop.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.allclose(saved, loaded)
    for saved, loaded in zip(loop.ema_model.parameters(), restored.ema_model.parameters(), strict=True):
        assert torch.allclose(saved, loaded)
    _assert_optimizer_states_match(loop.optimizer.state_dict(), restored.optimizer.state_dict())

    logs = restored.fit([batch], max_steps=2, fixed_t=1.0)
    assert restored.global_step == 2
    assert logs


def test_ema_tracks_training_and_validation_uses_ema_weights(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=31)
    initial_ema = [parameter.detach().clone() for parameter in loop.ema_model.parameters()]

    loop.fit([batch], max_steps=1, fixed_t=1.0)

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(initial_ema, loop.ema_model.parameters(), strict=True)
    )
    assert any(
        not torch.allclose(raw, ema)
        for raw, ema in zip(loop.model.parameters(), loop.ema_model.parameters(), strict=True)
    )

    validation = loop.validate([batch], fixed_t=1.0)
    expected = loop.compute_batch_loss(batch, fixed_t=1.0, model=loop.ema_model)
    assert validation.loss == pytest.approx(float(expected.loss.detach()))
    assert validation.per_class


def test_confidence_loss_is_weighted_and_disableable(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=41)
    weighted = loop.compute_batch_loss(batch, fixed_t=1.0)

    off_config = replace(config, train=replace(config.train, confidence_loss_weight=0.0))
    off_loop, off_batch = _loop_and_batch(off_config, seed=41)
    disabled = off_loop.compute_batch_loss(off_batch, fixed_t=1.0)

    assert torch.allclose(disabled.loss, disabled.denoising_loss)
    assert disabled.confidence_loss.item() == pytest.approx(0.0)
    if weighted.confidence_loss.item() > 0:
        assert torch.allclose(weighted.loss, weighted.denoising_loss + weighted.confidence_loss)

    logits = torch.zeros(1, 2, 8)
    logits[0, 0, 3] = 4.0
    logits[0, 1, 2] = 4.0
    targets = torch.tensor([[3, 7]])
    scored = torch.tensor([[True, True]])
    assert auxiliary_confidence_loss(logits, targets, scored).item() > 0.0


def test_periodic_validation_logs_held_out_loss_from_ema(tmp_path: Path) -> None:
    config = replace(_small_config(tmp_path), train=replace(_small_config(tmp_path).train, val_interval=1))
    torch.manual_seed(51)
    train_examples = make_synthetic_examples(config, count=2)
    val_examples = make_synthetic_examples(config, count=4)[2:]
    assert {example.window_start for example in train_examples}.isdisjoint(
        {example.window_start for example in val_examples}
    )
    train_loader = DataLoader(train_examples, batch_size=2, shuffle=False, collate_fn=collate_diffusion_examples)
    val_loader = DataLoader(val_examples, batch_size=2, shuffle=False, collate_fn=collate_diffusion_examples)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=51)

    logs = loop.fit(train_loader, val_dataloader=val_loader, max_steps=1, fixed_t=1.0)

    assert logs[0].validation is not None
    assert logs[0].validation.loss > 0
    assert logs[0].validation.per_class


def test_seeded_smoke_runs_are_deterministic(tmp_path: Path) -> None:
    first = run_smoke_train(max_steps=5, seed=99, checkpoint_dir=tmp_path / "a")
    second = run_smoke_train(max_steps=5, seed=99, checkpoint_dir=tmp_path / "b")

    assert [log.loss for log in first] == pytest.approx([log.loss for log in second])
    assert [log.masked_fraction for log in first] == pytest.approx([log.masked_fraction for log in second])
    assert [log.per_class for log in first] == [log.per_class for log in second]


def _small_config(tmp_path: Path | None = None) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_window_timesteps=4, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64),
        train=replace(
            config.train,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            adam_eps=1e-8,
            warmup=1,
            lr_floor_ratio=0.1,
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=8,
            val_interval=0,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path / "checkpoints") if tmp_path is not None else "checkpoints/test",
            ema_decay=0.9,
            confidence_loss_weight=0.1,
            precision="fp32",
        ),
    )


def _loop_and_batch(config: ProjectConfig, *, seed: int) -> tuple[TrainingLoop, object]:
    torch.manual_seed(seed)
    examples = make_synthetic_examples(config, count=2)
    dataloader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=collate_diffusion_examples)
    batch = next(iter(dataloader))
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    return TrainingLoop(model=model, config=config, seed=seed), batch


class CountingDiffusionModel(SC2StrategyDiffusionModel):
    def __init__(self, config: ProjectConfig, *, vocab_size: int) -> None:
        super().__init__(config, vocab_size=vocab_size)
        self.forward_records: list[tuple[bool, bool]] = []

    def forward(self, *args, canvas_self_conditioning=None, **kwargs):
        self.forward_records.append((torch.is_grad_enabled(), canvas_self_conditioning is not None))
        return super().forward(*args, canvas_self_conditioning=canvas_self_conditioning, **kwargs)


def _assert_optimizer_states_match(first: dict, second: dict) -> None:
    assert first["param_groups"] == second["param_groups"]
    assert first["state"].keys() == second["state"].keys()
    for key in first["state"]:
        for state_name, state_value in first["state"][key].items():
            loaded_value = second["state"][key][state_name]
            if isinstance(state_value, torch.Tensor):
                assert torch.allclose(state_value, loaded_value)
            else:
                assert state_value == loaded_value
