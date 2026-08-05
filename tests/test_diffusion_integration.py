"""Bounded integration coverage across diffusion training and inference boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import DiffusionBatch, collate_diffusion_examples
from thesis_ml.inference.sampler import load_sampling_checkpoint, sample_canvas
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.train.train import make_synthetic_examples
from thesis_ml.vocab.special_tokens import MASK_ID


def test_uniform_real_training_loss_backward_and_inference_flow(tmp_path: Path) -> None:
    config = _integration_config(tmp_path, process="uniform")
    batch = _batch(config)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=101)
    original_input = batch.input_token_ids.clone()

    batch_loss = loop.compute_batch_loss(batch, fixed_t=0.5)
    assert torch.equal(batch_loss.corruption.input_token_ids, original_input)
    assert torch.equal(batch_loss.scored_mask, batch.canvas_loss_mask)
    assert torch.isfinite(batch_loss.loss)
    batch_loss.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    sampled = sample_canvas(model, batch, config)
    assert sampled.canvas.shape == batch.target_canvas.shape
    assert 1 <= sampled.steps <= config.sampler.max_steps
    assert (sampled.canvas[batch.canvas_loss_mask] != MASK_ID).all()
    assert torch.equal(sampled.input_token_ids, original_input)
    assert torch.equal(sampled.initial_input_token_ids, original_input)


def test_absorbing_real_corruption_loss_and_inference_are_process_isolated(
    tmp_path: Path,
) -> None:
    config = _integration_config(tmp_path, process="absorbing")
    batch = _batch(config)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=102)

    batch_loss = loop.compute_batch_loss(batch, fixed_t=1.0)
    assert batch_loss.corruption.corrupted_positions.all()
    assert batch_loss.corruption.noised_canvas.eq(MASK_ID).all()
    assert torch.equal(batch_loss.scored_mask, batch.canvas_loss_mask)
    assert torch.allclose(
        batch_loss.corruption.position_weights,
        torch.ones_like(batch_loss.corruption.position_weights),
    )
    assert torch.isfinite(batch_loss.loss)
    batch_loss.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    sampled = sample_canvas(model, batch, config)
    assert sampled.steps == 1
    assert sampled.stop_reasons == ("absorbing_complete", "absorbing_complete")
    assert not sampled.trace[0].renoised_mask.any()
    assert (sampled.canvas[batch.canvas_loss_mask] != MASK_ID).all()
    assert model.diffusion_process == "absorbing"


def test_self_conditioning_continues_from_training_gate_to_inference_reuse(
    tmp_path: Path,
) -> None:
    config = _integration_config(tmp_path, process="uniform", self_cond_prob=1.0)
    batch = _batch(config)
    model = RecordingDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=103)

    batch_loss = loop.compute_batch_loss(batch, fixed_t=0.5)
    assert batch_loss.self_conditioning_mask.all()
    assert len(model.conditioning_calls) == 2
    assert model.conditioning_calls[0] is None
    training_signal = model.conditioning_calls[1]
    assert training_signal is not None
    assert training_signal.shape == (*batch.target_canvas.shape, config.model.d_model)
    assert not training_signal.requires_grad

    model.conditioning_calls.clear()
    sampled = sample_canvas(model, batch, config)
    assert sampled.steps == config.sampler.max_steps
    assert len(model.conditioning_calls) == sampled.steps
    assert model.conditioning_calls[0] is None
    assert all(signal is not None for signal in model.conditioning_calls[1:])
    assert all(not signal.requires_grad for signal in model.conditioning_calls[1:] if signal is not None)


def test_training_checkpoint_round_trips_to_inference_and_rejects_cross_process(
    tmp_path: Path,
) -> None:
    uniform = _integration_config(tmp_path, process="uniform")
    source_model = SC2StrategyDiffusionModel(uniform, vocab_size=128)
    loop = TrainingLoop(model=source_model, config=uniform, seed=104)
    checkpoint = loop.save_checkpoint(tmp_path / "uniform.pt")

    inference_model = SC2StrategyDiffusionModel(uniform, vocab_size=128)
    load_sampling_checkpoint(inference_model, checkpoint)
    for name, tensor in loop.ema_model.state_dict().items():
        assert torch.equal(inference_model.state_dict()[name], tensor)

    absorbing = _integration_config(tmp_path, process="absorbing")
    cross_process_model = SC2StrategyDiffusionModel(absorbing, vocab_size=128)
    before = {
        name: tensor.detach().clone()
        for name, tensor in cross_process_model.state_dict().items()
    }
    with pytest.raises(ValueError, match="diffusion process mismatch"):
        load_sampling_checkpoint(cross_process_model, checkpoint)
    for name, tensor in cross_process_model.state_dict().items():
        assert torch.equal(tensor, before[name])


class RecordingDiffusionModel(SC2StrategyDiffusionModel):
    def __init__(self, config: ProjectConfig, *, vocab_size: int) -> None:
        super().__init__(config, vocab_size=vocab_size)
        self.conditioning_calls: list[torch.Tensor | None] = []

    def forward(self, *args, canvas_self_conditioning=None, **kwargs):
        self.conditioning_calls.append(
            canvas_self_conditioning
            if canvas_self_conditioning is None
            else canvas_self_conditioning.detach().clone()
        )
        return super().forward(
            *args,
            canvas_self_conditioning=canvas_self_conditioning,
            **kwargs,
        )


def _integration_config(
    tmp_path: Path,
    *,
    process: str,
    self_cond_prob: float = 0.5,
) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(
            config.data,
            input_budget_tokens=64,
            canvas_budget_tokens=12,
        ),
        model=replace(
            config.model,
            d_model=16,
            layers=1,
            heads=4,
            ffn=32,
        ),
        diffusion=replace(config.diffusion, process=process),
        train=replace(
            config.train,
            precision="fp32",
            self_cond_prob=self_cond_prob,
            confidence_loss_weight=0.0,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
        sampler=replace(
            config.sampler,
            max_steps=2,
            entropy_bound=100.0,
            adaptive_stop=False,
        ),
    )


def _batch(config: ProjectConfig) -> DiffusionBatch:
    examples = make_synthetic_examples(config, count=2)
    return collate_diffusion_examples(examples, debut_mode=False)
