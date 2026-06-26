"""Canvas-only absorbing-state corruption for MDLM/LLaDA training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thesis_ml.config import MaskScheduleConfig
from thesis_ml.vocab.special_tokens import MASK_ID

MIN_T = 1e-4


@dataclass(frozen=True)
class CorruptionOutput:
    input_token_ids: torch.Tensor
    noised_canvas: torch.Tensor
    masked_positions: torch.Tensor
    t: torch.Tensor
    position_weights: torch.Tensor


def corrupt_batch(
    *,
    input_token_ids: torch.Tensor,
    target_canvas: torch.Tensor,
    schedule: MaskScheduleConfig,
    generator: torch.Generator | None = None,
    t: torch.Tensor | float | None = None,
    mask_token_id: int = MASK_ID,
) -> CorruptionOutput:
    """Apply linear MDLM/LLaDA absorbing-mask corruption to the canvas only.

    The project uses the LLaDA/MDLM linear masking objective: sample one global
    masking level t per example, mask canvas positions iid with probability t,
    and weight masked-position CE by approximately 1/t. The clamped input is
    returned by reference and is never edited here.
    """

    if schedule.name != "linear":
        raise ValueError(f"unsupported mask schedule: {schedule.name}")
    if schedule.t_distribution != "uniform":
        raise ValueError(f"unsupported t distribution: {schedule.t_distribution}")
    if schedule.loss_reweight != "inverse_t":
        raise ValueError(f"unsupported loss reweighting: {schedule.loss_reweight}")

    sampled_t = _resolve_t(target_canvas, schedule, generator=generator, t=t)
    probabilities = sampled_t.unsqueeze(1).expand_as(target_canvas)
    random = torch.rand(
        target_canvas.shape,
        device=target_canvas.device,
        generator=generator,
    )
    masked_positions = random < probabilities
    noised_canvas = torch.where(masked_positions, torch.full_like(target_canvas, mask_token_id), target_canvas)
    position_weights = inverse_t_weights(sampled_t, target_canvas.shape[1])

    return CorruptionOutput(
        input_token_ids=input_token_ids,
        noised_canvas=noised_canvas,
        masked_positions=masked_positions,
        t=sampled_t,
        position_weights=position_weights,
    )


def inverse_t_weights(t: torch.Tensor, canvas_len: int) -> torch.Tensor:
    """Return the linear MDLM/LLaDA per-position schedule weight."""

    return (1.0 / t.clamp_min(MIN_T)).unsqueeze(1).expand(-1, canvas_len)


def _resolve_t(
    target_canvas: torch.Tensor,
    schedule: MaskScheduleConfig,
    *,
    generator: torch.Generator | None,
    t: torch.Tensor | float | None,
) -> torch.Tensor:
    batch_size = target_canvas.shape[0]
    device = target_canvas.device
    if t is None:
        sampled = torch.rand(batch_size, device=device, generator=generator)
        sampled = schedule.min + sampled * (schedule.max - schedule.min)
    elif isinstance(t, torch.Tensor):
        sampled = t.to(device=device, dtype=torch.float32)
        if sampled.ndim == 0:
            sampled = sampled.expand(batch_size)
    else:
        sampled = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)

    if sampled.shape != (batch_size,):
        raise ValueError(f"t must be scalar or shape ({batch_size},), got {tuple(sampled.shape)}")
    return sampled.clamp_min(MIN_T)
