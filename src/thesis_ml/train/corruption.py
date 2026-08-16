"""Canvas-only corruption for uniform diffusion and its absorbing ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thesis_ml.config import DiffusionScheduleConfig
from thesis_ml.vocab.special_tokens import (
    BOS_ID,
    CONTENT_TOKEN_OFFSET,
    DELIMITER_ID,
    MASK_ID,
    PAD_ID,
)

MIN_T = 1e-4
SUPPORTED_PROCESSES = frozenset({"uniform", "absorbing"})


@dataclass(frozen=True)
class CorruptionOutput:
    input_token_ids: torch.Tensor
    noised_canvas: torch.Tensor
    corrupted_positions: torch.Tensor
    changed_positions: torch.Tensor
    t: torch.Tensor
    position_weights: torch.Tensor


def corrupt_batch(
    *,
    input_token_ids: torch.Tensor,
    target_canvas: torch.Tensor,
    process: str,
    schedule: DiffusionScheduleConfig,
    vocab_size: int,
    generator: torch.Generator | None = None,
    t: torch.Tensor | float | None = None,
    mask_token_id: int = MASK_ID,
    canvas_noise_mask: torch.Tensor | None = None,
) -> CorruptionOutput:
    """Apply the configured linear corruption process to the canvas only.

    ``corrupted_positions`` records the Bernoulli corruption branch. In uniform
    mode a sampled replacement may equal the target, so ``changed_positions``
    separately records actual token inequality. The clamped input is returned
    unchanged and is never noised.
    """

    _validate_arguments(process, schedule, vocab_size, mask_token_id)
    sampled_t = _resolve_t(target_canvas, schedule, generator=generator, t=t)
    probabilities = sampled_t.unsqueeze(1).expand_as(target_canvas)
    branch_draw = torch.rand(
        target_canvas.shape,
        device=target_canvas.device,
        generator=generator,
    )
    eligible = torch.ones_like(target_canvas, dtype=torch.bool)
    if canvas_noise_mask is not None:
        if canvas_noise_mask.shape != target_canvas.shape:
            raise ValueError("canvas_noise_mask must match target_canvas shape")
        eligible &= canvas_noise_mask.to(device=target_canvas.device, dtype=torch.bool)
    # Defense in depth: [BOS] is fixed even if a caller supplies an overly broad
    # mask. Production also excludes it through DiffusionBatch.canvas_loss_mask.
    eligible &= target_canvas.ne(BOS_ID)
    corrupted_positions = (branch_draw < probabilities) & eligible

    if process == "uniform":
        replacement = sample_uniform_noise(
            target_canvas.shape,
            vocab_size=vocab_size,
            device=target_canvas.device,
            generator=generator,
        )
        noised_canvas = torch.where(corrupted_positions, replacement, target_canvas)
        position_weights = torch.ones_like(target_canvas, dtype=torch.float32)
    else:
        noised_canvas = torch.where(
            corrupted_positions,
            torch.full_like(target_canvas, mask_token_id),
            target_canvas,
        )
        position_weights = inverse_t_weights(sampled_t, target_canvas.shape[1])

    return CorruptionOutput(
        input_token_ids=input_token_ids,
        noised_canvas=noised_canvas,
        corrupted_positions=corrupted_positions,
        changed_positions=noised_canvas.ne(target_canvas),
        t=sampled_t,
        position_weights=position_weights,
    )


def sample_uniform_noise(
    shape: torch.Size | tuple[int, ...],
    *,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw from the uniform diffusion noise support.

    Random replacement states are exactly ``[PAD]``, ``[DELIMITER]``, and all
    content IDs. ``[MASK]``, ``[END]``, ``[WIN]``, ``[LOSS]``, ``[BOS]``, and
    ``[EOS]`` remain meaningful targets/boundaries and are never injected as
    random noise.
    """

    support_size = uniform_noise_support_size(vocab_size)
    draws = torch.randint(
        0,
        support_size,
        shape,
        device=device,
        generator=generator,
    )
    content_ids = CONTENT_TOKEN_OFFSET + (draws - 2)
    return torch.where(
        draws.eq(0),
        torch.full_like(draws, PAD_ID),
        torch.where(draws.eq(1), torch.full_like(draws, DELIMITER_ID), content_ids),
    )


def uniform_noise_support_size(vocab_size: int) -> int:
    """Return the live ``[PAD]`` + ``[DELIMITER]`` + content support width."""

    if vocab_size < CONTENT_TOKEN_OFFSET:
        raise ValueError("vocab_size must include every reserved special token")
    return 2 + vocab_size - CONTENT_TOKEN_OFFSET


def inverse_t_weights(t: torch.Tensor, canvas_len: int) -> torch.Tensor:
    """Return absorbing diffusion's inverse-time per-position weight."""

    return (1.0 / t.clamp_min(MIN_T)).unsqueeze(1).expand(-1, canvas_len)


def _validate_arguments(
    process: str,
    schedule: DiffusionScheduleConfig,
    vocab_size: int,
    mask_token_id: int,
) -> None:
    if process not in SUPPORTED_PROCESSES:
        raise ValueError(f"unsupported diffusion process: {process!r}")
    if schedule.name != "linear":
        raise ValueError(f"unsupported diffusion schedule: {schedule.name}")
    if schedule.t_distribution not in {"uniform", "power"}:
        raise ValueError(f"unsupported t distribution: {schedule.t_distribution}")
    if mask_token_id != 0:
        raise ValueError("uniform diffusion requires [MASK] to be vocabulary id zero")
    if vocab_size < CONTENT_TOKEN_OFFSET:
        raise ValueError("vocab_size must include every reserved special token")


def _resolve_t(
    target_canvas: torch.Tensor,
    schedule: DiffusionScheduleConfig,
    *,
    generator: torch.Generator | None,
    t: torch.Tensor | float | None,
) -> torch.Tensor:
    """Resolve one global noise level per example.

    Exact-``t=1`` oversampling applies only to ordinary training calls where
    ``t`` is omitted. Explicit diagnostic and validation values are used
    exactly, including ``t=0``.
    """

    batch_size = target_canvas.shape[0]
    device = target_canvas.device
    if t is None:
        unit_draw = torch.rand(batch_size, device=device, generator=generator)
        if schedule.t_distribution == "power":
            unit_draw = unit_draw.pow(1.0 / schedule.t_distribution_power)
        sampled = schedule.min + unit_draw * (schedule.max - schedule.min)
        if schedule.t_one_fraction > 0.0:
            oversample_draw = torch.rand(batch_size, device=device, generator=generator)
            sampled = torch.where(
                oversample_draw < schedule.t_one_fraction,
                torch.ones_like(sampled),
                sampled,
            )
    elif isinstance(t, torch.Tensor):
        sampled = t.to(device=device, dtype=torch.float32)
        if sampled.ndim == 0:
            sampled = sampled.expand(batch_size)
    else:
        sampled = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)

    if sampled.shape != (batch_size,):
        raise ValueError(f"t must be scalar or shape ({batch_size},), got {tuple(sampled.shape)}")
    if not bool(((sampled >= 0.0) & (sampled <= 1.0)).all()):
        raise ValueError("t must be in [0, 1]")
    return sampled
