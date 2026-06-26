"""Confidence-based iterative denoising sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.vocab.special_tokens import MASK_ID


@dataclass(frozen=True)
class SamplerStep:
    step: int
    temperature: float
    committed_this_step: torch.Tensor
    committed_mask: torch.Tensor
    canvas: torch.Tensor


@dataclass(frozen=True)
class SamplerOutput:
    canvas: torch.Tensor
    input_token_ids: torch.Tensor
    initial_input_token_ids: torch.Tensor
    committed_mask: torch.Tensor
    steps: int
    trace: list[SamplerStep]


@torch.no_grad()
def sample_canvas(
    model: nn.Module,
    batch: DiffusionBatch,
    config: ProjectConfig,
    *,
    device: torch.device | str = "cpu",
) -> SamplerOutput:
    """Denoise an all-[MASK] canvas by monotonic confidence-based commits."""

    active_device = torch.device(device)
    model = model.to(active_device)
    model.eval()
    input_token_ids = batch.input_token_ids.to(active_device)
    input_attention_mask = batch.input_attention_mask.to(active_device)
    initial_input = input_token_ids.clone()

    batch_size = input_token_ids.shape[0]
    canvas = torch.full(
        (batch_size, config.data.canvas_budget_tokens),
        MASK_ID,
        dtype=torch.long,
        device=active_device,
    )
    committed = torch.zeros_like(canvas, dtype=torch.bool)
    trace: list[SamplerStep] = []

    for step_index in range(config.sampler.max_steps):
        temperature = sampler_temperature(config, step_index)
        output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas,
            input_attention_mask=input_attention_mask,
            canvas_attention_mask=torch.ones_like(canvas, dtype=torch.bool),
            input_records=batch.input_records,
        )
        canvas_logits = output.logits[:, input_token_ids.shape[1] :, :] / temperature
        probs = torch.softmax(canvas_logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        commit_mask = _select_commits(
            entropy=entropy,
            confidence=confidence,
            masked=~committed,
            entropy_bound=config.sampler.entropy_bound,
            confidence_threshold=config.sampler.confidence_threshold,
            min_commit_per_step=config.sampler.min_commit_per_step,
        )
        canvas = torch.where(commit_mask, predicted, canvas)
        committed = committed | commit_mask
        trace.append(
            SamplerStep(
                step=step_index + 1,
                temperature=temperature,
                committed_this_step=commit_mask.detach().cpu(),
                committed_mask=committed.detach().cpu(),
                canvas=canvas.detach().cpu(),
            )
        )
        if committed.all():
            break

    return SamplerOutput(
        canvas=canvas.detach().cpu(),
        input_token_ids=input_token_ids.detach().cpu(),
        initial_input_token_ids=initial_input.detach().cpu(),
        committed_mask=committed.detach().cpu(),
        steps=len(trace),
        trace=trace,
    )


def sampler_temperature(config: ProjectConfig, step_index: int) -> float:
    max_steps = max(1, config.sampler.max_steps)
    if max_steps == 1:
        return float(config.sampler.temperature.end)
    progress = min(1.0, step_index / float(max_steps - 1))
    start = config.sampler.temperature.start
    end = config.sampler.temperature.end
    return float(start + (end - start) * progress)


def load_sampling_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, device: torch.device | str = "cpu") -> nn.Module:
    """Load EMA weights for sampling when present, falling back to raw weights."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint.get("ema_model", checkpoint["model"])
    model.load_state_dict(state)
    return model


def _select_commits(
    *,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    masked: torch.Tensor,
    entropy_bound: float,
    confidence_threshold: float,
    min_commit_per_step: int,
) -> torch.Tensor:
    selected = torch.zeros_like(masked, dtype=torch.bool)
    for row in range(masked.shape[0]):
        candidates = torch.nonzero(masked[row], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        candidates = candidates[torch.argsort(entropy[row, candidates])]
        cumulative = 0.0
        committed_count = 0
        for index in candidates.tolist():
            if confidence[row, index].item() < confidence_threshold:
                continue
            next_entropy = float(entropy[row, index].item())
            within_budget = cumulative + next_entropy <= entropy_bound
            need_minimum = committed_count < min_commit_per_step
            if not within_budget and not need_minimum:
                break
            selected[row, index] = True
            cumulative += next_entropy
            committed_count += 1
    return selected
