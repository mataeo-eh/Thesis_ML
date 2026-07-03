"""Confidence-based iterative denoising sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.model import canvas_self_conditioning_from_logits
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
    """Denoise an all-[MASK] canvas by monotonic confidence-based commits.

    Each step runs the model over the current (partly masked) canvas, then commits
    the most-confident still-masked positions. Committed positions are never
    remasked, so the canvas fills in monotonically until every position is set.

    Fine-tune constraint (`config.sampler.outcome_last`): when True, canvas
    position 0 holds the win/loss outcome token and is forced to denoise LAST. It
    is excluded from the commit candidates until every other position `[1:]` is
    committed, then force-committed with the model's prediction (ignoring the
    confidence/entropy gates so sampling cannot stall). When the flag is False the
    behavior is byte-for-byte identical to the pre-training sampler.
    """

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
    canvas_self_conditioning: torch.Tensor | None = None
    trace: list[SamplerStep] = []

    for step_index in range(config.sampler.max_steps):
        temperature = sampler_temperature(config, step_index)
        forward_kwargs = {
            "input_token_ids": input_token_ids,
            "canvas_token_ids": canvas,
            "input_attention_mask": input_attention_mask,
            "canvas_attention_mask": torch.ones_like(canvas, dtype=torch.bool),
            "input_features": batch.input_features,
        }
        if config.model.self_conditioning:
            forward_kwargs["canvas_self_conditioning"] = canvas_self_conditioning
        output = model(**forward_kwargs)
        raw_canvas_logits = output.logits[:, input_token_ids.shape[1] :, :]
        canvas_logits = raw_canvas_logits / temperature
        probs = torch.softmax(canvas_logits, dim=-1)
        if config.model.self_conditioning:
            canvas_self_conditioning = canvas_self_conditioning_from_logits(raw_canvas_logits)
        confidence, predicted = probs.max(dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        # Candidate set for this step: every position that is still masked.
        # `selectable` is what we allow `_select_commits` to consider. When the
        # fine-tune constraint `config.sampler.outcome_last` is OFF this is
        # exactly `~committed`, so the pre-training sampling path is unchanged.
        selectable = ~committed
        if config.sampler.outcome_last:
            # Outcome-last constraint (Worker 2): canvas position 0 holds the
            # [WIN]/[LOSS] outcome token and must denoise LAST. Remove position 0
            # from the candidate set for any row whose remaining positions `[1:]`
            # are not yet fully committed. `rest_all_committed` is computed from
            # `committed` BEFORE this step's commits, so on the step that finishes
            # `[1:]`, position 0 is still not selectable here (it is force-committed
            # below on that same step instead).
            rest_all_committed = committed[:, 1:].all(dim=1)  # shape (batch,)
            selectable = selectable.clone()
            selectable[:, 0] = selectable[:, 0] & rest_all_committed

        commit_mask = _select_commits(
            entropy=entropy,
            confidence=confidence,
            masked=selectable,
            entropy_bound=config.sampler.entropy_bound,
            confidence_threshold=config.sampler.confidence_threshold,
            min_commit_per_step=config.sampler.min_commit_per_step,
        )

        if config.sampler.outcome_last:
            # Guarantee the outcome token commits. Once a row's `[1:]` positions
            # are ALL committed (looking at this step's commits via
            # `committed | commit_mask`), force-commit position 0 using the
            # model's own prediction, bypassing the confidence/entropy gates in
            # `_select_commits` so sampling can never stall on the final token.
            # Because `_select_commits` above never selects position 0 (it was
            # cleared from `selectable`), this force path is the only way the
            # outcome token is ever committed under the flag, so it is always the
            # last position to transition from masked -> committed. This also
            # covers the max_steps edge case: if `[1:]` finishes on the final
            # allowed step, the outcome token still commits on that same step.
            rest_done_after_step = (committed | commit_mask)[:, 1:].all(dim=1)
            outcome_still_masked = ~(committed | commit_mask)[:, 0]
            force_outcome = rest_done_after_step & outcome_still_masked  # (batch,)
            commit_mask[:, 0] = commit_mask[:, 0] | force_outcome

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
