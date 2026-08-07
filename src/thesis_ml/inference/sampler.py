"""Process-compatible entropy-bounded clean-state sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.embedding import InputFeatures
from thesis_ml.model.model import (
    canvas_self_conditioning_from_logits,
    validate_checkpoint_compatibility,
)
from thesis_ml.train.corruption import corrupt_batch, sample_uniform_non_mask
from thesis_ml.vocab.special_tokens import MASK_ID


@dataclass(frozen=True)
class SamplerStep:
    step: int
    temperature: float
    accepted_mask: torch.Tensor
    unaccepted_mask: torch.Tensor
    renoised_mask: torch.Tensor
    entropy: torch.Tensor
    mean_entropy: torch.Tensor
    argmax_predictions: torch.Tensor
    argmax_stable: torch.Tensor
    done_rows: torch.Tensor
    stop_reasons: tuple[str | None, ...]
    canvas: torch.Tensor


@dataclass(frozen=True)
class SamplerOutput:
    canvas: torch.Tensor
    input_token_ids: torch.Tensor
    initial_input_token_ids: torch.Tensor
    accepted_mask: torch.Tensor
    done_rows: torch.Tensor
    stop_reasons: tuple[str, ...]
    steps: int
    trace: list[SamplerStep]
    final_canvas_logits: torch.Tensor | None = None
    revealed_mask: torch.Tensor | None = None


def _initial_canvas(
    batch: DiffusionBatch,
    config: ProjectConfig,
    *,
    noise_rate: float,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the selected process's prior/infill state and clamping masks."""

    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1]")
    target = batch.target_canvas.to(device)
    eligible = batch.canvas_loss_mask.to(device=device, dtype=torch.bool)
    if noise_rate == 1.0:
        if config.diffusion.process == "uniform":
            prior = _sample_uniform_at_positions(
                target,
                eligible,
                vocab_size=vocab_size,
                generator=generator,
            )
        else:
            prior = torch.full_like(target, MASK_ID)
        canvas = torch.where(eligible, prior, target)
        revealed = torch.zeros_like(eligible)
        mutable = eligible
        return canvas, revealed, mutable

    corruption = corrupt_batch(
        input_token_ids=batch.input_token_ids.to(device),
        target_canvas=target,
        process=config.diffusion.process,
        schedule=config.diffusion.schedule,
        vocab_size=vocab_size,
        generator=generator,
        t=noise_rate,
    )
    mutable = corruption.corrupted_positions & eligible
    revealed = eligible & ~corruption.corrupted_positions
    canvas = torch.where(eligible, corruption.noised_canvas, target)
    return canvas, revealed, mutable


@torch.no_grad()
def denoise_canvas_once(
    model: nn.Module,
    batch: DiffusionBatch,
    config: ProjectConfig,
    *,
    device: torch.device | str = "cpu",
    return_final_logits: bool = False,
    noise_rate: float = 1.0,
) -> SamplerOutput:
    """Run exactly one denoiser pass from the selected process's prior."""

    active_device = torch.device(device)
    model = model.to(active_device)
    model.eval()
    generator = _make_generator(active_device, config.pipeline.seed)
    input_token_ids = batch.input_token_ids.to(active_device)
    initial_input = input_token_ids.clone()
    vocab_size = int(getattr(model, "vocab_size"))
    canvas, revealed, mutable = _initial_canvas(
        batch,
        config,
        noise_rate=noise_rate,
        vocab_size=vocab_size,
        device=active_device,
        generator=generator,
    )
    output = model(
        input_token_ids=input_token_ids,
        canvas_token_ids=canvas,
        input_attention_mask=batch.input_attention_mask.to(active_device),
        canvas_attention_mask=batch.canvas_attention_mask.to(active_device),
        input_features=_move_input_features(batch.input_features, active_device),
        canvas_self_conditioning=None if config.model.self_conditioning else None,
    )
    canvas_logits = output.logits[:, input_token_ids.shape[1] :, :]
    probs = _allowed_probabilities(canvas_logits)
    predicted = probs.argmax(dim=-1)
    final_canvas = torch.where(mutable, predicted, canvas)
    entropy = _entropy(probs)
    mean_entropy = _row_mean(entropy, mutable)
    done_rows = torch.ones(input_token_ids.shape[0], dtype=torch.bool, device=active_device)
    accepted = mutable
    trace = [
        SamplerStep(
            step=1,
            temperature=1.0,
            accepted_mask=accepted.detach().cpu(),
            unaccepted_mask=torch.zeros_like(accepted).cpu(),
            renoised_mask=torch.zeros_like(accepted).cpu(),
            entropy=entropy.detach().cpu(),
            mean_entropy=mean_entropy.detach().cpu(),
            argmax_predictions=predicted.detach().cpu(),
            argmax_stable=torch.zeros_like(done_rows).cpu(),
            done_rows=done_rows.cpu(),
            stop_reasons=tuple("single_pass" for _ in range(input_token_ids.shape[0])),
            canvas=final_canvas.detach().cpu(),
        )
    ]
    return SamplerOutput(
        canvas=final_canvas.detach().cpu(),
        input_token_ids=input_token_ids.detach().cpu(),
        initial_input_token_ids=initial_input.detach().cpu(),
        accepted_mask=accepted.detach().cpu(),
        done_rows=done_rows.cpu(),
        stop_reasons=tuple("single_pass" for _ in range(input_token_ids.shape[0])),
        steps=1,
        trace=trace,
        final_canvas_logits=canvas_logits.detach().cpu() if return_final_logits else None,
        revealed_mask=revealed.detach().cpu(),
    )


@torch.no_grad()
def sample_canvas(
    model: nn.Module,
    batch: DiffusionBatch,
    config: ProjectConfig,
    *,
    device: torch.device | str = "cpu",
    return_final_logits: bool = False,
    noise_rate: float = 1.0,
) -> SamplerOutput:
    """Sample with nonmonotonic uniform EB or monotonic absorbing EB."""

    active_device = torch.device(device)
    model = model.to(active_device)
    model.eval()
    generator = _make_generator(active_device, config.pipeline.seed)
    input_token_ids = batch.input_token_ids.to(active_device)
    input_attention_mask = batch.input_attention_mask.to(active_device)
    canvas_attention_mask = batch.canvas_attention_mask.to(active_device)
    input_features = _move_input_features(batch.input_features, active_device)
    initial_input = input_token_ids.clone()
    batch_size = input_token_ids.shape[0]
    vocab_size = int(getattr(model, "vocab_size"))
    canvas, revealed, mutable = _initial_canvas(
        batch,
        config,
        noise_rate=noise_rate,
        vocab_size=vocab_size,
        device=active_device,
        generator=generator,
    )

    done_rows = ~mutable.any(dim=1)
    stop_reasons: list[str | None] = [
        "no_eligible" if bool(done_rows[row]) else None for row in range(batch_size)
    ]
    previous_argmax: torch.Tensor | None = None
    stable_counts = torch.zeros(batch_size, dtype=torch.long, device=active_device)
    canvas_self_conditioning: torch.Tensor | None = None
    accepted = torch.zeros_like(mutable)
    trace: list[SamplerStep] = []

    for step_index in range(config.sampler.max_steps):
        if bool(done_rows.all()):
            break
        active_rows = ~done_rows
        temperature = sampler_temperature(config, step_index)
        output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas,
            input_attention_mask=input_attention_mask,
            canvas_attention_mask=canvas_attention_mask,
            input_features=input_features,
            canvas_self_conditioning=canvas_self_conditioning
            if config.model.self_conditioning
            else None,
        )
        raw_canvas_logits = output.logits[:, input_token_ids.shape[1] :, :]
        probs = _allowed_probabilities(raw_canvas_logits / temperature)
        entropy = _entropy(probs)
        argmax_predictions = probs.argmax(dim=-1)

        same_as_previous = torch.zeros(batch_size, dtype=torch.bool, device=active_device)
        if previous_argmax is not None:
            same_as_previous = active_rows & (
                (argmax_predictions == previous_argmax) | ~mutable
            ).all(dim=1)
        stable_counts = torch.where(
            active_rows,
            torch.where(
                same_as_previous,
                stable_counts + 1,
                torch.ones_like(stable_counts),
            ),
            stable_counts,
        )
        argmax_stable = stable_counts >= config.sampler.stability_steps
        if previous_argmax is None:
            previous_argmax = argmax_predictions.detach()
        else:
            previous_argmax = torch.where(
                active_rows[:, None],
                argmax_predictions.detach(),
                previous_argmax,
            )

        if config.diffusion.process == "uniform":
            selectable = mutable & active_rows[:, None]
            candidates = _sample_categorical_at_positions(
                probs,
                selectable,
                fallback=canvas,
                generator=generator,
            )
            accepted = _entropy_bounded_acceptance(
                entropy,
                selectable,
                config.sampler.entropy_bound,
            )
            unaccepted = selectable & ~accepted
            fresh_noise = _sample_uniform_at_positions(
                canvas,
                selectable,
                vocab_size=vocab_size,
                generator=generator,
            )
            canvas = torch.where(
                accepted,
                candidates,
                torch.where(unaccepted, fresh_noise, canvas),
            )
            renoised = unaccepted
            mean_entropy = _row_mean(entropy, mutable)
            if config.sampler.adaptive_stop:
                newly_done = (
                    active_rows
                    & (mean_entropy < config.sampler.entropy_threshold)
                    & argmax_stable
                )
                for row in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
                    stop_reasons[row] = "adaptive_entropy_stability"
                done_rows = done_rows | newly_done
        else:
            selectable = mutable & canvas.eq(MASK_ID) & active_rows[:, None]
            candidates = _sample_categorical_at_positions(
                probs,
                selectable,
                fallback=canvas,
                generator=generator,
            )
            accepted = _entropy_bounded_acceptance(
                entropy,
                selectable,
                config.sampler.entropy_bound,
            )
            unaccepted = selectable & ~accepted
            renoised = torch.zeros_like(unaccepted)
            canvas = torch.where(accepted, candidates, canvas)
            mean_entropy = _row_mean(entropy, selectable)
            newly_done = active_rows & ~(mutable & canvas.eq(MASK_ID)).any(dim=1)
            for row in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
                stop_reasons[row] = "absorbing_complete"
            done_rows = done_rows | newly_done

        if config.model.self_conditioning:
            next_self_conditioning = canvas_self_conditioning_from_logits(
                raw_canvas_logits,
                model.embedding.token_embedding.weight,
                probabilities=probs,
            )
            if canvas_self_conditioning is None:
                canvas_self_conditioning = torch.zeros_like(next_self_conditioning)
            canvas_self_conditioning = torch.where(
                active_rows[:, None, None],
                next_self_conditioning,
                canvas_self_conditioning,
            )

        trace.append(
            SamplerStep(
                step=step_index + 1,
                temperature=temperature,
                accepted_mask=accepted.detach().cpu(),
                unaccepted_mask=unaccepted.detach().cpu(),
                renoised_mask=renoised.detach().cpu(),
                entropy=entropy.detach().cpu(),
                mean_entropy=mean_entropy.detach().cpu(),
                argmax_predictions=argmax_predictions.detach().cpu(),
                argmax_stable=argmax_stable.detach().cpu(),
                done_rows=done_rows.detach().cpu(),
                stop_reasons=tuple(stop_reasons),
                canvas=canvas.detach().cpu(),
            )
        )

    for row, reason in enumerate(stop_reasons):
        if reason is None:
            stop_reasons[row] = "max_steps"
    if trace:
        trace[-1] = replace(trace[-1], stop_reasons=tuple(stop_reasons))

    final_canvas_logits = None
    if return_final_logits:
        final_output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas,
            input_attention_mask=input_attention_mask,
            canvas_attention_mask=canvas_attention_mask,
            input_features=input_features,
            canvas_self_conditioning=canvas_self_conditioning
            if config.model.self_conditioning
            else None,
        )
        final_canvas_logits = final_output.logits[:, input_token_ids.shape[1] :, :].detach().cpu()

    return SamplerOutput(
        canvas=canvas.detach().cpu(),
        input_token_ids=input_token_ids.detach().cpu(),
        initial_input_token_ids=initial_input.detach().cpu(),
        accepted_mask=accepted.detach().cpu(),
        done_rows=done_rows.detach().cpu(),
        stop_reasons=tuple(str(reason) for reason in stop_reasons),
        steps=len(trace),
        trace=trace,
        final_canvas_logits=final_canvas_logits,
        revealed_mask=revealed.detach().cpu(),
    )


def sampler_temperature(config: ProjectConfig, step_index: int) -> float:
    max_steps = max(1, config.sampler.max_steps)
    if max_steps == 1:
        return float(config.sampler.temperature.end)
    progress = min(1.0, step_index / float(max_steps - 1))
    shaped_progress = progress ** config.sampler.temperature.exponent
    start = config.sampler.temperature.start
    end = config.sampler.temperature.end
    return float(start + (end - start) * shaped_progress)


def load_sampling_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load compatible EMA weights for sampling."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(checkpoint, model, str(path))
    expected = getattr(model, "feature_statistics_identity", None)
    observed = checkpoint.get("feature_statistics_identity")
    if not isinstance(observed, str) or observed != expected:
        raise ValueError(f"checkpoint {path} feature statistics are missing or incompatible")
    if "ema_model" not in checkpoint:
        raise ValueError(f"checkpoint {path} has no EMA weights and is incompatible")
    model.load_state_dict(checkpoint["ema_model"])
    return model


def _allowed_probabilities(logits: torch.Tensor) -> torch.Tensor:
    allowed_logits = logits.float().clone()
    allowed_logits[..., MASK_ID] = -torch.inf
    return torch.softmax(allowed_logits, dim=-1)


def _sample_categorical_at_positions(
    probabilities: torch.Tensor,
    eligible: torch.Tensor,
    *,
    fallback: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample candidates only at mutable positions, preserving all others."""

    sampled = fallback.clone()
    if bool(eligible.any()):
        sampled[eligible] = torch.multinomial(
            probabilities[eligible],
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
    return sampled


def _sample_uniform_at_positions(
    fallback: torch.Tensor,
    eligible: torch.Tensor,
    *,
    vocab_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw uniform non-mask states only at mutable positions."""

    sampled = fallback.clone()
    count = int(eligible.sum().item())
    if count:
        sampled[eligible] = sample_uniform_non_mask(
            (count,),
            vocab_size=vocab_size,
            device=fallback.device,
            generator=generator,
        )
    return sampled


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)


def _entropy_bounded_acceptance(
    entropy: torch.Tensor,
    eligible: torch.Tensor,
    entropy_bound: float,
) -> torch.Tensor:
    """Accept the exact prefix where prior cumulative entropy is within gamma."""

    accepted = torch.zeros_like(eligible, dtype=torch.bool)
    for row in range(eligible.shape[0]):
        indices = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        order = torch.argsort(entropy[row, indices], stable=True)
        sorted_indices = indices[order]
        sorted_entropy = entropy[row, sorted_indices]
        prefix = torch.cumsum(sorted_entropy, dim=0) - sorted_entropy
        accepted[row, sorted_indices[prefix <= entropy_bound]] = True
    return accepted


def _row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=1)
    totals = (values * mask.to(values.dtype)).sum(dim=1)
    return torch.where(counts > 0, totals / counts.clamp_min(1), torch.zeros_like(totals))


def _move_input_features(
    features: InputFeatures | None,
    device: torch.device,
) -> InputFeatures | None:
    if features is None:
        return None
    return InputFeatures(
        continuous_values=features.continuous_values.to(device),
        continuous_validity=features.continuous_validity.to(device),
        categorical_values=features.categorical_values.to(device),
        allegiance_values=features.allegiance_values.to(device),
        feature_mask=features.feature_mask.to(device),
    )


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
