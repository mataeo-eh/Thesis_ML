"""Process-compatible entropy-bounded clean-state sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.backbone import FrozenInputKV
from thesis_ml.model.embedding import InputFeatures
from thesis_ml.model.model import (
    canvas_self_conditioning_from_logits,
    validate_checkpoint_compatibility,
)
from thesis_ml.train.corruption import corrupt_batch, sample_uniform_noise
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
    # --- per-step forward telemetry ------------------------------------------
    # Wall time of THIS step's single model forward call, in seconds (the unit
    # every other timing figure in this repo uses, e.g. `step_wall_seconds` in
    # train/loop.py). Measured with CUDA events on GPU so an async launch cannot
    # be mistaken for a fast forward; see `_ForwardTimer`.
    forward_wall_seconds: float = 0.0
    # True when this step reused a previously computed `FrozenInputKV` instead
    # of re-running the input region through all L backbone blocks. Step 1 of a
    # frozen-KV run is False (it is the step that BUILDS the cache), every later
    # step is True, and every step of a toggle-off run is False. Comparing the
    # `forward_wall_seconds` of the False step against the True ones is what
    # makes the frozen-KV payoff observable rather than merely asserted.
    used_cached_input_kv: bool = False


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


def _frozen_input_kv_enabled(model: nn.Module, input_len: int) -> bool:
    """Report whether this model can hand back and accept a frozen input-KV cache.

    The backbone raises `ValueError` when a cache is requested or supplied while
    the `frozen_input_kv` ablation is off or the input region is empty, and it
    never silently degrades. This predicate reproduces that exact condition
    (`self.frozen_input_kv and input_len is not None and input_len > 0` in
    `BidirectionalTransformer.forward`) so the sampler can only ever ask for a
    cache on a model that will actually produce one.

    The setting is read off the MODEL rather than off `ProjectConfig`, because
    the model is what decides whether the call raises: a model built by some
    other code path, or one of the lightweight stand-in models used in the test
    suite, may not match the config the sampler was handed. Anything without a
    backbone (every stand-in model) reports False and takes the untouched
    baseline path.

    Parameters:
        model: the denoiser being sampled from.
        input_len: padded width of the input region, i.e.
            `input_token_ids.shape[1]`.

    Returns:
        True only when a `FrozenInputKV` may legally be requested/reused.

    Calls: nothing; reads attributes with `getattr` defaults.
    """

    backbone = getattr(model, "backbone", None)
    return bool(getattr(backbone, "frozen_input_kv", False)) and input_len > 0


def _input_lengths_kwargs(
    model: nn.Module,
    batch: DiffusionBatch,
    device: torch.device,
) -> dict[str, object]:
    """Build the optional `input_lengths=` kwarg for a model forward call.

    `SC2StrategyDiffusionModel.forward` documents two equally valid ways of
    obtaining the per-row count of real input tokens: a caller that already holds
    `batch.input_lengths` passes it (the direct, cheapest route), and a caller
    that does not gets it derived from the input attention mask as
    `attention_mask[:, :input_len].sum(dim=1)`. The two are identically equal by
    construction, because the collater sets exactly `input_lengths` mask entries
    True per row -- so this choice is a micro-optimization and can never change a
    single output value.

    Two conditions must BOTH hold before the kwarg is passed:

    1. The model declares that it consumes it. Only the `per_segment_positions`
       ablation reads `input_lengths`; every other model ignores it. Reading the
       flag off the MODEL (not off `ProjectConfig`) matches the rule used by
       `_frozen_input_kv_enabled` -- the model is what the call actually reaches,
       and the sampler is also invoked with lightweight stand-in models whose
       `forward` signatures accept only the historical arguments.
    2. The batch actually carries it. `sample_canvas`/`denoise_canvas_once` are
       also called with duck-typed batch objects (diagnostics, tests) that expose
       only the tensors the sampler itself reads.

    When either fails the kwarg is omitted and the model's documented mask-derived
    fallback produces the same value.

    Parameters:
        model: the denoiser being sampled from.
        batch: the batch being sampled, or any object exposing its tensors.
        device: active device; `input_lengths` is moved onto it because a
            `DiffusionBatch` always arrives on CPU and building CUDA position ids
            from a CPU length tensor would fail.

    Returns:
        Either `{"input_lengths": <tensor on device>}` or an empty dict.

    Calls: nothing; reads attributes with `getattr` defaults.
    """

    if not getattr(model, "per_segment_positions", False):
        return {}
    input_lengths = getattr(batch, "input_lengths", None)
    if not isinstance(input_lengths, torch.Tensor):
        return {}
    return {"input_lengths": input_lengths.to(device)}


class _ForwardTimer:
    """Context manager measuring one model forward call's wall time, in seconds.

    Follows the timing convention `train/loop.py` already established: CUDA work
    is launched asynchronously, so a bare `perf_counter` around a GPU forward
    would time only the kernel launches and would report an entirely fictional
    speedup. On CUDA this brackets the call with `torch.cuda.Event`s and reads
    them back after synchronizing; on CPU there is no async queue, so
    `perf_counter` around the call is already exact.

    Synchronizing on the end event here is affordable (the training loop instead
    defers its read by one step) because sampling is inherently serialized
    already: the statements immediately after each forward pull scalars to the
    host via `.tolist()` / `bool(...)` / `.item()`, which block on exactly the
    same work.

    Parameters (constructor):
        device: the device the forward runs on; selects the measurement method.

    Attributes:
        seconds: elapsed forward time. Only meaningful after the `with` block
            has exited; it is 0.0 before that.

    Calls: `torch.cuda.Event`, `time.perf_counter`.
    """

    def __init__(self, device: torch.device) -> None:
        self._use_cuda_events = device.type == "cuda"
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._start_perf_counter = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "_ForwardTimer":
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_perf_counter = time.perf_counter()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        if self._use_cuda_events:
            self._end_event.record()
            self._end_event.synchronize()
            # CUDA events report MILLISECONDS; every timing figure in this repo
            # is in seconds, so convert here rather than at each read site.
            self.seconds = self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            self.seconds = time.perf_counter() - self._start_perf_counter


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
        canvas_noise_mask=eligible,
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
    # This endpoint performs exactly ONE forward, so there is no second step to
    # amortize a frozen input-KV cache over; building one would be pure cost.
    with _ForwardTimer(active_device) as forward_timer:
        output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas,
            input_attention_mask=batch.input_attention_mask.to(active_device),
            canvas_attention_mask=batch.canvas_attention_mask.to(active_device),
            input_features=_move_input_features(batch.input_features, active_device),
            canvas_self_conditioning=None if config.model.self_conditioning else None,
            **_input_lengths_kwargs(model, batch, active_device),
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
            forward_wall_seconds=forward_timer.seconds,
            used_cached_input_kv=False,
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
    # Built once, alongside the other input-region tensors, because the input
    # region is identical on every one of the forward calls below.
    input_lengths_kwargs = _input_lengths_kwargs(model, batch, active_device)
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

    # --- frozen input-KV reuse -----------------------------------------------
    # The input region is fixed for the whole of sampling: `input_token_ids`,
    # `input_attention_mask`, `input_features` and `input_lengths` are all bound
    # once above and no statement in the loop below rebinds or mutates any of
    # them -- only `canvas` changes between steps. That is exactly the precondition
    # the frozen-KV cache requires, so the input region's per-layer keys/values
    # can be computed on the first step and reused verbatim by every later step.
    #
    # The cache is built by the FIRST denoising step rather than by an extra
    # pre-loop forward on purpose: a pre-loop forward would be a whole additional
    # full-length model call, which both wastes the work this optimization exists
    # to avoid and breaks the subpackage contract that normal sampling makes
    # exactly one model call per step. Step 1 therefore asks for the cache back,
    # and steps 2..N hand it in.
    frozen_input_kv_enabled = _frozen_input_kv_enabled(model, input_token_ids.shape[1])
    cached_input_kv: FrozenInputKV | None = None

    for step_index in range(config.sampler.max_steps):
        if bool(done_rows.all()):
            break
        active_rows = ~done_rows
        temperature = sampler_temperature(config, step_index)
        # Kept as a kwargs dict so that with the ablation off NOTHING new is
        # passed and the call is the historical one, argument for argument.
        frozen_input_kv_kwargs: dict[str, object] = {}
        if frozen_input_kv_enabled:
            if cached_input_kv is None:
                frozen_input_kv_kwargs["return_cached_input_kv"] = True
            else:
                frozen_input_kv_kwargs["cached_input_kv"] = cached_input_kv
        step_used_cached_input_kv = "cached_input_kv" in frozen_input_kv_kwargs
        with _ForwardTimer(active_device) as forward_timer:
            output = model(
                input_token_ids=input_token_ids,
                canvas_token_ids=canvas,
                input_attention_mask=input_attention_mask,
                canvas_attention_mask=canvas_attention_mask,
                input_features=input_features,
                canvas_self_conditioning=canvas_self_conditioning
                if config.model.self_conditioning
                else None,
                **input_lengths_kwargs,
                **frozen_input_kv_kwargs,
            )
        if frozen_input_kv_enabled and cached_input_kv is None:
            # Capture what step 1 produced. Left None on the (contract-forbidden)
            # chance the model returns nothing, in which case the next step simply
            # asks again instead of feeding the model a bad cache.
            cached_input_kv = getattr(output, "cached_input_kv", None)
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
                forward_wall_seconds=forward_timer.seconds,
                used_cached_input_kv=step_used_cached_input_kv,
            )
        )

    for row, reason in enumerate(stop_reasons):
        if reason is None:
            stop_reasons[row] = "max_steps"
    if trace:
        trace[-1] = replace(trace[-1], stop_reasons=tuple(stop_reasons))

    final_canvas_logits = None
    if return_final_logits:
        # The diagnostics-only estimate pass conditions on the same, still
        # unchanged input region, so it reuses the cache too when one exists.
        # `cached_input_kv` is None when the loop never ran (every row was
        # already done), in which case this falls back to a plain full forward.
        final_frozen_input_kv_kwargs: dict[str, object] = {}
        if frozen_input_kv_enabled and cached_input_kv is not None:
            final_frozen_input_kv_kwargs["cached_input_kv"] = cached_input_kv
        final_output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas,
            input_attention_mask=input_attention_mask,
            canvas_attention_mask=canvas_attention_mask,
            input_features=input_features,
            canvas_self_conditioning=canvas_self_conditioning
            if config.model.self_conditioning
            else None,
            **input_lengths_kwargs,
            **final_frozen_input_kv_kwargs,
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
        sampled[eligible] = sample_uniform_noise(
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
