"""Held-out build-order evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import DatasetExample
from thesis_ml.eval.buildorder import BuildOrderEvent, extract_build_order, extract_build_order_from_parquet
from thesis_ml.eval.metrics import BuildOrderMetrics, aggregate_metrics, compare_build_orders
from thesis_ml.inference.decode import decode_canvas
from thesis_ml.inference.sampler import denoise_canvas_once, sample_canvas
from thesis_ml.inference.timing import TimedTimestep, attach_absolute_times
from thesis_ml.vocab.content_vocab import ContentVocabulary


@dataclass(frozen=True)
class EvaluationExampleResult:
    predicted_events: tuple[BuildOrderEvent, ...]
    ground_truth_events: tuple[BuildOrderEvent, ...]
    metrics: BuildOrderMetrics
    prediction_valid: bool
    dropped_final_timestep: bool
    # Per-timestep entity-type count grids at model (sampling-interval)
    # resolution, exposed so read-only diagnostics (e.g. the viz module) can
    # render the intermediates this harness already decodes without rebuilding
    # any sampling/decode logic. Both are decoded canvases (predicted vs the
    # clamped ground-truth target canvas), so they are resolution-matched to
    # each other. Empty when the predicted canvas failed grammar validation.
    predicted_counts: tuple[TimedTimestep, ...] = ()
    ground_truth_counts: tuple[TimedTimestep, ...] = ()
    predicted_canvas: tuple[int, ...] = ()
    ground_truth_canvas: tuple[int, ...] = ()
    final_canvas_logits: torch.Tensor | None = field(default=None, repr=False, compare=False)
    # Per-position provenance of ``predicted_canvas`` (aligned 1:1): True where the
    # position was REVEALED as ground truth (an infill start, see ``mask_rate``)
    # and False where the MODEL predicted it. Empty when not tracked; under the
    # default ``mask_rate=1.0`` every position is model-predicted (all False).
    predicted_canvas_revealed_mask: tuple[bool, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    split: str
    weight_source: str
    metrics: BuildOrderMetrics
    examples: list[EvaluationExampleResult]

    def to_metrics_dict(self) -> dict[str, float | int | str]:
        return {
            "split": self.split,
            "weight_source": self.weight_source,
            "accuracy": self.metrics.accuracy,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1": self.metrics.f1,
            "true_positives": self.metrics.true_positives,
            "predicted_count": self.metrics.predicted_count,
            "ground_truth_count": self.metrics.ground_truth_count,
        }


def evaluate_examples(
    *,
    model: nn.Module,
    examples: Sequence[DatasetExample],
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    device: torch.device | str = "cpu",
    weight_source: str = "ema",
) -> EvaluationReport:
    results: list[EvaluationExampleResult] = []
    for example in examples:
        result = evaluate_example(
            model=model,
            example=example,
            vocabulary=vocabulary,
            config=config,
            device=device,
        )
        results.append(result)
    return EvaluationReport(
        split=config.eval.heldout_split,
        weight_source=weight_source,
        metrics=aggregate_metrics([result.metrics for result in results]),
        examples=results,
    )


def evaluate_example(
    *,
    model: nn.Module,
    example: DatasetExample,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    device: torch.device | str = "cpu",
    include_canvas_logits: bool = False,
    bypass_sampler: bool = False,
    mask_rate: float = 1.0,
) -> EvaluationExampleResult:
    batch = collate_diffusion_examples([example], debut_mode=config.data.debut_mode)
    prediction_fn = denoise_canvas_once if bypass_sampler else sample_canvas
    sampled = prediction_fn(
        model,
        batch,
        config,
        device=device,
        return_final_logits=include_canvas_logits,
        mask_rate=mask_rate,
    )
    predicted = decode_canvas(sampled.canvas[0].tolist(), vocabulary)

    predicted_timesteps = _timed(predicted.timesteps, example, config)
    predicted_events = extract_build_order(predicted_timesteps, drop_final_timestep=False) if predicted.validation.valid else ()
    truth_events = _ground_truth_events(example, vocabulary, config)
    # Decode the clamped ground-truth target canvas to a per-timestep count grid
    # at the same model resolution as the prediction. This reuses the identical
    # decode_canvas + attach_absolute_times path already used for the prediction,
    # so the two grids are resolution-matched and share the §7 whole-timestep
    # truncation handling (decode_canvas never emits a partial final timestep).
    truth_decoded = decode_canvas(example.target_canvas.tolist(), vocabulary)
    ground_truth_timesteps = _timed(truth_decoded.timesteps, example, config)
    metrics = compare_build_orders(
        predicted_events,
        truth_events,
        timing_tolerance_buckets=config.eval.timing_tolerance_buckets,
    )
    return EvaluationExampleResult(
        predicted_events=predicted_events,
        ground_truth_events=truth_events,
        metrics=metrics,
        prediction_valid=predicted.validation.valid,
        dropped_final_timestep=False,
        predicted_counts=tuple(predicted_timesteps),
        ground_truth_counts=tuple(ground_truth_timesteps),
        predicted_canvas=tuple(int(token_id) for token_id in sampled.canvas[0].tolist()),
        ground_truth_canvas=tuple(int(token_id) for token_id in example.target_canvas.tolist()),
        final_canvas_logits=(
            sampled.final_canvas_logits[0] if sampled.final_canvas_logits is not None else None
        ),
        predicted_canvas_revealed_mask=(
            tuple(bool(flag) for flag in sampled.revealed_mask[0].tolist())
            if sampled.revealed_mask is not None
            else ()
        ),
    )


def _timed(
    timesteps: Sequence[dict[str, int]],
    example: DatasetExample,
    config: ProjectConfig,
) -> list[TimedTimestep]:
    last_clock = _last_input_clock(example)
    return attach_absolute_times(
        timesteps,
        last_input_clock=last_clock,
        sampling_interval_s=config.data.sampling_interval_s,
    )


def _last_input_clock(example: DatasetExample) -> float:
    clocks = [
        float(record.timestamp_seconds)
        for record in example.input_records
        if getattr(record, "timestamp_seconds", None) is not None
    ]
    return max(clocks) if clocks else float(example.window_start)


def _ground_truth_events(
    example: DatasetExample,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
) -> tuple[BuildOrderEvent, ...]:
    if example.replay_path is not None:
        return extract_build_order_from_parquet(
            example.replay_path,
            config,
            vocabulary,
            perspective_player=example.perspective_player,
            start=example.window_start,
            drop_final_timestep=False,
        )
    ground_truth = decode_canvas(example.target_canvas.tolist(), vocabulary)
    truth_timesteps = _timed(ground_truth.timesteps, example, config)
    return extract_build_order(truth_timesteps, drop_final_timestep=False)
