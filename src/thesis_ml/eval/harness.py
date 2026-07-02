"""Held-out build-order evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import DatasetExample
from thesis_ml.eval.buildorder import BuildOrderEvent, extract_build_order, extract_build_order_from_parquet
from thesis_ml.eval.metrics import BuildOrderMetrics, aggregate_metrics, compare_build_orders
from thesis_ml.inference.decode import decode_canvas
from thesis_ml.inference.sampler import sample_canvas
from thesis_ml.inference.timing import TimedTimestep, attach_absolute_times
from thesis_ml.vocab.content_vocab import ContentVocabulary


@dataclass(frozen=True)
class EvaluationExampleResult:
    predicted_events: tuple[BuildOrderEvent, ...]
    ground_truth_events: tuple[BuildOrderEvent, ...]
    metrics: BuildOrderMetrics
    prediction_valid: bool
    dropped_final_timestep: bool


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
) -> EvaluationExampleResult:
    batch = collate_diffusion_examples([example])
    sampled = sample_canvas(model, batch, config, device=device)
    predicted = decode_canvas(sampled.canvas[0].tolist(), vocabulary)

    predicted_timesteps = _timed(predicted.timesteps, example, config)
    predicted_events = extract_build_order(predicted_timesteps, drop_final_timestep=False) if predicted.validation.valid else ()
    truth_events = _ground_truth_events(example, vocabulary, config)
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
