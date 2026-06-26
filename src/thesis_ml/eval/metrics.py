"""Resolution-matched build-order metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from thesis_ml.eval.buildorder import BuildOrderEvent


@dataclass(frozen=True)
class BuildOrderMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted_count: int
    ground_truth_count: int
    per_entity_type: dict[str, "BuildOrderMetrics"]


def compare_build_orders(
    predicted: Sequence[BuildOrderEvent],
    ground_truth: Sequence[BuildOrderEvent],
    *,
    timing_tolerance_buckets: int,
) -> BuildOrderMetrics:
    matches = _count_matches(predicted, ground_truth, timing_tolerance_buckets=timing_tolerance_buckets)
    return _metrics_from_counts(matches, len(predicted), len(ground_truth), _per_type(predicted, ground_truth, timing_tolerance_buckets))


def aggregate_metrics(metrics: Sequence[BuildOrderMetrics]) -> BuildOrderMetrics:
    tp = sum(metric.true_positives for metric in metrics)
    predicted = sum(metric.predicted_count for metric in metrics)
    ground_truth = sum(metric.ground_truth_count for metric in metrics)
    return _metrics_from_counts(tp, predicted, ground_truth, {})


def _count_matches(
    predicted: Sequence[BuildOrderEvent],
    ground_truth: Sequence[BuildOrderEvent],
    *,
    timing_tolerance_buckets: int,
) -> int:
    unmatched_truth = set(range(len(ground_truth)))
    true_positives = 0
    for prediction in predicted:
        candidates = [
            index
            for index in unmatched_truth
            if ground_truth[index].entity_type == prediction.entity_type
            and abs(ground_truth[index].bucket - prediction.bucket) <= timing_tolerance_buckets
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(ground_truth[index].bucket - prediction.bucket))
        unmatched_truth.remove(best)
        true_positives += 1
    return true_positives


def _metrics_from_counts(
    true_positives: int,
    predicted_count: int,
    ground_truth_count: int,
    per_entity_type: dict[str, BuildOrderMetrics],
) -> BuildOrderMetrics:
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / ground_truth_count if ground_truth_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = true_positives / ground_truth_count if ground_truth_count else 0.0
    return BuildOrderMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        predicted_count=predicted_count,
        ground_truth_count=ground_truth_count,
        per_entity_type=per_entity_type,
    )


def _per_type(
    predicted: Sequence[BuildOrderEvent],
    ground_truth: Sequence[BuildOrderEvent],
    timing_tolerance_buckets: int,
) -> dict[str, BuildOrderMetrics]:
    names = sorted({event.entity_type for event in predicted} | {event.entity_type for event in ground_truth})
    per_type: dict[str, BuildOrderMetrics] = {}
    for name in names:
        pred_subset = [event for event in predicted if event.entity_type == name]
        truth_subset = [event for event in ground_truth if event.entity_type == name]
        matches = _count_matches(pred_subset, truth_subset, timing_tolerance_buckets=timing_tolerance_buckets)
        per_type[name] = _metrics_from_counts(matches, len(pred_subset), len(truth_subset), {})
    return per_type
