"""Fine-tuning (debut-mode) evaluation and report writer.

Role in the system
------------------
This module is the fine-tuning counterpart to ``eval/harness.py``. Where the
pre-training harness reports a single build-order F1, the fine-tuning
evaluation must characterise the *debut* head along several independent axes and
emit a nested JSON report. The final ``finetune_report.json`` (assembled and
written by the training pipeline) carries TWO sibling sections with identical
metric keys:

    {
      "memorized": { ...section metrics... },
      "test":      { ...section metrics... }
    }

The "memorized" section scores the exact replays the model was fine-tuned on
(a diagnostic of how well the outcome/debut targets were absorbed); the "test"
section scores held-out replays. Both are produced by calling
``build_debut_report`` once per set, then merged with
``assemble_finetune_report`` and written with ``write_finetune_report``.

Metrics produced per section
---------------------------
1. ``win_loss_accuracy`` -- fraction of examples whose generated position-0
   token matches the ground-truth outcome (``[WIN]``/``[LOSS]``).
2. ``build_order_f1`` -- debut build-order precision/recall/F1, reported three
   ways: an overall ``aggregate``, ``by_fog_class`` (visible/fogged/future
   debut), and ``by_fog_bucket`` (per-example fog-rate buckets).
3. ``debut_mae`` / ``debut_mae_matched_count`` -- timing error (mean absolute
   bucket difference) over debut events matched by entity type only, so
   "right unit, wrong time" is separated from "wrong unit".
4. ``grammar_validity`` -- fraction of generated canvases that satisfy the
   RELAXED debut grammar (``inference.decode.validate_debut_canvas``).
5. ``win_loss_minute_buckets`` -- cumulative outcome accuracy at each minute
   checkpoint, keyed by how far into the game the INPUT window reaches.
6. ``win_loss_structural`` -- structural booleans: outcome token at position 0,
   and (from the sampler trace) the outcome position denoises last.

Everything that is a threshold or bucket boundary is read from config
(``config.eval.*``, ``config.data.sampling_interval_s``). Nothing here mutates
the dataset target builder, the sampler, the loss, or the config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import DEBUT_CLASS_ID_TO_NAME, DatasetExample
from thesis_ml.eval.buildorder import BuildOrderEvent
from thesis_ml.eval.metrics import BuildOrderMetrics, aggregate_metrics, compare_build_orders
from thesis_ml.inference.decode import validate_debut_canvas
from thesis_ml.inference.sampler import sample_canvas
from thesis_ml.vocab.content_vocab import ContentVocabulary
from thesis_ml.vocab.special_tokens import (
    DELIMITER_ID,
    END_ID,
    LOSS_ID,
    MASK_ID,
    PAD_ID,
    WIN_ID,
)

# The three fog-state classes that a debut event can carry. Read from the shared
# debut class map (ids 0/1/2) so this module labels events identically to the
# dataset target builder. Ids 3+ (delimiter/end/pad/win-loss) are structural and
# never describe a debut event.
FOG_CLASS_NAMES: tuple[str, ...] = (
    DEBUT_CLASS_ID_TO_NAME[0],  # "visible-debut"
    DEBUT_CLASS_ID_TO_NAME[1],  # "fogged-debut"
    DEBUT_CLASS_ID_TO_NAME[2],  # "future-debut"
)

# Class ids that identify a debut event (as opposed to a structural token). Used
# to pull ground-truth debut events out of the target canvas class labels.
_DEBUT_EVENT_CLASS_IDS = frozenset({0, 1, 2})

# Valid section labels for a report section.
_SECTION_LABELS = ("memorized", "test")


# ---------------------------------------------------------------------------
# Config parsing helpers (the bucket lists live in config as CSV strings)
# ---------------------------------------------------------------------------


def _parse_float_csv(text: str) -> list[float]:
    """Parse a comma-separated string of numbers into a list of floats.

    Config validation only supports scalar field types, so the minute
    checkpoints and fog-rate edges are stored as strings like ``"1,3,5,7,10"``
    and parsed here. Empty entries (from trailing commas) are ignored.

    Parameters:
        text: The comma-separated numeric string from config.

    Returns:
        A list of floats in the order written.
    """

    return [float(piece) for piece in text.split(",") if piece.strip()]


def _fog_edges(config: ProjectConfig) -> tuple[float, float]:
    """Return the (low, high) fog-rate boundaries from config, sorted ascending.

    ``config.eval.debut_fog_bucket_edges`` must contain exactly two boundaries
    (default ``"0.3,0.7"``) that split examples into three fog buckets.

    Raises:
        ValueError: if the config string does not contain exactly two numbers.
    """

    edges = _parse_float_csv(config.eval.debut_fog_bucket_edges)
    if len(edges) != 2:
        raise ValueError(
            f"debut_fog_bucket_edges must have exactly two values, got {edges!r}"
        )
    low, high = sorted(edges)
    return low, high


def _fog_bucket_labels(low: float, high: float) -> tuple[str, str, str]:
    """Build the three fog-bucket key names from the boundaries.

    With the default edges 0.3/0.7 this returns ``(">70", "30-70", "<30")`` --
    the exact keys the report emits. The percentages are derived from the edges
    so a different config produces matching labels.

    Returns:
        A 3-tuple of keys ordered high / middle / low.
    """

    low_pct = int(round(low * 100))
    high_pct = int(round(high * 100))
    return (f">{high_pct}", f"{low_pct}-{high_pct}", f"<{low_pct}")


def _fog_bucket_for_rate(rate: float, low: float, high: float) -> str:
    """Return which fog-bucket key an example's fog rate falls into.

    Boundaries are inclusive on the middle bucket: ``rate < low`` -> low bucket,
    ``low <= rate <= high`` -> middle bucket, ``rate > high`` -> high bucket.
    """

    low_key, mid_key, high_key = (
        f"<{int(round(low * 100))}",
        f"{int(round(low * 100))}-{int(round(high * 100))}",
        f">{int(round(high * 100))}",
    )
    if rate < low:
        return low_key
    if rate > high:
        return high_key
    return mid_key


def _minute_key(minute: float) -> str:
    """Format a minute checkpoint as a JSON key ("1" not "1.0" when integral)."""

    if float(minute).is_integer():
        return str(int(minute))
    return str(minute)


# ---------------------------------------------------------------------------
# Per-example feature helpers
# ---------------------------------------------------------------------------


def _example_fog_rate(example: DatasetExample) -> float:
    """Compute one example's fog rate: fraction of enemy debuts hidden from input.

    Definition (documented for the report): the fog rate is the number of
    fogged enemy entity tokens divided by the total number of enemy entity
    tokens that COULD have been observed in the input window, i.e.

        fog_rate = sum(fogged_counts) / (sum(fogged_counts) + sum(observed_counts))

    ``fogged_counts`` and ``observed_counts`` are produced by the dataset input
    builder and count, per (relative timestep, token name), how many enemy
    entities were hidden vs. shown. When the denominator is zero (no enemy
    entities in the window) the fog rate is defined as 0.0.

    Parameters:
        example: The dataset example to score.

    Returns:
        A fog rate in [0, 1].
    """

    fogged = sum(example.fogged_counts.values())
    observed = sum(example.observed_counts.values())
    total = fogged + observed
    if total == 0:
        return 0.0
    return float(fogged) / float(total)


def _last_input_clock_seconds(example: DatasetExample, config: ProjectConfig) -> float:
    """Return how far into the game (in seconds) the INPUT window reaches.

    Prefers the largest real timestamp among the input records (absolute game
    seconds). When timestamps are unavailable (e.g. synthetic examples), it
    falls back to converting the input window's end timestep into seconds via
    ``config.data.sampling_interval_s``.

    Parameters:
        example: The dataset example.
        config: Project config (for the sampling interval).

    Returns:
        Seconds from game start to the end of the input window.
    """

    clocks = [
        float(record.timestamp_seconds)
        for record in example.input_records
        if getattr(record, "timestamp_seconds", None) is not None
    ]
    if clocks:
        return max(clocks)
    interval = float(config.data.sampling_interval_s)
    if example.window_end is not None:
        return float(example.window_end) * interval
    return float(example.window_start) * interval


def _input_reach_minutes(example: DatasetExample, config: ProjectConfig) -> float:
    """Convert the input window's reach into minutes (seconds / 60)."""

    return _last_input_clock_seconds(example, config) / 60.0


# ---------------------------------------------------------------------------
# Debut event extraction (ground truth from target canvas; predicted from
# the generated canvas). Unlike pre-training snapshots, a debut canvas lists
# each first-appearance directly, so every content token is ONE debut event at
# its timestep -- no cross-timestep count differencing is needed.
# ---------------------------------------------------------------------------


def _ground_truth_debut_events(
    example: DatasetExample,
) -> list[tuple[BuildOrderEvent, str]]:
    """Extract ground-truth debut events with their fog class from an example.

    The debut target canvas was built with per-token class labels (visible /
    fogged / future debut). We read those labels straight off the example: for
    every position whose class id is a debut-event class, we emit a
    ``BuildOrderEvent`` (entity type + timestep bucket) tagged with its fog
    class name.

    The timestep bucket comes from the aligned ``canvas_metadata`` entry's
    ``timestep_index`` (the relative timestep from the window start), which is
    the same basis the predicted decoder uses (delimiters counted so far).

    Parameters:
        example: A debut-mode dataset example carrying ``target_canvas``,
            ``class_labels``, and aligned ``canvas_metadata``.

    Returns:
        A list of (event, fog_class_name) pairs. Empty if the example lacks
        aligned metadata (fine-tuning examples always carry it).
    """

    labels = example.class_labels.tolist()
    metadata = example.canvas_metadata
    # Fine-tuning examples always carry one metadata dict per canvas position.
    # If that invariant is broken we cannot attribute fog classes, so we return
    # nothing rather than guess.
    if len(metadata) != len(labels):
        return []

    events: list[tuple[BuildOrderEvent, str]] = []
    for label, meta in zip(labels, metadata):
        if label not in _DEBUT_EVENT_CLASS_IDS:
            continue
        token_name = meta.get("token_name")
        timestep_index = meta.get("timestep_index")
        if token_name is None or timestep_index is None:
            continue
        event = BuildOrderEvent(entity_type=str(token_name), bucket=int(timestep_index))
        events.append((event, DEBUT_CLASS_ID_TO_NAME[label]))
    return events


def _decode_predicted_debut_events(
    token_ids: Sequence[int],
    vocabulary: ContentVocabulary | Mapping[int, str],
) -> list[BuildOrderEvent]:
    """Decode a (validated) generated debut canvas into debut events.

    Walks the canvas starting AFTER the position-0 outcome token. Each
    ``[DELIMITER]`` advances the timestep bucket; each content token becomes one
    ``BuildOrderEvent`` at the current bucket. Decoding stops at the first
    terminal special token (``[END]``/``[PAD]``). Callers should only pass
    canvases that already passed ``validate_debut_canvas``.

    Parameters:
        token_ids: The generated canvas token ids (position 0 = outcome).
        vocabulary: Content vocabulary (or an id->name mapping) to name tokens.

    Returns:
        A list of debut events, one per content token, in canvas order.
    """

    id_to_name = vocabulary.id_to_name if isinstance(vocabulary, ContentVocabulary) else vocabulary
    events: list[BuildOrderEvent] = []
    current_timestep = 0
    for token_id in token_ids[1:]:  # skip the position-0 outcome token
        if token_id == DELIMITER_ID:
            current_timestep += 1
            continue
        if token_id in (END_ID, PAD_ID, MASK_ID, WIN_ID, LOSS_ID):
            break  # reached the terminal region; no more debut events
        name = id_to_name.get(int(token_id))
        if name is None:
            continue  # unknown content id -> skip defensively
        events.append(BuildOrderEvent(entity_type=str(name), bucket=current_timestep))
    return events


# ---------------------------------------------------------------------------
# Timing MAE over matches by entity type only
# ---------------------------------------------------------------------------


def _absolute_timing_diffs(
    predicted: Sequence[BuildOrderEvent],
    ground_truth: Sequence[BuildOrderEvent],
) -> list[int]:
    """Match predicted->ground-truth debuts by entity type and return |bucket diffs|.

    Matching here uses entity type ONLY (no timing tolerance): each predicted
    event greedily claims the nearest-bucket unmatched ground-truth event of the
    same entity type. This isolates timing error -- a matched pair with a large
    bucket gap is "right unit, wrong time", while an unmatched prediction is
    "wrong unit" and simply contributes no timing sample.

    Parameters:
        predicted: Predicted debut events.
        ground_truth: Ground-truth debut events.

    Returns:
        A list of absolute bucket differences, one per matched pair.
    """

    unmatched_indices = list(range(len(ground_truth)))
    diffs: list[int] = []
    for prediction in predicted:
        candidates = [
            index
            for index in unmatched_indices
            if ground_truth[index].entity_type == prediction.entity_type
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(ground_truth[index].bucket - prediction.bucket))
        unmatched_indices.remove(best)
        diffs.append(abs(ground_truth[best].bucket - prediction.bucket))
    return diffs


# ---------------------------------------------------------------------------
# Structural checks driven by the sampler trace
# ---------------------------------------------------------------------------


def _first_commit_steps(trace: Sequence[object], row: int = 0) -> dict[int, int]:
    """Map each canvas position to the 1-based sampler step it was first committed.

    The sampler records, per step, a boolean tensor of positions committed that
    step. We scan steps in order and record the earliest step for each position.

    Parameters:
        trace: The sampler's ``trace`` (a sequence of ``SamplerStep``).
        row: Batch row to inspect (single-example batches use row 0).

    Returns:
        A dict position -> first-commit step number.
    """

    steps: dict[int, int] = {}
    for step in trace:
        committed = step.committed_this_step[row]  # bool tensor over canvas positions
        for position in torch.nonzero(committed, as_tuple=False).flatten().tolist():
            steps.setdefault(int(position), int(step.step))
    return steps


def _denoise_last_ok(trace: Sequence[object], row: int = 0) -> bool:
    """Return True when the position-0 outcome token was committed LAST.

    "Last" means position 0's first-commit step is greater than or equal to
    every other position's first-commit step -- i.e. no canvas position was
    committed after the outcome token. This reflects the intended behaviour of
    the W2 sampler constraint ``sampler.outcome_last``; when that constraint is
    off, this check will honestly report False.

    Parameters:
        trace: The sampler ``trace``.
        row: Batch row to inspect.

    Returns:
        True if the outcome position denoised last, else False.
    """

    steps = _first_commit_steps(trace, row)
    if 0 not in steps:
        return False
    outcome_step = steps[0]
    return outcome_step >= max(steps.values())


# ---------------------------------------------------------------------------
# Per-example evaluation bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExampleEvaluation:
    """Everything computed for one example, before section-level aggregation."""

    valid: bool
    predicted_outcome: int
    ground_truth_outcome: int
    predicted_events: list[BuildOrderEvent]
    ground_truth_events: list[tuple[BuildOrderEvent, str]]
    fog_rate: float
    input_reach_minutes: float
    position0_ok: bool
    denoise_last_ok: bool
    # Per-example overall build-order metrics (predicted vs. all ground truth).
    aggregate_metrics: BuildOrderMetrics


def _evaluate_example(
    *,
    model: nn.Module,
    example: DatasetExample,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    device: torch.device | str,
) -> _ExampleEvaluation:
    """Generate a canvas for one example and compute all its per-example signals.

    Runs the confidence-based sampler once (retaining its trace for the
    denoise-last structural check), decodes the debut events, and gathers the
    fog rate, input reach, outcome tokens, and overall build-order metrics.

    Parameters:
        model: The fine-tuned diffusion model to sample from.
        example: The dataset example to evaluate.
        vocabulary: Content vocabulary to name predicted tokens.
        config: Project config (buckets, sampling interval, tolerance).
        device: Torch device for sampling.

    Returns:
        A populated ``_ExampleEvaluation``.

    Calls:
        ``collate_diffusion_examples``, ``sample_canvas``,
        ``validate_debut_canvas``, ``_decode_predicted_debut_events``,
        ``_ground_truth_debut_events``, ``compare_build_orders``.
    """

    # Fine-tuning report path; debut_mode is always True here, threaded from
    # config so collate scopes its future telemetry correctly.
    batch = collate_diffusion_examples([example], debut_mode=config.data.debut_mode)
    sampled = sample_canvas(model, batch, config, device=device)
    canvas = sampled.canvas[0].tolist()

    validation = validate_debut_canvas(canvas)
    predicted_events = (
        _decode_predicted_debut_events(canvas, vocabulary) if validation.valid else []
    )
    ground_truth_events = _ground_truth_debut_events(example)
    gt_only_events = [event for event, _ in ground_truth_events]

    # Overall (fog-class-agnostic) build-order metrics for this example.
    per_example_metrics = compare_build_orders(
        predicted_events,
        gt_only_events,
        timing_tolerance_buckets=config.eval.timing_tolerance_buckets,
    )

    # Ground-truth outcome is the position-0 token baked into the debut target
    # by the dataset builder (which set it via resolve_replay_outcome). Reading
    # it from the target keeps evaluation hermetic (no re-reading metadata).
    ground_truth_outcome = int(example.target_canvas[0].item())
    predicted_outcome = int(canvas[0])

    # Structural: outcome token appears exactly once, only at position 0.
    outcome_positions = [index for index, token in enumerate(canvas) if token in (WIN_ID, LOSS_ID)]
    position0_ok = outcome_positions == [0]

    return _ExampleEvaluation(
        valid=validation.valid,
        predicted_outcome=predicted_outcome,
        ground_truth_outcome=ground_truth_outcome,
        predicted_events=predicted_events,
        ground_truth_events=ground_truth_events,
        fog_rate=_example_fog_rate(example),
        input_reach_minutes=_input_reach_minutes(example, config),
        position0_ok=position0_ok,
        denoise_last_ok=_denoise_last_ok(sampled.trace),
        aggregate_metrics=per_example_metrics,
    )


# ---------------------------------------------------------------------------
# Metric serialisation
# ---------------------------------------------------------------------------


def _metrics_to_dict(metrics: BuildOrderMetrics) -> dict[str, float | int]:
    """Flatten a ``BuildOrderMetrics`` into the report's scalar metric block."""

    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "true_positives": metrics.true_positives,
        "predicted_count": metrics.predicted_count,
        "ground_truth_count": metrics.ground_truth_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_debut_report(
    examples: Sequence[DatasetExample],
    *,
    label: str,
    model: nn.Module,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Build one report SECTION ("memorized" or "test") for a set of examples.

    This is the public entry point the pipeline calls once per set. It generates
    a canvas for every example and assembles the nested metric dict described in
    this module's header. The returned dict is the CONTENTS of one section; the
    caller places it under the "memorized" or "test" key (see
    ``assemble_finetune_report``).

    Parameters:
        examples: The replays/examples to score for this section.
        label: Either "memorized" or "test" (stored on the section for
            traceability; the metric keys are identical across labels).
        model: The fine-tuned diffusion model to sample from.
        vocabulary: Content vocabulary for naming predicted tokens.
        config: Project config (bucket definitions, tolerance, sampling interval).
        device: Torch device for sampling.

    Returns:
        A section dict with keys: ``label``, ``example_count``,
        ``evaluated_count``, ``win_loss_accuracy``, ``build_order_f1``
        (``aggregate`` / ``by_fog_class`` / ``by_fog_bucket``), ``debut_mae``,
        ``debut_mae_matched_count``, ``grammar_validity``,
        ``win_loss_minute_buckets``, and ``win_loss_structural``.

    Raises:
        ValueError: if ``label`` is not "memorized"/"test", or if the fog edge
            config is malformed.
    """

    if label not in _SECTION_LABELS:
        raise ValueError(f"label must be one of {_SECTION_LABELS}, got {label!r}")

    low, high = _fog_edges(config)
    minute_checkpoints = _parse_float_csv(config.eval.debut_minute_buckets)
    tolerance = config.eval.timing_tolerance_buckets

    # Evaluate every example once (this is where sampling happens).
    evaluations = [
        _evaluate_example(
            model=model,
            example=example,
            vocabulary=vocabulary,
            config=config,
            device=device,
        )
        for example in examples
    ]
    example_count = len(evaluations)

    # --- Win/loss accuracy -------------------------------------------------
    correct_outcomes = sum(
        1 for ev in evaluations if ev.predicted_outcome == ev.ground_truth_outcome
    )
    win_loss_accuracy = correct_outcomes / example_count if example_count else 0.0

    # --- Grammar validity --------------------------------------------------
    valid_count = sum(1 for ev in evaluations if ev.valid)
    grammar_validity = valid_count / example_count if example_count else 0.0

    # --- Build-order F1: overall aggregate ---------------------------------
    aggregate = aggregate_metrics([ev.aggregate_metrics for ev in evaluations])

    # --- Build-order F1: split by fog class --------------------------------
    # For each fog class we micro-aggregate per-example comparisons of ALL
    # predicted events against the ground-truth events of that class only.
    # Documented precision semantics: precision_C = TP_C / (all predicted debuts)
    # -- the share of predictions that correctly landed on a class-C debut --
    # while recall_C = TP_C / (class-C ground-truth debuts).
    by_fog_class: dict[str, dict[str, float | int]] = {}
    for fog_class in FOG_CLASS_NAMES:
        per_example_class_metrics = []
        for ev in evaluations:
            gt_class_events = [event for event, cls in ev.ground_truth_events if cls == fog_class]
            per_example_class_metrics.append(
                compare_build_orders(
                    ev.predicted_events,
                    gt_class_events,
                    timing_tolerance_buckets=tolerance,
                )
            )
        by_fog_class[fog_class] = _metrics_to_dict(aggregate_metrics(per_example_class_metrics))

    # --- Build-order F1: split by per-example fog-rate bucket --------------
    bucket_keys = _fog_bucket_labels(low, high)
    metrics_by_bucket: dict[str, list[BuildOrderMetrics]] = {key: [] for key in bucket_keys}
    for ev in evaluations:
        bucket_key = _fog_bucket_for_rate(ev.fog_rate, low, high)
        metrics_by_bucket[bucket_key].append(ev.aggregate_metrics)
    by_fog_bucket = {
        key: _metrics_to_dict(aggregate_metrics(metrics))
        for key, metrics in metrics_by_bucket.items()
    }

    # --- Debut timing MAE over entity-type matches -------------------------
    all_timing_diffs: list[int] = []
    for ev in evaluations:
        gt_only = [event for event, _ in ev.ground_truth_events]
        all_timing_diffs.extend(_absolute_timing_diffs(ev.predicted_events, gt_only))
    debut_mae = sum(all_timing_diffs) / len(all_timing_diffs) if all_timing_diffs else 0.0
    debut_mae_matched_count = len(all_timing_diffs)

    # --- Cumulative win/loss accuracy by input-reach minute checkpoint -----
    # Cumulative: bucket M holds every example whose input window reaches AT
    # MOST M minutes into the game; the value is outcome accuracy over those.
    win_loss_minute_buckets: dict[str, float] = {}
    for minute in minute_checkpoints:
        subset = [ev for ev in evaluations if ev.input_reach_minutes <= minute]
        if subset:
            subset_correct = sum(1 for ev in subset if ev.predicted_outcome == ev.ground_truth_outcome)
            win_loss_minute_buckets[_minute_key(minute)] = subset_correct / len(subset)
        else:
            win_loss_minute_buckets[_minute_key(minute)] = 0.0

    # --- Structural booleans ----------------------------------------------
    position0_ok = all(ev.position0_ok for ev in evaluations) if evaluations else False
    denoise_last_ok = all(ev.denoise_last_ok for ev in evaluations) if evaluations else False

    return {
        "label": label,
        "example_count": example_count,
        "evaluated_count": valid_count,
        "win_loss_accuracy": win_loss_accuracy,
        "build_order_f1": {
            "aggregate": _metrics_to_dict(aggregate),
            "by_fog_class": by_fog_class,
            "by_fog_bucket": by_fog_bucket,
        },
        "debut_mae": debut_mae,
        "debut_mae_matched_count": debut_mae_matched_count,
        "grammar_validity": grammar_validity,
        "win_loss_minute_buckets": win_loss_minute_buckets,
        "win_loss_structural": {
            "position0_ok": position0_ok,
            "denoise_last_ok": denoise_last_ok,
        },
    }


def assemble_finetune_report(
    *,
    memorized: dict[str, object],
    test: dict[str, object],
) -> dict[str, object]:
    """Merge the two per-set sections into the final report dict.

    Parameters:
        memorized: Section dict from ``build_debut_report(label="memorized")``.
        test: Section dict from ``build_debut_report(label="test")``.

    Returns:
        ``{"memorized": memorized, "test": test}`` -- the shape the pipeline
        writes to ``finetune_report.json``. Both sections carry identical keys.
    """

    return {"memorized": memorized, "test": test}


def write_finetune_report(report: dict[str, object], path: str | Path) -> dict[str, object]:
    """Write an assembled report dict to ``path`` as pretty-printed JSON.

    Parameters:
        report: The assembled report (see ``assemble_finetune_report``).
        path: Destination file path (parent directories are created).

    Returns:
        The same ``report`` dict (for convenient chaining).
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_and_write_finetune_report(
    *,
    memorized_examples: Sequence[DatasetExample],
    test_examples: Sequence[DatasetExample],
    model: nn.Module,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Convenience: build both sections, assemble, and write the JSON report.

    This is the single call a pipeline can use end-to-end. It runs
    ``build_debut_report`` for the memorized and test sets, merges them, and
    writes ``finetune_report.json`` at ``path``.

    Parameters:
        memorized_examples: The fine-tuned-on replays (memorized section).
        test_examples: The held-out replays (test section).
        model: The fine-tuned diffusion model.
        vocabulary: Content vocabulary.
        config: Project config.
        path: Destination JSON path.
        device: Torch device for sampling.

    Returns:
        The assembled report dict that was written.
    """

    memorized = build_debut_report(
        memorized_examples,
        label="memorized",
        model=model,
        vocabulary=vocabulary,
        config=config,
        device=device,
    )
    test = build_debut_report(
        test_examples,
        label="test",
        model=model,
        vocabulary=vocabulary,
        config=config,
        device=device,
    )
    report = assemble_finetune_report(memorized=memorized, test=test)
    return write_finetune_report(report, path)
