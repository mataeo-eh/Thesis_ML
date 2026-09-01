"""Held-out build-order accuracy, scored two ways: strict grammar and salvaged.

Question this answers
---------------------
"Across many unseen games, how much of the opponent's actual build order does
the model get right, at the model's own time resolution?"

Method
------
This is the harness described in `EVAL.md`, run over the held-out test split.
For each scored window:

  1. the model generates a full output canvas with the real iterative sampler
     (``inference.sampler.sample_canvas``);
  2. the canvas is grammar-validated and decoded into per-timestep entity-type
     counts (``inference.decode.decode_canvas``);
  3. both the prediction and the parsed ground truth are reduced to the SAME
     event representation ``(entity_type, appearance_bucket)`` by the single
     build-order oracle (``eval.buildorder``);
  4. events are matched entity-type-exact with a bucket tolerance of
     ``config.eval.timing_tolerance_buckets``, one-to-one, giving precision,
     recall, F1, and accuracy (``eval.metrics.compare_build_orders``).

Two scores, and why
-------------------
``decode_canvas`` returns NOTHING the moment ``validate_canvas`` fails. That is
correct behaviour for the deployed pipeline -- an ill-formed canvas is not a
valid model output. But it makes the aggregate score nearly uninterpretable as a
measure of build-order knowledge: one misplaced ``[DELIMITER]`` zeroes an entire
window, and the windows that fail the grammar are exactly the ones whose content
would be most informative.

So every window is scored twice, against the identical ground truth:

  * **strict** -- the project grammar, byte-for-byte what ``EVAL.md`` and the
    training pipeline report. An invalid canvas scores zero. Use this number
    when comparing against anything else in the project.
  * **lenient** -- the content salvaged from the canvas regardless of grammar
    (``inference_test_api.salvage_canvas_timesteps``), then run through the SAME
    ``extract_build_order`` + ``compare_build_orders`` path. This isolates "does
    the model know the build order" from "does the model emit a well-formed
    envelope".

The gap between the two IS the cost of the grammar failures. Grammar validity is
reported alongside as its own first-class metric, now with the specific rule each
failing window violated, so a structural problem shows up as a structural problem
rather than as a mysteriously low F1.

Depends on: ``eval.harness.evaluate_example``, ``eval.metrics.aggregate_metrics`` /
``compare_build_orders``, ``eval.buildorder.extract_build_order``,
``inference.decode.validate_canvas``, ``inference_test_api``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window on a training box
import matplotlib.pyplot as plt  # noqa: E402

from thesis_ml.eval.buildorder import extract_build_order
from thesis_ml.eval.harness import evaluate_example
from thesis_ml.eval.metrics import BuildOrderMetrics, aggregate_metrics, compare_build_orders
from thesis_ml.inference.decode import validate_canvas

from inference_test_api import (  # noqa: E402
    TestContext,
    TestResult,
    portable_path,
    salvage_canvas_timesteps,
    save_figure,
    write_csv,
    write_json,
)


TEST_NAME = "build_order_accuracy"
TEST_TITLE = "Held-out build-order precision / recall / F1 (strict + salvaged)"
TEST_DESCRIPTION = """
Runs the project's build-order evaluation harness (`EVAL.md`) over windows drawn
from the held-out test replays. Each window is generated with the full iterative
sampler, decoded, reduced to `(entity_type, appearance_bucket)` events, and
matched one-to-one against the ground-truth build order extracted from the same
replay's parquet rows, with the configured timing tolerance.

Every window is scored **twice** against the same ground truth: `strict` (the
project grammar -- an invalid canvas scores zero, matching what the training
pipeline reports) and `lenient` (content salvaged regardless of grammar, through
the identical event-extraction and matching path). The gap between them is
exactly what the grammar failures cost, and it separates "does not know the build
order" from "knows it but emits a malformed envelope".

Grammar validity is reported as its own metric, broken down by the specific
validator rule each failing window violated.
""".strip()
TEST_OUTPUTS = (
    "`build_order_metrics.json` -- pooled strict and lenient precision/recall/F1/accuracy, the grammar-failure histogram by rule, per-entity-type breakdown, and full provenance",
    "`per_window_metrics.csv` -- one row per scored window: replay, perspective, window start, validity, the exact grammar diagnosis, and both strict and lenient precision/recall/F1",
    "`per_entity_type_metrics.csv` -- one row per entity type (lenient scoring): pooled true positives, predicted count, ground-truth count, precision/recall/F1",
    "`grammar_failures.csv` -- one row per distinct validator rule violated, with how many windows hit it",
    "`per_window_f1_distribution.png` / `.svg` -- overlaid histograms of per-window strict and lenient F1, with both pooled values marked",
    "`strict_vs_lenient_f1.png` / `.svg` -- pooled precision / recall / F1 side by side under both scorings",
    "`per_entity_type_recall.png` / `.svg` -- lenient recall by entity type, ordered by how often the type actually occurs",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False

# Iterative sampling dominates the cost here, so the default sample is bounded.
# Raise it with `--option build_order_windows=N` when a tighter confidence
# interval is worth the wall-clock time.
DEFAULT_MAX_WINDOWS = 40

# How many entity types the recall bar chart shows. The tail is long and mostly
# single-occurrence types, which make the figure unreadable without adding
# information; the full table is in the CSV either way.
TOP_ENTITY_TYPES = 25


def run(context: TestContext) -> TestResult:
    """Score held-out windows strictly and leniently, then write the report.

    Parameters:
        context: runner-supplied context.

    Returns:
        A :class:`TestResult` carrying both pooled metric sets.

    Calls: ``SharedResources.model`` / ``examples``, ``eval.harness.evaluate_example``,
    ``salvage_canvas_timesteps``, ``eval.buildorder.extract_build_order``,
    ``eval.metrics.compare_build_orders`` / ``aggregate_metrics``.
    """

    max_windows = context.option_int("build_order_windows", DEFAULT_MAX_WINDOWS)
    if context.max_examples > 0:
        max_windows = min(max_windows, context.max_examples)

    model, run_config = context.shared.model()
    vocabulary = context.shared.vocabulary()
    tolerance = run_config.eval.timing_tolerance_buckets

    examples = context.shared.examples(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=max_windows,
        run_config=run_config,
    )
    print(f"scoring {len(examples)} held-out window(s), timing tolerance {tolerance} bucket(s)")

    per_window_rows: list[dict[str, Any]] = []
    strict_metrics: list[BuildOrderMetrics] = []
    lenient_metrics: list[BuildOrderMetrics] = []
    diagnosis_counts: Counter[str] = Counter()
    salvage_totals = {"skipped_special_tokens": 0, "skipped_unknown_tokens": 0, "trailing_partial": 0}

    # Per-entity-type counts pooled across windows, from the LENIENT scoring so
    # the invalid windows -- the informative ones -- still contribute. Pooling raw
    # counts (not averaging per-window rates) is what makes a type appearing once
    # per game comparable to one appearing thirty times.
    type_true_positives: dict[str, int] = defaultdict(int)
    type_predicted: dict[str, int] = defaultdict(int)
    type_ground_truth: dict[str, int] = defaultdict(int)

    for index, example in enumerate(examples):
        result = evaluate_example(
            model=model,
            example=example,
            vocabulary=vocabulary,
            config=run_config,
            device=context.device,
        )
        strict = result.metrics
        strict_metrics.append(strict)

        # Re-validate the canvas locally to recover the DIAGNOSIS string. The
        # harness keeps only the boolean, and "which rule failed" is the whole
        # difference between an actionable structural finding and a mystery.
        validation = validate_canvas(list(result.predicted_canvas))
        if not validation.valid:
            diagnosis_counts[validation.diagnosis or "unknown"] += 1

        # Lenient pass: salvage the content, then run the IDENTICAL event
        # extraction and matching the strict path uses, against the identical
        # ground truth. `extract_build_order` buckets by positional index, which
        # is exactly what `attach_absolute_times` assigns, so the two scorings
        # are bucket-compatible.
        salvaged = salvage_canvas_timesteps(result.predicted_canvas, vocabulary)
        salvage_totals["skipped_special_tokens"] += salvaged.skipped_special_tokens
        salvage_totals["skipped_unknown_tokens"] += salvaged.skipped_unknown_tokens
        salvage_totals["trailing_partial"] += int(salvaged.trailing_partial_timestep)
        lenient_events = extract_build_order(salvaged.timesteps, drop_final_timestep=False)
        lenient = compare_build_orders(
            lenient_events,
            result.ground_truth_events,
            timing_tolerance_buckets=tolerance,
        )
        lenient_metrics.append(lenient)

        per_window_rows.append(
            {
                "replay_id": example.replay_id,
                "perspective": example.perspective_player,
                "window_start": example.window_start,
                "prediction_valid": validation.valid,
                "grammar_diagnosis": "" if validation.valid else (validation.diagnosis or "unknown"),
                "strict_precision": round(strict.precision, 6),
                "strict_recall": round(strict.recall, 6),
                "strict_f1": round(strict.f1, 6),
                "strict_true_positives": strict.true_positives,
                "strict_predicted_events": strict.predicted_count,
                "lenient_precision": round(lenient.precision, 6),
                "lenient_recall": round(lenient.recall, 6),
                "lenient_f1": round(lenient.f1, 6),
                "lenient_true_positives": lenient.true_positives,
                "lenient_predicted_events": lenient.predicted_count,
                "ground_truth_events": lenient.ground_truth_count,
            }
        )
        for entity_type, per_type in lenient.per_entity_type.items():
            type_true_positives[entity_type] += per_type.true_positives
            type_predicted[entity_type] += per_type.predicted_count
            type_ground_truth[entity_type] += per_type.ground_truth_count

        if (index + 1) % 5 == 0 or index + 1 == len(examples):
            print(f"  scored {index + 1}/{len(examples)} windows", flush=True)

    pooled_strict = aggregate_metrics(strict_metrics)
    pooled_lenient = aggregate_metrics(lenient_metrics)
    valid_windows = sum(1 for row in per_window_rows if row["prediction_valid"])
    validity_rate = valid_windows / len(per_window_rows) if per_window_rows else 0.0

    entity_rows = _entity_type_rows(type_true_positives, type_predicted, type_ground_truth)
    failure_rows = [
        {"grammar_diagnosis": diagnosis, "windows": count}
        for diagnosis, count in diagnosis_counts.most_common()
    ]

    written: list[Path] = []
    written.append(
        write_json(
            {
                "provenance": context.provenance(uses_model=True),
                "timing_tolerance_buckets": tolerance,
                "sampler": "iterative (inference.sampler.sample_canvas)",
                "n_windows": len(per_window_rows),
                "scoring_note": (
                    "'strict' is the project grammar (invalid canvas scores zero) and is the "
                    "number comparable to EVAL.md and the training pipeline. 'lenient' salvages "
                    "the canvas content regardless of grammar and scores it through the same "
                    "extract_build_order + compare_build_orders path. The gap is what the "
                    "grammar failures cost."
                ),
                "grammar": {
                    "valid_windows": valid_windows,
                    "validity_rate": validity_rate,
                    "failures_by_rule": dict(diagnosis_counts),
                    "salvage_totals": salvage_totals,
                },
                "pooled_strict": _metrics_dict(pooled_strict),
                "pooled_lenient": _metrics_dict(pooled_lenient),
                "per_entity_type_lenient": entity_rows,
            },
            context.out_dir / "build_order_metrics.json",
        )
    )
    written.append(
        write_csv(
            per_window_rows,
            [
                "replay_id",
                "perspective",
                "window_start",
                "prediction_valid",
                "grammar_diagnosis",
                "strict_precision",
                "strict_recall",
                "strict_f1",
                "strict_true_positives",
                "strict_predicted_events",
                "lenient_precision",
                "lenient_recall",
                "lenient_f1",
                "lenient_true_positives",
                "lenient_predicted_events",
                "ground_truth_events",
            ],
            context.out_dir / "per_window_metrics.csv",
        )
    )
    written.append(
        write_csv(
            entity_rows,
            [
                "entity_type",
                "true_positives",
                "predicted_events",
                "ground_truth_events",
                "precision",
                "recall",
                "f1",
            ],
            context.out_dir / "per_entity_type_metrics.csv",
        )
    )
    written.append(
        write_csv(
            failure_rows,
            ["grammar_diagnosis", "windows"],
            context.out_dir / "grammar_failures.csv",
        )
    )
    written.extend(_plot_f1_distribution(per_window_rows, pooled_strict, pooled_lenient, context))
    written.extend(_plot_strict_vs_lenient(pooled_strict, pooled_lenient, context))
    written.extend(_plot_entity_recall(entity_rows, context))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    top_failure = diagnosis_counts.most_common(1)
    headline = [
        f"strict  F1 {pooled_strict.f1:.4f} (P {pooled_strict.precision:.4f}, R {pooled_strict.recall:.4f})"
        f" over {len(per_window_rows)} held-out windows",
        f"lenient F1 {pooled_lenient.f1:.4f} (P {pooled_lenient.precision:.4f}, R {pooled_lenient.recall:.4f})"
        f" -- grammar failures cost {pooled_lenient.f1 - pooled_strict.f1:+.4f} F1",
        f"grammar-valid canvases: {valid_windows}/{len(per_window_rows)} ({100.0 * validity_rate:.0f}%)",
    ]
    if top_failure:
        rule, count = top_failure[0]
        headline.append(f"most common failure ({count} windows): {rule}")

    return TestResult(
        headline=headline,
        artifacts=written,
        metrics={
            "n_windows": len(per_window_rows),
            "strict": _metrics_dict(pooled_strict),
            "lenient": _metrics_dict(pooled_lenient),
            "grammar_validity_rate": validity_rate,
            "grammar_failures_by_rule": dict(diagnosis_counts),
            "timing_tolerance_buckets": tolerance,
        },
    )


def _metrics_dict(metrics: BuildOrderMetrics) -> dict[str, Any]:
    """Flatten a ``BuildOrderMetrics`` into JSON-ready scalars (no per-type nesting)."""

    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positives": metrics.true_positives,
        "predicted_events": metrics.predicted_count,
        "ground_truth_events": metrics.ground_truth_count,
    }


def _entity_type_rows(
    true_positives: dict[str, int],
    predicted: dict[str, int],
    ground_truth: dict[str, int],
) -> list[dict[str, Any]]:
    """Build the per-entity-type table, ordered most-frequent-first."""

    rows = []
    for entity_type in sorted(
        set(ground_truth) | set(predicted), key=lambda name: (-ground_truth[name], name)
    ):
        matched = true_positives[entity_type]
        predicted_count = predicted[entity_type]
        truth_count = ground_truth[entity_type]
        precision = matched / predicted_count if predicted_count else 0.0
        recall = matched / truth_count if truth_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "entity_type": entity_type,
                "true_positives": matched,
                "predicted_events": predicted_count,
                "ground_truth_events": truth_count,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return rows


def _plot_f1_distribution(
    rows: list[dict[str, Any]],
    pooled_strict: BuildOrderMetrics,
    pooled_lenient: BuildOrderMetrics,
    context: TestContext,
) -> list[Path]:
    """Overlaid per-window F1 histograms for both scorings.

    The strict histogram's spike at zero IS the grammar-failure population. Seeing
    it next to the lenient distribution shows at a glance whether those windows
    were genuinely empty or merely malformed.
    """

    if not rows:
        return []
    strict_values = [float(row["strict_f1"]) for row in rows]
    lenient_values = [float(row["lenient_f1"]) for row in rows]

    figure, axes = plt.subplots(figsize=(8.0, 4.4))
    bins = 20
    axes.hist(
        strict_values, bins=bins, range=(0.0, 1.0), alpha=0.65, color="#C4402A", label="strict"
    )
    axes.hist(
        lenient_values, bins=bins, range=(0.0, 1.0), alpha=0.65, color="#3B6EA5", label="lenient"
    )
    axes.axvline(pooled_strict.f1, color="#C4402A", linestyle="--", linewidth=2)
    axes.axvline(pooled_lenient.f1, color="#3B6EA5", linestyle="--", linewidth=2)
    axes.set_xlabel("per-window build-order F1")
    axes.set_ylabel("held-out windows")
    axes.set_title(
        f"Build-order F1 across {len(rows)} held-out windows\n"
        f"pooled strict {pooled_strict.f1:.3f} (dashed red) vs lenient "
        f"{pooled_lenient.f1:.3f} (dashed blue)  --  {context.model_label}",
        fontsize=10,
    )
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "per_window_f1_distribution", dpi=context.dpi)


def _plot_strict_vs_lenient(
    pooled_strict: BuildOrderMetrics, pooled_lenient: BuildOrderMetrics, context: TestContext
) -> list[Path]:
    """Grouped bars comparing pooled precision / recall / F1 under both scorings."""

    labels = ["precision", "recall", "F1"]
    strict_values = [pooled_strict.precision, pooled_strict.recall, pooled_strict.f1]
    lenient_values = [pooled_lenient.precision, pooled_lenient.recall, pooled_lenient.f1]
    positions = range(len(labels))
    width = 0.38

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    strict_bars = axes.bar(
        [p - width / 2 for p in positions], strict_values, width, label="strict", color="#C4402A"
    )
    lenient_bars = axes.bar(
        [p + width / 2 for p in positions], lenient_values, width, label="lenient", color="#3B6EA5"
    )
    for bars in (strict_bars, lenient_bars):
        for bar in bars:
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels)
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("pooled value")
    axes.set_title(
        "Strict grammar vs salvaged content, same ground truth\n"
        f"the gap is the cost of grammar failures  --  {context.model_label}",
        fontsize=10,
    )
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "strict_vs_lenient_f1", dpi=context.dpi)


def _plot_entity_recall(rows: list[dict[str, Any]], context: TestContext) -> list[Path]:
    """Lenient recall by entity type, ordered by ground-truth frequency.

    Ordering by frequency rather than by recall is deliberate: it turns the chart
    into a readable statement about the head-versus-tail of the entity
    distribution, which is the thing a reader actually wants to know.
    """

    top = [row for row in rows if row["ground_truth_events"] > 0][:TOP_ENTITY_TYPES]
    if not top:
        return []
    labels = [f"{row['entity_type']} ({row['ground_truth_events']})" for row in top]
    values = [float(row["recall"]) for row in top]
    height = max(3.0, 0.28 * len(top) + 1.2)
    figure, axes = plt.subplots(figsize=(8.0, height))
    positions = range(len(top))
    axes.barh(list(positions), values, color="#3B6EA5")
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels, fontsize=8)
    axes.invert_yaxis()  # most frequent type at the top
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("recall (fraction of ground-truth events matched, lenient scoring)")
    axes.set_title(
        "Recall by entity type, most frequent first\n"
        "(count in parentheses = ground-truth events pooled over scored windows)",
        fontsize=10,
    )
    axes.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "per_entity_type_recall", dpi=context.dpi)
