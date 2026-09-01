"""Visual prediction-vs-truth diagnostics for individual held-out windows.

Question this answers
---------------------
"Show me, for a game the model has never seen, what it actually predicted next
to what really happened." This is the eyeball test: per-window entity-count
heatmaps of ground truth, prediction, and their signed difference, plus the raw
token-by-token comparison behind them.

Method
------
Thin wrapper over the existing ``thesis_ml.viz.diagnostics`` module. For each
selected held-out window it runs the full evaluation path -- iterative
entropy-bounded sampling (``inference.sampler.sample_canvas``), grammar
validation and decode (``inference.decode``), and the build-order oracle on BOTH
the prediction and the ground-truth canvas (``eval.harness.evaluate_example``) --
then renders the module's own figures. No plotting or sampling logic is
reimplemented here; this script only chooses WHICH windows are scored (held-out
test replays) and WHERE the artifacts land.

Cost note
---------
Iterative sampling runs up to ``sampler.max_steps`` forward passes per window, so
this test deliberately caps itself at a handful of windows (``DEFAULT_MAX_WINDOWS``,
overridable with ``--option figure_windows=N``). The at-scale metrics live in
``test_02_build_order_accuracy``.

Depends on: ``viz.diagnostics.evaluate_selected`` / ``render_figures`` /
``write_canvas_comparison_csv_files`` / ``write_input_canvas_text_files`` /
``write_logits_json``, and ``inference_test_api``.
"""

from __future__ import annotations

from pathlib import Path

from thesis_ml.viz.diagnostics import (
    evaluate_selected,
    render_figures,
    write_canvas_comparison_csv_files,
    write_input_canvas_text_files,
    write_logits_json,
)

from inference_test_api import TestContext, TestResult, portable_path, write_json


TEST_NAME = "canvas_reconstruction_figures"
TEST_TITLE = "Per-window prediction vs ground-truth canvas figures"
TEST_DESCRIPTION = """
Renders, for a small number of individual held-out test windows, the aligned
ground-truth / predicted / signed-error entity-count heatmaps produced by
`thesis_ml.viz.diagnostics`, plus an aggregate mean-absolute-difference heatmap
across those windows. Alongside the figures it exports the position-by-position
token comparison (CSV), the exact model input sequence with self/enemy markers
(text), and the final-canvas top-k logits and softmax confidences (JSON).

This is the qualitative test: it shows WHERE in the canvas the model is right or
wrong (which timestep buckets, which entity types) rather than reducing that to a
single score. Sampling is the real iterative sampler, so what is rendered is what
the deployed model would emit.
""".strip()
TEST_OUTPUTS = (
    "`prediction_vs_truth_<window>.png` / `.svg` -- ground-truth, predicted, and signed-error entity-count heatmaps for one window, with a match summary",
    "`mean_abs_diff_aggregate.png` / `.svg` -- mean absolute count error per (entity type, timestep bucket) across every rendered window",
    "`diagnostics.pdf` -- every figure above collected into one multi-page PDF",
    "`canvas_comparison.csv` -- per canvas position: predicted token name, ground-truth token name, and whether it matched",
    "`input_canvas.txt` -- the exact input sequence fed to the model, each token tagged SELF / ENEMY",
    "`canvas_logits.json` -- final-canvas top-k raw logits and softmax confidence at every canvas position",
    "`figure_metrics.json` -- per-window build-order precision/recall/F1 and grammar validity for the rendered windows",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False

# Iterative sampling is expensive (up to sampler.max_steps forward passes per
# window), and these figures are meant to be read one at a time rather than
# skimmed by the hundred. Six windows is enough to show variation across replays
# without turning the run into an overnight job.
DEFAULT_MAX_WINDOWS = 6

# Output-canvas noise rate. 1.0 is the terminal prior: the entire canvas starts
# as noise and every rendered position is genuinely model-predicted, which is the
# honest condition for "what does the model generate for an unseen game". Lower
# values reveal part of the ground truth as an infill prompt.
DEFAULT_OUTPUT_NOISE = 1.0


def run(context: TestContext) -> TestResult:
    """Render the diagnostic figures for a few held-out windows.

    Parameters:
        context: runner-supplied context (model, split, output directory).

    Returns:
        A :class:`TestResult` with the headline match numbers and every written
        path.

    Calls: ``SharedResources.model``, ``SharedResources.examples``,
    ``viz.diagnostics.evaluate_selected`` and the diagnostics writers.
    """

    max_windows = context.option_int("figure_windows", DEFAULT_MAX_WINDOWS)
    output_noise = context.option_float("figure_output_noise", DEFAULT_OUTPUT_NOISE)

    model, run_config = context.shared.model()
    vocabulary = context.shared.vocabulary()

    # A small sample spread across several replays beats many windows from one
    # game: the point of the figures is to show how behaviour varies between
    # unseen matches.
    examples = context.shared.examples(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=max_windows,
        run_config=run_config,
    )
    print(f"rendering {len(examples)} held-out window(s) at output noise t={output_noise:.2f}")

    rendered = evaluate_selected(
        model,
        examples,
        vocabulary,
        run_config,
        device=context.device,
        include_canvas_logits=True,
        bypass_sampler=False,
        noise_rate=output_noise,
    )

    written: list[Path] = []
    written.extend(
        render_figures(
            rendered,
            context.out_dir,
            tolerance_buckets=run_config.eval.timing_tolerance_buckets,
            dpi=context.dpi,
            # First-appearance timelines are only meaningful for a debut
            # fine-tuned model; this test runs on pre-training checkpoints too.
            include_first_appearance=False,
        )
    )
    written.extend(write_input_canvas_text_files(rendered, context.out_dir))
    written.extend(write_canvas_comparison_csv_files(rendered, vocabulary, context.out_dir))
    written.append(write_logits_json(rendered, vocabulary, context.out_dir))

    # Per-window numbers so the figures are not the only record: a reader can
    # sort by F1 to find the interesting windows without opening every PNG.
    per_window = []
    for item in rendered:
        metrics = item.result.metrics
        per_window.append(
            {
                "window": item.label,
                "replay_id": item.example.replay_id,
                "perspective": item.example.perspective_player,
                "window_start": item.example.window_start,
                "prediction_valid": item.result.prediction_valid,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "accuracy": metrics.accuracy,
                "true_positives": metrics.true_positives,
                "predicted_events": metrics.predicted_count,
                "ground_truth_events": metrics.ground_truth_count,
            }
        )

    valid_count = sum(1 for row in per_window if row["prediction_valid"])
    mean_f1 = (
        sum(float(row["f1"]) for row in per_window) / len(per_window) if per_window else 0.0
    )
    payload = {
        "provenance": context.provenance(uses_model=True),
        "output_noise": output_noise,
        "sampler": "iterative (inference.sampler.sample_canvas)",
        "n_windows": len(per_window),
        "grammar_valid_windows": valid_count,
        "mean_f1": mean_f1,
        "windows": per_window,
    }
    written.append(write_json(payload, context.out_dir / "figure_metrics.json"))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    return TestResult(
        headline=[
            f"{len(per_window)} held-out window(s) rendered at t={output_noise:.2f}",
            f"grammar-valid canvases: {valid_count}/{len(per_window)}",
            f"mean per-window build-order F1: {mean_f1:.4f}",
        ],
        artifacts=written,
        metrics={
            "n_windows": len(per_window),
            "grammar_valid_windows": valid_count,
            "mean_f1": mean_f1,
            "output_noise": output_noise,
        },
    )
