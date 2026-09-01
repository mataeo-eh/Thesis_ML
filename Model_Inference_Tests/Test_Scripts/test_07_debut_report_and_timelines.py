"""Debut/build-order fine-tuning report on the held-out split. FINE-TUNED MODELS ONLY.

Question this answers
---------------------
"For a model fine-tuned on the debut task, how well does it call the outcome and
place each entity type's FIRST APPEARANCE in time, on games it has never seen?"

Why this test is gated
----------------------
The debut task is a different objective from pre-training. A pre-training
checkpoint (``data.debut_mode = false``) produces a full enemy reconstruction +
future roll-out canvas; a fine-tuned checkpoint (``data.debut_mode = true``)
produces the sparse debut canvas this report scores, under the RELAXED debut
grammar (``inference.decode.validate_debut_canvas``) and a different loss-class
taxonomy.

Running debut metrics against pre-training weights would not fail loudly -- it
would produce numbers, and they would be meaningless. So this module declares
``REQUIRES_DEBUT_FINETUNE = True`` and the runner SKIPS it, with the reason
stated, whenever the checkpoint under test was trained with
``data.debut_mode = false``.

Method
------
Delegates to the project's own fine-tuning evaluation, ``eval.finetune_report``,
which the fine-tuning pipeline uses to write `finetune_report.json`. Only the
"test" section is produced here -- the "memorized" section by definition needs the
replays the model was fine-tuned on, which is the opposite of what this directory
scores. Additionally renders the first-appearance timelines from
``viz.diagnostics`` for a handful of windows: one row per entity type, with the
ground-truth and predicted debut markers and the configured timing-tolerance band.

Depends on: ``eval.finetune_report.build_debut_report``,
``viz.diagnostics.evaluate_selected`` / ``render_figures``, ``inference_test_api``.
"""

from __future__ import annotations

from pathlib import Path

from thesis_ml.eval.finetune_report import build_debut_report
from thesis_ml.viz.diagnostics import evaluate_selected, render_figures

from inference_test_api import TestContext, TestResult, portable_path, write_json


TEST_NAME = "debut_report_and_timelines"
TEST_TITLE = "Debut build-order report and first-appearance timelines"
TEST_DESCRIPTION = """
Scores a DEBUT FINE-TUNED checkpoint on the held-out test replays with the
project's fine-tuning evaluation: win/loss accuracy overall and by how far into
the game the input window reaches, debut build-order precision/recall/F1 broken
out by fog class and fog bucket, debut timing mean-absolute-error in buckets, and
grammar validity under the relaxed debut grammar. Also renders per-window
first-appearance timelines showing each entity type's ground-truth versus
predicted debut with the timing-tolerance band.

**Gated.** This test only applies to a checkpoint trained with
`data.debut_mode = true`. On a pre-training checkpoint the runner skips it and
says so, because the debut metrics would silently produce meaningless numbers
rather than failing.
""".strip()
TEST_OUTPUTS = (
    "`debut_report_test_split.json` -- the `finetune_report` test section: win/loss accuracy, per-minute-bucket outcome accuracy, debut build-order F1 (aggregate, by fog class, by fog bucket), debut timing MAE, and grammar validity",
    "`first_appearance_<window>.png` / `.svg` -- per-window first-appearance timeline: one row per entity type with ground-truth and predicted debut markers and the tolerance band",
    "`diagnostics.pdf` -- the rendered timelines and count comparisons collected into one multi-page PDF",
)
USES_MODEL = True
# The gate. See the module docstring for why this is not merely advisory.
REQUIRES_DEBUT_FINETUNE = True

# Timelines are read one at a time and each costs a full sampler run, so only a
# few are rendered. Override with `--option debut_timeline_windows=N`.
DEFAULT_TIMELINE_WINDOWS = 6

# Windows scored for the metrics section. Override with `--option debut_windows=N`.
DEFAULT_REPORT_WINDOWS = 40


def run(context: TestContext) -> TestResult:
    """Build the debut report and render first-appearance timelines.

    Parameters:
        context: runner-supplied context. Only reached when the checkpoint is
            debut fine-tuned -- the runner enforces that gate.

    Returns:
        A :class:`TestResult` with the headline debut metrics.

    Calls: ``eval.finetune_report.build_debut_report``,
    ``viz.diagnostics.evaluate_selected`` and ``render_figures``.
    """

    report_windows = context.option_int("debut_windows", DEFAULT_REPORT_WINDOWS)
    timeline_windows = context.option_int("debut_timeline_windows", DEFAULT_TIMELINE_WINDOWS)
    if context.max_examples > 0:
        report_windows = min(report_windows, context.max_examples)

    model, run_config = context.shared.model()
    vocabulary = context.shared.vocabulary()

    if not run_config.data.debut_mode:
        # Defence in depth. The runner already gates on the checkpoint's stored
        # debut_mode; this catches a config/checkpoint mismatch that would make
        # the dataset serve the wrong canvas taxonomy.
        raise ValueError(
            "the resolved run config has data.debut_mode=false, so the dataset would "
            "serve pre-training canvases; refusing to compute debut metrics on them"
        )

    examples = context.shared.examples(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=report_windows,
        run_config=run_config,
    )
    print(f"scoring {len(examples)} held-out window(s) with the debut report")

    section = build_debut_report(
        examples,
        label="test",
        model=model,
        vocabulary=vocabulary,
        config=run_config,
        device=context.device,
    )

    written: list[Path] = [
        write_json(
            {
                "provenance": context.provenance(uses_model=True),
                "note": "the 'test' section only; the 'memorized' section requires the fine-tune replays",
                "test": section,
            },
            context.out_dir / "debut_report_test_split.json",
        )
    ]

    # A separate, smaller selection for the timelines: they are figures to read,
    # not a sample to average.
    timeline_examples = examples[:timeline_windows]
    print(f"rendering {len(timeline_examples)} first-appearance timeline(s)")
    rendered = evaluate_selected(
        model,
        timeline_examples,
        vocabulary,
        run_config,
        device=context.device,
    )
    written.extend(
        render_figures(
            rendered,
            context.out_dir,
            tolerance_buckets=run_config.eval.timing_tolerance_buckets,
            dpi=context.dpi,
            include_first_appearance=True,
        )
    )

    for path in written:
        print(f"  wrote {portable_path(path)}")

    aggregate = section.get("build_order_f1", {}).get("aggregate", {})
    return TestResult(
        headline=[
            f"win/loss accuracy on {section.get('example_count')} held-out windows: "
            f"{float(section.get('win_loss_accuracy', 0.0)):.4f}",
            f"debut build-order F1: {float(aggregate.get('f1', 0.0)):.4f}",
            f"debut timing MAE: {section.get('debut_mae')} bucket(s) over "
            f"{section.get('debut_mae_matched_count')} matched events; "
            f"grammar validity {float(section.get('grammar_validity', 0.0)):.4f}",
        ],
        artifacts=written,
        metrics={
            "example_count": section.get("example_count"),
            "win_loss_accuracy": section.get("win_loss_accuracy"),
            "debut_build_order_f1": aggregate.get("f1"),
            "debut_mae": section.get("debut_mae"),
            "grammar_validity": section.get("grammar_validity"),
        },
    )
