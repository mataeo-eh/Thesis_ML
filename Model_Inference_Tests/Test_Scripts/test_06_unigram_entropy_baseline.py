"""The no-model floor: what a constant predictor scores on the same held-out split.

Question this answers
---------------------
"Is the model's held-out cross-entropy actually good?" A loss number on its own
cannot answer that. This test computes what a predictor with NO knowledge of the
input replay -- just the best fixed distribution over canvas tokens -- achieves
on the identical held-out windows, under the identical live loss mask and class
weighting.

The difference between that floor and the number in
`test_03_heldout_canvas_cross_entropy` is the model's actual contribution. If the
two are close, the model is reproducing the token marginal rather than reading
the game.

Method
------
Thin wrapper over the existing `scripts/canvas_unigram_baseline.py`, run with
``--split test``. That script serves manifest windows through the production
``SC2DiffusionDataset`` + ``collate_diffusion_examples`` path, so its scored
positions are exactly the live ``canvas_loss_mask`` positions -- clamped `[BOS]`
and batch-shape padding excluded, semantic `[PAD]` still scored. It reports the
empirical unigram entropy H(p) and the best constant predictor under the live
class weights, overall and for each of the seven loss classes.

This is the ONE test in this directory that never loads the checkpoint. It is CPU
only, uses no model forward pass, and its result depends solely on the data --
which is precisely what makes it a valid reference.

Depends on: ``scripts.canvas_unigram_baseline.compute_baseline`` /
``format_summary``, ``inference_test_api``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.canvas_unigram_baseline import compute_baseline, format_summary

from inference_test_api import TestContext, TestResult, portable_path, write_json


TEST_NAME = "unigram_entropy_baseline"
TEST_TITLE = "Constant-predictor entropy floor on the held-out split"
TEST_DESCRIPTION = """
Computes the data-only baseline for the SAME held-out test windows the model is
scored on: the empirical unigram entropy of the canvas targets and the loss of
the best constant predictor under the live weighted objective, overall and per
loss class.

Uses no model and no GPU. Its purpose is to make every cross-entropy number in
this run interpretable -- the gap between this floor and the model's held-out
loss is the model's actual contribution over knowing nothing but the token
marginal. A rare class whose model loss sits at its floor has not been learned,
regardless of how small the absolute number looks.
""".strip()
TEST_OUTPUTS = (
    "`unigram_baseline.json` -- unweighted and live-weighted constant-predictor baselines, overall and for all seven loss classes, with config / manifest / token-dictionary hashes",
    "`unigram_baseline.summary.txt` -- the script's own compact human summary: scored-position counts, per-class conditional entropy, CE under the global weighted optimum, and each class's contribution to the objective",
    "`baseline_context.json` -- this run's provenance plus the headline floor values, so the numbers can be lined up against the model's held-out loss",
)
# Data-only: the runner never opens the checkpoint for this test.
USES_MODEL = False
REQUIRES_DEBUT_FINETUNE = False

# The full test split is ~1,100 windows and this is a CPU streaming scan, so it
# is bounded by default to keep the run time reasonable. 0 means every window;
# override with `--option baseline_windows=N`.
DEFAULT_MAX_WINDOWS = 400


def run(context: TestContext) -> TestResult:
    """Compute the constant-predictor floor for the held-out split.

    Parameters:
        context: runner-supplied context. Only the config, output directory, and
            window budget are used -- no model is loaded.

    Returns:
        A :class:`TestResult` carrying the headline floor values.

    Calls: ``scripts.canvas_unigram_baseline.compute_baseline`` and
    ``format_summary``.
    """

    max_windows = context.option_int("baseline_windows", DEFAULT_MAX_WINDOWS)

    # num_workers 0 keeps this single-process: a measurement tool should not
    # spawn workers that contend with whatever else is running on the machine,
    # and the scan is fast enough without them.
    print(f"scanning the test split (max_windows={max_windows or 'all'}) on CPU")
    report: dict[str, Any] = compute_baseline(
        context.shared.user_config,
        config_path=context.shared.config_path,
        split_name="test",
        max_windows=max_windows,
        manifest_filters=(),
        dataset_epoch=0,
        num_workers=0,
        include_position_conditional=True,
    )

    written: list[Path] = [write_json(report, context.out_dir / "unigram_baseline.json")]

    summary_text = format_summary(report)
    summary_path = context.out_dir / "unigram_baseline.summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    written.append(summary_path)
    print(summary_text, end="")

    overall = report.get("overall", {})
    provenance_block = {
        # uses_model=False: the checkpoint is deliberately never opened here, so
        # this block records the split and budget only.
        "provenance": context.provenance(uses_model=False),
        "note": (
            "Data-only floor. Compare against heldout_canvas_cross_entropy: the gap "
            "between the two is the model's contribution over the token marginal."
        ),
        "headline": {
            key: overall.get(key)
            for key in (
                "scored_positions",
                "unique_target_tokens",
                "unweighted_marginal_entropy_nats",
                "position_conditional_unweighted_entropy_nats",
                "weighted_optimal_constant_ce_nats",
                "weighted_ce_of_unweighted_marginal_nats",
            )
            if key in overall
        },
    }
    written.append(write_json(provenance_block, context.out_dir / "baseline_context.json"))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    headline = [
        f"scored positions: {overall.get('scored_positions', 'n/a')}",
    ]
    for key, label in (
        ("unweighted_marginal_entropy_nats", "unigram entropy H(p)"),
        (
            "weighted_optimal_constant_ce_nats",
            "best constant predictor under the live weighted objective",
        ),
    ):
        if key in overall:
            headline.append(f"{label}: {float(overall[key]):.6f} nats")

    return TestResult(
        headline=headline,
        artifacts=written,
        metrics={
            "scored_positions": overall.get("scored_positions"),
            "unweighted_marginal_entropy_nats": overall.get("unweighted_marginal_entropy_nats"),
            "weighted_optimal_constant_ce_nats": overall.get("weighted_optimal_constant_ce_nats"),
        },
    )
