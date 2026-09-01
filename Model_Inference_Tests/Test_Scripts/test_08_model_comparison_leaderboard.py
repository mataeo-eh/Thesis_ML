"""The one-number cross-model comparison: held-out bits per token.

Question this answers
---------------------
"Of every model I have trained, which one actually learned the most about
StarCraft II replays it has never seen?"

Every other test in this directory characterises ONE checkpoint in depth. This
one exists to produce a single scalar per checkpoint that is directly comparable
against every other checkpoint scored the same way, and to accumulate those
scalars into one growing CSV so the comparison survives across runs, days, and
architectures.

The scalar is `bits_per_token`: the average number of yes/no questions the model
still needs to identify the true canvas token, given everything it saw. It is the
same quantity the training loop writes into `epoch_metrics.csv`, computed here on
the held-out TEST split through the identical code path. Its exponent form,
`effective_token_choices` (= 2 ** bits_per_token), is the number of
equally-likely tokens the model is effectively deciding between at each scored
position -- 1.0 meaning "it already knows", the vocabulary size meaning "it is
guessing".

That second number is the SAME ARITHMETIC an autoregressive LM's perplexity uses,
and it plays the same role in a comparison: run two models over the same held-out
data and the lower value is the one that was less confused by it. It is
deliberately NOT called perplexity. Perplexity is the exponentiated per-token
negative log-likelihood of a sequence under a model that factorises that
sequence's probability -- which a discrete-diffusion denoiser does not do. What
is measured here is a per-position cross entropy conditioned on a partially
corrupted canvas, averaged over a corruption distribution; there is no sequence
likelihood behind it, so reporting it under the name "perplexity" would invite a
comparison against published LM perplexities that is not valid arithmetic. See
"What makes two rows comparable" below.

Why bits and not the loss column
--------------------------------
`train_loss` / `dev_loss` are the class-WEIGHTED training objective
(`config.loss.class_loss_weights` boosts `[END]` and damps `[PAD]`). Two models
trained under different weights produce loss numbers that are not on the same
scale, and exponentiating a weighted loss is not an entropy of any
distribution. `bits_per_token` comes from a separate UNWEIGHTED cross-entropy
accumulator inside the loss module, which is exactly what makes it portable
between runs. See ``train.loop._finalize_bits_per_token``.

Method
------
Reuses ``TrainingLoop.validate`` -- the routine that produced the run's own
`dev_loss` and `bits_per_token` columns -- against a measurement-only
``TrainingLoop`` built around the loaded checkpoint. Nothing is trained, no
optimizer step is taken, no checkpoint is written. Identical in construction to
`test_03_heldout_canvas_cross_entropy`; this test differs in what it does with
the result, not in how it obtains it.

Three scores are recorded per checkpoint:

  * **`bits_per_token`** -- at the run's training t-distribution. This is the
    headline, and the number directly comparable to the run's own CSV column.
    Because that distribution is dominated by lightly-corrupted examples whose
    canvas tokens are mostly already correct, it is the right number for
    "did training progress" and a flattering one for "does this model
    understand the game".
  * **`bits_per_token_t_1.00`** and its `effective_token_choices_t_1.00`
    partner -- the same measurement at t = 1.0, a canvas that is ENTIRELY noise.
    This is the condition the deployed sampler actually starts from, so it is
    the model doing the job it is asked to do at inference, with nothing to lean
    on but the input replay. It is always scored, even when an overridden grid
    leaves t = 1.0 out. Expect it to be far worse than the headline pair; that
    gap is the measurement, not a defect.
  * **`bits_per_token_uniform_t`** -- the mean over a FIXED grid of corruption
    levels (default t = 0.25 / 0.5 / 0.75 / 1.0). Because the grid is hardcoded
    here rather than read from the config, this score stays comparable even
    between two models whose `diffusion.schedule` differs, which the headline
    does not. Use it whenever the `t_*` columns of two rows disagree.

What makes two rows comparable
------------------------------
Everything that changes the scored positions, not just the model. The row carries
all of it explicitly -- split, fog condition, window budget, seed, vocabulary
size, the full t-schedule, and `window_selection_key` (a hash of WHICH windows
were scored, since two runs can both score 240 windows and score 240 different
ones) -- plus `eval_condition_key`, a short hash over exactly those fields.
**Two rows are comparable if and only if their `eval_condition_key` matches.**
Sort the CSV by that column first, then by `bits_per_token` within it.

Reproducibility
---------------
Re-scoring one checkpoint under one condition must give bit-identical numbers, or
none of the comparisons above mean anything. Two things are done to guarantee it,
both easy to get wrong:

  * **The corruption generator is reseeded before EVERY pass**, so a level's
    score never depends on how many passes preceded it.
  * **Every pass gets a freshly-built loader.** ``SC2DiffusionDataset`` draws fog
    per SERVING -- ``_rng_for_index`` mixes a per-window serve counter into its
    seed sequence, so handing the same window out twice yields different fog.
    That is right for training, where fog is a transient view resampled each
    epoch, and fatal for a measurement: a reused loader would make pass 2 score
    different inputs than pass 1, so a level's number would depend on its
    position in the pass order. Rebuilding the loader restarts the serve counter,
    giving every pass serving=0 and byte-identical inputs, with t as the only
    difference between them.

Output
------
Unlike every other test here, the primary artifact is NOT confined to this run's
own subdirectory. One row is appended to a single shared CSV that lives at the
root of `output/` and grows across every invocation:

    Model_Inference_Tests/output/model_comparison_bits_per_token.csv

Re-running the same checkpoint appends another row rather than replacing the old
one; rows are distinguished by `run_datetime`. Identical repeat rows are
identifiable by their `dedupe_key`, which is equal exactly when the model, the
architecture, the evaluation condition, and the score are all the same -- sort by
it and delete the extras.

Depends on: ``train.loop.TrainingLoop.validate``, ``inference_test_api``.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from thesis_ml.train.loop import TrainingLoop

from inference_test_api import (
    TestContext,
    TestResult,
    portable_path,
    write_csv,
    write_json,
)


TEST_NAME = "model_comparison_leaderboard"
TEST_TITLE = "Cross-model comparison: held-out bits per token"
TEST_DESCRIPTION = """
Scores the checkpoint's held-out bits-per-token -- the unweighted cross entropy
per scored canvas position, and the closest well-defined analogue of the number
an LLM's perplexity reports -- and APPENDS it as one row to a single shared CSV
at the root of `output/` that accumulates across every run of this suite.

This is the suite's one-number model comparison. Each row carries the score, the
architecture that produced it (shape, parameter count, toggles), the checkpoint
it came from, the date-time it was measured, and the full evaluation condition
hashed into `eval_condition_key`. Rows sharing an `eval_condition_key` are
directly comparable and can be ranked by `bits_per_token`; rows that do not share
one were measured on different scored positions and must not be ranked together.

Each score is paired with an `effective_token_choices` column -- `2 ** bits`,
read as "how many equally-likely tokens is the model effectively choosing
between". The `_t_1.00` pair is the same measurement at the corruption level the
deployed sampler starts from, so it is the one that describes inference-time
understanding rather than training progress.

These are not called perplexity on purpose: a discrete-diffusion denoiser assigns
no sequence likelihood, so the name would invite a comparison against published
LM perplexities that the quantity does not support.
""".strip()
TEST_OUTPUTS = (
    "`output/model_comparison_bits_per_token.csv` -- **shared, appended, at the root of `output/` (not in this run's subdirectory)**: one row per scored checkpoint, accumulating across every run of this suite",
    "`leaderboard_row.csv` -- this run's row alone, so the run directory stays self-contained",
    "`bits_per_token.json` -- the full measurement: provenance, all three scores, the inference-condition pair, the per-t grid, and the secondary token metrics behind the row",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False


# The shared, cross-run CSV. Deliberately at the root of `output/` rather than
# inside a per-run subdirectory: its whole purpose is to outlive any single run
# and be the one file you open to compare models. It is still git-ignored
# runtime state and is fully regenerable by re-running this test per checkpoint.
LEADERBOARD_FILENAME = "model_comparison_bits_per_token.csv"

# Fixed corruption grid behind `bits_per_token_uniform_t`. Hardcoded ON PURPOSE:
# the headline score is taken at the run's own configured t-distribution, which
# means two models trained under different schedules are not strictly comparable
# on it. This grid does not come from any config, so the score averaged over it
# is comparable between any two checkpoints regardless of their schedules. 1.0 is
# the terminal prior (canvas entirely noise) the deployed sampler starts from.
DEFAULT_T_GRID = (0.25, 0.5, 0.75, 1.0)

# The corruption level the DEPLOYED SAMPLER actually starts from: a canvas that
# is entirely noise. Scored on its own, beside the headline, and always -- even
# when an overridden `leaderboard_t_grid` leaves it out -- because it is the only
# level that answers "how well does this model do the job it is asked to do at
# inference". The headline at the training t-distribution is dominated by
# lightly-corrupted examples whose canvas tokens are already correct, so it is
# the right number for comparing training progress and the wrong one for
# comparing inference-time understanding.
INFERENCE_T = 1.0

# Windows scored per pass. Each pass is one forward pass per batch, so this is
# cheap relative to the sampler-bound tests and can afford a wider sample than
# they do -- and a comparison number deserves the tighter estimate. Override with
# `--option leaderboard_windows=N`.
DEFAULT_MAX_WINDOWS = 240

# Fog condition. `None` = draw per serving from `config.fog.rate_distribution`,
# i.e. the visibility the model actually TRAINED under. Matching
# `test_03_heldout_canvas_cross_entropy` and `test_06_unigram_entropy_baseline`
# keeps this number readable against the entropy floor those tests establish.
# The draw is seeded from `config.pipeline.seed` over a fixed window selection,
# so "per serving" is still fully reproducible across models. Pin a fixed rate
# with `--option leaderboard_fog=0.0`.
DEFAULT_FOG_RATE_OVERRIDE = None

# Fields hashed into `eval_condition_key`. Everything here changes WHICH
# positions get scored or HOW they are corrupted, so a difference in any one of
# them makes two scores incomparable. The model itself is deliberately absent --
# that is the thing the key is meant to hold constant while it varies.
EVAL_CONDITION_FIELDS = (
    "config",
    "split_source",
    "n_replays_in_split",
    "n_windows",
    "window_selection_key",
    "fog_condition",
    "seed",
    "batch_size",
    "vocab_size",
    "diffusion_process",
    "t_schedule",
    "t_distribution",
    "t_distribution_power",
    "t_min",
    "t_max",
    "t_one_fraction",
    "t_grid",
)


def run(context: TestContext) -> TestResult:
    """Score the checkpoint's held-out bits-per-token and append it to the shared CSV.

    Parameters:
        context: runner-supplied context (model, split, output directory, budget).

    Returns:
        A :class:`TestResult` whose headline is the comparison number itself.

    Calls: ``SharedResources.model`` / ``dataloader`` / ``vocabulary``,
        ``TrainingLoop.validate``, :func:`_build_row`,
        :func:`append_leaderboard_row`.
    """

    max_windows = context.option_int("leaderboard_windows", DEFAULT_MAX_WINDOWS)
    if context.max_examples > 0:
        max_windows = min(max_windows, context.max_examples)
    t_grid = _parse_t_grid(context)
    fog_override = (
        context.option_float("leaderboard_fog", float("nan"))
        if "leaderboard_fog" in context.extra
        else DEFAULT_FOG_RATE_OVERRIDE
    )

    model, run_config = context.shared.model()
    vocabulary = context.shared.vocabulary()

    # Measurement-only loop: no metrics path, no publisher, no optimizer step.
    # `validate` scores with `ema_model`, which the constructor deep-copies from
    # the model handed in -- i.e. exactly the weights loaded from the checkpoint.
    loop = TrainingLoop(
        model=model,
        config=run_config,
        device=context.device,
        seed=context.seed,
    )

    # A FRESH loader per pass, not one loader reused across passes. This is not
    # a style choice -- it is required for the numbers to be reproducible.
    #
    # `SC2DiffusionDataset` draws fog per SERVING: `_rng_for_index` mixes a
    # per-index serve counter into the seed sequence, so the second time a window
    # is handed out it gets different fog than the first. That is correct for
    # training (fog is a transient view, resampled every epoch) and fatal here:
    # reusing one loader would mean pass 2 scores different inputs than pass 1,
    # and a level's score would depend on HOW MANY PASSES RAN BEFORE IT rather
    # than only on t. Two runs with different `leaderboard_t_grid` values would
    # then disagree on a shared level.
    #
    # Rebuilding the loader constructs a new dataset with an empty serve counter,
    # so every pass sees serving=0 -- byte-identical inputs, with t as the only
    # thing that differs between them.
    def fresh_loader():
        """Build a loader whose per-window fog draws are the FIRST serving.

        Returns:
            ``(loader, dataset_indices)`` exactly as ``SharedResources.dataloader``
            does. Called once per scoring pass.
        """

        return context.shared.dataloader(
            n_replays=context.n_replays,
            n_windows_per_replay=context.n_windows_per_replay,
            max_examples=max_windows,
            run_config=run_config,
            fog_rate_override=fog_override,
        )

    loader, indices = fresh_loader()
    selected_windows = _selected_windows(loader)
    fog_label = (
        "training distribution (per serving)"
        if fog_override is None
        else f"fixed rate {fog_override}"
    )
    print(
        f"scoring {len(indices)} held-out window(s) "
        f"in batches of {run_config.pipeline.batch_size}; fog = {fog_label}"
    )

    # Pass 1 -- the headline, at the run's own training t-distribution. The
    # generator is reseeded first so the corruption draws depend only on the
    # seed, never on how many passes happened to run before this one.
    print("pass: training t-distribution (the headline comparison number)")
    loop.generator.manual_seed(context.seed)
    headline_log = loop.validate(loader)
    del loader  # every later pass builds its own; see fresh_loader above
    if headline_log.bits_per_token is None:
        raise RuntimeError(
            "the held-out pass scored zero positions, so there is no bits-per-token "
            "to report; check the window selection and the loss mask"
        )
    print(
        f"  bits_per_token = {headline_log.bits_per_token:.6f}  "
        f"effective_token_choices = {headline_log.perplexity:.4f}"
    )

    # Passes 2..N -- the fixed levels. This is the schedule-independent grid plus
    # INFERENCE_T, which is scored whether or not the grid happens to contain it
    # so the inference-condition columns are never blank. Reseeded before each
    # pass so the only thing differing between levels is t itself.
    scored_levels = tuple(t_grid) + (
        () if INFERENCE_T in t_grid else (INFERENCE_T,)
    )
    grid_bits: dict[float, float] = {}
    for level in scored_levels:
        pass_loader, _ = fresh_loader()
        loop.generator.manual_seed(context.seed)
        grid_log = loop.validate(pass_loader, fixed_t=level)
        if grid_log.bits_per_token is None:
            raise RuntimeError(f"the fixed t={level} pass scored zero positions")
        grid_bits[level] = float(grid_log.bits_per_token)
        print(
            f"pass: fixed t = {level:.2f}  ->  bits_per_token = {grid_bits[level]:.6f}"
            f"  effective_token_choices = {_effective_choices(grid_bits[level]):.4f}"
        )

    # Averaged over the GRID only. INFERENCE_T is excluded when it is not part of
    # the grid, because this score's whole purpose is to be the same average for
    # every model -- folding in an extra level for some rows and not others would
    # destroy exactly the comparability it exists to provide.
    uniform_t_bits = sum(grid_bits[level] for level in t_grid) / len(t_grid)

    # The no-knowledge ceiling: a model that has learned nothing still needs
    # log2(V) bits to name a token out of V. Reporting the score against this
    # turns an absolute number into "what fraction of the naive cost did the
    # model remove", which is the form that reads at a glance across models.
    vocab_size = int(vocabulary.vocab_size)
    uniform_prior_bits = math.log2(vocab_size)
    inference_choices = _effective_choices(grid_bits[INFERENCE_T])
    # Measured AT THE INFERENCE CONDITION, not at the headline t-distribution.
    # At the training t-distribution most scored positions already hold the
    # correct token, so the model clears the uniform prior almost trivially and
    # this fraction reads ~99% for any half-trained model -- a number that
    # separates nothing. At t=1.0 the canvas carries no information at all, so
    # the gap to log2(V) is the model's genuine contribution from the input
    # replay alone.
    bits_saved = uniform_prior_bits - grid_bits[INFERENCE_T]

    row = _build_row(
        context=context,
        model=model,
        run_config=run_config,
        headline_log=headline_log,
        grid_bits=grid_bits,
        uniform_t_bits=uniform_t_bits,
        t_grid=t_grid,
        n_windows=len(indices),
        window_selection_key=_window_selection_key(selected_windows),
        fog_label=fog_label,
        vocab_size=vocab_size,
        uniform_prior_bits=uniform_prior_bits,
        bits_saved=bits_saved,
    )

    # `run_dir` is `output/<model label>__<date>`, so its parent is `output/`.
    leaderboard_path = context.run_dir.parent / LEADERBOARD_FILENAME
    append_leaderboard_row(row, leaderboard_path)

    written: list[Path] = [leaderboard_path]
    written.append(write_csv([row], list(row), context.out_dir / "leaderboard_row.csv"))
    written.append(
        write_json(
            {
                "provenance": context.provenance(
                    uses_model=True, fog_rate_override=fog_override
                ),
                "method": (
                    "TrainingLoop.validate with the loaded (EMA by default) weights -- the "
                    "same routine that produced the run's own bits_per_token column"
                ),
                "headline_metric": (
                    "bits_per_token: unweighted cross entropy per scored canvas position"
                ),
                "comparability": (
                    "rows of the shared CSV are comparable if and only if their "
                    "eval_condition_key matches; see EVAL_CONDITION_FIELDS"
                ),
                "leaderboard_csv": portable_path(leaderboard_path),
                "row": row,
                "training_t_distribution": {
                    "bits_per_token": float(headline_log.bits_per_token),
                    "effective_token_choices": float(headline_log.perplexity),
                    "weighted_loss": float(headline_log.loss),
                    "accuracy": {
                        name: float(value) for name, value in headline_log.accuracy.items()
                    },
                    "macro_f1": {
                        name: float(value) for name, value in headline_log.macro_f1.items()
                    },
                    "per_class": {
                        name: float(value) for name, value in headline_log.per_class.items()
                    },
                },
                "inference_condition": {
                    "t": INFERENCE_T,
                    "why": (
                        "the corruption level the deployed sampler starts from -- a canvas "
                        "that is entirely noise, so this is the model doing its actual job"
                    ),
                    "bits_per_token": grid_bits[INFERENCE_T],
                    "effective_token_choices": inference_choices,
                    "uniform_prior_bits": uniform_prior_bits,
                    "bits_removed_vs_uniform_prior": bits_saved,
                    "fraction_of_uniform_prior_removed": bits_saved / uniform_prior_bits,
                },
                "fixed_t_grid": {
                    f"{level:.2f}": {
                        "bits_per_token": grid_bits[level],
                        "effective_token_choices": _effective_choices(grid_bits[level]),
                    }
                    for level in sorted(grid_bits)
                },
                "uniform_prior": {
                    "vocab_size": vocab_size,
                    "bits": uniform_prior_bits,
                    "bits_removed_by_model": bits_saved,
                },
            },
            context.out_dir / "bits_per_token.json",
        )
    )

    for path in written:
        print(f"  wrote {portable_path(path)}")
    _print_leaderboard_tail(leaderboard_path)

    return TestResult(
        headline=[
            f"held-out bits per token: {headline_log.bits_per_token:.6f} "
            f"({headline_log.perplexity:.3f} effective token choices, "
            f"{len(indices)} windows)",
            f"at the inference condition t={INFERENCE_T:.2f}: "
            f"{grid_bits[INFERENCE_T]:.6f} bits/token "
            f"({inference_choices:.3f} effective token choices, "
            f"vs {vocab_size} for a uniform prior)",
            f"schedule-independent score over fixed t={_format_grid(t_grid)}: "
            f"{uniform_t_bits:.6f} bits/token",
            f"appended to {portable_path(leaderboard_path)} "
            f"(eval_condition_key {row['eval_condition_key']})",
        ],
        artifacts=written,
        metrics={
            "bits_per_token": float(headline_log.bits_per_token),
            "effective_token_choices": float(headline_log.perplexity),
            "bits_per_token_at_inference_t": grid_bits[INFERENCE_T],
            "effective_token_choices_at_inference_t": inference_choices,
            "bits_per_token_uniform_t": uniform_t_bits,
            "bits_per_token_by_t": {f"{level:.2f}": grid_bits[level] for level in t_grid},
            "fraction_of_uniform_prior_removed_at_inference_t": (
                bits_saved / uniform_prior_bits
            ),
            "n_windows": len(indices),
            "fog_condition": fog_label,
            "eval_condition_key": row["eval_condition_key"],
            "leaderboard_csv": portable_path(leaderboard_path),
        },
    )


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _build_row(
    *,
    context: TestContext,
    model: Any,
    run_config: Any,
    headline_log: Any,
    grid_bits: dict[float, float],
    uniform_t_bits: float,
    t_grid: tuple[float, ...],
    n_windows: int,
    window_selection_key: str,
    fog_label: str,
    vocab_size: int,
    uniform_prior_bits: float,
    bits_saved: float,
) -> dict[str, Any]:
    """Assemble the single CSV row describing this checkpoint and its score.

    The dict's INSERTION ORDER is the CSV's column order, grouped so the file
    reads left to right as: when it was run, what it scored, what the model is,
    where the weights came from, and under what conditions. Every value is a
    plain str/int/float so the row survives a CSV round trip unchanged.

    Parameters:
        context: the runner context (model label, run directory, seed, split).
        model: the loaded checkpoint, used only for its parameter count.
        run_config: the config with the checkpoint's own `model` section.
        headline_log: the ``ValidationLog`` from the training-t-distribution pass.
        grid_bits: bits-per-token keyed by fixed corruption level. Must contain
            :data:`INFERENCE_T`, whose pair of columns sits beside the headline.
        uniform_t_bits: mean of ``grid_bits``, the schedule-independent score.
        t_grid: the corruption levels behind ``grid_bits``, in order.
        n_windows: how many held-out windows were scored.
        window_selection_key: hash identifying WHICH windows those were.
        fog_label: human-readable fog condition.
        vocab_size: size of the output vocabulary.
        uniform_prior_bits: log2(vocab_size), the no-knowledge ceiling.
        bits_saved: how far below that ceiling the model scored.

    Returns:
        The row, ready for :func:`append_leaderboard_row`.

    Calls: :func:`_architecture_shape`, :func:`_eval_condition_key`,
        :func:`_dedupe_key`, :func:`_run_moment`.
    """

    model_config = run_config.model
    schedule = run_config.diffusion.schedule
    checkpoint = context.shared.checkpoint_facts()
    split = context.shared.test_split()
    moment = _run_moment(context.run_dir)

    params_total = sum(parameter.numel() for parameter in model.parameters())
    params_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    checkpoint_path = context.shared.checkpoint_path
    checkpoint_mb = (
        round(checkpoint_path.stat().st_size / (1024 * 1024), 1)
        if checkpoint_path.exists()
        else ""
    )

    # Built first because both the row and its comparability hash need it.
    condition: dict[str, Any] = {
        "config": portable_path(context.shared.config_path),
        "split_source": split.source,
        "n_replays_in_split": len(split.replay_ids),
        "n_windows": n_windows,
        "window_selection_key": window_selection_key,
        "fog_condition": fog_label,
        "seed": context.seed,
        "batch_size": int(run_config.pipeline.batch_size),
        "vocab_size": vocab_size,
        "diffusion_process": str(run_config.diffusion.process),
        "t_schedule": str(schedule.name),
        "t_distribution": str(schedule.t_distribution),
        "t_distribution_power": float(schedule.t_distribution_power),
        "t_min": float(schedule.min),
        "t_max": float(schedule.max),
        "t_one_fraction": float(schedule.t_one_fraction),
        "t_grid": _format_grid(t_grid),
    }
    condition_key = _eval_condition_key(condition)
    arch_shape = _architecture_shape(model_config)

    row: dict[str, Any] = {
        # -- when and on what --------------------------------------------------
        "run_datetime": moment.strftime("%Y-%m-%d %H:%M:%S"),
        "run_label": context.run_dir.name,
        "model_label": context.model_label,
        # -- the comparison number --------------------------------------------
        "bits_per_token": _round(headline_log.bits_per_token),
        # 2 ** bits_per_token. Named for what it measures rather than
        # "perplexity": the source field on ValidationLog carries that name, but
        # a discrete-diffusion denoiser assigns no sequence likelihood, so
        # publishing the column as a perplexity would invite a comparison against
        # LM perplexities that the quantity does not support.
        "effective_token_choices": _round(headline_log.perplexity, 4),
        # -- the same pair at the INFERENCE condition --------------------------
        # Placed immediately beside the headline because these are the numbers
        # that describe the model doing its actual job: denoising a canvas that
        # is entirely noise, which is where the deployed sampler begins. Expect
        # them to be much worse than the headline pair -- that gap is the point,
        # not a defect.
        f"bits_per_token_t_{INFERENCE_T:.2f}": _round(grid_bits[INFERENCE_T]),
        f"effective_token_choices_t_{INFERENCE_T:.2f}": _round(
            _effective_choices(grid_bits[INFERENCE_T]), 4
        ),
        "bits_per_token_uniform_t": _round(uniform_t_bits),
        "eval_condition_key": condition_key,
        # -- the number in context --------------------------------------------
        "uniform_prior_bits": _round(uniform_prior_bits, 4),
        # Both suffixed with the condition they were measured at, because an
        # unsuffixed "fraction of the prior removed" invites being read as though
        # it applied to the headline number, where it would be a far more
        # flattering and far less meaningful figure.
        f"bits_removed_vs_uniform_prior_t_{INFERENCE_T:.2f}": _round(bits_saved, 4),
        f"fraction_of_uniform_prior_removed_t_{INFERENCE_T:.2f}": _round(
            bits_saved / uniform_prior_bits, 4
        ),
        "weighted_loss": _round(headline_log.loss),
        "accuracy_noised": _round(headline_log.accuracy.get("noised")),
        "accuracy_ground_truth_preserved": _round(
            headline_log.accuracy.get("ground_truth_preserved")
        ),
        "macro_f1_noised": _round(headline_log.macro_f1.get("noised")),
        "macro_f1_ground_truth_preserved": _round(
            headline_log.macro_f1.get("ground_truth_preserved")
        ),
        # -- the architecture --------------------------------------------------
        "arch_shape": arch_shape,
        "params_millions": round(params_total / 1e6, 2),
        "params_total": params_total,
        "params_trainable": params_trainable,
        "d_model": int(model_config.d_model),
        "layers": int(model_config.layers),
        "heads": int(model_config.heads),
        "ffn": int(model_config.ffn),
        "qk_norm": _flag(model_config.qk_norm),
        "self_conditioning": _flag(model_config.self_conditioning),
        "frozen_input_kv": _flag(model_config.frozen_input_kv),
        "segment_embeddings": _flag(model_config.segment_embeddings),
        "per_segment_positions": _flag(model_config.per_segment_positions),
        "rope_theta": float(model_config.rope_theta),
        "architecture_identity": str(checkpoint.get("architecture_identity", "")),
        # -- the weights -------------------------------------------------------
        "checkpoint": str(checkpoint.get("checkpoint", "")),
        "checkpoint_mb": checkpoint_mb,
        "weights": str(checkpoint.get("weights", "")),
        "completed_epochs": int(checkpoint.get("completed_epochs", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_dev_loss": _round(checkpoint.get("best_dev_loss")),
        "debut_mode": _flag(checkpoint.get("debut_mode")),
    }
    # Per-t columns, one per remaining grid point, so a row carries its own noise
    # curve. INFERENCE_T is skipped here -- it is already reported beside the
    # headline above, and a duplicate column would be a second source of truth.
    for level in t_grid:
        if level == INFERENCE_T:
            continue
        row[f"bits_per_token_t_{level:.2f}"] = _round(grid_bits[level])
    # -- the evaluation condition, spelled out beside its hash -----------------
    row.update(condition)
    # -- the redundancy handle -------------------------------------------------
    # Equal exactly when the model, the architecture, the condition, AND the
    # score are all identical -- i.e. when a row adds nothing the file did not
    # already have. Sort by this column to find repeats and delete the extras.
    row["dedupe_key"] = _dedupe_key(
        model_label=context.model_label,
        architecture_identity=str(checkpoint.get("architecture_identity", "")),
        arch_shape=arch_shape,
        global_step=int(checkpoint.get("global_step", 0)),
        condition_key=condition_key,
        bits_per_token=float(headline_log.bits_per_token),
    )
    return row


def _architecture_shape(model_config: Any) -> str:
    """Compact one-cell architecture label, e.g. ``d768-L12-H12-F3072``.

    Exists so the CSV can be grouped or pivoted on "which shape is this" without
    comparing four numeric columns by eye. The individual dimensions are still
    present as their own columns for anything that needs to sort numerically.

    Parameters:
        model_config: the checkpoint's ``ModelConfig``.

    Returns:
        The label string.
    """

    return (
        f"d{int(model_config.d_model)}"
        f"-L{int(model_config.layers)}"
        f"-H{int(model_config.heads)}"
        f"-F{int(model_config.ffn)}"
    )


def _selected_windows(loader: Any) -> list[Any]:
    """Recover the manifest windows behind a loader, in the order it serves them.

    ``SharedResources.dataloader`` wraps the dataset in a ``torch.utils.data``
    ``Subset``, so the scored windows are the dataset's own window tuple indexed
    by the subset's positions. Read back rather than re-derived, so this can
    never disagree with what was actually scored.

    Parameters:
        loader: the ``DataLoader`` returned by ``SharedResources.dataloader``.

    Returns:
        The ``WindowManifestEntry`` objects served, in loader order.

    Called by: :func:`run`.
    """

    subset = loader.dataset
    return [subset.dataset.windows[position] for position in subset.indices]


def _window_selection_key(windows: list[Any]) -> str:
    """Hash the identity of every scored window, in order.

    WHY this is not covered by the `n_windows` count: two runs can both score
    240 windows and score 240 DIFFERENT windows -- change `--n-replays` and the
    striding lands somewhere else entirely. A count cannot tell those apart, so
    without this the comparability key would call two unrelated measurements
    comparable. Hashing the replay, perspective, and timestep span of each window
    makes "the same windows" a verifiable claim.

    Parameters:
        windows: the scored ``WindowManifestEntry`` objects, in loader order.

    Returns:
        The first 10 hex characters of the SHA-256 over their identities.

    Called by: :func:`run`.
    """

    payload = ";".join(
        f"{window.replay_id}:{window.perspective_player}:"
        f"{window.start_timestep}-{window.end_timestep}"
        for window in windows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _eval_condition_key(condition: dict[str, Any]) -> str:
    """Hash the evaluation condition into a short, stable comparability handle.

    Two rows may only be ranked against each other when this matches. A hash is
    used rather than asking the reader to compare sixteen columns by eye; those
    columns are still written out beside it so a mismatch can be diagnosed.

    Parameters:
        condition: field -> value mapping, which must cover exactly
            :data:`EVAL_CONDITION_FIELDS`.

    Returns:
        The first 10 hex characters of the SHA-256 over the field values, taken
        in the fixed order of :data:`EVAL_CONDITION_FIELDS` so the key never
        depends on dict ordering.

    Raises:
        KeyError: a declared field is missing from ``condition``.

    Called by: :func:`_build_row`.
    """

    payload = "|".join(f"{name}={condition[name]}" for name in EVAL_CONDITION_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _dedupe_key(
    *,
    model_label: str,
    architecture_identity: str,
    arch_shape: str,
    global_step: int,
    condition_key: str,
    bits_per_token: float,
) -> str:
    """Hash everything that makes a row redundant with an earlier one.

    The score is rounded to six decimals before hashing: two passes over the same
    windows with the same seed agree far past that, while a genuinely different
    model differs well before it.

    Parameters:
        model_label: the runner's identity for the weights under test.
        architecture_identity: the checkpoint's stamped architecture id.
        arch_shape: the compact shape label from :func:`_architecture_shape`.
        global_step: the optimizer step the checkpoint was saved at.
        condition_key: the value from :func:`_eval_condition_key`.
        bits_per_token: the headline score.

    Returns:
        The first 10 hex characters of the SHA-256 over those fields.

    Called by: :func:`_build_row`.
    """

    payload = (
        f"{model_label}|{architecture_identity}|{arch_shape}|{global_step}"
        f"|{condition_key}|{bits_per_token:.6f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# The shared, appended CSV
# ---------------------------------------------------------------------------


def append_leaderboard_row(row: dict[str, Any], path: Path) -> Path:
    """Append one row to the shared comparison CSV, never losing an existing one.

    The file is read, merged, and rewritten whole rather than opened in append
    mode. WHY: append mode is only correct while the header never changes, and
    this row's columns depend on the configured t-grid and on whatever columns a
    future version of this test adds. Rewriting lets an old row keep every value
    it had, with blanks under any column that did not exist when it was written,
    instead of silently sliding its values under the wrong headers. The file is
    one row per model evaluation, so rewriting it is trivially cheap.

    Column order rule: the existing header is preserved exactly as found --
    including any manual reordering done in a spreadsheet -- and genuinely new
    columns are appended on the right. A brand-new file simply takes this row's
    own order.

    Parameters:
        row: the new row. Keys are column names.
        path: the shared CSV. Created, with a header, if absent.

    Returns:
        ``path``, for the caller's artifact list.

    Called by: :func:`run`.
    """

    existing_rows: list[dict[str, Any]] = []
    columns: list[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            existing_rows = [dict(item) for item in reader]

    for name in row:
        if name not in columns:
            columns.append(name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in existing_rows + [row]:
            writer.writerow({name: item.get(name, "") for name in columns})
    return path


def _print_leaderboard_tail(path: Path, *, limit: int = 10) -> None:
    """Print the most recent leaderboard rows so the console shows the comparison.

    Only the identity and score columns are printed; the full row is in the file.
    Rows whose ``eval_condition_key`` differs from this run's are still shown, so
    a mismatch is visible rather than hidden.

    Parameters:
        path: the shared CSV.
        limit: how many trailing rows to print.

    Called by: :func:`run`.
    """

    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(item) for item in csv.DictReader(handle)]
    if not rows:
        return
    print(f"\n{portable_path(path)} -- last {min(limit, len(rows))} of {len(rows)} row(s):")
    inference_bits_column = f"bits_per_token_t_{INFERENCE_T:.2f}"
    inference_choices_column = f"effective_token_choices_t_{INFERENCE_T:.2f}"
    print(
        f"  {'run_datetime':<20} {'model_label':<34} {'bits/tok':>9} "
        f"{'eff.choices':>12} {'bits@t=1':>10} {'choices@t=1':>12}  cond"
    )
    for item in rows[-limit:]:
        print(
            f"  {item.get('run_datetime', ''):<20} {item.get('model_label', ''):<34} "
            f"{item.get('bits_per_token', ''):>9} "
            f"{item.get('effective_token_choices', ''):>12} "
            f"{item.get(inference_bits_column, ''):>10} "
            f"{item.get(inference_choices_column, ''):>12}  "
            f"{item.get('eval_condition_key', '')}"
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_t_grid(context: TestContext) -> tuple[float, ...]:
    """Read the fixed corruption grid from ``--option leaderboard_t_grid=...``.

    Changing the grid changes ``eval_condition_key``, so an overridden grid can
    never be accidentally compared against rows written with the default one.

    Parameters:
        context: the runner context carrying ``extra`` options.

    Returns:
        The grid, or :data:`DEFAULT_T_GRID` when the option is absent.

    Raises:
        ValueError: the option was given but empty, or holds a value outside
            ``[0, 1]``.
    """

    raw = context.extra.get("leaderboard_t_grid")
    if not raw:
        return DEFAULT_T_GRID
    levels = tuple(float(item) for item in raw.split(",") if item.strip())
    if not levels:
        raise ValueError("leaderboard_t_grid was given but contained no values")
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"leaderboard_t_grid values must be in [0, 1]; got {level}")
    return levels


def _format_grid(levels: tuple[float, ...]) -> str:
    """Render a corruption grid as a stable, hashable string (``0.25/0.50/...``)."""

    return "/".join(f"{level:.2f}" for level in levels)


def _run_moment(run_dir: Path) -> datetime:
    """Recover the run's date-time from the runner's directory name.

    The runner names each run ``<model label>__<%Y-%b-%d_%I-%M%p>``, optionally
    with a ``-2`` collision suffix. Parsing it back -- rather than calling
    ``datetime.now()`` -- ties the CSV row to the run directory sitting beside it
    even when the suite has been running for an hour before reaching this test.

    Parameters:
        run_dir: the runner-allocated run directory.

    Returns:
        The parsed moment, or the current time when the name carries none (which
        happens only when a test is driven outside the runner).
    """

    name = run_dir.name
    if "__" in name:
        stamp = re.sub(r"-\d+$", "", name.split("__", 1)[1])
        try:
            return datetime.strptime(stamp, "%Y-%b-%d_%I-%M%p")
        except ValueError:
            pass
    return datetime.now()


def _effective_choices(bits: float) -> float:
    """Convert bits per token into effective equally-likely token choices.

    ``2 ** bits`` -- the same arithmetic an autoregressive LM's perplexity uses,
    reported under a name that does not claim a sequence likelihood the model
    never computes. 1.0 means the model already knows the token; the vocabulary
    size means it is guessing uniformly.

    Parameters:
        bits: mean unweighted cross entropy in bits per scored position.

    Returns:
        The equivalent number of equally-likely choices.

    Called by: :func:`run` (console lines) and :func:`_build_row`.
    """

    return 2.0**bits


def _round(value: Any, digits: int = 6) -> Any:
    """Round a float for the CSV, rendering ``None`` as an empty cell."""

    if value is None:
        return ""
    return round(float(value), digits)


def _flag(value: Any) -> str:
    """Render a boolean as ``true``/``false``, and an unknown as an empty cell."""

    if value is None:
        return ""
    return "true" if bool(value) else "false"
