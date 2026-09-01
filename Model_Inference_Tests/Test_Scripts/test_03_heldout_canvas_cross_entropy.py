"""Held-out canvas cross-entropy: the training objective, measured on unseen games.

Question this answers
---------------------
"Does the number the model was optimizing actually generalize?" The training run
reports `train_loss` and `dev_loss` per epoch. This test computes the SAME
quantity on the test split -- replays that were in neither -- and breaks it down
so the loss can be attributed rather than just observed.

Method
------
Reuses ``TrainingLoop.validate``, the exact routine that produced the `dev_loss`
column in the run's `epoch_metrics.csv`. A ``TrainingLoop`` is constructed around
the loaded checkpoint purely as a measurement harness:

  * ``validate`` scores with ``self.ema_model``, which here is a copy of the
    weights this run loaded (EMA by default), so the numbers are directly
    comparable to the run's own dev curve;
  * ``fit`` is never called, no optimizer step is ever taken, no checkpoint is
    written, no EMA update is applied, and no metrics path is configured, so the
    probed run is untouched.

The pass is run twice over:

  1. **the training t-distribution** (``fixed_t=None``) -- directly comparable to
     the run's reported `dev_loss`; and
  2. **a sweep of fixed noise levels t** -- the same held-out windows corrupted at
     t = 0.1 ... 1.0. This turns one scalar into a curve, which is what separates
     "the model denoises lightly-corrupted canvases well and collapses at high
     noise" from "it is uniformly mediocre".

Every pass also carries the loss module's own decompositions: per loss class
(observed / fogged / future reconstruction, delimiter, END, PAD, win-loss), per
t-bucket, per player perspective, and per canvas state (positions the model was
shown correctly versus positions actually corrupted).

Fog condition
-------------
Examples are served under the model's TRAINING fog distribution
(`config.fog.rate_distribution`, redrawn per serving), not at a fixed eval fog
rate. This matches how `test_06_unigram_entropy_baseline` builds its dataset, so
the two are directly comparable -- and it keeps the `enemy-fogged` loss class
populated, which a fixed rate of 0.0 would silently empty. Pin a fixed rate with
`--option ce_fog=0.0` if you want the deterministic eval condition instead.

Read the numbers against the entropy floor from
`test_06_unigram_entropy_baseline`, which is the loss a model with no knowledge
of the input would achieve on the same split.

Depends on: ``train.loop.TrainingLoop.validate``, ``inference_test_api``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thesis_ml.train.loop import TrainingLoop

from inference_test_api import (  # noqa: E402
    TestContext,
    TestResult,
    portable_path,
    save_figure,
    write_csv,
    write_json,
)


TEST_NAME = "heldout_canvas_cross_entropy"
TEST_TITLE = "Held-out canvas cross-entropy and its decomposition"
TEST_DESCRIPTION = """
Measures the model's own training objective -- weighted canvas cross-entropy --
on windows from the held-out test replays, using the identical code path that
produced the run's `dev_loss` column (`TrainingLoop.validate` with the EMA
weights). Reports the loss at the training noise distribution and across a sweep
of fixed noise levels t, each broken down by loss class, t-bucket, player
perspective, and canvas state.

This is the generalization test in the model's own currency: comparing the value
here against the run's final `dev_loss` shows how much of the dev performance was
dev-specific, and the t-sweep shows at which corruption levels the model is
actually contributing over the data prior.
""".strip()
TEST_OUTPUTS = (
    "`heldout_cross_entropy.json` -- loss at the training t-distribution and at every swept t, each with per-class / per-t-bucket / per-perspective / per-canvas-state breakdowns",
    "`cross_entropy_by_noise.csv` -- one row per swept noise level t: total loss plus every per-class loss",
    "`cross_entropy_by_class.csv` -- one row per loss class: loss at the training t-distribution and at each swept t",
    "`cross_entropy_vs_noise.png` / `.svg` -- total held-out loss as a function of corruption level t",
    "`cross_entropy_by_class.png` / `.svg` -- per-class held-out loss versus t, one line per loss class",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False

# Noise levels swept. 1.0 is the terminal prior (canvas is entirely noise, the
# hardest condition and the one the deployed sampler starts from); 0.1 is a
# nearly-clean canvas. The training run itself draws t from a power distribution
# over [0, 1] with a 5% spike at exactly 1.0.
DEFAULT_NOISE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

# Windows scored per pass. Each pass is one forward pass per batch, so this is
# far cheaper per window than the sampler-bound tests and can afford a larger
# sample. Override with `--option ce_windows=N`.
DEFAULT_MAX_WINDOWS = 120

# Fog condition. `None` = draw per serving from config.fog.rate_distribution,
# i.e. the visibility the model actually TRAINED under.
#
# This is deliberately NOT the runner's configured eval fog rate. Two reasons:
#   1. It matches `unigram_entropy_baseline`, which builds its dataset the same
#      way. The whole point of that baseline is to be the floor these numbers are
#      read against, and a floor computed under different visibility is not a
#      floor for these numbers.
#   2. At a fixed fog rate of 0.0 nothing is fogged, so the `enemy-fogged` loss
#      class has zero scored positions and silently vanishes from the report --
#      hiding an entire third of the reconstruction objective.
# Pin a fixed rate instead with `--option ce_fog=0.0`.
DEFAULT_FOG_RATE_OVERRIDE = None


def run(context: TestContext) -> TestResult:
    """Measure held-out loss at the training t-distribution and across a t sweep.

    Parameters:
        context: runner-supplied context.

    Returns:
        A :class:`TestResult` with the headline held-out loss values.

    Calls: ``SharedResources.model`` / ``dataloader``, ``TrainingLoop.validate``.
    """

    max_windows = context.option_int("ce_windows", DEFAULT_MAX_WINDOWS)
    if context.max_examples > 0:
        max_windows = min(max_windows, context.max_examples)
    noise_levels = _parse_noise_levels(context)
    fog_override = (
        context.option_float("ce_fog", float("nan"))
        if "ce_fog" in context.extra
        else DEFAULT_FOG_RATE_OVERRIDE
    )

    model, run_config = context.shared.model()

    # A measurement-only TrainingLoop: no metrics paths, no publishers, no
    # optimizer step, no checkpoint write. `validate` scores with `ema_model`,
    # which the constructor deep-copies from the model handed in -- i.e. the
    # weights this run loaded from the checkpoint.
    loop = TrainingLoop(
        model=model,
        config=run_config,
        device=context.device,
        seed=context.seed,
    )

    loader, indices = context.shared.dataloader(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=max_windows,
        run_config=run_config,
        fog_rate_override=fog_override,
    )
    fog_label = (
        "training distribution (per serving)"
        if fog_override is None
        else f"fixed rate {fog_override}"
    )
    print(
        f"scoring {len(indices)} held-out window(s) "
        f"in batches of {run_config.pipeline.batch_size}; fog = {fog_label}"
    )

    passes: dict[str, dict[str, Any]] = {}

    # Pass 1: the training noise distribution. Directly comparable to the
    # dev_loss the run recorded each epoch.
    print("pass: training t-distribution (comparable to the run's dev_loss)")
    loop.generator.manual_seed(context.seed)
    passes["training_t_distribution"] = _validation_to_dict(loop.validate(loader))
    print(f"  loss = {passes['training_t_distribution']['loss']:.6f}")

    # Pass 2..N: one pass per fixed noise level. The generator is reseeded before
    # each pass so the corruption draws are identical across levels and the only
    # thing that changes between them is t itself.
    for level in noise_levels:
        print(f"pass: fixed t = {level:.2f}")
        loop.generator.manual_seed(context.seed)
        passes[f"t_{level:.2f}"] = _validation_to_dict(loop.validate(loader, fixed_t=level))
        print(f"  loss = {passes[f't_{level:.2f}']['loss']:.6f}")

    class_names = sorted(
        {name for entry in passes.values() for name in entry["per_class"]}
    )

    noise_rows = [
        {
            "t": f"{level:.2f}",
            "loss": round(passes[f"t_{level:.2f}"]["loss"], 6),
            **{
                f"class_{name}": round(passes[f"t_{level:.2f}"]["per_class"].get(name, float("nan")), 6)
                for name in class_names
            },
        }
        for level in noise_levels
    ]
    class_rows = [
        {
            "loss_class": name,
            "training_t_distribution": round(
                passes["training_t_distribution"]["per_class"].get(name, float("nan")), 6
            ),
            **{
                f"t_{level:.2f}": round(passes[f"t_{level:.2f}"]["per_class"].get(name, float("nan")), 6)
                for level in noise_levels
            },
        }
        for name in class_names
    ]

    written: list[Path] = []
    written.append(
        write_json(
            {
                "provenance": context.provenance(
                    uses_model=True, fog_rate_override=fog_override
                ),
                "method": (
                    "TrainingLoop.validate with the loaded (EMA by default) weights -- the "
                    "identical routine that produced the run's dev_loss column"
                ),
                "fog_note": (
                    "served under the training fog distribution so these numbers are "
                    "directly comparable to unigram_entropy_baseline's floor"
                    if fog_override is None
                    else f"served at a fixed fog rate of {fog_override}"
                ),
                "n_windows": len(indices),
                "batch_size": run_config.pipeline.batch_size,
                "diffusion_process": run_config.diffusion.process,
                "noise_levels": list(noise_levels),
                "passes": passes,
            },
            context.out_dir / "heldout_cross_entropy.json",
        )
    )
    written.append(
        write_csv(
            noise_rows,
            ["t", "loss"] + [f"class_{name}" for name in class_names],
            context.out_dir / "cross_entropy_by_noise.csv",
        )
    )
    written.append(
        write_csv(
            class_rows,
            ["loss_class", "training_t_distribution"] + [f"t_{level:.2f}" for level in noise_levels],
            context.out_dir / "cross_entropy_by_class.csv",
        )
    )
    written.extend(_plot_loss_vs_noise(noise_levels, passes, context))
    written.extend(_plot_class_loss_vs_noise(noise_levels, passes, class_names, context))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    training_loss = passes["training_t_distribution"]["loss"]
    terminal_key = f"t_{noise_levels[-1]:.2f}"
    return TestResult(
        headline=[
            f"held-out loss at the training t-distribution: {training_loss:.6f}"
            f" ({len(indices)} windows, fog = {fog_label})",
            f"held-out loss at the highest swept t={noise_levels[-1]:.2f}: "
            f"{passes[terminal_key]['loss']:.6f}",
            f"held-out loss at the lowest swept t={noise_levels[0]:.2f}: "
            f"{passes[f't_{noise_levels[0]:.2f}']['loss']:.6f}",
        ],
        artifacts=written,
        metrics={
            "n_windows": len(indices),
            "fog_condition": fog_label,
            "loss_training_t_distribution": training_loss,
            "loss_by_t": {f"{level:.2f}": passes[f"t_{level:.2f}"]["loss"] for level in noise_levels},
        },
    )


def _parse_noise_levels(context: TestContext) -> tuple[float, ...]:
    """Read the t sweep from ``--option ce_noise_levels=0.1,0.5,1.0`` or use the default."""

    raw = context.extra.get("ce_noise_levels")
    if not raw:
        return DEFAULT_NOISE_LEVELS
    levels = tuple(float(item) for item in raw.split(",") if item.strip())
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"ce_noise_levels values must be in [0, 1]; got {level}")
    return levels


def _validation_to_dict(log) -> dict[str, Any]:
    """Flatten a ``ValidationLog`` into a JSON-ready dict.

    Kept as one function so every pass in the sweep is serialized identically and
    the JSON's shape never depends on which pass wrote it.
    """

    return {
        "loss": float(log.loss),
        "per_class": {name: float(value) for name, value in log.per_class.items()},
        "per_t_bucket": {name: float(value) for name, value in log.t_bucket.items()},
        "per_perspective": {name: float(value) for name, value in log.perspective.items()},
        "per_canvas_state": {name: float(value) for name, value in log.canvas_state.items()},
        "per_future_distance": {name: float(value) for name, value in log.future_distance.items()},
        "rare_class_t_bucket": {name: float(value) for name, value in log.rare_class_t_bucket.items()},
        "rare_class_t_bucket_counts": {
            name: int(value) for name, value in log.rare_class_t_bucket_counts.items()
        },
    }


def _plot_loss_vs_noise(
    noise_levels: tuple[float, ...], passes: dict[str, dict[str, Any]], context: TestContext
) -> list[Path]:
    """Total held-out loss as a function of corruption level t.

    The training-t-distribution value is drawn as a horizontal reference line so
    the single number the run reported can be located on the curve.
    """

    values = [passes[f"t_{level:.2f}"]["loss"] for level in noise_levels]
    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    axes.plot(list(noise_levels), values, marker="o", color="#3B6EA5", linewidth=2)
    axes.axhline(
        passes["training_t_distribution"]["loss"],
        color="#C4402A",
        linestyle="--",
        linewidth=1.5,
        label=f"training t-distribution = {passes['training_t_distribution']['loss']:.4f}",
    )
    axes.set_xlabel("canvas corruption level t  (1.0 = terminal prior, entirely noise)")
    axes.set_ylabel("weighted canvas cross-entropy")
    axes.set_title(f"Held-out loss vs corruption level\n{context.model_label}", fontsize=10)
    axes.grid(alpha=0.25)
    axes.legend(fontsize=9)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "cross_entropy_vs_noise", dpi=context.dpi)


def _plot_class_loss_vs_noise(
    noise_levels: tuple[float, ...],
    passes: dict[str, dict[str, Any]],
    class_names: list[str],
    context: TestContext,
) -> list[Path]:
    """Per-loss-class held-out loss versus t, one line per class.

    Shows where the loss actually lives. A flat, high line for a rare class next
    to a low line for a common one is the signature of a model that learned the
    bulk distribution and not the tail.
    """

    if not class_names:
        return []
    figure, axes = plt.subplots(figsize=(8.0, 4.8))
    for name in class_names:
        values = [
            passes[f"t_{level:.2f}"]["per_class"].get(name, float("nan"))
            for level in noise_levels
        ]
        axes.plot(list(noise_levels), values, marker="o", linewidth=1.6, label=name)
    axes.set_xlabel("canvas corruption level t")
    axes.set_ylabel("weighted canvas cross-entropy")
    axes.set_title(f"Held-out loss by class vs corruption level\n{context.model_label}", fontsize=10)
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "cross_entropy_by_class", dpi=context.dpi)
