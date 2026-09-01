"""How much of an unseen canvas does the model recover, as a function of corruption?

Question this answers
---------------------
"Given a held-out game's output canvas with a fraction t of its positions
destroyed, how many of those destroyed positions does the model put back
EXACTLY right -- and how does that degrade as t goes from a nearly-clean canvas
to the terminal all-noise prior the deployed sampler actually starts from?"

This is the hard-decision counterpart to `test_03_heldout_canvas_cross_entropy`.
Cross-entropy rewards putting probability in roughly the right place; this test
asks the blunt question of whether the argmax token is the correct one. A model
can improve its CE substantially while its exact-token recovery stays flat, and
those are different claims about what it learned.

Method
------
For each noise level t in the sweep, and for each held-out batch:

  1. ``inference.sampler.denoise_canvas_once`` builds the infill state exactly as
     the deployed sampler's first step does -- corrupt the scored canvas
     positions with probability t, leave the remainder revealed as ground truth
     -- and runs ONE denoising forward pass;
  2. accuracy is measured ONLY over the positions the model actually had to
     predict (scored AND not revealed), so a low-t pass is not credited for the
     ground truth it was handed;
  3. the same accuracy is broken out per loss class, so recovery on common
     reconstruction tokens is separated from recovery on the rare structural and
     outcome tokens;
  4. the resulting canvas is grammar-checked with ``inference.decode.validate_canvas``.

A single forward pass rather than the full sampler is deliberate: it isolates the
weights' one-shot denoising ability from the sampler's iterative schedule, which
is a separate component with its own settings.

Depends on: ``inference.sampler.denoise_canvas_once``,
``inference.decode.validate_canvas``, ``model.loss.active_class_id_to_name``,
``inference_test_api``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from thesis_ml.inference.decode import validate_canvas
from thesis_ml.inference.sampler import denoise_canvas_once
from thesis_ml.model.loss import active_class_id_to_name

from inference_test_api import (  # noqa: E402
    TestContext,
    TestResult,
    portable_path,
    save_figure,
    write_csv,
    write_json,
)


TEST_NAME = "noise_recovery_sweep"
TEST_TITLE = "Exact-token recovery vs canvas corruption level"
TEST_DESCRIPTION = """
Corrupts a fraction t of the scored canvas positions of held-out test windows,
runs one denoising forward pass, and measures the fraction of the CORRUPTED
positions the model restores to exactly the right token -- swept across t from a
nearly-clean canvas to the terminal all-noise prior the deployed sampler starts
from.

Accuracy is scored only on positions the model actually had to predict, never on
positions it was handed as revealed ground truth, so the curve is comparable
across noise levels. Broken out per loss class (observed / fogged / future enemy
reconstruction, delimiter, END, PAD, win-loss) and accompanied by the fraction of
resulting canvases that pass grammar validation.
""".strip()
TEST_OUTPUTS = (
    "`noise_recovery.json` -- per noise level: overall exact-token recovery, per-class recovery, scored/predicted position counts, and grammar validity rate",
    "`noise_recovery.csv` -- one row per noise level: overall accuracy, grammar validity, and per-class accuracy columns",
    "`noise_recovery_curve.png` / `.svg` -- exact-token recovery versus corruption level t, with the grammar validity rate on a second axis",
    "`noise_recovery_by_class.png` / `.svg` -- per-loss-class recovery versus t, one line per class",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False

# Swept corruption levels. 1.0 is the terminal prior -- the entire scored canvas
# is noise and every scored position must be predicted, which is what the
# deployed sampler faces on its first step.
DEFAULT_NOISE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

# One forward pass per batch per level, so this is cheap relative to the
# sampler-bound tests. Override with `--option recovery_windows=N`.
DEFAULT_MAX_WINDOWS = 120


def run(context: TestContext) -> TestResult:
    """Sweep corruption levels and measure exact-token recovery on held-out windows.

    Parameters:
        context: runner-supplied context.

    Returns:
        A :class:`TestResult` with recovery at the easiest and hardest swept levels.

    Calls: ``SharedResources.model`` / ``dataloader``, ``denoise_canvas_once``,
    ``validate_canvas``.
    """

    max_windows = context.option_int("recovery_windows", DEFAULT_MAX_WINDOWS)
    if context.max_examples > 0:
        max_windows = min(max_windows, context.max_examples)
    noise_levels = _parse_noise_levels(context)

    model, run_config = context.shared.model()
    class_id_to_name = active_class_id_to_name(run_config)

    loader, indices = context.shared.dataloader(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=max_windows,
        run_config=run_config,
    )
    print(f"sweeping {len(noise_levels)} noise level(s) over {len(indices)} held-out window(s)")

    per_level: dict[str, dict[str, Any]] = {}
    for level in noise_levels:
        print(f"level t = {level:.2f}")
        per_level[f"{level:.2f}"] = _measure_level(
            model=model,
            loader=loader,
            run_config=run_config,
            device=context.device,
            noise_rate=level,
            class_id_to_name=class_id_to_name,
        )
        stats = per_level[f"{level:.2f}"]
        print(
            f"  recovered {stats['correct_positions']}/{stats['predicted_positions']} "
            f"= {stats['exact_token_recovery']:.4f}; "
            f"grammar-valid canvases {stats['grammar_valid_canvases']}/{stats['n_windows']}"
        )

    class_names = sorted(
        {name for stats in per_level.values() for name in stats["per_class_recovery"]}
    )
    rows = [
        {
            "t": f"{level:.2f}",
            "exact_token_recovery": round(per_level[f"{level:.2f}"]["exact_token_recovery"], 6),
            "predicted_positions": per_level[f"{level:.2f}"]["predicted_positions"],
            "scored_positions": per_level[f"{level:.2f}"]["scored_positions"],
            "grammar_validity_rate": round(per_level[f"{level:.2f}"]["grammar_validity_rate"], 6),
            **{
                f"class_{name}": _round_or_blank(
                    per_level[f"{level:.2f}"]["per_class_recovery"].get(name)
                )
                for name in class_names
            },
        }
        for level in noise_levels
    ]

    written: list[Path] = []
    written.append(
        write_json(
            {
                "provenance": context.provenance(uses_model=True),
                "method": (
                    "one denoising forward pass per level "
                    "(inference.sampler.denoise_canvas_once); accuracy scored only on "
                    "positions that were corrupted, never on revealed ground truth"
                ),
                "n_windows": len(indices),
                "noise_levels": list(noise_levels),
                "levels": per_level,
            },
            context.out_dir / "noise_recovery.json",
        )
    )
    written.append(
        write_csv(
            rows,
            [
                "t",
                "exact_token_recovery",
                "predicted_positions",
                "scored_positions",
                "grammar_validity_rate",
            ]
            + [f"class_{name}" for name in class_names],
            context.out_dir / "noise_recovery.csv",
        )
    )
    written.extend(_plot_recovery_curve(noise_levels, per_level, context))
    written.extend(_plot_recovery_by_class(noise_levels, per_level, class_names, context))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    easiest = per_level[f"{noise_levels[0]:.2f}"]
    hardest = per_level[f"{noise_levels[-1]:.2f}"]
    return TestResult(
        headline=[
            f"exact-token recovery at t={noise_levels[0]:.2f}: {easiest['exact_token_recovery']:.4f}",
            f"exact-token recovery at t={noise_levels[-1]:.2f} (terminal prior): "
            f"{hardest['exact_token_recovery']:.4f}",
            f"grammar-valid canvases at t={noise_levels[-1]:.2f}: "
            f"{hardest['grammar_valid_canvases']}/{hardest['n_windows']}",
        ],
        artifacts=written,
        metrics={
            "n_windows": len(indices),
            "recovery_by_t": {
                key: stats["exact_token_recovery"] for key, stats in per_level.items()
            },
            "grammar_validity_by_t": {
                key: stats["grammar_validity_rate"] for key, stats in per_level.items()
            },
        },
    )


def _measure_level(
    *,
    model,
    loader,
    run_config,
    device: torch.device,
    noise_rate: float,
    class_id_to_name: dict[int, str],
) -> dict[str, Any]:
    """Run one denoising pass per batch at one noise level and pool the counts.

    Counts are pooled across every position of every window rather than averaged
    per window, so windows with longer canvases carry the weight they should.

    Parameters:
        model: the loaded model in eval mode.
        loader: the held-out dataloader.
        run_config: the checkpoint's config (drives the corruption process).
        device: compute device.
        noise_rate: the corruption level t for this pass.
        class_id_to_name: loss-class id to human name, from the run's taxonomy.

    Returns:
        A dict of pooled counts and rates for this level.
    """

    correct_positions = 0
    predicted_positions = 0
    scored_positions = 0
    grammar_valid = 0
    n_windows = 0
    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)

    for batch in loader:
        sampled = denoise_canvas_once(
            model,
            batch,
            run_config,
            device=device,
            return_final_logits=False,
            noise_rate=noise_rate,
        )
        # Everything below is compared on CPU: denoise_canvas_once already
        # returns CPU tensors, and the batch's mask/label tensors never left it.
        canvas = sampled.canvas
        target = batch.target_canvas
        scored = batch.canvas_loss_mask.to(torch.bool)
        revealed = (
            sampled.revealed_mask.to(torch.bool)
            if sampled.revealed_mask is not None
            else torch.zeros_like(scored)
        )
        # The positions the model genuinely had to produce: scored AND not
        # handed back as revealed ground truth. At t = 1.0 this is every scored
        # position.
        predicted_mask = scored & ~revealed
        correct_mask = predicted_mask & canvas.eq(target)

        correct_positions += int(correct_mask.sum())
        predicted_positions += int(predicted_mask.sum())
        scored_positions += int(scored.sum())

        labels = batch.class_labels
        for class_id, class_name in class_id_to_name.items():
            in_class = predicted_mask & labels.eq(class_id)
            total = int(in_class.sum())
            if total:
                class_total[class_name] += total
                class_correct[class_name] += int((in_class & canvas.eq(target)).sum())

        for row in range(canvas.shape[0]):
            n_windows += 1
            if validate_canvas(canvas[row].tolist()).valid:
                grammar_valid += 1

    return {
        "noise_rate": noise_rate,
        "n_windows": n_windows,
        "scored_positions": scored_positions,
        "predicted_positions": predicted_positions,
        "correct_positions": correct_positions,
        "exact_token_recovery": correct_positions / predicted_positions if predicted_positions else 0.0,
        "grammar_valid_canvases": grammar_valid,
        "grammar_validity_rate": grammar_valid / n_windows if n_windows else 0.0,
        "per_class_recovery": {
            name: class_correct[name] / class_total[name]
            for name in sorted(class_total)
            if class_total[name]
        },
        "per_class_predicted_positions": {name: class_total[name] for name in sorted(class_total)},
    }


def _parse_noise_levels(context: TestContext) -> tuple[float, ...]:
    """Read the sweep from ``--option recovery_noise_levels=0.1,0.5,1.0`` or use the default."""

    raw = context.extra.get("recovery_noise_levels")
    if not raw:
        return DEFAULT_NOISE_LEVELS
    levels = tuple(float(item) for item in raw.split(",") if item.strip())
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"recovery_noise_levels values must be in [0, 1]; got {level}")
    return levels


def _round_or_blank(value: float | None) -> Any:
    """Round a rate for CSV output, leaving cells with no observations empty."""

    return "" if value is None else round(value, 6)


def _plot_recovery_curve(
    noise_levels: tuple[float, ...], per_level: dict[str, dict[str, Any]], context: TestContext
) -> list[Path]:
    """Recovery versus t, with grammar validity on a secondary axis.

    The two curves belong on one figure because they trade off: a model can keep
    emitting grammatically valid canvases while recovering almost nothing, and
    seeing both at once prevents reading either in isolation.
    """

    recovery = [per_level[f"{level:.2f}"]["exact_token_recovery"] for level in noise_levels]
    validity = [per_level[f"{level:.2f}"]["grammar_validity_rate"] for level in noise_levels]

    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    axes.plot(list(noise_levels), recovery, marker="o", color="#3B6EA5", linewidth=2, label="exact-token recovery")
    axes.set_xlabel("canvas corruption level t  (1.0 = terminal prior, entirely noise)")
    axes.set_ylabel("fraction of corrupted positions restored exactly")
    axes.set_ylim(0.0, 1.0)
    axes.grid(alpha=0.25)

    secondary = axes.twinx()
    secondary.plot(
        list(noise_levels),
        validity,
        marker="s",
        linestyle="--",
        color="#C4402A",
        linewidth=1.6,
        label="grammar validity",
    )
    secondary.set_ylabel("fraction of canvases passing grammar validation")
    secondary.set_ylim(0.0, 1.0)

    handles = axes.get_lines() + secondary.get_lines()
    axes.legend(handles, [line.get_label() for line in handles], fontsize=9, loc="best")
    axes.set_title(f"Held-out canvas recovery vs corruption\n{context.model_label}", fontsize=10)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "noise_recovery_curve", dpi=context.dpi)


def _plot_recovery_by_class(
    noise_levels: tuple[float, ...],
    per_level: dict[str, dict[str, Any]],
    class_names: list[str],
    context: TestContext,
) -> list[Path]:
    """Per-loss-class recovery versus t, one line per class.

    Splits "recovers the bulk of the canvas" from "recovers the rare structural
    and outcome tokens", which the pooled curve deliberately blends.
    """

    if not class_names:
        return []
    figure, axes = plt.subplots(figsize=(8.0, 4.8))
    for name in class_names:
        values = [
            per_level[f"{level:.2f}"]["per_class_recovery"].get(name, float("nan"))
            for level in noise_levels
        ]
        axes.plot(list(noise_levels), values, marker="o", linewidth=1.6, label=name)
    axes.set_xlabel("canvas corruption level t")
    axes.set_ylabel("fraction of corrupted positions restored exactly")
    axes.set_ylim(0.0, 1.0)
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, ncol=2)
    axes.set_title(f"Held-out recovery by loss class\n{context.model_label}", fontsize=10)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "noise_recovery_by_class", dpi=context.dpi)
