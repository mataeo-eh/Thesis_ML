"""Outcome-slot probe: can the model call the winner of a game it has never seen?

Question this answers
---------------------
Canvas position 1 holds the perspective player's `[WIN]` / `[LOSS]` outcome
token. Cross-entropy at that position collapses two completely different failure
modes into one number:

  * the model never located canvas index 1, so its distribution there is
    basically the noise marginal; versus
  * the model located index 1 AND learned that only `[WIN]`/`[LOSS]` live there,
    but is outvoted by a copy prior that keeps re-emitting whatever noise token
    it was shown.

They imply different fixes, so they have to be told apart by measurement. This
test runs the existing `thesis_ml.viz.outcome_probe` measurement against the
HELD-OUT test replays -- the probe's own CLI is wired to the ablation arms and to
the train/dev splits, so this script builds the loader over the test split and
calls the probe's public measurement functions directly.

Method
------
One single denoising forward pass per window at a HIGH noise level (default
t ~ U[0.9, 1.0]) -- the regime where the canvas is almost entirely noise and
positional identification is the only signal available. No iterative sampling,
deliberately: later sampler steps would overwrite position 1 and the result would
describe the sampler's schedule rather than the weights' distribution at a known
t. Self-conditioning is passed as `None`, the honest first-denoising-step
condition.

What to read
------------
* `pair_mass` -- total probability the model puts on `[WIN]` + `[LOSS]`. Those two
  tokens are outside the forward process's noise support, so mass there can only
  come from the model. High pair mass = it found the slot.
* `p_true_given_pair` -- its call BETWEEN the two outcomes with "did it find the
  slot" divided out. 0.5 is a coin flip; above 0.5 on unseen games is real
  predictive signal about who wins.
* `by_perspective` -- the same behaviour split p1 / p2. Because the outcome label
  is perspective-relative, a gap between the sides separates "reads the game"
  from "has a standing class preference".

Depends on: ``viz.outcome_probe.probe_batch`` / ``summarize`` /
``_print_arm_summary``, ``inference_test_api``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from thesis_ml.viz.outcome_probe import (
    _draw_noise_levels,
    _print_arm_summary,
    probe_batch,
    summarize,
)

from inference_test_api import (  # noqa: E402
    TestContext,
    TestResult,
    portable_path,
    save_figure,
    write_json,
)


TEST_NAME = "outcome_position_probe"
TEST_TITLE = "Win/loss outcome-slot probe at high noise"
TEST_DESCRIPTION = """
Reads the model's output distribution at canvas position 1 -- the
`[WIN]`/`[LOSS]` outcome slot -- for windows drawn from the held-out test
replays, under a single denoising forward pass at high corruption
(t ~ U[0.9, 1.0]).

Separates "never found the outcome position" from "found it and knows only
`[WIN]`/`[LOSS]` live there, but is outvoted by a copy prior", which per-class
cross-entropy provably cannot distinguish. Because the two outcome tokens sit
outside the forward process's noise support, any probability mass on them is
attributable to the model rather than to the corruption. Reports that mass, the
model's call between the two outcomes with slot-finding divided out
(`P(true | pair)`), the rank of the true outcome token in the full vocabulary,
and the whole picture split by player perspective and by t-bucket.
""".strip()
TEST_OUTPUTS = (
    "`outcome_position_probe.json` -- the full probe payload: reference values, three subset summaries (all / genuinely-noised / clean-survived), a per-t-bucket breakdown, and one record per probed window",
    "`outcome_probe_summary.txt` -- the probe's own console summary: pair mass, shown-noise mass, true-token rank, P(true | pair), argmax behaviour, outcome balance, and the p1/p2 comparison",
    "`outcome_probe_summary.png` / `.svg` -- bar chart of the headline probabilities (pair mass, mass on the true outcome, mass on the shown noise token) with the coin-flip reference marked",
)
USES_MODEL = True
REQUIRES_DEBUT_FINETUNE = False

# High-noise band. At t near 1.0 the canvas carries essentially no information,
# so anything the model puts on the outcome pair comes from the input replay
# rather than from a surviving ground-truth token.
DEFAULT_T_MIN = 0.9
DEFAULT_T_MAX = 1.0

# Windows probed. One forward pass each, so this is cheap; the default is a few
# dozen batches' worth. Override with `--option probe_windows=N`.
DEFAULT_MAX_WINDOWS = 120


def run(context: TestContext) -> TestResult:
    """Probe canvas position 1 across held-out windows and write the report.

    Parameters:
        context: runner-supplied context.

    Returns:
        A :class:`TestResult` with the headline outcome-slot numbers.

    Calls: ``SharedResources.model`` / ``dataloader``,
    ``viz.outcome_probe.probe_batch`` and ``summarize``.
    """

    max_windows = context.option_int("probe_windows", DEFAULT_MAX_WINDOWS)
    if context.max_examples > 0:
        max_windows = min(max_windows, context.max_examples)
    t_min = context.option_float("probe_t_min", DEFAULT_T_MIN)
    t_max = context.option_float("probe_t_max", DEFAULT_T_MAX)
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(f"probe noise band must satisfy 0 <= t_min <= t_max <= 1; got {t_min}, {t_max}")

    model, run_config = context.shared.model()
    vocabulary = context.shared.vocabulary()

    loader, indices = context.shared.dataloader(
        n_replays=context.n_replays,
        n_windows_per_replay=context.n_windows_per_replay,
        max_examples=max_windows,
        run_config=run_config,
    )
    print(f"probing {len(indices)} held-out window(s) at t in [{t_min:.2f}, {t_max:.2f}]")

    # Two separate seeded generators, matching the probe's own convention: the t
    # generator lives on CPU (so noise levels are reproducible regardless of
    # device) and the corruption generator must live on the compute device
    # because corrupt_batch draws on the target canvas's device.
    t_generator = torch.Generator().manual_seed(context.seed)
    corruption_generator = torch.Generator(device=context.device).manual_seed(context.seed)

    records = []
    consumed = 0
    for batch in loader:
        rows = batch.target_canvas.shape[0]
        noise_levels = _draw_noise_levels(rows, t_min=t_min, t_max=t_max, generator=t_generator)
        records.extend(
            probe_batch(
                model,
                batch,
                vocabulary=vocabulary,
                config=run_config,
                device=context.device,
                noise_levels=noise_levels,
                corruption_generator=corruption_generator,
                row_indices=indices[consumed : consumed + rows],
            )
        )
        consumed += rows
        print(f"  probed {consumed}/{len(indices)} windows", flush=True)

    from dataclasses import asdict

    summary = summarize(records, vocab_size=int(model.vocab_size))
    payload: dict[str, Any] = {
        "provenance": context.provenance(uses_model=True),
        "probe": {
            "split": "test (held out)",
            "examples_probed": len(records),
            "t_range": [t_min, t_max],
            "forward_passes_per_example": 1,
            "self_conditioning_input": "zeros (first denoising step)"
            if run_config.model.self_conditioning
            else "disabled",
            "diffusion_process": run_config.diffusion.process,
            "probed_dataset_indices": indices,
        },
        "summary": summary,
        "records": [asdict(record) for record in records],
    }

    written: list[Path] = [write_json(payload, context.out_dir / "outcome_position_probe.json")]

    # Reuse the probe's own console summary rather than writing a second,
    # divergent renderer. It expects the arm-style payload keys, so the few
    # fields it reads are supplied from the checkpoint facts.
    facts = context.shared.checkpoint_facts()
    console_payload = dict(payload)
    console_payload.update(
        {
            "weights": facts["weights"],
            "global_step": facts["global_step"],
            "completed_epochs": facts["completed_epochs"],
            "architecture_identity": facts["architecture_identity"],
        }
    )
    written.append(_write_console_summary(console_payload, context.out_dir))
    written.extend(_plot_summary(summary, context))

    for path in written:
        print(f"  wrote {portable_path(path)}")

    noised = summary.get("position_1_noised", {})
    headline = [f"probed {len(records)} held-out window(s) at t in [{t_min:.2f}, {t_max:.2f}]"]
    if noised.get("n_examples"):
        conditional = noised.get("p_true_given_pair", {})
        headline.append(
            f"mass on [WIN]+[LOSS]: mean {noised['pair_mass']['mean']:.4f}"
            f" (median {noised['pair_mass']['median']:.4f})"
        )
        headline.append(
            f"P(true outcome | it picked one of the two): pooled {conditional.get('pooled', float('nan')):.3f}"
            "  [0.5 = coin flip]"
        )
    else:
        headline.append("no genuinely-noised position-1 examples in this sample")

    return TestResult(
        headline=headline,
        artifacts=written,
        metrics={
            "examples_probed": len(records),
            "t_range": [t_min, t_max],
            "pair_mass_mean": noised.get("pair_mass", {}).get("mean"),
            "p_true_given_pair_pooled": noised.get("p_true_given_pair", {}).get("pooled"),
            "argmax_equals_true_fraction": noised.get("argmax_equals_true_fraction"),
        },
    )


def _write_console_summary(payload: dict[str, Any], out_dir: Path) -> Path:
    """Capture ``_print_arm_summary``'s output into a text artifact.

    The probe's summary printer is the canonical human rendering of these
    numbers; capturing it keeps this test from growing a second, drifting copy.
    """

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _print_arm_summary(payload)
    text = buffer.getvalue()
    print(text, end="")
    path = out_dir / "outcome_probe_summary.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _plot_summary(summary: dict[str, Any], context: TestContext) -> list[Path]:
    """Bar chart of the three headline probabilities on genuinely-noised rows.

    Restricted to genuinely-noised rows because on a row whose position 1
    survived the corruption uncorrupted, "mass on the shown token" IS
    correctness wearing a different name, and mixing the two makes the copy-prior
    reading impossible.
    """

    noised = summary.get("position_1_noised", {})
    if not noised.get("n_examples"):
        return []

    labels = [
        "mass on\n[WIN] + [LOSS]",
        "mass on the\ntrue outcome",
        "mass on the shown\nnoise token",
    ]
    values = [
        float(noised["pair_mass"]["mean"]),
        float(noised["true_mass"]["mean"]) if "true_mass" in noised else float("nan"),
        float(noised["shown_token_mass"]["mean"]),
    ]
    colors = ["#3B6EA5", "#2E8B57", "#C4402A"]

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    bars = axes.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        if value == value:  # skip NaN
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes.set_ylabel("mean probability at canvas position 1")
    axes.set_title(
        f"Outcome-slot probe on {noised['n_examples']} held-out windows "
        f"(genuinely-noised rows)\n{context.model_label}",
        fontsize=10,
    )
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, context.out_dir, "outcome_probe_summary", dpi=context.dpi)
