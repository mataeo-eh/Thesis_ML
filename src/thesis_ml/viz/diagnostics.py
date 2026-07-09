"""Diagnostic figures for a trained checkpoint against held-out replays.

Role in the system
------------------
This module is a THIN, READ-ONLY consumer of the existing evaluation pipeline.
Given a checkpoint and a directory of extractor parquet replays it renders a
prediction-vs-ground-truth count comparison per selected window:

  * Figure A -- aligned ground-truth, predicted, and signed-error count
    heatmaps with high-contrast palettes and a compact match summary.
  * Optional Figure B -- a build-order first-appearance timeline: one row per entity
    type with a ground-truth marker and a predicted marker at each type's first
    appearance bucket, plus a timing-tolerance band (tolerance from config).
    This is emitted only with ``--first-appearance`` because it is meaningful
    only for a model fine-tuned to emit debut/build-order targets.

It reuses, and never re-implements, every stage of the pipeline:
  * ingestion / tokenization / clamped-input / ground-truth target: the same
    ``preprocess_replays`` + ``SC2DiffusionDataset`` path the training pipeline
    uses (``thesis_ml.data``);
  * weight loading: ``load_diagnostic_model`` loads the EMA weights by default
    (matching what the sampler/eval path serves), or the raw weights when
    ``--raw`` is passed;
  * prediction + decode + the single build-order oracle on BOTH prediction and
    ground truth: ``eval.harness.evaluate_example`` (which defaults to
    ``sample_canvas`` and can explicitly use one-pass denoising through
    ``--bypass-sampler`` before the shared decode/oracle stages).

The ``--output-mask`` flag controls how masked the output canvas starts: the
fraction ``t`` of canvas positions the model must predict, with the remaining
``1 - t`` revealed as ground truth (the same LLaDA/MDLM corruption the training
pipeline applies at that ``t``). It accepts several rates and runs inference once
per rate (default ``0.5`` -- the pre-training average; ``1.0`` -- fully masked, the
previous behavior). It composes with either prediction path: without
``--bypass-sampler`` the iterative sampler infills the masked positions; with it,
a single forward pass does. The predicted-canvas JSON export flags each
token as MODEL (predicted) or TRUTH (revealed) for readability.

Because the counts and events come straight out of ``evaluate_example``, the
figures are guaranteed to be at model (sampling-interval) resolution and to use
the same SPEC-7 whole-timestep truncation handling as the reported metrics.

Nothing here mutates the checkpoint, the source replays, or the config. All
derived tokenization artifacts are written under a scratch subdirectory of
``--out-dir`` so the run touches nothing outside ``--out-dir``. The default
deliverables are static image files (PNG + vector SVG per figure and one
combined multi-page PDF); ``--csv`` and ``--json`` independently enable a
side-by-side prediction-vs-truth comparison CSV and final-canvas logit exports,
and ``--show-input`` dumps model input canvases (self vs enemy tokens marked)
alongside the other artifacts. Non-image exports consolidate multiple windows
into one labelled CSV/text/JSON artifact per output-mask directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

# Force a non-interactive backend: this tool only writes files, never opens a
# window or a web/interactive app.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import torch  # noqa: E402

from thesis_ml.config import ProjectConfig, load_config  # noqa: E402
from thesis_ml.data.dataset import DatasetExample, SC2DiffusionDataset  # noqa: E402
from thesis_ml.data.windowing import (  # noqa: E402
    WindowManifestEntry,
    load_window_manifest,
    preprocess_replays,
)
from thesis_ml.eval.buildorder import BuildOrderEvent  # noqa: E402
from thesis_ml.eval.harness import EvaluationExampleResult, evaluate_example  # noqa: E402
from thesis_ml.inference.timing import TimedTimestep  # noqa: E402
from thesis_ml.model.model import SC2StrategyDiffusionModel  # noqa: E402
from thesis_ml.vocab.content_vocab import ContentVocabulary, load_content_vocabulary  # noqa: E402
from thesis_ml.vocab.special_tokens import SPECIAL_TOKENS  # noqa: E402


# ---------------------------------------------------------------------------
# Model / ingestion wiring (all calls delegate to existing interfaces)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedExample:
    """One selected window plus its harness evaluation result.

    Attributes:
        example: the ``DatasetExample`` produced by the existing dataset path.
        result: the ``EvaluationExampleResult`` from ``evaluate_example`` (which
            already ran sampling, decode, and the build-order oracle on both
            sides). Carries the predicted/ground-truth count grids and event
            sets this module renders.
        label: a short human-readable identifier (replay id / perspective /
            window start) used in figure titles and output filenames.
        output_mask: the output-canvas mask rate ``t`` this example was rendered
            at (fraction of the canvas masked / model-predicted; the rest revealed
            as ground truth). Shown in figure titles so a reader can tell which
            ``--output-mask`` value produced the figure. ``1.0`` = fully masked.
    """

    example: DatasetExample
    result: EvaluationExampleResult
    label: str
    output_mask: float = 1.0


def load_diagnostic_model(
    checkpoint_path: str | Path,
    config: ProjectConfig,
    *,
    device: torch.device | str,
    use_raw: bool = False,
) -> tuple[SC2StrategyDiffusionModel, ProjectConfig]:
    """Build the model matching a checkpoint and load its EMA or raw weights.

    The architecture (``model.*``) and vocabulary size must match the saved
    tensors, so both are taken from the checkpoint itself: ``model.*`` from the
    ``config`` object the training loop stored, and ``vocab_size`` inferred from
    the output-head weight shape.

    Weight selection is deterministic and controlled solely by ``use_raw``:

      * ``use_raw=False`` (the DEFAULT) loads the EMA (exponential moving
        average) weights -- exactly what the sampler/eval path serves, so the
        rendered diagnostics match reported metrics (SPEC 5/7). This default
        NEVER silently falls back to the raw weights: a checkpoint without an
        ``ema_model`` entry raises, so "no flag" always means "EMA weights".
      * ``use_raw=True`` loads the raw (final optimizer step) ``model`` weights
        instead. Use this to inspect the non-averaged weights directly.

    Parameters:
        checkpoint_path: a checkpoint written by ``TrainingLoop.save_checkpoint``.
        config: the user-supplied eval config; everything except ``model.*`` is
            preserved (fog, sampler, eval tolerances, data budgets, paths).
        device: torch device string/object.
        use_raw: load raw weights instead of EMA (default ``False`` -> EMA).

    Returns:
        ``(model, run_config)`` where ``run_config`` is ``config`` with its
        ``model`` section replaced by the checkpoint's, so the sampler's
        ``self_conditioning`` flag matches the loaded weights.

    Raises:
        KeyError: if the default (EMA) weights are requested but the checkpoint
            has no ``ema_model`` entry -- re-run with ``use_raw=True`` (``--raw``)
            to use the raw weights explicitly.

    Calls:
        ``torch.load``, ``SC2StrategyDiffusionModel``.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    stored_config = checkpoint.get("config")
    model_config = stored_config.model if isinstance(stored_config, ProjectConfig) else config.model
    run_config = replace(config, model=model_config)

    # Pick the weight set explicitly. The default path requires the EMA weights
    # to be present rather than degrading to raw, so "no --raw flag" is always,
    # deterministically, the EMA weights.
    if use_raw:
        state = checkpoint["model"]
    else:
        if "ema_model" not in checkpoint:
            raise KeyError(
                f"checkpoint {Path(checkpoint_path)} has no EMA weights ('ema_model'); "
                "re-run with --raw to load the raw weights explicitly"
            )
        state = checkpoint["ema_model"]

    # Infer vocab size from the selected output head (shape [vocab_size, d_model]).
    vocab_size = int(state["output_head.weight"].shape[0])

    model = SC2StrategyDiffusionModel(run_config, vocab_size=vocab_size)
    model.load_state_dict(state)
    model.eval()
    return model, run_config


def _replay_paths(replay_dir: str | Path, glob: str) -> list[str]:
    """Return sorted extractor parquet paths under ``replay_dir``.

    Sorting keeps replay selection deterministic across runs. Uses the config's
    replay glob so the same file-matching rule as the training pipeline applies.
    """

    paths = sorted(str(path) for path in Path(replay_dir).glob(glob))
    if not paths:
        raise FileNotFoundError(f"no replays matching {glob!r} under {replay_dir}")
    return paths


def _select_windows(
    windows: Sequence[WindowManifestEntry],
    *,
    n_windows: int,
) -> list[WindowManifestEntry]:
    """Pick the first ``n_windows`` windows from every selected replay.

    There is intentionally no second overall cap: requesting five windows from
    two replays yields up to ten examples.

    Parameters:
        windows: manifest entries already filtered to the selected replays.
        n_windows: windows to render per replay id; ``<= 0`` means every
            available window from every selected replay.

    Returns:
        The selected manifest entries, in manifest order.
    """

    per_replay_count: dict[str, int] = {}
    selected: list[WindowManifestEntry] = []
    for window in windows:
        count = per_replay_count.get(window.replay_id, 0)
        if n_windows > 0 and count >= n_windows:
            continue
        selected.append(window)
        per_replay_count[window.replay_id] = count + 1
    return selected


def ingest_examples(
    config: ProjectConfig,
    replay_dir: str | Path,
    *,
    fog_rate: float,
    n_replays: int,
    n_windows: int,
    artifact_root: Path,
    manifest_path: Path,
) -> tuple[ProjectConfig, ContentVocabulary, list[DatasetExample]]:
    """Build ``DatasetExample`` objects from a replay dir via the existing path.

    This composes the SAME primitives the training pipeline uses --
    ``preprocess_replays`` (tokenization + window manifest) and
    ``SC2DiffusionDataset`` (clamped input + ground-truth target). It writes the
    derived tokenization artifacts and manifest UNDER ``--out-dir`` (via a
    config with those two paths redirected) so nothing outside ``--out-dir`` is
    mutated and the source data is untouched.

    Parameters:
        config: user eval config (with the checkpoint's ``model`` section).
        replay_dir: directory of extractor ``*_game_state.parquet`` files.
        fog_rate: fixed fog rate applied to every example (the eval fog
            condition). Passed as ``fog_rate_override`` so fog is deterministic.
        n_replays: how many replays from ``replay_dir`` to ingest (``<= 0`` =
            all replays in the directory).
        n_windows: windows to keep per selected replay (``<= 0`` = all).
        artifact_root: scratch dir under out-dir for tokenized replay artifacts.
        manifest_path: scratch window-manifest path under out-dir.

    Returns:
        ``(run_config, vocabulary, examples)``.

    Calls:
        ``load_content_vocabulary``, ``preprocess_replays``,
        ``load_window_manifest``, ``SC2DiffusionDataset``.
    """

    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    replay_paths = _replay_paths(replay_dir, config.pipeline.replay_glob)
    if n_replays > 0:
        replay_paths = replay_paths[:n_replays]

    # Redirect derived-artifact paths under out-dir so ingestion is read-only
    # with respect to the user's data tree.
    run_config = replace(
        config,
        data=replace(
            config.data,
            tokenized_replay_dir=str(artifact_root),
            window_manifest_path=str(manifest_path),
        ),
    )

    perspectives = tuple(
        item.strip() for item in config.pipeline.perspectives.split(",") if item.strip()
    )
    preprocess_replays(replay_paths, run_config, vocabulary, perspectives=perspectives, force=False)
    windows = load_window_manifest(manifest_path, config=run_config, replay_paths=replay_paths)
    selected = _select_windows(windows, n_windows=n_windows)
    if not selected:
        raise RuntimeError("no windows selected from the ingested replays")

    dataset = SC2DiffusionDataset(
        selected,
        run_config,
        vocabulary,
        seed=config.pipeline.seed,
        fog_rate_override=fog_rate,
    )
    examples = [dataset[index] for index in range(len(dataset))]
    return run_config, vocabulary, examples


def evaluate_selected(
    model: SC2StrategyDiffusionModel,
    examples: Sequence[DatasetExample],
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    *,
    device: torch.device | str,
    include_canvas_logits: bool = False,
    bypass_sampler: bool = False,
    mask_rate: float = 1.0,
) -> list[RenderedExample]:
    """Run the existing harness once per example and keep the intermediates.

    Delegates entirely to ``eval.harness.evaluate_example`` (sampling, decode,
    and the single build-order oracle on both sides). No pipeline logic is
    duplicated here.

    ``mask_rate`` is the output-canvas mask level ``t`` passed straight through to
    the harness/sampler: the fraction of the canvas masked (model-predicted) with
    the remainder revealed as ground truth. It is stored on each
    ``RenderedExample`` so figure titles can show which value produced them.
    """

    rendered: list[RenderedExample] = []
    for index, example in enumerate(examples):
        result = evaluate_example(
            model=model,
            example=example,
            vocabulary=vocabulary,
            config=config,
            device=device,
            include_canvas_logits=include_canvas_logits,
            bypass_sampler=bypass_sampler,
            mask_rate=mask_rate,
        )
        rendered.append(
            RenderedExample(
                example=example,
                result=result,
                label=_label(example, index),
                output_mask=mask_rate,
            )
        )
    return rendered


def _label(example: DatasetExample, index: int) -> str:
    """Build a short identifier for titles/filenames from example identity."""

    replay_id = example.replay_id or (example.replay_path.stem if example.replay_path else f"ex{index}")
    return f"{replay_id}_{example.perspective_player}_t{example.window_start}"


# ---------------------------------------------------------------------------
# Count-grid assembly (turns the harness count grids into a dense matrix)
# ---------------------------------------------------------------------------


def _counts_matrix(
    predicted: Sequence[TimedTimestep],
    ground_truth: Sequence[TimedTimestep],
) -> tuple[list[str], list[int], list[list[float]]]:
    """Assemble a signed diff matrix from two per-timestep count grids.

    Parameters:
        predicted: predicted per-timestep entity-type counts (model resolution).
        ground_truth: ground-truth per-timestep entity-type counts (same
            resolution -- both are decoded canvases).

    Returns:
        ``(entity_types, buckets, matrix)`` where ``matrix[i][j]`` is
        ``predicted_count - ground_truth_count`` for entity type ``i`` at
        timestep bucket ``j``. Types are sorted; buckets span the union of both
        grids. Missing cells count as zero.
    """

    entity_types = sorted(
        {name for step in predicted for name in step.counts}
        | {name for step in ground_truth for name in step.counts}
    )
    n_buckets = max(len(predicted), len(ground_truth))
    buckets = list(range(n_buckets))

    def _at(grid: Sequence[TimedTimestep], bucket: int, name: str) -> float:
        if bucket >= len(grid):
            return 0.0
        return float(grid[bucket].counts.get(name, 0))

    matrix = [
        [_at(predicted, bucket, name) - _at(ground_truth, bucket, name) for bucket in buckets]
        for name in entity_types
    ]
    return entity_types, buckets, matrix


def _figure_dimensions(
    n_cols: int,
    n_rows: int,
    *,
    col_size: float,
    row_size: float,
    min_w: float,
    min_h: float,
    max_w: float,
    max_h: float,
) -> tuple[float, float]:
    """Size a figure so it scales with its column/row counts (readability).

    Long (many timesteps) or tall (many entity types) grids get proportionally
    larger canvases, clamped to sane bounds. Combined with vector (SVG/PDF)
    output this keeps large grids legible instead of squashing every grid into a
    fixed box.
    """

    width = min(max_w, max(min_w, col_size * max(1, n_cols)))
    height = min(max_h, max(min_h, row_size * max(1, n_rows)))
    return width, height


# ---------------------------------------------------------------------------
# Figure A -- aligned prediction-vs-ground-truth count comparison
# ---------------------------------------------------------------------------


def _count_grid(
    steps: Sequence[TimedTimestep], entity_types: Sequence[str], buckets: Sequence[int]
) -> torch.Tensor:
    """Return a dense entity-by-timestep count grid for one decoded canvas."""

    return torch.tensor(
        [
            [float(steps[bucket].counts.get(name, 0)) if bucket < len(steps) else 0.0 for bucket in buckets]
            for name in entity_types
        ]
    )


def _sparse_bucket_ticks(buckets: Sequence[int], *, maximum: int = 12) -> list[int]:
    """Choose readable x ticks without labelling every column of a long window."""

    if len(buckets) <= maximum:
        return list(buckets)
    stride = max(1, (len(buckets) - 1 + maximum - 2) // (maximum - 1))
    ticks = list(range(0, len(buckets), stride))
    if ticks[-1] != len(buckets) - 1:
        ticks.append(len(buckets) - 1)
    return ticks


def plot_count_comparison(rendered: RenderedExample):
    """Render aligned truth, prediction, and signed-error count heatmaps.

    Truth and prediction use the same quantitative scale and saturated
    ``cividis`` palette. The error panel deliberately encodes only direction
    with three high-contrast colors (under / exact / over); signed magnitudes
    are printed in mismatched cells when the grid is small enough.

    Parameters:
        rendered: a ``RenderedExample`` carrying the harness count grids.

    Returns:
        The matplotlib ``Figure``. The caller owns saving/closing it.
    """

    entity_types, buckets, _ = _counts_matrix(
        rendered.result.predicted_counts,
        rendered.result.ground_truth_counts,
    )
    width = min(18.0, max(10.0, len(buckets) * 0.18))
    height = min(30.0, max(8.0, 5.0 + len(entity_types) * 0.72))
    figure, axes = plt.subplots(3, 1, figsize=(width, height), sharex=True, sharey=True)

    mask_tag = f"mask t={rendered.output_mask:.2f}"
    if not entity_types or not buckets:
        axis = axes[0]
        axis.text(0.5, 0.5, "no decoded timesteps\n(prediction invalid?)", ha="center", va="center")
        for item in axes:
            item.set_axis_off()
        figure.suptitle(f"prediction vs ground truth  {rendered.label}  ({mask_tag})")
        figure.tight_layout()
        return figure

    truth = _count_grid(rendered.result.ground_truth_counts, entity_types, buckets)
    prediction = _count_grid(rendered.result.predicted_counts, entity_types, buckets)
    diff = prediction - truth
    count_limit = max(1.0, float(torch.maximum(truth.max(), prediction.max()).item()))
    count_image = None
    for axis, data, title in zip(
        axes[:2], (truth, prediction), ("GROUND TRUTH counts", "MODEL PREDICTION counts")
    ):
        count_image = axis.imshow(
            data.numpy(), aspect="auto", cmap="cividis", vmin=0.0, vmax=count_limit, interpolation="nearest"
        )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")

    status = torch.sign(diff)
    status_cmap = ListedColormap(["#0067A5", "#D8D8D8", "#D1495B"])
    status_image = axes[2].imshow(
        status.numpy(), aspect="auto", cmap=status_cmap, vmin=-1.5, vmax=1.5, interpolation="nearest"
    )
    axes[2].set_title("ERROR direction  (signed count shown where space permits)", loc="left", fontsize=11, fontweight="bold")
    if diff.numel() <= 300:
        for row in range(diff.shape[0]):
            for col in range(diff.shape[1]):
                value = int(diff[row, col].item())
                if value:
                    axes[2].text(col, row, f"{value:+d}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    tick_positions = _sparse_bucket_ticks(buckets)
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels([buckets[index] for index in tick_positions], fontsize=9)
    axes[2].set_xlabel("timestep bucket (model resolution)", fontsize=10)
    for axis in axes:
        axis.set_yticks(range(len(entity_types)))
        axis.set_yticklabels(entity_types, fontsize=9)
        axis.set_ylabel("entity type", fontsize=9)
        axis.set_xticks([index - 0.5 for index in range(1, len(buckets))], minor=True)
        axis.set_yticks([index - 0.5 for index in range(1, len(entity_types))], minor=True)
        axis.grid(which="minor", color="#FFFFFF", linewidth=0.35, alpha=0.45)
        axis.tick_params(which="minor", bottom=False, left=False)

    valid = "valid" if rendered.result.prediction_valid else "INVALID prediction"
    exact = int((diff == 0).sum().item())
    total = diff.numel()
    mae = float(diff.abs().mean().item())
    figure.suptitle(
        f"{rendered.label}  |  {valid}  |  {mask_tag}  |  exact cells {exact}/{total} ({100.0 * exact / total:.1f}%)  |  MAE {mae:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    assert count_image is not None
    count_bar = figure.colorbar(count_image, ax=axes[:2], fraction=0.018, pad=0.015)
    count_bar.set_label("entity count")
    status_bar = figure.colorbar(status_image, ax=axes[2], fraction=0.018, pad=0.015, ticks=[-1, 0, 1])
    status_bar.ax.set_yticklabels(["UNDER", "EXACT", "OVER"])
    figure.subplots_adjust(left=0.18, right=0.91, top=0.92, bottom=0.07, hspace=0.26)
    return figure


def plot_mean_abs_diff_heatmap(rendered: Sequence[RenderedExample]):
    """Render the optional aggregate mean-absolute-diff heatmap over the set.

    For every (entity type, timestep bucket) cell, averages ``|predicted -
    ground_truth|`` across all rendered windows. A sequential colormap shows
    where, on average, predictions deviate most from ground truth. One figure
    for the whole set.

    Parameters:
        rendered: all rendered examples.

    Returns:
        The matplotlib ``Figure``, or ``None`` if there is nothing to aggregate.
    """

    # Union of entity types and the max bucket count across the set.
    entity_types: set[str] = set()
    max_buckets = 0
    per_example: list[tuple[list[str], list[int], list[list[float]]]] = []
    for item in rendered:
        types, buckets, matrix = _counts_matrix(
            item.result.predicted_counts, item.result.ground_truth_counts
        )
        per_example.append((types, buckets, matrix))
        entity_types.update(types)
        max_buckets = max(max_buckets, len(buckets))
    if not entity_types or max_buckets == 0:
        return None

    ordered_types = sorted(entity_types)
    type_index = {name: row for row, name in enumerate(ordered_types)}
    accum = torch.zeros(len(ordered_types), max_buckets)
    for types, buckets, matrix in per_example:
        for row, name in enumerate(types):
            for col in buckets:
                accum[type_index[name], col] += abs(matrix[row][col])
    mean_abs = accum / max(1, len(per_example))

    width = min(18.0, max(10.0, max_buckets * 0.18))
    height = min(20.0, max(5.0, 2.5 + len(ordered_types) * 0.42))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(mean_abs.numpy(), aspect="auto", cmap="inferno", interpolation="nearest")
    tick_positions = _sparse_bucket_ticks(list(range(max_buckets)))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_positions, fontsize=9)
    axis.set_yticks(range(len(ordered_types)))
    axis.set_yticklabels(ordered_types, fontsize=9)
    axis.set_xlabel("timestep bucket (model resolution)")
    axis.set_ylabel("entity type")
    mask_tag = f"mask t={rendered[0].output_mask:.2f}" if rendered else "mask t=?"
    axis.set_title(
        f"mean |predicted - ground-truth| over {len(per_example)} window(s)  ({mask_tag})"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("mean absolute count diff")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Figure B -- build-order first-appearance timeline
# ---------------------------------------------------------------------------


def _first_appearance(events: Sequence[BuildOrderEvent]) -> dict[str, int]:
    """Reduce a build-order event set to the first bucket per entity type."""

    first: dict[str, int] = {}
    for event in events:
        if event.entity_type not in first or event.bucket < first[event.entity_type]:
            first[event.entity_type] = event.bucket
    return first


def plot_first_appearance_timeline(rendered: RenderedExample, *, tolerance_buckets: int):
    """Render Figure B: predicted vs ground-truth first-appearance timeline.

    One row per entity type. A ground-truth marker and a predicted marker sit at
    each type's first-appearance bucket, with a shaded timing-tolerance band of
    +/- ``tolerance_buckets`` around the ground-truth bucket. A predicted marker
    missing where ground truth exists is a false negative; a predicted-only type
    is a false positive; both within the band is a hit; both but outside the
    band is a timing miss.

    Parameters:
        rendered: the ``RenderedExample`` carrying predicted/ground-truth events.
        tolerance_buckets: timing tolerance (from ``config.eval``), the same
            tolerance the reported metrics use.

    Returns:
        The matplotlib ``Figure``.
    """

    predicted = _first_appearance(rendered.result.predicted_events)
    ground_truth = _first_appearance(rendered.result.ground_truth_events)
    entity_types = sorted(set(predicted) | set(ground_truth))

    width, height = _figure_dimensions(
        max(len(predicted), len(ground_truth), 1),
        len(entity_types),
        col_size=0.0,  # width is driven by bucket span below, not type count
        row_size=0.34,
        min_w=8.0,
        min_h=3.0,
        max_w=40.0,
        max_h=40.0,
    )
    # Widen with the temporal span so long games stay readable.
    max_bucket = max(
        [bucket for bucket in list(predicted.values()) + list(ground_truth.values())] + [1]
    )
    width = min(40.0, max(width, 0.28 * (max_bucket + 2 * tolerance_buckets + 2)))
    figure, axis = plt.subplots(figsize=(width, height))

    mask_tag = f"mask t={rendered.output_mask:.2f}"
    if not entity_types:
        axis.text(0.5, 0.5, "no build-order events", ha="center", va="center")
        axis.set_axis_off()
        figure.suptitle(f"first appearance  {rendered.label}  ({mask_tag})")
        figure.tight_layout()
        return figure

    # Track which classification colors are actually used, for a lean legend.
    used: dict[str, str] = {}

    def _mark(kind: str, color: str) -> None:
        used[kind] = color

    for row, name in enumerate(entity_types):
        gt_bucket = ground_truth.get(name)
        pred_bucket = predicted.get(name)

        if gt_bucket is not None:
            # Tolerance band around the ground-truth first-appearance bucket.
            axis.axhspan(
                row - 0.35,
                row + 0.35,
                xmin=0,
                xmax=1,
                color="none",
            )
            axis.add_patch(
                plt.Rectangle(
                    (gt_bucket - tolerance_buckets - 0.5, row - 0.35),
                    2 * tolerance_buckets + 1,
                    0.7,
                    color="#cccccc",
                    alpha=0.5,
                    zorder=0,
                )
            )
            axis.scatter([gt_bucket], [row], marker="o", s=60, color="#1f77b4", zorder=3, label="_gt")
        if pred_bucket is not None:
            axis.scatter([pred_bucket], [row], marker="X", s=60, color="#333333", zorder=4, label="_pred")

        # Classify the row and color-connect predicted<->ground-truth.
        if gt_bucket is not None and pred_bucket is not None:
            within = abs(pred_bucket - gt_bucket) <= tolerance_buckets
            color = "#2ca02c" if within else "#ff7f0e"
            _mark("hit" if within else "timing miss", color)
            axis.plot([gt_bucket, pred_bucket], [row, row], color=color, linewidth=2, zorder=2)
        elif gt_bucket is not None:
            _mark("false negative (missed)", "#1f77b4")
        elif pred_bucket is not None:
            _mark("false positive (spurious)", "#d62728")
            axis.scatter([pred_bucket], [row], marker="X", s=90, facecolors="none", edgecolors="#d62728", zorder=5)

    axis.set_yticks(range(len(entity_types)))
    axis.set_yticklabels(entity_types, fontsize=7)
    axis.set_ylim(-0.6, len(entity_types) - 0.4)
    axis.set_xlabel(f"first-appearance timestep bucket  (tolerance +/-{tolerance_buckets})")
    axis.set_ylabel("entity type")
    axis.set_title(f"build-order first appearance  |  {rendered.label}  ({mask_tag})")
    axis.grid(axis="x", linestyle=":", alpha=0.4)

    # Build a compact legend: markers + any classification colors seen.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color="#1f77b4", label="ground truth"),
        plt.Line2D([], [], marker="X", linestyle="none", color="#333333", label="predicted"),
    ]
    for kind, color in used.items():
        handles.append(plt.Line2D([], [], color=color, linewidth=3, label=kind))
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Rendering + saving
# ---------------------------------------------------------------------------


def render_figures(
    rendered: Sequence[RenderedExample],
    out_dir: str | Path,
    *,
    tolerance_buckets: int,
    dpi: int,
    include_first_appearance: bool = False,
) -> list[Path]:
    """Render count comparisons, optional debut timelines, and the aggregate.

    Each figure is written as PNG (raster, required deliverable) and SVG
    (vector, so long/wide grids stay readable at any zoom). All figures are also
    collected into one combined multi-page PDF.

    Parameters:
        rendered: the evaluated windows to plot.
        out_dir: output directory; created if missing. The only write target.
        tolerance_buckets: timing tolerance for the optional debut timeline.
        dpi: raster resolution for the PNG exports.
        include_first_appearance: render debut timelines only when explicitly
            requested for a compatible fine-tuned model.

    Returns:
        The list of written file paths (PNGs, SVGs, and the combined PDF).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdf_path = out_path / "diagnostics.pdf"

    with PdfPages(pdf_path) as pdf:
        for item in rendered:
            figures = [(f"prediction_vs_truth_{item.label}", plot_count_comparison(item))]
            if include_first_appearance:
                figures.append(
                    (
                        f"first_appearance_{item.label}",
                        plot_first_appearance_timeline(item, tolerance_buckets=tolerance_buckets),
                    )
                )
            for name, figure in figures:
                written.extend(_save_figure(figure, out_path, name, dpi=dpi))
                pdf.savefig(figure)
                plt.close(figure)

        aggregate = plot_mean_abs_diff_heatmap(rendered)
        if aggregate is not None:
            written.extend(_save_figure(aggregate, out_path, "mean_abs_diff_aggregate", dpi=dpi))
            pdf.savefig(aggregate)
            plt.close(aggregate)

    written.append(pdf_path)
    return written


def _save_figure(figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    """Write one figure as both PNG (raster) and SVG (vector)."""

    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in stem)
    png_path = out_dir / f"{safe}.png"
    svg_path = out_dir / f"{safe}.svg"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")  # vector output for readability
    return [png_path, svg_path]


# ---------------------------------------------------------------------------
# Optional raw-canvas and logit exports
# ---------------------------------------------------------------------------


_SPECIAL_ID_TO_TOKEN = {token_id: token for token, token_id in SPECIAL_TOKENS.items()}


def _token_name(token_id: int, vocabulary: ContentVocabulary) -> str:
    """Resolve a special/content token id without decoding canvas grammar."""

    if token_id in _SPECIAL_ID_TO_TOKEN:
        return _SPECIAL_ID_TO_TOKEN[token_id]
    try:
        return vocabulary.token_name_for(token_id)
    except KeyError:
        return f"[UNKNOWN:{token_id}]"


def write_canvas_comparison_csv_files(
    rendered: Sequence[RenderedExample],
    vocabulary: ContentVocabulary,
    out_dir: str | Path,
) -> list[Path]:
    """Write side-by-side prediction-vs-truth rows for the rendered examples.

    A single rendered window keeps its labelled filename and existing four-column
    schema. Multiple windows are concatenated into ``canvas_comparison.csv``;
    that aggregate adds a leading ``window`` column so every row remains
    attributable to its replay/perspective/window. Canvas positions reset to zero
    for each window. Columns:

      * ``window`` -- rendered-example label (multi-window aggregate only).
      * ``sequenceindex`` -- the position in the canvas (0-based).
      * ``modelprediction`` -- the human-readable name of the token the model
        predicted at that position.
      * ``groundtruth`` -- the human-readable name of the ground-truth token at
        that position.
      * ``correct`` -- ``True`` if the model predicted the token correctly,
        ``False`` if it predicted the wrong token, or ``Unmasked`` if the
        position was never masked (revealed as ground truth under
        ``--output-mask`` < 1.0, i.e. handed to the model unchanged and so not
        actually predicted).

    The correctness test compares the raw token ids (not the resolved names) so
    it is exact even if two distinct ids ever mapped to the same display name.
    ``Unmasked`` positions are reported distinctly rather than as ``True`` so a
    reader never mistakes a revealed token for a genuine model prediction.

    Parameters:
        rendered: the evaluated windows; each carries the predicted and
            ground-truth canvases (equal-length token-id sequences).
        vocabulary: content vocabulary used to resolve token ids to names.
        out_dir: output directory; created if missing. The only write target.

    Returns:
        One path: ``canvas_comparison_<label>.csv`` for a single example or
        ``canvas_comparison.csv`` for a multi-window aggregate. No paths when
        ``rendered`` is empty.

    Calls:
        ``_token_name`` (id -> human-readable name).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not rendered:
        return []

    aggregate = len(rendered) > 1
    csv_path = out_path / (
        "canvas_comparison.csv" if aggregate else f"canvas_comparison_{rendered[0].label}.csv"
    )
    header = ["sequenceindex", "modelprediction", "groundtruth", "correct"]
    if aggregate:
        header.insert(0, "window")

    # newline="" is the documented way to let the csv module control line
    # endings itself (avoids blank rows on Windows).
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for item in rendered:
            # Which positions were revealed as ground truth (unmasked) rather
            # than predicted by the model. Same provenance the JSON export uses.
            revealed = item.result.predicted_canvas_revealed_mask
            for index, (predicted_id, truth_id) in enumerate(
                zip(item.result.predicted_canvas, item.result.ground_truth_canvas)
            ):
                # Unmasked positions were never predicted, so they are neither a
                # correct nor an incorrect model call -- flag them distinctly.
                if index < len(revealed) and revealed[index]:
                    correctness: bool | str = "Unmasked"
                else:
                    correctness = predicted_id == truth_id
                row: list[object] = [
                    index,
                    _token_name(predicted_id, vocabulary),
                    _token_name(truth_id, vocabulary),
                    correctness,
                ]
                if aggregate:
                    row.insert(0, item.label)
                writer.writerow(row)
    return [csv_path]


def _allegiance_marker(allegiance: str | None) -> str:
    """Map a token's allegiance to a fixed-width, human-readable side marker.

    The model input canvas interleaves the perspective player's own tokens with
    the (fogged) opponent tokens. ``TokenRecord.allegiance`` records which side a
    token belongs to -- ``"self"`` for the perspective player, ``"enemy"`` for the
    opponent, and ``None`` for structural tokens such as ``[DELIMITER]`` that
    belong to neither. This returns a padded, upper-case tag so the two sides line
    up in columns and are unmistakable to a human reading the raw ``.txt`` dump.

    Parameters:
        allegiance: the ``allegiance`` field of a ``TokenRecord`` ("self",
            "enemy", or ``None``).

    Returns:
        ``"SELF "``, ``"ENEMY"``, or ``"-----"`` (delimiter / no side).
    """

    if allegiance == "self":
        return "SELF "
    if allegiance == "enemy":
        return "ENEMY"
    return "-----"


def write_input_canvas_text_files(
    rendered: Sequence[RenderedExample],
    out_dir: str | Path,
) -> list[Path]:
    """Write annotated model-input canvases, consolidating multiple windows.

    This is the sequence actually fed to the model (``DatasetExample.input_records``
    -> ``input_token_ids``): the perspective player's own tokens followed by the
    fog-filtered opponent tokens, with ``[DELIMITER]`` tokens separating
    timesteps. Every line is tagged with a self/enemy marker (see
    ``_allegiance_marker``) so a human reader can immediately tell which side each
    token came from -- the key ambiguity when eyeballing a raw input dump. Written
    alongside the other output artifacts under ``out_dir``.

    Parameters:
        rendered: the evaluated windows; each carries the ``DatasetExample`` whose
            ``input_records`` are the model's input tokens (in order).
        out_dir: output directory; created if missing. The only write target.

    Returns:
        One path: ``input_canvas_<label>.txt`` for a single example or
        ``input_canvas.txt`` containing labelled sections for multiple examples.
        No paths when ``rendered`` is empty.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not rendered:
        return []

    aggregate = len(rendered) > 1
    path = out_path / ("input_canvas.txt" if aggregate else f"input_canvas_{rendered[0].label}.txt")
    # Header comments explain the columns and the self/enemy convention so the
    # file is self-describing without needing this source.
    lines = [
        "# model input canvas (perspective-self tokens, then fog-filtered enemy tokens)",
        "# columns: index<TAB>token_id<TAB>side<TAB>token_name",
        "# side: SELF = perspective player, ENEMY = opponent, ----- = [DELIMITER]/structural",
    ]
    for item in rendered:
        if aggregate:
            lines.extend(["", f"# window: {item.label}"])
        if not item.example.input_records:
            # Pre-training: input is literally absent, so there are no input
            # records to dump. Emit an explicit marker rather than a silently
            # empty section so a reader knows the absence is intentional.
            lines.append("# (no input -- pre-training example has absent input)")
            continue
        for index, record in enumerate(item.example.input_records):
            marker = _allegiance_marker(record.allegiance)
            lines.append(f"{index}\t{record.token_id}\t{marker}\t{record.token_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [path]


def write_logits_json(
    rendered: Sequence[RenderedExample],
    vocabulary: ContentVocabulary,
    out_dir: str | Path,
    *,
    top_k: int = 5,
) -> Path:
    """Write final-canvas top-k raw logits and softmax confidence per position."""

    examples: list[dict[str, object]] = []
    for item in rendered:
        logits = item.result.final_canvas_logits
        if logits is None:
            raise ValueError("final canvas logits were not requested during evaluation")
        probabilities = torch.softmax(logits.float(), dim=-1)
        k = min(top_k, logits.shape[-1])
        top_logits, top_ids = torch.topk(logits, k=k, dim=-1)
        top_confidences = torch.gather(probabilities, dim=-1, index=top_ids)
        revealed = item.result.predicted_canvas_revealed_mask
        positions: list[dict[str, object]] = []
        for index, (predicted_id, truth_id) in enumerate(
            zip(item.result.predicted_canvas, item.result.ground_truth_canvas)
        ):
            candidates = [
                {
                    "token": _token_name(int(token_id), vocabulary),
                    "token_id": int(token_id),
                    "logit": float(logit),
                    "confidence": float(confidence),
                }
                for token_id, logit, confidence in zip(
                    top_ids[index].tolist(),
                    top_logits[index].tolist(),
                    top_confidences[index].tolist(),
                )
            ]
            positions.append(
                {
                    "sequence_index": index,
                    # MODEL = predicted by the model; TRUTH = revealed ground truth
                    # (an unmasked position under --output-mask < 1.0).
                    "source": "TRUTH" if index < len(revealed) and revealed[index] else "MODEL",
                    "predicted_token": _token_name(predicted_id, vocabulary),
                    "predicted_token_id": predicted_id,
                    "ground_truth_token": _token_name(truth_id, vocabulary),
                    "ground_truth_token_id": truth_id,
                    "top_k": candidates,
                }
            )
        examples.append({"label": item.label, "positions": positions})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "canvas_logits.json"
    path.write_text(
        json.dumps({"top_k": top_k, "examples": examples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    *,
    checkpoint: str | Path,
    replay_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    n_replays: int,
    n_windows: int,
    fog_rate: float | None,
    dpi: int,
    device: str,
    write_json: bool = False,
    write_csv: bool = False,
    write_input: bool = False,
    write_first_appearance: bool = False,
    bypass_sampler: bool = False,
    use_raw: bool = False,
    output_masks: Sequence[float] = (0.5,),
) -> list[Path]:
    """Full read-only render: ingest -> load weights -> evaluate -> plot -> save.

    Parameters mirror the CLI flags. ``fog_rate`` of ``None`` uses the eval fog
    condition from config (``config.eval.fog_rate``). ``use_raw`` selects the raw
    model weights instead of the default EMA weights.

    ``output_masks`` is one or more output-canvas mask rates ``t`` in ``[0, 1]``:
    the fraction of the canvas masked (model-predicted), with the remainder
    revealed as ground truth. Ingestion and weight loading happen ONCE; the model
    is then evaluated and rendered once per mask rate. With a single rate the
    artifacts are written flat into ``out_dir`` (the original layout); with two or
    more rates each rate's artifacts go under an ``output_mask_<t>/`` subdirectory
    so the sweeps never collide. The default ``(0.5,)`` is the pre-training average
    mask rate (``t ~ U(0, 1)``). Returns every written path across all rates.
    """

    user_config = load_config(config_path)
    effective_fog = user_config.eval.fog_rate if fog_rate is None else fog_rate

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Scratch subdir under out-dir for derived tokenization artifacts, so the run
    # writes nothing outside out-dir and mutates no source data.
    scratch = out_path / "_ingest_cache"
    artifact_root = scratch / "tokenized_replays"
    manifest_path = scratch / "window_manifest.jsonl"

    torch_device = torch.device(device)
    model, model_config = load_diagnostic_model(
        checkpoint, user_config, device=torch_device, use_raw=use_raw
    )
    print(f"loaded {'RAW' if use_raw else 'EMA'} weights from {Path(checkpoint)}", flush=True)

    # Ingest the replays ONCE; every mask rate re-uses the same examples so the
    # only per-rate work is the (cheap relative to ingestion) evaluate + render.
    run_config, vocabulary, examples = ingest_examples(
        model_config,
        replay_dir,
        fog_rate=effective_fog,
        n_replays=n_replays,
        n_windows=n_windows,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
    )

    mask_rates = list(output_masks) if output_masks else [0.5]
    # A single rate stays flat in out_dir (original layout); multiple rates are
    # split into per-rate subdirectories so their figures/text/json never clash.
    nest_per_rate = len(mask_rates) > 1
    written: list[Path] = []
    for mask_rate in mask_rates:
        target_dir = out_path / f"output_mask_{mask_rate:.2f}" if nest_per_rate else out_path
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"rendering output-mask t={mask_rate:.2f} -> {target_dir}", flush=True)
        rendered = evaluate_selected(
            model,
            examples,
            vocabulary,
            run_config,
            device=torch_device,
            include_canvas_logits=write_json,
            bypass_sampler=bypass_sampler,
            mask_rate=mask_rate,
        )
        written.extend(
            render_figures(
                rendered,
                target_dir,
                tolerance_buckets=run_config.eval.timing_tolerance_buckets,
                dpi=dpi,
                include_first_appearance=write_first_appearance,
            )
        )
        if write_input:
            written.extend(write_input_canvas_text_files(rendered, target_dir))
        if write_csv:
            written.extend(write_canvas_comparison_csv_files(rendered, vocabulary, target_dir))
        if write_json:
            written.append(write_logits_json(rendered, vocabulary, target_dir))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render read-only diagnostic figures from a checkpoint + replay dir."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint written by the training loop")
    parser.add_argument("--replay-dir", type=Path, required=True, help="dir of extractor *_game_state.parquet files")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="the existing project YAML")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory (only write target)")
    parser.add_argument("--n-replays", type=int, default=2, help="how many replays from --replay-dir to ingest (0 = all)")
    parser.add_argument(
        "--n-windows",
        type=int,
        default=1,
        help="windows to render per selected replay (0 = all)",
    )
    parser.add_argument(
        "--fog-rate",
        type=float,
        default=None,
        help="fog rate applied to every example; default = config.eval.fog_rate",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG raster resolution")
    parser.add_argument("--device", type=str, default="cuda", help="torch device (cpu/cuda)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="write canvas_logits.json with final-canvas top-10 logits and confidence values",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help=(
            "write side-by-side comparison CSV data; multiple windows are consolidated "
            "with a window label column"
        ),
    )
    parser.add_argument(
        "--show-input",
        action="store_true",
        help="write model input canvas text; multiple windows become labelled sections in one file",
    )
    parser.add_argument(
        "--first-appearance",
        action="store_true",
        help="also render first-appearance timelines (only meaningful for debut/build-order fine-tuned models)",
    )
    parser.add_argument(
        "--bypass-sampler",
        action="store_true",
        help="replace iterative sampling with one denoising forward pass (per --output-mask rate)",
    )
    parser.add_argument(
        "--output-mask",
        type=float,
        nargs="+",
        default=[0.5],
        metavar="RATE",
        help=(
            "output-canvas mask rate(s) t in [0,1]: the fraction of the canvas "
            "masked (model-predicted); the rest is revealed as ground truth. Pass "
            "several to run inference once per rate, e.g. --output-mask 0.4 0.9 1.0 "
            "(each rate's artifacts land in its own output_mask_<t>/ subdir). "
            "Default 0.5 = the pre-training average mask rate (t~U(0,1)); 1.0 = "
            "fully masked (the previous behavior)"
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "use the raw (final optimizer step) model weights; default (flag "
            "omitted) deterministically uses the EMA weights the sampler serves"
        ),
    )
    args = parser.parse_args(argv)

    # Reject nonsensical mask rates early with a clear message rather than letting
    # them silently degrade inside the corruption schedule.
    for rate in args.output_mask:
        if not 0.0 <= rate <= 1.0:
            parser.error(f"--output-mask values must be in [0, 1]; got {rate}")

    written = run(
        checkpoint=args.checkpoint,
        replay_dir=args.replay_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        n_replays=args.n_replays,
        n_windows=args.n_windows,
        fog_rate=args.fog_rate,
        dpi=args.dpi,
        device=args.device,
        write_json=args.json,
        write_csv=args.csv,
        write_input=args.show_input,
        write_first_appearance=args.first_appearance,
        bypass_sampler=args.bypass_sampler,
        use_raw=args.raw,
        output_masks=args.output_mask,
    )
    print(f"wrote {len(written)} file(s) to {args.out_dir}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
