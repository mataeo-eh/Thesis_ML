"""Batch-versus-batch interference probe (read-only training diagnostic).

WHAT THIS ANSWERS
-----------------
A capacity-limited model cannot fit every training batch at once: an optimizer
step that helps the batch it was taken on must "pay" for that improvement by
getting *worse* somewhere else. A model with capacity to spare shows no such
tension -- a step on one batch either helps or is neutral everywhere.

This script measures that tension directly on an already-trained checkpoint:

    1. Freeze one epoch's worth of training batches, and freeze one corrupted
       "view" of each (fixed diffusion timestep t, fixed noise draw, fixed
       self-conditioning row mask). Every loss in the whole probe is measured on
       these same frozen views, so nothing moves except the model weights.
    2. Record the baseline loss of the checkpoint on every frozen view.
       -> `pre_loss`
    3. For each batch i: restore the checkpoint exactly, take ONE real optimizer
       step on batch i (same optimizer state, same LR, same precision, same
       gradient clipping the run itself used), then re-measure the loss on
       EVERY batch j.                                        -> `post_loss`
    4. Record `delta_loss = pre_loss - post_loss`, so POSITIVE means the step
       made batch j better and NEGATIVE means the step made batch j worse.

The signal to look for is negative deltas on batches other than the one stepped
on. If `self_delta` is comfortably positive while `mean_other_delta` is negative
-- or while a good share of the other batches show negative deltas -- the
batches are competing for the same parameters, which is the positive signal for
capacity limitation.

WHAT THIS DOES *NOT* DO
-----------------------
Nothing is written back into the run. No checkpoint is saved, no EMA update is
applied, no metrics file of the original run is touched. The model and optimizer
are restored from an in-memory snapshot before every single step, so the probe
is idempotent: the Nth step-batch sees exactly the same weights the 1st did.
Outputs go only to the probe's own output directory.

MEASUREMENT PRECISION (why evaluation defaults to fp32)
-------------------------------------------------------
A single optimizer step at the end of a decayed LR schedule moves the weights
very little, so the loss changes we are trying to read are small. The training
run itself computes losses under bf16 autocast, which carries roughly 3 decimal
digits -- coarse enough to quantize those small changes to zero. So by default:

  * the optimizer STEP runs in the run's own configured precision, because that
    step must be a faithful replica of what training would actually do; but
  * every loss MEASUREMENT runs in fp32, purely to resolve the difference.

Pass `--eval-precision config` to measure in the run's precision instead.

OUTPUTS (in `scripts/output/batch_interference/<arm>/`, existing files archived
first). They live under `scripts/output/` rather than beside the run's own
artifacts because this utility must not write into a completed run's directory.
--------------------------------------------------------------------------
  batch_interference_long.csv     one row per (step batch, evaluated batch)
  batch_interference_matrix.csv   the same deltas as a step-by-eval matrix
  batch_interference_summary.csv  one row per step batch (self vs others)
  batch_interference_meta.json    provenance: checkpoint, step, LR, seeds, drift

USAGE
-----
    ./.venv/Scripts/python.exe scripts/batch_interference_probe.py \
        --config configs/memorization_01_no_regularization.yaml

Calls into: thesis_ml.pipeline.train_pipeline (dataset/loader construction, so
this probe stays byte-identical to how the run built its data), and
thesis_ml.train.loop.TrainingLoop (checkpoint load, corruption, loss).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.feature_stats import load_feature_statistics
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import load_window_manifest
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.storage import StorageResolver
from thesis_ml.pipeline.train_pipeline import (
    _ensure_window_manifest,
    _explicit_replay_selection,
    _local_checkpoint_dir,
    _make_dataloader,
    _materialize_file,
    _materialize_replay_paths,
    _select_replays,
)
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.vocab.content_vocab import load_content_vocabulary


# Multiplier used to spread the per-batch corruption seeds far apart in the
# generator's state space, so batch 0 and batch 1 do not get near-identical
# noise draws. Any large odd number works; this one is a prime.
SEED_STRIDE = 1_000_003


def portable_path(path: Path) -> str:
    """Render paths inside the checkout relative to the current directory."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    """Define and read the probe's command line.

    Returns:
        Parsed arguments. `--config` is the only required one; every other knob
        has a default that reproduces the "faithful resume" reading.

    Called by: main.
    """

    parser = argparse.ArgumentParser(
        description="Measure cross-batch loss interference of a single optimizer step",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="training profile YAML of the arm to probe (e.g. configs/memorization_01_*.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to probe; defaults to <config checkpoint dir>/last.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where CSVs are written; defaults to scripts/output/batch_interference/<arm>",
    )
    parser.add_argument(
        "--epoch-index",
        type=int,
        default=None,
        help=(
            "which epoch's batch order and per-serving draws (fog, window serving) "
            "to freeze; defaults to the epoch the checkpoint would train next"
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="probe only the first N batches of the epoch (0 = all). Use for a smoke run.",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=20260812,
        help="base seed for the frozen corruption views (t draw, noise, self-conditioning)",
    )
    parser.add_argument(
        "--eval-precision",
        choices=("fp32", "config"),
        default="fp32",
        help=(
            "precision for loss MEASUREMENT only. fp32 (default) resolves the small "
            "loss changes a single step produces; 'config' uses the run's own bf16."
        ),
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help=(
            "multiply the checkpoint's current learning rate by this factor for the "
            "probe steps. 1.0 (default) is the faithful resume. A larger value "
            "amplifies an otherwise near-zero step so the interference PATTERN is "
            "readable; it no longer describes a step training would actually take."
        ),
    )
    parser.add_argument(
        "--batch-cache-gb",
        type=float,
        default=1.5,
        help=(
            "if the frozen batches fit in this much VRAM they are held on the GPU "
            "(much faster, as each batch is evaluated once per step batch); "
            "otherwise they stay in host memory. 0 disables GPU caching."
        ),
    )
    return parser.parse_args()


def build_loop_and_batches(
    config: ProjectConfig,
    *,
    checkpoint_path: Path,
    epoch_index: int | None,
    max_batches: int,
) -> tuple[TrainingLoop, list, int]:
    """Rebuild the run's model/optimizer/data exactly, then freeze one epoch of batches.

    Mirrors `train_pipeline._run_real_pipeline` up to the point where it would
    call `fit()`: same replay split, same window manifest, same feature
    statistics, same dataset, same batch sampler. It deliberately reuses that
    module's private helpers rather than reimplementing them, so the probe can
    never drift from how the run itself constructed its data.

    Two deliberate differences from the training pipeline:
      * `num_workers` is forced to 0, so the per-serving RNG draws happen in this
        process and are reproducible without worker-seeding subtleties;
      * feature statistics are only LOADED, never recomputed, keeping the probe
        read-only against `data/processed/`. `loop.load_checkpoint` validates
        that the loaded statistics match the ones the checkpoint was trained
        under, so a stale file fails loudly instead of silently skewing losses.

    Args:
        config: the arm's loaded project configuration.
        checkpoint_path: `last.pt` to restore model + optimizer + scheduler from.
        epoch_index: epoch whose batch order and per-serving draws to freeze, or
            None to use the epoch the checkpoint would train next.
        max_batches: keep only the first N batches (0 = the whole epoch).

    Returns:
        `(loop, batches, epoch_index)` where `batches` is a list of collated
        DiffusionBatch objects held in host memory.

    Calls: load_content_vocabulary, _materialize_file, _materialize_replay_paths,
    _ensure_window_manifest, _explicit_replay_selection / split_replays +
    _select_replays, load_window_manifest, load_feature_statistics,
    SC2DiffusionDataset, _make_dataloader, SC2StrategyDiffusionModel,
    TrainingLoop.load_checkpoint.
    """

    resolver = StorageResolver()
    torch.manual_seed(config.pipeline.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_dictionary = _materialize_file(
        config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver
    )
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)

    # Same two mutually exclusive routes the training pipeline takes: an
    # explicitly named replay subset (what the memorization profiles use), or
    # the reproducible seeded split.
    explicit = _explicit_replay_selection(replay_paths, config)
    if explicit is not None:
        train_replays, _dev_replays, _test_replays = explicit
    else:
        split = split_replays(
            replay_paths,
            seed=config.pipeline.split_seed,
            test_fraction=config.pipeline.test_fraction,
            dev_fraction=config.pipeline.dev_fraction,
            train_count=config.pipeline.train_replay_count,
            dev_count=config.pipeline.validation_replay_count,
        )
        train_replays, _dev_replays = _select_replays(list(split.train), list(split.dev), config)

    train_windows = load_window_manifest(
        config.data.window_manifest_path, config=config, replay_paths=train_replays
    )
    feature_statistics = load_feature_statistics(
        config.data.feature_statistics_path,
        expected_source_replay_ids=[Path(path).name for path in train_replays],
    )

    # Force single-process loading so the frozen batches are reproducible from
    # this process's RNG alone.
    loader_config = replace(config, pipeline=replace(config.pipeline, num_workers=0))
    train_dataset = SC2DiffusionDataset(
        train_windows,
        loader_config,
        vocabulary,
        seed=config.pipeline.seed,
        fog_rate_override=None,
    )
    train_loader = _make_dataloader(train_dataset, loader_config, shuffle=True, device=device)

    # The loop needs `train.max_steps` populated the same way the run populated
    # it (`len(loader) * epochs` when the YAML leaves it at 0), because the LR
    # scheduler's decay horizon is derived from it and we are restoring that
    # scheduler's state.
    planned_steps = config.train.max_steps
    if planned_steps <= 0:
        planned_steps = len(train_loader) * config.train.epochs
    training_config = replace(
        config,
        train=replace(
            config.train,
            checkpoint_dir=str(_local_checkpoint_dir(config, resolver)),
            max_steps=planned_steps,
        ),
    )

    model = SC2StrategyDiffusionModel(
        training_config,
        vocab_size=vocabulary.vocab_size,
        feature_statistics=feature_statistics,
    )
    # No metrics paths and no publishers: this loop must never write anything.
    loop = TrainingLoop(
        model=model,
        config=training_config,
        device=device,
        seed=config.pipeline.seed,
    )
    loop.load_checkpoint(checkpoint_path)

    # Freeze one epoch: same per-epoch dataset draws and same shuffled batch
    # order the run would have used for this epoch.
    resolved_epoch = loop.completed_epochs if epoch_index is None else epoch_index
    train_dataset.set_epoch(resolved_epoch)
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(resolved_epoch)

    batches = []
    for batch in train_loader:
        batches.append(batch)
        if max_batches > 0 and len(batches) >= max_batches:
            break
    return loop, batches, resolved_epoch


def batch_nbytes(batch) -> int:
    """Total bytes of every tensor field on a collated batch.

    Used to decide whether the frozen batches can be parked in VRAM (each batch
    is re-evaluated once per step batch, so avoiding N^2 host->device copies is
    worth real time).

    Args:
        batch: a DiffusionBatch.

    Returns:
        Sum of `element_size() * numel()` over its tensors, including the nested
        InputFeatures tensors.

    Called by: cache_batches_on_device.
    """

    total = 0
    for value in vars(batch).values():
        if torch.is_tensor(value):
            total += value.element_size() * value.numel()
        elif hasattr(value, "__dict__"):  # InputFeatures and friends
            for nested in vars(value).values():
                if torch.is_tensor(nested):
                    total += nested.element_size() * nested.numel()
    return total


def cache_batches_on_device(batches: list, device: torch.device, budget_gb: float) -> tuple[list, bool]:
    """Move the frozen batches to `device` when they fit inside `budget_gb`.

    Args:
        batches: frozen batches in host memory.
        device: the training device.
        budget_gb: VRAM the caller is willing to spend; 0 disables caching.

    Returns:
        `(batches, cached)` -- the same list (moved) and whether caching happened.

    Calls: batch_nbytes, thesis_ml.train.loop.move_batch_to_device.
    """

    from thesis_ml.train.loop import move_batch_to_device

    if device.type != "cuda" or budget_gb <= 0:
        return batches, False
    total_bytes = sum(batch_nbytes(batch) for batch in batches)
    if total_bytes > budget_gb * 1024**3:
        print(
            f"batch_cache=host reason=over_budget bytes={total_bytes / 1024**3:.2f}GiB "
            f"budget={budget_gb:.2f}GiB",
            flush=True,
        )
        return batches, False
    moved = [move_batch_to_device(batch, device) for batch in batches]
    print(f"batch_cache=cuda bytes={total_bytes / 1024**3:.2f}GiB", flush=True)
    return moved, True


def evaluate_all(loop: TrainingLoop, batches: list, seeds: list[int]) -> list[float]:
    """Loss of the CURRENT weights on every frozen batch view.

    Determinism is what makes this comparable across calls: the generator is
    reseeded to that batch's own fixed seed immediately before the loss is
    computed, so the diffusion timestep t, the noise draw, and the
    self-conditioning row mask are byte-identical on every pass. The model is
    put in eval mode (also disabling gradient checkpointing, which the backbone
    only applies while training).

    Args:
        loop: the training loop holding the model, generator, and loss.
        batches: frozen batches.
        seeds: one corruption seed per batch, index-aligned with `batches`.

    Returns:
        One float loss per batch, in batch order.

    Called by: run_probe. Calls: TrainingLoop.compute_batch_loss.
    """

    loop.model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch, seed in zip(batches, seeds):
            loop.generator.manual_seed(seed)
            losses.append(float(loop.compute_batch_loss(batch).loss))
    return losses


def take_one_step(loop: TrainingLoop, batch, seed: int) -> float:
    """Run exactly one faithful optimizer step on a single frozen batch view.

    Replicates what `TrainingLoop.fit` does per step MINUS everything that would
    persist state beyond this call: no scheduler advance, no EMA update, no
    checkpoint write, no metrics line. Gradient clipping is applied exactly as
    configured, because clipping changes the update the step actually makes.

    Args:
        loop: the training loop (weights are mutated in place -- the caller is
            responsible for restoring its snapshot afterwards).
        batch: the frozen batch to step on.
        seed: that batch's fixed corruption seed, so the step is taken on the
            same view the losses are measured on.

    Returns:
        The training-mode loss of the batch before the update was applied.

    Called by: run_probe.
    """

    loop.model.train()
    loop.optimizer.zero_grad(set_to_none=True)
    loop.generator.manual_seed(seed)
    batch_loss = loop.compute_batch_loss(batch)
    batch_loss.loss.backward()
    if loop.config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(loop.model.parameters(), loop.config.train.grad_clip)
    loop.optimizer.step()
    loop.optimizer.zero_grad(set_to_none=True)
    return float(batch_loss.loss.detach())


def archive_existing(output_dir: Path) -> Path | None:
    """Move any previous probe outputs into a timestamped subfolder.

    Run outputs are archived rather than overwritten, so an earlier probe's
    numbers stay available for comparison after a re-run.

    Args:
        output_dir: the probe's output directory (created if missing).

    Returns:
        The archive directory, or None when there was nothing to archive.

    Called by: run_probe.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = output_dir / f"_archive-{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.rename(archive_dir / path.name)
    print(f"archived_previous_outputs dir={portable_path(archive_dir)}", flush=True)
    return archive_dir


def write_outputs(
    output_dir: Path,
    *,
    arm: str,
    pre_losses: list[float],
    post_matrix: list[list[float]],
    step_train_losses: list[float],
    meta: dict,
) -> None:
    """Write the long, matrix, and summary CSVs plus the provenance JSON.

    Args:
        output_dir: destination directory (already archived/created).
        arm: the arm's name, carried in every long/summary row so several arms'
            CSVs can be concatenated without losing which is which.
        pre_losses: baseline loss per batch.
        post_matrix: `post_matrix[i][j]` = loss on batch j after a step on batch i.
        step_train_losses: training-mode loss of batch i at the moment its step
            was taken (reported for reference; it is measured in the run's own
            precision and in train mode, so it is not directly comparable to
            `pre_loss`).
        meta: provenance dictionary written verbatim to the JSON sidecar.

    Called by: run_probe.
    """

    count = len(pre_losses)
    long_path = output_dir / "batch_interference_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "step_batch",
                "eval_batch",
                "is_self",
                "pre_loss",
                "post_loss",
                "delta_loss",
                "relative_delta",
            ]
        )
        for step_index in range(count):
            for eval_index in range(count):
                pre = pre_losses[eval_index]
                post = post_matrix[step_index][eval_index]
                delta = pre - post
                writer.writerow(
                    [
                        arm,
                        step_index,
                        eval_index,
                        int(step_index == eval_index),
                        f"{pre:.10g}",
                        f"{post:.10g}",
                        f"{delta:.10g}",
                        f"{delta / pre:.10g}" if pre else "",
                    ]
                )

    matrix_path = output_dir / "batch_interference_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_batch"] + [f"eval_batch_{index}" for index in range(count)])
        for step_index in range(count):
            writer.writerow(
                [step_index]
                + [
                    f"{pre_losses[eval_index] - post_matrix[step_index][eval_index]:.10g}"
                    for eval_index in range(count)
                ]
            )

    summary_path = output_dir / "batch_interference_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "step_batch",
                "step_train_loss",
                "self_pre_loss",
                "self_post_loss",
                "self_delta",
                "other_count",
                "other_improved",
                "other_hurt",
                "other_unchanged",
                "mean_other_delta",
                "median_other_delta",
                "sum_other_delta",
                "worst_other_delta",
                "worst_other_batch",
                "best_other_delta",
                "best_other_batch",
                "net_delta_all_batches",
            ]
        )
        for step_index in range(count):
            deltas = [
                pre_losses[eval_index] - post_matrix[step_index][eval_index]
                for eval_index in range(count)
            ]
            self_delta = deltas[step_index]
            others = [
                (eval_index, delta)
                for eval_index, delta in enumerate(deltas)
                if eval_index != step_index
            ]
            other_deltas = [delta for _, delta in others]
            worst_index, worst_delta = min(others, key=lambda pair: pair[1])
            best_index, best_delta = max(others, key=lambda pair: pair[1])
            writer.writerow(
                [
                    arm,
                    step_index,
                    f"{step_train_losses[step_index]:.10g}",
                    f"{pre_losses[step_index]:.10g}",
                    f"{post_matrix[step_index][step_index]:.10g}",
                    f"{self_delta:.10g}",
                    len(other_deltas),
                    sum(1 for delta in other_deltas if delta > 0),
                    sum(1 for delta in other_deltas if delta < 0),
                    sum(1 for delta in other_deltas if delta == 0),
                    f"{statistics.fmean(other_deltas):.10g}",
                    f"{statistics.median(other_deltas):.10g}",
                    f"{sum(other_deltas):.10g}",
                    f"{worst_delta:.10g}",
                    worst_index,
                    f"{best_delta:.10g}",
                    best_index,
                    f"{sum(deltas):.10g}",
                ]
            )

    (output_dir / "batch_interference_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {portable_path(long_path)}\n"
        f"wrote {portable_path(matrix_path)}\n"
        f"wrote {portable_path(summary_path)}",
        flush=True,
    )


def run_probe(args: argparse.Namespace) -> None:
    """Execute the full probe for one arm and write its outputs.

    Called by: main. Calls: build_loop_and_batches, cache_batches_on_device,
    evaluate_all, take_one_step, archive_existing, write_outputs.
    """

    config = load_config(args.config)
    resolver = StorageResolver()
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else _local_checkpoint_dir(config, resolver) / "last.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    started = time.perf_counter()
    loop, batches, epoch_index = build_loop_and_batches(
        config,
        checkpoint_path=checkpoint_path,
        epoch_index=args.epoch_index,
        max_batches=args.max_batches,
    )
    count = len(batches)
    if count < 2:
        raise ValueError(f"need at least 2 batches to measure interference, got {count}")
    batches, _cached = cache_batches_on_device(batches, loop.device, args.batch_cache_gb)

    # One fixed corruption seed per batch. Every measurement of batch j -- and
    # the step taken ON batch j -- reuses this seed, so the corrupted view is
    # frozen for the entire probe.
    seeds = [args.probe_seed + SEED_STRIDE * index for index in range(count)]

    # The optimizer's LR comes from the restored scheduler state, i.e. the rate
    # the run was training at when the checkpoint was written.
    resume_lr = float(loop.optimizer.param_groups[0]["lr"])
    probe_lr = resume_lr * args.lr_scale
    if args.lr_scale != 1.0:
        for group in loop.optimizer.param_groups:
            group["lr"] = group["lr"] * args.lr_scale

    # Measurement precision is decoupled from step precision: swap the loop's
    # config around every measurement, leave it alone for the step itself.
    train_config = loop.config
    eval_config = (
        replace(train_config, train=replace(train_config.train, precision="fp32"))
        if args.eval_precision == "fp32"
        else train_config
    )

    print(
        f"probe arm={args.config.stem} checkpoint={portable_path(checkpoint_path)} "
        f"global_step={loop.global_step} completed_epochs={loop.completed_epochs} "
        f"batches={count} epoch_index={epoch_index} device={loop.device} "
        f"resume_lr={resume_lr:.3e} probe_lr={probe_lr:.3e} "
        f"step_precision={train_config.train.precision} "
        f"eval_precision={eval_config.train.precision}",
        flush=True,
    )

    # ---- Baseline -------------------------------------------------------
    loop.config = eval_config
    pre_losses = evaluate_all(loop, batches, seeds)
    # Determinism guard: re-measuring one batch with untouched weights must
    # reproduce the same number to the bit. If it does not, some source of
    # randomness is still live and every delta below would be noise.
    recheck = evaluate_all(loop, batches[:1], seeds[:1])[0]
    if recheck != pre_losses[0]:
        raise RuntimeError(
            "measurement is not deterministic: batch 0 evaluated twice on identical "
            f"weights gave {pre_losses[0]!r} then {recheck!r}"
        )
    loop.config = train_config
    print(
        "baseline_loss mean={:.6f} min={:.6f} max={:.6f}".format(
            statistics.fmean(pre_losses), min(pre_losses), max(pre_losses)
        ),
        flush=True,
    )

    # ---- Snapshot to restore before every step --------------------------
    # Deep copies are required: `optimizer.step()` mutates both the parameters
    # and the Adam moment buffers in place, so a shallow state_dict would be
    # corrupted by the first step it is supposed to protect against.
    model_snapshot = copy.deepcopy(loop.model.state_dict())
    optimizer_snapshot = copy.deepcopy(loop.optimizer.state_dict())

    post_matrix: list[list[float]] = []
    step_train_losses: list[float] = []
    for step_index in range(count):
        loop.model.load_state_dict(model_snapshot)
        loop.optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
        step_loss = take_one_step(loop, batches[step_index], seeds[step_index])
        step_train_losses.append(step_loss)

        loop.config = eval_config
        post_losses = evaluate_all(loop, batches, seeds)
        loop.config = train_config
        post_matrix.append(post_losses)

        deltas = [pre - post for pre, post in zip(pre_losses, post_losses)]
        others = [delta for index, delta in enumerate(deltas) if index != step_index]
        print(
            f"step_batch={step_index}/{count - 1} "
            f"self_delta={deltas[step_index]:+.6e} "
            f"mean_other_delta={statistics.fmean(others):+.6e} "
            f"others_hurt={sum(1 for delta in others if delta < 0)}/{len(others)} "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )

    # ---- Restore verification -------------------------------------------
    # Restore one final time and re-measure the baseline. Any drift here means
    # the snapshot/restore cycle was not exact and the deltas above inherited
    # accumulated state rather than each starting from the checkpoint.
    loop.model.load_state_dict(model_snapshot)
    loop.optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
    loop.config = eval_config
    restored_losses = evaluate_all(loop, batches, seeds)
    loop.config = train_config
    restore_drift = max(
        abs(restored - original) for restored, original in zip(restored_losses, pre_losses)
    )
    print(f"restore_check max_abs_drift={restore_drift:.3e}", flush=True)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "output" / "batch_interference" / args.config.stem
    )
    archive_existing(output_dir)
    write_outputs(
        output_dir,
        arm=args.config.stem,
        pre_losses=pre_losses,
        post_matrix=post_matrix,
        step_train_losses=step_train_losses,
        meta={
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config": portable_path(args.config),
            "arm": args.config.stem,
            "checkpoint": portable_path(checkpoint_path),
            "checkpoint_global_step": loop.global_step,
            "checkpoint_completed_epochs": loop.completed_epochs,
            "epoch_index_frozen": epoch_index,
            "batch_count": count,
            "batch_size": config.pipeline.batch_size,
            "device": str(loop.device),
            "resume_lr": resume_lr,
            "probe_lr": probe_lr,
            "lr_scale": args.lr_scale,
            "grad_clip": config.train.grad_clip,
            "step_precision": train_config.train.precision,
            "eval_precision": eval_config.train.precision,
            "probe_seed": args.probe_seed,
            "seed_stride": SEED_STRIDE,
            "corruption_seeds": seeds,
            "restore_max_abs_drift": restore_drift,
            "baseline_losses": pre_losses,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    print(f"done wall_seconds={time.perf_counter() - started:.0f}", flush=True)


def main() -> None:
    """Command-line entry point. Calls: parse_args, run_probe."""

    run_probe(parse_args())


if __name__ == "__main__":
    main()
