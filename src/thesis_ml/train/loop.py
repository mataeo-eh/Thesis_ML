"""Plain PyTorch training loop for SC2 clean-state discrete diffusion."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.embedding import InputFeatures
from thesis_ml.model.loss import (
    CANVAS_STATE_NAMES,
    FUTURE_DISTANCE_BUCKETS,
    PERSPECTIVE_NAMES,
    RARE_CLASS_T_BUCKET_NAMES,
    T_BUCKET_NAMES,
    CanvasCrossEntropyLoss,
    LossOutput,
    active_class_id_to_name,
)
from thesis_ml.model.model import (
    canvas_self_conditioning_from_logits,
    validate_checkpoint_compatibility,
)
from thesis_ml.train.corruption import CorruptionOutput, corrupt_batch


@dataclass(frozen=True)
class BatchLoss:
    loss: torch.Tensor
    denoising_loss: torch.Tensor
    confidence_loss: torch.Tensor
    loss_output: LossOutput
    corruption: CorruptionOutput
    scored_mask: torch.Tensor
    canvas_logits: torch.Tensor
    self_conditioning_mask: torch.Tensor


@dataclass(frozen=True)
class ValidationLog:
    loss: float
    per_class: dict[str, float]
    future_distance: dict[str, float]
    t_bucket: dict[str, float]
    perspective: dict[str, float]
    canvas_state: dict[str, float]
    # Rare-class x t-bucket cross decomposition. `rare_class_t_bucket` holds the
    # count-weighted mean loss for each populated cell and omits cells that
    # scored nothing; `rare_class_t_bucket_counts` holds ALL 12 cells including
    # the zeros, because "no [END] token landed in this corruption bucket" is
    # itself a reportable observation. See loss.RARE_CLASS_T_BUCKET_NAMES.
    rare_class_t_bucket: dict[str, float]
    rare_class_t_bucket_counts: dict[str, int]


@dataclass(frozen=True)
class TrainStepLog:
    step: int
    loss: float
    denoising_loss: float
    confidence_loss: float
    per_class: dict[str, float]
    future_distance: dict[str, float]
    # Clean-state CE broken down by the example's sampled t-bucket, by player
    # perspective, and by per-position canvas state (ground-truth-preserved vs
    # actually-noised). Emitted in BOTH pipelines. Empty keys are simply absent
    # from these dicts (per_class convention).
    t_bucket_loss: dict[str, float]
    perspective_loss: dict[str, float]
    canvas_state_loss: dict[str, float]
    # Rare-class x t-bucket cross decomposition for this step's last microbatch:
    # mean loss per populated cell, plus the scored-position count of every cell.
    rare_class_t_bucket_loss: dict[str, float]
    rare_class_t_bucket_count: dict[str, int]
    lr: float
    t_mean: float
    noise_fraction: float
    step_wall_seconds: float
    tokens_per_second: float
    cuda_max_memory_allocated_bytes: int
    cuda_memory_allocated_bytes: int
    cuda_memory_reserved_bytes: int
    cuda_inactive_split_bytes: int
    cuda_device_memory_used_bytes: int
    cuda_device_memory_gap_bytes: int
    validation: ValidationLog | None = None


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    dev_loss: float | None
    train_per_class: dict[str, float]
    dev_per_class: dict[str, float]
    average_input_timesteps: float
    average_enemy_future_timesteps: float
    input_timestep_percentiles: dict[str, float]
    enemy_future_timestep_percentiles: dict[str, float]
    train_future_distance: dict[str, float]
    dev_future_distance: dict[str, float]
    train_t_bucket_loss: dict[str, float]
    dev_t_bucket_loss: dict[str, float]
    train_perspective_loss: dict[str, float]
    dev_perspective_loss: dict[str, float]
    train_canvas_state_loss: dict[str, float]
    dev_canvas_state_loss: dict[str, float]
    # Rare-class x t-bucket cross decomposition, pooled over the whole epoch by
    # total scored positions (not by averaging per-microbatch means -- see
    # LossOutput.rare_class_t_bucket_sums for why). The `_counts` dicts carry all
    # 12 cells including zeros; the loss dicts omit cells that scored nothing.
    train_rare_class_t_bucket_loss: dict[str, float]
    dev_rare_class_t_bucket_loss: dict[str, float]
    train_rare_class_t_bucket_counts: dict[str, int]
    dev_rare_class_t_bucket_counts: dict[str, int]
    total_tokens_ingested: int
    total_unique_tokens_seen: int
    tokens_per_second: float
    wall_clock_elapsed_seconds: float
    average_cuda_device_memory_used_bytes: float
    average_cuda_device_memory_gap_bytes: float


# How many diagnostic reports are emitted per epoch, evenly spaced across the
# epoch's batches. WHY this exists in addition to the per-epoch CSV: on a corpus
# large enough that pre-training only needs ONE epoch, a per-epoch breakdown
# yields exactly one data point and shows no trend at all. Reporting at every
# ~10% of an epoch gives ten ordered observations of each loss sub-class within
# a single epoch, so the same diagnostics that are readable on a 30-epoch
# overfit run stay readable on a 1-epoch full run.
INTERVAL_REPORTS_PER_EPOCH = 10


@dataclass(frozen=True)
class IntervalMetrics:
    """One intra-epoch diagnostic report (see INTERVAL_REPORTS_PER_EPOCH).

    Every loss field is scoped to the SLICE of the epoch since the previous
    interval report -- not to the epoch so far -- so consecutive rows show the
    losses actually moving rather than a slowly-updating running average.

    The ``dev_*`` fields come from a full pass over the dev loader taken at this
    interval boundary with EMA weights, and are empty dicts (``dev_loss`` None)
    when the run has no dev loader or ``train.interval_dev_evaluation`` is false.

    The ``train_*`` fields are likewise empty (``train_loss`` None) when
    ``train.interval_train_evaluation`` is false, in which case train loss is
    reported once per epoch in the epoch CSV instead. Both sides being disabled
    suppresses the row entirely rather than emitting an all-blank one; see
    TrainingLoop.fit.
    """

    epoch: int
    # 1..INTERVAL_REPORTS_PER_EPOCH within the epoch.
    interval: int
    # Fraction of the epoch completed at this boundary (interval / reports).
    epoch_fraction: float
    global_step: int
    # Batch index within the epoch that triggered this report.
    epoch_batch_index: int
    batches_in_epoch: int
    train_loss: float | None
    dev_loss: float | None
    train_per_class: dict[str, float]
    dev_per_class: dict[str, float]
    train_t_bucket_loss: dict[str, float]
    dev_t_bucket_loss: dict[str, float]
    train_canvas_state_loss: dict[str, float]
    dev_canvas_state_loss: dict[str, float]
    train_perspective_loss: dict[str, float]
    dev_perspective_loss: dict[str, float]
    # Rare-class x t-bucket cross decomposition for this slice, pooled by scored
    # positions exactly as the epoch row pools it over the whole epoch.
    train_rare_class_t_bucket_loss: dict[str, float]
    dev_rare_class_t_bucket_loss: dict[str, float]
    train_rare_class_t_bucket_counts: dict[str, int]
    dev_rare_class_t_bucket_counts: dict[str, int]
    lr: float
    wall_clock_elapsed_seconds: float


def optimizer_steps_per_epoch(batches_per_epoch: int, accumulation_steps: int) -> int:
    """Return optimizer updates produced by one full dataloader epoch."""

    if batches_per_epoch < 0:
        raise ValueError("batches_per_epoch must be >= 0")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    return math.ceil(batches_per_epoch / accumulation_steps)


def interval_boundaries(batches_per_epoch: int, reports: int = INTERVAL_REPORTS_PER_EPOCH) -> list[int]:
    """Batch indices at which an intra-epoch report fires.

    Returns the 1-indexed batch numbers closest to each ``k / reports`` share of
    the epoch, for k = 1..reports. The final boundary is always
    ``batches_per_epoch``, so the last interval report coincides with the epoch
    end and the per-epoch CSV row.

    Duplicates are removed, which matters only when an epoch has fewer batches
    than ``reports``: a 4-batch epoch then reports 4 times, not 10, because
    there is no finer granularity available to report at.

    Args:
        batches_per_epoch: number of batches the epoch will yield.
        reports: how many evenly spaced reports to request.

    Returns:
        Sorted, strictly increasing batch indices in ``[1, batches_per_epoch]``.
        Empty when the epoch has no batches.

    Called by: TrainingLoop.fit (to schedule reports) and the tests.
    """

    if batches_per_epoch <= 0 or reports <= 0:
        return []
    boundaries = sorted(
        {math.ceil(batches_per_epoch * index / reports) for index in range(1, reports + 1)}
    )
    return [boundary for boundary in boundaries if boundary > 0]


class TrainingLoop:
    """Owns output-side corruption, optimization, logging, and checkpointing."""

    def __init__(
        self,
        *,
        model: nn.Module,
        config: ProjectConfig,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: CanvasCrossEntropyLoss | None = None,
        seed: int | None = None,
        metrics_path: str | Path | None = None,
        epoch_metrics_path: str | Path | None = None,
        interval_metrics_path: str | Path | None = None,
        checkpoint_publisher: Callable[[Path], None] | None = None,
        metrics_publisher: Callable[[Path], None] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        # Optional persistence/observability hooks (used by the cloud pipeline,
        # absent in unit tests). metrics_path: local JSONL file that receives
        # one line per logged step so a multi-day run can be monitored and
        # killed early if loss curves go wrong. *_publisher callbacks copy a
        # local artifact to durable remote storage (e.g. S3) so a preempted
        # spot instance loses only minutes of progress.
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.epoch_metrics_path = (
            _latest_metrics_csv_path(Path(epoch_metrics_path))
            if epoch_metrics_path is not None
            else None
        )
        if self.epoch_metrics_path is not None:
            self.epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # Intra-epoch diagnostic CSV: INTERVAL_REPORTS_PER_EPOCH rows per epoch
        # instead of one, so loss sub-class trends stay legible even on a run
        # that only trains for a single epoch. Same append-and-migrate handling
        # as the epoch CSV.
        self.interval_metrics_path = (
            _latest_metrics_csv_path(Path(interval_metrics_path))
            if interval_metrics_path is not None
            else None
        )
        if self.interval_metrics_path is not None:
            self.interval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_publisher = checkpoint_publisher
        self.metrics_publisher = metrics_publisher
        self.loss_fn = loss_fn or CanvasCrossEntropyLoss(config)
        self.loss_fn.to(self.device)
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            betas=(config.train.beta1, config.train.beta2),
            weight_decay=config.train.weight_decay,
            eps=config.train.adam_eps,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_multiplier)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.global_step = 0
        self.completed_epochs = 0
        # How many DataLoader batches of the CURRENT (in-progress) epoch have
        # been consumed. Persisted in checkpoints so a run killed mid-epoch
        # resumes at the exact batch it left off instead of replaying the epoch
        # from batch 1. Reset to 0 at each epoch boundary. See fit() for how it
        # drives the resumable batch sampler's skip-ahead.
        self.batches_completed_in_epoch = 0
        # `best_dev_loss` owns best-so-far checkpoint replacement. Early
        # stopping uses a separate thresholded best so any strict improvement
        # can be retained without letting negligible noise reset patience.
        self.best_dev_loss = math.inf
        self.early_stopping_best_dev_loss = math.inf
        self.epochs_without_improvement = 0
        # Wall-clock seconds accumulated by *previously finished* fit()
        # calls (restored from the checkpoint on resume). This is a
        # baseline, NOT the live total: time spent inside the currently
        # running fit() is not folded in here until that fit() returns.
        # Read `_cumulative_wall_seconds()` whenever you need the true
        # running total.
        self.elapsed_wall_seconds = 0.0
        # `time.perf_counter()` reading taken when the in-flight fit()
        # started, or None when no fit() is running. Paired with
        # `elapsed_wall_seconds` by `_cumulative_wall_seconds()`.
        self._fit_started_at: float | None = None
        self.total_tokens_ingested = 0
        self.unique_token_ids_seen: set[int] = set()
        # Lazily-built cache of (float ema tensors, float raw tensors, non-float
        # pairs) used by _update_ema so the per-step EMA update fuses into a
        # couple of _foreach_ kernels instead of re-walking state_dict and
        # launching two tiny kernels per parameter every step. Populated on the
        # first _update_ema call; see that method for why the references stay
        # valid across optimizer steps and checkpoint resumes.
        self._ema_tensor_cache: tuple[
            list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]
        ] | None = None
        # Target EMA decay for THIS run, sized to the run's own step horizon
        # rather than left at a fixed constant -- see _resolve_ema_decay. Seeded
        # here from the configured horizon so a loop driven directly (tests,
        # smoke paths) has a valid decay before fit() runs; fit() recomputes it
        # from the step budget it actually derives, which is the authoritative
        # value for a real run.
        self._ema_target_decay = self._resolve_ema_decay(self.config.train.max_steps)
        generator_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
        # Store the base seed so fit() can RESEED the generator at every epoch
        # boundary as manual_seed(base_seed + epoch_index). Because the generator
        # then becomes a deterministic function of (base_seed, epoch_index) at
        # each epoch start -- not of how many draws happened since construction --
        # a run resumed mid-training (fit() picks up at self.completed_epochs)
        # reproduces exactly the same corruption / self-conditioning draw stream
        # an uninterrupted run would have had. This FIXES the previous resume
        # misalignment where the generator was seeded once at construction and
        # never checkpointed, so every restart replayed the draws from seed
        # rather than continuing the intended per-epoch stream.
        self._base_seed = seed
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def fit(
        self,
        dataloader: Iterable[DiffusionBatch],
        *,
        max_steps: int | None = None,
        val_dataloader: Iterable[DiffusionBatch] | None = None,
        fixed_t: float | None = None,
        epochs: int | None = None,
        retain_logs: bool = True,
    ) -> list[TrainStepLog]:
        """Run optimizer steps and return per-step logs."""

        configured_steps = self.config.train.max_steps if max_steps is None else max_steps
        epoch_count = self.config.train.epochs if epochs is None else epochs
        try:
            batches_per_epoch = len(dataloader)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("training requires a sized dataloader for progress reporting") from exc
        optimizer_steps_in_epoch = optimizer_steps_per_epoch(
            batches_per_epoch,
            self.config.train.accumulation_steps,
        )
        if configured_steps > 0:
            target_steps = configured_steps
            epoch_limit = max(
                epoch_count,
                math.ceil(target_steps / max(1, optimizer_steps_in_epoch)),
            )
        else:
            if epoch_count < 1:
                raise ValueError("train.epochs must be >= 1 when train.max_steps is 0")
            target_steps = optimizer_steps_in_epoch * epoch_count
            epoch_limit = epoch_count
        if self.config.train.early_stopping_patience_epochs > 0 and val_dataloader is None:
            raise ValueError("dev-loss early stopping requires a validation dataloader")
        # Re-fit the EMA averaging window to the step budget THIS run will
        # actually train through, now that it is known. target_steps is used
        # rather than config.train.max_steps so that a run bounded by an explicit
        # `--max-steps` cap still gets an EMA that completes inside the steps it
        # is given. See _resolve_ema_decay.
        self._ema_target_decay = self._resolve_ema_decay(target_steps)
        base_accumulation_steps = self.config.train.accumulation_steps
        if base_accumulation_steps < 1:
            raise ValueError("train.accumulation_steps must be >= 1")

        logs: list[TrainStepLog] = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        fit_started = time.perf_counter()
        # Fold in any time from an earlier fit() that raised before it
        # could close its own window, then mark this fit() as in flight so
        # checkpoints written mid-run record the true cumulative elapsed
        # time rather than the value as of this process's startup.
        self.elapsed_wall_seconds = self._cumulative_wall_seconds()
        self._fit_started_at = fit_started

        # Surface the effective LR schedule up front so a run never silently
        # trains at a near-zero rate. base_lr is the peak; the linear warmup
        # means the EFFECTIVE lr only reaches base_lr after `warmup` steps, so a
        # warmup that is large relative to target_steps keeps the whole run at a
        # tiny fraction of base_lr. warmup_end_lr shows the lr at the first step
        # AFTER warmup; if it is far below base_lr, warmup is eating the run.
        base_lr = self.config.train.lr
        warmup, stable, decay = self._schedule_phase_steps(target_steps)
        print(
            f"lr_schedule name={self.config.train.lr_schedule} base_lr={base_lr:.3e} "
            f"warmup_steps={warmup} stable_steps={stable} decay_steps={decay} "
            f"target_steps={target_steps} "
            f"effective_lr_at_step_1={base_lr * self._lr_multiplier(0):.3e} "
            f"effective_lr_after_warmup={base_lr * self._lr_multiplier(warmup):.3e} "
            f"effective_lr_at_final_step={base_lr * self._lr_multiplier(target_steps):.3e}"
            + (
                "  [WARNING: warmup >= target_steps -> the run never leaves warmup; "
                "lower train.warmup]"
                if self.config.train.lr_schedule != "wsd" and warmup >= target_steps
                else ""
            ),
            flush=True,
        )
        # Same idea for the EMA: print the window the derived decay corresponds
        # to so it is visible up front that the EMA finishes inside this run
        # rather than trailing a window sized for some other run's length.
        ema_window_steps = (
            math.inf
            if self._ema_target_decay >= 1.0
            else 1.0 / (1.0 - self._ema_target_decay)
        )
        print(
            f"ema_schedule horizon_ratio={self.config.train.ema_horizon_ratio} "
            f"decay_ceiling={self.config.train.ema_decay} "
            f"effective_decay={self._ema_target_decay:.6f} "
            f"averaging_window_steps={ema_window_steps:.0f} "
            f"target_steps={target_steps} "
            f"windows_per_run={target_steps / ema_window_steps:.1f}",
            flush=True,
        )

        for epoch_index in range(self.completed_epochs, epoch_limit):
            if self.global_step >= target_steps:
                break
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch_index)
            # Reseed the corruption / self-conditioning generator for THIS epoch,
            # mirroring the dataset's and batch sampler's per-epoch reseeds (and
            # ResumableBatchSampler's established `base_seed + epoch` idiom). This
            # is what makes the corruption draw stream a deterministic function of
            # (base_seed, epoch_index), so a resumed run reproduces the same
            # stream as an uninterrupted one (see __init__). Only reseed when a
            # seed was configured; an unseeded run stays nondeterministic.
            if self._base_seed is not None:
                self.generator.manual_seed(self._base_seed + epoch_index)

            # ---- Deterministic ordering + mid-epoch resume ------------------
            # Seed the batch sampler for THIS epoch so its shuffle is
            # reproducible across process restarts; without this a resume would
            # draw a different order and skipping batches would be meaningless.
            # `completed_epochs` equals the in-progress epoch index during that
            # epoch, so on a mid-epoch resume `epoch_index` here matches the
            # epoch the checkpoint was taken in and the ordering lines up.
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch_index)
            # `batches_completed_in_epoch` is non-zero only on the first epoch
            # after a mid-epoch resume (it is reset to 0 at every epoch
            # boundary). Skip that many already-trained batches so the epoch
            # PROGRESSES instead of restarting.
            resume_skip = self.batches_completed_in_epoch
            if resume_skip >= batches_per_epoch and batches_per_epoch > 0:
                # The checkpoint landed on the final batch of this epoch (killed
                # between the last step and the epoch-end bookkeeping). Treat the
                # epoch as finished rather than replaying it or yielding zero
                # batches (which would prematurely end the whole run).
                self.completed_epochs = epoch_index + 1
                self.batches_completed_in_epoch = 0
                continue
            if resume_skip > 0 and batch_sampler is not None and hasattr(
                batch_sampler, "set_start_batch"
            ):
                batch_sampler.set_start_batch(resume_skip)

            epoch_started = time.perf_counter()
            epoch_losses: list[float] = []
            epoch_class_losses: dict[str, list[float]] = {}
            epoch_tokens = 0
            epoch_examples = 0
            epoch_input_timesteps = 0
            epoch_enemy_future_timesteps = 0
            epoch_input_timestep_counts: list[int] = []
            epoch_enemy_future_timestep_counts: list[int] = []
            epoch_future_distance_sums: dict[str, float] = {}
            epoch_future_distance_counts: dict[str, int] = {}
            # t-bucket / perspective loss accumulation mirrors per_class exactly
            # (a list of per-microbatch means, later simple-averaged).
            epoch_t_bucket_losses: dict[str, list[float]] = {}
            epoch_perspective_losses: dict[str, list[float]] = {}
            epoch_canvas_state_losses: dict[str, list[float]] = {}
            # The rare-class x t-bucket cells accumulate as SUM + COUNT rather
            # than as a list of per-microbatch means. A cell can hold 6 positions
            # in one microbatch and 0 in the next, so a simple average of
            # per-microbatch means would weight those equally and misreport the
            # epoch; pooling by total scored positions is the correct reduction.
            epoch_rare_class_sums: dict[str, float] = {}
            epoch_rare_class_counts: dict[str, int] = {}
            # ---- Intra-epoch diagnostic reporting ---------------------------
            # `interval_*` accumulators mirror the epoch ones but are CLEARED
            # after every report, so each row covers only the slice of the epoch
            # since the previous boundary. `interval_cursor` walks `boundaries`;
            # on a mid-epoch resume it is advanced past the boundaries this epoch
            # already reported so a resumed epoch does not emit duplicate rows
            # (with empty accumulators) for slices it already trained through.
            boundaries = interval_boundaries(batches_per_epoch)
            interval_cursor = sum(
                1 for boundary in boundaries if boundary <= self.batches_completed_in_epoch
            )
            interval_losses: list[float] = []
            interval_class_losses: dict[str, list[float]] = {}
            interval_t_bucket_losses: dict[str, list[float]] = {}
            interval_perspective_losses: dict[str, list[float]] = {}
            interval_canvas_state_losses: dict[str, list[float]] = {}
            interval_rare_class_sums: dict[str, float] = {}
            interval_rare_class_counts: dict[str, int] = {}
            epoch_cuda_device_memory_used = 0
            epoch_cuda_device_memory_gap = 0
            epoch_cuda_memory_samples = 0
            # Continue the batch counter from where a resume left off so the
            # progress display ("batch=K/N") and the persisted intra-epoch
            # position both count actual batches consumed this epoch.
            epoch_batch_index = self.batches_completed_in_epoch
            data_iter = iter(dataloader)

            # ---- Asynchronous, one-step-lagged execution --------------------
            # To keep the GPU saturated, each iteration first LAUNCHES this
            # step's GPU work (forward/backward/optimizer/EMA -- all queued
            # asynchronously) and only THEN finalizes the PREVIOUS step:
            # host-side logging, the GPU->CPU metric transfers, and epoch
            # aggregation. Because the previous step's kernels were queued a
            # full iteration ago, the GPU stays busy running the CURRENT step
            # while the CPU does that serial, sync-heavy bookkeeping -- removing
            # the per-step "compute, then sit idle while we log" bubble that was
            # starving the GPU (the sawtooth). Step timing uses CUDA events read
            # one step late, so reading the timer never blocks the launch thread
            # (a fresh cuda.synchronize() every step is exactly what we removed).
            pending: dict | None = None

            def _finalize(record: dict, step_wall_seconds: float) -> None:
                """Log and epoch-aggregate one already-launched step.

                Runs a full step after `record` was queued, so the GPU is busy
                with the next step throughout. Blocks only on that step's own
                CUDA end-event (long since complete) to read its GPU time, then
                pulls the small scalar metrics to the host. Mutates the enclosing
                epoch accumulators; the arithmetic is identical to the previous
                inline per-batch version, just performed one step later.
                """

                nonlocal epoch_cuda_device_memory_used, epoch_cuda_device_memory_gap
                nonlocal epoch_cuda_memory_samples, interval_cursor
                if self.device.type == "cuda":
                    record["end_evt"].synchronize()
                    compute_seconds = (
                        record["start_evt"].elapsed_time(record["end_evt"]) / 1000.0
                    )
                else:
                    compute_seconds = record["cpu_compute_seconds"]
                # Read allocator/device memory AFTER the sync so the figures
                # reflect the fully-executed step.
                cuda_max_allocated = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                )
                cuda_reserved = (
                    int(torch.cuda.memory_reserved(self.device))
                    if self.device.type == "cuda"
                    else 0
                )
                cuda_allocated = (
                    int(torch.cuda.memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                )
                cuda_inactive_split = (
                    int(
                        torch.cuda.memory_stats(self.device).get(
                            "inactive_split_bytes.all.current", 0
                        )
                    )
                    if self.device.type == "cuda"
                    else 0
                )
                cuda_device_used = 0
                if self.device.type == "cuda":
                    cuda_free, cuda_total = torch.cuda.mem_get_info(self.device)
                    cuda_device_used = int(cuda_total - cuda_free)
                cuda_device_gap = max(0, cuda_device_used - cuda_reserved)
                epoch_cuda_device_memory_used += cuda_device_used
                epoch_cuda_device_memory_gap += cuda_device_gap
                epoch_cuda_memory_samples += 1

                # Epoch loss/per-class/future-distance aggregation over every
                # microbatch of the step (same values the inline loop appended).
                for mb in record["microbatches"]:
                    loss_value = float(mb["loss"].cpu())
                    epoch_losses.append(loss_value)
                    interval_losses.append(loss_value)
                    for name, value in mb["per_class"].items():
                        class_value = float(value.cpu())
                        epoch_class_losses.setdefault(name, []).append(class_value)
                        interval_class_losses.setdefault(name, []).append(class_value)
                    for name, value in mb["future_distance"].items():
                        count = mb["future_distance_counts"][name]
                        epoch_future_distance_sums[name] = (
                            epoch_future_distance_sums.get(name, 0.0)
                            + float(value.cpu()) * count
                        )
                        epoch_future_distance_counts[name] = (
                            epoch_future_distance_counts.get(name, 0) + count
                        )
                    for name, value in mb["t_bucket"].items():
                        bucket_value = float(value.cpu())
                        epoch_t_bucket_losses.setdefault(name, []).append(bucket_value)
                        interval_t_bucket_losses.setdefault(name, []).append(bucket_value)
                    for name, value in mb["perspective"].items():
                        perspective_value = float(value.cpu())
                        epoch_perspective_losses.setdefault(name, []).append(perspective_value)
                        interval_perspective_losses.setdefault(name, []).append(perspective_value)
                    for name, value in mb["canvas_state"].items():
                        state_value = float(value.cpu())
                        epoch_canvas_state_losses.setdefault(name, []).append(state_value)
                        interval_canvas_state_losses.setdefault(name, []).append(state_value)
                    # Rare-class cells: accumulate the on-device sum/count pair
                    # into running host totals. All 12 keys are always present,
                    # so a cell that scored nothing accumulates 0.0 over 0 and
                    # is reported as a real zero count rather than going missing.
                    for name, value in mb["rare_class_sums"].items():
                        cell_sum = float(value.cpu())
                        epoch_rare_class_sums[name] = (
                            epoch_rare_class_sums.get(name, 0.0) + cell_sum
                        )
                        interval_rare_class_sums[name] = (
                            interval_rare_class_sums.get(name, 0.0) + cell_sum
                        )
                    for name, value in mb["rare_class_counts"].items():
                        count = int(value.cpu())
                        epoch_rare_class_counts[name] = (
                            epoch_rare_class_counts.get(name, 0) + count
                        )
                        interval_rare_class_counts[name] = (
                            interval_rare_class_counts.get(name, 0) + count
                        )

                tokens_per_second = record["step_tokens"] / step_wall_seconds
                print(
                    f"step={record['step']} step_wall_seconds={step_wall_seconds:.3f} "
                    f"data_wait_seconds={record['data_wait_seconds']:.3f} "
                    f"compute_seconds={compute_seconds:.3f} "
                    f"tokens_per_second={tokens_per_second:.1f} "
                    f"lr={record['lr']:.3e} "
                    f"cuda_max_memory_allocated_gb={cuda_max_allocated / 1024**3:.3f} "
                    f"cuda_memory_allocated_gb={cuda_allocated / 1024**3:.3f} "
                    f"cuda_memory_reserved_gb={cuda_reserved / 1024**3:.3f} "
                    f"cuda_inactive_split_gb={cuda_inactive_split / 1024**3:.3f} "
                    f"cuda_device_memory_used_gb={cuda_device_used / 1024**3:.3f} "
                    f"cuda_device_memory_gap_gb={cuda_device_gap / 1024**3:.3f}",
                    flush=True,
                )
                self._enforce_cuda_memory_limit(cuda_reserved)

                validation = self._maybe_validate(
                    val_dataloader, step=record["step"], fixed_t=fixed_t
                )
                last = record["microbatches"][-1]
                step_log = self._make_log(
                    step=record["step"],
                    loss=float(last["loss"].cpu()),
                    denoising_loss=float(last["denoising"].cpu()),
                    confidence_loss=float(last["confidence"].cpu()),
                    per_class={name: float(v.cpu()) for name, v in last["per_class"].items()},
                    future_distance={
                        name: float(v.cpu()) for name, v in last["future_distance"].items()
                    },
                    t_bucket_loss={
                        name: float(v.cpu()) for name, v in last["t_bucket"].items()
                    },
                    perspective_loss={
                        name: float(v.cpu()) for name, v in last["perspective"].items()
                    },
                    canvas_state_loss={
                        name: float(v.cpu()) for name, v in last["canvas_state"].items()
                    },
                    # Per-step rare-class cells are reduced to a mean here so the
                    # JSONL matches the other decompositions' shape; the count
                    # dict rides alongside because a mean over 1 position and a
                    # mean over 40 are not the same observation.
                    rare_class_t_bucket_loss=_finalize_rare_class_t_bucket(
                        {name: float(v.cpu()) for name, v in last["rare_class_sums"].items()},
                        {name: int(v.cpu()) for name, v in last["rare_class_counts"].items()},
                    ),
                    rare_class_t_bucket_count={
                        name: int(v.cpu()) for name, v in last["rare_class_counts"].items()
                    },
                    lr=record["lr"],
                    t_mean=float(last["t_mean"].cpu()),
                    noise_fraction=float(last["noise_fraction"].cpu()),
                    validation=validation,
                    step_wall_seconds=step_wall_seconds,
                    tokens_per_second=tokens_per_second,
                    cuda_max_memory_allocated_bytes=cuda_max_allocated,
                    cuda_memory_allocated_bytes=cuda_allocated,
                    cuda_memory_reserved_bytes=cuda_reserved,
                    cuda_inactive_split_bytes=cuda_inactive_split,
                    cuda_device_memory_used_bytes=cuda_device_used,
                    cuda_device_memory_gap_bytes=cuda_device_gap,
                )
                if retain_logs:
                    logs.append(step_log)
                self._write_metrics_line(step_log)

                # ---- Intra-epoch diagnostic report --------------------------
                # A single step can consume several microbatches (gradient
                # accumulation) and so cross more than one boundary at once. When
                # that happens we advance the cursor past ALL of them but emit
                # ONE row, labelled with the last boundary crossed: the slice
                # since the previous report is a single indivisible set of
                # microbatches, so splitting it into several rows would mean
                # emitting rows with no data behind them.
                crossed = 0
                while (
                    interval_cursor + crossed < len(boundaries)
                    and record["epoch_batch_index"] >= boundaries[interval_cursor + crossed]
                ):
                    crossed += 1
                if crossed:
                    interval_cursor += crossed
                    # Each SIDE of the interval row is independently config-gated.
                    #
                    # Dev (train.interval_dev_evaluation): the pass runs with EMA
                    # weights over the whole dev loader, exactly as epoch-end
                    # validation does, so interval and epoch dev numbers are
                    # directly comparable -- but on a small run it can cost more
                    # than the slice of training it follows.
                    #
                    # Train (train.interval_train_evaluation): costs nothing
                    # extra (the values are already accumulated), but on a run
                    # measured in many epochs a ~10%-of-epoch slice is only a
                    # handful of batches, so the rows are noise around what the
                    # epoch row already reports.
                    #
                    # Either side disabled leaves its columns blank and its
                    # numbers reported once per epoch in the epoch CSV instead.
                    # BOTH disabled means there is nothing left to report, so no
                    # row is written at all rather than an all-blank one -- the
                    # accumulators above still run untouched, so re-enabling
                    # either side needs no other change.
                    report_train = self.config.train.interval_train_evaluation
                    report_dev = (
                        val_dataloader is not None
                        and self.config.train.interval_dev_evaluation
                    )
                    interval_validation = (
                        self.validate(val_dataloader, fixed_t=fixed_t)
                        if report_dev
                        else None
                    )
                    if report_train or report_dev:
                        self._write_interval_metrics(
                            IntervalMetrics(
                                epoch=epoch_index + 1,
                                interval=interval_cursor,
                                epoch_fraction=interval_cursor / len(boundaries),
                                global_step=record["step"],
                                epoch_batch_index=record["epoch_batch_index"],
                                batches_in_epoch=batches_per_epoch,
                                train_loss=(
                                    sum(interval_losses) / len(interval_losses)
                                    if report_train and interval_losses
                                    else None
                                ),
                                dev_loss=(
                                    interval_validation.loss
                                    if interval_validation is not None
                                    else None
                                ),
                                train_per_class=(
                                    _mean_of_lists(interval_class_losses)
                                    if report_train
                                    else {}
                                ),
                                dev_per_class=(
                                    interval_validation.per_class
                                    if interval_validation is not None
                                    else {}
                                ),
                                train_t_bucket_loss=(
                                    _mean_of_lists(interval_t_bucket_losses)
                                    if report_train
                                    else {}
                                ),
                                dev_t_bucket_loss=(
                                    interval_validation.t_bucket
                                    if interval_validation is not None
                                    else {}
                                ),
                                train_canvas_state_loss=(
                                    _mean_of_lists(interval_canvas_state_losses)
                                    if report_train
                                    else {}
                                ),
                                dev_canvas_state_loss=(
                                    interval_validation.canvas_state
                                    if interval_validation is not None
                                    else {}
                                ),
                                train_perspective_loss=(
                                    _mean_of_lists(interval_perspective_losses)
                                    if report_train
                                    else {}
                                ),
                                dev_perspective_loss=(
                                    interval_validation.perspective
                                    if interval_validation is not None
                                    else {}
                                ),
                                train_rare_class_t_bucket_loss=(
                                    _finalize_rare_class_t_bucket(
                                        interval_rare_class_sums,
                                        interval_rare_class_counts,
                                    )
                                    if report_train
                                    else {}
                                ),
                                dev_rare_class_t_bucket_loss=(
                                    interval_validation.rare_class_t_bucket
                                    if interval_validation is not None
                                    else {}
                                ),
                                train_rare_class_t_bucket_counts=(
                                    dict(interval_rare_class_counts) if report_train else {}
                                ),
                                dev_rare_class_t_bucket_counts=(
                                    interval_validation.rare_class_t_bucket_counts
                                    if interval_validation is not None
                                    else {}
                                ),
                                lr=record["lr"],
                                wall_clock_elapsed_seconds=self._cumulative_wall_seconds(),
                            )
                        )
                    # Scope the NEXT row to the next slice only. Done regardless
                    # of whether a row was written, so a re-enabled train side
                    # never inherits a stale backlog of earlier slices.
                    interval_losses.clear()
                    interval_class_losses.clear()
                    interval_t_bucket_losses.clear()
                    interval_perspective_losses.clear()
                    interval_canvas_state_losses.clear()
                    interval_rare_class_sums.clear()
                    interval_rare_class_counts.clear()

            while self.global_step < target_steps:
                iter_top = time.perf_counter()
                try:
                    first_batch = next(data_iter)
                except StopIteration:
                    break
                accumulation_steps = self._effective_accumulation_steps(first_batch)
                microbatches = [first_batch]
                for _ in range(1, accumulation_steps):
                    try:
                        microbatches.append(next(data_iter))
                    except StopIteration:
                        break
                # Time spent BLOCKED on the DataLoader for this step's batches.
                # With prefetching this is ~0 when the loader keeps up; a large
                # value means the input pipeline is starving the GPU (loader
                # bound). Compare against compute_seconds in the per-step print.
                data_wait_seconds = time.perf_counter() - iter_top

                # Mark the start of this step's GPU work. On CUDA we time with
                # events (no host stall); on CPU there is no async queue, so a
                # perf_counter around the compute block is exact.
                if self.device.type == "cuda":
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    cpu_compute_start = None
                else:
                    start_evt = end_evt = None
                    cpu_compute_start = time.perf_counter()

                mb_scalars: list[dict] = []
                step_tokens = 0
                for batch in microbatches:
                    epoch_batch_index += 1
                    # Mirror into instance state so the checkpoint written by
                    # _maybe_checkpoint() below (after this step) records how far
                    # into the epoch we are.
                    self.batches_completed_in_epoch = epoch_batch_index
                    print(
                        f"phase=train epoch={epoch_index + 1}/{epoch_limit} "
                        f"batch={epoch_batch_index}/{batches_per_epoch}",
                        flush=True,
                    )
                    batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t)
                    (batch_loss.loss / len(microbatches)).backward()
                    # Capture ONLY the small scalar tensors the logging/epoch
                    # aggregation needs, still on-device and NOT synced here, so
                    # the large logits/mask tensors held by batch_loss are freed
                    # immediately (as `batch_loss` is reassigned next iteration)
                    # rather than pinned alive until the lagged finalize.
                    mb_scalars.append(
                        {
                            "loss": batch_loss.loss.detach(),
                            "denoising": batch_loss.denoising_loss.detach(),
                            "confidence": batch_loss.confidence_loss.detach(),
                            "per_class": {
                                name: value.detach()
                                for name, value in batch_loss.loss_output.per_class.items()
                            },
                            "future_distance": {
                                name: value.detach()
                                for name, value in batch_loss.loss_output.future_distance.items()
                            },
                            "future_distance_counts": dict(
                                batch_loss.loss_output.future_distance_counts
                            ),
                            "t_bucket": {
                                name: value.detach()
                                for name, value in batch_loss.loss_output.t_bucket.items()
                            },
                            "perspective": {
                                name: value.detach()
                                for name, value in batch_loss.loss_output.perspective.items()
                            },
                            "canvas_state": {
                                name: value.detach()
                                for name, value in batch_loss.loss_output.canvas_state.items()
                            },
                            # Rare-class cells travel as the raw sum/count pair
                            # so the lagged finalize can pool them by scored
                            # positions across microbatches.
                            "rare_class_sums": {
                                name: value.detach()
                                for name, value in
                                batch_loss.loss_output.rare_class_t_bucket_sums.items()
                            },
                            "rare_class_counts": {
                                name: value.detach()
                                for name, value in
                                batch_loss.loss_output.rare_class_t_bucket_counts.items()
                            },
                            "t_mean": batch_loss.corruption.t.detach().mean(),
                            "noise_fraction": (
                                batch_loss.corruption.corrupted_positions
                                & batch.canvas_loss_mask.to(batch_loss.corruption.corrupted_positions.device)
                            ).sum().float() / batch.canvas_loss_mask.sum().clamp_min(1).float(),
                        }
                    )
                    # Host-tensor accumulation (no GPU dependency) stays inline
                    # so it is attributed to the correct step without a sync.
                    batch_tokens = self._record_training_batch_metrics(batch)
                    epoch_tokens += batch_tokens
                    step_tokens += batch_tokens
                    epoch_examples += int(batch.input_timestep_counts.numel())
                    epoch_input_timesteps += int(batch.input_timestep_counts.sum().item())
                    epoch_enemy_future_timesteps += int(
                        batch.enemy_future_timestep_counts.sum().item()
                    )
                    epoch_input_timestep_counts.extend(batch.input_timestep_counts.tolist())
                    epoch_enemy_future_timestep_counts.extend(
                        batch.enemy_future_timestep_counts.tolist()
                    )

                if self.config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                # Record the lr AFTER scheduler.step(), matching the previous
                # loop (which read it in _make_log at the end of the step).
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.device.type == "cuda":
                    end_evt.record()
                    cpu_compute_seconds = None
                else:
                    cpu_compute_seconds = time.perf_counter() - cpu_compute_start

                # Checkpoint INLINE using the just-incremented self.global_step,
                # so cadence and the step-N.pt filename stay exact. Its
                # serialization sync is infrequent (only on the interval).
                self._maybe_checkpoint()

                record = {
                    "step": self.global_step,
                    "microbatches": mb_scalars,
                    # Batch index within the epoch AFTER this step's microbatches
                    # were consumed. The lagged _finalize uses it to decide which
                    # intra-epoch report boundaries this step crossed.
                    "epoch_batch_index": epoch_batch_index,
                    "step_tokens": step_tokens,
                    "data_wait_seconds": data_wait_seconds,
                    "iter_top": iter_top,
                    "start_evt": start_evt,
                    "end_evt": end_evt,
                    "cpu_compute_seconds": cpu_compute_seconds,
                    "lr": current_lr,
                }
                # Finalize the PREVIOUS step now, while the GPU runs THIS one.
                # step_wall for the previous step is the wall time of one full
                # iteration (its iter_top to this one's) -- the true per-step
                # throughput period once compute and logging overlap.
                if pending is not None:
                    _finalize(pending, step_wall_seconds=max(iter_top - pending["iter_top"], 1e-9))
                pending = record

            # Flush the final launched-but-unfinalized step of the epoch before
            # computing epoch metrics (so its loss is included in the averages).
            if pending is not None:
                _finalize(
                    pending,
                    step_wall_seconds=max(time.perf_counter() - pending["iter_top"], 1e-9),
                )
                pending = None

            if not epoch_losses:
                break
            # A bounded --max-steps verification may intentionally stop in the
            # middle of a very large epoch. Preserve its partial epoch metrics,
            # but do not launch the full epoch-end validation pass.
            partial_epoch = (
                epoch_batch_index < batches_per_epoch and self.global_step >= target_steps
            )
            epoch_training_duration = max(time.perf_counter() - epoch_started, 1e-9)
            epoch_validation = (
                self.validate(val_dataloader, fixed_t=fixed_t)
                if val_dataloader is not None and not partial_epoch
                else None
            )
            self.completed_epochs = epoch_index + 1
            # Epoch finished: the next epoch starts at batch 0. Reset before any
            # checkpoint of the next epoch can capture a stale offset.
            self.batches_completed_in_epoch = 0
            epoch_metrics = EpochMetrics(
                epoch=self.completed_epochs,
                train_loss=sum(epoch_losses) / len(epoch_losses),
                dev_loss=epoch_validation.loss if epoch_validation is not None else None,
                train_per_class=_mean_of_lists(epoch_class_losses),
                dev_per_class=epoch_validation.per_class if epoch_validation is not None else {},
                average_input_timesteps=epoch_input_timesteps / epoch_examples,
                average_enemy_future_timesteps=epoch_enemy_future_timesteps / epoch_examples,
                input_timestep_percentiles=_timestep_percentiles(epoch_input_timestep_counts),
                enemy_future_timestep_percentiles=_timestep_percentiles(
                    epoch_enemy_future_timestep_counts
                ),
                train_future_distance=_finalize_future_distance(
                    epoch_future_distance_sums,
                    epoch_future_distance_counts,
                ),
                dev_future_distance=(
                    epoch_validation.future_distance if epoch_validation is not None else {}
                ),
                train_t_bucket_loss=_mean_of_lists(epoch_t_bucket_losses),
                dev_t_bucket_loss=(
                    epoch_validation.t_bucket if epoch_validation is not None else {}
                ),
                train_perspective_loss=_mean_of_lists(epoch_perspective_losses),
                dev_perspective_loss=(
                    epoch_validation.perspective if epoch_validation is not None else {}
                ),
                train_canvas_state_loss=_mean_of_lists(epoch_canvas_state_losses),
                dev_canvas_state_loss=(
                    epoch_validation.canvas_state if epoch_validation is not None else {}
                ),
                train_rare_class_t_bucket_loss=_finalize_rare_class_t_bucket(
                    epoch_rare_class_sums,
                    epoch_rare_class_counts,
                ),
                dev_rare_class_t_bucket_loss=(
                    epoch_validation.rare_class_t_bucket
                    if epoch_validation is not None
                    else {}
                ),
                train_rare_class_t_bucket_counts=dict(epoch_rare_class_counts),
                dev_rare_class_t_bucket_counts=(
                    epoch_validation.rare_class_t_bucket_counts
                    if epoch_validation is not None
                    else {}
                ),
                total_tokens_ingested=self.total_tokens_ingested,
                total_unique_tokens_seen=len(self.unique_token_ids_seen),
                tokens_per_second=epoch_tokens / epoch_training_duration,
                wall_clock_elapsed_seconds=self._cumulative_wall_seconds(),
                average_cuda_device_memory_used_bytes=(
                    epoch_cuda_device_memory_used / epoch_cuda_memory_samples
                ),
                average_cuda_device_memory_gap_bytes=(
                    epoch_cuda_device_memory_gap / epoch_cuda_memory_samples
                ),
            )
            self._write_epoch_metrics(epoch_metrics)
            should_stop = (
                epoch_metrics.dev_loss is not None
                and self._should_stop_early(epoch_metrics.dev_loss)
            )
            self._save_epoch_checkpoints(epoch_metrics.dev_loss)
            if self.device.type == "cuda" and self.config.train.empty_cuda_cache_after_epoch:
                torch.cuda.empty_cache()
            if should_stop:
                print(
                    f"early_stopping=triggered epoch={self.completed_epochs} "
                    f"best_dev_loss={self.early_stopping_best_dev_loss:.6f} "
                    f"patience={self.config.train.early_stopping_patience_epochs}",
                    flush=True,
                )
                break

        # Close this fit()'s window: fold its duration into the baseline so
        # the next fit() (or the final save below) starts from the right
        # total without double-counting the window.
        self.elapsed_wall_seconds = self._cumulative_wall_seconds()
        self._fit_started_at = None

        if target_steps > 0:
            # Final durable checkpoint + metrics flush so a clean finish leaves
            # the same resumable state a mid-run preemption would.
            self.save_checkpoint(self.resume_checkpoint_path)
            self._publish_metrics()
        return logs

    def compute_batch_loss(
        self,
        batch: DiffusionBatch,
        *,
        fixed_t: float | None = None,
        model: nn.Module | None = None,
    ) -> BatchLoss:
        batch = move_batch_to_device(batch, self.device)
        corruption = corrupt_batch(
            input_token_ids=batch.input_token_ids,
            target_canvas=batch.target_canvas,
            process=self.config.diffusion.process,
            schedule=self.config.diffusion.schedule,
            vocab_size=int(getattr(self.model, "vocab_size")),
            generator=self.generator,
            t=fixed_t,
            canvas_noise_mask=batch.canvas_loss_mask,
        )

        if self.config.diffusion.process == "uniform":
            scored_mask = batch.canvas_loss_mask
        else:
            scored_mask = corruption.corrupted_positions & batch.canvas_loss_mask
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.train.precision == "bf16" and self.device.type in {"cpu", "cuda"},
        ):
            active_model = self.model if model is None else model
            input_len = batch.input_token_ids.shape[1]
            self_conditioning_mask = torch.zeros(
                batch.target_canvas.shape[0], device=self.device, dtype=torch.bool
            )
            canvas_self_conditioning = None
            if model is None and self.config.model.self_conditioning:
                self_conditioning_mask = self._sample_self_conditioning_rows(
                    batch.target_canvas.shape[0]
                )
                with torch.no_grad():
                    estimate = active_model(
                        input_token_ids=corruption.input_token_ids,
                        canvas_token_ids=corruption.noised_canvas,
                        input_attention_mask=batch.input_attention_mask,
                        canvas_attention_mask=batch.canvas_attention_mask,
                        input_features=batch.input_features,
                        # Real (non-pad) input token counts, needed by the
                        # per-segment RoPE position ablation. Passed explicitly
                        # because the batch already carries it; the model would
                        # otherwise re-derive the same values from the mask.
                        input_lengths=batch.input_lengths,
                    )
                    canvas_self_conditioning = canvas_self_conditioning_from_logits(
                        estimate.logits[:, input_len:, :],
                        active_model.embedding.token_embedding.weight,
                    )
                    canvas_self_conditioning = canvas_self_conditioning * self_conditioning_mask[
                        :, None, None
                    ].to(dtype=canvas_self_conditioning.dtype)
            forward_kwargs = {
                "input_token_ids": corruption.input_token_ids,
                "canvas_token_ids": corruption.noised_canvas,
                "input_attention_mask": batch.input_attention_mask,
                "canvas_attention_mask": batch.canvas_attention_mask,
                "input_features": batch.input_features,
                # See the self-conditioning estimate above: the per-segment RoPE
                # position ablation needs the real input token count per row.
                "input_lengths": batch.input_lengths,
            }
            if self.config.model.self_conditioning:
                forward_kwargs["canvas_self_conditioning"] = canvas_self_conditioning
            output = active_model(**forward_kwargs)
            canvas_logits = output.logits[:, input_len:, :]
            loss_output = self.loss_fn(
                canvas_logits.float(),
                batch.target_canvas,
                batch.class_labels,
                scored_mask=scored_mask,
                position_weights=corruption.position_weights,
                prediction_distances=batch.canvas_prediction_distances,
                # Per-example sampled t (from the corruption step) and player
                # perspective (a batch field) drive the t-bucket / perspective
                # loss breakdowns; both are [B] tensors aligned to the batch rows.
                sampled_t=corruption.t,
                perspective_ids=batch.perspective_ids,
                # Token-level inequality between the canvas the model was shown
                # and the target, which is what separates "already correct, keep
                # it" positions from "actually corrupted, repair it" positions.
                # See CANVAS_STATE_NAMES for why this is NOT corrupted_positions.
                changed_positions=corruption.changed_positions,
            )
            confidence_loss = auxiliary_confidence_loss(canvas_logits.float(), batch.target_canvas, scored_mask)
            weighted_confidence_loss = confidence_loss * self.config.train.confidence_loss_weight
            total_loss = loss_output.loss + weighted_confidence_loss

        return BatchLoss(
            loss=total_loss,
            denoising_loss=loss_output.loss,
            confidence_loss=weighted_confidence_loss,
            loss_output=loss_output,
            corruption=corruption,
            scored_mask=scored_mask,
            canvas_logits=canvas_logits,
            self_conditioning_mask=self_conditioning_mask.detach(),
        )

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.config.train.checkpoint_dir)

    @property
    def resume_checkpoint_path(self) -> Path:
        subdir = self.config.train.resume_checkpoint_subdir.strip()
        root = self.checkpoint_dir / subdir if subdir else self.checkpoint_dir
        return root / "last.pt"

    def _cumulative_wall_seconds(self) -> float:
        """Total wall-clock seconds this training run has spent in fit().

        Spans resumes: `elapsed_wall_seconds` carries the total accrued by
        every previous process (restored by `load_checkpoint`), and this
        adds the time the currently-running fit() has been going. The two
        must never be added together by hand -- callers use this method so
        the "is a fit in flight?" test lives in exactly one place.

        Returns:
            Seconds since the run first began training, counting all
            resumes. Equals `elapsed_wall_seconds` when no fit() is
            running (e.g. before the first one, or after the last).

        Depends on: `_fit_started_at` / `elapsed_wall_seconds`, both set by
        `__init__`, updated by `fit`, and restored by `load_checkpoint`.
        """

        if self._fit_started_at is None:
            return self.elapsed_wall_seconds
        return self.elapsed_wall_seconds + (time.perf_counter() - self._fit_started_at)

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "completed_epochs": self.completed_epochs,
                "batches_completed_in_epoch": self.batches_completed_in_epoch,
                "best_dev_loss": self.best_dev_loss,
                "early_stopping_best_dev_loss": self.early_stopping_best_dev_loss,
                "epochs_without_improvement": self.epochs_without_improvement,
                # The LIVE cumulative total, so a run killed mid-fit()
                # resumes with its wall clock intact instead of rewinding
                # to whatever it was when this process started.
                "elapsed_wall_seconds": self._cumulative_wall_seconds(),
                "total_tokens_ingested": self.total_tokens_ingested,
                "unique_token_ids_seen": sorted(self.unique_token_ids_seen),
                "config": self.config,
                "feature_statistics_identity": getattr(
                    self.model, "feature_statistics_identity", None
                ),
                "architecture_identity": getattr(self.model, "architecture_identity", None),
                "diffusion_process": getattr(self.model, "diffusion_process", None),
            },
            checkpoint_path,
        )
        # Push to durable remote storage when a publisher is configured so the
        # checkpoint survives instance loss. No-op (publisher is None) in tests
        # and in purely-local runs.
        if self.checkpoint_publisher is not None:
            self.checkpoint_publisher(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        validate_checkpoint_compatibility(checkpoint, self.model, str(Path(path)))
        self._validate_feature_statistics_identity(checkpoint, path)
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint["global_step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
        # Older checkpoints predate intra-epoch resume; absent key -> start the
        # resumed epoch at batch 0 (the previous behavior).
        self.batches_completed_in_epoch = int(
            checkpoint.get("batches_completed_in_epoch", 0)
        )
        self.best_dev_loss = float(checkpoint.get("best_dev_loss", math.inf))
        self.early_stopping_best_dev_loss = float(
            checkpoint.get("early_stopping_best_dev_loss", math.inf)
        )
        self.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        self.elapsed_wall_seconds = float(checkpoint.get("elapsed_wall_seconds", 0.0))
        # A full restore replaces every counter, so any in-flight fit()
        # window belongs to the state we just discarded. Clearing it keeps
        # the restored total from being inflated by that stale window.
        self._fit_started_at = None
        self.total_tokens_ingested = int(checkpoint.get("total_tokens_ingested", 0))
        self.unique_token_ids_seen = {
            int(token_id) for token_id in checkpoint.get("unique_token_ids_seen", [])
        }

    def load_model_weights(self, path: str | Path) -> None:
        """Warm-start ONLY the model weights from a checkpoint (fine-tuning).

        This is deliberately a *different* code path from `load_checkpoint`
        (full resume). A full resume is for continuing an interrupted run of
        the SAME training job: it restores the optimizer's momentum/variance
        buffers, the LR-schedule position (`global_step`), and every training
        counter (`completed_epochs`, best-dev state, etc.) so training
        picks up exactly where it left off.

        A "warm start" for fine-tuning is different: we want to begin a BRAND
        NEW training run (fresh optimizer, fresh LR schedule starting at
        step 0, fresh epoch counters) but initialize the model's learned
        weights from a previously pretrained checkpoint instead of random
        initialization. Copying the optimizer/scheduler/step state across
        would be wrong here because:
          - the fine-tune uses a different (much smaller) learning rate, so
            reusing the old optimizer's Adam moment estimates would apply
            stale momentum computed under a different LR regime;
          - the fine-tune's LR schedule (warmup + cosine decay) is meant to
            restart from step 0 over its own `epochs`/`max_steps` budget, not
            continue partway through the pretrain schedule;
          - epoch/step counters must start at 0 so fine-tune metrics files
            (which begin at epoch 1) are not confused with pretrain epochs.

        Only two tensors are copied out of the checkpoint dict: the plain
        model's `state_dict()` and, if present, the EMA (exponential moving
        average) model's `state_dict()`. Everything else in the checkpoint
        (optimizer, scheduler, global_step, completed_epochs, ...) is
        ignored entirely — `self.optimizer`, `self.scheduler`,
        `self.global_step`, and the other counters are left exactly as they
        were set by `__init__` (i.e. fresh).

        Args:
            path: filesystem path to a checkpoint previously written by
                `save_checkpoint` (e.g. the pretrained run's `last.pt`).
        """

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        validate_checkpoint_compatibility(checkpoint, self.model, str(Path(path)))
        self._validate_feature_statistics_identity(checkpoint, path)
        # Copy the plain model's weights.
        self.model.load_state_dict(checkpoint["model"])
        # The EMA (shadow) model tracks a smoothed copy of the weights used at
        # evaluation time. Older checkpoints may lack an "ema_model" key, in
        # which case we fall back to seeding the EMA copy with the same plain
        # model weights so both start out identical, as they would for a
        # freshly constructed loop.
        self.ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        # NOTE: optimizer, scheduler, global_step, completed_epochs,
        # best-dev state, epochs_without_improvement, elapsed_wall_seconds,
        # total_tokens_ingested, and unique_token_ids_seen are intentionally
        # left untouched here -- that is what makes this a "warm start"
        # rather than a "resume".

    def _validate_feature_statistics_identity(self, checkpoint: dict, path: str | Path) -> None:
        expected = getattr(self.model, "feature_statistics_identity", None)
        observed = checkpoint.get("feature_statistics_identity")
        if not isinstance(observed, str):
            raise ValueError(
                f"checkpoint {Path(path)} has no feature_statistics_identity; "
                "it predates the joint static-feature contract and is incompatible"
            )
        if observed != expected:
            raise ValueError(
                f"checkpoint {Path(path)} feature statistics mismatch: "
                f"expected {expected}, got {observed}"
            )

    def _schedule_phase_steps(self, horizon: int) -> tuple[int, int, int]:
        """Resolve warmup/stable/decay optimizer-step counts for logging/math."""

        horizon = max(1, horizon)
        if self.config.train.lr_schedule != "wsd":
            warmup = max(1, self.config.train.warmup)
            return warmup, 0, max(0, horizon - warmup)
        # WSD literature treats warmup as a fixed step-count phase, independent
        # of the eventual run horizon. The final decay remains horizon-relative;
        # stable training owns every optimizer step between those two phases.
        warmup = max(1, min(horizon, self.config.train.warmup))
        decay = int(round(horizon * self.config.train.lr_decay_ratio))
        decay = max(0, min(horizon - warmup, decay))
        stable = horizon - warmup - decay
        return warmup, stable, decay

    def _lr_multiplier(self, step_index: int) -> float:
        """Return the configured cosine, linear, or WSD LR multiplier."""

        max_steps = max(1, self.config.train.max_steps)
        warmup, stable, decay_steps = self._schedule_phase_steps(max_steps)
        if step_index < warmup:
            return float(step_index + 1) / float(warmup)
        if self.config.train.lr_schedule == "wsd":
            decay_start = warmup + stable
            if step_index < decay_start or decay_steps <= 0:
                return 1.0
            progress = min(1.0, float(step_index - decay_start) / float(decay_steps))
            return self.config.train.lr_floor_ratio + (1.0 - self.config.train.lr_floor_ratio) * (
                1.0 - progress
            )
        progress = min(1.0, float(step_index - warmup) / float(max(1, max_steps - warmup)))
        floor = self.config.train.lr_floor_ratio
        if self.config.train.lr_schedule == "linear":
            decay = 1.0 - progress
        else:
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * decay

    def _make_log(
        self,
        *,
        step: int,
        loss: float,
        denoising_loss: float,
        confidence_loss: float,
        per_class: dict[str, float],
        future_distance: dict[str, float],
        t_bucket_loss: dict[str, float],
        perspective_loss: dict[str, float],
        canvas_state_loss: dict[str, float],
        rare_class_t_bucket_loss: dict[str, float],
        rare_class_t_bucket_count: dict[str, int],
        lr: float,
        t_mean: float,
        noise_fraction: float,
        validation: ValidationLog | None,
        step_wall_seconds: float,
        tokens_per_second: float,
        cuda_max_memory_allocated_bytes: int,
        cuda_memory_allocated_bytes: int,
        cuda_memory_reserved_bytes: int,
        cuda_inactive_split_bytes: int,
        cuda_device_memory_used_bytes: int,
        cuda_device_memory_gap_bytes: int,
    ) -> TrainStepLog:
        # Values arrive already moved to the host (the caller batches the
        # GPU->CPU transfers in the lagged finalize). We only assemble the record
        # here, preserving the previous sorted ordering of the per-class and
        # future-distance dicts so the emitted JSONL is byte-stable.
        return TrainStepLog(
            step=step,
            loss=loss,
            denoising_loss=denoising_loss,
            confidence_loss=confidence_loss,
            per_class=dict(sorted(per_class.items())),
            future_distance=dict(sorted(future_distance.items())),
            t_bucket_loss=dict(sorted(t_bucket_loss.items())),
            perspective_loss=dict(sorted(perspective_loss.items())),
            canvas_state_loss=dict(sorted(canvas_state_loss.items())),
            rare_class_t_bucket_loss=dict(sorted(rare_class_t_bucket_loss.items())),
            rare_class_t_bucket_count=dict(sorted(rare_class_t_bucket_count.items())),
            lr=lr,
            t_mean=t_mean,
            noise_fraction=noise_fraction,
            step_wall_seconds=step_wall_seconds,
            tokens_per_second=tokens_per_second,
            cuda_max_memory_allocated_bytes=cuda_max_memory_allocated_bytes,
            cuda_memory_allocated_bytes=cuda_memory_allocated_bytes,
            cuda_memory_reserved_bytes=cuda_memory_reserved_bytes,
            cuda_inactive_split_bytes=cuda_inactive_split_bytes,
            cuda_device_memory_used_bytes=cuda_device_memory_used_bytes,
            cuda_device_memory_gap_bytes=cuda_device_memory_gap_bytes,
            validation=validation,
        )

    def _enforce_cuda_memory_limit(self, reserved_bytes: int) -> None:
        limit_gb = self.config.train.max_cuda_reserved_gb
        if self.device.type != "cuda" or limit_gb <= 0:
            return
        limit_bytes = int(limit_gb * 1024**3)
        if reserved_bytes < limit_bytes:
            return

        # `memory_reserved` includes completely unused blocks held by PyTorch's
        # caching allocator. Dynamic padding can visit several large shapes in
        # one shuffled epoch and cache a segment for each even though the live
        # allocation returns to a small, stable baseline after every step. Treat
        # the configured ceiling as a cache-trim trigger first; otherwise a
        # healthy run can be killed merely because reclaimable blocks happen to
        # sum above the limit. `empty_cache` preserves live tensors and makes the
        # post-trim reservation the meaningful safety signal.
        torch.cuda.empty_cache()
        reserved_after_trim = int(torch.cuda.memory_reserved(self.device))
        print(
            "cuda_cache_trim reason=reserved_memory_ceiling "
            f"reserved_before_gb={reserved_bytes / 1024**3:.3f} "
            f"reserved_after_gb={reserved_after_trim / 1024**3:.3f} "
            f"limit_gb={limit_gb:.3f}",
            flush=True,
        )
        if reserved_after_trim >= limit_bytes:
            raise RuntimeError(
                "CUDA reserved-memory safety limit exceeded after cache trim: "
                f"reserved_before={reserved_bytes / 1024**3:.3f} GiB, "
                f"reserved_after={reserved_after_trim / 1024**3:.3f} GiB, "
                f"limit={limit_gb:.3f} GiB"
            )

    def _maybe_checkpoint(self) -> None:
        """Persist a resumable checkpoint on the configured step cadence.

        Every `checkpoint_interval` steps this overwrites `last.pt` (the single
        file the resume path reads) and publishes it remotely, so a crash or
        spot preemption loses at most one interval of training. When
        `keep_step_checkpoints` is set it also retains a `step-N.pt` snapshot.
        Metrics are flushed remotely on the same cadence for live monitoring.
        """

        interval = self.config.train.checkpoint_interval
        if interval <= 0 or self.global_step % interval != 0:
            return
        self.save_checkpoint(self.resume_checkpoint_path)
        if self.config.train.keep_step_checkpoints:
            self.save_checkpoint(
                self.resume_checkpoint_path.parent / f"step-{self.global_step}.pt"
            )
        self._publish_metrics()

    def _write_metrics_line(self, log: TrainStepLog) -> None:
        """Append one JSON line describing this step to the metrics file.

        Cheap append-per-step (no remote I/O) so it never bottlenecks training;
        remote publishing happens on the checkpoint cadence. Includes loss,
        per-class losses, lr, noise fraction, and any validation log so the
        run can be tracked and aborted early from the JSONL alone.

        Both modes retain input/fog/future telemetry because both now consume
        the shared clamped input grammar.
        """

        if self.metrics_path is None:
            return
        record = asdict(log)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _publish_metrics(self) -> None:
        if self.metrics_publisher is not None and self.metrics_path is not None and self.metrics_path.exists():
            self.metrics_publisher(self.metrics_path)
        if (
            self.metrics_publisher is not None
            and self.epoch_metrics_path is not None
            and self.epoch_metrics_path.exists()
        ):
            self.metrics_publisher(self.epoch_metrics_path)
        if (
            self.metrics_publisher is not None
            and self.interval_metrics_path is not None
            and self.interval_metrics_path.exists()
        ):
            self.metrics_publisher(self.interval_metrics_path)

    def _write_epoch_metrics(self, metrics: EpochMetrics) -> None:
        if self.epoch_metrics_path is None:
            return
        # Use the SAME taxonomy map that CanvasCrossEntropyLoss used to build
        # per-class losses (5-entry pre-training map vs 7-entry debut map), so
        # the CSV columns declared here always match the keys that
        # `train_per_class`/`dev_per_class` (below) actually contain. See
        # `active_class_id_to_name`'s docstring in model/loss.py for why this
        # single shared helper is required.
        active_class_map = active_class_id_to_name(self.config)
        class_names = [_metric_class_name(name) for name in active_class_map.values()]
        # Columns emitted in BOTH pipelines: the loss headline, per-class losses,
        # and the new t-bucket / perspective breakdowns.
        fieldnames = [
            "epoch",
            "train_loss",
            "dev_loss",
            *(f"train_{name}_loss" for name in class_names),
            *(f"dev_{name}_loss" for name in class_names),
            *(f"train_t_bucket_loss_{name}" for name in T_BUCKET_NAMES),
            *(f"dev_t_bucket_loss_{name}" for name in T_BUCKET_NAMES),
            *(f"train_perspective_loss_{name}" for name in PERSPECTIVE_NAMES),
            *(f"dev_perspective_loss_{name}" for name in PERSPECTIVE_NAMES),
            *(f"train_canvas_state_loss_{name}" for name in CANVAS_STATE_NAMES),
            *(f"dev_canvas_state_loss_{name}" for name in CANVAS_STATE_NAMES),
            # Rare-class x t-bucket cross decomposition: 12 loss cells and their
            # 12 scored-position counts, per split. The count columns are not
            # redundant -- a loss cell averaged over 2 positions and one averaged
            # over 200 look identical without them, and a 0 there is the finding.
            *(f"train_rare_class_loss_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"dev_rare_class_loss_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"train_rare_class_count_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"dev_rare_class_count_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
        ]
        fieldnames += [
                "average_input_timesteps",
                "average_enemy_future_timesteps",
                "input_timestep_p50",
                "input_timestep_p90",
                "input_timestep_p95",
                "enemy_future_timestep_p50",
                "enemy_future_timestep_p90",
                "enemy_future_timestep_p95",
                *(
                    f"train_enemy_future_loss_distance_{name}"
                    for name in FUTURE_DISTANCE_BUCKETS
                ),
                *(
                    f"dev_enemy_future_loss_distance_{name}"
                    for name in FUTURE_DISTANCE_BUCKETS
                ),
        ]
        fieldnames += [
            "total_tokens_ingested",
            "total_unique_tokens_seen",
            "tokens_per_second",
            "wall_clock_elapsed_seconds",
            "average_cuda_device_memory_used_bytes",
            "average_cuda_device_memory_gap_bytes",
        ]
        row: dict[str, object] = {
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "dev_loss": "" if metrics.dev_loss is None else metrics.dev_loss,
            "total_tokens_ingested": metrics.total_tokens_ingested,
            "total_unique_tokens_seen": metrics.total_unique_tokens_seen,
            "tokens_per_second": metrics.tokens_per_second,
            "wall_clock_elapsed_seconds": metrics.wall_clock_elapsed_seconds,
            "average_cuda_device_memory_used_bytes": metrics.average_cuda_device_memory_used_bytes,
            "average_cuda_device_memory_gap_bytes": metrics.average_cuda_device_memory_gap_bytes,
        }
        for source_name in active_class_map.values():
            name = _metric_class_name(source_name)
            row[f"train_{name}_loss"] = metrics.train_per_class.get(source_name, "")
            row[f"dev_{name}_loss"] = metrics.dev_per_class.get(source_name, "")
        # Empty bucket/perspective -> "" (the same convention per-class columns
        # use for a class that scored no tokens this epoch).
        for name in T_BUCKET_NAMES:
            row[f"train_t_bucket_loss_{name}"] = metrics.train_t_bucket_loss.get(name, "")
            row[f"dev_t_bucket_loss_{name}"] = metrics.dev_t_bucket_loss.get(name, "")
        for name in PERSPECTIVE_NAMES:
            row[f"train_perspective_loss_{name}"] = metrics.train_perspective_loss.get(name, "")
            row[f"dev_perspective_loss_{name}"] = metrics.dev_perspective_loss.get(name, "")
        for name in CANVAS_STATE_NAMES:
            row[f"train_canvas_state_loss_{name}"] = metrics.train_canvas_state_loss.get(name, "")
            row[f"dev_canvas_state_loss_{name}"] = metrics.dev_canvas_state_loss.get(name, "")
        # Loss cells follow the blank-when-empty convention. Count cells do NOT:
        # a populated split reports every count including 0 (that zero is the
        # observation), and only a split that never ran at all leaves them blank.
        for name in RARE_CLASS_T_BUCKET_NAMES:
            row[f"train_rare_class_loss_{name}"] = (
                metrics.train_rare_class_t_bucket_loss.get(name, "")
            )
            row[f"dev_rare_class_loss_{name}"] = (
                metrics.dev_rare_class_t_bucket_loss.get(name, "")
            )
            row[f"train_rare_class_count_{name}"] = (
                metrics.train_rare_class_t_bucket_counts.get(name, "")
            )
            row[f"dev_rare_class_count_{name}"] = (
                metrics.dev_rare_class_t_bucket_counts.get(name, "")
            )
        row["average_input_timesteps"] = metrics.average_input_timesteps
        row["average_enemy_future_timesteps"] = metrics.average_enemy_future_timesteps
        for percentile in ("p50", "p90", "p95"):
            row[f"input_timestep_{percentile}"] = metrics.input_timestep_percentiles[percentile]
            row[f"enemy_future_timestep_{percentile}"] = (
                metrics.enemy_future_timestep_percentiles[percentile]
            )
        for name in FUTURE_DISTANCE_BUCKETS:
            row[f"train_enemy_future_loss_distance_{name}"] = (
                metrics.train_future_distance.get(name, "")
            )
            row[f"dev_enemy_future_loss_distance_{name}"] = (
                metrics.dev_future_distance.get(name, "")
            )
        self.epoch_metrics_path = _append_csv_row(self.epoch_metrics_path, fieldnames, row)

    def _write_interval_metrics(self, metrics: IntervalMetrics) -> None:
        """Append one intra-epoch diagnostic row to the interval CSV.

        Emits INTERVAL_REPORTS_PER_EPOCH rows per epoch (see that constant for
        why). Every loss column is scoped to the slice of the epoch since the
        previous row, so reading a column top-to-bottom shows that sub-class's
        loss actually moving -- including within a single epoch, which is the
        case a per-epoch CSV cannot report on at all.

        Columns mirror the epoch CSV's loss columns exactly (same class taxonomy
        helper, same t-bucket / perspective / canvas-state name tuples), so the
        two files are directly comparable. No-op when no interval metrics path
        was configured.

        Args:
            metrics: the assembled row. Empty breakdown keys become blank cells,
                matching the epoch CSV's convention.

        Calls: active_class_id_to_name, _metric_class_name, _append_csv_row.
        """

        if self.interval_metrics_path is None:
            return
        active_class_map = active_class_id_to_name(self.config)
        class_names = [_metric_class_name(name) for name in active_class_map.values()]
        fieldnames = [
            "epoch",
            "interval",
            "epoch_fraction",
            "global_step",
            "epoch_batch_index",
            "batches_in_epoch",
            "train_loss",
            "dev_loss",
            *(f"train_{name}_loss" for name in class_names),
            *(f"dev_{name}_loss" for name in class_names),
            *(f"train_t_bucket_loss_{name}" for name in T_BUCKET_NAMES),
            *(f"dev_t_bucket_loss_{name}" for name in T_BUCKET_NAMES),
            *(f"train_canvas_state_loss_{name}" for name in CANVAS_STATE_NAMES),
            *(f"dev_canvas_state_loss_{name}" for name in CANVAS_STATE_NAMES),
            *(f"train_perspective_loss_{name}" for name in PERSPECTIVE_NAMES),
            *(f"dev_perspective_loss_{name}" for name in PERSPECTIVE_NAMES),
            *(f"train_rare_class_loss_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"dev_rare_class_loss_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"train_rare_class_count_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            *(f"dev_rare_class_count_{name}" for name in RARE_CLASS_T_BUCKET_NAMES),
            "lr",
            "wall_clock_elapsed_seconds",
        ]
        row: dict[str, object] = {
            "epoch": metrics.epoch,
            "interval": metrics.interval,
            "epoch_fraction": metrics.epoch_fraction,
            "global_step": metrics.global_step,
            "epoch_batch_index": metrics.epoch_batch_index,
            "batches_in_epoch": metrics.batches_in_epoch,
            "train_loss": "" if metrics.train_loss is None else metrics.train_loss,
            "dev_loss": "" if metrics.dev_loss is None else metrics.dev_loss,
            "lr": metrics.lr,
            "wall_clock_elapsed_seconds": metrics.wall_clock_elapsed_seconds,
        }
        for source_name in active_class_map.values():
            name = _metric_class_name(source_name)
            row[f"train_{name}_loss"] = metrics.train_per_class.get(source_name, "")
            row[f"dev_{name}_loss"] = metrics.dev_per_class.get(source_name, "")
        for name in T_BUCKET_NAMES:
            row[f"train_t_bucket_loss_{name}"] = metrics.train_t_bucket_loss.get(name, "")
            row[f"dev_t_bucket_loss_{name}"] = metrics.dev_t_bucket_loss.get(name, "")
        for name in CANVAS_STATE_NAMES:
            row[f"train_canvas_state_loss_{name}"] = metrics.train_canvas_state_loss.get(name, "")
            row[f"dev_canvas_state_loss_{name}"] = metrics.dev_canvas_state_loss.get(name, "")
        for name in PERSPECTIVE_NAMES:
            row[f"train_perspective_loss_{name}"] = metrics.train_perspective_loss.get(name, "")
            row[f"dev_perspective_loss_{name}"] = metrics.dev_perspective_loss.get(name, "")
        for name in RARE_CLASS_T_BUCKET_NAMES:
            row[f"train_rare_class_loss_{name}"] = (
                metrics.train_rare_class_t_bucket_loss.get(name, "")
            )
            row[f"dev_rare_class_loss_{name}"] = (
                metrics.dev_rare_class_t_bucket_loss.get(name, "")
            )
            row[f"train_rare_class_count_{name}"] = (
                metrics.train_rare_class_t_bucket_counts.get(name, "")
            )
            row[f"dev_rare_class_count_{name}"] = (
                metrics.dev_rare_class_t_bucket_counts.get(name, "")
            )
        self.interval_metrics_path = _append_csv_row(
            self.interval_metrics_path,
            fieldnames,
            row,
        )
        print(
            f"interval_report epoch={metrics.epoch} "
            f"interval={metrics.interval}/{INTERVAL_REPORTS_PER_EPOCH} "
            f"step={metrics.global_step} "
            + (
                "train_loss=none"
                if metrics.train_loss is None
                else f"train_loss={metrics.train_loss:.6f}"
            )
            + " "
            + (
                "dev_loss=none"
                if metrics.dev_loss is None
                else f"dev_loss={metrics.dev_loss:.6f}"
            ),
            flush=True,
        )

    def _record_training_batch_metrics(self, batch: DiffusionBatch) -> int:
        input_tokens = batch.input_token_ids[batch.input_attention_mask]
        canvas_tokens = batch.target_canvas[batch.canvas_attention_mask]
        token_count = int(input_tokens.numel() + canvas_tokens.numel())
        self.total_tokens_ingested += token_count
        unique_batch_tokens = torch.unique(torch.cat((input_tokens, canvas_tokens)))
        self.unique_token_ids_seen.update(int(token_id) for token_id in unique_batch_tokens.tolist())
        return token_count

    def _save_epoch_checkpoints(self, dev_loss: float | None) -> None:
        """Retain one best-dev checkpoint and immutable epoch milestones."""

        epoch = self.completed_epochs
        if (
            self.config.train.save_best_checkpoint
            and dev_loss is not None
            and math.isfinite(dev_loss)
            and dev_loss < self.best_dev_loss
        ):
            best_dir = self.checkpoint_dir / self.config.train.best_checkpoint_subdir
            best_dir.mkdir(parents=True, exist_ok=True)
            self.best_dev_loss = dev_loss
            current = self.save_checkpoint(best_dir / f"epoch-{epoch:04d}.pt")
            # Write the improved checkpoint before removing its predecessor so
            # a serialization failure never destroys the last known-good best.
            for prior in best_dir.glob("epoch-*.pt"):
                if prior != current:
                    prior.unlink()

        interval = self.config.train.durable_checkpoint_interval_epochs
        if interval > 0 and epoch > 0 and epoch % interval == 0:
            durable_dir = self.checkpoint_dir / self.config.train.durable_checkpoint_subdir
            self.save_checkpoint(durable_dir / f"epoch-{epoch:04d}.pt")

    def _should_stop_early(self, dev_loss: float) -> bool:
        patience = self.config.train.early_stopping_patience_epochs
        if patience <= 0:
            return False
        minimum = self.config.train.early_stopping_min_relative_improvement
        if not 0.0 <= minimum < 1.0:
            raise ValueError("train.early_stopping_min_relative_improvement must be in [0, 1)")
        if (
            not math.isfinite(self.early_stopping_best_dev_loss)
            or dev_loss <= self.early_stopping_best_dev_loss * (1.0 - minimum)
        ):
            self.early_stopping_best_dev_loss = dev_loss
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= patience

    def _effective_accumulation_steps(self, batch: DiffusionBatch) -> int:
        configured = self.config.train.accumulation_steps
        target_tokens = self.config.train.target_effective_batch_tokens
        if target_tokens <= 0:
            return configured
        microbatch_tokens = int(batch.input_attention_mask.sum().item() + batch.target_canvas.numel())
        if microbatch_tokens <= 0:
            return configured
        return max(configured, math.ceil(target_tokens / microbatch_tokens))

    def _sample_self_conditioning_rows(self, batch_size: int) -> torch.Tensor:
        """Choose stopped estimate conditioning independently for each row."""

        probability = self.config.train.self_cond_prob
        if probability <= 0.0:
            return torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        if probability >= 1.0:
            return torch.ones(batch_size, device=self.device, dtype=torch.bool)
        generator_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        return torch.rand(
            batch_size, device=generator_device, generator=self.generator
        ) < probability

    def _resolve_ema_decay(self, total_steps: int) -> float:
        """Size the EMA averaging window to a run of `total_steps` steps.

        An EMA with decay `d` averages over an effective window of `1/(1-d)`
        steps. Holding `d` at a constant therefore pins that window to a fixed
        step count regardless of how long the run is: the previous fixed 0.9999
        is a ~10,000-step window, so a 3,400-step overfit run ended with an EMA
        that had not traversed its own window even once and whose served weights
        still carried a large share of early-training (near-init) parameters.

        This derives the decay from the run instead. `train.ema_horizon_ratio` is
        the window expressed as a fraction of the run's steps, so at 0.1 the EMA
        always averages over roughly the final 10% of training -- about ten
        window lengths before the last step, which is enough for the average to
        have fully turned over -- whatever the epoch budget happens to be.
        `train.ema_decay` stays in play as a CEILING, so a very long run tops out
        at a sane window rather than growing one without bound.

        Parameters:
            total_steps: the run's total optimizer-step horizon. Values <= 0 are
                treated as a 1-step horizon, which yields decay 0.0 (the EMA
                simply mirrors the raw weights) -- the right answer for a run too
                short to average over.

        Returns:
            The target decay in `[0, train.ema_decay]`.

        Calls: nothing. Reads only `self.config.train`.

        Called by: `__init__` (initial value) and `fit` (authoritative value,
        from the step budget fit() derives).
        """

        ratio = self.config.train.ema_horizon_ratio
        ceiling = self.config.train.ema_decay
        if not 0.0 < ratio <= 1.0:
            raise ValueError("train.ema_horizon_ratio must be in (0, 1]")
        if not 0.0 <= ceiling <= 1.0:
            raise ValueError("train.ema_decay must be in [0, 1]")
        # max(1.0, ...) keeps the window at least one step so the reciprocal
        # below stays <= 1 and the decay stays in [0, 1).
        window_steps = max(1.0, ratio * float(max(0, total_steps)))
        derived_decay = 1.0 - 1.0 / window_steps
        return min(ceiling, derived_decay)

    def _update_ema(self) -> None:
        # Ramp the decay in over the first steps instead of jumping straight to
        # the target. `self.global_step` has not been incremented yet when this
        # is called, so `updates` is how many EMA updates INCLUDING this one have
        # happened; it is restored from the checkpoint on resume, so the ramp
        # picks up where it left off rather than restarting.
        #
        # Why the ramp: the EMA buffer starts as a copy of the randomly
        # initialized weights, and at the target decay it takes a full window to
        # forget them. (1+n)/(10+n) is 0.18 at the first update and climbs past
        # 0.9 by n=90, so the first handful of updates are dominated by the live
        # weights and the EMA is a usable set of weights from epoch 1 -- which
        # matters here because dev validation and every periodic checkpoint read
        # the EMA weights, not the raw ones. The ramp only ever LOWERS the decay,
        # so it can never overshoot the run-fitted target.
        updates = self.global_step + 1
        decay = min(self._ema_target_decay, float(1 + updates) / float(10 + updates))
        # Build the tensor cache once. state_dict() returns references to the
        # SAME underlying parameter/buffer tensors on every call, and both the
        # optimizer step and load_state_dict mutate those tensors in place, so
        # caching the references stays correct across steps and resumes. We
        # split float tensors (which get the decayed moving-average update) from
        # non-float buffers (e.g. integer counters, copied verbatim) so all the
        # float work can be fused below.
        if self._ema_tensor_cache is None:
            raw_state = self.model.state_dict()
            float_ema: list[torch.Tensor] = []
            float_raw: list[torch.Tensor] = []
            nonfloat_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for name, ema_value in self.ema_model.state_dict().items():
                raw_value = raw_state[name]
                if torch.is_floating_point(ema_value):
                    float_ema.append(ema_value)
                    float_raw.append(raw_value)
                else:
                    nonfloat_pairs.append((ema_value, raw_value))
            self._ema_tensor_cache = (float_ema, float_raw, nonfloat_pairs)
        float_ema, float_raw, nonfloat_pairs = self._ema_tensor_cache
        with torch.no_grad():
            # ema = decay * ema + (1 - decay) * raw, computed identically to the
            # previous per-tensor `mul_(decay).add_(raw, alpha=1-decay)` loop but
            # fused across every float tensor into two kernel launches instead of
            # two per parameter. This is the same arithmetic, just far less
            # main-thread dispatch and GPU launch overhead per step.
            torch._foreach_mul_(float_ema, decay)
            torch._foreach_add_(float_ema, float_raw, alpha=1.0 - decay)
            for ema_value, raw_value in nonfloat_pairs:
                ema_value.copy_(raw_value)

    def _maybe_validate(
        self,
        val_dataloader: Iterable[DiffusionBatch] | None,
        *,
        step: int,
        fixed_t: float | None,
    ) -> ValidationLog | None:
        # `step` is passed explicitly (rather than reading self.global_step)
        # because validation now runs in the lagged finalize, one step after
        # self.global_step has already advanced; the interval must be checked
        # against the step the log line actually belongs to.
        interval = self.config.train.val_interval
        if val_dataloader is None or interval <= 0 or step % interval != 0:
            return None
        return self.validate(val_dataloader, fixed_t=fixed_t)

    @torch.no_grad()
    def validate(self, dataloader: Iterable[DiffusionBatch], *, fixed_t: float | None = None) -> ValidationLog:
        """Evaluate held-out loss with EMA weights."""

        was_training = self.ema_model.training
        self.ema_model.eval()
        loss_sum = 0.0
        loss_count = 0
        class_sums: dict[str, float] = {}
        class_counts: dict[str, int] = {}
        future_distance_sums: dict[str, float] = {}
        future_distance_counts: dict[str, int] = {}
        # t-bucket / perspective validation aggregation mirrors per_class: sum of
        # per-batch means over the batches that actually populated each key,
        # divided by that count.
        t_bucket_sums: dict[str, float] = {}
        t_bucket_counts: dict[str, int] = {}
        perspective_sums: dict[str, float] = {}
        perspective_counts: dict[str, int] = {}
        canvas_state_sums: dict[str, float] = {}
        canvas_state_counts: dict[str, int] = {}
        # Rare-class cells pool by scored positions rather than by batch, so
        # these are running totals of the loss sums and the position counts --
        # not sums of per-batch means like the dicts above. See
        # LossOutput.rare_class_t_bucket_sums.
        rare_class_sums: dict[str, float] = {}
        rare_class_counts: dict[str, int] = {}
        for batch in dataloader:
            batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t, model=self.ema_model)
            loss_sum += float(batch_loss.loss.detach().cpu())
            loss_count += 1
            for name, value in batch_loss.loss_output.per_class.items():
                class_sums[name] = class_sums.get(name, 0.0) + float(value.detach().cpu())
                class_counts[name] = class_counts.get(name, 0) + 1
            _accumulate_future_distance(
                future_distance_sums,
                future_distance_counts,
                batch_loss.loss_output,
            )
            for name, value in batch_loss.loss_output.t_bucket.items():
                t_bucket_sums[name] = t_bucket_sums.get(name, 0.0) + float(value.detach().cpu())
                t_bucket_counts[name] = t_bucket_counts.get(name, 0) + 1
            for name, value in batch_loss.loss_output.perspective.items():
                perspective_sums[name] = perspective_sums.get(name, 0.0) + float(value.detach().cpu())
                perspective_counts[name] = perspective_counts.get(name, 0) + 1
            for name, value in batch_loss.loss_output.canvas_state.items():
                canvas_state_sums[name] = (
                    canvas_state_sums.get(name, 0.0) + float(value.detach().cpu())
                )
                canvas_state_counts[name] = canvas_state_counts.get(name, 0) + 1
            for name, value in batch_loss.loss_output.rare_class_t_bucket_sums.items():
                rare_class_sums[name] = (
                    rare_class_sums.get(name, 0.0) + float(value.detach().cpu())
                )
            for name, value in batch_loss.loss_output.rare_class_t_bucket_counts.items():
                rare_class_counts[name] = (
                    rare_class_counts.get(name, 0) + int(value.detach().cpu())
                )
        if was_training:
            self.ema_model.train()
        if loss_count == 0:
            raise ValueError("validation dataloader yielded no batches")
        per_class = {
            name: class_sums[name] / class_counts[name]
            for name in sorted(class_sums)
        }
        return ValidationLog(
            loss=loss_sum / loss_count,
            per_class=per_class,
            future_distance=_finalize_future_distance(
                future_distance_sums,
                future_distance_counts,
            ),
            t_bucket={
                name: t_bucket_sums[name] / t_bucket_counts[name]
                for name in sorted(t_bucket_sums)
            },
            perspective={
                name: perspective_sums[name] / perspective_counts[name]
                for name in sorted(perspective_sums)
            },
            canvas_state={
                name: canvas_state_sums[name] / canvas_state_counts[name]
                for name in sorted(canvas_state_sums)
            },
            rare_class_t_bucket=_finalize_rare_class_t_bucket(
                rare_class_sums,
                rare_class_counts,
            ),
            rare_class_t_bucket_counts=dict(rare_class_counts),
        )


def auxiliary_confidence_loss(
    canvas_logits: torch.Tensor,
    target_canvas: torch.Tensor,
    scored_mask: torch.Tensor,
) -> torch.Tensor:
    """LLaDA2.0 CAP-style entropy sharpening on already-correct canvas predictions."""

    predicted = canvas_logits.argmax(dim=-1)
    active = scored_mask.to(torch.bool) & (predicted == target_canvas)
    if not active.any():
        return canvas_logits.new_zeros(())
    log_probs = torch.log_softmax(canvas_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy[active].mean()


def _metric_class_name(name: str) -> str:
    return name.strip("[]").replace("-", "_").lower()


def _append_csv_row(
    path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> Path:
    """Append one row without allowing a Windows file lock to kill training.

    A run whose CSV was written by an older column schema (e.g. before the
    canvas-state breakdown existed) would otherwise raise on the first append.
    Instead the existing rows are rewritten under the CURRENT header -- dropping
    columns that no longer exist and leaving newly added ones blank -- so a run
    resumed after a schema change keeps its history in one readable file.

    Args:
        path: destination CSV; parent directories must already exist. This may
            already be a continuation selected after an earlier lock.
        fieldnames: the current column schema, in emission order.
        row: values keyed by ``fieldnames``; missing keys become blank cells.

    Returns:
        The path that received the row. Normally this is ``path``. If another
        Windows process keeps ``path`` write-locked through the bounded retries,
        this is a new timestamped continuation CSV containing the readable
        history plus the new row. Callers retain the returned path for all later
        rows and publishing, so logging remains contiguous and training proceeds.

    Called by: TrainingLoop._write_epoch_metrics and
        TrainingLoop._write_interval_metrics.
    """

    retry_delays = (0.25, 0.5, 1.0, 2.0)
    last_error: PermissionError | None = None
    for attempt in range(len(retry_delays) + 1):
        if attempt:
            time.sleep(retry_delays[attempt - 1])
        try:
            _append_csv_row_at_path(path, fieldnames, row)
            return path
        except PermissionError as exc:
            last_error = exc

    continuation_path, copied_rows = _write_csv_continuation(path, fieldnames, row)
    print(
        "metrics_csv_locked action=continued "
        f"locked_path={path} continuation_path={continuation_path} "
        f"history_rows_copied={copied_rows} error={last_error}",
        flush=True,
    )
    return continuation_path


def _append_csv_row_at_path(
    path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> None:
    """Append or schema-migrate one CSV path; propagate a writer lock."""

    write_header = True
    migration_path: Path | None = None
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fieldnames:
                write_header = False
                existing_rows = None
            else:
                existing_rows = list(reader)
        if existing_rows is not None:
            # Rewrite the whole file under the new header, then swap it in.
            migration_path = path.with_suffix(f"{path.suffix}.schema-migration")
            try:
                with migration_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=fieldnames,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(existing_rows)
                migration_path.replace(path)
            finally:
                # A locked destination can make replace() fail after the
                # migration file was fully written. It is only a generated temp
                # artifact, so do not leave it behind across every retry.
                if migration_path.exists():
                    try:
                        migration_path.unlink()
                    except OSError:
                        pass
            write_header = False
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_csv_continuation(
    locked_path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> tuple[Path, int]:
    """Write a full-history continuation beside a persistently locked CSV."""

    existing_rows: list[dict[str, str]] = []
    try:
        if locked_path.exists() and locked_path.stat().st_size > 0:
            with locked_path.open(newline="", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))
    except PermissionError:
        # Some lockers deny reads as well as writes. The current row is still
        # persisted; the console message's copied-row count makes the absent
        # history explicit instead of silently pretending it was recovered.
        existing_rows = []

    stamp = time.strftime("%Y%m%d-%H%M%S")
    continuation_path = locked_path.with_name(
        f"{locked_path.stem}-continued-{stamp}-{time.time_ns()}{locked_path.suffix}"
    )
    with continuation_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
    return continuation_path, len(existing_rows)


def _latest_metrics_csv_path(canonical_path: Path) -> Path:
    """Resume the newest continuation, if a prior launch escaped a file lock."""

    continuations = sorted(
        canonical_path.parent.glob(
            f"{canonical_path.stem}-continued-*{canonical_path.suffix}"
        )
    )
    return continuations[-1] if continuations else canonical_path


def _mean_of_lists(accumulated: dict[str, list[float]]) -> dict[str, float]:
    """Simple-average each named list of per-microbatch means, key-sorted.

    Used for every loss breakdown that accumulates one mean per microbatch
    (per-class, t-bucket, perspective, canvas-state). Names that accumulated no
    values never appear as keys in the input and so are absent from the output,
    which is the empty-key convention the CSV writers turn into a blank cell.

    Args:
        accumulated: name -> list of per-microbatch mean losses.

    Returns:
        name -> mean of that list, iterated in sorted key order so emitted rows
        are byte-stable.
    """

    return {
        name: sum(values) / len(values)
        for name, values in sorted(accumulated.items())
        if values
    }


def _timestep_percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        name: float(torch.quantile(tensor, quantile).item())
        for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95))
    }


def _accumulate_future_distance(
    sums: dict[str, float],
    counts: dict[str, int],
    loss_output: LossOutput,
) -> None:
    for name, value in loss_output.future_distance.items():
        count = loss_output.future_distance_counts[name]
        sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * count
        counts[name] = counts.get(name, 0) + count


def _finalize_future_distance(
    sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    return {
        name: sums[name] / counts[name]
        for name in FUTURE_DISTANCE_BUCKETS
        if counts.get(name, 0) > 0
    }


def _finalize_rare_class_t_bucket(
    sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    """Reduce accumulated rare-class cell sums/counts to per-cell mean losses.

    The division is what makes these cells POOLED means over every scored
    position, rather than means of per-microbatch means: a cell holding 6
    positions in one microbatch and 0 in the next must not weight those equally.

    Args:
        sums: cell name -> total cross-entropy summed over its scored positions.
        counts: cell name -> number of scored positions, including explicit
            zeros for cells that scored nothing.

    Returns:
        Cell name -> mean loss, in the canonical RARE_CLASS_T_BUCKET_NAMES order,
        with zero-count cells OMITTED. That omission is the same blank-cell
        convention every other decomposition uses; the count itself is reported
        separately and does keep its zero, which is how a reader distinguishes
        "this bucket contained no [END] token" from "this bucket was not
        evaluated".

    Called by: TrainingLoop.fit (epoch and interval rows), TrainingLoop.validate,
        and the per-step log assembly in _finalize.
    """

    return {
        name: sums[name] / counts[name]
        for name in RARE_CLASS_T_BUCKET_NAMES
        if counts.get(name, 0) > 0
    }


def move_batch_to_device(batch: DiffusionBatch, device: torch.device) -> DiffusionBatch:
    non_blocking = device.type == "cuda"
    features = batch.input_features
    moved_features = InputFeatures(
        continuous_values=features.continuous_values.to(device, non_blocking=non_blocking),
        continuous_validity=features.continuous_validity.to(
            device, non_blocking=non_blocking
        ),
        categorical_values=features.categorical_values.to(
            device, non_blocking=non_blocking
        ),
        allegiance_values=features.allegiance_values.to(device, non_blocking=non_blocking),
        feature_mask=features.feature_mask.to(device, non_blocking=non_blocking),
    )
    return DiffusionBatch(
        input_token_ids=batch.input_token_ids.to(device, non_blocking=non_blocking),
        input_attention_mask=batch.input_attention_mask.to(device, non_blocking=non_blocking),
        input_lengths=batch.input_lengths.to(device, non_blocking=non_blocking),
        target_canvas=batch.target_canvas.to(device, non_blocking=non_blocking),
        canvas_attention_mask=batch.canvas_attention_mask.to(device, non_blocking=non_blocking),
        class_labels=batch.class_labels.to(device, non_blocking=non_blocking),
        canvas_loss_mask=batch.canvas_loss_mask.to(device, non_blocking=non_blocking),
        terminated=batch.terminated.to(device, non_blocking=non_blocking),
        truncated=batch.truncated.to(device, non_blocking=non_blocking),
        perspective_ids=batch.perspective_ids.to(device, non_blocking=non_blocking),
        input_timestep_counts=batch.input_timestep_counts,
        enemy_future_timestep_counts=batch.enemy_future_timestep_counts,
        canvas_prediction_distances=batch.canvas_prediction_distances.to(
            device,
            non_blocking=non_blocking,
        ),
        input_records=batch.input_records,
        canvas_metadata=batch.canvas_metadata,
        input_features=moved_features,
    )
