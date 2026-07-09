"""Plain PyTorch training loop for SC2 masked discrete diffusion."""

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
    FUTURE_DISTANCE_BUCKETS,
    PERSPECTIVE_NAMES,
    T_BUCKET_NAMES,
    CanvasCrossEntropyLoss,
    LossOutput,
    active_class_id_to_name,
)
from thesis_ml.model.model import canvas_self_conditioning_from_logits
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
    self_conditioning_used: bool


@dataclass(frozen=True)
class ValidationLog:
    loss: float
    per_class: dict[str, float]
    future_distance: dict[str, float]
    t_bucket: dict[str, float]
    perspective: dict[str, float]


@dataclass(frozen=True)
class TrainStepLog:
    step: int
    loss: float
    denoising_loss: float
    confidence_loss: float
    per_class: dict[str, float]
    future_distance: dict[str, float]
    # Masked-CE broken down by the example's sampled t-bucket and by player
    # perspective. Emitted in BOTH pipelines. Empty buckets/perspectives are
    # simply absent from these dicts (per_class convention).
    t_bucket_loss: dict[str, float]
    perspective_loss: dict[str, float]
    lr: float
    t_mean: float
    masked_fraction: float
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
    total_tokens_ingested: int
    total_unique_tokens_seen: int
    tokens_per_second: float
    wall_clock_elapsed_seconds: float
    average_cuda_device_memory_used_bytes: float
    average_cuda_device_memory_gap_bytes: float


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
        self.epoch_metrics_path = Path(epoch_metrics_path) if epoch_metrics_path is not None else None
        if self.epoch_metrics_path is not None:
            self.epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.best_train_loss = math.inf
        self.epochs_without_improvement = 0
        self.elapsed_wall_seconds = 0.0
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
        if configured_steps > 0:
            target_steps = configured_steps
            epoch_limit = max(epoch_count, math.ceil(target_steps / max(1, batches_per_epoch)))
        else:
            if epoch_count < 1:
                raise ValueError("train.epochs must be >= 1 when train.max_steps is 0")
            target_steps = batches_per_epoch * epoch_count
            epoch_limit = epoch_count
        base_accumulation_steps = self.config.train.accumulation_steps
        if base_accumulation_steps < 1:
            raise ValueError("train.accumulation_steps must be >= 1")

        logs: list[TrainStepLog] = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        fit_started = time.perf_counter()

        # Surface the effective LR schedule up front so a run never silently
        # trains at a near-zero rate. base_lr is the peak; the linear warmup
        # means the EFFECTIVE lr only reaches base_lr after `warmup` steps, so a
        # warmup that is large relative to target_steps keeps the whole run at a
        # tiny fraction of base_lr. warmup_end_lr shows the lr at the first step
        # AFTER warmup; if it is far below base_lr, warmup is eating the run.
        base_lr = self.config.train.lr
        warmup = max(1, self.config.train.warmup)
        print(
            f"lr_schedule base_lr={base_lr:.3e} warmup_steps={warmup} "
            f"target_steps={target_steps} "
            f"effective_lr_at_step_1={base_lr * self._lr_multiplier(0):.3e} "
            f"effective_lr_after_warmup={base_lr * self._lr_multiplier(warmup):.3e}"
            + (
                "  [WARNING: warmup >= target_steps -> the run never leaves warmup; "
                "lower train.warmup]"
                if warmup >= target_steps
                else ""
            ),
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
                nonlocal epoch_cuda_memory_samples
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
                    epoch_losses.append(float(mb["loss"].cpu()))
                    for name, value in mb["per_class"].items():
                        epoch_class_losses.setdefault(name, []).append(float(value.cpu()))
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
                        epoch_t_bucket_losses.setdefault(name, []).append(float(value.cpu()))
                    for name, value in mb["perspective"].items():
                        epoch_perspective_losses.setdefault(name, []).append(float(value.cpu()))

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
                    lr=record["lr"],
                    t_mean=float(last["t_mean"].cpu()),
                    masked_fraction=float(last["masked_fraction"].cpu()),
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
                            "t_mean": batch_loss.corruption.t.detach().mean(),
                            "masked_fraction": batch_loss.scored_mask.float().mean().detach(),
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
                train_per_class={
                    name: sum(values) / len(values)
                    for name, values in sorted(epoch_class_losses.items())
                },
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
                train_t_bucket_loss={
                    name: sum(values) / len(values)
                    for name, values in sorted(epoch_t_bucket_losses.items())
                },
                dev_t_bucket_loss=(
                    epoch_validation.t_bucket if epoch_validation is not None else {}
                ),
                train_perspective_loss={
                    name: sum(values) / len(values)
                    for name, values in sorted(epoch_perspective_losses.items())
                },
                dev_perspective_loss=(
                    epoch_validation.perspective if epoch_validation is not None else {}
                ),
                total_tokens_ingested=self.total_tokens_ingested,
                total_unique_tokens_seen=len(self.unique_token_ids_seen),
                tokens_per_second=epoch_tokens / epoch_training_duration,
                wall_clock_elapsed_seconds=self.elapsed_wall_seconds + (time.perf_counter() - fit_started),
                average_cuda_device_memory_used_bytes=(
                    epoch_cuda_device_memory_used / epoch_cuda_memory_samples
                ),
                average_cuda_device_memory_gap_bytes=(
                    epoch_cuda_device_memory_gap / epoch_cuda_memory_samples
                ),
            )
            self._write_epoch_metrics(epoch_metrics)
            if self.device.type == "cuda" and self.config.train.empty_cuda_cache_after_epoch:
                torch.cuda.empty_cache()
            if self._should_stop_early(epoch_metrics.train_loss):
                print(
                    f"early_stopping=triggered epoch={self.completed_epochs} "
                    f"best_train_loss={self.best_train_loss:.6f} "
                    f"patience={self.config.train.early_stopping_patience_epochs}",
                    flush=True,
                )
                break

        self.elapsed_wall_seconds += time.perf_counter() - fit_started

        if target_steps > 0:
            # Final durable checkpoint + metrics flush so a clean finish leaves
            # the same resumable state a mid-run preemption would.
            self.save_checkpoint(self.checkpoint_dir / "last.pt")
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
            schedule=self.config.diffusion.mask_schedule,
            generator=self.generator,
            t=fixed_t,
        )

        scored_mask = corruption.masked_positions & batch.canvas_loss_mask
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.train.precision == "bf16" and self.device.type in {"cpu", "cuda"},
        ):
            active_model = self.model if model is None else model
            input_len = batch.input_token_ids.shape[1]
            canvas_self_conditioning = None
            self_conditioning_used = model is None and self._use_self_conditioning()
            if self_conditioning_used:
                with torch.no_grad():
                    estimate = active_model(
                        input_token_ids=corruption.input_token_ids,
                        canvas_token_ids=corruption.noised_canvas,
                        input_attention_mask=batch.input_attention_mask,
                        canvas_attention_mask=batch.canvas_attention_mask,
                        input_features=batch.input_features,
                    )
                    canvas_self_conditioning = canvas_self_conditioning_from_logits(
                        estimate.logits[:, input_len:, :]
                    )
            forward_kwargs = {
                "input_token_ids": corruption.input_token_ids,
                "canvas_token_ids": corruption.noised_canvas,
                "input_attention_mask": batch.input_attention_mask,
                "canvas_attention_mask": batch.canvas_attention_mask,
                "input_features": batch.input_features,
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
            self_conditioning_used=self_conditioning_used,
        )

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.config.train.checkpoint_dir)

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
                "best_train_loss": self.best_train_loss,
                "epochs_without_improvement": self.epochs_without_improvement,
                "elapsed_wall_seconds": self.elapsed_wall_seconds,
                "total_tokens_ingested": self.total_tokens_ingested,
                "unique_token_ids_seen": sorted(self.unique_token_ids_seen),
                "config": self.config,
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
        self.best_train_loss = float(checkpoint.get("best_train_loss", math.inf))
        self.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        self.elapsed_wall_seconds = float(checkpoint.get("elapsed_wall_seconds", 0.0))
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
        counter (`completed_epochs`, `best_train_loss`, etc.) so training
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
        # Copy the plain model's weights.
        self.model.load_state_dict(checkpoint["model"])
        # The EMA (shadow) model tracks a smoothed copy of the weights used at
        # evaluation time. Older checkpoints may lack an "ema_model" key, in
        # which case we fall back to seeding the EMA copy with the same plain
        # model weights so both start out identical, as they would for a
        # freshly constructed loop.
        self.ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        # NOTE: optimizer, scheduler, global_step, completed_epochs,
        # best_train_loss, epochs_without_improvement, elapsed_wall_seconds,
        # total_tokens_ingested, and unique_token_ids_seen are intentionally
        # left untouched here -- that is what makes this a "warm start"
        # rather than a "resume".

    def _lr_multiplier(self, step_index: int) -> float:
        warmup = max(1, self.config.train.warmup)
        max_steps = max(warmup + 1, self.config.train.max_steps)
        if step_index < warmup:
            return float(step_index + 1) / float(warmup)
        progress = min(1.0, float(step_index - warmup) / float(max_steps - warmup))
        floor = self.config.train.lr_floor_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

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
        lr: float,
        t_mean: float,
        masked_fraction: float,
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
            lr=lr,
            t_mean=t_mean,
            masked_fraction=masked_fraction,
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
        if reserved_bytes >= limit_bytes:
            raise RuntimeError(
                "CUDA reserved-memory safety limit exceeded: "
                f"reserved={reserved_bytes / 1024**3:.3f} GiB, "
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
        self.save_checkpoint(self.checkpoint_dir / "last.pt")
        if self.config.train.keep_step_checkpoints:
            self.save_checkpoint(self.checkpoint_dir / f"step-{self.global_step}.pt")
        self._publish_metrics()

    def _write_metrics_line(self, log: TrainStepLog) -> None:
        """Append one JSON line describing this step to the metrics file.

        Cheap append-per-step (no remote I/O) so it never bottlenecks training;
        remote publishing happens on the checkpoint cadence. Includes loss,
        per-class losses, lr, masked fraction, and any validation log so the
        run can be tracked and aborted early from the JSONL alone.

        In PRE-TRAINING (``debut_mode`` False) the fine-tuning-only
        "future_distance" key is stripped from the serialized JSON entirely --
        not even emitted as an empty ``{}`` -- both at the top level and inside
        the nested "validation" sub-object. The dataclass fields themselves are
        kept (in-process consumers see a fixed shape); only the emitted JSON
        drops the key. Fine-tuning output is unchanged.
        """

        if self.metrics_path is None:
            return
        record = asdict(log)
        if not self.config.data.debut_mode:
            # Pre-training never has a future class, so the JSONL must contain
            # no "future_distance" key at all (see docstring above).
            record.pop("future_distance", None)
            validation_record = record.get("validation")
            if isinstance(validation_record, dict):
                validation_record.pop("future_distance", None)
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
        debut_mode = self.config.data.debut_mode
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
        ]
        # Input-side / fog-derived / future-distance columns are FINE-TUNING ONLY.
        # Pre-training has no input, no fog, and no future class, so these columns
        # are omitted entirely from a pre-training epoch CSV.
        if debut_mode:
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
        if debut_mode:
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
        write_header = self._prepare_epoch_metrics_file(fieldnames)
        with self.epoch_metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _prepare_epoch_metrics_file(self, fieldnames: list[str]) -> bool:
        if self.epoch_metrics_path is None:
            return False
        if not self.epoch_metrics_path.exists() or self.epoch_metrics_path.stat().st_size == 0:
            return True
        with self.epoch_metrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fieldnames:
                return False
            existing_rows = list(reader)
        migration_path = self.epoch_metrics_path.with_suffix(
            f"{self.epoch_metrics_path.suffix}.schema-migration"
        )
        with migration_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
        migration_path.replace(self.epoch_metrics_path)
        return False

    def _record_training_batch_metrics(self, batch: DiffusionBatch) -> int:
        input_tokens = batch.input_token_ids[batch.input_attention_mask]
        canvas_tokens = batch.target_canvas[batch.canvas_attention_mask]
        token_count = int(input_tokens.numel() + canvas_tokens.numel())
        self.total_tokens_ingested += token_count
        unique_batch_tokens = torch.unique(torch.cat((input_tokens, canvas_tokens)))
        self.unique_token_ids_seen.update(int(token_id) for token_id in unique_batch_tokens.tolist())
        return token_count

    def _should_stop_early(self, train_loss: float) -> bool:
        patience = self.config.train.early_stopping_patience_epochs
        if patience <= 0:
            return False
        minimum = self.config.train.early_stopping_min_relative_improvement
        if not 0.0 <= minimum < 1.0:
            raise ValueError("train.early_stopping_min_relative_improvement must be in [0, 1)")
        if not math.isfinite(self.best_train_loss) or train_loss <= self.best_train_loss * (1.0 - minimum):
            self.best_train_loss = train_loss
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

    def _use_self_conditioning(self) -> bool:
        if not self.config.model.self_conditioning:
            return False
        probability = self.config.train.self_cond_prob
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        generator_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        return bool(torch.rand((), device=generator_device, generator=self.generator).item() < probability)

    def _update_ema(self) -> None:
        decay = self.config.train.ema_decay
        if not 0.0 <= decay <= 1.0:
            raise ValueError("train.ema_decay must be in [0, 1]")
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


def move_batch_to_device(batch: DiffusionBatch, device: torch.device) -> DiffusionBatch:
    non_blocking = device.type == "cuda"
    features = batch.input_features
    moved_features = InputFeatures(
        map_values=features.map_values.to(device, non_blocking=non_blocking),
        stat_values=features.stat_values.to(device, non_blocking=non_blocking),
        team_ids=features.team_ids.to(device, non_blocking=non_blocking),
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
