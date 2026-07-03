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


@dataclass(frozen=True)
class TrainStepLog:
    step: int
    loss: float
    denoising_loss: float
    confidence_loss: float
    per_class: dict[str, float]
    future_distance: dict[str, float]
    lr: float
    t_mean: float
    masked_fraction: float
    step_wall_seconds: float
    tokens_per_second: float
    cuda_max_memory_allocated_bytes: int
    cuda_memory_reserved_bytes: int
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
    total_tokens_ingested: int
    total_unique_tokens_seen: int
    tokens_per_second: float
    wall_clock_elapsed_seconds: float


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
        self.best_train_loss = math.inf
        self.epochs_without_improvement = 0
        self.elapsed_wall_seconds = 0.0
        self.total_tokens_ingested = 0
        self.unique_token_ids_seen: set[int] = set()
        generator_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
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

        for epoch_index in range(self.completed_epochs, epoch_limit):
            if self.global_step >= target_steps:
                break
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch_index)
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
            epoch_batch_index = 0
            data_iter = iter(dataloader)

            while self.global_step < target_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_started = time.perf_counter()
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

                batch_losses: list[BatchLoss] = []
                step_tokens = 0
                for batch in microbatches:
                    epoch_batch_index += 1
                    print(
                        f"phase=train epoch={epoch_index + 1}/{epoch_limit} "
                        f"batch={epoch_batch_index}/{batches_per_epoch}",
                        flush=True,
                    )
                    batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t)
                    (batch_loss.loss / len(microbatches)).backward()
                    batch_losses.append(batch_loss)
                    epoch_losses.append(float(batch_loss.loss.detach().cpu()))
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
                    for name, value in batch_loss.loss_output.per_class.items():
                        epoch_class_losses.setdefault(name, []).append(float(value.detach().cpu()))
                    _accumulate_future_distance(
                        epoch_future_distance_sums,
                        epoch_future_distance_counts,
                        batch_loss.loss_output,
                    )

                if self.config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_wall_seconds = max(time.perf_counter() - step_started, 1e-9)
                tokens_per_second = step_tokens / step_wall_seconds
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
                print(
                    f"step={self.global_step} step_wall_seconds={step_wall_seconds:.3f} "
                    f"tokens_per_second={tokens_per_second:.1f} "
                    f"cuda_max_memory_allocated_gb={cuda_max_allocated / 1024**3:.3f} "
                    f"cuda_memory_reserved_gb={cuda_reserved / 1024**3:.3f}",
                    flush=True,
                )
                self._enforce_cuda_memory_limit(cuda_reserved)

                last_batch_loss = batch_losses[-1]
                validation = self._maybe_validate(val_dataloader, fixed_t=fixed_t)
                step_log = self._make_log(
                    last_batch_loss,
                    validation=validation,
                    step_wall_seconds=step_wall_seconds,
                    tokens_per_second=tokens_per_second,
                    cuda_max_memory_allocated_bytes=cuda_max_allocated,
                    cuda_memory_reserved_bytes=cuda_reserved,
                )
                logs.append(step_log)
                self._write_metrics_line(step_log)
                self._maybe_checkpoint()

            if not epoch_losses:
                break
            epoch_training_duration = max(time.perf_counter() - epoch_started, 1e-9)
            epoch_validation = (
                self.validate(val_dataloader, fixed_t=fixed_t)
                if val_dataloader is not None
                else None
            )
            self.completed_epochs = epoch_index + 1
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
                total_tokens_ingested=self.total_tokens_ingested,
                total_unique_tokens_seen=len(self.unique_token_ids_seen),
                tokens_per_second=epoch_tokens / epoch_training_duration,
                wall_clock_elapsed_seconds=self.elapsed_wall_seconds + (time.perf_counter() - fit_started),
            )
            self._write_epoch_metrics(epoch_metrics)
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
        batch_loss: BatchLoss,
        *,
        validation: ValidationLog | None,
        step_wall_seconds: float,
        tokens_per_second: float,
        cuda_max_memory_allocated_bytes: int,
        cuda_memory_reserved_bytes: int,
    ) -> TrainStepLog:
        per_class = {
            name: float(value.detach().cpu())
            for name, value in sorted(batch_loss.loss_output.per_class.items())
        }
        future_distance = {
            name: float(value.detach().cpu())
            for name, value in sorted(batch_loss.loss_output.future_distance.items())
        }
        return TrainStepLog(
            step=self.global_step,
            loss=float(batch_loss.loss.detach().cpu()),
            denoising_loss=float(batch_loss.denoising_loss.detach().cpu()),
            confidence_loss=float(batch_loss.confidence_loss.detach().cpu()),
            per_class=per_class,
            future_distance=future_distance,
            lr=float(self.optimizer.param_groups[0]["lr"]),
            t_mean=float(batch_loss.corruption.t.detach().mean().cpu()),
            masked_fraction=float(batch_loss.scored_mask.float().mean().detach().cpu()),
            step_wall_seconds=step_wall_seconds,
            tokens_per_second=tokens_per_second,
            cuda_max_memory_allocated_bytes=cuda_max_memory_allocated_bytes,
            cuda_memory_reserved_bytes=cuda_memory_reserved_bytes,
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

    def _write_epoch_metrics(self, metrics: EpochMetrics) -> None:
        if self.epoch_metrics_path is None:
            return
        # Use the SAME 6-class (pretraining) or 7-class (debut) name map that
        # CanvasCrossEntropyLoss used to build per-class losses, so the CSV
        # columns declared here always match the keys that
        # `train_per_class`/`dev_per_class` (below) actually contain. See
        # `active_class_id_to_name`'s docstring in model/loss.py for why this
        # single shared helper is required to keep pretraining's columns
        # byte-for-byte unchanged when debut mode is off.
        active_class_map = active_class_id_to_name(self.config)
        class_names = [_metric_class_name(name) for name in active_class_map.values()]
        fieldnames = [
            "epoch",
            "train_loss",
            "dev_loss",
            *(f"train_{name}_loss" for name in class_names),
            *(f"dev_{name}_loss" for name in class_names),
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
            "total_tokens_ingested",
            "total_unique_tokens_seen",
            "tokens_per_second",
            "wall_clock_elapsed_seconds",
        ]
        row: dict[str, object] = {
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "dev_loss": "" if metrics.dev_loss is None else metrics.dev_loss,
            "average_input_timesteps": metrics.average_input_timesteps,
            "average_enemy_future_timesteps": metrics.average_enemy_future_timesteps,
            "total_tokens_ingested": metrics.total_tokens_ingested,
            "total_unique_tokens_seen": metrics.total_unique_tokens_seen,
            "tokens_per_second": metrics.tokens_per_second,
            "wall_clock_elapsed_seconds": metrics.wall_clock_elapsed_seconds,
        }
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
        for source_name in active_class_map.values():
            name = _metric_class_name(source_name)
            row[f"train_{name}_loss"] = metrics.train_per_class.get(source_name, "")
            row[f"dev_{name}_loss"] = metrics.dev_per_class.get(source_name, "")
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
        with torch.no_grad():
            raw_state = self.model.state_dict()
            for name, ema_value in self.ema_model.state_dict().items():
                raw_value = raw_state[name]
                if torch.is_floating_point(ema_value):
                    ema_value.mul_(decay).add_(raw_value.detach(), alpha=1.0 - decay)
                else:
                    ema_value.copy_(raw_value)

    def _maybe_validate(
        self,
        val_dataloader: Iterable[DiffusionBatch] | None,
        *,
        fixed_t: float | None,
    ) -> ValidationLog | None:
        interval = self.config.train.val_interval
        if val_dataloader is None or interval <= 0 or self.global_step % interval != 0:
            return None
        return self.validate(val_dataloader, fixed_t=fixed_t)

    @torch.no_grad()
    def validate(self, dataloader: Iterable[DiffusionBatch], *, fixed_t: float | None = None) -> ValidationLog:
        """Evaluate held-out loss with EMA weights."""

        was_training = self.ema_model.training
        self.ema_model.eval()
        losses: list[torch.Tensor] = []
        class_totals: dict[str, list[torch.Tensor]] = {}
        future_distance_sums: dict[str, float] = {}
        future_distance_counts: dict[str, int] = {}
        for batch in dataloader:
            batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t, model=self.ema_model)
            losses.append(batch_loss.loss.detach())
            for name, value in batch_loss.loss_output.per_class.items():
                class_totals.setdefault(name, []).append(value.detach())
            _accumulate_future_distance(
                future_distance_sums,
                future_distance_counts,
                batch_loss.loss_output,
            )
        if was_training:
            self.ema_model.train()
        if not losses:
            raise ValueError("validation dataloader yielded no batches")
        per_class = {
            name: float(torch.stack(values).mean().cpu())
            for name, values in sorted(class_totals.items())
        }
        return ValidationLog(
            loss=float(torch.stack(losses).mean().cpu()),
            per_class=per_class,
            future_distance=_finalize_future_distance(
                future_distance_sums,
                future_distance_counts,
            ),
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
