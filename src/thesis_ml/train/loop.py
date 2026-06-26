"""Plain PyTorch training loop for SC2 masked discrete diffusion."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from thesis_ml.config import ProjectConfig
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.loss import CanvasCrossEntropyLoss, LossOutput
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


@dataclass(frozen=True)
class TrainStepLog:
    step: int
    loss: float
    denoising_loss: float
    confidence_loss: float
    per_class: dict[str, float]
    lr: float
    t_mean: float
    masked_fraction: float
    validation: ValidationLog | None = None


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
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
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
    ) -> list[TrainStepLog]:
        """Run optimizer steps and return per-step logs."""

        target_steps = self.config.train.max_steps if max_steps is None else max_steps
        base_accumulation_steps = self.config.train.accumulation_steps
        if base_accumulation_steps < 1:
            raise ValueError("train.accumulation_steps must be >= 1")

        logs: list[TrainStepLog] = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        data_iter = iter(dataloader)
        while self.global_step < target_steps:
            last_batch_loss: BatchLoss | None = None
            try:
                first_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                first_batch = next(data_iter)
            accumulation_steps = self._effective_accumulation_steps(first_batch)
            for microstep in range(accumulation_steps):
                try:
                    batch = first_batch if microstep == 0 else next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t)
                (batch_loss.loss / accumulation_steps).backward()
                last_batch_loss = batch_loss

            if self.config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self._update_ema()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if last_batch_loss is None:
                raise RuntimeError("optimizer step completed without a batch")
            validation = self._maybe_validate(val_dataloader, fixed_t=fixed_t)
            logs.append(self._make_log(last_batch_loss, validation=validation))
            self._maybe_checkpoint()

        if target_steps > 0:
            self.save_checkpoint(self.checkpoint_dir / "last.pt")
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
                        canvas_attention_mask=batch.canvas_loss_mask,
                        input_records=batch.input_records,
                    )
                    canvas_self_conditioning = canvas_self_conditioning_from_logits(
                        estimate.logits[:, input_len:, :]
                    )
            forward_kwargs = {
                "input_token_ids": corruption.input_token_ids,
                "canvas_token_ids": corruption.noised_canvas,
                "input_attention_mask": batch.input_attention_mask,
                "canvas_attention_mask": batch.canvas_loss_mask,
                "input_records": batch.input_records,
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
                "config": self.config,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint["global_step"])

    def _lr_multiplier(self, step_index: int) -> float:
        warmup = max(1, self.config.train.warmup)
        max_steps = max(warmup + 1, self.config.train.max_steps)
        if step_index < warmup:
            return float(step_index + 1) / float(warmup)
        progress = min(1.0, float(step_index - warmup) / float(max_steps - warmup))
        floor = self.config.train.lr_floor_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    def _make_log(self, batch_loss: BatchLoss, *, validation: ValidationLog | None) -> TrainStepLog:
        per_class = {
            name: float(value.detach().cpu())
            for name, value in sorted(batch_loss.loss_output.per_class.items())
        }
        return TrainStepLog(
            step=self.global_step,
            loss=float(batch_loss.loss.detach().cpu()),
            denoising_loss=float(batch_loss.denoising_loss.detach().cpu()),
            confidence_loss=float(batch_loss.confidence_loss.detach().cpu()),
            per_class=per_class,
            lr=float(self.optimizer.param_groups[0]["lr"]),
            t_mean=float(batch_loss.corruption.t.detach().mean().cpu()),
            masked_fraction=float(batch_loss.scored_mask.float().mean().detach().cpu()),
            validation=validation,
        )

    def _maybe_checkpoint(self) -> None:
        interval = self.config.train.checkpoint_interval
        if interval > 0 and self.global_step % interval == 0:
            self.save_checkpoint(self.checkpoint_dir / f"step-{self.global_step}.pt")

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
        for batch in dataloader:
            batch_loss = self.compute_batch_loss(batch, fixed_t=fixed_t, model=self.ema_model)
            losses.append(batch_loss.loss.detach())
            for name, value in batch_loss.loss_output.per_class.items():
                class_totals.setdefault(name, []).append(value.detach())
        if was_training:
            self.ema_model.train()
        if not losses:
            raise ValueError("validation dataloader yielded no batches")
        per_class = {
            name: float(torch.stack(values).mean().cpu())
            for name, values in sorted(class_totals.items())
        }
        return ValidationLog(loss=float(torch.stack(losses).mean().cpu()), per_class=per_class)


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


def move_batch_to_device(batch: DiffusionBatch, device: torch.device) -> DiffusionBatch:
    return DiffusionBatch(
        input_token_ids=batch.input_token_ids.to(device),
        input_attention_mask=batch.input_attention_mask.to(device),
        input_lengths=batch.input_lengths.to(device),
        target_canvas=batch.target_canvas.to(device),
        class_labels=batch.class_labels.to(device),
        canvas_loss_mask=batch.canvas_loss_mask.to(device),
        terminated=batch.terminated.to(device),
        truncated=batch.truncated.to(device),
        input_records=batch.input_records,
        canvas_metadata=batch.canvas_metadata,
    )
