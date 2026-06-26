"""Canvas-only cross-entropy with per-class decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from thesis_ml.config import ProjectConfig
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
)

CLASS_ID_TO_NAME = {
    CLASS_ENEMY_OBSERVED: "enemy-observed",
    CLASS_ENEMY_FOGGED: "enemy-fogged",
    CLASS_ENEMY_FUTURE: "enemy-future",
    CLASS_DELIMITER: "[DELIMITER]",
    CLASS_END: "[END]",
    CLASS_PAD: "[PAD]",
}


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    per_class: dict[str, torch.Tensor]


class CanvasCrossEntropyLoss(nn.Module):
    """Loss for canvas positions only.

    Fused cross entropy is deliberately optional and off by default. With this
    project's small vocabulary, its memory savings are marginal and do not
    justify an extra dependency in v1.
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.use_fused_cross_entropy = config.loss.use_fused_cross_entropy
        weights = config.loss.class_loss_weights
        class_weights = torch.ones(len(CLASS_ID_TO_NAME), dtype=torch.float32)
        class_weights[CLASS_ENEMY_OBSERVED] = weights.enemy_observed_reconstruction
        class_weights[CLASS_ENEMY_FOGGED] = weights.enemy_fogged_reconstruction
        class_weights[CLASS_ENEMY_FUTURE] = weights.enemy_future_prediction
        class_weights[CLASS_DELIMITER] = weights.delimiter
        class_weights[CLASS_END] = weights.end
        class_weights[CLASS_PAD] = weights.pad
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        canvas_logits: torch.Tensor,
        target_canvas: torch.Tensor,
        class_labels: torch.Tensor,
        *,
        scored_mask: torch.Tensor | None = None,
        position_weights: torch.Tensor | None = None,
    ) -> LossOutput:
        ce = F.cross_entropy(
            canvas_logits.transpose(1, 2),
            target_canvas,
            reduction="none",
        )
        active = torch.ones_like(ce, dtype=torch.bool) if scored_mask is None else scored_mask.to(torch.bool)
        weights = self.class_weights.to(ce.device)[class_labels]
        if position_weights is not None:
            weights = weights * position_weights.to(ce.device)
        weighted = ce * weights
        denominator = weights[active].sum().clamp_min(1e-8)
        aggregate = weighted[active].sum() / denominator

        per_class: dict[str, torch.Tensor] = {}
        for class_id, name in CLASS_ID_TO_NAME.items():
            class_mask = active & (class_labels == class_id)
            if class_mask.any():
                per_class[name] = ce[class_mask].mean()

        return LossOutput(loss=aggregate, per_class=per_class)
