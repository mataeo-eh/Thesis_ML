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
    CLASS_WINLOSS,
    DEBUT_CLASS_ID_TO_NAME,
)

# The ORIGINAL 6-class pre-training id->name map. This constant must NEVER be
# mutated, renamed, or reordered: `train/loop.py` and existing tests import it
# directly, and pretraining's per-class loss keys / epoch-CSV columns must stay
# byte-for-byte identical to what they were before debut-mode support existed.
CLASS_ID_TO_NAME = {
    CLASS_ENEMY_OBSERVED: "enemy-observed",
    CLASS_ENEMY_FOGGED: "enemy-fogged",
    CLASS_ENEMY_FUTURE: "enemy-future",
    CLASS_DELIMITER: "[DELIMITER]",
    CLASS_END: "[END]",
    CLASS_PAD: "[PAD]",
}


def active_class_id_to_name(config: ProjectConfig) -> dict[int, str]:
    """Pick the class-id -> class-name map that matches the current run.

    WHY this function exists: pre-training canvases only ever contain 6
    classes (reconstruction/fogged/future/delimiter/end/pad), while debut
    fine-tuning canvases (``config.data.debut_mode`` True) add a 7th class --
    the single win/loss outcome token (``CLASS_WINLOSS``, id 6) -- and rename
    the first three classes to describe "debut" state instead of plain
    reconstruction (see ``DEBUT_CLASS_ID_TO_NAME`` in data/dataset.py, the
    shared map produced by the dataset worker).

    Both the loss module (class-weight buffer sizing + per-class loss
    decomposition, below) and the training loop (epoch-CSV column names, in
    train/loop.py) call this SAME helper so they always agree on which class
    names exist for a given config. If either module built its own map
    independently, the two could drift apart (e.g. loss.py emitting a
    "win-loss" per-class key that loop.py has no CSV column for). Centralizing
    the choice here is also what guarantees pretraining's per-class keys and
    CSV columns stay byte-for-byte unchanged: when debut_mode is False this
    always returns the original, untouched CLASS_ID_TO_NAME map.

    Args:
        config: The full project config. Only ``config.data.debut_mode`` is
            read.

    Returns:
        ``DEBUT_CLASS_ID_TO_NAME`` (7 classes) when debut mode is active,
        otherwise the original ``CLASS_ID_TO_NAME`` (6 classes).
    """

    if config.data.debut_mode:
        return DEBUT_CLASS_ID_TO_NAME
    return CLASS_ID_TO_NAME


FUTURE_DISTANCE_BUCKETS = {
    "1": (1, 1),
    "2_5": (2, 5),
    "6_10": (6, 10),
    "11_30": (11, 30),
    "31_plus": (31, None),
}


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    per_class: dict[str, torch.Tensor]
    future_distance: dict[str, torch.Tensor]
    future_distance_counts: dict[str, int]


class CanvasCrossEntropyLoss(nn.Module):
    """Loss for canvas positions only.

    Fused cross entropy is deliberately optional and off by default. With this
    project's small vocabulary, its memory savings are marginal and do not
    justify an extra dependency in v1.
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.use_fused_cross_entropy = config.loss.use_fused_cross_entropy
        # Pick the 6-class (pretraining) or 7-class (debut fine-tuning) name
        # map ONCE up front. Every downstream computation in this class (the
        # weight buffer's length below, and the per-class decomposition in
        # forward()) is derived from this single map, so pretraining and
        # debut mode can never disagree with each other or with the CSV
        # columns train/loop.py writes (which call the same helper). See
        # active_class_id_to_name's docstring for the full rationale.
        self.class_id_to_name = active_class_id_to_name(config)
        weights = config.loss.class_loss_weights
        class_weights = torch.ones(len(self.class_id_to_name), dtype=torch.float32)
        # Ids 0-5 mean the same positions in BOTH taxonomies (debut mode only
        # renames them, it does not renumber them), so these assignments are
        # correct regardless of which mode is active.
        class_weights[CLASS_ENEMY_OBSERVED] = weights.enemy_observed_reconstruction
        class_weights[CLASS_ENEMY_FOGGED] = weights.enemy_fogged_reconstruction
        class_weights[CLASS_ENEMY_FUTURE] = weights.enemy_future_prediction
        class_weights[CLASS_DELIMITER] = weights.delimiter
        class_weights[CLASS_END] = weights.end
        class_weights[CLASS_PAD] = weights.pad
        if config.data.debut_mode:
            # Id 6 (CLASS_WINLOSS) only exists in the debut taxonomy. Setting
            # it only in this branch keeps the pretraining buffer exactly
            # length-6 with exactly the same values as before this change.
            class_weights[CLASS_WINLOSS] = weights.win_loss
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        canvas_logits: torch.Tensor,
        target_canvas: torch.Tensor,
        class_labels: torch.Tensor,
        *,
        scored_mask: torch.Tensor | None = None,
        position_weights: torch.Tensor | None = None,
        prediction_distances: torch.Tensor | None = None,
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
        for class_id, name in self.class_id_to_name.items():
            class_mask = active & (class_labels == class_id)
            if class_mask.any():
                per_class[name] = ce[class_mask].mean()

        future_distance: dict[str, torch.Tensor] = {}
        future_distance_counts: dict[str, int] = {}
        if prediction_distances is not None:
            distances = prediction_distances.to(ce.device)
            future_mask = active & (class_labels == CLASS_ENEMY_FUTURE)
            for name, (minimum, maximum) in FUTURE_DISTANCE_BUCKETS.items():
                bucket_mask = future_mask & (distances >= minimum)
                if maximum is not None:
                    bucket_mask &= distances <= maximum
                count = int(bucket_mask.sum().item())
                if count:
                    future_distance[name] = ce[bucket_mask].mean()
                    future_distance_counts[name] = count

        return LossOutput(
            loss=aggregate,
            per_class=per_class,
            future_distance=future_distance,
            future_distance_counts=future_distance_counts,
        )
