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
    PRETRAIN_CLASS_ID_TO_NAME,
)


def active_class_id_to_name(config: ProjectConfig) -> dict[int, str]:
    """Pick the class-id -> class-name map that matches the current run.

    WHY this function exists: pre-training and debut fine-tuning use DIFFERENT
    class taxonomies, but the loss module (per-class decomposition, below) and
    the training loop (epoch-CSV column names, in train/loop.py) must always
    agree on which class names exist. Both call this SAME helper so they can
    never drift apart (e.g. loss.py emitting a per-class key that loop.py has
    no CSV column for).

    The two maps are:
      - Pre-training (``config.data.debut_mode`` False):
        ``PRETRAIN_CLASS_ID_TO_NAME`` -- observed/fogged/future reconstruction
        names plus structural and outcome classes.
      - Fine-tuning (``config.data.debut_mode`` True): ``DEBUT_CLASS_ID_TO_NAME``
        -- the dense 7-entry debut taxonomy (visible/fogged/future-debut, the
        structural tokens, and the win/loss outcome token).

    Args:
        config: The full project config. Only ``config.data.debut_mode`` is
            read.

    Returns:
        ``DEBUT_CLASS_ID_TO_NAME`` when debut mode is active, otherwise
        ``PRETRAIN_CLASS_ID_TO_NAME``.
    """

    if config.data.debut_mode:
        return DEBUT_CLASS_ID_TO_NAME
    return PRETRAIN_CLASS_ID_TO_NAME


# Future-distance decomposition buckets shared by full-rollout pretraining and
# debut fine-tuning.
FUTURE_DISTANCE_BUCKETS = {
    "1": (1, 1),
    "2_5": (2, 5),
    "6_10": (6, 10),
    "11_30": (11, 30),
    "31_plus": (31, None),
}

# t-bucket loss-breakdown names, in the canonical order used for CSV columns.
# Each training/eval example's sampled noise level t (from the corruption
# step) lands in EXACTLY ONE of these contiguous, exhaustive buckets over [0, 1]:
#   t == 1.0            -> "t_eq_1"
#   0.7 <= t < 1.0      -> "t_0_7_to_1_0"
#   0.5 <= t < 0.7      -> "t_0_5_to_0_7"
#   0.3 <= t < 0.5      -> "t_0_3_to_0_5"
#   0.0 <= t < 0.3      -> "t_0_0_to_0_3"
# Emitted in BOTH pre-training and fine-tuning.
T_BUCKET_NAMES = (
    "t_eq_1",
    "t_0_7_to_1_0",
    "t_0_5_to_0_7",
    "t_0_3_to_0_5",
    "t_0_0_to_0_3",
)

# Perspective-split loss-breakdown names. Each example is built from one player's
# perspective (``DatasetExample.perspective_player``); "p1" means p1 is the
# viewer and p2 is the reconstructed enemy, and vice versa for "p2". Emitted in
# BOTH pipelines. Integer ids below are the representation carried on the batch
# (see data/collate.py) so the perspective survives ``.to(device)``.
PERSPECTIVE_NAMES = ("p1", "p2")
PERSPECTIVE_P1 = 1
PERSPECTIVE_P2 = 2


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    per_class: dict[str, torch.Tensor]
    future_distance: dict[str, torch.Tensor]
    future_distance_counts: dict[str, int]
    # Clean-state CE broken down by the example's sampled t-bucket and by the
    # example's player perspective. Both follow the SAME emptiness convention as
    # ``per_class``: a bucket/perspective with zero scored tokens is simply
    # ABSENT from the dict (no key), rather than present-with-a-sentinel.
    t_bucket: dict[str, torch.Tensor]
    perspective: dict[str, torch.Tensor]


class CanvasCrossEntropyLoss(nn.Module):
    """Loss for canvas positions only.

    Fused cross entropy is deliberately optional and off by default. With this
    project's small vocabulary, its memory savings are marginal and do not
    justify an extra dependency in v1.
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.use_fused_cross_entropy = config.loss.use_fused_cross_entropy
        # Pick the taxonomy map ONCE up front (see active_class_id_to_name). The
        # per-class decomposition in forward() derives its keys from this single
        # map, so pre-training and debut mode can never disagree with each other
        # or with the CSV columns train/loop.py writes (which call the same
        # helper). ``debut_mode`` is cached because it gates configured class
        # weighting, while distance decomposition remains shared.
        self.class_id_to_name = active_class_id_to_name(config)
        self.debut_mode = config.data.debut_mode

        # The weight buffer is indexed by raw class-id, so it MUST be sized by
        # max(id) + 1, NOT len(map), so stable raw ids remain safe.
        buffer_size = max(self.class_id_to_name) + 1
        class_weights = torch.ones(buffer_size, dtype=torch.float32)
        if self.debut_mode:
            # Fine-tuning: per-class config weighting (unchanged behavior). The
            # config guarantees class_loss_weights is populated in debut mode.
            weights = config.loss.class_loss_weights
            class_weights[CLASS_ENEMY_OBSERVED] = weights.enemy_observed_reconstruction
            class_weights[CLASS_ENEMY_FOGGED] = weights.enemy_fogged_reconstruction
            class_weights[CLASS_ENEMY_FUTURE] = weights.enemy_future_prediction
            class_weights[CLASS_DELIMITER] = weights.delimiter
            class_weights[CLASS_END] = weights.end
            class_weights[CLASS_PAD] = weights.pad
            class_weights[CLASS_WINLOSS] = weights.win_loss
        # Pre-training leaves every class at 1.0, including semantic [PAD].
        # Batch-shape padding is excluded by the caller's scored mask.
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
        sampled_t: torch.Tensor | None = None,
        perspective_ids: torch.Tensor | None = None,
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

        # t-bucket breakdown (BOTH pipelines). Each example's single sampled t
        # (shape [B]) assigns ALL that example's scored canvas positions to one
        # bucket; the clean-state CE mean is then taken over every scored position in
        # that bucket across the batch. Empty buckets are omitted (per_class
        # convention). See T_BUCKET_NAMES for the exact, exhaustive boundaries.
        t_bucket: dict[str, torch.Tensor] = {}
        if sampled_t is not None:
            t_row = sampled_t.to(ce.device)
            bucket_row_masks = {
                "t_eq_1": t_row == 1.0,
                "t_0_7_to_1_0": (t_row >= 0.7) & (t_row < 1.0),
                "t_0_5_to_0_7": (t_row >= 0.5) & (t_row < 0.7),
                "t_0_3_to_0_5": (t_row >= 0.3) & (t_row < 0.5),
                "t_0_0_to_0_3": t_row < 0.3,
            }
            for name in T_BUCKET_NAMES:
                bucket_mask = active & bucket_row_masks[name].unsqueeze(1)
                if bucket_mask.any():
                    t_bucket[name] = ce[bucket_mask].mean()

        # Perspective breakdown (BOTH pipelines). Same shape of logic as the
        # t-bucket split, but partitioning examples by which player perspective
        # they were built from. Empty perspectives are omitted (per_class
        # convention).
        perspective: dict[str, torch.Tensor] = {}
        if perspective_ids is not None:
            perspective_row = perspective_ids.to(ce.device)
            for name, perspective_id in (
                ("p1", PERSPECTIVE_P1),
                ("p2", PERSPECTIVE_P2),
            ):
                perspective_mask = active & (perspective_row == perspective_id).unsqueeze(1)
                if perspective_mask.any():
                    perspective[name] = ce[perspective_mask].mean()

        return LossOutput(
            loss=aggregate,
            per_class=per_class,
            future_distance=future_distance,
            future_distance_counts=future_distance_counts,
            t_bucket=t_bucket,
            perspective=perspective,
        )
