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
        ``PRETRAIN_CLASS_ID_TO_NAME`` -- a SPARSE 5-entry map (ids 0/3/4/5/6).
        Pre-training collapses the observed/fogged/future split into a single
        "content" class (id 0) and never emits ids 1 or 2, so those two ids are
        intentionally absent from the map. CRITICAL: because the map is sparse,
        any id-indexed buffer (e.g. the class-weight buffer below) must be sized
        ``max(map) + 1`` (= 7), NOT ``len(map)`` (= 5).
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


# Future-distance decomposition buckets. FINE-TUNING ONLY: pre-training collapses
# the future class into "content" and never labels a token CLASS_ENEMY_FUTURE, so
# these buckets are computed only when ``config.data.debut_mode`` is True (see
# CanvasCrossEntropyLoss.forward, which gates on ``self.debut_mode``).
FUTURE_DISTANCE_BUCKETS = {
    "1": (1, 1),
    "2_5": (2, 5),
    "6_10": (6, 10),
    "11_30": (11, 30),
    "31_plus": (31, None),
}

# t-bucket loss-breakdown names, in the canonical order used for CSV columns.
# Each training/eval example's sampled masking level t (from the corruption
# step) lands in EXACTLY ONE of these contiguous, exhaustive buckets over [0, 1]:
#   t == 1.0            -> "t_eq_1"
#   0.7 <= t < 1.0      -> "t_0_7_to_1_0"
#   0.5 <= t < 0.7      -> "t_0_5_to_0_7"
#   0.3 <= t < 0.5      -> "t_0_3_to_0_5"
#   0.0 <= t < 0.3      -> "t_0_0_to_0_3"  (MIN_T clamping keeps t > 0, still here)
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
    # Masked-CE broken down by the example's sampled t-bucket and by the
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
        # helper). ``debut_mode`` is cached because it also gates the
        # future-distance decomposition (fine-tuning-only) in forward().
        self.class_id_to_name = active_class_id_to_name(config)
        self.debut_mode = config.data.debut_mode

        # The weight buffer is indexed by raw class-id, so it MUST be sized by
        # max(id) + 1, NOT len(map). Pre-training's map is SPARSE (ids 0/3/4/5/6,
        # so len == 5) but ids still range up to 6 -- sizing by len would
        # under-allocate and crash when indexing class-id 6. max(id)+1 == 7 in
        # both taxonomies today.
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
        else:
            # Pre-training: fully uniform, published-MDLM-style loss weighting --
            # 1.0 for every class (already the buffer's fill value) except PAD,
            # which is zeroed so padding positions never contribute. NOTE: we
            # deliberately do NOT read config.loss.class_loss_weights here; it is
            # None for pre-training configs (fog / per-class weights are a
            # fine-tuning-only concern per config validation).
            class_weights[CLASS_PAD] = 0.0
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
        # Future-distance decomposition is FINE-TUNING ONLY. Pre-training never
        # labels a token CLASS_ENEMY_FUTURE (the future class is collapsed into
        # "content"), so we gate explicitly on the mode rather than relying on
        # the label simply never appearing -- pre-training must emit no
        # future-distance keys at all.
        if self.debut_mode and prediction_distances is not None:
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
        # bucket; the masked-CE mean is then taken over every scored position in
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
