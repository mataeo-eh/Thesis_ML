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
# Each training/eval example's sampled noise level t (from the corruption step)
# is that example's CORRUPTION LEVEL: the probability each canvas position was
# independently replaced. Every example lands in EXACTLY ONE of these
# contiguous, exhaustive buckets over [0, 1]:
#   t == 1.0             -> "t_eq_1"          (fully corrupted: pure generation)
#   0.75 <= t < 1.0      -> "t_0_75_to_1_0"   (nearly fully corrupted)
#   0.25 <= t < 0.75     -> "t_0_25_to_0_75"  (mid corruption)
#   0.00 <= t < 0.25     -> "t_0_0_to_0_25"   (lightly corrupted: mostly clean)
#
# WHY these boundaries: t == 1.0 is separated from "almost 1.0" on purpose. At
# exactly t=1 the model sees NO surviving ground-truth canvas token and must
# generate the sequence from the input alone, which is the hardest and most
# diagnostic regime; at t=0.99 a handful of true tokens survive and can anchor
# the rest, so averaging the two would hide how the model actually does at full
# corruption. Read these together with the CANVAS_STATE_NAMES split below:
# t-buckets say "how corrupted was the example", the canvas-state split says
# "how well did the model handle the clean vs corrupted positions inside it".
#
# Emitted in BOTH pre-training and fine-tuning.
T_BUCKET_NAMES = (
    "t_eq_1",
    "t_0_75_to_1_0",
    "t_0_25_to_0_75",
    "t_0_0_to_0_25",
)

# Canvas-state loss-breakdown names. This splits every scored canvas position by
# whether the token the MODEL ACTUALLY SAW at that position was already the
# correct one:
#   "ground_truth_preserved" -> the noised canvas token equals the target, so the
#       model's job is to RECOGNIZE it as already-correct and keep it. A falling
#       loss here means the model has learned what a valid sequence looks like
#       well enough to leave true tokens alone.
#   "noised"                 -> the noised canvas token differs from the target,
#       so the model must REPAIR it. This is the denoising work proper.
#
# WHY this is keyed on actual token inequality rather than on the corruption
# coin-flip: under uniform diffusion a "corrupted" position can be re-drawn as
# the very same token it already held. From the model's point of view that
# position is indistinguishable from an untouched one, so scoring it as "noised"
# would blur exactly the distinction this breakdown exists to measure. The
# caller therefore passes ``CorruptionOutput.changed_positions`` (token
# inequality), not ``corrupted_positions`` (the Bernoulli branch).
#
# Under the absorbing ablation only corrupted positions are scored at all, so
# "ground_truth_preserved" is legitimately empty there and is simply omitted
# (the same empty-key convention ``per_class`` uses).
CANVAS_STATE_NAMES = ("ground_truth_preserved", "noised")

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
    # Clean-state CE broken down by the example's sampled t-bucket, by the
    # example's player perspective, and by per-position canvas state
    # (ground-truth-preserved vs actually-noised). All three follow the SAME
    # emptiness convention as ``per_class``: a bucket/perspective/state with zero
    # scored tokens is simply ABSENT from the dict (no key), rather than
    # present-with-a-sentinel.
    t_bucket: dict[str, torch.Tensor]
    perspective: dict[str, torch.Tensor]
    canvas_state: dict[str, torch.Tensor]


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
        changed_positions: torch.Tensor | None = None,
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
                "t_0_75_to_1_0": (t_row >= 0.75) & (t_row < 1.0),
                "t_0_25_to_0_75": (t_row >= 0.25) & (t_row < 0.75),
                "t_0_0_to_0_25": t_row < 0.25,
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

        # Canvas-state breakdown (BOTH pipelines). Unlike the t-bucket split,
        # which partitions whole EXAMPLES by their sampled corruption level, this
        # partitions individual POSITIONS by whether the token the model was shown
        # there already matched the target. See CANVAS_STATE_NAMES for why this
        # uses token inequality rather than the corruption coin-flip.
        canvas_state: dict[str, torch.Tensor] = {}
        if changed_positions is not None:
            changed = changed_positions.to(device=ce.device, dtype=torch.bool)
            state_masks = {
                "ground_truth_preserved": active & ~changed,
                "noised": active & changed,
            }
            for name in CANVAS_STATE_NAMES:
                state_mask = state_masks[name]
                if state_mask.any():
                    canvas_state[name] = ce[state_mask].mean()

        return LossOutput(
            loss=aggregate,
            per_class=per_class,
            future_distance=future_distance,
            future_distance_counts=future_distance_counts,
            t_bucket=t_bucket,
            perspective=perspective,
            canvas_state=canvas_state,
        )
