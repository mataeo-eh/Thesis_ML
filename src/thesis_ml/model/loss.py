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

# Row order of the per-canvas-state token count matrices returned in
# ``LossOutput.token_class_counts``. Each matrix has shape ``[3, vocab_size]``
# and every column is one token id in the output vocabulary:
#
#   row 0 "true_positive" -> positions where the argmax prediction equalled the
#                            target AND the target was this token id
#   row 1 "predicted"     -> positions where the argmax prediction was this
#                            token id, regardless of whether it was right
#   row 2 "target"        -> positions where the target was this token id
#                            (this token's support)
#
# WHY counts rather than finished accuracy/F1 numbers: counts are ADDITIVE, so
# the training loop can pool them across microbatches, across an epoch, and
# across a whole dev pass by plain summation and compute one set of ratios at
# the very end. Averaging per-microbatch F1 scores would instead weight a
# microbatch holding 6 scored positions the same as one holding 6000, which is
# the same pooling bug ``rare_class_t_bucket_sums`` exists to avoid. From these
# three rows both reported metrics follow directly:
#
#   accuracy  = true_positive.sum() / target.sum()
#   precision = true_positive / predicted     (per token id)
#   recall    = true_positive / target        (per token id)
#   macro F1  = unweighted mean of the per-token-id F1 over the token ids that
#               actually occur (see ``macro_f1_from_counts`` in train/loop.py)
#
# Note that micro-averaged F1 is deliberately NOT reported: with exactly one
# prediction per scored position it is algebraically identical to accuracy, so
# it would be a duplicate column. Macro F1 is the one that carries new
# information, because it gives a token appearing 5 times the same weight as one
# appearing 500,000 times -- which is precisely the signal needed to detect the
# model collapsing onto the handful of very common tokens.
TOKEN_COUNT_ROWS = ("true_positive", "predicted", "target")

# Perspective-split loss-breakdown names. Each example is built from one player's
# perspective (``DatasetExample.perspective_player``); "p1" means p1 is the
# viewer and p2 is the reconstructed enemy, and vice versa for "p2". Emitted in
# BOTH pipelines. Integer ids below are the representation carried on the batch
# (see data/collate.py) so the perspective survives ``.to(device)``.
PERSPECTIVE_NAMES = ("p1", "p2")
PERSPECTIVE_P1 = 1
PERSPECTIVE_P2 = 2

# ---------------------------------------------------------------------------
# Rare-class x t-bucket cross decomposition
# ---------------------------------------------------------------------------
# The three classes below are RARE in the canvas: a window carries exactly one
# win/loss token, at most one [END], and one [DELIMITER] per timestep, against a
# 4096-position canvas dominated by enemy reconstruction tokens. They therefore
# receive gradient on nearly every step, but only through a handful of positions
# out of thousands, so their signal is easy to lose in the aggregate loss.
#
# WHY cross them with the t-buckets rather than reading `per_class` and
# `t_bucket` side by side: those two are MARGINALS. `per_class["[END]"]` averages
# [END] over every corruption level at once, and `t_bucket["t_0_75_to_1_0"]`
# averages every class at that corruption level. Neither can answer the actual
# question -- does the win/loss token survive HIGH corruption, or only get
# memorized at low corruption? -- because a trend that runs along the corruption
# axis is exactly what marginalizing over it destroys. This decomposition keeps
# both axes, so a rare class that is learned at t < 0.25 and hopeless at
# t >= 0.75 is visible as such instead of averaging into one mediocre number.
#
# Keys are "<class>_<t-bucket>", e.g. "win_loss_t_eq_1", giving 3 x 4 = 12 cells.
# Class spellings match `_metric_class_name` in train/loop.py, so these names are
# identical in pre-training ("[END]") and debut ("end") taxonomies.
RARE_CLASS_IDS: dict[str, int] = {
    "win_loss": CLASS_WINLOSS,
    "end": CLASS_END,
    "delimiter": CLASS_DELIMITER,
}

RARE_CLASS_T_BUCKET_NAMES: tuple[str, ...] = tuple(
    f"{class_name}_{bucket_name}"
    for class_name in RARE_CLASS_IDS
    for bucket_name in T_BUCKET_NAMES
)


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
    # Rare-class x t-bucket cross decomposition (see RARE_CLASS_T_BUCKET_NAMES).
    # These two dicts BREAK the emptiness convention above on purpose: they carry
    # all 12 keys unconditionally, because "this bucket scored zero [END] tokens"
    # is itself the observation being recorded -- an absent key could not be
    # distinguished from a bucket that was never evaluated.
    #
    # They are also the only decomposition reported as SUM + COUNT rather than as
    # a finished mean. Two reasons: (1) the consumer needs the raw count anyway,
    # and (2) summing lets the training loop pool across microbatches by total
    # scored positions instead of averaging per-microbatch means, which is the
    # only correct pooling when a bucket holds 6 positions in one microbatch and
    # 0 in the next. Both are on-device tensors, left unsynchronized so the
    # training loop's lagged finalize (not the forward pass) pays the transfer.
    rare_class_t_bucket_sums: dict[str, torch.Tensor]
    rare_class_t_bucket_counts: dict[str, torch.Tensor]
    # ---- Token-level correctness and information-theory reporting ----------
    #
    # These four fields exist so the per-epoch and per-interval CSVs can report
    # something other than a loss value: how often the model's argmax token is
    # actually right, whether it is right across the whole vocabulary or only on
    # the common tokens, and what the model's uncertainty costs in bits.
    #
    # They are REPORTING ONLY. Nothing here is differentiable, nothing here
    # feeds the optimizer, and none of it changes ``loss``.
    #
    # ``unweighted_ce_sum`` / ``scored_token_count`` deliberately do NOT reuse
    # the ``loss`` field. ``loss`` is the class-WEIGHTED training objective
    # (config.loss.class_loss_weights currently boosts ``[END]`` by ~24.6x and
    # damps ``[PAD]`` to 0.1x), so exp(loss) is not a perplexity of anything and
    # is not comparable between two runs whose weights differ. These two raw
    # accumulators give the plain, unweighted mean cross entropy in nats over
    # every scored position, from which bits-per-token and perplexity follow:
    #
    #   mean_nats       = unweighted_ce_sum / scored_token_count
    #   bits_per_token  = mean_nats / ln(2)
    #   perplexity      = exp(mean_nats) == 2 ** bits_per_token
    #
    # Both are on-device, unsynchronized scalars, matching
    # ``rare_class_t_bucket_sums``: the caller's lagged finalize pays the
    # GPU->CPU transfer, not the forward pass.
    unweighted_ce_sum: torch.Tensor
    scored_token_count: torch.Tensor
    # Per-canvas-state token count matrices, keyed by CANVAS_STATE_NAMES and
    # shaped ``[3, vocab_size]`` with rows ordered by TOKEN_COUNT_ROWS. Absent
    # entirely when ``changed_positions`` was not supplied (no canvas-state split
    # is knowable then), and following the ``per_class`` emptiness convention: a
    # state with zero scored positions is omitted rather than present-with-zeros.
    #
    # WHY the accuracy/F1 reporting is split by canvas state instead of pooled
    # into one number: a scored position whose noised canvas token ALREADY equals
    # the target only asks the model to leave a correct token alone, and at low
    # corruption levels those positions dominate. A single pooled accuracy would
    # therefore mostly track the sampled corruption schedule rather than the
    # model. The "noised" half is the denoising work proper; the
    # "ground_truth_preserved" half measures whether the model knows to keep what
    # is already right. See CANVAS_STATE_NAMES for why the split keys on token
    # inequality rather than on the corruption coin-flip.
    token_class_counts: dict[str, torch.Tensor]


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
        # helper).
        self.class_id_to_name = active_class_id_to_name(config)
        self.debut_mode = config.data.debut_mode

        # The weight buffer is indexed by raw class-id, so it MUST be sized by
        # max(id) + 1, NOT len(map), so stable raw ids remain safe.
        buffer_size = max(self.class_id_to_name) + 1
        class_weights = torch.ones(buffer_size, dtype=torch.float32)
        weights = config.loss.class_loss_weights
        class_weights[CLASS_ENEMY_OBSERVED] = weights.enemy_observed_reconstruction
        class_weights[CLASS_ENEMY_FOGGED] = weights.enemy_fogged_reconstruction
        class_weights[CLASS_ENEMY_FUTURE] = weights.enemy_future_prediction
        class_weights[CLASS_DELIMITER] = weights.delimiter
        class_weights[CLASS_END] = weights.end
        class_weights[CLASS_PAD] = weights.pad
        class_weights[CLASS_WINLOSS] = weights.win_loss
        # Semantic [PAD] is still scored; its configured weight only changes its
        # contribution to the normalized objective. Batch-shape padding remains
        # excluded by the caller's scored mask.
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
        # Inactive positions may carry CLASS_CLAMPED=-1 for the fixed [BOS]
        # anchor. Index weights only for scored positions so that sentinel can
        # never alias the final class or index outside the taxonomy.
        weights = torch.zeros_like(ce)
        weights[active] = self.class_weights.to(ce.device)[class_labels[active]]
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
        # Rare-class x t-bucket cross decomposition, computed in the same block
        # because it reuses the very same per-example bucket row masks. See
        # RARE_CLASS_T_BUCKET_NAMES for why the marginals above cannot answer the
        # question this one answers.
        rare_class_t_bucket_sums: dict[str, torch.Tensor] = {}
        rare_class_t_bucket_counts: dict[str, torch.Tensor] = {}
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

            for class_name, class_id in RARE_CLASS_IDS.items():
                class_mask = active & (class_labels == class_id)
                for bucket_name in T_BUCKET_NAMES:
                    cell_mask = (
                        class_mask & bucket_row_masks[bucket_name].unsqueeze(1)
                    ).to(ce.dtype)
                    key = f"{class_name}_{bucket_name}"
                    # Multiply-and-sum rather than boolean-index-and-mean: this
                    # keeps all 12 cells sync-free (a boolean index has to read
                    # the mask's popcount back to the host to size its output,
                    # which would stall the forward pass 12 extra times per
                    # microbatch), and it yields the sum/count pair the caller
                    # pools with. An empty cell is a clean 0.0 sum over a 0
                    # count, never a NaN.
                    rare_class_t_bucket_sums[key] = (ce * cell_mask).sum()
                    rare_class_t_bucket_counts[key] = cell_mask.sum()

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
        # Declared out here (rather than inside the branch below) because the
        # token-level accuracy/F1 counts further down split on the SAME masks and
        # must not recompute them. Stays empty when no corruption information was
        # supplied, which suppresses both the canvas-state losses and the
        # canvas-state token counts together.
        state_masks: dict[str, torch.Tensor] = {}
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

        # ---- Token-level correctness and unweighted cross entropy ------------
        # Reporting only: nothing below is differentiable or feeds `loss`.
        # See LossOutput.token_class_counts / .unweighted_ce_sum for the full
        # rationale (why the split, why raw counts, why not exp(loss)).
        active_weight = active.to(ce.dtype)
        # Plain, UNWEIGHTED cross-entropy total in nats over every scored
        # position, plus how many positions that was. The caller divides one by
        # the other to get bits-per-token and perplexity. Kept separate from
        # `aggregate` above, which is class-weighted and so cannot be
        # exponentiated into a perplexity.
        unweighted_ce_sum = (ce * active_weight).sum()
        scored_token_count = active_weight.sum()

        token_class_counts: dict[str, torch.Tensor] = {}
        if state_masks:
            vocab_size = canvas_logits.shape[-1]
            predicted_tokens = canvas_logits.argmax(dim=-1)
            correct_positions = predicted_tokens == target_canvas
            # Flattened once and shared by every state below; `scatter_add_`
            # wants the index and the source to be 1-D and the same length.
            flat_predicted = predicted_tokens.reshape(-1)
            flat_target = target_canvas.reshape(-1)
            # Iterate the states the canvas_state block above found non-empty
            # rather than re-testing `state_mask.any()` here. Two reasons: each
            # `.any()` reads a value back to the host, and reusing that result
            # makes it structurally impossible for the loss split and the token
            # split to disagree about which states exist in a given batch. This
            # also carries the `per_class` emptiness convention across for free:
            # a state that scored nothing is absent, not a row of zeros.
            for name in canvas_state:
                state_mask = state_masks[name]
                # Multiply-and-scatter rather than boolean-index-and-count, for
                # the reason given on rare_class_t_bucket_sums: a boolean index
                # has to read the mask's popcount back to the host to size its
                # output, which would stall the forward pass. Masking to 0.0
                # instead lets every position take part in the scatter while
                # contributing nothing outside the state.
                in_state = state_mask.to(torch.float32).reshape(-1)
                in_state_and_correct = (
                    (state_mask & correct_positions).to(torch.float32).reshape(-1)
                )
                # Scatter in float32, then hand back int64 (below). float32 is
                # exact for integer counts under 2**24 -- ~16.7 million positions
                # in ONE microbatch, orders of magnitude above anything this
                # project batches -- while avoiding int64 atomics on CUDA. The
                # int64 result is what matters to the caller, whose epoch-level
                # totals genuinely can exceed that bound once pooled.
                counts = torch.zeros(
                    len(TOKEN_COUNT_ROWS), vocab_size, device=ce.device, dtype=torch.float32
                )
                counts[0].scatter_add_(0, flat_target, in_state_and_correct)
                counts[1].scatter_add_(0, flat_predicted, in_state)
                counts[2].scatter_add_(0, flat_target, in_state)
                # Exact integers throughout, so the cast loses nothing and gives
                # the caller a type it can pool over a whole epoch safely.
                token_class_counts[name] = counts.to(torch.int64)

        return LossOutput(
            loss=aggregate,
            per_class=per_class,
            future_distance=future_distance,
            future_distance_counts=future_distance_counts,
            t_bucket=t_bucket,
            perspective=perspective,
            canvas_state=canvas_state,
            rare_class_t_bucket_sums=rare_class_t_bucket_sums,
            rare_class_t_bucket_counts=rare_class_t_bucket_counts,
            unweighted_ce_sum=unweighted_ce_sum,
            scored_token_count=scored_token_count,
            token_class_counts=token_class_counts,
        )
