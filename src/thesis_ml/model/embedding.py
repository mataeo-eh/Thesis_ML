"""Token embeddings plus an input-only joint static-feature residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from thesis_ml.data.features import (
    BUFF_VALIDITY_INDEX,
    BUFF_VALUE_OFFSET,
    CATEGORICAL_FEATURE_WIDTH,
    CLOAK_STATE_COUNT,
    CONTINUOUS_FEATURE_NAMES,
    STAT_KEYS,
    encode_entity_features,
    parse_numeric_feature,
    parse_position,
)
from thesis_ml.data.feature_stats import (
    FeatureStatistics,
)
from thesis_ml.model.backbone import GeGLU, RMSNorm
from thesis_ml.serialize import TokenRecord

# Row indexes into the optional segment-embedding table. The ordering is part of
# the public contract of this module (downstream tests assert it), so it is named
# here rather than written as bare 0/1 literals at the two use sites.
INPUT_SEGMENT_INDEX = 0
CANVAS_SEGMENT_INDEX = 1
# Number of distinct regions the model sees: the input region and the canvas.
SEGMENT_COUNT = 2


@dataclass(frozen=True)
class InputFeatures:
    """Pre-parsed, batched contextual fields for the input region.

    These are the model-approved numeric values (map position, unit stats, team)
    extracted once per batch in the DataLoader workers (see
    ``thesis_ml.data.collate``), instead of re-parsing TokenRecord objects in a
    Python loop on every forward pass. Shapes are ``[batch, seq_len, ...]``:
      - continuous_values: [B, L, F] approved scalar features
      - continuous_validity: [B, L, F] distinguishes valid zero from missing
      - categorical_values: [B, L, C] cloak one-hot plus sparse buff multi-hot
      - allegiance_values: [B, L, 1] self +1, enemy -1, structural/pad 0
      - feature_mask: [B, L] true only for self/enemy content records

    Absolute game time is intentionally absent. ``TokenRecord`` may retain a
    timestamp for dataset ordering or output-side evaluation, but this type is
    the boundary that prevents that metadata from entering the model.
    """

    continuous_values: torch.Tensor
    continuous_validity: torch.Tensor
    categorical_values: torch.Tensor
    allegiance_values: torch.Tensor
    feature_mask: torch.Tensor


def build_input_features(
    records: Sequence[Sequence[TokenRecord]],
    seq_len: int,
    *,
    left_pad: bool = False,
) -> InputFeatures:
    """Parse a batch of input-token record rows into batched feature tensors.

    Runs in the DataLoader worker (via collate) so this CPU-bound parsing is
    parallelized and happens once per batch per epoch rather than every step.
    Builds CPU float32 tensors; the model moves/casts them at use time.

    Parameters:
        records: one list of TokenRecord per batch row (the input region).
        seq_len: padded sequence length the batch was collated to.
    Returns:
        InputFeatures with [batch, seq_len, ...] tensors.
    Calls: _records_to_tensors, the sole allowlist from records to model inputs.
    """

    batch = len(records)
    (
        continuous_values,
        continuous_validity,
        categorical_values,
        allegiance_values,
        feature_mask,
    ) = _records_to_tensors(
        records,
        torch.Size((batch, seq_len)),
        device=torch.device("cpu"),
        dtype=torch.float32,
        left_pad=left_pad,
    )
    return InputFeatures(
        continuous_values=continuous_values,
        continuous_validity=continuous_validity,
        categorical_values=categorical_values,
        allegiance_values=allegiance_values,
        feature_mask=feature_mask,
    )


class InputContextEmbedding(nn.Module):
    """Shared type embedding plus the exact learned joint input residual."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        feature_statistics: FeatureStatistics,
        self_conditioning: bool = True,
        segment_embeddings: bool = False,
    ) -> None:
        """Build the shared token table plus the optional per-region extras.

        Parameters:
            vocab_size: number of token ids the shared type embedding covers.
            d_model: model width; every embedding this module emits has it.
            feature_statistics: train-split standardization statistics for the
                input-only joint static-feature branch. Its means/stds are held
                as buffers so they travel with the checkpoint.
            self_conditioning: when true, build the canvas self-conditioning
                branch (pre-norm, GeGLU, post-norm). When false the three
                submodules are left as plain ``None`` attributes, which keeps
                them out of the ``state_dict`` entirely.
            segment_embeddings: when true, build a learned two-row table whose
                row 0 is added to every input-region embedding and row 1 to
                every canvas-region embedding. Defaults false, which is the
                baseline: the table is not created, so no new ``state_dict``
                keys appear in the all-off architecture-v2 baseline.
        Returns:
            None; this is a constructor.
        Calls: nn.Embedding/nn.Linear/nn.Sequential construction plus RMSNorm
            and GeGLU from thesis_ml.model.backbone. Zero-initialization of the
            segment table is delegated to self.reset_segment_embeddings.
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.self_conditioning = self_conditioning
        self.segment_embeddings = segment_embeddings
        self.feature_statistics_identity = feature_statistics.identity
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "feature_means",
            torch.tensor(feature_statistics.means, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_stds",
            torch.tensor(feature_statistics.stds, dtype=torch.float32),
        )
        feature_width = (
            2 * len(CONTINUOUS_FEATURE_NAMES) + CATEGORICAL_FEATURE_WIDTH + 1
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_width, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.joint_mixer = nn.Sequential(
            nn.Linear(d_model + 32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if self_conditioning:
            self.self_cond_norm = RMSNorm(d_model)
            self.self_cond_ffn = GeGLU(d_model, d_model)
            self.self_cond_post_norm = RMSNorm(d_model, scale=False)
        else:
            self.self_cond_norm = None
            self.self_cond_ffn = None
            self.self_cond_post_norm = None
        if segment_embeddings:
            self.segment_embedding = nn.Embedding(SEGMENT_COUNT, d_model)
            self.reset_segment_embeddings()
        else:
            # Assigning a plain ``None`` (rather than an nn.Embedding that is
            # then ignored) is what keeps the toggle-off state_dict key-for-key
            # identical to its matching toggle-off v2 arm, so those checkpoints
            # load under strict load_state_dict. nn.Module only registers a
            # submodule when the assigned value is an nn.Module, so this adds no keys.
            # Same pattern as the self-conditioning branch directly above.
            self.segment_embedding = None

    def reset_joint_output(self) -> None:
        """Make the initialized joint branch exactly zero, preserving E."""

        output = self.joint_mixer[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def reset_segment_embeddings(self) -> None:
        """Make the segment table exactly zero; no-op when the toggle is off.

        Zero is the deliberate starting point: at zero the segment term adds
        nothing, so day-0 behavior with the toggle on is identical to the
        baseline and any later divergence is attributable to LEARNING rather
        than to initialization noise.

        This exists as a separate method for the same reason
        ``reset_joint_output`` does: ``SC2StrategyDiffusionModel._init_weights``
        sweeps every ``nn.Embedding`` in the model and re-initializes it at
        ``std=0.02``, which would clobber the zeros set in ``__init__``. The
        model calls this again after that sweep. It is safe to call
        unconditionally because the table does not exist when the toggle is off.

        Parameters:
            None.
        Returns:
            None; mutates self.segment_embedding.weight in place.
        Calls: nn.init.zeros_.
        """

        if self.segment_embedding is None:
            return
        nn.init.zeros_(self.segment_embedding.weight)

    def _add_segment_embedding(
        self,
        embeddings: torch.Tensor,
        segment_index: int,
    ) -> torch.Tensor:
        """Add one segment vector to every position of a region's embeddings.

        Applied to the FINAL per-region embedding — after the joint feature
        residual in ``embed_input`` and after the self-conditioning post-norm in
        ``embed_canvas``. That placement is load-bearing: adding the signal
        earlier would let a downstream normalization wash it out, and washing it
        out is exactly the failure this ablation is meant to rule out.

        The vector is added to EVERY position in the region, padded slots
        included. Padded positions are excluded from attention as keys and are
        never scored by the loss, so their embedding value cannot affect either
        the forward result or the gradient; skipping them would cost a mask
        multiply and buy nothing. Uniform application is also the point of the
        toggle — the signal must mean "this position is canvas" at every index,
        not only at content indexes.

        Parameters:
            embeddings: [batch, seq_len, d_model] final embeddings of one region.
            segment_index: INPUT_SEGMENT_INDEX or CANVAS_SEGMENT_INDEX.
        Returns:
            [batch, seq_len, d_model]; the input tensor unchanged (same object)
            when the toggle is off, otherwise a new tensor with the segment
            vector broadcast-added over batch and sequence.
        Calls: nothing beyond torch primitives. Called by embed_input and
            embed_canvas.
        """

        if self.segment_embedding is None:
            return embeddings
        # Indexing the weight table directly is exactly an nn.Embedding lookup
        # for a constant id, and keeps the gradient path, but avoids allocating
        # a [batch, seq_len] index tensor on every forward pass just to look up
        # the same row repeatedly. The result is [d_model] and broadcasts over
        # the batch and sequence axes.
        segment_vector = self.segment_embedding.weight[segment_index]
        return embeddings + segment_vector.to(
            device=embeddings.device, dtype=embeddings.dtype
        )

    def forward(
        self,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        *,
        input_features: InputFeatures | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        canvas_embeddings = self.embed_canvas(canvas_token_ids, canvas_self_conditioning=canvas_self_conditioning)
        if input_token_ids.shape[1] == 0:
            return canvas_embeddings
        input_embeddings = self.embed_input(
            input_token_ids,
            input_features=input_features,
        )
        return torch.cat([input_embeddings, canvas_embeddings], dim=1)

    def embed_input(
        self,
        input_token_ids: torch.Tensor,
        *,
        input_features: InputFeatures | None = None,
    ) -> torch.Tensor:
        type_embeddings = self.token_embedding(input_token_ids)

        if input_features is None:
            raise ValueError("input_features are required for every non-empty input region")

        device = type_embeddings.device
        continuous = input_features.continuous_values.to(
            device=device, dtype=type_embeddings.dtype
        )
        continuous_validity = input_features.continuous_validity.to(
            device=device, dtype=torch.bool
        )
        categorical = input_features.categorical_values.to(
            device=device, dtype=type_embeddings.dtype
        )
        allegiance = input_features.allegiance_values.to(
            device=device, dtype=type_embeddings.dtype
        )
        feature_mask = input_features.feature_mask.to(device=device, dtype=torch.bool)
        if continuous.shape[-1] != len(CONTINUOUS_FEATURE_NAMES):
            raise ValueError(
                f"continuous input feature width must be {len(CONTINUOUS_FEATURE_NAMES)}"
            )
        if continuous_validity.shape != continuous.shape:
            raise ValueError("continuous feature validity must match continuous values")
        if categorical.shape[-1] != CATEGORICAL_FEATURE_WIDTH:
            raise ValueError(
                f"categorical input feature width must be {CATEGORICAL_FEATURE_WIDTH}"
            )
        standardized = (
            continuous - self.feature_means.to(dtype=type_embeddings.dtype)
        ) / self.feature_stds.to(dtype=type_embeddings.dtype)
        standardized = torch.where(
            continuous_validity,
            standardized,
            torch.zeros_like(standardized),
        )
        branch_input = torch.cat(
            [
                standardized,
                continuous_validity.to(dtype=type_embeddings.dtype),
                categorical,
                allegiance,
            ],
            dim=-1,
        )
        hidden_features = self.feature_mlp(branch_input)
        residual = self.joint_mixer(torch.cat([type_embeddings, hidden_features], dim=-1))
        residual = residual * feature_mask.unsqueeze(-1).to(dtype=residual.dtype)
        # Segment row 0 marks the input region. Added last, on top of the joint
        # residual, so no later transform inside this module can renormalize it
        # away. A no-op when the segment_embeddings toggle is off.
        return self._add_segment_embedding(
            type_embeddings + residual, INPUT_SEGMENT_INDEX
        )

    def embed_canvas(
        self,
        canvas_token_ids: torch.Tensor,
        *,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(canvas_token_ids)
        if not self.self_conditioning:
            # Segment row 1 marks the canvas region. Both of this method's exit
            # paths must apply it, or the signal would silently disappear
            # whenever self-conditioning happened to be disabled.
            return self._add_segment_embedding(embeddings, CANVAS_SEGMENT_INDEX)
        expected_shape = (*canvas_token_ids.shape, embeddings.shape[-1])
        if canvas_self_conditioning is None:
            canvas_self_conditioning = torch.zeros_like(embeddings)
        if tuple(canvas_self_conditioning.shape) != expected_shape:
            raise ValueError(
                "canvas_self_conditioning must have shape "
                f"{expected_shape}, got {tuple(canvas_self_conditioning.shape)}"
            )
        signal = canvas_self_conditioning.to(device=embeddings.device, dtype=embeddings.dtype)
        residual = self.self_cond_ffn(self.self_cond_norm(signal))
        # Segment row 1 is added AFTER the post-norm, deliberately: applying it
        # before would let self_cond_post_norm rescale the region signal away.
        return self._add_segment_embedding(
            self.self_cond_post_norm(embeddings + residual), CANVAS_SEGMENT_INDEX
        )


def _records_to_tensors(
    records: Sequence[Sequence[TokenRecord]],
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    left_pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len = shape
    continuous_values = torch.zeros(
        batch, seq_len, len(CONTINUOUS_FEATURE_NAMES), device=device, dtype=dtype
    )
    continuous_validity = torch.zeros(
        batch, seq_len, len(CONTINUOUS_FEATURE_NAMES), device=device, dtype=torch.bool
    )
    categorical_values = torch.zeros(
        batch, seq_len, CATEGORICAL_FEATURE_WIDTH, device=device, dtype=dtype
    )
    allegiance_values = torch.zeros(batch, seq_len, 1, device=device, dtype=dtype)
    feature_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.bool)

    for batch_index, row_records in enumerate(records):
        offset = max(0, seq_len - len(row_records)) if left_pad else 0
        for token_index, record in enumerate(row_records[:seq_len]):
            token_index += offset
            if record.allegiance == "self":
                allegiance_values[batch_index, token_index, 0] = 1.0
                feature_mask[batch_index, token_index] = True
            elif record.allegiance == "enemy":
                allegiance_values[batch_index, token_index, 0] = -1.0
                feature_mask[batch_index, token_index] = True
            raw = record.raw_attributes or {}
            encoded = encode_entity_features(record.raw_position, raw)
            continuous_values[batch_index, token_index] = torch.tensor(
                encoded.continuous_values, device=device, dtype=dtype
            )
            continuous_validity[batch_index, token_index] = torch.tensor(
                encoded.continuous_validity, device=device, dtype=torch.bool
            )
            if encoded.cloak_state is not None:
                categorical_values[
                    batch_index, token_index, encoded.cloak_state
                ] = 1.0
            if encoded.buff_ids is not None:
                categorical_values[
                    batch_index, token_index, BUFF_VALIDITY_INDEX
                ] = 1.0
                if encoded.buff_ids:
                    buff_indexes = torch.tensor(
                        [BUFF_VALUE_OFFSET + buff_id for buff_id in encoded.buff_ids],
                        device=device,
                        dtype=torch.long,
                    )
                    categorical_values[batch_index, token_index, buff_indexes] = 1.0

    return (
        continuous_values,
        continuous_validity,
        categorical_values,
        allegiance_values,
        feature_mask,
    )


def _parse_position(value: Any) -> tuple[float, float, float] | None:
    return parse_position(value)


def _numeric_feature(value: Any) -> float:
    return parse_numeric_feature(value)[0]
