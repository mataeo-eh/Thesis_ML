"""Token embeddings plus an input-only joint static-feature residual."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any, Sequence

import torch
from torch import nn

from thesis_ml.data.feature_stats import (
    CONTINUOUS_FEATURE_NAMES,
    STAT_KEYS,
    FeatureStatistics,
)
from thesis_ml.model.backbone import GeGLU, RMSNorm
from thesis_ml.serialize import TokenRecord


@dataclass(frozen=True)
class InputFeatures:
    """Pre-parsed, batched contextual fields for the input region.

    These are the model-approved numeric values (map position, unit stats, team)
    extracted once per batch in the DataLoader workers (see
    ``thesis_ml.data.collate``), instead of re-parsing TokenRecord objects in a
    Python loop on every forward pass. Shapes are ``[batch, seq_len, ...]``:
      - continuous_values: [B, L, F] map X/Y followed by STAT_KEYS
      - allegiance_values: [B, L, 1] self +1, enemy -1, structural/pad 0
      - feature_mask: [B, L] true only for self/enemy content records

    Absolute game time is intentionally absent. ``TokenRecord`` may retain a
    timestamp for dataset ordering or output-side evaluation, but this type is
    the boundary that prevents that metadata from entering the model.
    """

    continuous_values: torch.Tensor
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
    continuous_values, allegiance_values, feature_mask = _records_to_tensors(
        records,
        torch.Size((batch, seq_len)),
        device=torch.device("cpu"),
        dtype=torch.float32,
        left_pad=left_pad,
    )
    return InputFeatures(
        continuous_values=continuous_values,
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
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.self_conditioning = self_conditioning
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
        feature_width = len(CONTINUOUS_FEATURE_NAMES) + 1
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

    def reset_joint_output(self) -> None:
        """Make the initialized joint branch exactly zero, preserving E."""

        output = self.joint_mixer[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

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
        allegiance = input_features.allegiance_values.to(
            device=device, dtype=type_embeddings.dtype
        )
        feature_mask = input_features.feature_mask.to(device=device, dtype=torch.bool)
        if continuous.shape[-1] != len(CONTINUOUS_FEATURE_NAMES):
            raise ValueError(
                f"continuous input feature width must be {len(CONTINUOUS_FEATURE_NAMES)}"
            )
        standardized = (
            continuous - self.feature_means.to(dtype=type_embeddings.dtype)
        ) / self.feature_stds.to(dtype=type_embeddings.dtype)
        branch_input = torch.cat([standardized, allegiance], dim=-1)
        hidden_features = self.feature_mlp(branch_input)
        residual = self.joint_mixer(torch.cat([type_embeddings, hidden_features], dim=-1))
        residual = residual * feature_mask.unsqueeze(-1).to(dtype=residual.dtype)
        return type_embeddings + residual

    def embed_canvas(
        self,
        canvas_token_ids: torch.Tensor,
        *,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(canvas_token_ids)
        if not self.self_conditioning:
            return embeddings
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
        return self.self_cond_post_norm(embeddings + residual)


def _records_to_tensors(
    records: Sequence[Sequence[TokenRecord]],
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    left_pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len = shape
    continuous_values = torch.zeros(
        batch, seq_len, len(CONTINUOUS_FEATURE_NAMES), device=device, dtype=dtype
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
            position = _parse_position(record.raw_position)
            if position is not None:
                continuous_values[batch_index, token_index, :2] = torch.tensor(
                    position[:2], device=device, dtype=dtype
                )
            raw = record.raw_attributes or {}
            for stat_index, key in enumerate(STAT_KEYS):
                continuous_values[batch_index, token_index, 2 + stat_index] = _numeric_feature(
                    raw.get(key)
                )

    return continuous_values, allegiance_values, feature_mask


def _parse_position(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("("):
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, tuple) and len(parsed) >= 3:
            return float(parsed[0]), float(parsed[1]), float(parsed[2])
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    return None


def _numeric_feature(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    if "/" in text:
        numerator = text.split("/", 1)[0]
        return _numeric_feature(numerator)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return 0.0
    return float(match.group(0))
