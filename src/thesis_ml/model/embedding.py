"""Token embeddings plus input-only contextual encodings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import torch
from torch import nn

from thesis_ml.serialize import TokenRecord

STAT_KEYS = (
    "health",
    "energy",
    "shields",
    "facing",
    "radius",
    "build_progress",
    "weapon_cooldown",
    "attack_upgrade_level",
    "armor_upgrade_level",
    "shield_upgrade_level",
    "cargo_space_taken",
    "cargo_space_max",
    "order_count",
    "is_flying",
    "is_burrowed",
    "is_hallucination",
    "is_active",
    "is_powered",
)


@dataclass(frozen=True)
class InputFeatures:
    """Pre-parsed, batched contextual fields for the input region.

    These are the model-approved numeric values (map position, unit stats, team)
    extracted once per batch in the DataLoader workers (see
    ``thesis_ml.data.collate``), instead of re-parsing TokenRecord objects in a
    Python loop on every forward pass. Shapes are ``[batch, seq_len, ...]``:
      - map_values:   [B, L, 2]   exact (X, Y) map coordinate
      - stat_values:  [B, L, S]   per-unit stats in STAT_KEYS order
      - team_ids:     [B, L]      0 = pad, 1 = self, 2 = enemy

    Absolute game time is intentionally absent. ``TokenRecord`` may retain a
    timestamp for dataset ordering or output-side evaluation, but this type is
    the boundary that prevents that metadata from entering the model.
    """

    map_values: torch.Tensor
    stat_values: torch.Tensor
    team_ids: torch.Tensor


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
    map_values, stat_values, team_ids = _records_to_tensors(
        records,
        torch.Size((batch, seq_len)),
        device=torch.device("cpu"),
        dtype=torch.float32,
        left_pad=left_pad,
    )
    return InputFeatures(
        map_values=map_values,
        stat_values=stat_values,
        team_ids=team_ids,
    )


class FourierFeatures(nn.Module):
    """Extrapolation-friendly continuous features."""

    def __init__(self, input_dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = input_dim * num_frequencies * 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        angles = values.unsqueeze(-1) * self.frequencies * math.pi
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoded.flatten(start_dim=-2)


class InputContextEmbedding(nn.Module):
    """Shared token embedding with additive contextual fields for input tokens only."""

    def __init__(self, vocab_size: int, d_model: int, *, self_conditioning: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.self_conditioning = self_conditioning
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.map_fourier = FourierFeatures(input_dim=2)
        self.map_projection = nn.Linear(self.map_fourier.output_dim, d_model, bias=False)
        self.stat_projection = nn.Linear(len(STAT_KEYS), d_model, bias=False)
        self.team_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.self_cond_projection = nn.Linear(vocab_size, d_model, bias=False) if self_conditioning else None

    def forward(
        self,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        *,
        input_features: InputFeatures | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_embeddings = self.embed_input(
            input_token_ids,
            input_features=input_features,
        )
        canvas_embeddings = self.embed_canvas(canvas_token_ids, canvas_self_conditioning=canvas_self_conditioning)
        return torch.cat([input_embeddings, canvas_embeddings], dim=1)

    def embed_input(
        self,
        input_token_ids: torch.Tensor,
        *,
        input_features: InputFeatures | None = None,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(input_token_ids)

        if input_features is None:
            return embeddings

        device = embeddings.device
        map_values = input_features.map_values.to(device=device, dtype=embeddings.dtype)
        stat_values = input_features.stat_values.to(device=device, dtype=embeddings.dtype)
        team_ids = input_features.team_ids.to(device=device, dtype=torch.long)

        return (
            embeddings
            + self.map_projection(self.map_fourier(map_values))
            + self.stat_projection(stat_values)
            + self.team_embedding(team_ids)
        )

    def embed_canvas(
        self,
        canvas_token_ids: torch.Tensor,
        *,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(canvas_token_ids)
        if not self.self_conditioning or canvas_self_conditioning is None:
            return embeddings
        expected_shape = (*canvas_token_ids.shape, self.vocab_size)
        if tuple(canvas_self_conditioning.shape) != expected_shape:
            raise ValueError(
                "canvas_self_conditioning must have shape "
                f"{expected_shape}, got {tuple(canvas_self_conditioning.shape)}"
            )
        projection = self.self_cond_projection(canvas_self_conditioning.to(dtype=embeddings.dtype))
        return embeddings + projection


def _records_to_tensors(
    records: Sequence[Sequence[TokenRecord]],
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    left_pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len = shape
    map_values = torch.zeros(batch, seq_len, 2, device=device, dtype=dtype)
    stat_values = torch.zeros(batch, seq_len, len(STAT_KEYS), device=device, dtype=dtype)
    team_ids = torch.zeros(batch, seq_len, device=device, dtype=torch.long)

    for batch_index, row_records in enumerate(records):
        offset = max(0, seq_len - len(row_records)) if left_pad else 0
        for token_index, record in enumerate(row_records[:seq_len]):
            token_index += offset
            if record.allegiance == "self":
                team_ids[batch_index, token_index] = 1
            elif record.allegiance == "enemy":
                team_ids[batch_index, token_index] = 2
            position = _parse_position(record.raw_position)
            if position is not None:
                map_values[batch_index, token_index] = torch.tensor(position[:2], device=device, dtype=dtype)
            raw = record.raw_attributes or {}
            for stat_index, key in enumerate(STAT_KEYS):
                stat_values[batch_index, token_index, stat_index] = _numeric_feature(raw.get(key))

    return map_values, stat_values, team_ids


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
