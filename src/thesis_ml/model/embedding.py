"""Token embeddings plus input-only contextual encodings."""

from __future__ import annotations

import ast
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

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.map_fourier = FourierFeatures(input_dim=2)
        self.clock_fourier = FourierFeatures(input_dim=1)
        self.map_projection = nn.Linear(self.map_fourier.output_dim, d_model, bias=False)
        self.clock_projection = nn.Linear(self.clock_fourier.output_dim, d_model, bias=False)
        self.stat_projection = nn.Linear(len(STAT_KEYS), d_model, bias=False)
        self.team_embedding = nn.Embedding(3, d_model, padding_idx=0)

    def forward(
        self,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        *,
        input_records: Sequence[Sequence[TokenRecord]] | None = None,
    ) -> torch.Tensor:
        input_embeddings = self.embed_input(input_token_ids, input_records=input_records)
        canvas_embeddings = self.embed_canvas(canvas_token_ids)
        return torch.cat([input_embeddings, canvas_embeddings], dim=1)

    def embed_input(
        self,
        input_token_ids: torch.Tensor,
        *,
        input_records: Sequence[Sequence[TokenRecord]] | None = None,
    ) -> torch.Tensor:
        embeddings = self.token_embedding(input_token_ids)
        if input_records is None:
            return embeddings

        device = input_token_ids.device
        map_values, clock_values, stat_values, team_ids = _records_to_tensors(
            input_records,
            input_token_ids.shape,
            device=device,
            dtype=embeddings.dtype,
        )
        return (
            embeddings
            + self.map_projection(self.map_fourier(map_values))
            + self.clock_projection(self.clock_fourier(clock_values))
            + self.stat_projection(stat_values)
            + self.team_embedding(team_ids)
        )

    def embed_canvas(self, canvas_token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(canvas_token_ids)


def _records_to_tensors(
    records: Sequence[Sequence[TokenRecord]],
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len = shape
    map_values = torch.zeros(batch, seq_len, 2, device=device, dtype=dtype)
    clock_values = torch.zeros(batch, seq_len, 1, device=device, dtype=dtype)
    stat_values = torch.zeros(batch, seq_len, len(STAT_KEYS), device=device, dtype=dtype)
    team_ids = torch.zeros(batch, seq_len, device=device, dtype=torch.long)

    for batch_index, row_records in enumerate(records):
        for token_index, record in enumerate(row_records[:seq_len]):
            if record.allegiance == "self":
                team_ids[batch_index, token_index] = 1
            elif record.allegiance == "enemy":
                team_ids[batch_index, token_index] = 2
            if record.timestamp_seconds is not None:
                clock_values[batch_index, token_index, 0] = float(record.timestamp_seconds)
            position = _parse_position(record.raw_position)
            if position is not None:
                map_values[batch_index, token_index] = torch.tensor(position[:2], device=device, dtype=dtype)
            raw = record.raw_attributes or {}
            for stat_index, key in enumerate(STAT_KEYS):
                stat_values[batch_index, token_index, stat_index] = _numeric_feature(raw.get(key))

    return map_values, clock_values, stat_values, team_ids


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
