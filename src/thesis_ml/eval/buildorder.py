"""Deterministic build-order extraction at model resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from thesis_ml.config import ProjectConfig
from thesis_ml.inference.timing import TimedTimestep
from thesis_ml.serialize import serialize_snapshot
from thesis_ml.vocab.content_vocab import ContentVocabulary


@dataclass(frozen=True, order=True)
class BuildOrderEvent:
    entity_type: str
    bucket: int


def extract_build_order(
    timesteps: Sequence[Mapping[str, int] | TimedTimestep],
    *,
    drop_final_timestep: bool = False,
) -> tuple[BuildOrderEvent, ...]:
    """Extract one event for each newly appearing count in decoded timesteps."""

    usable = timesteps[:-1] if drop_final_timestep and timesteps else timesteps
    previous_counts: dict[str, int] = {}
    events: list[BuildOrderEvent] = []
    for index, timestep in enumerate(usable):
        counts = timestep.counts if isinstance(timestep, TimedTimestep) else timestep
        bucket = timestep.timestep_index if isinstance(timestep, TimedTimestep) else index
        for entity_type in sorted(counts):
            count = max(0, int(counts[entity_type]))
            new_count = count - previous_counts.get(entity_type, 0)
            if new_count <= 0:
                continue
            events.extend(BuildOrderEvent(entity_type=entity_type, bucket=int(bucket)) for _ in range(new_count))
            previous_counts[entity_type] = count
    return tuple(sorted(events, key=lambda event: (event.bucket, event.entity_type)))


def extract_build_order_from_parquet(
    parquet_path: str | Path,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    *,
    perspective_player: str,
    start: int = 0,
    drop_final_timestep: bool = False,
) -> tuple[BuildOrderEvent, ...]:
    frame = pd.read_parquet(parquet_path).sort_values("game_loop").reset_index(drop=True)
    return extract_build_order_from_frame(
        frame,
        config,
        vocabulary,
        perspective_player=perspective_player,
        start=start,
        drop_final_timestep=drop_final_timestep,
    )


def extract_build_order_from_frame(
    frame: pd.DataFrame,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    *,
    perspective_player: str,
    start: int = 0,
    drop_final_timestep: bool = False,
) -> tuple[BuildOrderEvent, ...]:
    """Extract ground-truth build order from parsed parquet rows.

    Entities contribute an event when their instance ID first appears. Upgrades
    contribute when the upgrade token first appears in the owning player's
    cumulative upgrade list.
    """

    enemy_player = _enemy_player(perspective_player)
    rows = frame.iloc[start:]
    if drop_final_timestep and len(rows) > 0:
        rows = rows.iloc[:-1]

    seen_entities: set[tuple[str, str]] = set()
    seen_upgrades: set[str] = set()
    events: list[BuildOrderEvent] = []
    for bucket, (_, row) in enumerate(rows.iterrows()):
        records = serialize_snapshot(row, config, vocabulary, perspective_player=perspective_player)
        for record in records:
            if record.owner != enemy_player:
                continue
            if record.token_kind == "entity":
                key = (record.token_name, record.instance_id or "")
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                events.append(BuildOrderEvent(entity_type=record.token_name, bucket=bucket))
            elif record.token_kind == "upgrade":
                if record.token_name in seen_upgrades:
                    continue
                seen_upgrades.add(record.token_name)
                events.append(BuildOrderEvent(entity_type=record.token_name, bucket=bucket))
    return tuple(sorted(events, key=lambda event: (event.bucket, event.entity_type)))


def _enemy_player(perspective_player: str) -> str:
    if perspective_player == "p1":
        return "p2"
    if perspective_player == "p2":
        return "p1"
    raise ValueError("perspective_player must be 'p1' or 'p2'")
