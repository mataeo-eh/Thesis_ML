"""Serialization from extractor parquet snapshots to token records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from thesis_ml.config import ProjectConfig
from thesis_ml.vocab.content_vocab import ContentVocabulary, normalize_content_name
from thesis_ml.vocab.special_tokens import DELIMITER_ID, DELIMITER_TOKEN

ENTITY_COLUMN_RE = re.compile(r"^(p[12])_(.+)_(\d{3})_(.+)$")
UPGRADE_COLUMNS = ("p1_upgrades", "p2_upgrades")


@dataclass(frozen=True)
class EntityColumnGroup:
    owner: str
    bot_name: str
    entity_type: str
    instance_id: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class TokenRecord:
    token_id: int
    token_name: str
    token_kind: str
    owner: str | None
    allegiance: str | None
    game_loop: int | None
    timestamp_seconds: float | None
    entity_type: str | None = None
    instance_id: str | None = None
    raw_position: Any | None = None
    raw_attributes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_entity_columns(columns: Iterable[str]) -> tuple[EntityColumnGroup, ...]:
    grouped: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for column in columns:
        match = ENTITY_COLUMN_RE.match(column)
        if match is None:
            continue
        owner, middle, instance_id, attribute = match.groups()
        bot_name, entity_type = _split_bot_and_entity(middle)
        entity_type = normalize_content_name(entity_type)
        key = (owner, bot_name, entity_type, instance_id)
        grouped.setdefault(key, {})[attribute] = column

    groups = [
        EntityColumnGroup(
            owner=owner,
            bot_name=bot_name,
            entity_type=entity_type,
            instance_id=instance_id,
            attributes=dict(sorted(attributes.items())),
        )
        for (owner, bot_name, entity_type, instance_id), attributes in grouped.items()
    ]
    return tuple(sorted(groups, key=lambda group: (group.owner, group.bot_name, group.entity_type, group.instance_id)))


def serialize_snapshot(
    snapshot: pd.Series,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    *,
    perspective_player: str | None = None,
    entity_groups: tuple[EntityColumnGroup, ...] | None = None,
) -> list[TokenRecord]:
    groups = entity_groups if entity_groups is not None else parse_entity_columns(snapshot.index)
    records: list[TokenRecord] = []
    for group in groups:
        raw_attributes = _non_null_attributes(snapshot, group)
        if not raw_attributes:
            continue
        records.append(_entity_record(snapshot, group, raw_attributes, vocabulary, perspective_player))

    records.extend(_upgrade_records(snapshot, vocabulary, perspective_player))
    records.sort(key=lambda record: _record_sort_key(record, vocabulary, config))
    records.append(_delimiter_record(snapshot))
    return records


def serialize_sequence(
    snapshots: Iterable[pd.Series],
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    *,
    perspective_player: str | None = None,
) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    for snapshot in snapshots:
        records.extend(
            serialize_snapshot(
                snapshot,
                config,
                vocabulary,
                perspective_player=perspective_player,
            )
        )
    return records


def deserialize_counts(records: Iterable[TokenRecord], vocabulary: ContentVocabulary) -> list[dict[str, int]]:
    timesteps: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for record in records:
        if record.token_id == DELIMITER_ID:
            timesteps.append(current)
            current = {}
            continue
        name = vocabulary.token_name_for(record.token_id)
        current[name] = current.get(name, 0) + 1
    if current:
        timesteps.append(current)
    return timesteps


def snapshot_content_counts(snapshot: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in parse_entity_columns(snapshot.index):
        if _non_null_attributes(snapshot, group):
            counts[group.entity_type] = counts.get(group.entity_type, 0) + 1
    for owner in ("p1", "p2"):
        for upgrade in parse_upgrades(snapshot.get(f"{owner}_upgrades")):
            counts[upgrade] = counts.get(upgrade, 0) + 1
    return counts


def parse_upgrades(value: Any) -> tuple[str, ...]:
    """Normalize one parquet upgrade cell into content-token names.

    Extractor generations have stored these cumulative values as Python-literal
    strings, Arrow/Pandas list-like values, and (in older data) mappings.  Keep
    the normalization in one public helper so analysis scripts and model
    serialization count the same upgrade tokens.
    """

    if value is None:
        return ()

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [text]
    elif isinstance(value, (list, tuple, set)):
        parsed = value
    elif hasattr(value, "tolist"):
        parsed = value.tolist()
    elif pd.isna(value):
        return ()

    if not isinstance(parsed, (dict, list, tuple, set)) and pd.isna(parsed):
        return ()
    if isinstance(parsed, dict):
        items = (name for name, enabled in parsed.items() if enabled)
    elif isinstance(parsed, (list, tuple, set)):
        items = parsed
    else:
        items = (parsed,)
    return tuple(normalize_content_name(str(item)) for item in items if str(item).strip())


def records_to_plain(records: Iterable[TokenRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def _split_bot_and_entity(middle: str) -> tuple[str, str]:
    if "_" not in middle:
        return "", middle
    bot_name, entity_type = middle.rsplit("_", 1)
    return bot_name, entity_type


def _non_null_attributes(snapshot: pd.Series, group: EntityColumnGroup) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for attribute, column in group.attributes.items():
        value = snapshot[column]
        if pd.isna(value):
            continue
        raw[attribute] = value
    return raw


def _entity_record(
    snapshot: pd.Series,
    group: EntityColumnGroup,
    raw_attributes: dict[str, Any],
    vocabulary: ContentVocabulary,
    perspective_player: str | None,
) -> TokenRecord:
    return TokenRecord(
        token_id=vocabulary.token_id_for(group.entity_type),
        token_name=group.entity_type,
        token_kind="entity",
        owner=group.owner,
        allegiance=_allegiance(group.owner, perspective_player),
        game_loop=_optional_int(snapshot.get("game_loop")),
        timestamp_seconds=_optional_float(snapshot.get("timestamp_seconds")),
        entity_type=group.entity_type,
        instance_id=group.instance_id,
        raw_position=raw_attributes.get("pos_(X,Y,Z)"),
        raw_attributes=raw_attributes,
    )


def _upgrade_records(
    snapshot: pd.Series,
    vocabulary: ContentVocabulary,
    perspective_player: str | None,
) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    for owner in ("p1", "p2"):
        for upgrade in parse_upgrades(snapshot.get(f"{owner}_upgrades")):
            records.append(
                TokenRecord(
                    token_id=vocabulary.token_id_for(upgrade),
                    token_name=upgrade,
                    token_kind="upgrade",
                    owner=owner,
                    allegiance=_allegiance(owner, perspective_player),
                    game_loop=_optional_int(snapshot.get("game_loop")),
                    timestamp_seconds=_optional_float(snapshot.get("timestamp_seconds")),
                    entity_type=upgrade,
                    raw_attributes={"upgrade": upgrade},
                )
            )
    return records


def _delimiter_record(snapshot: pd.Series) -> TokenRecord:
    return TokenRecord(
        token_id=DELIMITER_ID,
        token_name=DELIMITER_TOKEN,
        token_kind="delimiter",
        owner=None,
        allegiance=None,
        game_loop=_optional_int(snapshot.get("game_loop")),
        timestamp_seconds=_optional_float(snapshot.get("timestamp_seconds")),
    )


def _record_sort_key(record: TokenRecord, vocabulary: ContentVocabulary, config: ProjectConfig) -> tuple[Any, ...]:
    if record.token_kind == "delimiter":
        return (2, 0, 0, "", "")
    if record.token_kind == "upgrade":
        return (1, vocabulary.source_id_for(record.token_name), 0, record.owner or "", "")
    if config.data.within_type_tiebreak not in {"unit_id", "instance_id"}:
        raise ValueError(f"unsupported within_type_tiebreak: {config.data.within_type_tiebreak}")
    return (
        0,
        vocabulary.source_id_for(record.token_name),
        int(record.instance_id or 0),
        record.owner or "",
        record.entity_type or "",
    )


def _allegiance(owner: str, perspective_player: str | None) -> str | None:
    if perspective_player is None:
        return None
    if perspective_player not in {"p1", "p2"}:
        raise ValueError("perspective_player must be 'p1', 'p2', or None")
    return "self" if owner == perspective_player else "enemy"


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
