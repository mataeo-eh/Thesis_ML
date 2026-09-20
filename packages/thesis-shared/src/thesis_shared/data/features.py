"""Strict, shared encoding of parquet entity attributes into model features."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math
import re
from typing import Any, Mapping


POSITION_KEY = "pos_(X,Y,Z)"

# Raw scalar fields approved for the learned per-token feature branch. Facing
# is handled separately because its lossless model representation is sin/cos.
STAT_KEYS = (
    "health",
    "energy",
    "shields",
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
    "ideal_harvesters",
    "buff_duration_remain",
    "buff_duration_max",
    "detect_range",
)

CONTINUOUS_FEATURE_NAMES = (
    "map_x",
    "map_y",
    "health",
    "energy",
    "shields",
    "facing_sin",
    "facing_cos",
    *STAT_KEYS[3:],
)

CLOAK_STATE_NAMES = (
    "CloakedUnknown",
    "Cloaked",
    "CloakedDetected",
    "NotCloaked",
    "CloakedAllied",
)
CLOAK_STATE_COUNT = len(CLOAK_STATE_NAMES)

# PySC2 4.0.0's convenience Buffs enum stops at 289, but the extractor records
# raw SC2 protocol IDs and the full 943-replay corpus contains IDs through 302.
# Keep the observed raw range lossless and fail loudly above it so a future game
# or corpus expansion cannot silently discard an unrepresented category.
BUFF_ID_MAX = 302
BUFF_ID_SPACE = BUFF_ID_MAX + 1
BUFF_VALIDITY_INDEX = CLOAK_STATE_COUNT
BUFF_VALUE_OFFSET = BUFF_VALIDITY_INDEX + 1
CATEGORICAL_FEATURE_WIDTH = BUFF_VALUE_OFFSET + BUFF_ID_SPACE

_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


@dataclass(frozen=True)
class EncodedEntityFeatures:
    continuous_values: tuple[float, ...]
    continuous_validity: tuple[bool, ...]
    cloak_state: int | None
    buff_ids: tuple[int, ...] | None


def parse_position(value: Any) -> tuple[float, float, float] | None:
    """Return a finite XYZ tuple, or ``None`` for null/sentinel/malformed data."""

    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("("):
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (tuple, list)) or len(parsed) < 3:
        return None
    try:
        coordinates = tuple(float(component) for component in parsed[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in coordinates):
        return None
    return coordinates


def parse_numeric_feature(value: Any) -> tuple[float, bool]:
    """Parse only exact supported scalar forms and report semantic validity."""

    if value is None:
        return 0.0, False
    if isinstance(value, bool):
        return (1.0 if value else 0.0), True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return (parsed, True) if math.isfinite(parsed) else (0.0, False)

    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return 1.0, True
    if lowered == "false":
        return 0.0, True
    if "/" in text:
        pieces = text.split("/")
        if len(pieces) != 2:
            return 0.0, False
        numerator, numerator_valid = _parse_exact_number(pieces[0])
        denominator, denominator_valid = _parse_exact_number(pieces[1])
        if not numerator_valid or not denominator_valid or denominator == 0.0:
            return 0.0, False
        ratio = numerator / denominator
        return (ratio, True) if math.isfinite(ratio) else (0.0, False)
    return _parse_exact_number(text)


def encode_entity_features(
    raw_position: Any,
    raw_attributes: Mapping[str, Any],
) -> EncodedEntityFeatures:
    """Encode all approved fields without conflating valid zero with missing."""

    values: list[float] = []
    validity: list[bool] = []

    position = parse_position(raw_position)
    if position is None:
        values.extend((0.0, 0.0))
        validity.extend((False, False))
    else:
        values.extend(position[:2])
        validity.extend((True, True))

    for key in ("health", "energy", "shields"):
        value, valid = parse_numeric_feature(raw_attributes.get(key))
        values.append(value)
        validity.append(valid)

    if "facing_sin" in raw_attributes and "facing_cos" in raw_attributes:
        facing_sin, sin_valid = parse_numeric_feature(raw_attributes.get("facing_sin"))
        facing_cos, cos_valid = parse_numeric_feature(raw_attributes.get("facing_cos"))
        facing_valid = sin_valid and cos_valid
        values.extend((facing_sin if facing_valid else 0.0, facing_cos if facing_valid else 0.0))
        validity.extend((facing_valid, facing_valid))
    else:
        facing, facing_valid = parse_numeric_feature(raw_attributes.get("facing"))
        values.extend((math.sin(facing), math.cos(facing)) if facing_valid else (0.0, 0.0))
        validity.extend((facing_valid, facing_valid))

    for key in STAT_KEYS[3:]:
        value, valid = parse_numeric_feature(raw_attributes.get(key))
        values.append(value)
        validity.append(valid)

    if len(values) != len(CONTINUOUS_FEATURE_NAMES):
        raise AssertionError("entity feature codec disagrees with the continuous schema")
    return EncodedEntityFeatures(
        continuous_values=tuple(values),
        continuous_validity=tuple(validity),
        cloak_state=parse_cloak_state(raw_attributes.get("cloak")),
        buff_ids=parse_buff_ids(raw_attributes.get("buff_ids")),
    )


def parse_cloak_state(value: Any) -> int | None:
    parsed, valid = parse_numeric_feature(value)
    if not valid or not parsed.is_integer():
        return None
    state = int(parsed)
    return state if 0 <= state < CLOAK_STATE_COUNT else None


def parse_buff_ids(value: Any) -> tuple[int, ...] | None:
    """Parse a valid buff-ID list; malformed/sentinel values are missing."""

    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("["):
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    elif hasattr(value, "tolist"):
        parsed = value.tolist()
    if not isinstance(parsed, (list, tuple, set)):
        return None

    result: set[int] = set()
    for item in parsed:
        if isinstance(item, bool):
            raise ValueError("buff IDs must be integer enum values, not booleans")
        try:
            buff_id = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid buff ID: {item!r}") from exc
        if isinstance(item, float) and not item.is_integer():
            raise ValueError(f"invalid non-integral buff ID: {item!r}")
        if not 0 <= buff_id <= BUFF_ID_MAX:
            raise ValueError(
                f"buff ID {buff_id} exceeds the supported SC2 enum range 0..{BUFF_ID_MAX}"
            )
        result.add(buff_id)
    return tuple(sorted(result))


def pack_continuous_validity(validity: tuple[bool, ...]) -> int:
    if len(validity) > 32:
        raise ValueError("continuous validity packing supports at most 32 features")
    result = 0
    for index, valid in enumerate(validity):
        if valid:
            result |= 1 << index
    return result


def continuous_feature_is_valid(mask: int, index: int) -> bool:
    return bool(mask & (1 << index))


def _parse_exact_number(text: str) -> tuple[float, bool]:
    stripped = text.strip()
    if _NUMBER_RE.fullmatch(stripped) is None:
        return 0.0, False
    parsed = float(stripped)
    return (parsed, True) if math.isfinite(parsed) else (0.0, False)
