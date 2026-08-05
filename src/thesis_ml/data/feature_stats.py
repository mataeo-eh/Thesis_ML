"""Frozen training-split statistics for input entity features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FEATURE_STATISTICS_VERSION = 1
ZERO_VARIANCE_POLICY = "unit-scale"

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
CONTINUOUS_FEATURE_NAMES = ("map_x", "map_y", *STAT_KEYS)


class FeatureStatisticsError(ValueError):
    """Raised when the frozen feature-statistics artifact is unusable."""


@dataclass(frozen=True)
class FeatureStatistics:
    version: int
    feature_names: tuple[str, ...]
    counts: tuple[int, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    zero_variance_features: tuple[str, ...]
    source_replay_ids: tuple[str, ...]
    identity: str

    @classmethod
    def identity_for_tests(cls) -> "FeatureStatistics":
        """Neutral statistics for isolated synthetic/unit tests only."""

        width = len(CONTINUOUS_FEATURE_NAMES)
        payload = {
            "version": FEATURE_STATISTICS_VERSION,
            "feature_names": list(CONTINUOUS_FEATURE_NAMES),
            "counts": [1] * width,
            "means": [0.0] * width,
            "stds": [1.0] * width,
            "zero_variance_features": [],
            "source_replay_ids": ["synthetic-test-only"],
            "zero_variance_policy": ZERO_VARIANCE_POLICY,
        }
        return cls(
            version=FEATURE_STATISTICS_VERSION,
            feature_names=CONTINUOUS_FEATURE_NAMES,
            counts=(1,) * width,
            means=(0.0,) * width,
            stds=(1.0,) * width,
            zero_variance_features=(),
            source_replay_ids=("synthetic-test-only",),
            identity=_payload_identity(payload),
        )


def compute_feature_statistics(
    artifact_paths: Iterable[str | Path],
    *,
    source_replay_ids: Sequence[str],
) -> FeatureStatistics:
    """Compute population statistics once over unique training replay artifacts."""

    # Local import avoids a module cycle: windowing owns TokenizedReplay and
    # imports this module's stable feature schema when it writes artifacts.
    from thesis_ml.data.windowing import TokenizedReplay

    ordered_paths = sorted({str(Path(path)) for path in artifact_paths})
    if not ordered_paths:
        raise FeatureStatisticsError("cannot compute feature statistics from an empty training split")

    width = len(CONTINUOUS_FEATURE_NAMES)
    count = 0
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for artifact_path in ordered_paths:
        values = np.asarray(TokenizedReplay(artifact_path).features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != width:
            raise FeatureStatisticsError(
                f"feature artifact {artifact_path} has shape {values.shape}; expected [N, {width}]"
            )
        if not np.isfinite(values).all():
            raise FeatureStatisticsError(f"feature artifact {artifact_path} contains non-finite values")
        count += int(values.shape[0])
        total += values.sum(axis=0, dtype=np.float64)
        total_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if count <= 0:
        raise FeatureStatisticsError("training split contains no entity feature rows")
    means = total / count
    variances = np.maximum(total_sq / count - np.square(means), 0.0)
    raw_stds = np.sqrt(variances)
    zero_variance = raw_stds == 0.0
    stds = np.where(zero_variance, 1.0, raw_stds)
    payload = {
        "version": FEATURE_STATISTICS_VERSION,
        "feature_names": list(CONTINUOUS_FEATURE_NAMES),
        "counts": [count] * width,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "zero_variance_features": [
            name for name, is_zero in zip(CONTINUOUS_FEATURE_NAMES, zero_variance, strict=True) if is_zero
        ],
        "source_replay_ids": sorted(set(source_replay_ids)),
        "zero_variance_policy": ZERO_VARIANCE_POLICY,
    }
    return _statistics_from_payload(payload, identity=_payload_identity(payload))


def write_feature_statistics(statistics: FeatureStatistics, path: str | Path) -> Path:
    """Persist statistics deterministically as versioned canonical JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(statistics)
    payload["identity"] = statistics.identity
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_feature_statistics(
    path: str | Path,
    *,
    expected_identity: str | None = None,
    expected_source_replay_ids: Sequence[str] | None = None,
) -> FeatureStatistics:
    """Load and strictly validate a frozen statistics artifact."""

    target = Path(path)
    if not target.exists():
        raise FeatureStatisticsError(
            f"feature statistics artifact is missing at {target}; run the explicit "
            "feature-statistics preprocessing step before training/evaluation"
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureStatisticsError(f"feature statistics artifact is unreadable: {target}") from exc
    if not isinstance(raw, dict):
        raise FeatureStatisticsError("feature statistics artifact must contain a JSON object")
    identity = raw.pop("identity", None)
    computed_identity = _payload_identity(raw)
    if not isinstance(identity, str) or identity != computed_identity:
        raise FeatureStatisticsError("feature statistics identity is missing or does not match its payload")
    statistics = _statistics_from_payload(raw, identity=identity)
    if expected_identity is not None and statistics.identity != expected_identity:
        raise FeatureStatisticsError(
            "feature statistics are incompatible with the checkpoint: "
            f"expected {expected_identity}, got {statistics.identity}"
        )
    if expected_source_replay_ids is not None:
        expected = tuple(sorted(set(expected_source_replay_ids)))
        if statistics.source_replay_ids != expected:
            raise FeatureStatisticsError(
                "feature statistics training split does not match the selected training replays"
            )
    return statistics


def _statistics_from_payload(payload: dict[str, object], *, identity: str) -> FeatureStatistics:
    required = {
        "version",
        "feature_names",
        "counts",
        "means",
        "stds",
        "zero_variance_features",
        "source_replay_ids",
        "zero_variance_policy",
    }
    if set(payload) != required:
        raise FeatureStatisticsError("feature statistics artifact has an incompatible schema")
    if payload["version"] != FEATURE_STATISTICS_VERSION:
        raise FeatureStatisticsError(
            f"unsupported feature statistics version: {payload['version']!r}"
        )
    if payload["zero_variance_policy"] != ZERO_VARIANCE_POLICY:
        raise FeatureStatisticsError("feature statistics zero-variance policy is incompatible")
    try:
        names_raw = _json_list(payload["feature_names"], "feature_names")
        counts_raw = _json_list(payload["counts"], "counts")
        means_raw = _json_list(payload["means"], "means")
        stds_raw = _json_list(payload["stds"], "stds")
        zero_variance_raw = _json_list(
            payload["zero_variance_features"], "zero_variance_features"
        )
        replay_ids_raw = _json_list(payload["source_replay_ids"], "source_replay_ids")
        if not all(isinstance(value, str) for value in names_raw):
            raise TypeError("feature_names must contain strings")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in counts_raw):
            raise TypeError("counts must contain integers")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (*means_raw, *stds_raw)
        ):
            raise TypeError("means and stds must contain numbers")
        if not all(isinstance(value, str) for value in zero_variance_raw):
            raise TypeError("zero_variance_features must contain strings")
        if not all(isinstance(value, str) for value in replay_ids_raw):
            raise TypeError("source_replay_ids must contain strings")
        names = tuple(names_raw)
        counts = tuple(counts_raw)
        means = tuple(float(value) for value in means_raw)
        stds = tuple(float(value) for value in stds_raw)
        zero_variance = tuple(zero_variance_raw)
        replay_ids = tuple(replay_ids_raw)
    except (TypeError, ValueError) as exc:
        raise FeatureStatisticsError("feature statistics values are malformed") from exc
    if names != CONTINUOUS_FEATURE_NAMES:
        raise FeatureStatisticsError(
            f"feature statistics order is incompatible: expected {CONTINUOUS_FEATURE_NAMES}, got {names}"
        )
    width = len(names)
    if not (len(counts) == len(means) == len(stds) == width):
        raise FeatureStatisticsError("feature statistics vectors do not match the feature schema")
    if any(value <= 0 for value in counts):
        raise FeatureStatisticsError("feature statistics counts must all be positive")
    if any(not math.isfinite(value) for value in (*means, *stds)) or any(value <= 0 for value in stds):
        raise FeatureStatisticsError("feature statistics means/stds must be finite and stds positive")
    if tuple(sorted(set(replay_ids))) != replay_ids or not replay_ids:
        raise FeatureStatisticsError("feature statistics source replay ids must be non-empty and sorted")
    if any(name not in names for name in zero_variance):
        raise FeatureStatisticsError("feature statistics zero-variance names are incompatible")
    return FeatureStatistics(
        version=FEATURE_STATISTICS_VERSION,
        feature_names=names,
        counts=counts,
        means=means,
        stds=stds,
        zero_variance_features=zero_variance,
        source_replay_ids=replay_ids,
        identity=identity,
    )


def _payload(statistics: FeatureStatistics) -> dict[str, object]:
    return {
        "version": statistics.version,
        "feature_names": list(statistics.feature_names),
        "counts": list(statistics.counts),
        "means": list(statistics.means),
        "stds": list(statistics.stds),
        "zero_variance_features": list(statistics.zero_variance_features),
        "source_replay_ids": list(statistics.source_replay_ids),
        "zero_variance_policy": ZERO_VARIANCE_POLICY,
    }


def _payload_identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array")
    return value
