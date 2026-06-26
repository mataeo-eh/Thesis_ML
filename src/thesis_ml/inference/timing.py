"""Arithmetic time recovery for decoded canvases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TimedTimestep:
    timestep_index: int
    timestamp_seconds: float
    counts: dict[str, int]


def attach_absolute_times(
    timesteps: Sequence[dict[str, int]],
    *,
    last_input_clock: float,
    sampling_interval_s: int,
) -> list[TimedTimestep]:
    """Assign times by SPEC §7 arithmetic; canvas contents are not inspected."""

    return [
        TimedTimestep(
            timestep_index=index,
            timestamp_seconds=float(last_input_clock + sampling_interval_s * index),
            counts=dict(counts),
        )
        for index, counts in enumerate(timesteps)
    ]
