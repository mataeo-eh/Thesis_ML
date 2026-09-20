"""Reproducible train/dev/test partitioning over replays.

Role in the system: the corpus is not pre-split into train/dev/test on disk.
Splitting at training time keeps the data store simple, but it must be
*reproducible* so every run (and a resumed run on a fresh cloud instance) sees
the exact same partition. We therefore shuffle the replay list with a fixed seed
and partition deterministically.

We split by REPLAY, not by training window/example. Two windows from the same
replay are highly correlated; placing one in train and another in test would
leak information and inflate held-out metrics. Splitting whole replays avoids
that leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReplaySplit:
    """The three disjoint replay-path groups produced by `split_replays`."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]


def split_replays(
    replay_paths: Sequence[str | Path],
    *,
    seed: int,
    test_fraction: float,
    dev_fraction: float,
    train_count: int = 0,
    dev_count: int = 0,
) -> ReplaySplit:
    """Deterministically partition replay paths into train/dev/test.

    The split is two-stage, matching the project convention: first hold out
    `test_fraction` of all replays as the test set; then hold out
    `dev_fraction` of the REMAINING (train) replays as the dev set. With the
    defaults (0.15, 0.10) this yields roughly 76.5% train, 8.5% dev, 15% test.

    Parameters:
        replay_paths: all available replay file paths.
        seed: fixed seed controlling the shuffle, independent of the training
            seed so re-seeding a run does not reshuffle the split.
        test_fraction: fraction of all replays reserved for test (0..1).
        dev_fraction: fraction of the post-test (train) replays reserved for
            dev (0..1).
    Returns:
        ReplaySplit with disjoint, sorted-by-shuffle path tuples. Every input
        replay appears in exactly one group.
    Calls: numpy.random.default_rng (seeded permutation).
    """

    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")

    paths = [str(path) for path in replay_paths]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    shuffled = [paths[index] for index in order]

    total = len(shuffled)
    if train_count < 0 or dev_count < 0:
        raise ValueError("train_count and dev_count must be non-negative")
    if train_count > 0:
        if train_count + dev_count > total:
            raise ValueError(
                f"requested {train_count} train + {dev_count} dev replays, "
                f"but only {total} are available"
            )
        train = shuffled[:train_count]
        dev = shuffled[train_count : train_count + dev_count]
        test = shuffled[train_count + dev_count :]
        return ReplaySplit(train=tuple(train), dev=tuple(dev), test=tuple(test))

    test_count = int(round(total * test_fraction))
    test = shuffled[:test_count]
    train_and_dev = shuffled[test_count:]

    dev_count = int(round(len(train_and_dev) * dev_fraction))
    dev = train_and_dev[:dev_count]
    train = train_and_dev[dev_count:]

    return ReplaySplit(train=tuple(train), dev=tuple(dev), test=tuple(test))
