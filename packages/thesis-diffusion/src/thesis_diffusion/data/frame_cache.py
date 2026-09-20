"""RAM-bounded least-recently-used cache for replay DataFrames.

Role in the system: the training dataset reads one parquet file per replay and
would, naively, keep every frame it ever touched in memory. On the real corpus
(hundreds of GB of replays) that exhausts host RAM and crashes the run. This
module provides a cache that holds as many recently-used replay frames as fit
inside a byte budget and evicts the least-recently-used frame when the budget is
exceeded. The budget is derived from actual host RAM at runtime, so the same
code scales from a laptop to a large cloud instance without changes.

When the DataLoader uses multiple worker processes, each worker holds its own
cache; the per-worker budget is the global RAM fraction divided by the worker
count so the aggregate footprint stays under the configured fraction.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd

# Conservative fallback budget (bytes) used only when host RAM cannot be
# detected (e.g. psutil missing). 2 GB keeps a small instance safe.
_FALLBACK_BUDGET_BYTES = 2 * 1024**3


def detect_total_ram_bytes() -> int:
    """Return total host RAM in bytes, or a conservative fallback.

    Uses psutil when available (cross-platform). Falls back to the 2 GB
    constant if psutil is not installed so the cache still functions.
    Calls: psutil.virtual_memory (optional dependency).
    """

    try:
        import psutil
    except ImportError:
        return _FALLBACK_BUDGET_BYTES
    return int(psutil.virtual_memory().total)


def resolve_cache_budget_bytes(ram_fraction: float, *, num_workers: int) -> int:
    """Compute this process's frame-cache byte budget.

    Parameters:
        ram_fraction: fraction of total host RAM the cache (aggregated across
            all loader workers) may use. Clamped to (0, 0.95].
        num_workers: number of DataLoader worker processes sharing the budget
            (use 1 for the main process / no workers). The global budget is
            divided evenly so the combined footprint respects ram_fraction.
    Returns:
        Per-process budget in bytes (at least 256 MB so a single large frame
        can always be cached).
    Calls: detect_total_ram_bytes.
    """

    fraction = max(0.0, min(ram_fraction, 0.95))
    workers = max(1, num_workers)
    global_budget = detect_total_ram_bytes() * fraction
    per_worker = int(global_budget / workers)
    return max(per_worker, 256 * 1024**2)


def estimate_frame_bytes(frame: pd.DataFrame) -> int:
    """Return the deep in-memory size of a DataFrame in bytes.

    `deep=True` accounts for Python object columns (strings), which dominate
    these replay frames, so the budget reflects true memory use.
    """

    return int(frame.memory_usage(deep=True).sum())


class BoundedFrameCache:
    """LRU cache of replay DataFrames bounded by a byte budget.

    The budget is resolved lazily on first use (and re-resolved when running
    inside a DataLoader worker, where the worker count becomes known) so the
    cache can size itself to the host it actually runs on.
    """

    def __init__(self, ram_fraction: float) -> None:
        self._ram_fraction = ram_fraction
        self._entries: "OrderedDict[Path, pd.DataFrame]" = OrderedDict()
        self._sizes: dict[Path, int] = {}
        self._current_bytes = 0
        self._budget_bytes: int | None = None

    def _budget(self) -> int:
        """Resolve (once) the per-process budget, accounting for workers."""

        if self._budget_bytes is None:
            num_workers = _current_worker_count()
            self._budget_bytes = resolve_cache_budget_bytes(
                self._ram_fraction,
                num_workers=num_workers,
            )
        return self._budget_bytes

    def get(self, key: Path, loader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
        """Return the frame for `key`, loading and caching it on a miss.

        On a hit, marks the entry most-recently-used. On a miss, loads via
        `loader`, evicts least-recently-used entries until the new frame fits
        the budget, then inserts it. A frame larger than the whole budget is
        still returned (and kept until the next insertion evicts it).
        Parameters:
            key: replay file path (cache key).
            loader: callable that reads and returns the DataFrame for `key`.
        """

        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]

        frame = loader(key)
        size = estimate_frame_bytes(frame)
        self._evict_until_fits(size)
        self._entries[key] = frame
        self._sizes[key] = size
        self._current_bytes += size
        return frame

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        """Evict least-recently-used frames until `incoming_bytes` fits."""

        budget = self._budget()
        while self._entries and self._current_bytes + incoming_bytes > budget:
            evicted_key, _ = self._entries.popitem(last=False)
            self._current_bytes -= self._sizes.pop(evicted_key)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes


def _current_worker_count() -> int:
    """Return the DataLoader worker count for this process (1 if not a worker).

    Uses torch's worker info so each worker divides the global RAM budget by the
    number of sibling workers. Imported lazily to keep this module torch-light.
    """

    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return 1
    info = get_worker_info()
    if info is None:
        return 1
    return max(1, info.num_workers)
