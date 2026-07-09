"""Deterministic, resumable batch sampler for interrupted training runs.

Local training of the SC2 diffusion model is too slow to finish in one session,
so a run is expected to be killed and resumed many times. The training loop
(``thesis_ml.train.loop.TrainingLoop``) checkpoints the optimizer, model, step
counter, and -- crucially for this file -- HOW FAR INTO THE CURRENT EPOCH it has
progressed. For that intra-epoch position to mean anything on resume, the order
in which batches are drawn from the dataset must be reproducible: if the shuffle
were re-randomized on every process start (as PyTorch's default ``shuffle=True``
``RandomSampler`` is, because it draws from the global RNG), then "skip the
first K batches" would skip K *arbitrary* batches and the model would keep
re-seeing the same early slice of the epoch over and over.

This sampler provides the missing guarantee. Given an epoch index it produces a
fixed permutation of dataset indices (seeded by ``base_seed + epoch``), chunks
it into batches, and can fast-forward past a number of already-consumed batches
before yielding the rest. It is passed to ``DataLoader(batch_sampler=...)``.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[list[int]]):
    """Yield deterministic, per-epoch-shuffled batches with a resume offset.

    The DataLoader calls ``iter()`` on this object once per epoch. Each call
    rebuilds the SAME index permutation for the currently-set epoch (so the
    ordering is reproducible across process restarts) and skips the first
    ``start_batch`` batches, which is how a mid-epoch resume avoids replaying
    batches the model already trained on.

    Args:
        dataset_size: Number of examples in the dataset (``len(dataset)``).
        batch_size: Examples per batch.
        base_seed: Base RNG seed; the per-epoch seed is ``base_seed + epoch`` so
            each epoch has a distinct but reproducible shuffle.
        drop_last: If True, drop a trailing partial batch (matches
            ``DataLoader(drop_last=...)`` semantics). Defaults to False.

    Coordinating methods (called by ``TrainingLoop.fit``):
        set_epoch(epoch): Select which epoch's permutation to produce.
        set_start_batch(n): Skip the first ``n`` batches on the NEXT iteration
            only; the offset auto-clears after one epoch so later (freshly
            started) epochs begin at batch 0.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        base_seed: int,
        drop_last: bool = False,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._dataset_size = int(dataset_size)
        self._batch_size = int(batch_size)
        self._base_seed = int(base_seed)
        self._drop_last = bool(drop_last)
        # Which epoch's permutation to emit; updated by the training loop each
        # epoch via set_epoch so the seed (base_seed + epoch) shifts per epoch.
        self._epoch = 0
        # How many leading batches to skip on the next __iter__ only. Set to the
        # checkpoint's intra-epoch progress on resume; cleared back to 0 after
        # one iteration so subsequent epochs are not truncated.
        self._start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose reproducible permutation will be produced."""
        self._epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Skip this many leading batches on the next iteration only."""
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        self._start_batch = int(start_batch)

    def __len__(self) -> int:
        """Total batches in a FULL epoch (ignores the resume offset).

        The training loop reads ``len(dataloader)`` to report progress and to
        size the epoch, so this must reflect the whole epoch regardless of how
        many batches a resume will skip.
        """
        if self._drop_last:
            return self._dataset_size // self._batch_size
        return (self._dataset_size + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[list[int]]:
        # Rebuild the epoch's permutation deterministically. Seeding a fresh
        # generator with base_seed + epoch means every process that iterates a
        # given epoch -- including one that just resumed from a checkpoint --
        # gets byte-for-byte the same batch order, which is what makes the
        # skip-ahead below land on the correct next batch.
        generator = torch.Generator()
        generator.manual_seed(self._base_seed + self._epoch)
        order = torch.randperm(self._dataset_size, generator=generator).tolist()

        # Chunk the flat index order into fixed-size batches.
        batches: list[list[int]] = [
            order[start : start + self._batch_size]
            for start in range(0, self._dataset_size, self._batch_size)
        ]
        if self._drop_last and batches and len(batches[-1]) < self._batch_size:
            batches.pop()

        # Fast-forward past already-consumed batches for a mid-epoch resume,
        # then clear the offset so the NEXT epoch starts from the beginning.
        start_batch = self._start_batch
        self._start_batch = 0
        for batch in batches[start_batch:]:
            yield batch
