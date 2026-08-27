"""Measure how fixed-index token cross-entropy prices small SC2 semantic edits.

Role in the larger system
-------------------------
This is a READ-ONLY training-objective diagnostic. It does not change the
production loss, serializer, model, corruption process, or sampler, and it never
writes into a training run or into ``Model_Inference_Tests/``. Its only durable
side effect is a bounded JSON/text/CSV bundle under
``scripts/output/timestep_alignment_probe/``.

The question it answers
-----------------------
The production canvas grammar is::

    [BOS] [WIN|LOSS] (content* [DELIMITER])+ ([END] [PAD]* | [PAD]*)

Each delimiter-bounded group is ONE one-second SC2 timestep, and the decoded
semantic state of a timestep is the multiset/count of its content token types
(``decode_canvas`` does not retain entity order inside a group). The training
objective, however, is positionwise cross entropy against fixed serialized
COORDINATES. A single missing or extra content token inside one timestep shifts
the expected index of every remaining content token and of that timestep's
delimiter, so a prediction whose delimiter-local semantic edit distance is small
can still collect many wrong-coordinate penalties.

Those penalties are ADDITIVE across positions, never exponential. What one
semantic insertion/deletion can do is create MANY additive positionwise
penalties before the sequence re-aligns; that multiplicity is what this probe
measures.

Three arms
----------
A. ``run_geometry_experiment`` -- model-independent objective geometry. Real
   clean target canvases, deterministic high-confidence pseudo-logits, and five
   controlled perturbations. This arm establishes what the objective rewards
   without reference to anything a checkpoint learned, so it cannot be confounded
   by the checkpoint having been trained under this very objective.
B. ``run_model_experiment`` -- one denoiser forward pass per controlled
   corruption level on real recorded replay windows with the named V3 EMA
   checkpoint. Positional, structural, delimiter-local semantic, and oracle
   aligned-content scores over the SAME logits and targets.
C. controls inside ``run_model_experiment`` -- correct input versus shuffled
   input versus enemy-stripped input on identical noised canvases, plus a
   persistence baseline and a train-split content unigram baseline.

Causal caveat, restated in code because it governs how the numbers may be read
-------------------------------------------------------------------------------
The probed checkpoint was itself trained under positional CE. Arm B is therefore
OBSERVATIONAL: a large positional-versus-aligned gap shows the trained model's
current errors have an alignment component, and a small one does not prove an
alignment-aware objective would fail. Only a matched training ablation can
establish causation. Arm A is the model-independent half and is the only arm
that can prove an objective-geometry claim on its own.

Calls into the production package rather than reimplementing it:
``thesis_ml.config.load_config``, ``thesis_ml.data.dataset.SC2DiffusionDataset``,
``thesis_ml.data.collate.collate_diffusion_examples`` (through
``_make_dataloader``), ``thesis_ml.train.corruption.corrupt_batch``,
``thesis_ml.model.loss.CanvasCrossEntropyLoss``,
``thesis_ml.viz.diagnostics.load_diagnostic_model``,
``thesis_ml.train.loop._macro_f1_from_counts``, and the training pipeline's own
replay-selection helpers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_PAD,
    CLASS_WINLOSS,
    SC2DiffusionDataset,
)
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import (
    WindowManifestEntry,
    load_window_manifest,
    read_manifest_metadata,
)
from thesis_ml.model.embedding import InputFeatures
from thesis_ml.model.loss import (
    FUTURE_DISTANCE_BUCKETS,
    CanvasCrossEntropyLoss,
    active_class_id_to_name,
)
from thesis_ml.pipeline.storage import StorageResolver
from thesis_ml.pipeline.train_pipeline import (
    _ensure_window_manifest,
    _explicit_replay_selection,
    _make_dataloader,
    _materialize_file,
    _materialize_replay_paths,
    _select_replays,
    _shutdown_dataloader,
)
from thesis_ml.train.loop import _macro_f1_from_counts
from thesis_ml.train.corruption import corrupt_batch
from thesis_ml.viz.diagnostics import load_diagnostic_model
from thesis_ml.vocab.content_vocab import ContentVocabulary, load_content_vocabulary
from thesis_ml.vocab.special_tokens import (
    BOS_ID,
    CONTENT_TOKEN_OFFSET,
    DELIMITER_ID,
    END_ID,
    LOSS_ID,
    PAD_ID,
    WIN_ID,
)

# Repository-root-relative defaults. Nothing here embeds a workstation path; the
# provenance writer renders every path through `portable_path` below.
DEFAULT_OUTPUT_DIR = Path("scripts/output/timestep_alignment_probe")
DEFAULT_CONFIG = Path("configs/smallTrainingTestV3.yaml")
DEFAULT_CHECKPOINT = Path(
    "tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt"
)
# The corruption sweep required by the investigation. Exact 1.0 is separated from
# 0.99 on purpose: at t=1 no truthful canvas token survives as an alignment
# landmark, which is the regime the hypothesis is specifically about.
DEFAULT_NOISE_LEVELS = (0.75, 0.90, 0.99, 1.00)
# Dominant Protoss economy/base tokens. Delimiter-local semantic metrics are
# reported both with and without these, because a multiset score can be carried
# almost entirely by getting the probe count roughly right.
DOMINANT_CONTENT_TOKEN_NAMES = ("probe", "nexus")

# Structural (non-content) canvas token ids. A content token is any id at or
# above CONTENT_TOKEN_OFFSET; everything below it is one of the reserved
# specials and is never part of a timestep's semantic multiset.
STRUCTURAL_TOKEN_IDS = frozenset(range(CONTENT_TOKEN_OFFSET))


# ---------------------------------------------------------------------------
# Canvas segmentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimestepSpan:
    """One delimiter-bounded group of a canvas: exactly one SC2 timestep.

    Attributes:
        ordinal: 0-based index of this timestep within the canvas.
        content_start: first content position (inclusive).
        content_end: one past the last content position; equals
            ``delimiter_index``, so an EMPTY timestep has
            ``content_start == content_end``.
        delimiter_index: canvas index holding this group's ``[DELIMITER]``.
    """

    ordinal: int
    content_start: int
    content_end: int
    delimiter_index: int

    @property
    def content_length(self) -> int:
        return self.content_end - self.content_start


@dataclass(frozen=True)
class CanvasLayout:
    """Structural parse of one canvas row, batch-shape padding already excluded.

    Attributes:
        active_length: number of real canvas positions in the row (the row's
            ``canvas_attention_mask`` popcount). Batch-shape padding beyond this
            is not part of the canvas at all and is never scored.
        outcome_index: index of the single ``[WIN]``/``[LOSS]`` token (always 1).
        timesteps: the delimiter-bounded groups, in order.
        delimiter_indices: canvas indices of every ``[DELIMITER]``, in order.
        end_index: index of the terminal ``[END]``, or None for a
            boundary-truncated horizon that ends on a delimiter and pads directly.
        semantic_pad_indices: indices of the SEMANTIC ``[PAD]`` run -- surplus
            canvas positions that are genuine prediction targets. This never
            includes batch-shape padding, which lies beyond ``active_length``.
        terminated: True when a terminal ``[END]`` is present.

    Calls: nothing; a pure scan over the id sequence.
    """

    active_length: int
    outcome_index: int
    timesteps: tuple[TimestepSpan, ...]
    delimiter_indices: tuple[int, ...]
    end_index: int | None
    semantic_pad_indices: tuple[int, ...]
    terminated: bool


def parse_canvas_layout(token_ids: Sequence[int], active_length: int | None = None) -> CanvasLayout:
    """Segment a ground-truth canvas row into timesteps, terminator, and pads.

    This is the segmentation every semantic metric in this module is anchored to.
    It deliberately mirrors ``inference.decode.validate_canvas``'s notion of the
    grammar but returns COORDINATES instead of decoded names, because the whole
    question here is about coordinates.

    Parameters:
        token_ids: the canvas row's token ids, including clamped ``[BOS]``.
        active_length: real canvas width; ``None`` means the whole sequence.

    Returns:
        A :class:`CanvasLayout`.

    Raises:
        ValueError: the row does not begin ``[BOS] [WIN|LOSS]``, or a content
            position holds a token that cannot appear there.

    Calls: nothing.
    """

    ids = list(token_ids)
    width = len(ids) if active_length is None else int(active_length)
    if width < 2:
        raise ValueError("a canvas row must hold at least [BOS] and an outcome token")
    if ids[0] != BOS_ID:
        raise ValueError("canvas position 0 must be clamped [BOS]")
    if ids[1] not in (WIN_ID, LOSS_ID):
        raise ValueError("canvas position 1 must be [WIN] or [LOSS]")

    timesteps: list[TimestepSpan] = []
    delimiter_indices: list[int] = []
    end_index: int | None = None
    content_start = 2
    index = 2
    while index < width:
        token = ids[index]
        if token == DELIMITER_ID:
            timesteps.append(
                TimestepSpan(
                    ordinal=len(timesteps),
                    content_start=content_start,
                    content_end=index,
                    delimiter_index=index,
                )
            )
            delimiter_indices.append(index)
            content_start = index + 1
            index += 1
            continue
        if token == END_ID:
            end_index = index
            index += 1
            break
        if token == PAD_ID:
            # A boundary-truncated horizon pads directly from the last delimiter.
            break
        index += 1

    pad_start = index
    semantic_pad_indices = tuple(range(pad_start, width))
    return CanvasLayout(
        active_length=width,
        outcome_index=1,
        timesteps=tuple(timesteps),
        delimiter_indices=tuple(delimiter_indices),
        end_index=end_index,
        semantic_pad_indices=semantic_pad_indices,
        terminated=end_index is not None,
    )


def predicted_structure(token_ids: Sequence[int], active_length: int) -> dict[str, int]:
    """Summarize a PREDICTED canvas row without requiring it to be grammatical.

    A hard argmax row need not parse: it can hold two delimiters in a row, a
    ``[PAD]`` in the middle, or no terminator at all. So instead of parsing it,
    this reports the three structural quantities the investigation compares
    against the target:

      * ``delimiter_count`` -- how many ``[DELIMITER]`` tokens were emitted;
      * ``active_length`` -- the index of the FIRST terminal token
        (``[END]`` or ``[PAD]``), i.e. where the model thinks the canvas body
        stops. Falls back to the real canvas width when neither appears;
      * ``end_count`` -- how many ``[END]`` tokens were emitted.

    Parameters:
        token_ids: predicted canvas row ids.
        active_length: the row's real canvas width (batch padding excluded).

    Returns:
        A dict with ``delimiter_count``, ``active_length``, and ``end_count``.

    Calls: nothing.
    """

    ids = list(token_ids)[:active_length]
    delimiter_count = sum(1 for token in ids[2:] if token == DELIMITER_ID)
    end_count = sum(1 for token in ids[2:] if token == END_ID)
    predicted_active = active_length
    for index in range(2, len(ids)):
        if ids[index] in (END_ID, PAD_ID):
            predicted_active = index
            break
    return {
        "delimiter_count": delimiter_count,
        "active_length": predicted_active,
        "end_count": end_count,
    }


# ---------------------------------------------------------------------------
# Delimiter-local semantic comparisons
# ---------------------------------------------------------------------------


def content_counts(token_ids: Iterable[int], *, exclude: frozenset[int] = frozenset()) -> Counter:
    """Count CONTENT token types, dropping structural ids and any exclusions.

    Parameters:
        token_ids: ids covering one timestep's content span.
        exclude: content ids to leave out (used for the "excluding probe/nexus"
            variant of every multiset metric).

    Returns:
        A ``Counter`` from content token id to occurrence count. This is exactly
        the semantic state ``decode_canvas`` would report for the group, minus
        the name lookup.

    Calls: nothing.
    """

    counts: Counter = Counter()
    for token in token_ids:
        if token in STRUCTURAL_TOKEN_IDS or token in exclude:
            continue
        counts[int(token)] += 1
    return counts


def multiset_overlap(predicted: Mapping[int, int], target: Mapping[int, int]) -> int:
    """Return sum over token types of ``min(predicted, target)``.

    This is the multiset intersection size, i.e. how many predicted occurrences
    can be matched one-to-one with target occurrences of the same type. It is the
    numerator of both multiset precision and multiset recall.
    """

    return sum(min(count, target.get(token, 0)) for token, count in predicted.items())


def multiset_count_error(predicted: Mapping[int, int], target: Mapping[int, int]) -> tuple[int, int]:
    """Return (absolute count error, number of token-type cells compared).

    The cells compared are the UNION of the token types present in either
    multiset, so a token the model invents and a token it drops both count.
    """

    tokens = set(predicted) | set(target)
    error = sum(abs(predicted.get(token, 0) - target.get(token, 0)) for token in tokens)
    return error, len(tokens)


def levenshtein(left: Sequence[int], right: Sequence[int]) -> int:
    """Ordered edit distance (insert/delete/substitute) between two id sequences.

    Used for the delimiter-LOCAL edit distance: it is applied to one timestep
    group at a time, never across the whole canvas, so its cost stays bounded by
    the group length.

    Calls: nothing.
    """

    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class TimestepSemantics:
    """Delimiter-local semantic comparison for ONE timestep group."""

    overlap: int
    predicted_total: int
    target_total: int
    exact_multiset: bool
    count_error: int
    count_cells: int
    edit_distance: int


def compare_timestep(
    predicted_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    exclude: frozenset[int] = frozenset(),
) -> TimestepSemantics:
    """Compare one timestep's predicted content against its target content.

    Order is deliberately NOT required for the multiset half: the decoded
    semantic state of a timestep is its count vector, and entity order inside a
    group is serialization bookkeeping. The ordered ``edit_distance`` is reported
    alongside so the two views can be contrasted.

    Calls: ``content_counts``, ``multiset_overlap``, ``multiset_count_error``,
    ``levenshtein``.
    """

    predicted_counts = content_counts(predicted_tokens, exclude=exclude)
    target_counts = content_counts(target_tokens, exclude=exclude)
    overlap = multiset_overlap(predicted_counts, target_counts)
    error, cells = multiset_count_error(predicted_counts, target_counts)
    filtered_predicted = [
        int(token)
        for token in predicted_tokens
        if token not in STRUCTURAL_TOKEN_IDS and token not in exclude
    ]
    filtered_target = [
        int(token)
        for token in target_tokens
        if token not in STRUCTURAL_TOKEN_IDS and token not in exclude
    ]
    return TimestepSemantics(
        overlap=overlap,
        predicted_total=sum(predicted_counts.values()),
        target_total=sum(target_counts.values()),
        exact_multiset=predicted_counts == target_counts,
        count_error=error,
        count_cells=cells,
        edit_distance=levenshtein(filtered_predicted, filtered_target),
    )


# ---------------------------------------------------------------------------
# Minimum-cost assignment (oracle aligned content score)
# ---------------------------------------------------------------------------


def linear_sum_assignment_min(cost: np.ndarray) -> np.ndarray:
    """Solve a rectangular minimum-cost assignment problem (Jonker-Volgenant).

    Implemented here rather than imported because SciPy is deliberately not a
    dependency of this repository; the algorithm is the standard shortest-
    augmenting-path method with dual potentials, vectorized over columns with
    NumPy so each augmentation costs O(nc) NumPy operations rather than O(nc)
    Python-level operations.

    Parameters:
        cost: ``[nr, nc]`` finite cost matrix with ``nr <= nc``.

    Returns:
        ``col_for_row``: an integer array of length ``nr`` where entry ``i`` is
        the column assigned to row ``i``. The total cost is
        ``cost[np.arange(nr), col_for_row].sum()`` and is the global minimum.

    Raises:
        ValueError: ``nr > nc`` or the matrix holds a non-finite entry.

    Calls: ``_augmenting_path``.
    """

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be two-dimensional")
    rows, columns = matrix.shape
    if rows > columns:
        raise ValueError("linear_sum_assignment_min requires nr <= nc")
    if rows == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.isfinite(matrix).all():
        raise ValueError("cost matrix must be finite")

    u = np.zeros(rows, dtype=np.float64)
    v = np.zeros(columns, dtype=np.float64)
    path = np.full(columns, -1, dtype=np.int64)
    col_for_row = np.full(rows, -1, dtype=np.int64)
    row_for_col = np.full(columns, -1, dtype=np.int64)
    shortest = np.empty(columns, dtype=np.float64)

    for current_row in range(rows):
        sink, min_value, scanned_rows, scanned_cols = _augmenting_path(
            matrix, u, v, path, row_for_col, shortest, current_row
        )
        # Dual update: keep every reduced cost non-negative so later Dijkstra
        # passes stay valid.
        u[current_row] += min_value
        other_rows = scanned_rows.copy()
        other_rows[current_row] = False
        if other_rows.any():
            u[other_rows] += min_value - shortest[col_for_row[other_rows]]
        v[scanned_cols] -= min_value - shortest[scanned_cols]
        # Walk the augmenting path back to the free row, flipping matches.
        column = sink
        while True:
            row = int(path[column])
            row_for_col[column] = row
            col_for_row[row], column = column, col_for_row[row]
            if row == current_row:
                break
    return col_for_row


def _augmenting_path(
    cost: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    path: np.ndarray,
    row_for_col: np.ndarray,
    shortest: np.ndarray,
    start_row: int,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Dijkstra over reduced costs to find one shortest augmenting path.

    Internal helper for :func:`linear_sum_assignment_min`; see that docstring.
    Returns ``(sink_column, min_value, scanned_rows, scanned_columns)``.
    """

    rows, columns = cost.shape
    min_value = 0.0
    remaining = np.arange(columns - 1, -1, -1, dtype=np.int64)
    num_remaining = columns
    scanned_rows = np.zeros(rows, dtype=bool)
    scanned_cols = np.zeros(columns, dtype=bool)
    shortest[:] = np.inf
    row = start_row
    sink = -1
    while sink == -1:
        scanned_rows[row] = True
        active = remaining[:num_remaining]
        reduced = min_value + cost[row, active] - u[row] - v[active]
        improved = reduced < shortest[active]
        improved_cols = active[improved]
        path[improved_cols] = row
        shortest[improved_cols] = reduced[improved]
        distances = shortest[active]
        lowest = distances.min()
        candidates = np.flatnonzero(distances == lowest)
        # Prefer a free column on ties, exactly as the reference implementation
        # does: it terminates the search one step earlier.
        free = candidates[row_for_col[active[candidates]] == -1]
        index = int(free[0]) if free.size else int(candidates[0])
        min_value = float(lowest)
        column = int(active[index])
        if not math.isfinite(min_value):
            raise ValueError("assignment problem is infeasible")
        if row_for_col[column] == -1:
            sink = column
        else:
            row = int(row_for_col[column])
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]
        scanned_cols[column] = True
    return sink, min_value, scanned_rows, scanned_cols


def oracle_aligned_content_cost(
    log_probabilities: np.ndarray,
    target_tokens: Sequence[int],
) -> float:
    """Order-invariant lower bound on a timestep's content negative log-likelihood.

    Within ONE ground-truth timestep content span, every output slot is allowed
    to claim any single target occurrence in that same span, and each target
    occurrence is claimed exactly once. Duplicate occurrences of the same token
    type are treated as DISTINCT columns, so a timestep that truly contains three
    ``probe`` tokens still requires three slots to pay for them.

    The result is deliberately OPTIMISTIC. It is a diagnostic score, not a
    likelihood, not the production loss, and not a claim that a matching
    algorithm should be trained: it never has to produce a generatable delimiter
    sequence, and it is free to reorder within the span. Its only job is to say
    how much of the positional content CE is attributable to coordinate
    misalignment rather than to predicting the wrong token types.

    Parameters:
        log_probabilities: ``[n, vocab]`` per-slot log-softmax rows for the span.
        target_tokens: the ``n`` target token ids in that span.

    Returns:
        The minimum achievable sum of negative log probabilities, in nats.

    Raises:
        ValueError: the slot count and target count disagree.

    Calls: ``linear_sum_assignment_min``.
    """

    targets = list(target_tokens)
    if log_probabilities.shape[0] != len(targets):
        raise ValueError("slot count and target-occurrence count must match")
    if not targets:
        return 0.0
    cost = -log_probabilities[:, np.asarray(targets, dtype=np.int64)]
    assignment = linear_sum_assignment_min(cost)
    return float(cost[np.arange(cost.shape[0]), assignment].sum())


# ---------------------------------------------------------------------------
# Experiment A: model-independent objective geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Perturbation:
    """One controlled edit of a clean target canvas.

    Attributes:
        name: stable identifier used as the report key.
        description: what the edit does, in one sentence.
        predicted: the perturbed token row (same length as the target row).
        semantic_edits: the number of MINIMAL delimiter-local semantic edits the
            construction intends -- insertions/deletions/substitutions of content
            occurrences, counted per timestep group.
        focus_timestep: the ordinal of the timestep the edit was applied to.
        expected_realignment_index: the first canvas index at or after which the
            prediction is identical to the target again, for edits whose shift is
            deliberately bounded; ``None`` for the unbounded control.
    """

    name: str
    description: str
    predicted: list[int]
    semantic_edits: int
    focus_timestep: int
    expected_realignment_index: int | None


def select_focus_timestep(layout: CanvasLayout) -> TimestepSpan:
    """Pick the timestep group every controlled edit is applied to.

    The group must (a) hold at least four content tokens, so a middle-of-group
    edit has content on both sides, and (b) be followed by another group, so a
    bounded shift has somewhere to re-align to. Among the qualifying groups the
    one with the MEDIAN content length is chosen, because picking the first
    qualifying group would systematically sample the small early-game timesteps
    and understate how many coordinates one edit actually moves.

    Raises:
        ValueError: no group qualifies.

    Calls: nothing.
    """

    candidates = [span for span in layout.timesteps[:-1] if span.content_length >= 4]
    if not candidates:
        raise ValueError("no non-final timestep group with >= 4 content tokens")
    ordered = sorted(candidates, key=lambda span: (span.content_length, span.ordinal))
    return ordered[len(ordered) // 2]


def build_perturbations(
    target: Sequence[int],
    layout: CanvasLayout,
    *,
    vocabulary_size: int,
) -> list[Perturbation]:
    """Construct the five controlled cases (plus one unbounded control).

    All edits are applied to the median-length qualifying timestep group (see
    :func:`select_focus_timestep`) at that group's FIRST content position, which
    is the maximal within-group shift a single deletion or insertion can produce
    while still re-aligning at the group's delimiter. The offset sweep in
    :func:`deletion_offset_sweep` reports what happens at every other offset, so
    the maximal case is never mistaken for the typical one.

    The cases are:

    1. ``exact_target`` -- the prediction equals the target. Zero semantic edits,
       zero positional mismatches. This pins the floor.
    2. ``content_substitution`` -- one content occurrence is replaced by a
       different content token, in place. ONE semantic edit, ONE positional
       mismatch: the coordinate objective prices this correctly.
    3. ``content_deletion_local_shift`` -- one content occurrence is deleted and
       everything from there through the group's ``[DELIMITER]`` shifts one
       position LEFT, so the delimiter lands one position early and the following
       group's first token is duplicated into the vacated slot. TWO semantic
       edits (one deletion in group j, one duplication in group j+1); the shift
       is bounded and positions after the vacated slot are exact again. Delimiter
       COUNT is preserved and exactly one delimiter is displaced by -1.
    4. ``content_insertion_local_shift`` -- the mirror image: one occurrence is
       duplicated and everything through the delimiter shifts one position RIGHT,
       consuming the following group's first token. TWO semantic edits; delimiter
       count preserved, one delimiter displaced by +1.
    5. ``delimiter_displacement`` -- the group's ``[DELIMITER]`` swaps places with
       its last content token, so the content is delimiter-locally intact but
       assigned to the wrong group. TWO semantic edits, TWO positional
       mismatches: another case the coordinate objective prices correctly.
    6. ``content_deletion_global_shift`` (control, not one of the five) -- the
       same single deletion with the shift left UNBOUNDED to the end of the
       canvas. It exists purely to answer "how far does the penalty propagate
       when the model does not re-align", and it is what makes the bounded cases
       interpretable.

    Parameters:
        target: the clean target canvas row (active positions only).
        layout: that row's parsed layout.
        vocabulary_size: live vocabulary width, used to pick a substitute token
            that is guaranteed to be a legal content id and different from the
            original.

    Returns:
        The perturbations, in report order.

    Raises:
        ValueError: no timestep group satisfies the selection rule.

    Calls: nothing.
    """

    ids = list(target)
    focus = select_focus_timestep(layout)
    next_span = layout.timesteps[focus.ordinal + 1]
    if next_span.content_length < 1:
        raise ValueError("the group after the focus group is empty")

    # The group's FIRST content position: one deletion here shifts the whole
    # group, which is the maximal bounded case.
    edit_index = focus.content_start
    delimiter_index = focus.delimiter_index

    cases: list[Perturbation] = []
    cases.append(
        Perturbation(
            name="exact_target",
            description="prediction equals the clean target exactly",
            predicted=list(ids),
            semantic_edits=0,
            focus_timestep=focus.ordinal,
            expected_realignment_index=2,
        )
    )

    substituted = list(ids)
    original = ids[edit_index]
    substitute = CONTENT_TOKEN_OFFSET + ((original - CONTENT_TOKEN_OFFSET + 1) % (vocabulary_size - CONTENT_TOKEN_OFFSET))
    substituted[edit_index] = substitute
    cases.append(
        Perturbation(
            name="content_substitution",
            description="one content occurrence replaced in place by another content token",
            predicted=substituted,
            semantic_edits=1,
            focus_timestep=focus.ordinal,
            expected_realignment_index=edit_index + 1,
        )
    )

    # Deletion + bounded left shift. Positions [edit_index, delimiter_index] take
    # the target's next token, which slides the delimiter to delimiter_index - 1
    # and duplicates the following group's first token into delimiter_index.
    deleted = list(ids)
    deleted[edit_index : delimiter_index + 1] = ids[edit_index + 1 : delimiter_index + 2]
    cases.append(
        Perturbation(
            name="content_deletion_local_shift",
            description=(
                "one content occurrence deleted; the group shifts one position left "
                "through its delimiter and re-aligns immediately after it"
            ),
            predicted=deleted,
            semantic_edits=2,
            focus_timestep=focus.ordinal,
            expected_realignment_index=delimiter_index + 1,
        )
    )

    # Insertion + bounded right shift, the mirror image of the deletion case.
    inserted = list(ids)
    inserted[edit_index + 1 : delimiter_index + 2] = ids[edit_index : delimiter_index + 1]
    cases.append(
        Perturbation(
            name="content_insertion_local_shift",
            description=(
                "one content occurrence duplicated; the group shifts one position right "
                "through its delimiter and re-aligns immediately after it"
            ),
            predicted=inserted,
            semantic_edits=2,
            focus_timestep=focus.ordinal,
            expected_realignment_index=delimiter_index + 2,
        )
    )

    displaced = list(ids)
    displaced[delimiter_index - 1] = DELIMITER_ID
    displaced[delimiter_index] = ids[delimiter_index - 1]
    cases.append(
        Perturbation(
            name="delimiter_displacement",
            description=(
                "the group's delimiter swaps places with its last content token, moving "
                "one correct content occurrence into the following group"
            ),
            predicted=displaced,
            semantic_edits=2,
            focus_timestep=focus.ordinal,
            expected_realignment_index=delimiter_index + 1,
        )
    )

    unbounded = list(ids)
    unbounded[edit_index:] = list(ids[edit_index + 1 :]) + [PAD_ID]
    cases.append(
        Perturbation(
            name="content_deletion_global_shift",
            description=(
                "control: the same single deletion with the left shift left UNBOUNDED to "
                "the end of the canvas"
            ),
            predicted=unbounded,
            semantic_edits=1,
            focus_timestep=focus.ordinal,
            expected_realignment_index=None,
        )
    )
    return cases


def deletion_offset_sweep(
    target: Sequence[int], focus: TimestepSpan
) -> dict[str, object]:
    """How many coordinates ONE bounded deletion moves, at every offset.

    The named cases above delete at the focus group's first content position,
    which is the maximal bounded shift. This sweeps the deletion across every
    offset in the same group and reports the distribution of resulting positional
    mismatches, so the report can quote a typical amplification rather than only
    a worst case.

    Closed form: deleting the occurrence at offset ``k`` and shifting left through
    the delimiter makes position ``i`` mismatch exactly when
    ``target[i] != target[i + 1]``, for ``i`` from ``k`` through the group's
    delimiter index. No pseudo-logits are needed to count that.

    Returns:
        ``content_length``, ``min``/``mean``/``max`` positional mismatches, and
        the mean mismatch count expressed per intended semantic edit (the bounded
        construction always costs exactly two: one deletion in this group and one
        duplication in the next).

    Calls: nothing.
    """

    ids = list(target)
    delimiter_index = focus.delimiter_index
    mismatches: list[int] = []
    for offset in range(focus.content_start, focus.content_end):
        mismatches.append(
            sum(
                1
                for index in range(offset, delimiter_index + 1)
                if ids[index] != ids[index + 1]
            )
        )
    if not mismatches:
        return {"content_length": 0}
    return {
        "content_length": focus.content_length,
        "min_positional_mismatches": min(mismatches),
        "mean_positional_mismatches": sum(mismatches) / len(mismatches),
        "max_positional_mismatches": max(mismatches),
        "mean_mismatch_amplification": sum(mismatches) / len(mismatches) / 2.0,
    }


def pseudo_logits(
    predicted: Sequence[int],
    *,
    vocabulary_size: int,
    confidence: float,
) -> torch.Tensor:
    """Build deterministic high-confidence logits that argmax to ``predicted``.

    Every row puts probability ``confidence`` on its predicted token and spreads
    ``1 - confidence`` uniformly over the remaining ``vocabulary_size - 1`` ids.
    Cross entropy at a position therefore has exactly two possible values --
    ``-ln(confidence)`` when the prediction is right and
    ``-ln((1-confidence)/(V-1))`` when it is wrong -- which makes every number in
    the geometry arm a closed form that unit tests can pin.

    Parameters:
        predicted: the hard token row the logits should argmax to.
        vocabulary_size: live vocabulary width.
        confidence: probability mass on the predicted token, in (0, 1).

    Returns:
        A ``[len(predicted), vocabulary_size]`` float32 logit tensor.

    Calls: nothing.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    off = math.log((1.0 - confidence) / (vocabulary_size - 1))
    on = math.log(confidence)
    logits = torch.full((len(predicted), vocabulary_size), off, dtype=torch.float32)
    logits[torch.arange(len(predicted)), torch.tensor(predicted, dtype=torch.long)] = on
    return logits


def evaluate_perturbation(
    perturbation: Perturbation,
    *,
    target: Sequence[int],
    layout: CanvasLayout,
    class_labels: Sequence[int],
    vocabulary_size: int,
    confidence: float,
    class_id_to_name: Mapping[int, str],
) -> dict[str, object]:
    """Score one controlled perturbation on one real clean canvas.

    Reports, for the whole canvas row excluding clamped ``[BOS]``:

      * ``semantic_edits`` -- what the construction intended;
      * ``positional_mismatches`` -- how many coordinates the objective sees as
        wrong;
      * ``unweighted_ce_nats`` -- the additive positional CE those mismatches buy,
        split into outcome / content / delimiter / ``[END]`` / semantic ``[PAD]``;
      * ``amplification`` ratios -- mismatches and excess CE per intended edit;
      * delimiter-local semantics -- multiset precision/recall/F1, count error,
        and ordered edit distance, computed by aligning the PREDICTED canvas's
        own delimiter groups against the target's by ordinal;
      * propagation -- first and last mismatching index, and whether the damage
        stopped where the construction intended.

    Calls: ``pseudo_logits``, ``parse_canvas_layout``, ``compare_timestep``.
    """

    targets = list(target)
    predicted = list(perturbation.predicted)
    logits = pseudo_logits(predicted, vocabulary_size=vocabulary_size, confidence=confidence)
    per_position_ce = F.cross_entropy(
        logits, torch.tensor(targets, dtype=torch.long), reduction="none"
    ).tolist()

    # Position 0 is clamped [BOS]: attended context, never a scored target.
    scored = list(range(1, layout.active_length))
    mismatches = [index for index in scored if predicted[index] != targets[index]]
    ce_total = float(sum(per_position_ce[index] for index in scored))

    per_class_ce: dict[str, float] = defaultdict(float)
    per_class_positions: dict[str, int] = defaultdict(int)
    per_class_mismatches: dict[str, int] = defaultdict(int)
    for index in scored:
        name = class_id_to_name.get(int(class_labels[index]), f"class_{class_labels[index]}")
        per_class_ce[name] += float(per_position_ce[index])
        per_class_positions[name] += 1
        if predicted[index] != targets[index]:
            per_class_mismatches[name] += 1

    # The exact-target floor: every scored position pays -ln(confidence).
    floor_ce = len(scored) * -math.log(confidence)
    excess_ce = ce_total - floor_ce
    per_mismatch_excess = -math.log((1.0 - confidence) / (vocabulary_size - 1)) + math.log(confidence)

    predicted_layout: dict[str, object]
    semantics_all: TimestepSemantics | None = None
    semantics_by_group: list[dict[str, object]] = []
    try:
        parsed_prediction = parse_canvas_layout(predicted, layout.active_length)
        predicted_layout = {
            "parsed": True,
            "delimiter_count": len(parsed_prediction.delimiter_indices),
            "timestep_count": len(parsed_prediction.timesteps),
            "terminated": parsed_prediction.terminated,
        }
        pooled_overlap = 0
        pooled_predicted = 0
        pooled_target = 0
        pooled_edit = 0
        pooled_count_error = 0
        pooled_cells = 0
        exact_groups = 0
        compared = min(len(parsed_prediction.timesteps), len(layout.timesteps))
        for ordinal in range(compared):
            predicted_span = parsed_prediction.timesteps[ordinal]
            target_span = layout.timesteps[ordinal]
            semantics = compare_timestep(
                predicted[predicted_span.content_start : predicted_span.content_end],
                targets[target_span.content_start : target_span.content_end],
            )
            pooled_overlap += semantics.overlap
            pooled_predicted += semantics.predicted_total
            pooled_target += semantics.target_total
            pooled_edit += semantics.edit_distance
            pooled_count_error += semantics.count_error
            pooled_cells += semantics.count_cells
            exact_groups += int(semantics.exact_multiset)
            if semantics.edit_distance:
                semantics_by_group.append(
                    {
                        "timestep_ordinal": ordinal,
                        "edit_distance": semantics.edit_distance,
                        "count_error": semantics.count_error,
                    }
                )
        semantics_all = TimestepSemantics(
            overlap=pooled_overlap,
            predicted_total=pooled_predicted,
            target_total=pooled_target,
            exact_multiset=exact_groups == compared,
            count_error=pooled_count_error,
            count_cells=pooled_cells,
            edit_distance=pooled_edit,
        )
        predicted_layout["compared_timesteps"] = compared
        predicted_layout["exact_multiset_timesteps"] = exact_groups

        # Focus-scoped view. The pooled whole-canvas numbers above are dominated
        # by the dozens of untouched groups, so a one-group edit is invisible in
        # them; this narrows to the edited group and the one it can spill into,
        # which is where the whole hypothesis lives.
        focus_ordinals = [
            ordinal
            for ordinal in (perturbation.focus_timestep, perturbation.focus_timestep + 1)
            if ordinal < compared
        ]
        focus_overlap = focus_predicted = focus_target = 0
        focus_edit = focus_error = focus_cells = 0
        for ordinal in focus_ordinals:
            predicted_span = parsed_prediction.timesteps[ordinal]
            target_span = layout.timesteps[ordinal]
            semantics = compare_timestep(
                predicted[predicted_span.content_start : predicted_span.content_end],
                targets[target_span.content_start : target_span.content_end],
            )
            focus_overlap += semantics.overlap
            focus_predicted += semantics.predicted_total
            focus_target += semantics.target_total
            focus_edit += semantics.edit_distance
            focus_error += semantics.count_error
            focus_cells += semantics.count_cells
        focus_start = layout.timesteps[perturbation.focus_timestep].content_start
        focus_stop = layout.timesteps[focus_ordinals[-1]].delimiter_index
        predicted_layout["focus"] = {
            "timestep_ordinals": focus_ordinals,
            "positional_mismatches": sum(
                1
                for index in range(focus_start, focus_stop + 1)
                if predicted[index] != targets[index]
            ),
            "positions": focus_stop + 1 - focus_start,
            "overlap": focus_overlap,
            "predicted_total": focus_predicted,
            "target_total": focus_target,
            "multiset_precision": _ratio(focus_overlap, focus_predicted),
            "multiset_recall": _ratio(focus_overlap, focus_target),
            "multiset_f1": _f1(focus_overlap, focus_predicted, focus_target),
            "count_error": focus_error,
            "count_cells": focus_cells,
            "edit_distance": focus_edit,
        }
    except ValueError as error:
        predicted_layout = {"parsed": False, "reason": str(error)}

    first_mismatch = mismatches[0] if mismatches else None
    last_mismatch = mismatches[-1] if mismatches else None
    realignment = perturbation.expected_realignment_index
    stops_where_intended = (
        None
        if realignment is None
        else (last_mismatch is None or last_mismatch < realignment)
    )

    result: dict[str, object] = {
        "name": perturbation.name,
        "description": perturbation.description,
        "focus_timestep": perturbation.focus_timestep,
        "semantic_edits": perturbation.semantic_edits,
        "scored_positions": len(scored),
        "positional_mismatches": len(mismatches),
        "unweighted_ce_nats": ce_total,
        "unweighted_ce_floor_nats": floor_ce,
        "excess_ce_over_exact_nats": excess_ce,
        "per_position_excess_ce_nats": per_mismatch_excess,
        "mismatch_amplification": (
            None if perturbation.semantic_edits == 0 else len(mismatches) / perturbation.semantic_edits
        ),
        "excess_ce_amplification": (
            None
            if perturbation.semantic_edits == 0
            else excess_ce / (perturbation.semantic_edits * per_mismatch_excess)
        ),
        "per_class": {
            name: {
                "ce_nats": per_class_ce[name],
                "positions": per_class_positions[name],
                "mismatches": per_class_mismatches[name],
            }
            for name in sorted(per_class_ce)
        },
        "predicted_layout": predicted_layout,
        "first_mismatch_index": first_mismatch,
        "last_mismatch_index": last_mismatch,
        "expected_realignment_index": realignment,
        "penalty_stops_where_intended": stops_where_intended,
        "penalty_span": None if first_mismatch is None else last_mismatch - first_mismatch + 1,
    }
    if semantics_all is not None:
        result["delimiter_local"] = {
            "multiset_precision": _ratio(semantics_all.overlap, semantics_all.predicted_total),
            "multiset_recall": _ratio(semantics_all.overlap, semantics_all.target_total),
            "multiset_f1": _f1(
                semantics_all.overlap, semantics_all.predicted_total, semantics_all.target_total
            ),
            "total_count_error": semantics_all.count_error,
            "count_cells": semantics_all.count_cells,
            "count_mae": _ratio(semantics_all.count_error, semantics_all.count_cells),
            "edit_distance": semantics_all.edit_distance,
            "damaged_timesteps": semantics_by_group[:8],
        }
    return result


def run_geometry_experiment(
    batches: Sequence[DiffusionBatch],
    *,
    vocabulary_size: int,
    confidence: float,
    class_id_to_name: Mapping[int, str],
    max_canvases: int,
) -> dict[str, object]:
    """Run the model-independent geometry arm over real clean target canvases.

    Parameters:
        batches: collated batches whose ``target_canvas`` rows supply the clean
            canvases. No model, checkpoint, or GPU is involved.
        vocabulary_size: live vocabulary width.
        confidence: pseudo-logit confidence (see :func:`pseudo_logits`).
        class_id_to_name: the run's active class taxonomy.
        max_canvases: how many canvases to score.

    Returns:
        A report dict with per-canvas results and a pooled summary keyed by
        perturbation name.

    Calls: ``parse_canvas_layout``, ``build_perturbations``,
    ``evaluate_perturbation``.
    """

    per_canvas: list[dict[str, object]] = []
    for batch in batches:
        for row in range(batch.target_canvas.shape[0]):
            if len(per_canvas) >= max_canvases:
                break
            active_length = int(batch.canvas_attention_mask[row].sum().item())
            target = batch.target_canvas[row, :active_length].tolist()
            class_labels = batch.class_labels[row, :active_length].tolist()
            layout = parse_canvas_layout(target, active_length)
            try:
                focus = select_focus_timestep(layout)
                perturbations = build_perturbations(
                    target, layout, vocabulary_size=vocabulary_size
                )
            except ValueError:
                continue
            per_canvas.append(
                {
                    "canvas_index": len(per_canvas),
                    "active_length": active_length,
                    "timestep_count": len(layout.timesteps),
                    "terminated": layout.terminated,
                    "semantic_pad_positions": len(layout.semantic_pad_indices),
                    "focus_timestep": focus.ordinal,
                    "focus_content_length": focus.content_length,
                    "deletion_offset_sweep": deletion_offset_sweep(target, focus),
                    "cases": [
                        evaluate_perturbation(
                            perturbation,
                            target=target,
                            layout=layout,
                            class_labels=class_labels,
                            vocabulary_size=vocabulary_size,
                            confidence=confidence,
                            class_id_to_name=class_id_to_name,
                        )
                        for perturbation in perturbations
                    ],
                }
            )
        if len(per_canvas) >= max_canvases:
            break

    pooled: dict[str, dict[str, object]] = {}
    for canvas in per_canvas:
        for case in canvas["cases"]:  # type: ignore[index]
            entry = pooled.setdefault(
                case["name"],
                {
                    "description": case["description"],
                    "canvases": 0,
                    "semantic_edits": 0,
                    "positional_mismatches": 0,
                    "excess_ce_nats": 0.0,
                    "scored_positions": 0,
                    "penalty_span_sum": 0,
                    "stopped_where_intended": 0,
                    "stop_checked": 0,
                    "multiset_f1_sum": 0.0,
                    "edit_distance": 0,
                    "focus_positional_mismatches": 0,
                    "focus_positions": 0,
                    "focus_overlap": 0,
                    "focus_predicted": 0,
                    "focus_target": 0,
                    "focus_edit_distance": 0,
                    "focus_count_error": 0,
                },
            )
            entry["canvases"] += 1
            entry["semantic_edits"] += case["semantic_edits"]
            entry["positional_mismatches"] += case["positional_mismatches"]
            entry["excess_ce_nats"] += case["excess_ce_over_exact_nats"]
            entry["scored_positions"] += case["scored_positions"]
            entry["penalty_span_sum"] += case["penalty_span"] or 0
            if case["penalty_stops_where_intended"] is not None:
                entry["stop_checked"] += 1
                entry["stopped_where_intended"] += int(case["penalty_stops_where_intended"])
            local = case.get("delimiter_local")
            if local is not None:
                entry["multiset_f1_sum"] += local["multiset_f1"] or 0.0
                entry["edit_distance"] += local["edit_distance"]
            focus = case["predicted_layout"].get("focus")  # type: ignore[union-attr]
            if focus is not None:
                entry["focus_positional_mismatches"] += focus["positional_mismatches"]
                entry["focus_positions"] += focus["positions"]
                entry["focus_overlap"] += focus["overlap"]
                entry["focus_predicted"] += focus["predicted_total"]
                entry["focus_target"] += focus["target_total"]
                entry["focus_edit_distance"] += focus["edit_distance"]
                entry["focus_count_error"] += focus["count_error"]

    for name, entry in pooled.items():
        edits = entry["semantic_edits"]
        entry["mismatch_amplification"] = (
            None if edits == 0 else entry["positional_mismatches"] / edits
        )
        entry["mean_penalty_span"] = entry["penalty_span_sum"] / max(1, entry["canvases"])
        entry["mean_multiset_f1"] = entry["multiset_f1_sum"] / max(1, entry["canvases"])
        entry["mean_delimiter_local_edit_distance"] = entry["edit_distance"] / max(
            1, entry["canvases"]
        )
        # Focus-scoped ratios: the edited group plus the group it can spill into.
        entry["focus_multiset_f1"] = _f1(
            entry["focus_overlap"], entry["focus_predicted"], entry["focus_target"]
        )
        entry["focus_mismatch_amplification"] = (
            None if edits == 0 else entry["focus_positional_mismatches"] / edits
        )
        entry["focus_edit_distance_per_canvas"] = entry["focus_edit_distance"] / max(
            1, entry["canvases"]
        )

    sweeps = [
        canvas["deletion_offset_sweep"]
        for canvas in per_canvas
        if canvas["deletion_offset_sweep"].get("content_length")  # type: ignore[union-attr]
    ]
    offset_sweep = {}
    if sweeps:
        offset_sweep = {
            "canvases": len(sweeps),
            "mean_focus_content_length": sum(s["content_length"] for s in sweeps) / len(sweeps),
            "mean_positional_mismatches": sum(
                s["mean_positional_mismatches"] for s in sweeps
            )
            / len(sweeps),
            "min_positional_mismatches": min(s["min_positional_mismatches"] for s in sweeps),
            "max_positional_mismatches": max(s["max_positional_mismatches"] for s in sweeps),
            "mean_mismatch_amplification": sum(
                s["mean_mismatch_amplification"] for s in sweeps
            )
            / len(sweeps),
        }
    return {
        "deletion_offset_sweep": offset_sweep,
        "pseudo_logit_confidence": confidence,
        "per_position_correct_ce_nats": -math.log(confidence),
        "per_position_wrong_ce_nats": -math.log((1.0 - confidence) / (vocabulary_size - 1)),
        "canvases_scored": len(per_canvas),
        "pooled": pooled,
        "per_canvas": per_canvas,
    }


# ---------------------------------------------------------------------------
# Pooled accumulation for the model arm
# ---------------------------------------------------------------------------


class PooledCounters:
    """Additive counters pooled across every scored position before ratios.

    WHY additive counters instead of per-window means: support sizes differ
    wildly between windows (a 12-timestep window and a 60-timestep window are not
    equally informative), so averaging per-window ratios would silently reweight
    the corpus. Every ratio in the report is formed once, at the end, from pooled
    numerators and denominators.
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = defaultdict(float)

    def add(self, key: str, value: float) -> None:
        self.values[key] += float(value)

    def merge(self, other: "PooledCounters") -> None:
        for key, value in other.values.items():
            self.values[key] += value

    def get(self, key: str) -> float:
        return self.values.get(key, 0.0)


def finalize_counters(counters: PooledCounters) -> dict[str, object]:
    """Turn one slice's pooled counters into the reported ratios.

    Calls: ``_ratio``, ``_f1``.
    """

    values = counters.values
    scored = values.get("scored_positions", 0.0)
    noised = values.get("noised_positions", 0.0)
    content = values.get("content_positions", 0.0)
    timesteps = values.get("timesteps", 0.0)
    report: dict[str, object] = {
        "windows": int(values.get("windows", 0.0)),
        "scored_positions": int(scored),
        "weighted_objective_nats": _ratio(
            values.get("weighted_ce_sum", 0.0), values.get("weight_sum", 0.0)
        ),
        "unweighted_ce_nats": _ratio(values.get("unweighted_ce_sum", 0.0), scored),
        "noised_positions": int(noised),
        "noised_argmax_accuracy": _ratio(values.get("noised_correct", 0.0), noised),
        "preserved_positions": int(scored - noised),
        "preserved_argmax_accuracy": _ratio(
            values.get("preserved_correct", 0.0), scored - noised
        ),
    }
    if content:
        positional = values.get("content_ce_sum", 0.0)
        oracle = values.get("oracle_ce_sum", 0.0)
        oracle_positions = values.get("oracle_positions", 0.0)
        report["content_positions"] = int(content)
        report["content_positional_ce_nats"] = _ratio(positional, content)
        if oracle_positions:
            oracle_positional = values.get("oracle_positional_ce_sum", 0.0)
            report["oracle_covered_positions"] = int(oracle_positions)
            report["oracle_aligned_ce_nats"] = _ratio(oracle, oracle_positions)
            report["oracle_matched_positional_ce_nats"] = _ratio(
                oracle_positional, oracle_positions
            )
            report["alignment_gap_nats"] = _ratio(
                oracle_positional - oracle, oracle_positions
            )
            report["alignment_gap_fraction"] = _ratio(
                oracle_positional - oracle, oracle_positional
            )
    if timesteps:
        overlap = values.get("multiset_overlap", 0.0)
        report["timesteps"] = int(timesteps)
        report["exact_multiset_rate"] = _ratio(values.get("exact_multiset", 0.0), timesteps)
        report["multiset_precision"] = _ratio(overlap, values.get("multiset_predicted", 0.0))
        report["multiset_recall"] = _ratio(overlap, values.get("multiset_target", 0.0))
        report["multiset_f1"] = _f1(
            overlap, values.get("multiset_predicted", 0.0), values.get("multiset_target", 0.0)
        )
        report["count_mae"] = _ratio(
            values.get("count_error", 0.0), values.get("count_cells", 0.0)
        )
        report["delimiter_local_edit_distance_per_timestep"] = _ratio(
            values.get("edit_distance", 0.0), timesteps
        )
        rare_overlap = values.get("rare_multiset_overlap", 0.0)
        report["multiset_f1_excluding_dominant"] = _f1(
            rare_overlap,
            values.get("rare_multiset_predicted", 0.0),
            values.get("rare_multiset_target", 0.0),
        )
        report["exact_multiset_rate_excluding_dominant"] = _ratio(
            values.get("rare_exact_multiset", 0.0), timesteps
        )
    return report


# ---------------------------------------------------------------------------
# Experiment B/C: real checkpoint logits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowIdentity:
    """Portable identity for one scored replay window."""

    replay_id: str
    perspective: str
    start_timestep: int
    end_timestep: int

    def as_key(self) -> str:
        return f"{self.replay_id}:{self.perspective}:{self.start_timestep}-{self.end_timestep}"


def _forward_canvas_logits(
    model: torch.nn.Module,
    *,
    input_token_ids: torch.Tensor,
    input_attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    input_features: InputFeatures,
    noised_canvas: torch.Tensor,
    canvas_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run ONE denoiser forward pass and return the canvas-region logits.

    Self-conditioning is passed as ``None``, which the embedding turns into a
    zero signal. That is exactly what the iterative sampler feeds on its FIRST
    step, so this single pass is the sampler's first denoising estimate rather
    than an unrelated configuration.

    Returns: ``[batch, canvas_len, vocab]`` float32 logits.
    """

    with torch.no_grad():
        output = model(
            input_token_ids=input_token_ids,
            canvas_token_ids=noised_canvas,
            input_attention_mask=input_attention_mask,
            canvas_attention_mask=canvas_attention_mask,
            input_features=input_features,
            input_lengths=input_lengths,
            canvas_self_conditioning=None,
        )
    return output.logits[:, input_token_ids.shape[1] :, :].float()


def _shuffle_input_rows(batch: DiffusionBatch) -> dict[str, object]:
    """Return input tensors with rows rotated by one, canvas untouched.

    This is the conditioning-use control: the canvas the model is asked to
    denoise is IDENTICAL, but the clamped input belongs to a different replay
    window. A model whose t=1 prediction is driven by a learned marginal rather
    than by its conditioning will barely notice.

    Calls: nothing.
    """

    order = torch.roll(torch.arange(batch.target_canvas.shape[0]), shifts=1)
    features = batch.input_features
    return {
        "input_token_ids": batch.input_token_ids[order],
        "input_attention_mask": batch.input_attention_mask[order],
        "input_lengths": batch.input_lengths[order],
        "input_features": InputFeatures(
            continuous_values=features.continuous_values[order],
            continuous_validity=features.continuous_validity[order],
            categorical_values=features.categorical_values[order],
            allegiance_values=features.allegiance_values[order],
            feature_mask=features.feature_mask[order],
        ),
    }


def _strip_enemy_input(batch: DiffusionBatch) -> dict[str, object]:
    """Return input tensors with enemy content removed, structure retained.

    Enemy content tokens are exactly the positions whose allegiance feature is
    ``-1``. Delimiters, the terminal ``[EOS]``, and every self record keep their
    place, so the surviving input is still a legal interleaved input sequence --
    it has simply lost the observed enemy evidence. Rows are re-packed
    LEFT-padded, matching the collater's convention.

    Calls: nothing.
    """

    features = batch.input_features
    allegiance = features.allegiance_values.squeeze(-1)
    keep = batch.input_attention_mask & ~(
        (allegiance < -0.5) & features.feature_mask
    )
    rows = batch.input_token_ids.shape[0]
    lengths = keep.sum(dim=1)
    width = int(lengths.max().item())
    device = batch.input_token_ids.device

    token_ids = torch.full((rows, width), PAD_ID, dtype=torch.long, device=device)
    attention = torch.zeros((rows, width), dtype=torch.bool, device=device)
    continuous = torch.zeros(
        (rows, width, features.continuous_values.shape[-1]),
        dtype=features.continuous_values.dtype,
        device=device,
    )
    validity = torch.zeros(
        (rows, width, features.continuous_validity.shape[-1]),
        dtype=features.continuous_validity.dtype,
        device=device,
    )
    categorical = torch.zeros(
        (rows, width, features.categorical_values.shape[-1]),
        dtype=features.categorical_values.dtype,
        device=device,
    )
    allegiance_out = torch.zeros(
        (rows, width, 1), dtype=features.allegiance_values.dtype, device=device
    )
    feature_mask = torch.zeros((rows, width), dtype=torch.bool, device=device)

    for row in range(rows):
        indices = torch.nonzero(keep[row], as_tuple=False).squeeze(-1)
        length = int(indices.numel())
        if length == 0:
            continue
        start = width - length
        token_ids[row, start:] = batch.input_token_ids[row, indices]
        attention[row, start:] = True
        continuous[row, start:] = features.continuous_values[row, indices]
        validity[row, start:] = features.continuous_validity[row, indices]
        categorical[row, start:] = features.categorical_values[row, indices]
        allegiance_out[row, start:] = features.allegiance_values[row, indices]
        feature_mask[row, start:] = features.feature_mask[row, indices]

    return {
        "input_token_ids": token_ids,
        "input_attention_mask": attention,
        "input_lengths": lengths,
        "input_features": InputFeatures(
            continuous_values=continuous,
            continuous_validity=validity,
            categorical_values=categorical,
            allegiance_values=allegiance_out,
            feature_mask=feature_mask,
        ),
    }


def score_batch(
    *,
    canvas_logits: torch.Tensor,
    batch: DiffusionBatch,
    changed_positions: torch.Tensor,
    class_weights: torch.Tensor,
    class_id_to_name: Mapping[int, str],
    identities: Sequence[WindowIdentity],
    noise_level: float,
    dominant_ids: frozenset[int],
    slices: dict[tuple[str, str], PooledCounters],
    token_counts: dict[str, np.ndarray],
    per_window_rows: list[dict[str, object]],
    compute_oracle: bool,
    oracle_max_span: int,
    vocab_size: int,
    overall_facet: str = "overall",
) -> None:
    """Score one batch of canvas logits into the pooled slice accumulators.

    Everything is computed from the SAME logits and targets so the positional and
    aligned views are directly comparable. Slice keys are ``(facet, value)``
    pairs; every scored position contributes to the ``overall_facet`` slice and to
    one value of each other facet it belongs to.

    ``overall_facet`` is what keeps the Experiment C control arms separate: the
    controls reuse this whole scoring path unchanged and simply route their
    headline slice to ``control:<name>`` instead of ``overall``.

    Facets populated here: ``overall_facet``, ``t``, ``perspective``, ``replay``,
    ``future_distance``, ``delimiter_count_correct``, and ``class``.

    Calls: ``parse_canvas_layout``, ``predicted_structure``, ``compare_timestep``,
    ``oracle_aligned_content_cost``.
    """

    targets = batch.target_canvas
    scored_mask = batch.canvas_loss_mask
    log_probabilities = torch.log_softmax(canvas_logits, dim=-1)
    per_position_ce = F.nll_loss(
        log_probabilities.transpose(1, 2), targets, reduction="none"
    )
    predicted_tokens = canvas_logits.argmax(dim=-1)
    correct = predicted_tokens == targets
    weights = class_weights.to(canvas_logits.device)[batch.class_labels.clamp_min(0)]
    weights = weights * scored_mask.to(weights.dtype)

    rows = targets.shape[0]
    ce_cpu = per_position_ce.detach().cpu().numpy()
    log_probabilities_cpu = log_probabilities.detach().cpu().numpy()
    predicted_cpu = predicted_tokens.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    scored_cpu = scored_mask.detach().cpu().numpy().astype(bool)
    changed_cpu = changed_positions.detach().cpu().numpy().astype(bool)
    correct_cpu = correct.detach().cpu().numpy()
    weights_cpu = weights.detach().cpu().numpy()
    labels_cpu = batch.class_labels.detach().cpu().numpy()
    distances_cpu = batch.canvas_prediction_distances.detach().cpu().numpy()
    perspectives = batch.perspective_ids.detach().cpu().tolist()

    def counters(facet: str, value: str) -> PooledCounters:
        return slices.setdefault((facet, value), PooledCounters())

    for row in range(rows):
        identity = identities[row]
        active_length = int(batch.canvas_attention_mask[row].sum().item())
        target_row = targets_cpu[row, :active_length].tolist()
        predicted_row = predicted_cpu[row, :active_length].tolist()
        layout = parse_canvas_layout(target_row, active_length)
        structure = predicted_structure(predicted_row, active_length)
        delimiter_correct = structure["delimiter_count"] == len(layout.delimiter_indices)

        perspective = "p1" if perspectives[row] == 1 else "p2"
        row_facets = [
            (overall_facet, "all"),
            ("t", f"{noise_level:.2f}"),
            ("perspective", perspective),
            ("replay", identity.replay_id),
            (
                "delimiter_count_correct",
                "true" if delimiter_correct else "false",
            ),
        ]

        row_counters = [counters(facet, value) for facet, value in row_facets]
        for counter in row_counters:
            counter.add("windows", 1)

        scored_indices = np.flatnonzero(scored_cpu[row, :active_length])
        row_ce = ce_cpu[row]
        row_weight = weights_cpu[row]
        row_changed = changed_cpu[row]
        row_correct = correct_cpu[row]

        scored_ce = float(row_ce[scored_indices].sum())
        weighted_ce = float((row_ce[scored_indices] * row_weight[scored_indices]).sum())
        weight_sum = float(row_weight[scored_indices].sum())
        noised_indices = scored_indices[row_changed[scored_indices]]
        preserved_indices = scored_indices[~row_changed[scored_indices]]
        for counter in row_counters:
            counter.add("scored_positions", len(scored_indices))
            counter.add("unweighted_ce_sum", scored_ce)
            counter.add("weighted_ce_sum", weighted_ce)
            counter.add("weight_sum", weight_sum)
            counter.add("noised_positions", len(noised_indices))
            counter.add("noised_correct", float(row_correct[noised_indices].sum()))
            counter.add("preserved_correct", float(row_correct[preserved_indices].sum()))

        # Per-class CE (the structural half of the report: outcome, delimiter,
        # [END], semantic [PAD], and the three content classes).
        for class_id, class_name in class_id_to_name.items():
            class_indices = scored_indices[labels_cpu[row, scored_indices] == class_id]
            if class_indices.size == 0:
                continue
            class_counter = counters("class", class_name)
            class_counter.add("scored_positions", class_indices.size)
            class_counter.add("unweighted_ce_sum", float(row_ce[class_indices].sum()))
            class_counter.add(
                "weighted_ce_sum", float((row_ce[class_indices] * row_weight[class_indices]).sum())
            )
            class_counter.add("weight_sum", float(row_weight[class_indices].sum()))
            class_noised = class_indices[row_changed[class_indices]]
            class_counter.add("noised_positions", class_noised.size)
            class_counter.add("noised_correct", float(row_correct[class_noised].sum()))
            class_counter.add(
                "preserved_correct",
                float(row_correct[class_indices[~row_changed[class_indices]]].sum()),
            )
            # Same class slice, but split by noise level so the per-class trend
            # along the corruption axis stays visible.
            level_counter = counters("class_by_t", f"{class_name}@{noise_level:.2f}")
            level_counter.add("scored_positions", class_indices.size)
            level_counter.add("unweighted_ce_sum", float(row_ce[class_indices].sum()))
            level_counter.add("noised_positions", class_noised.size)
            level_counter.add("noised_correct", float(row_correct[class_noised].sum()))

        # Future-distance buckets over the enemy-future content positions.
        for bucket_name, (minimum, maximum) in FUTURE_DISTANCE_BUCKETS.items():
            row_distances = distances_cpu[row, scored_indices]
            selector = row_distances >= minimum
            if maximum is not None:
                selector &= row_distances <= maximum
            bucket_indices = scored_indices[selector]
            if bucket_indices.size == 0:
                continue
            bucket_counter = counters("future_distance", bucket_name)
            bucket_counter.add("scored_positions", bucket_indices.size)
            bucket_counter.add("unweighted_ce_sum", float(row_ce[bucket_indices].sum()))
            bucket_counter.add(
                "noised_positions", int(row_changed[bucket_indices].sum())
            )
            bucket_counter.add(
                "noised_correct",
                float(row_correct[bucket_indices][row_changed[bucket_indices]].sum()),
            )

        # Structural quantities the investigation compares directly.
        outcome_ce = float(row_ce[1])
        outcome_pair_mass = float(
            np.exp(log_probabilities_cpu[row, 1, WIN_ID])
            + np.exp(log_probabilities_cpu[row, 1, LOSS_ID])
        )
        delimiter_drift = _delimiter_drift(
            predicted_row, list(layout.delimiter_indices)
        )
        for counter in row_counters:
            counter.add("outcome_ce_sum", outcome_ce)
            counter.add("outcome_positions", 1)
            counter.add("outcome_pair_mass_sum", outcome_pair_mass)
            counter.add("outcome_correct", float(row_correct[1]))
            counter.add("target_active_length", layout.active_length)
            counter.add("predicted_active_length", structure["active_length"])
            counter.add("target_delimiters", len(layout.delimiter_indices))
            counter.add("predicted_delimiters", structure["delimiter_count"])
            counter.add("delimiter_count_correct", float(delimiter_correct))
            counter.add("delimiter_drift_abs_sum", delimiter_drift["abs_sum"])
            counter.add("delimiter_drift_signed_sum", delimiter_drift["signed_sum"])
            counter.add("delimiter_drift_pairs", delimiter_drift["pairs"])

        # ---- Delimiter-local semantics + oracle aligned content score --------
        window_content_positions = 0
        window_content_ce = 0.0
        window_oracle_ce = 0.0
        window_oracle_positional = 0.0
        window_oracle_positions = 0
        for span in layout.timesteps:
            if span.content_length == 0:
                continue
            span_slice = slice(span.content_start, span.content_end)
            span_target = target_row[span_slice]
            span_predicted = predicted_row[span_slice]
            semantics = compare_timestep(span_predicted, span_target)
            rare = compare_timestep(span_predicted, span_target, exclude=dominant_ids)
            span_ce = float(row_ce[span.content_start : span.content_end].sum())
            window_content_positions += span.content_length
            window_content_ce += span_ce

            for counter in row_counters:
                counter.add("timesteps", 1)
                counter.add("multiset_overlap", semantics.overlap)
                counter.add("multiset_predicted", semantics.predicted_total)
                counter.add("multiset_target", semantics.target_total)
                counter.add("exact_multiset", float(semantics.exact_multiset))
                counter.add("count_error", semantics.count_error)
                counter.add("count_cells", semantics.count_cells)
                counter.add("edit_distance", semantics.edit_distance)
                counter.add("rare_multiset_overlap", rare.overlap)
                counter.add("rare_multiset_predicted", rare.predicted_total)
                counter.add("rare_multiset_target", rare.target_total)
                counter.add("rare_exact_multiset", float(rare.exact_multiset))

            if compute_oracle and span.content_length <= oracle_max_span:
                oracle = oracle_aligned_content_cost(
                    log_probabilities_cpu[row, span.content_start : span.content_end],
                    span_target,
                )
                window_oracle_ce += oracle
                window_oracle_positional += span_ce
                window_oracle_positions += span.content_length

        for counter in row_counters:
            counter.add("content_positions", window_content_positions)
            counter.add("content_ce_sum", window_content_ce)
            counter.add("oracle_ce_sum", window_oracle_ce)
            counter.add("oracle_positional_ce_sum", window_oracle_positional)
            counter.add("oracle_positions", window_oracle_positions)

        per_window_rows.append(
            {
                "t": round(noise_level, 4),
                "window": identity.as_key(),
                "replay_id": identity.replay_id,
                "perspective": perspective,
                "scored_positions": int(len(scored_indices)),
                "unweighted_ce_nats": _ratio(scored_ce, len(scored_indices)),
                "target_delimiters": len(layout.delimiter_indices),
                "predicted_delimiters": structure["delimiter_count"],
                "target_active_length": layout.active_length,
                "predicted_active_length": structure["active_length"],
                "content_positional_ce_nats": _ratio(
                    window_content_ce, window_content_positions
                ),
                "oracle_aligned_ce_nats": _ratio(
                    window_oracle_ce, window_oracle_positions
                ),
                "alignment_gap_nats": _ratio(
                    window_oracle_positional - window_oracle_ce, window_oracle_positions
                ),
            }
        )

    # Per-token TP/predicted/target counts on genuinely noised positions, kept per
    # noise level so macro-F1 can be reported by t. Rows follow the production
    # TOKEN_COUNT_ROWS order so `_macro_f1_from_counts` can consume them directly.
    key = f"{noise_level:.2f}"
    matrix = token_counts.setdefault(key, np.zeros((3, vocab_size), dtype=np.int64))
    active = scored_cpu & changed_cpu
    flat_target = targets_cpu[active]
    flat_predicted = predicted_cpu[active]
    flat_correct = correct_cpu[active]
    np.add.at(matrix[0], flat_target[flat_correct], 1)
    np.add.at(matrix[1], flat_predicted, 1)
    np.add.at(matrix[2], flat_target, 1)


def _delimiter_drift(
    predicted_row: Sequence[int], target_delimiters: Sequence[int]
) -> dict[str, float]:
    """Compare the i-th predicted delimiter index against the i-th target one.

    Only the first ``min(count)`` delimiters are paired; a missing or surplus
    delimiter is reported through the delimiter-count comparison instead, so a
    single count error cannot masquerade as an enormous drift.
    """

    predicted_delimiters = [
        index for index, token in enumerate(predicted_row) if index >= 2 and token == DELIMITER_ID
    ]
    pairs = min(len(predicted_delimiters), len(target_delimiters))
    abs_sum = 0.0
    signed_sum = 0.0
    for ordinal in range(pairs):
        drift = predicted_delimiters[ordinal] - target_delimiters[ordinal]
        abs_sum += abs(drift)
        signed_sum += drift
    return {"abs_sum": abs_sum, "signed_sum": signed_sum, "pairs": float(pairs)}


def persistence_and_unigram_baselines(
    batches: Sequence[DiffusionBatch],
    *,
    unigram_counts: np.ndarray | None,
    dominant_ids: frozenset[int],
) -> dict[str, object]:
    """Two model-free comparators for the delimiter-local multiset metrics.

    * ``previous_timestep_persistence`` -- predict each timestep's content
      multiset by copying the PREVIOUS ground-truth timestep's multiset. This is
      a strong oracle-flavoured baseline (it reads ground truth one step back)
      and is reported as such: it bounds how much of a good multiset score is
      simply SC2 state being slow-moving.
    * ``train_unigram_constant`` -- fill every content slot with the single most
      frequent train-split content token. Leakage-free: the histogram comes from
      the train split only, and it is evaluated on the held-out split.

    Parameters:
        batches: the scored batches (targets only; no model output is used).
        unigram_counts: train-split content-token histogram, or None to skip.
        dominant_ids: probe/nexus ids, for the excluding-dominant variant.

    Returns:
        A dict of pooled multiset metrics per baseline.

    Calls: ``parse_canvas_layout``, ``compare_timestep``, ``content_counts``.
    """

    persistence = PooledCounters()
    unigram = PooledCounters()
    top_token = (
        int(np.argmax(unigram_counts)) if unigram_counts is not None and unigram_counts.sum() else None
    )
    for batch in batches:
        for row in range(batch.target_canvas.shape[0]):
            active_length = int(batch.canvas_attention_mask[row].sum().item())
            target_row = batch.target_canvas[row, :active_length].tolist()
            layout = parse_canvas_layout(target_row, active_length)
            previous: list[int] = []
            for span in layout.timesteps:
                current = target_row[span.content_start : span.content_end]
                if span.content_length == 0:
                    continue
                # Persistence predicts the previous timestep's content, padded or
                # truncated to this timestep's slot count so the comparison stays
                # a like-for-like multiset comparison.
                predicted = list(previous[: span.content_length])
                predicted += [PAD_ID] * (span.content_length - len(predicted))
                semantics = compare_timestep(predicted, current)
                rare = compare_timestep(predicted, current, exclude=dominant_ids)
                _accumulate_multiset(persistence, semantics, rare)
                if top_token is not None:
                    constant = [top_token] * span.content_length
                    constant_semantics = compare_timestep(constant, current)
                    constant_rare = compare_timestep(constant, current, exclude=dominant_ids)
                    _accumulate_multiset(unigram, constant_semantics, constant_rare)
                previous = list(current)
    result: dict[str, object] = {
        "previous_timestep_persistence": finalize_counters(persistence),
    }
    if top_token is not None:
        result["train_unigram_constant"] = finalize_counters(unigram)
        result["train_unigram_top_token_id"] = top_token
    return result


def _accumulate_multiset(
    counters: PooledCounters, semantics: TimestepSemantics, rare: TimestepSemantics
) -> None:
    """Add one timestep's multiset comparison into a pooled counter set."""

    counters.add("timesteps", 1)
    counters.add("multiset_overlap", semantics.overlap)
    counters.add("multiset_predicted", semantics.predicted_total)
    counters.add("multiset_target", semantics.target_total)
    counters.add("exact_multiset", float(semantics.exact_multiset))
    counters.add("count_error", semantics.count_error)
    counters.add("count_cells", semantics.count_cells)
    counters.add("edit_distance", semantics.edit_distance)
    counters.add("rare_multiset_overlap", rare.overlap)
    counters.add("rare_multiset_predicted", rare.predicted_total)
    counters.add("rare_multiset_target", rare.target_total)
    counters.add("rare_exact_multiset", float(rare.exact_multiset))


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------


def resolve_split_replays(
    replay_paths: Sequence[str], *, config: ProjectConfig, split_name: str
) -> list[str]:
    """Return the replay paths belonging to one config-derived split.

    Reuses the training pipeline's own selection helpers rather than
    reimplementing them, because a second implementation of "which replays were
    held out" is the one bug that would invalidate every number here.

    Calls: ``_explicit_replay_selection``, ``split_replays``, ``_select_replays``.
    """

    explicit = _explicit_replay_selection(list(replay_paths), config)
    if explicit is not None:
        train_replays, dev_replays, test_replays = explicit
    else:
        split = split_replays(
            list(replay_paths),
            seed=config.pipeline.split_seed,
            test_fraction=config.pipeline.test_fraction,
            dev_fraction=config.pipeline.dev_fraction,
            train_count=config.pipeline.train_replay_count,
            dev_count=config.pipeline.validation_replay_count,
        )
        train_replays, dev_replays = _select_replays(
            list(split.train), list(split.dev), config
        )
        test_replays = list(split.test)
    return {"train": train_replays, "dev": dev_replays, "test": test_replays}[split_name]


def verify_recorded_split(
    derived_paths: Sequence[str],
    *,
    split_name: str,
    replay_selection_path: Path,
) -> dict[str, object]:
    """Fail closed when the live split disagrees with the run's recorded one.

    Raises:
        ValueError: the recorded and derived replay id sets differ, which would
            mean the checkpoint may have trained on the windows about to be
            scored.
    """

    if not replay_selection_path.exists():
        return {"verified": False, "reason": "no replay_selection.json found"}
    recorded = json.loads(replay_selection_path.read_text(encoding="utf-8"))
    recorded_ids = sorted(recorded.get(f"{split_name}_replay_ids", []))
    derived_ids = sorted(Path(path).stem for path in derived_paths)
    if recorded_ids and recorded_ids != derived_ids:
        raise ValueError(
            f"the run's recorded {split_name} split ({len(recorded_ids)} replays) does not "
            f"match the config-derived split ({len(derived_ids)} replays); refusing to score "
            "a partition the checkpoint may have trained on"
        )
    return {
        "verified": bool(recorded_ids),
        "recorded_replay_count": len(recorded_ids),
        "path": portable_path(replay_selection_path),
    }


def select_windows(
    windows: Sequence[WindowManifestEntry],
    *,
    max_examples: int,
    windows_per_replay: int,
    position: str = "first",
) -> list[WindowManifestEntry]:
    """Spread the window budget across every (replay, perspective) group.

    Taking the first N manifest entries would concentrate every scored window in
    one or two replays, which is exactly the narrowness this investigation must
    avoid. Grouping by replay ALONE is not enough either: the manifest lists a
    replay's p1 windows before its p2 windows, so a replay-only round robin with
    a small per-replay budget would silently score one perspective and leave the
    perspective breakdown empty. This walks (replay, perspective) groups round
    robin, so both perspectives enter at the first slot.

    ``windows_per_replay`` is the number of slots taken from EACH group, so with
    two perspectives it yields up to ``2 * windows_per_replay`` windows per
    replay.

    ``position`` chooses which end of each group is sampled. ``"first"`` (the
    default) takes the earliest windows, which is the representative mid-game
    condition. ``"last"`` takes the latest windows, which is the ONLY place a
    terminal ``[END]`` target exists: every other window is boundary-truncated
    and pads directly off its last delimiter, so an ``[END]`` measurement needs
    this option.

    Calls: nothing.
    """

    if position not in {"first", "last"}:
        raise ValueError("position must be 'first' or 'last'")
    by_group: dict[tuple[str, str], list[WindowManifestEntry]] = defaultdict(list)
    for window in windows:
        perspective = str(getattr(window, "perspective_player", ""))
        by_group[(window.replay_id, perspective)].append(window)
    if position == "last":
        for entries in by_group.values():
            entries.reverse()
    ordered_groups = sorted(by_group)
    selected: list[WindowManifestEntry] = []
    for slot in range(max(1, windows_per_replay)):
        for group in ordered_groups:
            entries = by_group[group]
            if slot < len(entries):
                selected.append(entries[slot])
                if 0 < max_examples <= len(selected):
                    return selected
    return selected


def collect_train_unigram(
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    train_replay_paths: Sequence[str],
    *,
    max_windows: int,
    dataset_epoch: int,
) -> np.ndarray | None:
    """Build a leakage-free train-split content-token histogram.

    Only train-split replays are read, and only their TARGET canvases (no model,
    no checkpoint). Bounded by ``max_windows`` so this stays a few seconds of
    work rather than a full corpus pass.

    Returns: a ``[vocab]`` int64 histogram, or None when disabled.

    Calls: ``load_window_manifest``, ``SC2DiffusionDataset``.
    """

    if max_windows <= 0 or not train_replay_paths:
        return None
    windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=list(train_replay_paths),
    )
    windows = select_windows(windows, max_examples=max_windows, windows_per_replay=1)
    if not windows:
        return None
    dataset = SC2DiffusionDataset(
        windows, config, vocabulary, seed=config.pipeline.seed, fog_rate_override=None
    )
    dataset.set_epoch(dataset_epoch)
    counts = np.zeros(vocabulary.vocab_size, dtype=np.int64)
    for index in range(len(dataset)):
        example = dataset[index]
        tokens = example.target_canvas.numpy()
        content = tokens[tokens >= CONTENT_TOKEN_OFFSET]
        if content.size:
            counts += np.bincount(content, minlength=vocabulary.vocab_size).astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Execute the configured arms and return the full JSON-ready report.

    Calls: ``load_config``, ``load_content_vocabulary``, ``_materialize_*``,
    ``resolve_split_replays``, ``verify_recorded_split``, ``select_windows``,
    ``SC2DiffusionDataset``, ``_make_dataloader``, ``load_diagnostic_model``,
    ``corrupt_batch``, ``_forward_canvas_logits``, ``score_batch``,
    ``run_geometry_experiment``, ``persistence_and_unigram_baselines``.
    """

    started_at = time.perf_counter()
    config_path = args.config.resolve()
    config = load_config(config_path)
    resolver = StorageResolver()
    token_dictionary = _materialize_file(
        config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver
    )
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)

    split_paths = resolve_split_replays(
        replay_paths, config=config, split_name=args.split
    )
    replay_selection_path = (
        args.replay_selection
        if args.replay_selection is not None
        else Path(config.storage.log_uri) / "replay_selection.json"
    )
    split_verification = verify_recorded_split(
        split_paths, split_name=args.split, replay_selection_path=replay_selection_path
    )
    manifest_windows = load_window_manifest(
        config.data.window_manifest_path, config=config, replay_paths=list(split_paths)
    )
    windows = select_windows(
        manifest_windows,
        max_examples=args.max_examples,
        windows_per_replay=args.windows_per_replay,
        position=args.window_position,
    )
    if not windows:
        raise ValueError(f"the {args.split} split contains no manifest windows")

    identities = [
        WindowIdentity(
            replay_id=window.replay_id,
            perspective=window.perspective_player,
            start_timestep=window.start_timestep,
            end_timestep=window.end_timestep,
        )
        for window in windows
    ]

    dominant_ids = frozenset(
        token_id
        for token_id, name in vocabulary.id_to_name.items()
        if name in DOMINANT_CONTENT_TOKEN_NAMES
    )
    class_id_to_name = active_class_id_to_name(config)
    criterion = CanvasCrossEntropyLoss(config)

    print(
        "timestep_alignment_probe configuration\n"
        f"  config={portable_path(config_path)}\n"
        f"  checkpoint={portable_path(args.checkpoint)}\n"
        f"  weights={'raw' if args.raw else 'ema'}\n"
        f"  split={args.split} split_replays={len(split_paths)} "
        f"manifest_windows={len(manifest_windows)} selected_windows={len(windows)}\n"
        f"  noise_levels={[round(level, 4) for level in args.noise_level]} "
        f"window_position={args.window_position}\n"
        f"  device={args.device} num_workers={args.num_workers} seed={args.seed} "
        f"batch_size={config.pipeline.batch_size}\n"
        f"  oracle={'off' if args.no_oracle else 'on'} "
        f"geometry_canvases={args.geometry_canvases} "
        f"dominant_excluded={sorted(dominant_ids)}",
        file=sys.stderr,
        flush=True,
    )

    dataset = SC2DiffusionDataset(
        windows, config, vocabulary, seed=config.pipeline.seed, fog_rate_override=None
    )
    dataset.set_epoch(args.dataset_epoch)
    loader_config = replace(
        config, pipeline=replace(config.pipeline, num_workers=args.num_workers)
    )
    loader = _make_dataloader(dataset, loader_config, shuffle=False, device="cpu")
    batches: list[DiffusionBatch] = []
    batch_identities: list[list[WindowIdentity]] = []
    try:
        cursor = 0
        for batch in loader:
            size = batch.target_canvas.shape[0]
            batches.append(batch)
            batch_identities.append(identities[cursor : cursor + size])
            cursor += size
    finally:
        _shutdown_dataloader(loader)

    report: dict[str, object] = {}

    # ---- Experiment A ----------------------------------------------------
    if not args.skip_geometry:
        geometry_started = time.perf_counter()
        report["geometry"] = run_geometry_experiment(
            batches,
            vocabulary_size=vocabulary.vocab_size,
            confidence=args.geometry_confidence,
            class_id_to_name=class_id_to_name,
            max_canvases=args.geometry_canvases,
        )
        print(
            f"experiment_a_geometry canvases={report['geometry']['canvases_scored']} "
            f"elapsed={_format_duration(time.perf_counter() - geometry_started)}",
            file=sys.stderr,
            flush=True,
        )

    if args.geometry_only:
        report["provenance"] = _provenance(
            args,
            config=config,
            config_path=config_path,
            token_dictionary=token_dictionary,
            windows=windows,
            split_paths=split_paths,
            split_verification=split_verification,
            vocabulary=vocabulary,
            elapsed=time.perf_counter() - started_at,
            device_name=None,
        )
        return report

    # ---- Experiment B/C --------------------------------------------------
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but no CUDA device is visible; refusing to "
            "report GPU behavior from a CPU fallback"
        )
    model, run_config = load_diagnostic_model(
        args.checkpoint, config, device=device, use_raw=args.raw
    )
    model.to(device)
    model.eval()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    class_weights = criterion.class_weights.detach()
    slices: dict[tuple[str, str], PooledCounters] = {}
    control_slices: dict[tuple[str, str], PooledCounters] = {}
    token_counts: dict[str, np.ndarray] = {}
    per_window_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    total_units = len(batches) * len(args.noise_level)
    completed = 0
    forward_started = time.perf_counter()
    for batch_index, batch in enumerate(batches):
        device_batch = _batch_to_device(batch, device)
        for noise_level in args.noise_level:
            # Couple the corruption draws across t: re-seeding an identical
            # generator before every call makes `corrupt_batch` draw the SAME
            # Bernoulli uniforms and the SAME replacement tokens at each level,
            # so the sweep changes only which truthful anchors survive. Because
            # the branch draw is shared, the corrupted sets are nested.
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + batch_index)
            corruption = corrupt_batch(
                input_token_ids=device_batch.input_token_ids,
                target_canvas=device_batch.target_canvas,
                process=run_config.diffusion.process,
                schedule=run_config.diffusion.schedule,
                vocab_size=int(model.vocab_size),
                generator=generator,
                t=noise_level,
                canvas_noise_mask=device_batch.canvas_loss_mask,
            )
            canvas_logits = _forward_canvas_logits(
                model,
                input_token_ids=device_batch.input_token_ids,
                input_attention_mask=device_batch.input_attention_mask,
                input_lengths=device_batch.input_lengths,
                input_features=device_batch.input_features,
                noised_canvas=corruption.noised_canvas,
                canvas_attention_mask=device_batch.canvas_attention_mask,
            )
            score_batch(
                canvas_logits=canvas_logits,
                batch=device_batch,
                changed_positions=corruption.changed_positions,
                class_weights=class_weights,
                class_id_to_name=class_id_to_name,
                identities=batch_identities[batch_index],
                noise_level=noise_level,
                dominant_ids=dominant_ids,
                slices=slices,
                token_counts=token_counts,
                per_window_rows=per_window_rows,
                compute_oracle=not args.no_oracle,
                oracle_max_span=args.oracle_max_span,
                vocab_size=vocabulary.vocab_size,
            )

            # ---- Experiment C: conditioning-use controls, identical canvas ---
            # The noised canvas, targets, and corruption draw are byte-identical
            # to the primary arm above; ONLY the clamped input changes. Any
            # difference is therefore attributable to conditioning use.
            if (
                device_batch.target_canvas.shape[0] >= 2
                and math.isclose(noise_level, args.control_noise_level, abs_tol=1e-9)
            ):
                for control_name, overrides in (
                    ("shuffled_input", _shuffle_input_rows(device_batch)),
                    ("enemy_stripped_input", _strip_enemy_input(device_batch)),
                ):
                    control_logits = _forward_canvas_logits(
                        model,
                        input_token_ids=overrides["input_token_ids"],
                        input_attention_mask=overrides["input_attention_mask"],
                        input_lengths=overrides["input_lengths"],
                        input_features=overrides["input_features"],
                        noised_canvas=corruption.noised_canvas,
                        canvas_attention_mask=device_batch.canvas_attention_mask,
                    )
                    score_batch(
                        canvas_logits=control_logits,
                        batch=device_batch,
                        changed_positions=corruption.changed_positions,
                        class_weights=class_weights,
                        class_id_to_name=class_id_to_name,
                        identities=batch_identities[batch_index],
                        noise_level=noise_level,
                        dominant_ids=dominant_ids,
                        slices=control_slices,
                        token_counts={},
                        per_window_rows=control_rows,
                        compute_oracle=not args.no_oracle,
                        oracle_max_span=args.oracle_max_span,
                        vocab_size=vocabulary.vocab_size,
                        overall_facet=f"control:{control_name}",
                    )

            completed += 1
            elapsed = time.perf_counter() - forward_started
            rate = completed / max(elapsed, 1e-9)
            print(
                f"forward_pass={completed}/{total_units} "
                f"elapsed={_format_duration(elapsed)} "
                f"rate={rate:.2f}_passes_per_s "
                f"eta={_format_duration((total_units - completed) / rate if rate else math.inf)}",
                file=sys.stderr,
                flush=True,
            )

    train_paths = resolve_split_replays(replay_paths, config=config, split_name="train")
    unigram_counts = collect_train_unigram(
        config,
        vocabulary,
        train_paths,
        max_windows=args.baseline_max_windows,
        dataset_epoch=args.dataset_epoch,
    )

    report["model"] = {
        "slices": {
            f"{facet}/{value}": finalize_counters(counters)
            for (facet, value), counters in sorted(slices.items())
        },
        "structural": {
            f"{facet}/{value}": _finalize_structural(counters)
            for (facet, value), counters in sorted(slices.items())
            if facet in {"overall", "t", "delimiter_count_correct"}
        },
        "noised_macro_f1_by_t": {
            key: _macro_f1_from_counts(torch.from_numpy(matrix))
            for key, matrix in sorted(token_counts.items())
        },
        "per_window": per_window_rows,
    }
    # The matched "correct input" arm is the primary slice at the same t, scored
    # on the same windows with the same corruption draw, so the three
    # conditioning arms are directly comparable.
    conditioning: dict[str, object] = {}
    correct_key = ("t", f"{args.control_noise_level:.2f}")
    if correct_key in slices:
        conditioning["correct_input"] = finalize_counters(slices[correct_key])
    for (facet, _value), counters in sorted(control_slices.items()):
        if facet.startswith("control:"):
            conditioning[facet.split(":", 1)[1]] = finalize_counters(counters)

    report["controls"] = {
        "conditioning": conditioning,
        "control_noise_level": args.control_noise_level,
        "baselines": persistence_and_unigram_baselines(
            batches, unigram_counts=unigram_counts, dominant_ids=dominant_ids
        ),
        "per_window": control_rows,
    }
    report["provenance"] = _provenance(
        args,
        config=config,
        config_path=config_path,
        token_dictionary=token_dictionary,
        windows=windows,
        split_paths=split_paths,
        split_verification=split_verification,
        vocabulary=vocabulary,
        elapsed=time.perf_counter() - started_at,
        device_name=device_name,
    )
    return report


def _finalize_structural(counters: PooledCounters) -> dict[str, object]:
    """Report the structural comparisons pooled over one slice."""

    values = counters.values
    windows = values.get("windows", 0.0)
    return {
        "windows": int(windows),
        "outcome_ce_nats": _ratio(
            values.get("outcome_ce_sum", 0.0), values.get("outcome_positions", 0.0)
        ),
        "outcome_pair_mass": _ratio(
            values.get("outcome_pair_mass_sum", 0.0), values.get("outcome_positions", 0.0)
        ),
        "outcome_accuracy": _ratio(values.get("outcome_correct", 0.0), windows),
        "mean_target_active_length": _ratio(values.get("target_active_length", 0.0), windows),
        "mean_predicted_active_length": _ratio(
            values.get("predicted_active_length", 0.0), windows
        ),
        "mean_target_delimiters": _ratio(values.get("target_delimiters", 0.0), windows),
        "mean_predicted_delimiters": _ratio(values.get("predicted_delimiters", 0.0), windows),
        "delimiter_count_exact_rate": _ratio(values.get("delimiter_count_correct", 0.0), windows),
        "mean_abs_delimiter_drift": _ratio(
            values.get("delimiter_drift_abs_sum", 0.0), values.get("delimiter_drift_pairs", 0.0)
        ),
        "mean_signed_delimiter_drift": _ratio(
            values.get("delimiter_drift_signed_sum", 0.0),
            values.get("delimiter_drift_pairs", 0.0),
        ),
    }


def _batch_to_device(batch: DiffusionBatch, device: torch.device) -> DiffusionBatch:
    """Move every model-facing tensor of a collated batch onto ``device``."""

    features = batch.input_features
    return replace(
        batch,
        input_token_ids=batch.input_token_ids.to(device),
        input_attention_mask=batch.input_attention_mask.to(device),
        input_lengths=batch.input_lengths.to(device),
        target_canvas=batch.target_canvas.to(device),
        canvas_attention_mask=batch.canvas_attention_mask.to(device),
        class_labels=batch.class_labels.to(device),
        canvas_loss_mask=batch.canvas_loss_mask.to(device),
        canvas_prediction_distances=batch.canvas_prediction_distances.to(device),
        perspective_ids=batch.perspective_ids.to(device),
        input_features=InputFeatures(
            continuous_values=features.continuous_values.to(device),
            continuous_validity=features.continuous_validity.to(device),
            categorical_values=features.categorical_values.to(device),
            allegiance_values=features.allegiance_values.to(device),
            feature_mask=features.feature_mask.to(device),
        ),
    )


def _provenance(
    args: argparse.Namespace,
    *,
    config: ProjectConfig,
    config_path: Path,
    token_dictionary: Path,
    windows: Sequence[WindowManifestEntry],
    split_paths: Sequence[str],
    split_verification: Mapping[str, object],
    vocabulary: ContentVocabulary,
    elapsed: float,
    device_name: str | None,
) -> dict[str, object]:
    """Assemble portable, repository-relative provenance for the report."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_profile": config_path.stem,
        "config_path": portable_path(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": portable_path(args.checkpoint),
        "checkpoint_weights": "raw" if args.raw else "ema",
        "split": args.split,
        "split_replay_count": len(split_paths),
        "split_verification": dict(split_verification),
        "selected_window_count": len(windows),
        "selected_windows": [
            f"{window.replay_id}:{window.perspective_player}:"
            f"{window.start_timestep}-{window.end_timestep}"
            for window in windows
        ],
        "selected_replay_ids": sorted({window.replay_id for window in windows}),
        "manifest_path": portable_path(Path(config.data.window_manifest_path)),
        "manifest_metadata": read_manifest_metadata(config.data.window_manifest_path),
        "token_dictionary_path": portable_path(token_dictionary),
        "token_dictionary_sha256": _sha256(token_dictionary),
        "vocabulary_width": vocabulary.vocab_size,
        "noise_levels": [round(level, 6) for level in args.noise_level],
        "device": args.device,
        "device_name": device_name,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "dataset_epoch": args.dataset_epoch,
        "window_position": args.window_position,
        "batch_size": config.pipeline.batch_size,
        "oracle_enabled": not args.no_oracle,
        "oracle_max_span": args.oracle_max_span,
        "geometry_confidence": args.geometry_confidence,
        "elapsed_seconds": round(elapsed, 3),
        "read_only": (
            "reads config, manifest, processed arrays, replay parquet, and the checkpoint; "
            "writes only below scripts/output/timestep_alignment_probe/"
        ),
    }


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def format_summary(report: Mapping[str, object]) -> str:
    """Render a compact human-readable summary of the whole report."""

    lines: list[str] = ["Timestep alignment probe"]
    provenance = report.get("provenance", {})
    if provenance:
        lines.append(
            f"config={provenance['config_profile']} split={provenance['split']} "
            f"windows={provenance['selected_window_count']} "
            f"replays={len(provenance['selected_replay_ids'])} "
            f"weights={provenance['checkpoint_weights']} device={provenance['device_name']}"
        )
        lines.append(f"checkpoint={provenance['checkpoint_path']}")

    geometry = report.get("geometry")
    if geometry:
        lines.append("")
        lines.append(
            "EXPERIMENT A -- model-independent objective geometry "
            f"({geometry['canvases_scored']} real clean canvases, pseudo-logit confidence "
            f"{geometry['pseudo_logit_confidence']:.3f})"
        )
        lines.append(
            f"  correct-position CE={geometry['per_position_correct_ce_nats']:.6f} nats, "
            f"wrong-position CE={geometry['per_position_wrong_ce_nats']:.6f} nats"
        )
        lines.append(
            "  case                              edits  mismatch  ampl.   excess_CE  span   "
            "focusMS_F1  focus_ampl  focus_edit"
        )
        for name, entry in geometry["pooled"].items():
            lines.append(
                f"  {name:<32} {entry['semantic_edits']:>6} {entry['positional_mismatches']:>9} "
                f"{_format_optional(entry['mismatch_amplification']):>7} "
                f"{entry['excess_ce_nats']:>11.2f} "
                f"{entry['mean_penalty_span']:>6.1f} "
                f"{_format_optional(entry['focus_multiset_f1']):>11} "
                f"{_format_optional(entry['focus_mismatch_amplification']):>11} "
                f"{entry['focus_edit_distance_per_canvas']:>11.2f}"
            )
        sweep = geometry.get("deletion_offset_sweep") or {}
        if sweep:
            lines.append(
                "  one bounded deletion swept over every offset of the focus group: "
                f"group_len={sweep['mean_focus_content_length']:.1f} "
                f"mismatches min/mean/max="
                f"{sweep['min_positional_mismatches']}/"
                f"{sweep['mean_positional_mismatches']:.1f}/"
                f"{sweep['max_positional_mismatches']} "
                f"mean_amplification={sweep['mean_mismatch_amplification']:.2f}x"
            )

    model = report.get("model")
    if model:
        lines.append("")
        lines.append("EXPERIMENT B -- real V3 EMA checkpoint logits, one denoiser pass per t")
        lines.append(
            "  slice                weighted_obj  unweighted_CE  noised_acc  content_CE  "
            "oracle_CE   gap_nats  gap_frac  multisetF1  exactMS"
        )
        for key, entry in model["slices"].items():
            if not key.startswith(("overall/", "t/")):
                continue
            lines.append(
                f"  {key:<20} {_format_optional(entry.get('weighted_objective_nats')):>12} "
                f"{_format_optional(entry.get('unweighted_ce_nats')):>14} "
                f"{_format_optional(entry.get('noised_argmax_accuracy')):>11} "
                f"{_format_optional(entry.get('content_positional_ce_nats')):>11} "
                f"{_format_optional(entry.get('oracle_aligned_ce_nats')):>10} "
                f"{_format_optional(entry.get('alignment_gap_nats')):>10} "
                f"{_format_optional(entry.get('alignment_gap_fraction')):>9} "
                f"{_format_optional(entry.get('multiset_f1')):>11} "
                f"{_format_optional(entry.get('exact_multiset_rate')):>8}"
            )
        lines.append("")
        lines.append("  macro-F1 on genuinely noised positions, by t:")
        for key, value in model["noised_macro_f1_by_t"].items():
            lines.append(f"    t={key}: {value:.6f}")
        lines.append("")
        lines.append("  structural:")
        for key, entry in model["structural"].items():
            lines.append(
                f"    {key:<28} outcome_CE={entry['outcome_ce_nats']:.4f} "
                f"pair_mass={entry['outcome_pair_mass']:.4f} "
                f"delim_exact={entry['delimiter_count_exact_rate']:.4f} "
                f"|drift|={entry['mean_abs_delimiter_drift']:.2f} "
                f"len_pred/target="
                f"{entry['mean_predicted_active_length']:.0f}/{entry['mean_target_active_length']:.0f}"
            )
        lines.append("")
        lines.append("  per class (pooled over every t):")
        for key, entry in model["slices"].items():
            if not key.startswith("class/"):
                continue
            lines.append(
                f"    {key:<28} positions={entry['scored_positions']:>9} "
                f"CE={_format_optional(entry.get('unweighted_ce_nats')):>10} "
                f"noised_acc={_format_optional(entry.get('noised_argmax_accuracy')):>10}"
            )

    controls = report.get("controls")
    if controls:
        lines.append("")
        lines.append(
            f"EXPERIMENT C -- controls at t={controls['control_noise_level']:.2f}"
        )
        for key, entry in controls["conditioning"].items():
            lines.append(
                f"  {key:<28} unweighted_CE={_format_optional(entry.get('unweighted_ce_nats')):>10} "
                f"noised_acc={_format_optional(entry.get('noised_argmax_accuracy')):>10} "
                f"multisetF1={_format_optional(entry.get('multiset_f1')):>10}"
            )
        for key, entry in controls["baselines"].items():
            if not isinstance(entry, dict):
                lines.append(f"  {key}={entry}")
                continue
            lines.append(
                f"  {key:<28} multisetF1={_format_optional(entry.get('multiset_f1')):>10} "
                f"exactMS={_format_optional(entry.get('exact_multiset_rate')):>10} "
                f"count_MAE={_format_optional(entry.get('count_mae')):>10}"
            )

    lines.append("")
    lines.append(
        "READ THESE CAREFULLY: Experiment A is model-independent and proves an objective-"
        "geometry property. Experiment B is OBSERVATIONAL -- the probed checkpoint was itself "
        "trained under positional CE, so its alignment gap cannot establish causation. The "
        "oracle aligned score is deliberately optimistic: it is order-invariant inside a "
        "ground-truth timestep span, never generates a delimiter sequence, and is not a "
        "likelihood or the production loss."
    )
    return "\n".join(lines) + "\n"


def write_per_window_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write the bounded per-window table used by the durable diagnostic."""

    if not rows:
        return
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how positional canvas cross-entropy prices small delimiter-local "
            "SC2 semantic edits, with a model-independent geometry arm and an "
            "observational real-checkpoint arm."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"merged run profile YAML (default: {DEFAULT_CONFIG.as_posix()})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"training checkpoint to probe (default: {DEFAULT_CHECKPOINT.as_posix()})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="load raw weights; the default is the EMA weights the sampler serves",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="test",
        help="recorded replay split to score (default: test)",
    )
    parser.add_argument(
        "--replay-selection",
        type=Path,
        default=None,
        help="run's replay_selection.json; defaults to <storage.log_uri>/replay_selection.json",
    )
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; 0 is the safe in-process fallback (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=20260826, help="corruption seed")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=48,
        help="maximum replay windows to score (default: 48)",
    )
    parser.add_argument(
        "--windows-per-replay",
        type=int,
        default=2,
        help=(
            "windows taken from each (replay, perspective) group before moving on "
            "(default: 2)"
        ),
    )
    parser.add_argument(
        "--window-position",
        choices=("first", "last"),
        default="first",
        help=(
            "which end of each (replay, perspective) group to sample; 'last' is the "
            "only way to reach terminal [END] targets, because every other window is "
            "boundary-truncated (default: first)"
        ),
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        action="append",
        default=None,
        help=(
            "corruption level for one denoiser pass; repeat to sweep "
            f"(default: {list(DEFAULT_NOISE_LEVELS)})"
        ),
    )
    parser.add_argument(
        "--control-noise-level",
        type=float,
        default=1.0,
        help="noise level at which the input-conditioning controls run (default: 1.0)",
    )
    parser.add_argument(
        "--dataset-epoch",
        type=int,
        default=0,
        help="deterministic per-serving fog epoch (default: 0)",
    )
    parser.add_argument(
        "--geometry-canvases",
        type=int,
        default=12,
        help="clean canvases used by the model-independent arm (default: 12)",
    )
    parser.add_argument(
        "--geometry-confidence",
        type=float,
        default=0.9,
        help="pseudo-logit probability on the predicted token (default: 0.9)",
    )
    parser.add_argument(
        "--geometry-only", action="store_true", help="run only the model-independent arm"
    )
    parser.add_argument(
        "--skip-geometry", action="store_true", help="skip the model-independent arm"
    )
    parser.add_argument(
        "--no-oracle", action="store_true", help="skip the assignment-based aligned score"
    )
    parser.add_argument(
        "--oracle-max-span",
        type=int,
        default=1024,
        help="largest timestep content span the oracle assignment is run on (default: 1024)",
    )
    parser.add_argument(
        "--baseline-max-windows",
        type=int,
        default=64,
        help="train-split windows read for the unigram baseline (default: 64; 0 disables)",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON artifact path")
    args = parser.parse_args(argv)
    if args.noise_level is None:
        args.noise_level = list(DEFAULT_NOISE_LEVELS)
    for level in args.noise_level:
        if not 0.0 <= level <= 1.0:
            parser.error("--noise-level must lie in [0, 1]")
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.windows_per_replay <= 0:
        parser.error("--windows-per-replay must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0.0 < args.geometry_confidence < 1.0:
        parser.error("--geometry-confidence must lie strictly between 0 and 1")
    if args.geometry_only and args.skip_geometry:
        parser.error("--geometry-only and --skip-geometry are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    venv_python = Path(".venv/Scripts/python.exe")
    if not venv_python.exists() and not Path(".venv/bin/python").exists():
        print(
            "warning: no project virtual environment found at .venv/; run this script "
            "through .venv\\Scripts\\python.exe",
            file=sys.stderr,
        )
    report = run_probe(args)
    output_path = args.output
    if output_path is None:
        suffix = "geometry" if args.geometry_only else args.split
        output_path = DEFAULT_OUTPUT_DIR / f"{args.config.stem}-{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = format_summary(report)
    summary_path = output_path.with_suffix(".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    model = report.get("model")
    if model:
        write_per_window_csv(
            model["per_window"], output_path.with_suffix(".per_window.csv")
        )
    print(summary, end="")
    print(f"json_artifact={portable_path(output_path)}")
    print(f"summary_artifact={portable_path(summary_path)}")
    return 0


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float | None:
    """Pooled ratio, or None when the slice has no support."""

    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _f1(overlap: float, predicted_total: float, target_total: float) -> float | None:
    """Harmonic mean of pooled multiset precision and recall."""

    precision = _ratio(overlap, predicted_total)
    recall = _ratio(overlap, target_total)
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    return f'"{text}"' if "," in text else text


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unserializable value of type {type(value)!r}")


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes}m{seconds_part:02d}s"
    return f"{seconds_part}s"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Render a path relative to the checkout root so evidence stays portable."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
