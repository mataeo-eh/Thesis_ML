"""Decode and validate generated output canvases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from thesis_ml.vocab.content_vocab import ContentVocabulary
from thesis_ml.vocab.special_tokens import (
    BOS_ID,
    DELIMITER_ID,
    END_ID,
    EOS_ID,
    LOSS_ID,
    MASK_ID,
    PAD_ID,
    WIN_ID,
)


@dataclass(frozen=True)
class CanvasValidation:
    valid: bool
    diagnosis: str | None
    end_index: int | None
    truncated: bool
    partial_final_timestep: bool


@dataclass(frozen=True)
class DecodedCanvas:
    validation: CanvasValidation
    timesteps: list[dict[str, int]]
    truncated: bool
    partial_final_timestep: bool


def validate_canvas(token_ids: Sequence[int]) -> CanvasValidation:
    """Validate the SPEC §7 canvas grammar without repairing it."""

    if not token_ids:
        return CanvasValidation(False, "canvas is empty", None, False, False)
    if MASK_ID in token_ids:
        return CanvasValidation(False, "canvas still contains [MASK]", None, False, False)

    try:
        first_pad = token_ids.index(PAD_ID)
    except ValueError:
        first_pad = None
    try:
        end_index = token_ids.index(END_ID)
    except ValueError:
        end_index = None

    prefix_error = _canvas_prefix_error(token_ids)
    if prefix_error is not None:
        return CanvasValidation(False, prefix_error, None, False, False)
    if EOS_ID in token_ids:
        return CanvasValidation(False, "[EOS] is input-only and invalid on the canvas", None, False, False)

    if first_pad is not None and end_index is not None and first_pad < end_index:
        return CanvasValidation(False, "[PAD] appears before [END]", end_index, False, False)
    if first_pad is not None:
        for index, token_id in enumerate(token_ids[first_pad:], start=first_pad):
            if token_id != PAD_ID:
                return CanvasValidation(False, f"non-[PAD] token after padding at position {index}", end_index, False, False)
    if end_index is not None:
        if first_pad is not None and first_pad != end_index + 1:
            return CanvasValidation(False, "[END] must be followed immediately by [PAD]", end_index, False, False)
        for index, token_id in enumerate(token_ids[end_index + 1 :], start=end_index + 1):
            if token_id != PAD_ID:
                return CanvasValidation(False, f"non-[PAD] token after [END] at position {index}", end_index, False, False)
        if end_index <= 1 or token_ids[end_index - 1] != DELIMITER_ID:
            return CanvasValidation(False, "[END] must follow a complete timestep", end_index, False, False)
        return CanvasValidation(True, None, end_index, False, False)

    active_end = first_pad if first_pad is not None else len(token_ids)
    # active_end <= 2 means only [BOS] + outcome precede padding.
    if active_end <= 2 or token_ids[active_end - 1] != DELIMITER_ID:
        return CanvasValidation(False, "truncated canvas must end on a timestep boundary", None, True, False)
    return CanvasValidation(True, None, None, True, False)


def validate_debut_canvas(token_ids: Sequence[int]) -> CanvasValidation:
    """Validate the RELAXED fine-tuning (debut-mode) canvas grammar.

    This is a SEPARATE, additive validator for fine-tuning: it does NOT touch
    ``validate_canvas`` (the pre-training grammar) and is never used by
    ``decode_canvas``. Both grammars require [BOS] at 0 and one outcome at 1;
    debut mode differs only by permitting empty timestep groups.

    The debut grammar accepted here is:

        [BOS]                                 # fixed anchor at position 0
        [WIN | LOSS]                          # exactly one, at position 1
        ( [DELIMITER] | timestep-tokens [DELIMITER] )+   # one group per timestep
        ( [END] [PAD]* | [PAD]* )             # optional terminal [END], then pad

    Notes on the relaxations relative to the pre-training grammar:
      * A timestep may be EMPTY: two ``[DELIMITER]`` tokens back-to-back are
        legal (an empty debut timestep emits a bare delimiter). The
        pre-training grammar has no such notion because it reconstructs full
        snapshots.
      * Position 0 is fixed [BOS]; position 1 is the single outcome token.

    Everything else mirrors the structural rules of ``validate_canvas``:
      * No residual ``[MASK]`` may remain.
      * ``[PAD]`` may only appear as a trailing run and may not precede ``[END]``.
      * ``[END]`` must be immediately followed by ``[PAD]`` (or the sequence end)
        and must sit on a completed timestep boundary (preceded by a
        ``[DELIMITER]``).
      * A truncated canvas (no ``[END]``) must still end on a timestep boundary.

    Parameters:
        token_ids: The full generated canvas token id sequence.

    Returns:
        A ``CanvasValidation``. ``valid`` is True only when the sequence matches
        the debut grammar above; ``diagnosis`` explains the first violation.

    Calls:
        Nothing else; pure structural checks over the id sequence.
    """

    if not token_ids:
        return CanvasValidation(False, "canvas is empty", None, False, False)
    if MASK_ID in token_ids:
        return CanvasValidation(False, "canvas still contains [MASK]", None, False, False)

    prefix_error = _canvas_prefix_error(token_ids, debut=True)
    if prefix_error is not None:
        return CanvasValidation(False, prefix_error, None, False, False)
    if EOS_ID in token_ids:
        return CanvasValidation(False, "[EOS] is input-only and invalid on the canvas", None, False, False)

    # Locate the first [PAD] and the (single expected) [END] over the whole
    # sequence, mirroring validate_canvas's bookkeeping.
    try:
        first_pad = token_ids.index(PAD_ID)
    except ValueError:
        first_pad = None
    try:
        end_index = token_ids.index(END_ID)
    except ValueError:
        end_index = None

    # [PAD] may never appear before [END].
    if first_pad is not None and end_index is not None and first_pad < end_index:
        return CanvasValidation(False, "[PAD] appears before [END]", end_index, False, False)
    # Once padding starts it must run uninterrupted to the end.
    if first_pad is not None:
        for index, token_id in enumerate(token_ids[first_pad:], start=first_pad):
            if token_id != PAD_ID:
                return CanvasValidation(False, f"non-[PAD] token after padding at position {index}", end_index, False, False)

    if end_index is not None:
        # [END] must be immediately followed by [PAD] (if any padding exists).
        if first_pad is not None and first_pad != end_index + 1:
            return CanvasValidation(False, "[END] must be followed immediately by [PAD]", end_index, False, False)
        for index, token_id in enumerate(token_ids[end_index + 1 :], start=end_index + 1):
            if token_id != PAD_ID:
                return CanvasValidation(False, f"non-[PAD] token after [END] at position {index}", end_index, False, False)
        # [END] must land on a completed timestep: the token before it is a
        # [DELIMITER], and it cannot be part of the two-token prefix.
        if end_index <= 1 or token_ids[end_index - 1] != DELIMITER_ID:
            return CanvasValidation(False, "[END] must follow a complete timestep", end_index, False, False)
        return CanvasValidation(True, None, end_index, False, False)

    # No [END]: the canvas is truncated. The active region (everything before the
    # trailing pad) must still end on a timestep boundary, i.e. a [DELIMITER].
    active_end = first_pad if first_pad is not None else len(token_ids)
    # active_end <= 2 means only [BOS] + outcome precede the padding.
    if active_end <= 2 or token_ids[active_end - 1] != DELIMITER_ID:
        return CanvasValidation(False, "truncated debut canvas must end on a timestep boundary", None, True, False)
    return CanvasValidation(True, None, None, True, False)


def decode_canvas(
    token_ids: Sequence[int],
    vocabulary: ContentVocabulary | Mapping[int, str],
) -> DecodedCanvas:
    validation = validate_canvas(token_ids)
    if not validation.valid:
        return DecodedCanvas(validation, [], validation.truncated, validation.partial_final_timestep)

    names = _id_to_name(vocabulary)
    # Skip [BOS] and the [WIN]/[LOSS] outcome prefix.
    if validation.end_index is not None:
        active = token_ids[2 : validation.end_index]
    else:
        try:
            active = token_ids[2 : token_ids.index(PAD_ID)]
        except ValueError:
            active = token_ids[2:]
    timesteps: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for index, token_id in enumerate(active, start=2):
        if token_id == DELIMITER_ID:
            timesteps.append(current)
            current = {}
            continue
        if token_id in {PAD_ID, END_ID, MASK_ID, WIN_ID, LOSS_ID, BOS_ID, EOS_ID}:
            diagnosis = f"unexpected special token {token_id} at position {index}"
            invalid = CanvasValidation(False, diagnosis, validation.end_index, validation.truncated, validation.partial_final_timestep)
            return DecodedCanvas(invalid, [], invalid.truncated, invalid.partial_final_timestep)
        try:
            name = names[token_id]
        except KeyError:
            diagnosis = f"unknown content token id {token_id} at position {index}"
            invalid = CanvasValidation(False, diagnosis, validation.end_index, validation.truncated, validation.partial_final_timestep)
            return DecodedCanvas(invalid, [], invalid.truncated, invalid.partial_final_timestep)
        current[name] = current.get(name, 0) + 1

    return DecodedCanvas(validation, timesteps, validation.truncated, validation.partial_final_timestep)


def _canvas_prefix_error(token_ids: Sequence[int], *, debut: bool = False) -> str | None:
    label = "debut canvas" if debut else "canvas"
    if len(token_ids) < 2:
        return f"{label} must begin with [BOS] then [WIN]/[LOSS]"
    if token_ids[0] != BOS_ID:
        return f"{label} must start with [BOS]"
    if BOS_ID in token_ids[1:]:
        return "[BOS] may appear only at position 0"
    if token_ids[1] not in (WIN_ID, LOSS_ID):
        return f"{label} position 1 must be [WIN] or [LOSS]"
    if WIN_ID in token_ids[2:] or LOSS_ID in token_ids[2:]:
        return "outcome token may appear only at position 1"
    return None


def _id_to_name(vocabulary: ContentVocabulary | Mapping[int, str]) -> Mapping[int, str]:
    if isinstance(vocabulary, ContentVocabulary):
        return vocabulary.id_to_name
    return vocabulary
