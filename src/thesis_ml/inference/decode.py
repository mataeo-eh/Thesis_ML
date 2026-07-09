"""Decode and validate generated output canvases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from thesis_ml.vocab.content_vocab import ContentVocabulary
from thesis_ml.vocab.special_tokens import (
    DELIMITER_ID,
    END_ID,
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

    # The pre-training canvas now leads with exactly one [WIN]/[LOSS] outcome
    # token at position 0 (denoised last), the same leading-outcome-token layout
    # as the debut grammar. It may not appear at any other position.
    if token_ids[0] not in (WIN_ID, LOSS_ID):
        return CanvasValidation(False, "canvas must start with a [WIN]/[LOSS] outcome token", None, False, False)
    if WIN_ID in token_ids[1:] or LOSS_ID in token_ids[1:]:
        return CanvasValidation(False, "outcome token may appear only at position 0", None, False, False)

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
        if end_index == 0 or token_ids[end_index - 1] != DELIMITER_ID:
            return CanvasValidation(False, "[END] must follow a complete timestep", end_index, False, False)
        return CanvasValidation(True, None, end_index, False, False)

    active_end = first_pad if first_pad is not None else len(token_ids)
    # active_end <= 1 means only the outcome token precedes the padding (no
    # timesteps at all); token_ids[active_end - 1] would be the outcome token.
    if active_end <= 1 or token_ids[active_end - 1] != DELIMITER_ID:
        return CanvasValidation(False, "truncated canvas must end on a timestep boundary", None, True, False)
    return CanvasValidation(True, None, None, True, False)


def validate_debut_canvas(token_ids: Sequence[int]) -> CanvasValidation:
    """Validate the RELAXED fine-tuning (debut-mode) canvas grammar.

    This is a SEPARATE, additive validator for fine-tuning: it does NOT touch
    ``validate_canvas`` (the pre-training grammar) and is never used by
    ``decode_canvas``. The pre-training grammar explicitly REJECTS win/loss
    outcome tokens; the fine-tuning grammar REQUIRES exactly one at position 0.

    The debut grammar accepted here is:

        [WIN | LOSS]                          # exactly one, at position 0
        ( [DELIMITER] | timestep-tokens [DELIMITER] )+   # one group per timestep
        ( [END] [PAD]* | [PAD]* )             # optional terminal [END], then pad

    Notes on the relaxations relative to the pre-training grammar:
      * A timestep may be EMPTY: two ``[DELIMITER]`` tokens back-to-back are
        legal (an empty debut timestep emits a bare delimiter). The
        pre-training grammar has no such notion because it reconstructs full
        snapshots.
      * Position 0 is the single outcome token, which the pre-training grammar
        forbids anywhere.

    Everything else mirrors the structural rules of ``validate_canvas``:
      * No residual ``[MASK]`` may remain.
      * ``[PAD]`` may only appear as a trailing run and may not precede ``[END]``.
      * ``[END]`` must be immediately followed by ``[PAD]`` (or the sequence end)
        and must sit on a completed timestep boundary (preceded by a
        ``[DELIMITER]``).
      * A truncated canvas (no ``[END]``) must still end on a timestep boundary.

    Parameters:
        token_ids: The full generated canvas token id sequence (position 0
            included).

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

    # Rule 1: position 0 must be exactly one win/loss outcome token, and no other
    # position may contain a win/loss token.
    if token_ids[0] not in (WIN_ID, LOSS_ID):
        return CanvasValidation(False, "debut canvas must start with a [WIN]/[LOSS] token", None, False, False)
    rest = token_ids[1:]
    if WIN_ID in rest or LOSS_ID in rest:
        return CanvasValidation(False, "outcome token may appear only at position 0", None, False, False)

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
        # [DELIMITER], and it cannot be the outcome token at position 0.
        if end_index == 0 or token_ids[end_index - 1] != DELIMITER_ID:
            return CanvasValidation(False, "[END] must follow a complete timestep", end_index, False, False)
        return CanvasValidation(True, None, end_index, False, False)

    # No [END]: the canvas is truncated. The active region (everything before the
    # trailing pad) must still end on a timestep boundary, i.e. a [DELIMITER].
    active_end = first_pad if first_pad is not None else len(token_ids)
    # active_end == 1 means only the outcome token precedes the padding (no
    # timesteps at all); token_ids[active_end - 1] would be the outcome token,
    # which is not a delimiter, so this is correctly rejected below.
    if active_end <= 1 or token_ids[active_end - 1] != DELIMITER_ID:
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
    # Skip position 0 (the [WIN]/[LOSS] outcome token) -- timestep parsing begins
    # after it. validate_canvas has already confirmed exactly one outcome token
    # sits there.
    if validation.end_index is not None:
        active = token_ids[1 : validation.end_index]
    else:
        try:
            active = token_ids[1 : token_ids.index(PAD_ID)]
        except ValueError:
            active = token_ids[1:]
    timesteps: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for index, token_id in enumerate(active, start=1):
        if token_id == DELIMITER_ID:
            timesteps.append(current)
            current = {}
            continue
        if token_id in {PAD_ID, END_ID, MASK_ID, WIN_ID, LOSS_ID}:
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


def _id_to_name(vocabulary: ContentVocabulary | Mapping[int, str]) -> Mapping[int, str]:
    if isinstance(vocabulary, ContentVocabulary):
        return vocabulary.id_to_name
    return vocabulary
