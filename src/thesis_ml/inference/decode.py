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

    if WIN_ID in token_ids or LOSS_ID in token_ids:
        return CanvasValidation(False, "outcome token is not valid in pretraining canvas", None, False, False)

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
    if active_end == 0 or token_ids[active_end - 1] != DELIMITER_ID:
        return CanvasValidation(False, "truncated canvas must end on a timestep boundary", None, True, False)
    return CanvasValidation(True, None, None, True, False)


def decode_canvas(
    token_ids: Sequence[int],
    vocabulary: ContentVocabulary | Mapping[int, str],
) -> DecodedCanvas:
    validation = validate_canvas(token_ids)
    if not validation.valid:
        return DecodedCanvas(validation, [], validation.truncated, validation.partial_final_timestep)

    names = _id_to_name(vocabulary)
    if validation.end_index is not None:
        active = token_ids[: validation.end_index]
    else:
        try:
            active = token_ids[: token_ids.index(PAD_ID)]
        except ValueError:
            active = token_ids
    timesteps: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for index, token_id in enumerate(active):
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
