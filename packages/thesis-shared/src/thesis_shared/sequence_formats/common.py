"""Shared perspective validation and conversion from canonical entity records."""

from thesis_shared.serialize import TokenRecord
from thesis_shared.sequence_formats.types import SequenceToken
from thesis_shared.vocab.sequence_vocabulary import SequenceVocabulary


def enemy_player(perspective: str) -> str:
    if perspective not in ("p1", "p2"):
        raise ValueError("perspective must be p1 or p2")
    return "p2" if perspective == "p1" else "p1"


def structural_token(vocabulary: SequenceVocabulary, name: str, timestep: int | None = None,
                     owner: str | None = None, role: str = "structure") -> SequenceToken:
    return SequenceToken(vocabulary.token_id(name), name, timestep, owner, role)


def owner_content(records: list[TokenRecord], owner: str, timestep: int) -> list[SequenceToken]:
    # Preserve the serializer's canonical order; discard raw attributes here.
    return [SequenceToken(r.token_id, r.token_name, timestep, owner, "content")
            for r in records if r.owner == owner]
