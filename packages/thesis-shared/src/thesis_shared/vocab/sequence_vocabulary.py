"""Presentation-specific vocabulary overlays; existing content IDs never move."""

from dataclasses import dataclass
import hashlib
import json

from thesis_shared.vocab.content_vocab import ContentVocabulary
from thesis_shared.vocab.special_tokens import SPECIAL_TOKENS

CONDITIONED = "conditioned_enemy_v1"
JOINT = "unconditioned_joint_v1"


def validate_sequence_format(sequence_format: str) -> str:
    if sequence_format not in (CONDITIONED, JOINT):
        raise ValueError(f"sequence_format must explicitly select {CONDITIONED!r} or {JOINT!r}")
    return sequence_format


@dataclass(frozen=True)
class SequenceVocabulary:
    sequence_format: str
    names: tuple[str, ...]
    content_ids: tuple[int, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.names)

    @property
    def identity(self) -> str:
        payload = json.dumps((self.sequence_format, self.names, self.content_ids))
        return hashlib.sha256(payload.encode()).hexdigest()

    def token_id(self, name: str) -> int:
        return self.names.index(name)


def build_sequence_vocabulary(content: ContentVocabulary, *, sequence_format: str) -> SequenceVocabulary:
    validate_sequence_format(sequence_format)
    names = [""] * content.vocab_size
    for name, token_id in SPECIAL_TOKENS.items():
        names[token_id] = name
    for token in content.tokens:
        names[token.token_id] = token.name
    if sequence_format == JOINT:
        # Cosmetic joint-format spelling; retain the existing outcome ID.
        names[SPECIAL_TOKENS["[LOSS]"]] = "[LOSE]"
        names.extend(("[SELF]", "[ENEMY]"))
    return SequenceVocabulary(sequence_format, tuple(names), tuple(t.token_id for t in content.tokens))
