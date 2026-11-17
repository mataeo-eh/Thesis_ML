"""Presentation-specific vocabulary overlays; existing content IDs never move.

The joint (unconditioned) format appends its own structural tokens after the
full content vocabulary: two ownership markers followed by four race tokens.
Appending, rather than inserting, keeps every legacy ID stable.
"""

from dataclasses import dataclass
import hashlib
import json

from thesis_shared.vocab.content_vocab import ContentVocabulary
from thesis_shared.vocab.special_tokens import SPECIAL_TOKENS

CONDITIONED = "conditioned_enemy_v1"
JOINT = "unconditioned_joint_v1"

# Ownership markers that open each player's block.
OWNERSHIP_TOKENS = ("[SELF]", "[ENEMY]")

# Replay-metadata race string -> race token name. Alphabetical by token name,
# which is also the order the tokens are appended to the joint vocabulary.
# "Random" is kept as its own token even though replay metadata normally
# records the race a Random player actually received.
RACE_TOKENS = {
    "Protoss": "[PROTOSS]",
    "Random": "[RANDOM]",
    "Terran": "[TERRAN]",
    "Zerg": "[ZERG]",
}


def validate_sequence_format(sequence_format: str) -> str:
    if sequence_format not in (CONDITIONED, JOINT):
        raise ValueError(f"sequence_format must explicitly select {CONDITIONED!r} or {JOINT!r}")
    return sequence_format


def race_token_name(race: str) -> str:
    """Map a replay-metadata race string (e.g. "Terran") to its race token name.

    Parameters:
        race: The race exactly as replay metadata records it.

    Returns:
        The race token name, e.g. "[TERRAN]".

    Raises:
        ValueError: If the race is not one of the four known races. An unknown
            race never silently falls back to another token.
    """
    if race not in RACE_TOKENS:
        raise ValueError(f"unknown race {race!r}; expected one of {sorted(RACE_TOKENS)}")
    return RACE_TOKENS[race]


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
    """Build the conditioned or joint vocabulary view over the content vocabulary.

    Joint layout: [special tokens][content tokens][SELF][ENEMY][PROTOSS][RANDOM][TERRAN][ZERG].
    """
    validate_sequence_format(sequence_format)
    names = [""] * content.vocab_size
    for name, token_id in SPECIAL_TOKENS.items():
        names[token_id] = name
    for token in content.tokens:
        names[token.token_id] = token.name
    if sequence_format == JOINT:
        # Cosmetic joint-format spelling; retain the existing outcome ID.
        names[SPECIAL_TOKENS["[LOSS]"]] = "[LOSE]"
        names.extend(OWNERSHIP_TOKENS)
        names.extend(sorted(RACE_TOKENS.values()))
    return SequenceVocabulary(sequence_format, tuple(names), tuple(t.token_id for t in content.tokens))
