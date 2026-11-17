"""Clean two-player timesteps, the per-replay race header, and the outcome footer.

Within-timestep token order (the binding rule, documented in
PRETRAINING_PIPELINE.md): inside each player's block, content tokens appear in
ascending content token ID. That order comes from ``serialize_snapshot``, which
sorts entities by the Token_Dictionary source ID (SC2 game-data type ID), then
upgrades/research by source ID. Content token IDs are assigned in that same
source-ID order, so "ascending token ID" and "serializer order" are one rule.
``iter_joint_timesteps`` checks it on every block.
"""

from collections.abc import Iterable, Iterator

import pandas as pd

from thesis_shared.config import ProjectConfig
from thesis_shared.serialize import parse_entity_columns, serialize_snapshot
from thesis_shared.sequence_formats.common import enemy_player, owner_content, structural_token
from thesis_shared.sequence_formats.types import SequenceToken
from thesis_shared.vocab.content_vocab import ContentVocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, RACE_TOKENS, SequenceVocabulary

# [BOS] [SELF] <self race> [ENEMY] <enemy race>
REPLAY_HEADER_LENGTH = 5
# [SELF] <self outcome> [ENEMY] <enemy outcome>
OUTCOME_FOOTER_LENGTH = 4


def iter_joint_timesteps(snapshots: Iterable[pd.Series], config: ProjectConfig,
                        content: ContentVocabulary, vocabulary: SequenceVocabulary, *,
                        perspective: str) -> Iterator[tuple[SequenceToken, ...]]:
    """Serialize complete timesteps without outcomes, fog, or feature conditioning.

    Each yielded timestep is ``[SELF] self-content [ENEMY] enemy-content [DELIMITER]``.

    Raises:
        ValueError: If a player's content is not in ascending token-ID order.
            That would mean the canonical serializer order changed.
    """
    enemy = enemy_player(perspective)
    if vocabulary.sequence_format != JOINT:
        raise ValueError("joint presentation requires the joint vocabulary")
    groups = None
    for timestep, snapshot in enumerate(snapshots):
        if groups is None:
            groups = parse_entity_columns(snapshot.index)
        records = serialize_snapshot(snapshot, config, content,
                                     perspective_player=perspective, entity_groups=groups)
        tokens = []
        for owner, marker in ((perspective, "[SELF]"), (enemy, "[ENEMY]")):
            tokens.append(structural_token(vocabulary, marker, timestep, owner, "ownership"))
            block = owner_content(records, owner, timestep)
            ids = [token.token_id for token in block]
            if ids != sorted(ids):
                raise ValueError(f"timestep {timestep} {owner} content is not in ascending token-ID order")
            tokens.extend(block)
        tokens.append(structural_token(vocabulary, "[DELIMITER]", timestep))
        yield tuple(tokens)


def build_replay_header(vocabulary: SequenceVocabulary, *, perspective: str,
                        races: dict[str, str]) -> tuple[SequenceToken, ...]:
    """Build ``[BOS] [SELF] <self race> [ENEMY] <enemy race>`` for one replay segment.

    With no surviving content tokens, an unconditioned model has nothing else
    that tells it each player's race. This header supplies it: diffusion clamps
    it in a window's leading position, and AR treats it as ordinary tokens.

    Parameters:
        vocabulary: The joint vocabulary (must contain the race tokens).
        perspective: "p1" or "p2"; that player's race follows [SELF].
        races: Player -> race token name, e.g. {"p1": "[TERRAN]", "p2": "[ZERG]"}.

    Returns:
        The five header tokens.

    Calls: enemy_player, structural_token.
    """
    enemy = enemy_player(perspective)
    if vocabulary.sequence_format != JOINT:
        raise ValueError("joint replay header requires the joint vocabulary")
    if set(races) != {"p1", "p2"} or not set(races.values()) <= set(RACE_TOKENS.values()):
        raise ValueError(f"races must map p1 and p2 to race tokens {sorted(RACE_TOKENS.values())}")
    tokens = [structural_token(vocabulary, "[BOS]")]
    for owner, marker in ((perspective, "[SELF]"), (enemy, "[ENEMY]")):
        tokens.append(structural_token(vocabulary, marker, owner=owner, role="ownership"))
        tokens.append(structural_token(vocabulary, races[owner], owner=owner, role="race"))
    return tuple(tokens)


def build_outcome_footer(vocabulary: SequenceVocabulary, *, perspective: str,
                         outcomes: dict[str, int]) -> tuple[SequenceToken, ...]:
    """Validate recorded labels, not model predictions; all four targets are scored."""
    enemy = enemy_player(perspective)
    if vocabulary.sequence_format != JOINT:
        raise ValueError("joint outcome footer requires the joint vocabulary")
    if set(outcomes) != {"p1", "p2"} or set(outcomes.values()) != {
        vocabulary.token_id("[WIN]"), vocabulary.token_id("[LOSE]")
    }:
        raise ValueError("joint format requires one recorded win and one recorded loss")
    tokens = []
    for owner, marker in ((perspective, "[SELF]"), (enemy, "[ENEMY]")):
        tokens.append(structural_token(vocabulary, marker, owner=owner, role="ownership"))
        tokens.append(structural_token(vocabulary, vocabulary.names[outcomes[owner]],
                                       owner=owner, role="outcome"))
    return tuple(tokens)
