"""Clean two-player timesteps and a separate generated window-outcome footer."""

from collections.abc import Iterable, Iterator

import pandas as pd

from thesis_shared.config import ProjectConfig
from thesis_shared.serialize import parse_entity_columns, serialize_snapshot
from thesis_shared.sequence_formats.common import enemy_player, owner_content, structural_token
from thesis_shared.sequence_formats.types import SequenceToken
from thesis_shared.vocab.content_vocab import ContentVocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, SequenceVocabulary


def iter_joint_timesteps(snapshots: Iterable[pd.Series], config: ProjectConfig,
                        content: ContentVocabulary, vocabulary: SequenceVocabulary, *,
                        perspective: str) -> Iterator[tuple[SequenceToken, ...]]:
    """Serialize complete timesteps without outcomes, fog, or feature conditioning."""
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
            tokens.extend(owner_content(records, owner, timestep))
        tokens.append(structural_token(vocabulary, "[DELIMITER]", timestep))
        yield tuple(tokens)


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
