import json
from pathlib import Path

import pandas as pd

from thesis_ml.config import load_config
from thesis_ml.serialize import (
    deserialize_counts,
    records_to_plain,
    serialize_snapshot,
    snapshot_content_counts,
)
from thesis_ml.vocab.content_vocab import load_content_vocabulary
from thesis_ml.vocab.special_tokens import CONTENT_TOKEN_OFFSET, DELIMITER_ID, SPECIAL_TOKENS


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "match_4745722_game_state.parquet"
CONFIG = ROOT / "config" / "default.yaml"
TOKEN_DICTIONARY = ROOT / "data" / "Token_Dictionary.json"


def _snapshot() -> pd.Series:
    frame = pd.read_parquet(FIXTURE)
    return frame.iloc[1]


def test_round_trip_serialization_fidelity() -> None:
    config = load_config(CONFIG)
    vocab = load_content_vocabulary(TOKEN_DICTIONARY)
    snapshot = _snapshot()

    records = serialize_snapshot(snapshot, config, vocab, perspective_player="p1")
    decoded = deserialize_counts(records, vocab)

    assert decoded == [snapshot_content_counts(snapshot)]


def test_serialization_and_vocabulary_are_deterministic() -> None:
    config = load_config(CONFIG)
    snapshot = _snapshot()
    vocab_a = load_content_vocabulary(TOKEN_DICTIONARY)
    vocab_b = load_content_vocabulary(TOKEN_DICTIONARY)

    records_a = serialize_snapshot(snapshot, config, vocab_a, perspective_player="p1")
    records_b = serialize_snapshot(snapshot, config, vocab_b, perspective_player="p1")

    assert vocab_a.tokens == vocab_b.tokens
    assert json.dumps(records_to_plain(records_a), sort_keys=True) == json.dumps(
        records_to_plain(records_b),
        sort_keys=True,
    )


def test_canonical_order_and_delimiter() -> None:
    config = load_config(CONFIG)
    vocab = load_content_vocabulary(TOKEN_DICTIONARY)
    records = serialize_snapshot(_snapshot(), config, vocab, perspective_player="p1")

    assert records[-1].token_id == DELIMITER_ID
    assert sum(1 for record in records if record.token_id == DELIMITER_ID) == 1

    entity_records = [record for record in records if record.token_kind == "entity"]
    sort_keys = [
        (
            vocab.source_id_for(record.token_name),
            int(record.instance_id or 0),
            record.owner or "",
            record.entity_type or "",
        )
        for record in entity_records
    ]
    assert sort_keys == sorted(sort_keys)


def test_vocabulary_contains_only_allowed_token_identities() -> None:
    vocab = load_content_vocabulary(TOKEN_DICTIONARY)
    names = [token.name for token in vocab.tokens]

    assert min(token.token_id for token in vocab.tokens) == CONTENT_TOKEN_OFFSET
    assert set(SPECIAL_TOKENS.values()).isdisjoint({token.token_id for token in vocab.tokens})
    assert "stimpack" in names
    assert "scv" in names
    assert "game_loop" not in names
    assert "timestamp_seconds" not in names

    banned_fragments = (
        "_count",
        "minerals",
        "vespene",
        "supply_used",
        "supply_cap",
        "collection_rate",
        "timestamp",
        "game_loop",
        "pos_",
        "(x,y,z)",
        "region",
    )
    for name in names:
        assert all(fragment not in name for fragment in banned_fragments), name
