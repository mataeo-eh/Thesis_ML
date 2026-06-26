"""Deterministic content vocabulary for entity and upgrade tokens."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from thesis_ml.vocab.special_tokens import CONTENT_TOKEN_OFFSET, SPECIAL_TOKENS

UPGRADE_SOURCE_ID_OFFSET = 100_000


@dataclass(frozen=True)
class ContentToken:
    name: str
    token_id: int
    source_id: int
    kind: str


@dataclass(frozen=True)
class ContentVocabulary:
    tokens: tuple[ContentToken, ...]

    def __post_init__(self) -> None:
        names = [token.name for token in self.tokens]
        ids = [token.token_id for token in self.tokens]
        if len(names) != len(set(names)):
            raise ValueError("content vocabulary has duplicate names")
        if len(ids) != len(set(ids)):
            raise ValueError("content vocabulary has duplicate token IDs")
        if ids and min(ids) < CONTENT_TOKEN_OFFSET:
            raise ValueError("content token IDs must start at CONTENT_TOKEN_OFFSET")

    @property
    def name_to_id(self) -> dict[str, int]:
        return {token.name: token.token_id for token in self.tokens}

    @property
    def id_to_name(self) -> dict[int, str]:
        return {token.token_id: token.name for token in self.tokens}

    @property
    def name_to_source_id(self) -> dict[str, int]:
        return {token.name: token.source_id for token in self.tokens}

    @property
    def name_to_kind(self) -> dict[str, str]:
        return {token.name: token.kind for token in self.tokens}

    @property
    def vocab_size(self) -> int:
        max_special = max(SPECIAL_TOKENS.values())
        max_content = max((token.token_id for token in self.tokens), default=max_special)
        return max(max_special, max_content) + 1

    def token_id_for(self, name: str) -> int:
        try:
            return self.name_to_id[name]
        except KeyError as exc:
            raise KeyError(f"unknown content token: {name}") from exc

    def token_name_for(self, token_id: int) -> str:
        try:
            return self.id_to_name[token_id]
        except KeyError as exc:
            raise KeyError(f"unknown content token ID: {token_id}") from exc

    def source_id_for(self, name: str) -> int:
        try:
            return self.name_to_source_id[name]
        except KeyError as exc:
            raise KeyError(f"unknown content token: {name}") from exc


def normalize_content_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("unknown(") and normalized.endswith(")"):
        return f"unknown_{normalized[8:-1]}"
    return normalized


def load_content_vocabulary(token_dictionary_path: str | Path) -> ContentVocabulary:
    path = Path(token_dictionary_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return build_content_vocabulary(raw["tokens"])


def build_content_vocabulary(raw_tokens: dict[str, str] | Iterable[tuple[str, str]]) -> ContentVocabulary:
    items = raw_tokens.items() if isinstance(raw_tokens, dict) else raw_tokens
    parsed: list[tuple[int, str]] = []
    for raw_source_id, raw_name in items:
        source_id = int(raw_source_id)
        name = normalize_content_name(str(raw_name))
        parsed.append((source_id, name))

    tokens: list[ContentToken] = []
    seen: set[str] = set()
    for index, (source_id, name) in enumerate(sorted(parsed, key=lambda item: (item[0], item[1]))):
        if name in seen:
            continue
        seen.add(name)
        kind = "upgrade" if source_id >= UPGRADE_SOURCE_ID_OFFSET else "entity"
        tokens.append(
            ContentToken(
                name=name,
                token_id=CONTENT_TOKEN_OFFSET + len(tokens),
                source_id=source_id,
                kind=kind,
            )
        )

    return ContentVocabulary(tokens=tuple(tokens))
