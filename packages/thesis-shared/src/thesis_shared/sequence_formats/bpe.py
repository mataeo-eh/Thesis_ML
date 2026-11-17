"""Deterministic token-level BPE for the isolated joint preview.

Atomic content IDs (entities and upgrades/research) are symbols (not characters or UTF-8 bytes). Every other
ID is a hard boundary. Counts are adjacent-pair occurrences in the current
segmentation, including overlapping occurrences, as in ordinary BPE.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
import hashlib
import json
from functools import cached_property
import heapq
from pathlib import Path
import sqlite3

from thesis_shared.sequence_formats.types import SequenceToken
from thesis_shared.vocab.sequence_vocabulary import SequenceVocabulary


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    result = []
    index = 0
    while index < len(ids):
        if ids[index:index + 2] == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(ids[index])
            index += 1
    return tuple(result)


@dataclass(frozen=True)
class ContentBPE:
    base: SequenceVocabulary
    mergeable_ids: tuple[int, ...]
    min_occurrences: int
    # left ID, right ID, new ID, adjacent frequency at selection
    merges: tuple[tuple[int, int, int, int], ...]

    @cached_property
    def pair_ranks(self):
        return {(left, right): (rank, new) for rank, (left, right, new, _) in enumerate(self.merges)}

    def encode_ids(self, word: Iterable[int]) -> tuple[int, ...]:
        """Encode ONE content-only owner/timestep word in ranked BPE order.

        A linked list and rank heap avoid scanning every merge for every word.
        Callers must separate structural/owner/timestep boundaries first.
        """
        ids = list(word)
        if not set(ids) <= set(self.mergeable_ids):
            raise ValueError("encode_ids requires an atomic content-only word")
        following = list(range(1, len(ids))) + [-1] if ids else []
        previous = list(range(-1, len(ids) - 1))
        alive = [True] * len(ids)
        heap = []

        def offer(left):
            right = following[left]
            if right != -1 and (entry := self.pair_ranks.get((ids[left], ids[right]))) is not None:
                heapq.heappush(heap, (entry[0], left, right, ids[left], ids[right], entry[1]))

        for left in range(len(ids)):
            offer(left)
        while heap:
            _, left, right, a, b, new = heapq.heappop(heap)
            if not (alive[left] and alive[right] and following[left] == right
                    and ids[left] == a and ids[right] == b):
                continue
            ids[left] = new
            alive[right] = False
            following[left] = following[right]
            if following[right] != -1:
                previous[following[right]] = left
            if previous[left] != -1:
                offer(previous[left])
            offer(left)
        return tuple(value for index, value in enumerate(ids) if alive[index])

    @property
    def vocabulary(self) -> SequenceVocabulary:
        names = self.base.names + tuple(f"<BPE_{new}>" for _, _, new, _ in self.merges)
        return SequenceVocabulary(self.base.sequence_format, names,
                                  self.base.content_ids + tuple(m[2] for m in self.merges))

    @property
    def expansions(self) -> dict[int, tuple[int, ...]]:
        expansions = {i: (i,) for i in range(self.base.vocab_size)}
        for left, right, new, _ in self.merges:
            expansions[new] = expansions[left] + expansions[right]
        return expansions

    def artifact(self) -> dict:
        payload = {"algorithm": "content_token_bpe_v1", "base_identity": self.base.identity,
                   "base_names": self.base.names, "mergeable_ids": self.mergeable_ids,
                   "min_occurrences": self.min_occurrences, "merges": self.merges,
                   "frequency": "current adjacent pairs, overlaps included; replacement is left-to-right non-overlapping",
                   "tie_break": "lexicographically smallest pair of token IDs"}
        payload["identity"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        expansions = self.expansions
        payload["vocabulary"] = [
            {"id": i, "name": name, "atomic_ids": expansions[i]}
            for i, name in enumerate(self.vocabulary.names)]
        return payload

    @classmethod
    def from_artifact(cls, payload: dict, base: SequenceVocabulary) -> "ContentBPE":
        """Load a frozen study/preview tokenizer, rejecting altered merge semantics."""
        if payload.get("algorithm") != "content_token_bpe_v1" or payload.get("base_identity") != base.identity:
            raise ValueError("BPE artifact/base vocabulary identity mismatch")
        allowed = tuple(payload["mergeable_ids"])
        minimum = payload["min_occurrences"]
        if type(minimum) is not int or minimum < 2 or not set(allowed) <= set(base.content_ids):
            raise ValueError("invalid BPE artifact settings")
        merges = tuple(tuple(row) for row in payload["merges"])
        known = set(allowed)
        for rank, (left, right, new, frequency) in enumerate(merges):
            if (left not in known or right not in known or new != base.vocab_size + rank
                    or frequency < minimum or not all(type(i) is int for i in (left, right, new, frequency))):
                raise ValueError("invalid ranked BPE merge")
            known.add(new)
        bpe = cls(base, allowed, minimum, merges)
        if json.dumps(bpe.artifact(), sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise ValueError("BPE artifact identity or expanded vocabulary mismatch")
        return bpe

    def encode(self, tokens: Iterable[SequenceToken]) -> tuple[SequenceToken, ...]:
        result = tuple(tokens)
        for left, right, new, _ in self.merges:
            merged = []
            index = 0
            while index < len(result):
                first = result[index]
                second = result[index + 1] if index + 1 < len(result) else None
                if (second is not None and (first.token_id, second.token_id) == (left, right)
                        and (first.owner, first.timestep) == (second.owner, second.timestep)):
                    merged.append(replace(first, token_id=new, name=f"<BPE_{new}>"))
                    index += 2
                else:
                    merged.append(first)
                    index += 1
            result = tuple(merged)
        return result

    def decode(self, tokens: Iterable[SequenceToken]) -> tuple[SequenceToken, ...]:
        expansions = self.expansions
        return tuple(replace(token, token_id=i, name=self.base.names[i])
                     for token in tokens for i in expansions[token.token_id])


def fit_content_bpe(timesteps: Iterable[tuple[SequenceToken, ...]], base: SequenceVocabulary,
                   mergeable_ids: Iterable[int], *, min_occurrences: int) -> ContentBPE:
    """Fit only supplied replay spans; retain every base ID, append ranked merges."""
    if type(min_occurrences) is not int or min_occurrences < 2:
        raise ValueError("BPE min_occurrences must be an integer of at least 2")
    allowed = frozenset(mergeable_ids)
    if not allowed.issubset(base.content_ids):
        raise ValueError("mergeable IDs must belong to the base content vocabulary")
    spans: Counter = Counter()
    for timestep in timesteps:
        span = []
        previous = None
        for token in timestep:
            key = (token.owner, token.timestep)
            if token.token_id not in allowed or (previous is not None and key != previous):
                if span:
                    spans[tuple(span)] += 1
                    span = []
            if token.token_id in allowed:
                span.append(token.token_id)
            previous = key
        if span:
            spans[tuple(span)] += 1
    with ContentBPETrainer(":memory:", base, allowed, min_occurrences=min_occurrences) as trainer:
        trainer.initialize(spans.items())
        return trainer.fit_to(None)


class ContentBPETrainer:
    """Incremental exact BPE with a disk-backed inverted pair index.

    The preview uses an in-memory SQLite store; corpus studies use a scratch
    file. Only words containing the chosen pair are rewritten. Each merge is
    one transaction, so interruption rolls back at most the active merge.
    The caller's identity binds a persistent store to its training corpus.
    """

    def __init__(self, path: str | Path, base: SequenceVocabulary, mergeable_ids: Iterable[int],
                 *, min_occurrences: int, identity: str = ""):
        if type(min_occurrences) is not int or min_occurrences < 2:
            raise ValueError("BPE min_occurrences must be an integer of at least 2")
        self.base = base
        self.allowed = tuple(sorted(set(mergeable_ids)))
        if not set(self.allowed) <= set(base.content_ids):
            raise ValueError("mergeable IDs must belong to the base content vocabulary")
        self.minimum = min_occurrences
        self.identity = json.dumps(["indexed_bpe_v1", base.identity, self.allowed,
                                    min_occurrences, identity])
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA cache_size=-32768")  # bounded 32 MiB page cache
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (identity TEXT);
            CREATE TABLE IF NOT EXISTS words (id INTEGER PRIMARY KEY, ids TEXT UNIQUE, weight INTEGER);
            CREATE TABLE IF NOT EXISTS members (
                left_id INTEGER, right_id INTEGER, word INTEGER, occurrences INTEGER,
                PRIMARY KEY(left_id, right_id, word));
            CREATE TABLE IF NOT EXISTS pairs (
                left_id INTEGER, right_id INTEGER, frequency INTEGER,
                PRIMARY KEY(left_id, right_id));
            CREATE INDEX IF NOT EXISTS best_pair ON pairs(frequency DESC, left_id, right_id);
            CREATE TABLE IF NOT EXISTS merges (
                new_id INTEGER PRIMARY KEY, left_id INTEGER, right_id INTEGER,
                frequency INTEGER, deleted INTEGER);
        """)
        row = self.db.execute("SELECT identity FROM metadata").fetchone()
        if row and row[0] != self.identity:
            self.db.close()
            raise ValueError("BPE fitting-store identity differs from the training corpus/settings")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def initialize(self, words: Iterable[tuple[tuple[int, ...], int]]) -> None:
        """Load weighted atomic words once, streaming without a corpus-sized list."""
        if self.db.execute("SELECT 1 FROM metadata").fetchone():
            return
        allowed = set(self.allowed)
        with self.db:
            for word, weight in words:
                if not set(word) <= allowed or type(weight) is not int or weight < 1:
                    raise ValueError("invalid atomic word or multiplicity")
                if not word:
                    continue
                self.db.execute("INSERT INTO words(ids,weight) VALUES (?,?) "
                                "ON CONFLICT(ids) DO UPDATE SET weight=weight+excluded.weight",
                                (json.dumps(word), weight))
            for word_id, text, weight in self.db.execute("SELECT id,ids,weight FROM words"):
                word = json.loads(text)
                for (left, right), count in Counter(zip(word, word[1:])).items():
                    self.db.execute("INSERT INTO members VALUES (?,?,?,?)", (left, right, word_id, count))
                    self._adjust(left, right, count * weight)
            self.db.execute("INSERT INTO metadata VALUES (?)", (self.identity,))

    def _adjust(self, left, right, delta):
        self.db.execute("INSERT INTO pairs VALUES (?,?,?) ON CONFLICT(left_id,right_id) "
                        "DO UPDATE SET frequency=frequency+excluded.frequency", (left, right, delta))

    def snapshot(self) -> ContentBPE:
        merges = tuple(self.db.execute("SELECT left_id,right_id,new_id,frequency FROM merges ORDER BY new_id"))
        return ContentBPE(self.base, self.allowed, self.minimum, merges)

    def fit_to(self, vocabulary_size: int | None, *, progress=None) -> ContentBPE:
        """Extend the ranked vocabulary to a TOTAL size, or pair exhaustion."""
        if vocabulary_size is not None and (type(vocabulary_size) is not int
                                           or vocabulary_size < self.base.vocab_size):
            raise ValueError("vocabulary_size must include every base ID")
        if not self.db.execute("SELECT 1 FROM metadata").fetchone():
            raise ValueError("initialize the trainer first")
        size = self.base.vocab_size + self.db.execute("SELECT COUNT(*) FROM merges").fetchone()[0]
        if vocabulary_size is not None and size > vocabulary_size:
            raise ValueError("fitting store has already passed this vocabulary size")
        while vocabulary_size is None or size < vocabulary_size:
            row = self.db.execute("SELECT left_id,right_id,frequency FROM pairs "
                                  "WHERE frequency>=? ORDER BY frequency DESC,left_id,right_id LIMIT 1",
                                  (self.minimum,)).fetchone()
            if row is None:
                break
            left, right, frequency = row
            deleted, last = 0, -1
            with self.db:
                while True:
                    affected = self.db.execute(
                        "SELECT w.id,w.ids,w.weight FROM members m JOIN words w ON w.id=m.word "
                        "WHERE m.left_id=? AND m.right_id=? AND w.id>? ORDER BY w.id LIMIT 256",
                        (left, right, last)).fetchall()
                    if not affected:
                        break
                    for word_id, text, weight in affected:
                        word = tuple(json.loads(text))
                        merged = _merge(word, (left, right), size)
                        deleted += (len(word) - len(merged)) * weight
                        before, after = Counter(zip(word, word[1:])), Counter(zip(merged, merged[1:]))
                        for pair in before.keys() | after.keys():
                            delta = after[pair] - before[pair]
                            if delta:
                                self._adjust(*pair, delta * weight)
                            if after[pair]:
                                self.db.execute("INSERT INTO members VALUES (?,?,?,?) "
                                                "ON CONFLICT(left_id,right_id,word) DO UPDATE "
                                                "SET occurrences=excluded.occurrences", (*pair, word_id, after[pair]))
                            else:
                                self.db.execute("DELETE FROM members WHERE left_id=? AND right_id=? AND word=?",
                                                (*pair, word_id))
                        self.db.execute("UPDATE words SET ids=? WHERE id=?", (json.dumps(merged), word_id))
                    last = affected[-1][0]
                self.db.execute("INSERT INTO merges VALUES (?,?,?,?,?)", (size, left, right, frequency, deleted))
            size += 1
            if progress is not None:
                progress(size, frequency, deleted)
        return self.snapshot()
