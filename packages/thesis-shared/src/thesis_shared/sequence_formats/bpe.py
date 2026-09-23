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
    merges = []
    while True:
        counts: Counter = Counter()
        for span, multiplicity in spans.items():
            for pair in zip(span, span[1:]):
                counts[pair] += multiplicity
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        frequency = counts[pair]
        if frequency < min_occurrences:
            break
        new = base.vocab_size + len(merges)
        merges.append((*pair, new, frequency))
        updated: Counter = Counter()
        for span, multiplicity in spans.items():
            updated[_merge(span, pair, new)] += multiplicity
        spans = updated
    return ContentBPE(base, tuple(sorted(allowed)), min_occurrences, tuple(merges))
