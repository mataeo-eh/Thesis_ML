"""Greedy joint windows; the same function serves preview and future preprocessing."""

from collections.abc import Iterable, Iterator

from thesis_shared.sequence_formats.common import structural_token
from thesis_shared.sequence_formats.unconditioned_joint import build_outcome_footer
from thesis_shared.sequence_formats.types import SequenceToken, SequenceWindow
from thesis_shared.vocab.sequence_vocabulary import JOINT, SequenceVocabulary


def iter_joint_windows(timesteps: Iterable[tuple[SequenceToken, ...]],
                       vocabulary: SequenceVocabulary, *, context_budget: int,
                       perspective: str, outcomes: dict[str, int]) -> Iterator[SequenceWindow]:
    """Reserve four footer slots in every window, plus END at actual replay end."""
    if vocabulary.sequence_format != JOINT or context_budget < 10:
        raise ValueError("joint windows require the joint vocabulary and a budget of at least 10")
    footer = build_outcome_footer(vocabulary, perspective=perspective, outcomes=outcomes)
    source = iter(timesteps)
    pending = next(source, None)
    if pending is None:
        raise ValueError("cannot window an empty replay")
    start = end = 0
    while pending is not None:
        tokens = [structural_token(vocabulary, "[BOS]"),
                  structural_token(vocabulary, "[DELIMITER]", start)]
        while pending is not None:
            following = next(source, None)
            terminal = following is None
            required = len(pending) + int(terminal)
            if len(tokens) + required + len(footer) > context_budget:
                if end == start:
                    raise ValueError(f"whole timestep {start} with footer needs {2 + required + len(footer)} tokens; budget={context_budget}")
                yield SequenceWindow(tuple(tokens) + footer, start, end, context_budget, False, required)
                start = end
                tokens = [structural_token(vocabulary, "[BOS]"),
                          structural_token(vocabulary, "[DELIMITER]", start)]
                if len(tokens) + required + len(footer) > context_budget:
                    raise ValueError(f"whole timestep {start} with footer needs {2 + required + len(footer)} tokens; budget={context_budget}")
            tokens.extend(pending)
            end += 1
            pending = following
            if terminal:
                tokens.extend(footer)
                tokens.append(structural_token(vocabulary, "[END]"))
                yield SequenceWindow(tuple(tokens), start, end, context_budget, True, None)
