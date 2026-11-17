"""Greedy whole-timestep joint windows, optionally packing consecutive replays.

The same function serves the local preview and future cloud preprocessing, and
returns plain-Python windows so both arms consume identical sequences.

Window grammar (one replay segment):

    [BOS] [SELF] <race> [ENEMY] <race> [DELIMITER]
    ([SELF] self-content [ENEMY] enemy-content [DELIMITER])+
    [SELF] <self outcome> [ENEMY] <enemy outcome>
    [END]            <- only when the segment reaches the replay's real end

With ``pack_replays=True``, a window whose replay ended with room to spare
continues directly with the next replay's segment (starting at its own [BOS]),
provided at least one whole timestep of it fits.
"""

from collections.abc import Iterable, Iterator

from thesis_shared.sequence_formats.common import structural_token
from thesis_shared.sequence_formats.unconditioned_joint import (
    OUTCOME_FOOTER_LENGTH,
    REPLAY_HEADER_LENGTH,
    build_outcome_footer,
    build_replay_header,
)
from thesis_shared.sequence_formats.types import JointReplay, SequenceToken, SequenceWindow, WindowSegment
from thesis_shared.vocab.sequence_vocabulary import JOINT, SequenceVocabulary

# Header + opening delimiter + footer + END: fixed overhead of a complete replay segment.
SEGMENT_OVERHEAD = REPLAY_HEADER_LENGTH + 1 + OUTCOME_FOOTER_LENGTH + 1


def _flag_last(items: Iterable) -> Iterator[tuple[object, bool]]:
    """Yield (item, is_last) pairs while reading only one item ahead."""
    source = iter(items)
    try:
        current = next(source)
    except StopIteration:
        return
    for following in source:
        yield current, False
        current = following
    yield current, True


def iter_joint_windows(replays: Iterable[JointReplay], vocabulary: SequenceVocabulary, *,
                       context_budget: int, pack_replays: bool) -> Iterator[SequenceWindow]:
    """Cut replays into windows of whole timesteps, at most ``context_budget`` tokens each.

    Parameters:
        replays: Replays in a fixed, deterministic order. Each is read once, lazily.
        vocabulary: The joint vocabulary.
        context_budget: Maximum tokens per window (no padding is added).
        pack_replays: If True, the tail of a window after one replay's [END] is
            filled with the start of the next replay. If False, every window
            holds exactly one replay segment.

    Yields:
        SequenceWindow objects in order. Windows never split a timestep, and each
        replay segment always ends with its four-token outcome footer.

    Raises:
        ValueError: For a non-joint vocabulary, a budget too small for one
            segment's overhead, an empty replay, or a single timestep too large
            to fit in an empty window.

    Calls: build_replay_header, build_outcome_footer, structural_token.
    """
    if vocabulary.sequence_format != JOINT or context_budget < SEGMENT_OVERHEAD:
        raise ValueError(f"joint windows require the joint vocabulary and a budget of at least {SEGMENT_OVERHEAD}")

    tokens: list[SequenceToken] = []       # tokens of the window being filled
    segments: list[WindowSegment] = []     # finished segments in that window

    def finish_window(next_timestep_tokens: int | None) -> SequenceWindow:
        return SequenceWindow(tuple(tokens), tuple(segments), context_budget,
                              next_timestep_tokens, REPLAY_HEADER_LENGTH)

    for replay in replays:
        header = build_replay_header(vocabulary, perspective=replay.perspective, races=replay.races)
        footer = build_outcome_footer(vocabulary, perspective=replay.perspective, outcomes=replay.outcomes)
        segment_start = None               # window position of this replay's open segment, if any
        start = end = 0                    # replay timestep span of the open segment
        for step, terminal in _flag_last(replay.timesteps):
            # Room this timestep needs, including the footer that must close the
            # segment and END if this is the replay's last timestep.
            required = len(step) + int(terminal)
            closing = required + len(footer)

            # The open segment cannot take this timestep: close it and the window.
            if segment_start is not None and len(tokens) + closing > context_budget:
                tokens.extend(footer)
                segments.append(WindowSegment(replay.replay_id, replay.perspective,
                                              segment_start, start, end, False))
                yield finish_window(required)
                tokens, segments = [], []
                segment_start = None
                start = end

            # Open a segment, first closing a packed window with no room left.
            if segment_start is None:
                opening = len(header) + 1
                if tokens and len(tokens) + opening + closing > context_budget:
                    yield finish_window(required)
                    tokens, segments = [], []
                if len(tokens) + opening + closing > context_budget:
                    raise ValueError(f"{replay.replay_id}: whole timestep {start} with header and footer "
                                     f"needs {opening + closing} tokens; budget={context_budget}")
                segment_start = len(tokens)
                tokens.extend(header)
                tokens.append(structural_token(vocabulary, "[DELIMITER]", start))

            tokens.extend(step)
            end += 1

            if terminal:
                tokens.extend(footer)
                tokens.append(structural_token(vocabulary, "[END]"))
                segments.append(WindowSegment(replay.replay_id, replay.perspective,
                                              segment_start, start, end, True))
                if not pack_replays:
                    yield finish_window(None)
                    tokens, segments = [], []

        if end == 0:
            raise ValueError(f"cannot window an empty replay: {replay.replay_id}")

    # The last replay ended with a partly filled packed window.
    if tokens:
        yield finish_window(None)
