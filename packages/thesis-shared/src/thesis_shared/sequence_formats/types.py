"""Immutable sequence values shared by previews and future arm adapters.

Everything here is plain Python (tuples of ints/bools), so both the PyTorch arm
and the TensorFlow arm can convert a window into their own tensor type at the
last possible moment.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceToken:
    token_id: int
    name: str
    timestep: int | None = None
    owner: str | None = None
    role: str = "structure"


@dataclass(frozen=True)
class JointReplay:
    """One replay, from one perspective, ready to be windowed.

    Attributes:
        replay_id: Stable replay name (inspection metadata only, never a feature).
        perspective: "p1" or "p2"; the player shown as [SELF].
        races: Player -> race token name, e.g. {"p1": "[TERRAN]", "p2": "[ZERG]"}.
        outcomes: Player -> outcome token ID ([WIN] or [LOSE]).
        timesteps: Serialized whole timesteps, in replay order. May be a lazy iterator.
    """

    replay_id: str
    perspective: str
    races: dict[str, str]
    outcomes: dict[str, int]
    timesteps: Iterable[tuple["SequenceToken", ...]]


@dataclass(frozen=True)
class WindowSegment:
    """The contiguous span of one replay inside a (possibly packed) window.

    Attributes:
        replay_id: Which replay this span came from.
        perspective: The perspective used for that replay.
        start_position: Window index of the segment's leading [BOS].
        start_timestep: First replay timestep in the span (zero-based, inclusive).
        end_timestep: Replay timestep after the last one in the span (exclusive).
        reached_replay_end: True when the span ends at the replay's real end,
            i.e. it closes with the outcome footer followed by [END].
    """

    replay_id: str
    perspective: str
    start_position: int
    start_timestep: int
    end_timestep: int
    reached_replay_end: bool


@dataclass(frozen=True)
class SequenceWindow:
    """One unpadded training sequence of at most ``context_budget`` tokens.

    A window holds one or more replay segments. It holds several only when
    replay packing is enabled and a replay ended with room left over.
    """

    tokens: tuple[SequenceToken, ...]
    segments: tuple[WindowSegment, ...]
    context_budget: int
    # Tokens the next timestep would have needed (inspection only); None when
    # the window closed because its last replay ended, not because it was full.
    next_timestep_tokens: int | None
    # Leading positions that diffusion clamps as observed input:
    # [BOS] [SELF] <self race> [ENEMY] <enemy race>.
    clamped_prefix_length: int

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token_id for token in self.tokens)

    @property
    def diffusion_clamp_mask(self) -> tuple[bool, ...]:
        """True where diffusion keeps the ground-truth token and does not score it.

        Clamping is by POSITION, not by ID. Only the window's first header is
        clamped. A later packed replay's [BOS]/race header, and any noise-
        inserted BOS or race token, stays mutable and scored, so training
        matches unconditional sampling where only the leading prefix is given.
        """
        clamped = min(self.clamped_prefix_length, len(self.tokens))
        return (True,) * clamped + (False,) * (len(self.tokens) - clamped)

    @property
    def diffusion_loss_mask(self) -> tuple[bool, ...]:
        """Positions diffusion scores: every position that is not clamped."""
        return tuple(not clamped for clamped in self.diffusion_clamp_mask)

    @property
    def ar_loss_mask(self) -> tuple[bool, ...]:
        """Positions the autoregressive arm scores as next-token targets.

        Position 0 ([BOS]) has no preceding context, so it is never a target.
        Every other token, including the race header, is an ordinary target.
        """
        return (False,) + (True,) * (len(self.tokens) - 1)
