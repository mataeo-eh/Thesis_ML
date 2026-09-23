"""Immutable sequence values shared by previews and future arm adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceToken:
    token_id: int
    name: str
    timestep: int | None = None
    owner: str | None = None
    role: str = "structure"


@dataclass(frozen=True)
class SequenceWindow:
    tokens: tuple[SequenceToken, ...]
    start_timestep: int
    end_timestep: int
    context_budget: int
    reached_replay_end: bool
    next_timestep_tokens: int | None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token_id for token in self.tokens)

    @property
    def loss_mask(self) -> tuple[bool, ...]:
        # Clamp by POSITION, not ID: a noise-inserted BOS elsewhere is mutable.
        return (False,) + (True,) * (len(self.tokens) - 1)
