"""Tests for the `config.sampler.outcome_last` fine-tune constraint (Worker 2).

These tests prove that when `outcome_last` is True the sampler holds canvas
position 0 (the win/loss outcome token) back until every other position `[1:]`
is committed, and that when the flag is False the pre-training behavior is
unchanged (position 0 is committed as freely as any other position).

The tests drive `sample_canvas` with a deterministic stub model
(`OutcomeStubModel`) so the commit order is fully predictable and no real
network weights are needed. The stub mirrors the fixture pattern used by
`tests/test_sampler.py`.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.inference.sampler import sample_canvas
from thesis_ml.train.train import make_synthetic_examples


class OutcomeStubModel(nn.Module):
    """Deterministic stand-in model that emits fixed per-position canvas logits.

    Every forward call returns the same logits regardless of the current canvas,
    so the sampler's commit order is decided purely by the confidence/entropy
    values these logits imply. That makes the outcome-last ordering easy to
    assert.

    Parameters
    ----------
    canvas_budget:
        Number of canvas positions. Position 0 is the outcome token.
    vocab_size:
        Size of the token vocabulary the logits span.
    top_logit:
        Logit placed on each position's target token for positions `[1:]`. A
        large value makes those positions high-confidence / low-entropy so they
        commit readily.
    outcome_logit:
        Logit placed on position 0's target token. Set this LOW to make the
        outcome token low-confidence, which lets us prove the outcome-last force
        path commits it even when a confidence threshold would otherwise block
        it.

    This model is called by `sample_canvas` during the denoising loop.
    """

    def __init__(
        self,
        *,
        canvas_budget: int,
        vocab_size: int = 128,
        top_logit: float = 8.0,
        outcome_logit: float = 8.0,
    ) -> None:
        super().__init__()
        self.canvas_budget = canvas_budget
        self.vocab_size = vocab_size
        self.top_logit = top_logit
        self.outcome_logit = outcome_logit

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask=None,
        canvas_attention_mask=None,
        input_records=None,
        input_features=None,
        canvas_self_conditioning="missing",
    ):
        """Return fixed logits for the input+canvas sequence.

        Position 0's target token gets `outcome_logit`; every other canvas
        position's target token gets `top_logit`. Target token ids are chosen
        arbitrarily but deterministically (position index + 10) so `predicted`
        is well defined.
        """
        batch, canvas_len = canvas_token_ids.shape
        input_len = input_token_ids.shape[1]
        logits = torch.zeros(
            batch, input_len + canvas_len, self.vocab_size, device=canvas_token_ids.device
        )
        for position in range(canvas_len):
            target_token = 10 + position  # arbitrary, deterministic, distinct
            logit_value = self.outcome_logit if position == 0 else self.top_logit
            logits[:, input_len + position, target_token] = logit_value
        return SimpleNamespace(logits=logits)


def _config(*, canvas_budget: int, outcome_last: bool, confidence_threshold: float = 0.0) -> ProjectConfig:
    """Build a tiny config with the outcome-last flag toggled.

    Uses `entropy_bound=0.0` with `min_commit_per_step=1` so exactly ONE canvas
    position commits per step. That gradual commit schedule makes the ordering
    (position 0 held to last) directly observable across trace steps.
    """
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=canvas_budget),
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64, self_conditioning=True),
        sampler=replace(
            config.sampler,
            max_steps=20,
            entropy_bound=0.0,
            confidence_threshold=confidence_threshold,
            min_commit_per_step=1,
            outcome_last=outcome_last,
        ),
    )


def _batch(config: ProjectConfig, *, count: int = 1):
    """Collate `count` synthetic examples into a diffusion batch.

    `make_synthetic_examples` builds PRE-TRAINING fixtures (absent input,
    collapsed labels), so the batch is collated in pre-training mode; the
    outcome-last constraint under test is a sampler behavior that applies
    identically in both modes.
    """
    examples = make_synthetic_examples(config, count=count)
    return collate_diffusion_examples(examples, debut_mode=False)


def _first_commit_steps(trace, row: int) -> list[int]:
    """Return, per canvas position, the 1-based step at which it first committed.

    Reads the cumulative `committed_mask` in each `SamplerStep` to find the first
    step where each position became True. A position that never commits is
    reported as a very large sentinel so ordering assertions still make sense.
    """
    canvas_len = trace[0].committed_mask.shape[1]
    first_step = [10**9] * canvas_len
    for step in trace:
        committed_row = step.committed_mask[row]
        for position in range(canvas_len):
            if committed_row[position].item() and first_step[position] == 10**9:
                first_step[position] = step.step
    return first_step


def test_outcome_last_true_commits_position_zero_last() -> None:
    """With the flag ON, position 0 is the last position to commit."""
    config = _config(canvas_budget=5, outcome_last=True)
    model = OutcomeStubModel(canvas_budget=5)

    output = sample_canvas(model, _batch(config), config)

    # Every position eventually commits.
    assert output.committed_mask.all()

    # Core invariant: at no trace step is position 0 committed while any of the
    # other positions [1:] is still masked. In other words position 0 only ever
    # commits together-with-or-after the rest.
    for step in output.trace:
        if step.committed_mask[0, 0].item():
            assert step.committed_mask[0, 1:].all(), (
                f"position 0 committed at step {step.step} before [1:] finished"
            )

    # Position 0 is the last (maximum) first-commit step across all positions.
    first_steps = _first_commit_steps(output.trace, row=0)
    assert first_steps[0] == max(first_steps)
    # And it is strictly after at least the earliest committed position, proving
    # it was genuinely held back rather than committed in the opening step.
    assert first_steps[0] > min(first_steps[1:])


def test_outcome_last_false_leaves_ordering_unconstrained() -> None:
    """With the flag OFF, position 0 is committed as freely as any position.

    Using the same gradual schedule, position 0 has the lowest index and ties on
    entropy with the rest, so the legacy sampler commits it in the very first
    step. This demonstrates the constraint is not applied and pre-training
    ordering is unchanged.
    """
    config = _config(canvas_budget=5, outcome_last=False)
    model = OutcomeStubModel(canvas_budget=5)

    output = sample_canvas(model, _batch(config), config)

    assert output.committed_mask.all()
    # Position 0 commits in the first step (not held to last).
    assert output.trace[0].committed_this_step[0, 0].item()
    first_steps = _first_commit_steps(output.trace, row=0)
    assert first_steps[0] == 1


def test_outcome_last_force_commits_low_confidence_outcome() -> None:
    """The force path commits position 0 even when confidence would block it.

    Position 0 is given a low logit (low confidence) and the confidence
    threshold is set high enough that `_select_commits` would never choose it.
    The outcome-last force path must still commit it once [1:] are done.
    """
    config = _config(canvas_budget=4, outcome_last=True, confidence_threshold=0.5)
    # Positions [1:] are high-confidence (~0.96); position 0 is low (~0.02).
    model = OutcomeStubModel(canvas_budget=4, outcome_logit=1.0)

    output = sample_canvas(model, _batch(config), config)

    # Despite low confidence and a 0.5 threshold, position 0 still commits...
    assert output.committed_mask[0, 0].item()
    assert output.committed_mask.all()
    # ...and it is still the last position to commit.
    first_steps = _first_commit_steps(output.trace, row=0)
    assert first_steps[0] == max(first_steps)


def test_outcome_last_true_handles_batch_rows() -> None:
    """The constraint is applied per-row for a multi-row batch."""
    config = _config(canvas_budget=4, outcome_last=True)
    model = OutcomeStubModel(canvas_budget=4)

    output = sample_canvas(model, _batch(config, count=2), config)

    assert output.committed_mask.all()
    for row in range(output.committed_mask.shape[0]):
        for step in output.trace:
            if step.committed_mask[row, 0].item():
                assert step.committed_mask[row, 1:].all()
        first_steps = _first_commit_steps(output.trace, row=row)
        assert first_steps[0] == max(first_steps)
