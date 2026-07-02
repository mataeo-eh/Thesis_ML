from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.inference.decode import decode_canvas, validate_canvas
from thesis_ml.inference.sampler import load_sampling_checkpoint, sample_canvas
from thesis_ml.inference.timing import attach_absolute_times
from thesis_ml.train.train import make_synthetic_examples
from thesis_ml.vocab.content_vocab import build_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, MASK_ID, PAD_ID


def test_sampler_generated_canvas_validates_and_input_is_clamped() -> None:
    config = _small_config(canvas_budget=7, max_steps=10)
    target = torch.tensor([100, DELIMITER_ID, 101, DELIMITER_ID, END_ID, PAD_ID, PAD_ID])
    model = FixedCanvasModel(target, vocab_size=128)
    batch = _batch(config)

    output = sample_canvas(model, batch, config)
    decoded = decode_canvas(output.canvas[0].tolist(), _vocab())

    assert decoded.validation.valid
    assert decoded.validation.truncated is False
    assert torch.equal(output.input_token_ids, output.initial_input_token_ids)
    assert not (output.input_token_ids == MASK_ID).any()


def test_sampler_commits_monotonically_without_remasking() -> None:
    config = _small_config(canvas_budget=5, max_steps=8, entropy_bound=0.01)
    target = torch.tensor([100, DELIMITER_ID, 101, END_ID, PAD_ID])
    model = FixedCanvasModel(target, vocab_size=128, top_logit=1.5)
    output = sample_canvas(model, _batch(config), config)

    previous = torch.zeros_like(output.trace[0].committed_mask)
    for step in output.trace:
        assert not (previous & (step.canvas == MASK_ID)).any()
        assert torch.equal(step.committed_mask, previous | step.committed_this_step)
        assert torch.all(step.committed_mask | previous.logical_not())
        previous = step.committed_mask
    assert output.committed_mask.all()


def test_sampler_reuses_self_conditioning_without_extra_model_calls() -> None:
    config = _small_config(canvas_budget=3, max_steps=3, entropy_bound=0.0)
    target = torch.tensor([100, 101, END_ID])
    model = FixedCanvasModel(target, vocab_size=128, top_logit=1.0)

    output = sample_canvas(model, _batch(config), config)

    assert len(model.self_conditioning_inputs) == output.steps
    assert model.self_conditioning_inputs[0] is None
    for self_conditioning in model.self_conditioning_inputs[1:]:
        assert self_conditioning is not None
        assert self_conditioning.shape == (1, config.data.canvas_budget_tokens, model.vocab_size)


def test_sampler_self_conditioning_off_preserves_legacy_call_contract() -> None:
    config = replace(
        _small_config(canvas_budget=3, max_steps=3, entropy_bound=0.0),
        model=replace(_small_config(canvas_budget=3, max_steps=3, entropy_bound=0.0).model, self_conditioning=False),
    )
    target = torch.tensor([100, 101, END_ID])
    model = FixedCanvasModel(target, vocab_size=128, top_logit=1.0)

    output = sample_canvas(model, _batch(config), config)

    assert len(model.self_conditioning_inputs) == output.steps
    assert model.self_conditioning_inputs == ["missing", "missing", "missing"]


def test_sampler_early_stops_and_respects_max_steps() -> None:
    config = _small_config(canvas_budget=5, max_steps=10, entropy_bound=100.0)
    target = torch.tensor([100, DELIMITER_ID, 101, END_ID, PAD_ID])
    output = sample_canvas(FixedCanvasModel(target, vocab_size=128), _batch(config), config)

    assert output.steps == 1
    assert output.steps <= config.sampler.max_steps
    assert output.committed_mask.all()


def test_decoder_roundtrips_known_canvas_and_flags_invalid() -> None:
    vocab = _vocab()
    valid = [100, 100, DELIMITER_ID, 101, DELIMITER_ID, END_ID, PAD_ID, PAD_ID]
    decoded = decode_canvas(valid, vocab)

    assert decoded.validation.valid
    assert decoded.timesteps == [{"marine": 2}, {"scv": 1}]
    assert decoded.truncated is False
    assert decoded.partial_final_timestep is False

    truncated = decode_canvas([100, DELIMITER_ID, PAD_ID, PAD_ID], vocab)
    assert truncated.validation.valid
    assert truncated.truncated is True
    assert truncated.partial_final_timestep is False
    assert truncated.timesteps == [{"marine": 1}]

    partial = decode_canvas([100, DELIMITER_ID, 101], vocab)
    assert partial.validation.valid is False
    assert "boundary" in (partial.validation.diagnosis or "")

    invalid = validate_canvas([100, PAD_ID, END_ID])
    assert invalid.valid is False
    assert "[PAD]" in (invalid.diagnosis or "")


def test_time_recovery_is_arithmetic_only() -> None:
    timed = attach_absolute_times(
        [{"marine": 2}, {"scv": 1}],
        last_input_clock=125.0,
        sampling_interval_s=5,
    )

    assert [item.timestamp_seconds for item in timed] == [125.0, 130.0]
    assert timed[1].counts == {"scv": 1}


def test_sampling_checkpoint_prefers_ema_weights(tmp_path) -> None:
    raw = FixedCanvasModel(torch.tensor([100, END_ID, PAD_ID]), vocab_size=128)
    ema = FixedCanvasModel(torch.tensor([101, END_ID, PAD_ID]), vocab_size=128)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": raw.state_dict(), "ema_model": ema.state_dict()}, checkpoint)

    loaded = FixedCanvasModel(torch.tensor([100, END_ID, PAD_ID]), vocab_size=128)
    load_sampling_checkpoint(loaded, checkpoint)

    assert torch.equal(loaded.target_canvas, ema.target_canvas)


class FixedCanvasModel(nn.Module):
    def __init__(self, target_canvas: torch.Tensor, *, vocab_size: int, top_logit: float = 8.0) -> None:
        super().__init__()
        self.register_buffer("target_canvas", target_canvas.clone())
        self.vocab_size = vocab_size
        self.top_logit = top_logit
        self.self_conditioning_inputs = []

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
        if isinstance(canvas_self_conditioning, torch.Tensor):
            self.self_conditioning_inputs.append(canvas_self_conditioning.detach().clone())
        else:
            self.self_conditioning_inputs.append(canvas_self_conditioning)
        batch, canvas_len = canvas_token_ids.shape
        input_len = input_token_ids.shape[1]
        logits = torch.zeros(batch, input_len + canvas_len, self.vocab_size, device=canvas_token_ids.device)
        for position, token_id in enumerate(self.target_canvas.tolist()):
            logits[:, input_len + position, token_id] = self.top_logit
        return SimpleNamespace(logits=logits)


def _small_config(
    *,
    canvas_budget: int,
    max_steps: int,
    entropy_bound: float = 0.1,
) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=canvas_budget),
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64, self_conditioning=True),
        sampler=replace(
            config.sampler,
            max_steps=max_steps,
            entropy_bound=entropy_bound,
            confidence_threshold=0.0,
            min_commit_per_step=1,
        ),
    )


def _batch(config: ProjectConfig):
    examples = make_synthetic_examples(config, count=1)
    return collate_diffusion_examples(examples)


def _vocab():
    return build_content_vocabulary({"1": "marine", "2": "scv"})
