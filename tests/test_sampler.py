from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.inference.decode import decode_canvas, validate_canvas
from thesis_ml.inference.sampler import (
    _entropy_bounded_acceptance,
    load_sampling_checkpoint,
    sample_canvas,
)
from thesis_ml.inference.timing import attach_absolute_times
from thesis_ml.train.train import make_synthetic_examples
from thesis_ml.vocab.content_vocab import build_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, MASK_ID, PAD_ID, WIN_ID


def test_uniform_sampler_can_generate_valid_canvas_and_clamps_input() -> None:
    config = _small_config(canvas_budget=7, max_steps=2, entropy_bound=100.0)
    target = torch.tensor([WIN_ID, 100, DELIMITER_ID, 101, DELIMITER_ID, END_ID, PAD_ID])
    model = FixedCanvasModel(target, config=config, top_logit=50.0)
    output = sample_canvas(model, _batch(config), config)
    decoded = decode_canvas(output.canvas[0].tolist(), _vocab())

    assert decoded.validation.valid
    assert torch.equal(output.input_token_ids, output.initial_input_token_ids)
    assert not (output.canvas == MASK_ID).any()
    assert output.steps == 2
    assert model.calls == output.steps


def test_exact_entropy_bounded_prefix_uses_prior_cumulative_entropy() -> None:
    entropy = torch.tensor([[0.08, 0.03, 0.06, 0.02]])
    eligible = torch.ones_like(entropy, dtype=torch.bool)
    accepted = _entropy_bounded_acceptance(entropy, eligible, entropy_bound=0.05)
    # Sorted: .02, .03, .06, .08. Prior cumulative values: 0, .02, .05, .11.
    assert torch.equal(accepted, torch.tensor([[False, True, True, True]]))


def test_uniform_acceptance_is_nonmonotonic_and_unaccepted_positions_renoise() -> None:
    config = _small_config(canvas_budget=3, max_steps=2, entropy_bound=0.0)
    model = ChangingEntropyModel(config=config, canvas_len=3)
    output = sample_canvas(model, _batch(config), config)

    assert output.trace[0].accepted_mask[0, 0]
    assert not output.trace[1].accepted_mask[0, 0]
    assert output.trace[1].unaccepted_mask[0, 0]
    assert output.trace[1].renoised_mask[0, 0]


def test_seeded_categorical_candidates_and_uniform_renoising_are_reproducible() -> None:
    config = _small_config(canvas_budget=5, max_steps=3, entropy_bound=0.01)
    target = torch.tensor([WIN_ID, 100, DELIMITER_ID, END_ID, PAD_ID])
    first = sample_canvas(FixedCanvasModel(target, config=config, top_logit=1.0), _batch(config), config)
    second = sample_canvas(FixedCanvasModel(target, config=config, top_logit=1.0), _batch(config), config)
    assert torch.equal(first.canvas, second.canvas)
    for left, right in zip(first.trace, second.trace, strict=True):
        assert torch.equal(left.accepted_mask, right.accepted_mask)
        assert torch.equal(left.canvas, right.canvas)


def test_self_conditioning_reuses_expected_embeddings_without_extra_calls() -> None:
    config = _small_config(canvas_budget=3, max_steps=3, entropy_bound=0.0)
    model = FixedCanvasModel(torch.tensor([WIN_ID, 100, END_ID]), config=config, top_logit=1.0)
    output = sample_canvas(model, _batch(config), config)

    assert model.calls == output.steps
    assert model.self_conditioning_inputs[0] is None
    for signal in model.self_conditioning_inputs[1:]:
        assert signal is not None
        assert signal.shape == (1, 3, config.model.d_model)


def test_optional_final_logits_are_the_only_extra_model_call() -> None:
    config = _small_config(canvas_budget=3, max_steps=2, entropy_bound=100.0)
    model = FixedCanvasModel(torch.tensor([WIN_ID, 100, END_ID]), config=config)
    output = sample_canvas(model, _batch(config), config, return_final_logits=True)
    assert output.final_canvas_logits is not None
    assert output.final_canvas_logits.shape == (1, 3, model.vocab_size)
    assert model.calls == output.steps + 1


def test_adaptive_stop_requires_entropy_and_argmax_stability() -> None:
    base = _small_config(canvas_budget=3, max_steps=3, entropy_bound=100.0)
    target = torch.tensor([WIN_ID, 100, END_ID])

    both = replace(base, sampler=replace(base.sampler, adaptive_stop=True, entropy_threshold=0.01))
    both_output = sample_canvas(FixedCanvasModel(target, config=both, top_logit=50.0), _batch(both), both)
    assert both_output.steps == 2
    assert both_output.stop_reasons == ("adaptive_entropy_stability",)

    stability_only = replace(
        base, sampler=replace(base.sampler, adaptive_stop=True, entropy_threshold=0.0)
    )
    stability_output = sample_canvas(
        FixedCanvasModel(target, config=stability_only, top_logit=50.0),
        _batch(stability_only),
        stability_only,
    )
    assert stability_output.steps == 3
    assert stability_output.stop_reasons == ("max_steps",)
    assert stability_output.trace[-1].stop_reasons == ("max_steps",)

    entropy_only_model = AlternatingArgmaxModel(config=both, canvas_len=3)
    entropy_output = sample_canvas(entropy_only_model, _batch(both), both)
    assert entropy_output.steps == 3
    assert entropy_output.stop_reasons == ("max_steps",)
    assert entropy_output.trace[-1].stop_reasons == ("max_steps",)


def test_done_batch_rows_freeze_while_other_rows_continue() -> None:
    config = _small_config(canvas_budget=3, max_steps=3, entropy_bound=100.0)
    config = replace(
        config, sampler=replace(config.sampler, adaptive_stop=True, entropy_threshold=0.01)
    )
    model = MixedBatchModel(config=config, canvas_len=3)
    output = sample_canvas(model, _batch(config, count=2), config)
    assert output.trace[1].done_rows[0]
    assert torch.equal(output.trace[1].canvas[0], output.trace[2].canvas[0])
    assert output.stop_reasons[0] == "adaptive_entropy_stability"
    assert output.stop_reasons[1] == "max_steps"
    assert output.trace[-1].stop_reasons == (
        "adaptive_entropy_stability",
        "max_steps",
    )


def test_absorbing_sampler_is_monotonic_and_never_renoises() -> None:
    base = _small_config(canvas_budget=4, max_steps=4, entropy_bound=0.0)
    config = replace(base, diffusion=replace(base.diffusion, process="absorbing"))
    target = torch.tensor([WIN_ID, 100, END_ID, PAD_ID])
    output = sample_canvas(FixedCanvasModel(target, config=config, top_logit=2.0), _batch(config), config)
    previous_unmasked = torch.zeros(1, 4, dtype=torch.bool)
    for step in output.trace:
        unmasked = step.canvas != MASK_ID
        assert (unmasked | ~previous_unmasked).all()
        assert not step.renoised_mask.any()
        previous_unmasked = unmasked
    assert (output.canvas != MASK_ID).all()
    assert output.stop_reasons == ("absorbing_complete",)


def test_infill_revealed_positions_are_clamped_and_excluded() -> None:
    config = _small_config(canvas_budget=7, max_steps=2, entropy_bound=100.0)
    target = torch.tensor([WIN_ID, 100, DELIMITER_ID, 101, DELIMITER_ID, END_ID, PAD_ID])
    batch = _batch(config)
    output = sample_canvas(
        FixedCanvasModel(target, config=config, top_logit=50.0),
        batch,
        config,
        noise_rate=0.5,
    )
    revealed = output.revealed_mask
    assert revealed is not None and revealed.any()
    assert torch.equal(output.canvas[revealed], batch.target_canvas[revealed])
    for step in output.trace:
        assert not step.accepted_mask[revealed].any()
        assert not step.renoised_mask[revealed].any()


def test_mask_is_excluded_but_outcome_tokens_are_not_position_restricted() -> None:
    config = _small_config(canvas_budget=3, max_steps=1, entropy_bound=100.0)
    target = torch.tensor([100, WIN_ID, WIN_ID])
    output = sample_canvas(FixedCanvasModel(target, config=config, top_logit=50.0), _batch(config), config)
    assert not (output.canvas == MASK_ID).any()
    assert output.canvas[0, 1] == WIN_ID
    assert output.canvas[0, 2] == WIN_ID


def test_sampling_checkpoint_validates_architecture_and_process_before_loading(tmp_path) -> None:
    config = _small_config(canvas_budget=3, max_steps=1)
    model = FixedCanvasModel(torch.tensor([WIN_ID, END_ID, PAD_ID]), config=config)
    model.feature_statistics_identity = "test-statistics"
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": model.state_dict(),
            "feature_statistics_identity": "test-statistics",
            "architecture_identity": model.architecture_identity,
            "diffusion_process": model.diffusion_process,
        },
        checkpoint,
    )
    load_sampling_checkpoint(model, checkpoint)

    retired = tmp_path / "retired.pt"
    torch.save({"model": model.state_dict(), "ema_model": model.state_dict()}, retired)
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        load_sampling_checkpoint(model, retired)

    cross_process = tmp_path / "cross_process.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["diffusion_process"] = "absorbing"
    torch.save(payload, cross_process)
    with pytest.raises(ValueError, match="diffusion process mismatch"):
        load_sampling_checkpoint(model, cross_process)


def test_decoder_and_time_recovery_contracts() -> None:
    vocab = _vocab()
    valid = [WIN_ID, 100, 100, DELIMITER_ID, 101, DELIMITER_ID, END_ID, PAD_ID]
    decoded = decode_canvas(valid, vocab)
    assert decoded.validation.valid
    assert decoded.timesteps == [{"marine": 2}, {"scv": 1}]
    assert not validate_canvas([100, DELIMITER_ID, END_ID]).valid
    assert not validate_canvas([WIN_ID, 100, PAD_ID, END_ID]).valid
    timed = attach_absolute_times(
        [{"marine": 2}, {"scv": 1}], last_input_clock=125.0, sampling_interval_s=5
    )
    assert [item.timestamp_seconds for item in timed] == [125.0, 130.0]


class FixedCanvasModel(nn.Module):
    def __init__(
        self,
        target_canvas: torch.Tensor,
        *,
        config: ProjectConfig,
        vocab_size: int = 128,
        top_logit: float = 8.0,
    ) -> None:
        super().__init__()
        self.register_buffer("target_canvas", target_canvas.clone())
        self.vocab_size = vocab_size
        self.top_logit = top_logit
        self.embedding = nn.Module()
        self.embedding.token_embedding = nn.Embedding(vocab_size, config.model.d_model)
        self.architecture_identity = "uniform-gemma4-dense-v1"
        self.diffusion_process = config.diffusion.process
        self.self_conditioning_inputs: list[torch.Tensor | None] = []
        self.calls = 0

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        canvas_self_conditioning=None,
        **kwargs,
    ):
        self.calls += 1
        self.self_conditioning_inputs.append(
            canvas_self_conditioning.detach().clone()
            if isinstance(canvas_self_conditioning, torch.Tensor)
            else None
        )
        batch, canvas_len = canvas_token_ids.shape
        input_len = input_token_ids.shape[1]
        logits = torch.zeros(batch, input_len + canvas_len, self.vocab_size, device=canvas_token_ids.device)
        for position, token_id in enumerate(self.target_canvas.tolist()):
            logits[:, input_len + position, token_id] = self.top_logit
        return SimpleNamespace(logits=logits)


class ChangingEntropyModel(FixedCanvasModel):
    def __init__(self, *, config: ProjectConfig, canvas_len: int) -> None:
        super().__init__(torch.full((canvas_len,), 10), config=config, top_logit=1.0)

    def forward(self, *, input_token_ids, canvas_token_ids, canvas_self_conditioning=None, **kwargs):
        output = super().forward(
            input_token_ids=input_token_ids,
            canvas_token_ids=canvas_token_ids,
            canvas_self_conditioning=canvas_self_conditioning,
            **kwargs,
        )
        input_len = input_token_ids.shape[1]
        logits = output.logits
        preferred = (self.calls - 1) % canvas_token_ids.shape[1]
        logits[:, input_len:, :] = 0.0
        logits[:, input_len + preferred, 10] = 8.0
        return SimpleNamespace(logits=logits)


class AlternatingArgmaxModel(FixedCanvasModel):
    def __init__(self, *, config: ProjectConfig, canvas_len: int) -> None:
        super().__init__(torch.full((canvas_len,), 10), config=config, top_logit=50.0)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        input_len = kwargs["input_token_ids"].shape[1]
        token = 10 if self.calls % 2 else 11
        output.logits[:, input_len:, :] = -50.0
        output.logits[:, input_len:, token] = 50.0
        return output


class MixedBatchModel(AlternatingArgmaxModel):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        input_len = kwargs["input_token_ids"].shape[1]
        output.logits[0, input_len:, :] = -50.0
        output.logits[0, input_len:, 10] = 50.0
        return output


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
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64),
        sampler=replace(
            config.sampler,
            max_steps=max_steps,
            entropy_bound=entropy_bound,
            adaptive_stop=False,
        ),
    )


def _batch(config: ProjectConfig, *, count: int = 1):
    examples = make_synthetic_examples(config, count=count)
    batch = collate_diffusion_examples(examples, debut_mode=False)
    width = config.data.canvas_budget_tokens
    return replace(
        batch,
        target_canvas=batch.target_canvas[:, :width],
        canvas_attention_mask=batch.canvas_attention_mask[:, :width],
        class_labels=batch.class_labels[:, :width],
        canvas_loss_mask=batch.canvas_loss_mask[:, :width],
        canvas_prediction_distances=batch.canvas_prediction_distances[:, :width],
        canvas_metadata=[row[:width] for row in batch.canvas_metadata],
    )


def _vocab():
    return build_content_vocabulary({"1": "marine", "2": "scv"})
