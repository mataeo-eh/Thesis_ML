from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.dataset import CLASS_DELIMITER, CLASS_ENEMY_FUTURE, CLASS_PAD
from thesis_ml.eval.buildorder import BuildOrderEvent, extract_build_order, extract_build_order_from_frame
from thesis_ml.eval.harness import evaluate_examples
from thesis_ml.eval.metrics import compare_build_orders
from thesis_ml.train.train import make_synthetic_examples
from thesis_ml.vocab.content_vocab import build_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, PAD_ID


def test_harness_computes_accuracy_f1_on_heldout_examples() -> None:
    config = _small_config(canvas_budget=12)
    examples = make_synthetic_examples(config, count=2)
    model = FixedCanvasModel(examples[0].target_canvas, vocab_size=128)

    report = evaluate_examples(
        model=model,
        examples=examples,
        vocabulary=_synthetic_vocab(),
        config=config,
    )
    metrics = report.to_metrics_dict()

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert "loss" not in metrics
    assert "cross_entropy" not in metrics


def test_oracle_determinism() -> None:
    timesteps = [{"synthetic_1": 1, "synthetic_0": 2}, {"synthetic_2": 1}]

    first = extract_build_order(timesteps)
    second = extract_build_order(timesteps)

    assert first == second


def test_parquet_oracle_extracts_instances_and_upgrades() -> None:
    config = _small_config(canvas_budget=12)
    vocab = build_content_vocabulary(
        {
            "1": "marine",
            "2": "barracks",
            "100001": "stimpack",
        }
    )
    frame = pd.DataFrame(
        [
            {
                "game_loop": 0,
                "timestamp_seconds": 0.0,
                "p2_bot_marine_001_health": 45.0,
                "p2_bot_marine_002_health": None,
                "p2_bot_barracks_001_health": None,
                "p2_upgrades": "[]",
            },
            {
                "game_loop": 5,
                "timestamp_seconds": 5.0,
                "p2_bot_marine_001_health": 45.0,
                "p2_bot_marine_002_health": None,
                "p2_bot_barracks_001_health": 1000.0,
                "p2_upgrades": "['stimpack']",
            },
            {
                "game_loop": 10,
                "timestamp_seconds": 10.0,
                "p2_bot_marine_001_health": 45.0,
                "p2_bot_marine_002_health": 45.0,
                "p2_bot_barracks_001_health": 1000.0,
                "p2_upgrades": "['stimpack']",
            },
        ]
    )

    events = extract_build_order_from_frame(frame, config, vocab, perspective_player="p1")

    assert events == (
        BuildOrderEvent("marine", 0),
        BuildOrderEvent("barracks", 1),
        BuildOrderEvent("stimpack", 1),
        BuildOrderEvent("marine", 2),
    )


def test_resolution_matched_extraction_known_answer() -> None:
    predicted_timesteps = [{"marine": 2}, {"marine": 3, "barracks": 1}, {"factory": 1}]
    ground_truth_timesteps = [{"marine": 2}, {"marine": 3, "barracks": 1}, {"factory": 1}]
    expected = (
        BuildOrderEvent("marine", 0),
        BuildOrderEvent("marine", 0),
        BuildOrderEvent("barracks", 1),
        BuildOrderEvent("marine", 1),
        BuildOrderEvent("factory", 2),
    )

    assert extract_build_order(predicted_timesteps) == expected
    assert extract_build_order(ground_truth_timesteps) == expected


def test_metric_correctness_with_tolerance_and_miss() -> None:
    predicted = (
        BuildOrderEvent("marine", 0),
        BuildOrderEvent("barracks", 3),
        BuildOrderEvent("factory", 8),
        BuildOrderEvent("starport", 9),
    )
    ground_truth = (
        BuildOrderEvent("marine", 1),
        BuildOrderEvent("barracks", 5),
        BuildOrderEvent("factory", 9),
    )

    metrics = compare_build_orders(predicted, ground_truth, timing_tolerance_buckets=1)

    assert metrics.true_positives == 2
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(4 / 7)


def test_boundary_truncated_timesteps_are_not_dropped() -> None:
    config = _small_config(canvas_budget=4)
    base = make_synthetic_examples(config, count=1)[0]
    truncated = replace(
        base,
        target_canvas=torch.tensor([100, DELIMITER_ID, PAD_ID, PAD_ID], dtype=torch.long),
        class_labels=torch.tensor([CLASS_ENEMY_FUTURE, CLASS_DELIMITER, CLASS_PAD, CLASS_PAD], dtype=torch.long),
        terminated=False,
        truncated=True,
        canvas_metadata=[],
    )
    model = FixedCanvasModel(torch.tensor([100, DELIMITER_ID, PAD_ID, PAD_ID]), vocab_size=128)

    report = evaluate_examples(
        model=model,
        examples=[truncated],
        vocabulary=_synthetic_vocab(),
        config=config,
    )

    assert report.examples[0].dropped_final_timestep is False
    assert report.examples[0].ground_truth_events == (BuildOrderEvent("synthetic_0", 0),)
    assert report.examples[0].predicted_events == (BuildOrderEvent("synthetic_0", 0),)
    assert report.metrics.accuracy == pytest.approx(1.0)


def test_ce_is_not_reported() -> None:
    config = _small_config(canvas_budget=12)
    example = make_synthetic_examples(config, count=1)[0]
    report = evaluate_examples(
        model=FixedCanvasModel(example.target_canvas, vocab_size=128),
        examples=[example],
        vocabulary=_synthetic_vocab(),
        config=config,
    )

    keys = set(report.to_metrics_dict())
    assert {"accuracy", "precision", "recall", "f1"} <= keys
    assert "ce" not in keys
    assert "cross_entropy" not in keys
    assert "loss" not in keys


class FixedCanvasModel(nn.Module):
    def __init__(self, target_canvas: torch.Tensor, *, vocab_size: int, top_logit: float = 8.0) -> None:
        super().__init__()
        self.register_buffer("target_canvas", target_canvas.clone())
        self.vocab_size = vocab_size
        self.top_logit = top_logit

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask=None,
        canvas_attention_mask=None,
        input_records=None,
        input_features=None,
        canvas_self_conditioning=None,
    ):
        batch, canvas_len = canvas_token_ids.shape
        input_len = input_token_ids.shape[1]
        logits = torch.zeros(batch, input_len + canvas_len, self.vocab_size, device=canvas_token_ids.device)
        for position, token_id in enumerate(self.target_canvas.tolist()):
            logits[:, input_len + position, token_id] = self.top_logit
        return SimpleNamespace(logits=logits)


def _small_config(*, canvas_budget: int) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=canvas_budget),
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64),
        sampler=replace(config.sampler, max_steps=4, entropy_bound=100.0),
        eval=replace(config.eval, heldout_split="test", timing_tolerance_buckets=1, fog_rate=0.0),
    )


def _synthetic_vocab():
    return build_content_vocabulary(
        {
            "1": "synthetic_0",
            "2": "synthetic_1",
            "3": "synthetic_2",
            "4": "synthetic_3",
            "5": "synthetic_4",
            "6": "synthetic_5",
        }
    )
