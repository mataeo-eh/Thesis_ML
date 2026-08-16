from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import torch

from scripts.canvas_unigram_baseline import UnigramAccumulator, summarize_counts
from thesis_ml.config import load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import (
    CLASS_CLAMPED,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    DatasetExample,
)
from thesis_ml.data.features import CATEGORICAL_FEATURE_WIDTH, CONTINUOUS_FEATURE_NAMES
from thesis_ml.model.loss import CanvasCrossEntropyLoss, active_class_id_to_name
from thesis_ml.vocab.special_tokens import BOS_ID, CONTENT_TOKEN_OFFSET, PAD_ID


ROOT = Path(__file__).resolve().parents[1]


def test_weighted_optimal_constant_has_known_closed_form() -> None:
    accumulator = UnigramAccumulator(
        vocab_size=3,
        canvas_length=4,
        class_ids=(0, 1),
        include_position_conditional=True,
    )
    batch = _batch(
        targets=[[0, 0, 1, 1]],
        labels=[[0, 0, 1, 1]],
        mask=[[True, True, True, True]],
    )
    accumulator.update(batch)

    report = summarize_counts(
        accumulator,
        class_id_to_name={0: "heavy", 1: "light"},
        class_weights=(1.0, 0.5),
    )
    overall = report["overall"]
    expected_weighted_entropy = -(2.0 / 3.0) * math.log(2.0 / 3.0) - (
        1.0 / 3.0
    ) * math.log(1.0 / 3.0)

    assert overall["scored_positions"] == 4
    assert math.isclose(
        overall["unweighted_marginal_entropy_nats"], math.log(2.0), rel_tol=1e-12
    )
    assert math.isclose(
        overall["weighted_optimal_constant_ce_nats"],
        expected_weighted_entropy,
        rel_tol=1e-12,
    )
    assert math.isclose(
        overall["position_conditional_unweighted_entropy_nats"], 0.0, abs_tol=1e-12
    )
    assert [entry["scored_positions"] for entry in report["classes"]] == [2, 2]
    assert all(
        math.isclose(entry["unweighted_conditional_entropy_nats"], 0.0, abs_tol=1e-12)
        for entry in report["classes"]
    )


def test_accumulator_selection_and_weighting_match_canvas_loss() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    weights = replace(config.loss.class_loss_weights, pad=0.1)
    config = replace(config, loss=replace(config.loss, class_loss_weights=weights))
    examples = [
        _example(
            [BOS_ID, CONTENT_TOKEN_OFFSET, PAD_ID],
            [CLASS_CLAMPED, CLASS_ENEMY_OBSERVED, CLASS_PAD],
        ),
        _example(
            [BOS_ID, CONTENT_TOKEN_OFFSET],
            [CLASS_CLAMPED, CLASS_ENEMY_OBSERVED],
        ),
    ]
    batch = collate_diffusion_examples(examples, debut_mode=False)
    criterion = CanvasCrossEntropyLoss(config)
    class_names = active_class_id_to_name(config)
    accumulator = UnigramAccumulator(
        vocab_size=CONTENT_TOKEN_OFFSET + 1,
        canvas_length=config.data.canvas_budget_tokens,
        class_ids=tuple(class_names),
        include_position_conditional=True,
    )
    accumulator.update(batch)
    report = summarize_counts(
        accumulator,
        class_id_to_name=class_names,
        class_weights=criterion.class_weights.tolist(),
    )

    # Two content targets plus one SEMANTIC [PAD] are scored. The two BOS
    # anchors and the shorter row's batch-shape PAD are excluded.
    assert report["overall"]["scored_positions"] == 3
    per_class_counts = {
        entry["class_id"]: entry["scored_positions"] for entry in report["classes"]
    }
    assert per_class_counts[CLASS_ENEMY_OBSERVED] == 2
    assert per_class_counts[CLASS_PAD] == 1

    total_weight = 2.0 + 0.1
    probabilities = torch.zeros(CONTENT_TOKEN_OFFSET + 1, dtype=torch.float32)
    probabilities[CONTENT_TOKEN_OFFSET] = 2.0 / total_weight
    probabilities[PAD_ID] = 0.1 / total_weight
    logits = probabilities.log().view(1, 1, -1).expand(
        batch.target_canvas.shape[0], batch.target_canvas.shape[1], -1
    )
    loss = criterion(
        logits,
        batch.target_canvas,
        batch.class_labels,
        scored_mask=batch.canvas_loss_mask,
    ).loss

    assert torch.isclose(
        loss,
        torch.tensor(report["overall"]["weighted_optimal_constant_ce_nats"]),
        atol=1e-7,
        rtol=1e-7,
    )


def _batch(
    *, targets: list[list[int]], labels: list[list[int]], mask: list[list[bool]]
):
    shape = (len(targets), len(targets[0]))
    empty_bool = torch.zeros(shape, dtype=torch.bool)
    empty_long = torch.zeros(shape, dtype=torch.long)
    from thesis_ml.data.collate import DiffusionBatch
    from thesis_ml.model.embedding import InputFeatures

    return DiffusionBatch(
        input_token_ids=torch.empty((shape[0], 0), dtype=torch.long),
        input_attention_mask=torch.empty((shape[0], 0), dtype=torch.bool),
        input_lengths=torch.zeros(shape[0], dtype=torch.long),
        target_canvas=torch.tensor(targets, dtype=torch.long),
        canvas_attention_mask=torch.ones(shape, dtype=torch.bool),
        class_labels=torch.tensor(labels, dtype=torch.long),
        canvas_loss_mask=torch.tensor(mask, dtype=torch.bool),
        terminated=torch.zeros(shape[0], dtype=torch.bool),
        truncated=torch.zeros(shape[0], dtype=torch.bool),
        perspective_ids=torch.ones(shape[0], dtype=torch.long),
        input_timestep_counts=torch.zeros(shape[0], dtype=torch.long),
        enemy_future_timestep_counts=torch.zeros(shape[0], dtype=torch.long),
        canvas_prediction_distances=empty_long - 1,
        input_records=[],
        canvas_metadata=[],
        input_features=InputFeatures(
            continuous_values=torch.empty(
                (shape[0], 0, len(CONTINUOUS_FEATURE_NAMES))
            ),
            continuous_validity=torch.empty(
                (shape[0], 0, len(CONTINUOUS_FEATURE_NAMES)), dtype=torch.bool
            ),
            categorical_values=torch.empty(
                (shape[0], 0, CATEGORICAL_FEATURE_WIDTH)
            ),
            allegiance_values=torch.empty((shape[0], 0, 1)),
            feature_mask=empty_bool[:, :0],
        ),
    )


def _example(targets: list[int], labels: list[int]) -> DatasetExample:
    return DatasetExample(
        input_records=[],
        input_token_ids=torch.empty(0, dtype=torch.long),
        target_canvas=torch.tensor(targets, dtype=torch.long),
        class_labels=torch.tensor(labels, dtype=torch.long),
        terminated=False,
        truncated=True,
        canvas_metadata=[],
        fogged_counts={},
        observed_counts={},
        window_start=0,
        perspective_player="p1",
    )
