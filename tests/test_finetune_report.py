"""Tests for the fine-tuning (debut-mode) evaluation report (Worker 4).

These tests use tiny synthetic inputs and a stub model so they never depend on a
trained checkpoint or large fixture:

  * ``FixedCanvasModel`` forces the sampler to reproduce a known debut canvas,
    letting us assert the whole report on a fully-determined generation.
  * The relaxed grammar validator, the timing-MAE helper, and the denoise-last
    structural helper are unit-tested directly on hand-built inputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from thesis_ml.config import load_config
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
)
from thesis_ml.eval.buildorder import BuildOrderEvent
from thesis_ml.eval.finetune_report import (
    _absolute_timing_diffs,
    _denoise_last_ok,
    _example_fog_rate,
    assemble_finetune_report,
    build_debut_report,
    write_finetune_report,
)
from thesis_ml.data.dataset import DatasetExample
from thesis_ml.inference.decode import validate_debut_canvas
from thesis_ml.train.train import _synthetic_input_records
from thesis_ml.vocab.content_vocab import build_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, LOSS_ID, MASK_ID, PAD_ID, WIN_ID

from dataclasses import replace


# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------

CANVAS_BUDGET = 16
MARINE_ID = 100  # build_content_vocabulary assigns ids from CONTENT_TOKEN_OFFSET (100)
MEDIVAC_ID = 101


def _vocab():
    """Two-token content vocabulary: marine -> 100, medivac -> 101."""

    return build_content_vocabulary({"1": "marine", "2": "medivac"})


def _debut_canvas() -> list[int]:
    """A valid debut canvas: [WIN] marine | (empty t1) | medivac | [END] pads.

    Timesteps (bucket index = delimiters seen before the token):
        t0: marine        -> visible-debut
        t1: (empty)       -> bare delimiter (back-to-back delimiters)
        t2: medivac       -> future-debut
    """

    tokens = [WIN_ID, MARINE_ID, DELIMITER_ID, DELIMITER_ID, MEDIVAC_ID, DELIMITER_ID, END_ID]
    tokens += [PAD_ID] * (CANVAS_BUDGET - len(tokens))
    return tokens


def _debut_class_labels() -> list[int]:
    labels = [
        CLASS_WINLOSS,          # [WIN]
        CLASS_ENEMY_OBSERVED,   # marine -> visible-debut
        CLASS_DELIMITER,
        CLASS_DELIMITER,        # empty t1
        CLASS_ENEMY_FUTURE,     # medivac -> future-debut
        CLASS_DELIMITER,
        CLASS_END,
    ]
    labels += [CLASS_PAD] * (CANVAS_BUDGET - len(labels))
    return labels


def _debut_metadata() -> list[dict]:
    meta = [
        {"token_kind": "outcome", "timestep_index": None, "token_name": "[WIN/LOSS]"},
        {"token_kind": "entity", "timestep_index": 0, "token_name": "marine"},
        {"token_kind": "delimiter", "timestep_index": 0, "token_name": "[DELIMITER]"},
        {"token_kind": "delimiter", "timestep_index": 1, "token_name": "[DELIMITER]"},
        {"token_kind": "entity", "timestep_index": 2, "token_name": "medivac"},
        {"token_kind": "delimiter", "timestep_index": 2, "token_name": "[DELIMITER]"},
        {"token_kind": "end", "timestep_index": None, "token_name": "[END]"},
    ]
    meta += [
        {"token_kind": "pad", "timestep_index": None, "token_name": "[PAD]"}
        for _ in range(CANVAS_BUDGET - len(meta))
    ]
    return meta


def _example() -> DatasetExample:
    """One synthetic debut-mode example whose target canvas is ``_debut_canvas``."""

    input_records = _synthetic_input_records(0)
    return DatasetExample(
        input_records=input_records,
        input_token_ids=torch.tensor([r.token_id for r in input_records], dtype=torch.long),
        target_canvas=torch.tensor(_debut_canvas(), dtype=torch.long),
        class_labels=torch.tensor(_debut_class_labels(), dtype=torch.long),
        terminated=True,
        truncated=False,
        canvas_metadata=_debut_metadata(),
        fogged_counts={},                       # no fog -> fog rate 0 -> "<30" bucket
        observed_counts={(0, "marine"): 1},
        window_start=0,
        perspective_player="p1",
        window_end=3,
    )


class FixedCanvasModel(nn.Module):
    """Stub model that makes the sampler reproduce ``target_canvas`` exactly."""

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


def _config():
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=CANVAS_BUDGET),
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64),
        sampler=replace(config.sampler, max_steps=4, entropy_bound=100.0),
        eval=replace(config.eval, timing_tolerance_buckets=1),
    )


# ---------------------------------------------------------------------------
# Report structure + values
# ---------------------------------------------------------------------------


def _build_section(label: str = "memorized"):
    config = _config()
    example = _example()
    model = FixedCanvasModel(example.target_canvas, vocab_size=128)
    return build_debut_report(
        [example],
        label=label,
        model=model,
        vocabulary=_vocab(),
        config=config,
    )


def test_section_has_all_required_nested_keys() -> None:
    section = _build_section()

    # Top-level keys the orchestrator / W5 rely on.
    assert section["label"] == "memorized"
    assert set(section) >= {
        "win_loss_accuracy",
        "build_order_f1",
        "debut_mae",
        "debut_mae_matched_count",
        "grammar_validity",
        "win_loss_minute_buckets",
        "win_loss_structural",
    }

    # Fog-class split keys are present (mandatory, never collapsed).
    by_fog_class = section["build_order_f1"]["by_fog_class"]
    assert set(by_fog_class) == {"visible-debut", "fogged-debut", "future-debut"}

    # Fog-rate bucket keys are present.
    by_fog_bucket = section["build_order_f1"]["by_fog_bucket"]
    assert set(by_fog_bucket) == {">70", "30-70", "<30"}

    # Minute-bucket keys are present (from config "1,3,5,7,10").
    assert set(section["win_loss_minute_buckets"]) == {"1", "3", "5", "7", "10"}

    # Structural booleans present and typed.
    structural = section["win_loss_structural"]
    assert isinstance(structural["position0_ok"], bool)
    assert isinstance(structural["denoise_last_ok"], bool)


def test_section_values_on_perfect_generation() -> None:
    section = _build_section()

    # Predicted canvas == target -> outcome correct, grammar valid, F1 perfect.
    assert section["win_loss_accuracy"] == pytest.approx(1.0)
    assert section["grammar_validity"] == pytest.approx(1.0)
    assert section["build_order_f1"]["aggregate"]["f1"] == pytest.approx(1.0)

    # Fog-class recall: the marine debut is visible, the medivac debut future.
    assert section["build_order_f1"]["by_fog_class"]["visible-debut"]["recall"] == pytest.approx(1.0)
    assert section["build_order_f1"]["by_fog_class"]["future-debut"]["recall"] == pytest.approx(1.0)
    # No fogged debuts exist -> zeroed metrics, but the key is still present.
    assert section["build_order_f1"]["by_fog_class"]["fogged-debut"]["f1"] == pytest.approx(0.0)

    # Fog rate is 0 (no fogged counts) -> example lands in the "<30" bucket.
    assert section["build_order_f1"]["by_fog_bucket"]["<30"]["ground_truth_count"] == 2

    # Timing MAE is 0 on an exact reproduction; both debuts matched.
    assert section["debut_mae"] == pytest.approx(0.0)
    assert section["debut_mae_matched_count"] == 2

    # Structural: the single outcome token sits at position 0.
    assert section["win_loss_structural"]["position0_ok"] is True

    # Cumulative outcome accuracy is 1.0 at every minute checkpoint.
    for value in section["win_loss_minute_buckets"].values():
        assert value == pytest.approx(1.0)


def test_assemble_and_write_round_trip(tmp_path) -> None:
    memorized = _build_section("memorized")
    test = _build_section("test")
    report = assemble_finetune_report(memorized=memorized, test=test)

    # Both sections present with identical metric keys.
    assert set(report) == {"memorized", "test"}
    assert set(report["memorized"]) == set(report["test"])

    destination = tmp_path / "finetune_report.json"
    write_finetune_report(report, destination)
    assert destination.exists()

    import json

    reloaded = json.loads(destination.read_text(encoding="utf-8"))
    assert reloaded["memorized"]["win_loss_accuracy"] == pytest.approx(1.0)
    assert reloaded["test"]["win_loss_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Relaxed grammar validator
# ---------------------------------------------------------------------------


def test_relaxed_validator_accepts_empty_timesteps_and_leading_outcome() -> None:
    # [WIN] marine | (empty) | medivac | [END] pads -> valid (back-to-back
    # delimiters for the empty timestep are legal).
    assert validate_debut_canvas(_debut_canvas()).valid is True

    # A canvas that is nothing but empty timesteps then [END] is also valid.
    only_empty = [LOSS_ID, DELIMITER_ID, DELIMITER_ID, END_ID, PAD_ID, PAD_ID]
    assert validate_debut_canvas(only_empty).valid is True


def test_relaxed_validator_rejects_malformed_canvases() -> None:
    # Missing the leading outcome token.
    assert validate_debut_canvas([MARINE_ID, DELIMITER_ID, END_ID, PAD_ID]).valid is False

    # Outcome token appearing somewhere other than position 0.
    assert validate_debut_canvas([WIN_ID, MARINE_ID, WIN_ID, DELIMITER_ID, END_ID, PAD_ID]).valid is False

    # [END] not sitting on a completed timestep (no preceding delimiter).
    assert validate_debut_canvas([WIN_ID, MARINE_ID, END_ID, PAD_ID]).valid is False

    # Residual [MASK] left in the canvas.
    assert validate_debut_canvas([WIN_ID, MASK_ID, DELIMITER_ID, END_ID, PAD_ID]).valid is False

    # [PAD] appearing before [END].
    assert validate_debut_canvas([WIN_ID, MARINE_ID, DELIMITER_ID, PAD_ID, END_ID]).valid is False

    # Truncated canvas that does not end on a delimiter boundary.
    assert validate_debut_canvas([WIN_ID, MARINE_ID, PAD_ID, PAD_ID]).valid is False


def test_pretraining_validator_untouched() -> None:
    # The existing pre-training grammar must still reject outcome tokens, proving
    # we added a separate validator rather than loosening the old one.
    from thesis_ml.inference.decode import validate_canvas

    assert validate_canvas([WIN_ID, MARINE_ID, DELIMITER_ID, END_ID, PAD_ID]).valid is False


# ---------------------------------------------------------------------------
# Timing MAE helper
# ---------------------------------------------------------------------------


def test_timing_mae_separates_wrong_time_from_wrong_unit() -> None:
    predicted = [
        BuildOrderEvent("marine", 0),    # matches marine@2 -> |0-2| = 2 (right unit, wrong time)
        BuildOrderEvent("medivac", 5),   # matches medivac@5 -> 0
        BuildOrderEvent("zealot", 1),    # wrong unit -> no ground-truth zealot -> unmatched
    ]
    ground_truth = [
        BuildOrderEvent("marine", 2),
        BuildOrderEvent("medivac", 5),
    ]

    diffs = _absolute_timing_diffs(predicted, ground_truth)

    # Two matches (marine, medivac); the wrong-unit zealot contributes nothing.
    assert sorted(diffs) == [0, 2]
    assert sum(diffs) / len(diffs) == pytest.approx(1.0)


def test_fog_rate_definition() -> None:
    example = replace(
        _example(),
        fogged_counts={(0, "marine"): 3},
        observed_counts={(0, "marine"): 1},
    )
    # 3 fogged / (3 fogged + 1 observed) = 0.75.
    assert _example_fog_rate(example) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Denoise-last structural helper
# ---------------------------------------------------------------------------


def _fake_trace(commit_steps: list[list[bool]]):
    """Build a fake sampler trace from a list of per-step commit masks."""

    return [
        SimpleNamespace(step=index + 1, committed_this_step=torch.tensor([mask], dtype=torch.bool))
        for index, mask in enumerate(commit_steps)
    ]


def test_denoise_last_ok_true_when_outcome_committed_last() -> None:
    # Step 1 commits positions 1 and 2; step 2 commits position 0 (outcome last).
    trace = _fake_trace([[False, True, True], [True, False, False]])
    assert _denoise_last_ok(trace) is True


def test_denoise_last_ok_false_when_outcome_committed_early() -> None:
    # Step 1 commits position 0 (outcome first); step 2 commits the rest.
    trace = _fake_trace([[True, False, False], [False, True, True]])
    assert _denoise_last_ok(trace) is False
