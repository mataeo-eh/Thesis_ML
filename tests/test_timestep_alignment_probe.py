"""Unit coverage for the timestep alignment probe (``scripts/timestep_alignment_probe.py``).

Role in the larger system
-------------------------
The probe is a training-objective diagnostic whose whole value is that its
numbers are trustworthy. These tests pin the pieces that could silently produce a
wrong headline:

* canvas segmentation, including terminal ``[END]`` versus a boundary-truncated
  ``[PAD]`` horizon and empty timestep groups;
* the exclusion of batch-shape padding while semantic ``[PAD]`` stays scored;
* every controlled perturbation's closed-form mismatch count and cross entropy;
* the minimum-cost assignment used by the oracle aligned score, including
  duplicate token occurrences;
* pooled aggregation (sums pooled before ratios, never a mean of means);
* determinism under a fixed seed with ``--num-workers 0``;
* the write boundary -- the probe must never target ``Model_Inference_Tests/`` or
  the training run it probes.

Nothing here loads a checkpoint, touches CUDA, or reads replay data, so the whole
file runs in the ordinary CPU suite.
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.data.dataset import (
    CLASS_CLAMPED,
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
    PRETRAIN_CLASS_ID_TO_NAME,
)
from thesis_ml.model.embedding import InputFeatures
from thesis_ml.vocab.special_tokens import (
    BOS_ID,
    CONTENT_TOKEN_OFFSET,
    DELIMITER_ID,
    END_ID,
    PAD_ID,
    WIN_ID,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.timestep_alignment_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    PooledCounters,
    build_perturbations,
    compare_timestep,
    content_counts,
    deletion_offset_sweep,
    evaluate_perturbation,
    finalize_counters,
    levenshtein,
    linear_sum_assignment_min,
    multiset_overlap,
    oracle_aligned_content_cost,
    parse_canvas_layout,
    predicted_structure,
    pseudo_logits,
    score_batch,
    select_focus_timestep,
    select_windows,
    _shuffle_input_rows,
    _strip_enemy_input,
)

PROBE_SOURCE = REPOSITORY_ROOT / "scripts" / "timestep_alignment_probe.py"
VOCAB_SIZE = 32
CONFIDENCE = 0.9


# ---------------------------------------------------------------------------
# Fixtures: one small terminated canvas whose every index is written out below
# ---------------------------------------------------------------------------
#
#   index: 0     1    2   3   4   5   6      7   8   9  10  11  12
#   token: BOS  WIN  10  10  11  12  DELIM  10  11  13  13  14  DELIM
#   index: 13  14  15  16  17     18   19  20  21
#   token: 12  15  16  17  DELIM  END  PAD PAD PAD
#
# timestep 0: content 2..5   delimiter 6   (length 4)
# timestep 1: content 7..11  delimiter 12  (length 5)  <- the median-length
#                                                          qualifying group
# timestep 2: content 13..16 delimiter 17  (length 4, final -> never the focus)


def terminated_canvas() -> list[int]:
    """A grammatical canvas that ends with a terminal ``[END]`` then padding."""

    return [
        BOS_ID,
        WIN_ID,
        10,
        10,
        11,
        12,
        DELIMITER_ID,
        10,
        11,
        13,
        13,
        14,
        DELIMITER_ID,
        12,
        15,
        16,
        17,
        DELIMITER_ID,
        END_ID,
        PAD_ID,
        PAD_ID,
        PAD_ID,
    ]


def truncated_canvas() -> list[int]:
    """A boundary-truncated canvas: no ``[END]``, padding straight off a delimiter."""

    tokens = terminated_canvas()
    tokens[18] = PAD_ID
    return tokens


def canvas_class_labels(tokens: list[int]) -> list[int]:
    """Assign the production class taxonomy to a hand-built canvas row."""

    labels = []
    for index, token in enumerate(tokens):
        if index == 0:
            labels.append(CLASS_CLAMPED)
        elif index == 1:
            labels.append(CLASS_WINLOSS)
        elif token == DELIMITER_ID:
            labels.append(CLASS_DELIMITER)
        elif token == END_ID:
            labels.append(CLASS_END)
        elif token == PAD_ID:
            labels.append(CLASS_PAD)
        else:
            labels.append(CLASS_ENEMY_OBSERVED)
    return labels


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_parse_terminated_canvas_segments_every_group():
    layout = parse_canvas_layout(terminated_canvas())
    assert [span.delimiter_index for span in layout.timesteps] == [6, 12, 17]
    assert [(span.content_start, span.content_end) for span in layout.timesteps] == [
        (2, 6),
        (7, 12),
        (13, 17),
    ]
    assert layout.end_index == 18
    assert layout.terminated is True
    assert layout.semantic_pad_indices == (19, 20, 21)
    assert layout.outcome_index == 1


def test_parse_boundary_truncated_canvas_has_no_end_and_pads_from_the_delimiter():
    layout = parse_canvas_layout(truncated_canvas())
    assert layout.end_index is None
    assert layout.terminated is False
    # Padding starts at the position the [END] occupied in the terminated case.
    assert layout.semantic_pad_indices == (18, 19, 20, 21)
    assert len(layout.timesteps) == 3


def test_parse_handles_an_empty_timestep_group():
    tokens = [BOS_ID, WIN_ID, DELIMITER_ID, 10, DELIMITER_ID, END_ID, PAD_ID]
    layout = parse_canvas_layout(tokens)
    assert len(layout.timesteps) == 2
    assert layout.timesteps[0].content_length == 0
    assert layout.timesteps[1].content_length == 1


def test_parse_rejects_a_row_that_is_not_bos_then_outcome():
    with pytest.raises(ValueError):
        parse_canvas_layout([WIN_ID, BOS_ID, 10, DELIMITER_ID])
    with pytest.raises(ValueError):
        parse_canvas_layout([BOS_ID, 10, DELIMITER_ID])


def test_active_length_excludes_batch_shape_padding():
    """Batch-shape padding beyond ``active_length`` is not part of the canvas.

    The collater right-pads short rows with ``[PAD]``. Those positions are NOT
    semantic ``[PAD]`` targets, and treating them as such would inflate every
    ``[PAD]`` statistic in the report.
    """

    tokens = terminated_canvas() + [PAD_ID] * 9
    full = parse_canvas_layout(tokens)
    trimmed = parse_canvas_layout(tokens, active_length=22)
    assert len(full.semantic_pad_indices) == 12
    assert trimmed.semantic_pad_indices == (19, 20, 21)
    assert trimmed.timesteps == full.timesteps


def test_predicted_structure_tolerates_an_ungrammatical_row():
    row = [BOS_ID, WIN_ID, 10, DELIMITER_ID, DELIMITER_ID, 11, PAD_ID, 12, PAD_ID]
    structure = predicted_structure(row, active_length=len(row))
    assert structure["delimiter_count"] == 2
    assert structure["end_count"] == 0
    # The first terminal token is where the model thinks the body stops.
    assert structure["active_length"] == 6


# ---------------------------------------------------------------------------
# Delimiter-local semantics
# ---------------------------------------------------------------------------


def test_content_counts_drops_structural_tokens_and_exclusions():
    counts = content_counts([10, 10, DELIMITER_ID, 11, PAD_ID, END_ID])
    assert counts == {10: 2, 11: 1}
    assert content_counts([10, 10, 11], exclude=frozenset({10})) == {11: 1}


def test_multiset_metrics_are_order_invariant_but_edit_distance_is_not():
    forward = compare_timestep([10, 11, 12], [12, 11, 10])
    assert forward.overlap == 3
    assert forward.exact_multiset is True
    assert forward.count_error == 0
    # Reversing three distinct tokens costs two ordered substitutions.
    assert forward.edit_distance == 2


def test_multiset_metrics_price_one_wrong_token():
    semantics = compare_timestep([10, 11, 13], [10, 11, 12])
    assert semantics.overlap == 2
    assert semantics.predicted_total == 3
    assert semantics.target_total == 3
    assert semantics.exact_multiset is False
    # 12 dropped and 13 invented: two cells wrong by one each.
    assert semantics.count_error == 2
    assert semantics.count_cells == 4
    assert semantics.edit_distance == 1


def test_structural_predictions_inside_a_span_reduce_recall():
    semantics = compare_timestep([10, DELIMITER_ID, PAD_ID], [10, 11, 12])
    assert semantics.predicted_total == 1
    assert semantics.target_total == 3
    assert multiset_overlap({10: 1}, {10: 1, 11: 1, 12: 1}) == 1


def test_levenshtein_matches_known_values():
    assert levenshtein([], [1, 2, 3]) == 3
    assert levenshtein([1, 2, 3], []) == 3
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert levenshtein([1, 2, 3], [1, 9, 3]) == 1


# ---------------------------------------------------------------------------
# Minimum-cost assignment / oracle aligned score
# ---------------------------------------------------------------------------


def test_assignment_matches_brute_force_on_random_small_problems():
    generator = np.random.default_rng(7)
    for _ in range(120):
        rows = int(generator.integers(1, 6))
        columns = int(generator.integers(rows, 7))
        cost = generator.normal(size=(rows, columns))
        assignment = linear_sum_assignment_min(cost)
        obtained = cost[np.arange(rows), assignment].sum()
        best = min(
            sum(cost[row, permutation[row]] for row in range(rows))
            for permutation in itertools.permutations(range(columns), rows)
        )
        assert obtained == pytest.approx(best)
        assert len(set(assignment.tolist())) == rows


def test_assignment_handles_duplicate_columns_as_distinct_occurrences():
    """Two occurrences of one token type are two columns, not one.

    A timestep holding ``probe`` twice must cost two slots, otherwise the oracle
    score would let a single confident slot pay for an arbitrary count.
    """

    cost = np.array([[0.1, 0.1, 5.0], [0.2, 0.2, 0.3], [9.0, 9.0, 9.5]])
    assignment = linear_sum_assignment_min(cost)
    assert sorted(assignment.tolist()) == [0, 1, 2]


def test_assignment_rejects_more_rows_than_columns():
    with pytest.raises(ValueError):
        linear_sum_assignment_min(np.zeros((3, 2)))


def test_oracle_score_never_exceeds_the_positional_score():
    """The identity assignment is always feasible, so the oracle is a lower bound."""

    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(6, VOCAB_SIZE, generator=generator)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    targets = [10, 11, 10, 12, 13, 10]
    positional = float(-sum(log_probabilities[i, token] for i, token in enumerate(targets)))
    oracle = oracle_aligned_content_cost(log_probabilities.numpy(), targets)
    assert oracle <= positional + 1e-9


def test_oracle_score_recovers_a_pure_permutation_for_free():
    """A prediction that is right up to order inside one group costs the floor."""

    targets = [10, 11, 12]
    predicted_order = [12, 10, 11]
    logits = pseudo_logits(predicted_order, vocabulary_size=VOCAB_SIZE, confidence=CONFIDENCE)
    log_probabilities = torch.log_softmax(logits, dim=-1).numpy()
    oracle = oracle_aligned_content_cost(log_probabilities, targets)
    assert oracle == pytest.approx(3 * -math.log(CONFIDENCE), rel=1e-5)


def test_oracle_score_requires_matching_lengths():
    with pytest.raises(ValueError):
        oracle_aligned_content_cost(np.zeros((2, VOCAB_SIZE)), [10, 11, 12])


# ---------------------------------------------------------------------------
# Experiment A: controlled perturbations
# ---------------------------------------------------------------------------


def test_pseudo_logits_produce_the_intended_two_valued_cross_entropy():
    logits = pseudo_logits([10, 11], vocabulary_size=VOCAB_SIZE, confidence=CONFIDENCE)
    probabilities = torch.softmax(logits, dim=-1)
    assert probabilities[0, 10].item() == pytest.approx(CONFIDENCE)
    assert probabilities[0, 11].item() == pytest.approx(
        (1.0 - CONFIDENCE) / (VOCAB_SIZE - 1)
    )
    assert logits.argmax(dim=-1).tolist() == [10, 11]


def test_focus_group_is_the_median_length_non_final_qualifying_group():
    layout = parse_canvas_layout(terminated_canvas())
    focus = select_focus_timestep(layout)
    # Groups 0 (length 4) and 1 (length 5) qualify; group 2 is final. The median
    # of a two-element sorted list is its second element.
    assert focus.ordinal == 1
    assert focus.content_length == 5


def test_focus_selection_rejects_a_canvas_with_no_qualifying_group():
    layout = parse_canvas_layout([BOS_ID, WIN_ID, 10, DELIMITER_ID, END_ID, PAD_ID])
    with pytest.raises(ValueError):
        select_focus_timestep(layout)


@pytest.mark.parametrize(
    "name, expected_mismatches, expected_edits, expected_realignment",
    [
        ("exact_target", 0, 0, 2),
        ("content_substitution", 1, 1, 8),
        ("content_deletion_local_shift", 5, 2, 13),
        ("content_insertion_local_shift", 5, 2, 14),
        ("delimiter_displacement", 2, 2, 13),
        ("content_deletion_global_shift", 11, 1, None),
    ],
)
def test_each_perturbation_has_the_expected_closed_form_shape(
    name, expected_mismatches, expected_edits, expected_realignment
):
    tokens = terminated_canvas()
    layout = parse_canvas_layout(tokens)
    labels = canvas_class_labels(tokens)
    cases = {
        case.name: case
        for case in build_perturbations(tokens, layout, vocabulary_size=VOCAB_SIZE)
    }
    assert set(cases) == {
        "exact_target",
        "content_substitution",
        "content_deletion_local_shift",
        "content_insertion_local_shift",
        "delimiter_displacement",
        "content_deletion_global_shift",
    }
    case = cases[name]
    assert case.semantic_edits == expected_edits
    assert case.expected_realignment_index == expected_realignment
    assert len(case.predicted) == len(tokens)

    result = evaluate_perturbation(
        case,
        target=tokens,
        layout=layout,
        class_labels=labels,
        vocabulary_size=VOCAB_SIZE,
        confidence=CONFIDENCE,
        class_id_to_name=PRETRAIN_CLASS_ID_TO_NAME,
    )
    assert result["positional_mismatches"] == expected_mismatches

    # Cross entropy is fully determined by the mismatch count under pseudo-logits.
    scored_positions = len(tokens) - 1
    correct_ce = -math.log(CONFIDENCE)
    wrong_ce = -math.log((1.0 - CONFIDENCE) / (VOCAB_SIZE - 1))
    expected_ce = (
        scored_positions - expected_mismatches
    ) * correct_ce + expected_mismatches * wrong_ce
    # float32 pseudo-logits, so compare relatively rather than to 1e-6 absolute.
    assert result["unweighted_ce_nats"] == pytest.approx(expected_ce, rel=1e-5)
    assert result["scored_positions"] == scored_positions

    if expected_edits:
        assert result["mismatch_amplification"] == pytest.approx(
            expected_mismatches / expected_edits
        )
    if expected_realignment is not None:
        assert result["penalty_stops_where_intended"] is True


def test_bounded_shifts_preserve_the_delimiter_count_and_displace_exactly_one():
    tokens = terminated_canvas()
    layout = parse_canvas_layout(tokens)
    cases = {
        case.name: case
        for case in build_perturbations(tokens, layout, vocabulary_size=VOCAB_SIZE)
    }
    target_delimiters = list(layout.delimiter_indices)
    for name, displacement in (
        ("content_deletion_local_shift", -1),
        ("content_insertion_local_shift", +1),
    ):
        predicted = cases[name].predicted
        predicted_delimiters = [
            index
            for index, token in enumerate(predicted)
            if index >= 2 and token == DELIMITER_ID
        ]
        assert len(predicted_delimiters) == len(target_delimiters)
        drifts = [
            predicted_delimiters[i] - target_delimiters[i]
            for i in range(len(target_delimiters))
        ]
        assert drifts.count(displacement) == 1
        assert sum(1 for drift in drifts if drift != 0) == 1


def test_unbounded_shift_propagates_much_further_than_the_bounded_one():
    tokens = terminated_canvas()
    layout = parse_canvas_layout(tokens)
    labels = canvas_class_labels(tokens)
    cases = {
        case.name: case
        for case in build_perturbations(tokens, layout, vocabulary_size=VOCAB_SIZE)
    }
    results = {
        name: evaluate_perturbation(
            case,
            target=tokens,
            layout=layout,
            class_labels=labels,
            vocabulary_size=VOCAB_SIZE,
            confidence=CONFIDENCE,
            class_id_to_name=PRETRAIN_CLASS_ID_TO_NAME,
        )
        for name, case in cases.items()
    }
    bounded = results["content_deletion_local_shift"]
    unbounded = results["content_deletion_global_shift"]
    assert unbounded["positional_mismatches"] > bounded["positional_mismatches"]
    assert unbounded["penalty_span"] > bounded["penalty_span"]
    assert unbounded["expected_realignment_index"] is None
    assert unbounded["penalty_stops_where_intended"] is None


def test_perturbation_cross_entropy_is_split_across_the_reported_classes():
    tokens = terminated_canvas()
    layout = parse_canvas_layout(tokens)
    labels = canvas_class_labels(tokens)
    case = next(
        entry
        for entry in build_perturbations(tokens, layout, vocabulary_size=VOCAB_SIZE)
        if entry.name == "content_deletion_local_shift"
    )
    result = evaluate_perturbation(
        case,
        target=tokens,
        layout=layout,
        class_labels=labels,
        vocabulary_size=VOCAB_SIZE,
        confidence=CONFIDENCE,
        class_id_to_name=PRETRAIN_CLASS_ID_TO_NAME,
    )
    per_class = result["per_class"]
    assert set(per_class) == {"win-loss", "enemy-observed", "[DELIMITER]", "[END]", "[PAD]"}
    assert sum(entry["positions"] for entry in per_class.values()) == len(tokens) - 1
    assert sum(entry["ce_nats"] for entry in per_class.values()) == pytest.approx(
        result["unweighted_ce_nats"], rel=1e-6
    )
    # Semantic [PAD] is scored: it holds real positions in this canvas.
    assert per_class["[PAD]"]["positions"] == 3
    # The bounded shift displaces this group's delimiter and duplicates the next
    # group's first token, so exactly one delimiter position mismatches.
    assert per_class["[DELIMITER]"]["mismatches"] == 1


def test_deletion_offset_sweep_matches_its_closed_form():
    tokens = terminated_canvas()
    layout = parse_canvas_layout(tokens)
    focus = select_focus_timestep(layout)
    sweep = deletion_offset_sweep(tokens, focus)
    assert sweep["content_length"] == 5
    assert sweep["min_positional_mismatches"] == 2
    assert sweep["max_positional_mismatches"] == 5
    assert sweep["mean_positional_mismatches"] == pytest.approx((5 + 4 + 3 + 3 + 2) / 5)
    assert sweep["mean_mismatch_amplification"] == pytest.approx(
        sweep["mean_positional_mismatches"] / 2.0
    )


def test_runs_of_identical_tokens_absorb_the_shift():
    """Canonical sorting damps the alignment penalty; the probe must show that.

    Serialization sorts by entity type, so a timestep holding many copies of one
    unit has long runs of identical ids. A left shift across such a run changes
    NOTHING at those coordinates, which is a real and reportable property of the
    objective's geometry -- not a bug in the measurement.
    """

    tokens = [BOS_ID, WIN_ID] + [10] * 8 + [DELIMITER_ID, 11, 12, DELIMITER_ID, END_ID, PAD_ID]
    layout = parse_canvas_layout(tokens)
    focus = select_focus_timestep(layout)
    sweep = deletion_offset_sweep(tokens, focus)
    # Only the run boundary and the delimiter positions can mismatch.
    assert sweep["max_positional_mismatches"] == 2


# ---------------------------------------------------------------------------
# Pooled aggregation
# ---------------------------------------------------------------------------


def test_pooled_counters_form_ratios_from_sums_not_from_means():
    """A big slice and a tiny slice must not be averaged as equals."""

    big = PooledCounters()
    big.add("scored_positions", 1000)
    big.add("unweighted_ce_sum", 100.0)
    small = PooledCounters()
    small.add("scored_positions", 1)
    small.add("unweighted_ce_sum", 10.0)

    pooled = PooledCounters()
    pooled.merge(big)
    pooled.merge(small)
    report = finalize_counters(pooled)
    assert report["unweighted_ce_nats"] == pytest.approx(110.0 / 1001.0)
    # A mean of per-slice means would have produced (0.1 + 10.0) / 2 = 5.05.
    assert report["unweighted_ce_nats"] < 1.0


def test_finalize_counters_returns_none_for_an_empty_slice():
    report = finalize_counters(PooledCounters())
    assert report["scored_positions"] == 0
    assert report["unweighted_ce_nats"] is None
    assert report["noised_argmax_accuracy"] is None
    assert "multiset_f1" not in report


def test_alignment_gap_is_reported_over_the_oracle_covered_positions_only():
    counters = PooledCounters()
    counters.add("content_positions", 100)
    counters.add("content_ce_sum", 200.0)
    counters.add("oracle_positions", 80)
    counters.add("oracle_positional_ce_sum", 160.0)
    counters.add("oracle_ce_sum", 120.0)
    report = finalize_counters(counters)
    assert report["content_positional_ce_nats"] == pytest.approx(2.0)
    assert report["oracle_aligned_ce_nats"] == pytest.approx(1.5)
    assert report["alignment_gap_nats"] == pytest.approx(0.5)
    assert report["alignment_gap_fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Batch scoring: masks, determinism, and control transforms
# ---------------------------------------------------------------------------


def _synthetic_batch(rows: tuple[list[int], ...]) -> DiffusionBatch:
    """Build a minimal ``DiffusionBatch`` over hand-written canvases.

    Rows of different lengths are right-padded with batch-shape ``[PAD]``, which
    is excluded from ``canvas_attention_mask`` and therefore from
    ``canvas_loss_mask`` -- exactly what the production collater does.
    """

    width = max(len(row) for row in rows)
    target = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    labels = torch.full((len(rows), width), CLASS_PAD, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        target[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        labels[index, : len(row)] = torch.tensor(
            canvas_class_labels(row), dtype=torch.long
        )
        attention[index, : len(row)] = True
    loss_mask = attention.clone()
    loss_mask[:, 0] = False
    features = InputFeatures(
        continuous_values=torch.zeros((len(rows), 1, 4)),
        continuous_validity=torch.zeros((len(rows), 1, 4), dtype=torch.bool),
        categorical_values=torch.zeros((len(rows), 1, 3)),
        allegiance_values=torch.zeros((len(rows), 1, 1)),
        feature_mask=torch.zeros((len(rows), 1), dtype=torch.bool),
    )
    return DiffusionBatch(
        input_token_ids=torch.full((len(rows), 1), PAD_ID, dtype=torch.long),
        input_attention_mask=torch.ones((len(rows), 1), dtype=torch.bool),
        input_lengths=torch.ones((len(rows),), dtype=torch.long),
        target_canvas=target,
        canvas_attention_mask=attention,
        class_labels=labels,
        canvas_loss_mask=loss_mask,
        terminated=torch.ones((len(rows),), dtype=torch.bool),
        truncated=torch.zeros((len(rows),), dtype=torch.bool),
        perspective_ids=torch.ones((len(rows),), dtype=torch.long),
        input_timestep_counts=torch.ones((len(rows),), dtype=torch.long),
        enemy_future_timestep_counts=torch.zeros((len(rows),), dtype=torch.long),
        canvas_prediction_distances=torch.full((len(rows), width), -1, dtype=torch.long),
        input_records=[],
        canvas_metadata=[],
        input_features=features,
    )


def _score(batch: DiffusionBatch, logits: torch.Tensor):
    from scripts.timestep_alignment_probe import WindowIdentity

    slices: dict = {}
    rows: list = []
    score_batch(
        canvas_logits=logits,
        batch=batch,
        changed_positions=torch.ones_like(batch.target_canvas, dtype=torch.bool),
        class_weights=torch.ones(max(PRETRAIN_CLASS_ID_TO_NAME) + 1),
        class_id_to_name=PRETRAIN_CLASS_ID_TO_NAME,
        identities=[
            WindowIdentity("replay", "p1", 0, 1) for _ in range(batch.target_canvas.shape[0])
        ],
        noise_level=1.0,
        dominant_ids=frozenset(),
        slices=slices,
        token_counts={},
        per_window_rows=rows,
        compute_oracle=True,
        oracle_max_span=64,
        vocab_size=VOCAB_SIZE,
    )
    return slices, rows


def test_scoring_counts_semantic_pad_but_never_batch_shape_padding():
    short = terminated_canvas()[:18] + [PAD_ID]  # ends on a delimiter then one pad
    long = terminated_canvas()
    batch = _synthetic_batch((short, long))
    logits = torch.zeros((2, batch.target_canvas.shape[1], VOCAB_SIZE))
    slices, _rows = _score(batch, logits)
    overall = finalize_counters(slices[("overall", "all")])
    # Exactly the live loss mask: every real position except clamped [BOS].
    assert overall["scored_positions"] == int(batch.canvas_loss_mask.sum().item())
    assert overall["scored_positions"] == (len(short) - 1) + (len(long) - 1)
    pad_slice = finalize_counters(slices[("class", "[PAD]")])
    # One semantic pad in the short row, three in the long one. The three
    # batch-shape pads appended to the short row are NOT counted.
    assert pad_slice["scored_positions"] == 4


def test_perfect_logits_give_zero_alignment_gap_and_a_perfect_multiset():
    tokens = terminated_canvas()
    batch = _synthetic_batch((tokens,))
    logits = pseudo_logits(tokens, vocabulary_size=VOCAB_SIZE, confidence=0.999).unsqueeze(0)
    slices, rows = _score(batch, logits)
    overall = finalize_counters(slices[("overall", "all")])
    assert overall["multiset_f1"] == pytest.approx(1.0)
    assert overall["exact_multiset_rate"] == pytest.approx(1.0)
    assert overall["alignment_gap_nats"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["predicted_delimiters"] == 3
    assert rows[0]["target_delimiters"] == 3


def test_reordered_prediction_shows_a_positive_alignment_gap():
    """The gap is exactly the point of the diagnostic; it must be non-zero here."""

    tokens = terminated_canvas()
    reordered = list(tokens)
    # Reverse timestep 1's content: same multiset, different coordinates.
    reordered[7:12] = list(reversed(tokens[7:12]))
    batch = _synthetic_batch((tokens,))
    logits = pseudo_logits(reordered, vocabulary_size=VOCAB_SIZE, confidence=0.99).unsqueeze(0)
    slices, _rows = _score(batch, logits)
    overall = finalize_counters(slices[("overall", "all")])
    assert overall["alignment_gap_nats"] > 0.0
    # Order-invariant multiset comparison sees no error at all.
    assert overall["exact_multiset_rate"] == pytest.approx(1.0)


def test_scoring_is_deterministic_for_identical_inputs():
    tokens = terminated_canvas()
    batch = _synthetic_batch((tokens,))
    logits = pseudo_logits(tokens, vocabulary_size=VOCAB_SIZE, confidence=0.8).unsqueeze(0)
    first, _ = _score(batch, logits)
    second, _ = _score(batch, logits)
    assert finalize_counters(first[("overall", "all")]) == finalize_counters(
        second[("overall", "all")]
    )


def test_coupled_corruption_is_nested_across_noise_levels():
    """Re-seeding one generator per level couples the sweep, as the probe relies on.

    The probe re-seeds an identical ``torch.Generator`` before every
    ``corrupt_batch`` call so the Bernoulli draw and the replacement tokens are
    shared across t. That makes the corrupted sets NESTED, so the sweep changes
    which truthful anchors survive rather than swapping in unrelated canvases.
    """

    from thesis_ml.train.corruption import corrupt_batch
    from thesis_ml.config import DiffusionScheduleConfig

    schedule = DiffusionScheduleConfig(
        name="linear",
        t_distribution="power",
        min=0.0,
        max=1.0,
        t_one_fraction=0.0,
        t_distribution_power=2.0,
    )
    canvas = torch.tensor([terminated_canvas()], dtype=torch.long)
    noise_mask = torch.ones_like(canvas, dtype=torch.bool)
    noise_mask[:, 0] = False
    outputs = {}
    for level in (0.25, 0.5, 0.9, 1.0):
        generator = torch.Generator().manual_seed(1234)
        outputs[level] = corrupt_batch(
            input_token_ids=torch.zeros((1, 1), dtype=torch.long),
            target_canvas=canvas,
            process="uniform",
            schedule=schedule,
            vocab_size=VOCAB_SIZE,
            generator=generator,
            t=level,
            canvas_noise_mask=noise_mask,
        )
    levels = sorted(outputs)
    for lower, higher in zip(levels, levels[1:]):
        low_mask = outputs[lower].corrupted_positions
        high_mask = outputs[higher].corrupted_positions
        assert bool((low_mask & ~high_mask).sum() == 0)
    # Replacement tokens are shared, so every commonly corrupted position holds
    # the SAME replacement at every level.
    shared = outputs[0.25].corrupted_positions
    assert torch.equal(
        outputs[0.25].noised_canvas[shared], outputs[1.0].noised_canvas[shared]
    )


def test_shuffled_input_rotates_rows_without_touching_the_canvas():
    batch = _synthetic_batch((terminated_canvas(), truncated_canvas()))
    batch = type(batch)(
        **{
            **batch.__dict__,
            "input_token_ids": torch.tensor([[10, 11], [12, 13]], dtype=torch.long),
            "input_attention_mask": torch.ones((2, 2), dtype=torch.bool),
            "input_lengths": torch.tensor([2, 2], dtype=torch.long),
            "input_features": InputFeatures(
                continuous_values=torch.zeros((2, 2, 4)),
                continuous_validity=torch.zeros((2, 2, 4), dtype=torch.bool),
                categorical_values=torch.zeros((2, 2, 3)),
                allegiance_values=torch.zeros((2, 2, 1)),
                feature_mask=torch.zeros((2, 2), dtype=torch.bool),
            ),
        }
    )
    overrides = _shuffle_input_rows(batch)
    assert overrides["input_token_ids"].tolist() == [[12, 13], [10, 11]]


def test_enemy_stripping_removes_enemy_tokens_and_keeps_structure():
    batch = _synthetic_batch((terminated_canvas(),))
    tokens = torch.tensor([[20, 21, 22, DELIMITER_ID]], dtype=torch.long)
    allegiance = torch.tensor([[[1.0], [-1.0], [-1.0], [0.0]]])
    feature_mask = torch.tensor([[True, True, True, False]])
    batch = type(batch)(
        **{
            **batch.__dict__,
            "input_token_ids": tokens,
            "input_attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "input_lengths": torch.tensor([4], dtype=torch.long),
            "input_features": InputFeatures(
                continuous_values=torch.zeros((1, 4, 4)),
                continuous_validity=torch.zeros((1, 4, 4), dtype=torch.bool),
                categorical_values=torch.zeros((1, 4, 3)),
                allegiance_values=allegiance,
                feature_mask=feature_mask,
            ),
        }
    )
    overrides = _strip_enemy_input(batch)
    kept = overrides["input_token_ids"][overrides["input_attention_mask"]]
    assert kept.tolist() == [20, DELIMITER_ID]
    assert overrides["input_lengths"].tolist() == [2]
    # Left padding, matching the collater's convention.
    assert overrides["input_attention_mask"].tolist() == [[True, True]]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


class _Window:
    """Minimal stand-in for a ``WindowManifestEntry``.

    Only ``replay_id`` and ``perspective_player`` are read by ``select_windows``.
    """

    def __init__(self, replay_id: str, perspective: str, ordinal: int) -> None:
        self.replay_id = replay_id
        self.perspective_player = perspective
        self.ordinal = ordinal

    def __repr__(self) -> str:  # pragma: no cover -- test failure output only
        return f"{self.replay_id}/{self.perspective_player}#{self.ordinal}"


def test_window_selection_spreads_across_replays_before_deepening():
    windows = [
        _Window(replay, "p1", ordinal)
        for replay in ("b", "a", "c")
        for ordinal in range(4)
    ]
    selected = select_windows(windows, max_examples=6, windows_per_replay=2)
    assert [(window.replay_id, window.ordinal) for window in selected] == [
        ("a", 0),
        ("b", 0),
        ("c", 0),
        ("a", 1),
        ("b", 1),
        ("c", 1),
    ]


def test_window_selection_reaches_both_perspectives_at_the_first_slot():
    """The manifest lists a replay's p1 windows before its p2 windows.

    Grouping by replay alone would therefore fill a small budget with p1 only and
    leave the perspective breakdown empty, so the grouping key must include the
    perspective.
    """

    windows = [
        _Window("a", perspective, ordinal)
        for perspective in ("p1", "p2")
        for ordinal in range(5)
    ]
    selected = select_windows(windows, max_examples=4, windows_per_replay=2)
    assert [window.perspective_player for window in selected] == ["p1", "p2", "p1", "p2"]


def test_window_selection_is_deterministic():
    windows = [
        _Window(replay, "p1", ordinal) for replay in ("z", "y") for ordinal in range(3)
    ]
    first = select_windows(windows, max_examples=4, windows_per_replay=3)
    second = select_windows(windows, max_examples=4, windows_per_replay=3)
    assert [w.replay_id for w in first] == [w.replay_id for w in second]
    assert [w.ordinal for w in first] == [w.ordinal for w in second]


# ---------------------------------------------------------------------------
# Write boundary
# ---------------------------------------------------------------------------


def test_generated_artifacts_stay_under_the_scripts_output_tree():
    assert DEFAULT_OUTPUT_DIR == Path("scripts/output/timestep_alignment_probe")
    assert DEFAULT_OUTPUT_DIR.parts[:2] == ("scripts", "output")


def test_probe_never_writes_into_model_inference_tests_or_the_probed_run():
    """No write API in the probe may target a forbidden tree.

    ``Model_Inference_Tests/`` is contractually off limits for training-objective
    diagnostics, and the probed training run must stay untouched. This walks the
    module's AST rather than grepping so a write hidden inside a helper is still
    caught.
    """

    source = PROBE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = ("Model_Inference_Tests", "checkpoints", "tests/output")
    writing_calls = {"write_text", "write_bytes", "mkdir", "open", "touch", "unlink", "rmdir"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else None
        if name not in writing_calls:
            continue
        rendered = ast.dump(node)
        # `open` in read-binary mode is how the sha256 helper reads a file.
        if name == "open" and "'rb'" in rendered:
            continue
        for marker in forbidden:
            assert marker not in rendered, f"{name} call may write into {marker}"

    # The checkpoint is only ever loaded, never saved back.
    assert "save_checkpoint" not in source
    assert "torch.save" not in source


def test_window_selection_can_sample_the_terminal_windows():
    """``[END]`` only exists in a replay's LAST window; the selector must reach it."""

    windows = [_Window("a", "p1", ordinal) for ordinal in range(5)]
    first = select_windows(windows, max_examples=2, windows_per_replay=2, position="first")
    last = select_windows(windows, max_examples=2, windows_per_replay=2, position="last")
    assert [window.ordinal for window in first] == [0, 1]
    assert [window.ordinal for window in last] == [4, 3]


def test_window_selection_rejects_an_unknown_position():
    with pytest.raises(ValueError):
        select_windows([_Window("a", "p1", 0)], max_examples=1, windows_per_replay=1, position="middle")
