"""Unit coverage for `scripts/rare_token_signal_probe.py`.

Pins the closed-form behaviour of the model-independent shift-exposure arm and
the three-level per-token scoring, so the diagnostic's headline ratios cannot
silently change meaning. Requires no CUDA, no checkpoint, and no replay data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rare_token_signal_probe import (  # noqa: E402
    FREQUENCY_BUCKETS,
    SIGNAL_FIELDS,
    SIGNAL_FIELD_INDEX,
    TECH_BUILDING_NAMES,
    accumulate_exposure,
    boundary_flags,
    bucket_token_rows,
    run_lengths,
    summarize_token_signal,
)
from thesis_ml.vocab.special_tokens import CONTENT_TOKEN_OFFSET, DELIMITER_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Arm A -- boundary/run structure
# ---------------------------------------------------------------------------


def test_boundary_flags_marks_only_run_ends():
    # Three runs: [10,10,10], [11,11], [12]. A left shift changes the token only
    # at the last coordinate of each run.
    content = [10, 10, 10, 11, 11, 12]
    assert boundary_flags(content).tolist() == [
        False,
        False,
        True,
        False,
        True,
        True,
    ]


def test_boundary_flags_final_coordinate_is_always_a_boundary():
    """The delimiter shifts into the last content slot, so it always breaks."""

    for content in ([7], [7, 7], [7, 7, 7, 7]):
        assert boundary_flags(content)[-1]


def test_boundary_flags_final_coordinate_breaks_even_against_delimiter_id():
    """A content id can never equal DELIMITER_ID, so the sentinel always differs."""

    assert DELIMITER_ID < CONTENT_TOKEN_OFFSET
    assert boundary_flags([CONTENT_TOKEN_OFFSET]).tolist() == [True]


def test_boundary_flags_empty_group():
    assert boundary_flags([]).tolist() == []


def test_run_lengths_matches_run_structure():
    assert run_lengths([10, 10, 10, 11, 11, 12]).tolist() == [3, 3, 3, 2, 2, 1]


def test_run_lengths_empty_group():
    assert run_lengths([]).tolist() == []


def test_singleton_type_is_fully_exposed_and_long_run_is_mostly_protected():
    """The central claim of Arm A, pinned as a closed form.

    In ``[10]*10 + [11]``, the singleton id 11 sits at a run of length 1 and is
    100% boundary; id 10 sits in a run of 10 and is 1/10 boundary. The pooled
    amplification is small BECAUSE the rare type absorbs all of it.
    """

    totals: dict[int, dict[str, float]] = {}
    accumulate_exposure([10] * 10 + [11], totals)

    common = totals[10]
    rare = totals[11]
    assert common["occurrences"] == 10
    assert common["boundary_occurrences"] == 1  # only the last of the run
    assert rare["occurrences"] == 1
    assert rare["boundary_occurrences"] == 1  # entirely boundary

    assert common["boundary_occurrences"] / common["occurrences"] == pytest.approx(0.1)
    assert rare["boundary_occurrences"] / rare["occurrences"] == pytest.approx(1.0)


def test_expected_hits_uses_uniform_deletion_offset():
    """``expected_hits`` marginalizes over which slot the deletion removes.

    In a group of length 11, the boundary coordinate for id 10 is index 9 and is
    reached by 10 of the 11 offsets; the singleton at index 10 is reached by all
    11. Both weights are ``(coordinate + 1) / length``.
    """

    totals: dict[int, dict[str, float]] = {}
    accumulate_exposure([10] * 10 + [11], totals)
    assert totals[10]["expected_hits"] == pytest.approx(10 / 11)
    assert totals[11]["expected_hits"] == pytest.approx(11 / 11)


def test_prefix_count_is_the_upstream_arithmetic_a_type_depends_on():
    """A type's coordinate is determined by how many tokens precede it."""

    totals: dict[int, dict[str, float]] = {}
    accumulate_exposure([10, 10, 10, 11], totals)
    # id 10 sits at prefixes 0, 1, 2; id 11 sits at prefix 3.
    assert totals[10]["prefix_sum"] == pytest.approx(0 + 1 + 2)
    assert totals[11]["prefix_sum"] == pytest.approx(3)


def test_accumulate_exposure_pools_across_timesteps():
    totals: dict[int, dict[str, float]] = {}
    accumulate_exposure([10, 10], totals)
    accumulate_exposure([10, 11], totals)
    assert totals[10]["occurrences"] == 3
    assert totals[10]["boundary_occurrences"] == 2  # last of group 1, first of group 2
    assert totals[11]["occurrences"] == 1


def test_accumulate_exposure_ignores_empty_group():
    totals: dict[int, dict[str, float]] = {}
    accumulate_exposure([], totals)
    assert totals == {}


# ---------------------------------------------------------------------------
# Arm B -- three-level scoring summary
# ---------------------------------------------------------------------------


VOCAB = CONTENT_TOKEN_OFFSET + 4


def _matrix() -> np.ndarray:
    return np.zeros((len(SIGNAL_FIELDS), VOCAB), dtype=np.float64)


def _set(matrix: np.ndarray, field: str, token_id: int, value: float) -> None:
    matrix[SIGNAL_FIELD_INDEX[field], token_id] = value


def test_summarize_forms_every_ratio_from_pooled_sums():
    token = CONTENT_TOKEN_OFFSET
    matrix = _matrix()
    _set(matrix, "target_positions", token, 100.0)
    _set(matrix, "positional_tp", token, 10.0)
    _set(matrix, "timestep_overlap", token, 80.0)
    _set(matrix, "spans_with_target", token, 50.0)
    _set(matrix, "spans_with_any_prediction", token, 45.0)
    _set(matrix, "soft_mass_at_target", token, 25.0)
    _set(matrix, "soft_expected_in_spans", token, 90.0)
    _set(matrix, "target_count_in_spans", token, 100.0)
    _set(matrix, "ce_sum_at_target", token, 200.0)
    _set(matrix, "weighted_ce_sum_at_target", token, 200.0)

    class _Vocab:
        id_to_name = {token: "cyberneticscore"}

    rows = summarize_token_signal(
        matrix,
        vocabulary=_Vocab(),
        timesteps_scored=50,
        total_weighted_ce=1000.0,
        min_occurrences=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["token_name"] == "cyberneticscore"
    assert row["is_tech_building"] is True
    assert row["positional_recall"] == pytest.approx(0.10)
    assert row["timestep_recall"] == pytest.approx(0.80)
    assert row["span_presence_recall"] == pytest.approx(0.90)
    assert row["mean_probability_at_target"] == pytest.approx(0.25)
    assert row["expected_over_target_count"] == pytest.approx(0.90)
    assert row["mean_ce_at_target_nats"] == pytest.approx(2.0)
    assert row["weighted_loss_share"] == pytest.approx(0.20)
    assert row["occurrences_per_timestep"] == pytest.approx(2.0)


def test_timestep_recall_is_never_below_positional_recall():
    """Order-invariant overlap forgives placement, so it can only be larger.

    This is the discriminant the whole arm rests on: the gap between the two is
    the alignment component of a type's error.
    """

    token = CONTENT_TOKEN_OFFSET + 1
    matrix = _matrix()
    _set(matrix, "target_positions", token, 40.0)
    _set(matrix, "positional_tp", token, 4.0)
    _set(matrix, "timestep_overlap", token, 36.0)

    class _Vocab:
        id_to_name = {token: "pylon"}

    row = summarize_token_signal(
        matrix,
        vocabulary=_Vocab(),
        timesteps_scored=10,
        total_weighted_ce=1.0,
        min_occurrences=1,
    )[0]
    assert row["timestep_recall"] >= row["positional_recall"]


def test_summarize_drops_low_support_and_structural_ids():
    matrix = _matrix()
    _set(matrix, "target_positions", CONTENT_TOKEN_OFFSET, 5.0)
    _set(matrix, "target_positions", DELIMITER_ID, 999.0)

    class _Vocab:
        id_to_name = {CONTENT_TOKEN_OFFSET: "probe"}

    assert (
        summarize_token_signal(
            matrix,
            vocabulary=_Vocab(),
            timesteps_scored=1,
            total_weighted_ce=1.0,
            min_occurrences=20,
        )
        == []
    )
    rows = summarize_token_signal(
        matrix,
        vocabulary=_Vocab(),
        timesteps_scored=1,
        total_weighted_ce=1.0,
        min_occurrences=1,
    )
    # The delimiter is below CONTENT_TOKEN_OFFSET and must never appear.
    assert [row["token_id"] for row in rows] == [CONTENT_TOKEN_OFFSET]


def test_buckets_pool_before_forming_ratios():
    """A bucket ratio is support-weighted, never the mean of member ratios."""

    rows = [
        {
            "token_name": "a",
            "is_tech_building": False,
            "target_positions": 900,
            "occurrences_per_timestep": 20.0,
            "positional_recall": 1.0,
            "timestep_recall": 1.0,
            "span_presence_recall": 1.0,
            "mean_probability_at_target": 1.0,
            "expected_over_target_count": 1.0,
            "present_absent_rate_ratio": 2.0,
            "mean_ce_at_target_nats": 0.0,
            "weighted_loss_share": 0.01,
        },
        {
            "token_name": "b",
            "is_tech_building": False,
            "target_positions": 100,
            "occurrences_per_timestep": 20.0,
            "positional_recall": 0.0,
            "timestep_recall": 0.0,
            "span_presence_recall": 0.0,
            "mean_probability_at_target": 0.0,
            "expected_over_target_count": 0.0,
            "present_absent_rate_ratio": 1.0,
            "mean_ce_at_target_nats": 10.0,
            "weighted_loss_share": 0.02,
        },
    ]
    buckets = bucket_token_rows(rows)
    high = buckets["frequency/f5_dominant_4_plus"]
    assert high["target_positions"] == 1000
    # Support-weighted: 900/1000, not the unweighted mean of 1.0 and 0.0.
    assert high["positional_recall"] == pytest.approx(0.9)
    assert high["mean_ce_at_target_nats"] == pytest.approx(1.0)
    # Loss share is additive across members, not weighted.
    assert high["weighted_loss_share"] == pytest.approx(0.03)
    # The pooled all-content bucket sees every row exactly once.
    assert buckets["set/all_content"]["target_positions"] == 1000
    assert buckets["set/all_content"]["positional_recall"] == pytest.approx(0.9)


def test_frequency_buckets_are_contiguous_and_exhaustive():
    assert FREQUENCY_BUCKETS[0][1] == 0.0
    assert FREQUENCY_BUCKETS[-1][2] == float("inf")
    for earlier, later in zip(FREQUENCY_BUCKETS, FREQUENCY_BUCKETS[1:]):
        assert earlier[2] == later[1]


def test_tech_and_non_tech_sets_partition_the_rows():
    rows = [
        {
            "token_name": "cyberneticscore",
            "is_tech_building": True,
            "target_positions": 10,
            "occurrences_per_timestep": 1.0,
            "positional_recall": 0.0,
            "timestep_recall": 0.0,
            "span_presence_recall": 0.0,
            "mean_probability_at_target": 0.0,
            "expected_over_target_count": 0.0,
            "present_absent_rate_ratio": 1.0,
            "mean_ce_at_target_nats": 1.0,
            "weighted_loss_share": 0.0,
        },
        {
            "token_name": "probe",
            "is_tech_building": False,
            "target_positions": 10,
            "occurrences_per_timestep": 30.0,
            "positional_recall": 1.0,
            "timestep_recall": 1.0,
            "span_presence_recall": 1.0,
            "mean_probability_at_target": 1.0,
            "expected_over_target_count": 1.0,
            "present_absent_rate_ratio": 2.0,
            "mean_ce_at_target_nats": 0.0,
            "weighted_loss_share": 0.0,
        },
    ]
    buckets = bucket_token_rows(rows)
    assert buckets["set/tech_buildings"]["token_types"] == 1
    assert buckets["set/non_tech"]["token_types"] == 1
    assert (
        buckets["set/tech_buildings"]["target_positions"]
        + buckets["set/non_tech"]["target_positions"]
        == buckets["set/all_content"]["target_positions"]
        == 20
    )


def test_tech_building_names_all_exist_in_the_shipped_dictionary():
    """The named tech set is an editorial choice; it must still be real."""

    from thesis_ml.config import load_config
    from thesis_ml.vocab.content_vocab import load_content_vocabulary

    config = load_config(Path("configs/smallTrainingTestV3.yaml"))
    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    assert TECH_BUILDING_NAMES <= set(vocabulary.name_to_id)


def test_tech_buildings_sort_after_protoss_economy_structures():
    """Pins the ordering fact the investigation's mechanism depends on.

    Canonical serialization sorts by SC2 source id, so a pylon/gateway miscount
    displaces every Protoss tech-unlock building in that timestep.
    """

    from thesis_ml.config import load_config
    from thesis_ml.vocab.content_vocab import load_content_vocabulary

    config = load_config(Path("configs/smallTrainingTestV3.yaml"))
    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    source = vocabulary.name_to_source_id
    economy = max(source[name] for name in ("nexus", "pylon", "assimilator", "gateway"))
    for name in (
        "fleetbeacon",
        "twilightcouncil",
        "stargate",
        "templararchive",
        "darkshrine",
        "roboticsbay",
        "roboticsfacility",
        "cyberneticscore",
    ):
        assert source[name] > economy


def test_present_absent_rate_ratio_detects_a_sprayed_base_rate():
    """The control that separates timestep knowledge from a corpus base rate.

    A model that puts the SAME soft rate per position inside spans that contain
    the type and spans that do not has no per-timestep knowledge of it, and the
    ratio must be exactly 1.0. Real knowledge pushes it above 1.
    """

    token = CONTENT_TOKEN_OFFSET + 2
    matrix = _matrix()
    _set(matrix, "target_positions", token, 50.0)
    _set(matrix, "soft_expected_in_spans", token, 10.0)
    _set(matrix, "present_span_positions", token, 100.0)   # rate 0.10
    _set(matrix, "soft_expected_in_absent_spans", token, 20.0)
    _set(matrix, "absent_span_positions", token, 200.0)    # rate 0.10

    class _Vocab:
        id_to_name = {token: "sprayed"}

    row = summarize_token_signal(
        matrix,
        vocabulary=_Vocab(),
        timesteps_scored=10,
        total_weighted_ce=1.0,
        min_occurrences=1,
    )[0]
    assert row["present_absent_rate_ratio"] == pytest.approx(1.0)

    # Quadruple the soft rate inside the spans that actually contain the type.
    _set(matrix, "soft_expected_in_spans", token, 40.0)
    row = summarize_token_signal(
        matrix,
        vocabulary=_Vocab(),
        timesteps_scored=10,
        total_weighted_ce=1.0,
        min_occurrences=1,
    )[0]
    assert row["present_absent_rate_ratio"] == pytest.approx(4.0)
