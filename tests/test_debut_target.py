"""Tests for the debut build-order + win/loss fine-tuning target (debut_mode).

These tests exercise the target builder ``_build_debut_target`` and the outcome
resolver ``resolve_replay_outcome`` added for the fine-tuning path. They use a
tiny synthetic in-memory replay so they do not depend on any large fixture, and
they gate the one test that needs real on-disk metadata so it skips cleanly when
that data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis_ml.config import load_config
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
    DEBUT_CLASS_ID_TO_NAME,
    _build_artifact_target,
    _build_debut_target,
    resolve_replay_outcome,
)
from thesis_ml.data.windowing import ENTITY_CODE, P1_CODE, P2_CODE, WindowManifestEntry
from thesis_ml.model.embedding import STAT_KEYS
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, LOSS_ID, PAD_ID, WIN_ID


ROOT = Path(__file__).resolve().parents[1]
# Real processed quickstart data ships parquet + sibling json/ metadata, which
# is what resolve_replay_outcome reads. Absent in some checkouts -> skip.
QUICKSTART = ROOT / "data" / "processed" / "quickstart"


# ---------------------------------------------------------------------------
# Synthetic replay fixtures
# ---------------------------------------------------------------------------

# Fake token ids (content tokens are >= 100 in the real vocab) and their names.
_TOKEN_NAMES = {100: "marine", 101: "marauder", 102: "medivac", 103: "scv"}


class _FakeVocabulary:
    """Minimal stand-in for ContentVocabulary exposing only token_name_for."""

    def token_name_for(self, token_id: int) -> str:
        return _TOKEN_NAMES[int(token_id)]


class _FakeReplay:
    """Tiny in-memory replay matching the attributes _artifact_* helpers read.

    Timesteps (enemy = p2, code 2):
        t0: p2 marine, p1 scv (self, ignored)
        t1: p2 marine            -> no new marine instance -> empty debut timestep
        t2: p2 medivac           -> first medivac -> visible-debut (in window)
        t3: p2 marine, p2 marine -> a second marine instance -> future-debut
        t4: p2 marauder          -> first marauder -> future-debut, final timestep
    """

    def __init__(self) -> None:
        token_ids = [100, 103, 100, 102, 100, 100, 101]
        owners = [P2_CODE, P1_CODE, P2_CODE, P2_CODE, P2_CODE, P2_CODE, P2_CODE]
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.owners = np.asarray(owners, dtype=np.uint8)
        self.kinds = np.full(len(token_ids), ENTITY_CODE, dtype=np.uint8)
        # offsets partition the 7 token positions into 5 timesteps.
        self.offsets = np.asarray([0, 2, 3, 4, 6, 7], dtype=np.int64)
        feature_width = 2 + len(STAT_KEYS)
        self.features = np.zeros((len(token_ids), feature_width), dtype=np.float32)
        self.game_loops = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        self.timestamps = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def token_slice(self, timestep: int) -> slice:
        return slice(int(self.offsets[timestep]), int(self.offsets[timestep + 1]))

    @property
    def timestep_count(self) -> int:
        return len(self.offsets) - 1


def _window() -> WindowManifestEntry:
    """Window over timesteps 0..2 (input length 3); t3, t4 are the future."""

    return WindowManifestEntry(
        replay_id="fake",
        replay_path="fake/parquet/match_fake_game_state.parquet",
        artifact_path="fake/artifact",
        perspective_player="p1",
        start_timestep=0,
        end_timestep=3,
        input_token_count=0,
        enemy_reconstruction_token_count=0,
        replay_timestep_count=5,
    )


def _build(fogged_counts: dict[tuple[int, str], int], *, outcome_id: int = WIN_ID, budget: int = 64):
    return _build_debut_target(
        _FakeReplay(),
        _window(),
        _FakeVocabulary(),
        "p2",
        fogged_counts=fogged_counts,
        budget=budget,
        outcome_id=outcome_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_outcome_token_at_position_zero_with_winloss_class() -> None:
    build = _build({}, outcome_id=LOSS_ID)
    assert build.token_ids[0] == LOSS_ID
    assert build.class_labels[0] == CLASS_WINLOSS
    # The outcome token must appear exactly once, only at position 0.
    assert build.token_ids.count(LOSS_ID) == 1
    assert build.class_labels.count(CLASS_WINLOSS) == 1
    # Canvas is padded to exactly the requested budget.
    assert len(build.token_ids) == 64
    assert len(build.class_labels) == len(build.token_ids)


def test_debut_event_timestep_is_first_appearance() -> None:
    build = _build({})
    # medivac first appears at timestep 2; its debut metadata must carry
    # timestep_index 2 (relative), not a later timestep.
    medivac_positions = [
        meta["timestep_index"]
        for meta in build.metadata
        if meta.get("token_name") == "medivac"
    ]
    assert medivac_positions == [2]
    # marine first appears at timestep 0 (its second instance debuts at t3).
    marine_positions = [
        meta["timestep_index"]
        for meta in build.metadata
        if meta.get("token_name") == "marine"
    ]
    assert marine_positions == [0, 3]


def test_empty_timestep_produces_back_to_back_delimiters() -> None:
    build = _build({})
    # Sequence should be: outcome, marine, DELIM (t0), DELIM (t1 has no debut),
    # medivac, DELIM (t2), ...
    tokens = build.token_ids
    # Find the first two positions that are delimiters and confirm they are
    # adjacent (t0 delimiter immediately followed by t1's bare delimiter).
    first_delim = tokens.index(DELIMITER_ID)
    assert tokens[first_delim] == DELIMITER_ID
    assert tokens[first_delim + 1] == DELIMITER_ID


def test_fog_class_labels_visible_fogged_and_future() -> None:
    # Mark the t0 marine as fogged in the input; it must become fogged-debut.
    build = _build({(0, "marine"): 1})
    labels_by_name_ts = {
        (meta.get("token_name"), meta.get("timestep_index")): label
        for meta, label in zip(build.metadata, build.class_labels)
    }
    # t0 marine debut was fogged in the input -> fogged-debut (class 1).
    assert labels_by_name_ts[("marine", 0)] == CLASS_ENEMY_FOGGED
    # t2 medivac debut is inside the window and not fogged -> visible-debut (0).
    assert labels_by_name_ts[("medivac", 2)] == CLASS_ENEMY_OBSERVED
    # t3 marine debut is beyond the window boundary (3) -> future-debut (2).
    assert labels_by_name_ts[("marine", 3)] == CLASS_ENEMY_FUTURE
    # t4 marauder debut is also in the future.
    assert labels_by_name_ts[("marauder", 4)] == CLASS_ENEMY_FUTURE


def test_terminates_with_end_then_pads() -> None:
    build = _build({})
    assert build.terminated is True
    end_index = build.token_ids.index(END_ID)
    # [END] is preceded by a delimiter and followed only by padding.
    assert build.token_ids[end_index - 1] == DELIMITER_ID
    assert all(token == PAD_ID for token in build.token_ids[end_index + 1 :])
    assert build.class_labels[end_index] == CLASS_END
    assert build.class_labels[-1] == CLASS_PAD


def test_whole_timestep_truncation_when_budget_overflows() -> None:
    # A tiny budget cannot hold the whole game; the builder truncates on whole
    # timesteps, emits no [END], and still pads to budget.
    build = _build({}, budget=5)
    assert len(build.token_ids) == 5
    assert build.truncated is True
    assert build.terminated is False
    assert END_ID not in build.token_ids
    assert PAD_ID in build.token_ids


def test_pretraining_artifact_path_has_no_winloss_token() -> None:
    # The debut_mode-off path (the existing artifact target) must never emit a
    # win/loss token or the CLASS_WINLOSS label. This proves debut logic did not
    # leak into the pretraining canvas.
    build = _build_artifact_target(
        _FakeReplay(),
        _window(),
        _FakeVocabulary(),
        "p2",
        fogged_counts={},
        budget=64,
    )
    assert WIN_ID not in build.token_ids
    assert LOSS_ID not in build.token_ids
    assert CLASS_WINLOSS not in build.class_labels


def test_debut_class_id_to_name_map_is_complete() -> None:
    assert DEBUT_CLASS_ID_TO_NAME == {
        0: "visible-debut",
        1: "fogged-debut",
        2: "future-debut",
        3: "delimiter",
        4: "end",
        5: "pad",
        6: "win-loss",
    }


def test_default_config_debut_mode_off() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    assert config.data.debut_mode is False


@pytest.mark.skipif(
    not (QUICKSTART / "json").exists(),
    reason="processed quickstart metadata not present",
)
def test_resolve_replay_outcome_reads_real_metadata() -> None:
    parquet = QUICKSTART / "parquet" / "match_4745721_game_state.parquet"
    # match_4745721 metadata: p1 = Defeat, p2 = Victory.
    assert resolve_replay_outcome(parquet, "p1") == LOSS_ID
    assert resolve_replay_outcome(parquet, "p2") == WIN_ID


def test_resolve_replay_outcome_fails_loudly_on_missing_metadata() -> None:
    # Point at a parquet whose sibling json/ metadata does not exist; the helper
    # must raise rather than silently defaulting to a win or loss.
    missing = ROOT / "does_not_exist" / "parquet" / "match_none_game_state.parquet"
    with pytest.raises(ValueError):
        resolve_replay_outcome(missing, "p1")
