"""Joint presentation boundaries, shared serialization, and legacy ID preservation."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from thesis_shared.config import ConfigError, load_config, load_sequence_preview_config
from thesis_shared.data.window_policies.unconditioned_joint import iter_joint_windows
from thesis_shared.sequence_formats.unconditioned_joint import build_outcome_footer, iter_joint_timesteps
from thesis_shared.serialize import serialize_snapshot
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import CONDITIONED, JOINT, build_sequence_vocabulary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    config = load_config(ROOT / "config/default.yaml")
    content = load_content_vocabulary(ROOT / "data/Token_Dictionary.json")
    vocabulary = build_sequence_vocabulary(content, sequence_format=JOINT)
    return config, content, vocabulary


def blocks(setup, rows, perspective="p1"):
    config, content, vocabulary = setup
    return list(iter_joint_timesteps(rows, config, content, vocabulary,
                                    perspective=perspective))


def test_legacy_vocabulary_unchanged_and_overlay_contiguous(setup):
    _, content, vocabulary = setup
    legacy = build_sequence_vocabulary(content, sequence_format=CONDITIONED)
    assert all(vocabulary.names[i] == name for i, name in enumerate(legacy.names) if i != 5)
    assert legacy.names[5] == "[LOSS]" and vocabulary.names[5] == "[LOSE]"
    assert vocabulary.names[-2:] == ("[SELF]", "[ENEMY]")
    assert vocabulary.content_ids == legacy.content_ids
    assert vocabulary.vocab_size == content.vocab_size + 2
    assert legacy.identity != vocabulary.identity
    assert vocabulary.token_id("[SELF]") not in vocabulary.content_ids
    with pytest.raises(ValueError, match="explicitly"):
        build_sequence_vocabulary(content, sequence_format="")


def test_real_snapshot_matches_canonical_serializer_in_both_perspectives(setup):
    config, content, _ = setup
    row = pd.read_parquet(ROOT / "tests/fixtures/match_4745722_game_state.parquet").iloc[1]
    baseline = serialize_snapshot(row, config, content, perspective_player="p1")
    for perspective in ("p1", "p2"):
        block = blocks(setup, [row], perspective)[0]
        for owner in ("p1", "p2"):
            assert [t.token_id for t in block if t.role == "content" and t.owner == owner] == [
                r.token_id for r in baseline if r.owner == owner]
        assert block[0].name == "[SELF]" and block[0].owner == perspective
        assert not any(t.role == "outcome" for t in block)
        assert not any(hasattr(t, "raw_attributes") for t in block)


def test_empty_players_still_have_markers_without_outcomes(setup):
    result = blocks(setup, [pd.Series(dtype=object)])[0]
    assert [t.name for t in result] == ["[SELF]", "[ENEMY]", "[DELIMITER]"]


def test_whole_timesteps_tile_and_only_actual_replay_end_gets_end(setup):
    *_, vocabulary = setup
    timesteps = blocks(setup, [pd.Series(dtype=object)] * 3)
    windows = list(iter_joint_windows(timesteps, vocabulary, context_budget=12, perspective="p1", outcomes={"p1": 4, "p2": 5}))
    assert [(w.start_timestep, w.end_timestep) for w in windows] == [(0, 2), (2, 3)]
    assert [t.name for t in windows[0].tokens[:2]] == ["[BOS]", "[DELIMITER]"]
    assert windows[0].tokens[-5].name == "[DELIMITER]"
    for window in windows:
        footer = window.tokens[-5:-1] if window.reached_replay_end else window.tokens[-4:]
        assert [t.name for t in footer] == ["[SELF]", "[WIN]", "[ENEMY]", "[LOSE]"]
        assert sum(t.role == "outcome" for t in window.tokens) == 2
        assert all(t.timestep is None for t in footer)
        assert all(window.loss_mask[-5:-1] if window.reached_replay_end else window.loss_mask[-4:])
    assert not windows[0].reached_replay_end
    assert windows[1].tokens[-1].name == "[END]"
    assert windows[1].reached_replay_end
    assert all(len(w.tokens) <= 12 for w in windows)
    assert all(w.loss_mask == (False,) + (True,) * (len(w.tokens) - 1) for w in windows)


def test_terminal_marker_is_budgeted_and_oversized_step_fails(setup):
    *_, vocabulary = setup
    timestep = blocks(setup, [pd.Series(dtype=object)])[0]
    assert len(next(iter_joint_windows([timestep], vocabulary, context_budget=10, perspective="p1", outcomes={"p1": 4, "p2": 5})).tokens) == 10
    with pytest.raises(ValueError, match="whole timestep"):
        list(iter_joint_windows([timestep + timestep], vocabulary, context_budget=10, perspective="p1", outcomes={"p1": 4, "p2": 5}))
    with pytest.raises(ValueError, match="empty replay"):
        list(iter_joint_windows([], vocabulary, context_budget=10, perspective="p1", outcomes={"p1": 4, "p2": 5}))


def test_missing_format_fails_before_any_replay_is_opened(tmp_path):
    source = (ROOT / "config/sequence_preview.yaml").read_text()
    path = tmp_path / "missing.yaml"
    path.write_text(source.replace("sequence_format: unconditioned_joint_v1\n", ""))
    with pytest.raises(ConfigError, match="sequence_format is required"):
        load_sequence_preview_config(path)


def test_unknown_format_and_noncomplementary_outcomes_fail(setup, tmp_path):
    source = (ROOT / "config/sequence_preview.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("unconditioned_joint_v1", "typo"))
    with pytest.raises(ValueError, match="explicitly"):
        load_sequence_preview_config(path)
    config, content, vocabulary = setup
    with pytest.raises(ValueError, match="one recorded win"):
        build_outcome_footer(vocabulary, perspective="p1", outcomes={"p1": 4, "p2": 4})


def test_shared_preview_import_is_framework_free():
    subprocess.run([sys.executable, "-c", "import sys; import thesis_shared.sequence_formats.preview; "
                    "assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules"], check=True)


@pytest.mark.parametrize("perspective,expected", [("p1", [4, 5]), ("p2", [5, 4])])
def test_footer_swaps_outcomes_and_keeps_all_positions_scored(setup, perspective, expected):
    *_, vocabulary = setup
    timesteps = blocks(setup, [pd.Series(dtype=object)], perspective)
    window = next(iter_joint_windows(timesteps, vocabulary, context_budget=10,
                                    perspective=perspective, outcomes={"p1": 4, "p2": 5}))
    footer = window.tokens[-5:-1]
    assert [t.token_id for t in footer if t.role == "outcome"] == expected
    assert footer[0].owner == perspective
    assert all(window.loss_mask[-5:-1])
    assert not any(t.role == "outcome" for t in window.tokens[:-5])
