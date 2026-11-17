"""Joint presentation boundaries, shared serialization, and legacy ID preservation."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from thesis_shared.config import ConfigError, load_config, load_sequence_preview_config
from thesis_shared.data.replay_outcomes import resolve_replay_race
from thesis_shared.data.window_policies.unconditioned_joint import iter_joint_windows
from thesis_shared.sequence_formats.types import JointReplay
from thesis_shared.sequence_formats.unconditioned_joint import (
    build_outcome_footer, build_replay_header, iter_joint_timesteps)
from thesis_shared.serialize import serialize_snapshot
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import (
    CONDITIONED, JOINT, RACE_TOKENS, build_sequence_vocabulary, race_token_name)

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


RACES = {"p1": "[TERRAN]", "p2": "[ZERG]"}
HEADER = ["[BOS]", "[SELF]", "[TERRAN]", "[ENEMY]", "[ZERG]"]


def replay(timesteps, name="r1", perspective="p1", outcomes=None, races=RACES):
    """Wrap test timesteps as one JointReplay with fixed races and outcomes."""
    return JointReplay(name, perspective, races, outcomes or {"p1": 4, "p2": 5}, timesteps)


def windows_of(vocabulary, *replays, budget, pack=True):
    return list(iter_joint_windows(replays, vocabulary, context_budget=budget, pack_replays=pack))


def test_legacy_vocabulary_unchanged_and_overlay_contiguous(setup):
    _, content, vocabulary = setup
    legacy = build_sequence_vocabulary(content, sequence_format=CONDITIONED)
    assert all(vocabulary.names[i] == name for i, name in enumerate(legacy.names) if i != 5)
    assert legacy.names[5] == "[LOSS]" and vocabulary.names[5] == "[LOSE]"
    assert vocabulary.names[-6:] == ("[SELF]", "[ENEMY]", "[PROTOSS]", "[RANDOM]", "[TERRAN]", "[ZERG]")
    assert vocabulary.content_ids == legacy.content_ids
    assert vocabulary.vocab_size == content.vocab_size + 6
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
    # Header 5 + delimiter 1 + two 3-token timesteps + footer 4 = 16.
    windows = windows_of(vocabulary, replay(timesteps), budget=16)
    assert [(w.segments[0].start_timestep, w.segments[0].end_timestep) for w in windows] == [(0, 2), (2, 3)]
    for window in windows:
        assert [t.name for t in window.tokens[:6]] == HEADER + ["[DELIMITER]"]
        segment = window.segments[0]
        footer = window.tokens[-5:-1] if segment.reached_replay_end else window.tokens[-4:]
        assert [t.name for t in footer] == ["[SELF]", "[WIN]", "[ENEMY]", "[LOSE]"]
        assert sum(t.role == "outcome" for t in window.tokens) == 2
        assert all(t.timestep is None for t in footer)
        assert window.diffusion_clamp_mask == (True,) * 5 + (False,) * (len(window.tokens) - 5)
        assert window.ar_loss_mask == (False,) + (True,) * (len(window.tokens) - 1)
    assert not windows[0].segments[0].reached_replay_end
    assert windows[1].tokens[-1].name == "[END]"
    assert windows[1].segments[0].reached_replay_end
    assert all(len(w.tokens) <= 16 for w in windows)


def test_terminal_marker_is_budgeted_and_oversized_step_fails(setup):
    *_, vocabulary = setup
    timestep = blocks(setup, [pd.Series(dtype=object)])[0]
    # 5 header + 1 delimiter + 3 timestep + 4 footer + END = 14.
    assert len(windows_of(vocabulary, replay([timestep]), budget=14)[0].tokens) == 14
    with pytest.raises(ValueError, match="whole timestep"):
        windows_of(vocabulary, replay([timestep + timestep]), budget=14)
    with pytest.raises(ValueError, match="empty replay"):
        windows_of(vocabulary, replay([]), budget=14)
    with pytest.raises(ValueError, match="at least 11"):
        windows_of(vocabulary, replay([timestep]), budget=10)


def test_replays_pack_after_end_with_their_own_header(setup):
    *_, vocabulary = setup
    step = blocks(setup, [pd.Series(dtype=object)])[0]
    first = replay([step], "a")
    second = replay([step] * 3, "b", races={"p1": "[PROTOSS]", "p2": "[RANDOM]"})
    # Replay a uses 14 tokens. Of the 16 left, one packed timestep of b needs
    # 5 + 1 + 3 + 4 = 13; a second one would need 3 more (16 + 13 > 30).
    windows = windows_of(vocabulary, first, second, budget=30)
    packed = windows[0]
    assert [s.replay_id for s in packed.segments] == ["a", "b"]
    boundary = packed.segments[1].start_position
    assert [t.name for t in packed.tokens[boundary - 5:boundary + 5]] == [
        "[SELF]", "[WIN]", "[ENEMY]", "[LOSE]", "[END]",
        "[BOS]", "[SELF]", "[PROTOSS]", "[ENEMY]", "[RANDOM]"]
    assert packed.segments[1].reached_replay_end is False
    assert packed.tokens[-1].name == "[LOSE]"
    # Only the window's first header is clamped; the packed header is generated.
    assert sum(packed.diffusion_clamp_mask) == 5
    assert all(packed.diffusion_loss_mask[boundary:boundary + 5])
    # The rest of replay b continues in the next window with a fresh header.
    assert windows[1].segments[0].replay_id == "b"
    assert windows[1].segments[0].start_timestep == packed.segments[1].end_timestep
    assert windows[-1].tokens[-1].name == "[END]"
    covered = [(s.replay_id, t) for w in windows for s in w.segments
               for t in range(s.start_timestep, s.end_timestep)]
    assert covered == [("a", 0)] + [("b", t) for t in range(3)]
    assert all(len(w.tokens) <= 30 for w in windows)


def test_packing_disabled_gives_one_replay_per_window(setup):
    *_, vocabulary = setup
    step = blocks(setup, [pd.Series(dtype=object)])[0]
    windows = windows_of(vocabulary, replay([step], "a"), replay([step], "b"), budget=100, pack=False)
    assert [[s.replay_id for s in w.segments] for w in windows] == [["a"], ["b"]]
    assert all(w.tokens[-1].name == "[END]" for w in windows)


def test_packed_tail_too_small_closes_window(setup):
    *_, vocabulary = setup
    step = blocks(setup, [pd.Series(dtype=object)])[0]
    # 14 used; replay b's single segment needs 14 more, so a 27-token budget cannot pack it.
    windows = windows_of(vocabulary, replay([step], "a"), replay([step], "b"), budget=27)
    assert [[s.replay_id for s in w.segments] for w in windows] == [["a"], ["b"]]
    assert windows[0].next_timestep_tokens == 4


def test_race_header_and_real_metadata(setup):
    *_, vocabulary = setup
    header = build_replay_header(vocabulary, perspective="p2", races=RACES)
    assert [t.name for t in header] == ["[BOS]", "[SELF]", "[ZERG]", "[ENEMY]", "[TERRAN]"]
    assert [t.owner for t in header] == [None, "p2", "p2", "p1", "p1"]
    with pytest.raises(ValueError, match="race tokens"):
        build_replay_header(vocabulary, perspective="p1", races={"p1": "[TERRAN]", "p2": "[SCV]"})
    assert set(RACE_TOKENS.values()) <= set(vocabulary.names)
    assert race_token_name("Random") == "[RANDOM]"
    with pytest.raises(ValueError, match="unknown race"):
        race_token_name("terran")
    parquet = ROOT.parent / "data/quickstart/parquet/match_4745722_game_state.parquet"
    if parquet.exists():
        assert resolve_replay_race(parquet, "p2") == "[PROTOSS]"


def test_content_order_is_ascending_token_id_per_player(setup):
    rows = pd.read_parquet(ROOT / "tests/fixtures/match_4745722_game_state.parquet").iloc[:200:20]
    for block in blocks(setup, [row for _, row in rows.iterrows()]):
        for owner in ("p1", "p2"):
            ids = [t.token_id for t in block if t.role == "content" and t.owner == owner]
            assert ids == sorted(ids)


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
    # Import every shared module; none may pull in a deep-learning framework.
    code = ("import sys, pkgutil, importlib, thesis_shared\n"
            "for m in pkgutil.walk_packages(thesis_shared.__path__, 'thesis_shared.'):\n"
            "    importlib.import_module(m.name)\n"
            "bad = {'torch', 'tensorflow', 'keras', 'jax'} & set(sys.modules)\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("perspective,expected", [("p1", [4, 5]), ("p2", [5, 4])])
def test_footer_swaps_outcomes_and_keeps_all_positions_scored(setup, perspective, expected):
    *_, vocabulary = setup
    timesteps = blocks(setup, [pd.Series(dtype=object)], perspective)
    window = windows_of(vocabulary, replay(timesteps, perspective=perspective), budget=14)[0]
    footer = window.tokens[-5:-1]
    assert [t.token_id for t in footer if t.role == "outcome"] == expected
    assert footer[0].owner == perspective
    assert all(window.diffusion_loss_mask[-5:-1]) and all(window.ar_loss_mask[-5:-1])
    assert not any(t.role == "outcome" for t in window.tokens[:-5])
