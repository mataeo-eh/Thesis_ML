from pathlib import Path

import pandas as pd

from scripts.estimate_context_window import build_report, estimate_replay


def test_estimate_replay_counts_unique_entities_upgrades_and_sequence_grammar(tmp_path: Path) -> None:
    parquet_path = tmp_path / "synthetic.parquet"
    pd.DataFrame(
        {
            "game_loop": [0, 1, 2],
            # No health column exercises the exact any-attribute fallback.
            "p1_bot_scv_001_pos_(X,Y,Z)": ["(1, 2, 3)", None, None],
            "p1_bot_scv_001_facing": [None, 1.5, None],
            "p1_bot_marine_002_health": [None, 45.0, 0.0],
            "p2_bot_probe_001_health": [20.0, 20.0, None],
            "p1_upgrades": [None, ["Stimpack"], ["Stimpack", "CombatShield"]],
            "p2_upgrades": ["[]", "['WarpGate']", "['WarpGate']"],
        }
    ).to_parquet(parquet_path, index=False)

    estimate = estimate_replay(parquet_path)

    # P1 entities: 2 + 2, upgrades: 0 + 1 + 2.
    assert estimate.p1_content_tokens == 7
    # P2 entities: 2, upgrades: 0 + 1 + 1.
    assert estimate.p2_content_tokens == 4
    # Pre-training's input is now LITERALLY ABSENT (no self/enemy blocks, no
    # fog, no delimiters) -- always 0, regardless of content-token counts.
    assert estimate.pretrain_input_tokens == 0
    # Fine-tuning's input interleaves [self][enemy] per timestep with exactly
    # ONE delimiter per timestep (not one delimiter per player per timestep):
    # p1_content (7) + p2_content (4) + timesteps (3) = 14.
    assert estimate.finetune_input_tokens == 14
    assert estimate.p1_perspective_output_tokens == 8
    assert estimate.p2_perspective_output_tokens == 11


def test_report_statistics_include_both_perspectives(tmp_path: Path) -> None:
    parquet_path = tmp_path / "minimal.parquet"
    pd.DataFrame(
        {
            "game_loop": [0],
            "p1_bot_scv_001_health": [45.0],
            "p2_bot_probe_001_health": [20.0],
            "p1_upgrades": [None],
            "p2_upgrades": [None],
        }
    ).to_parquet(parquet_path, index=False)

    report = build_report([estimate_replay(parquet_path)])

    # Report schema bumped 1 -> 2: pretrain_input_tokens / finetune_input_tokens
    # replace the old single input_tokens field (see ReplayTokenCounts).
    assert report["schema_version"] == 2
    assert report["dataset"]["perspective_samples"] == 2
    assert report["statistics"]["timesteps"] == {
        "minimum": 1,
        "maximum": 1,
        "mean": 1,
        "median": 1,
        "mode": 1,
        "mode_frequency": 1,
        "all_modes": [1],
        "sample_count": 1,
    }
    # Pre-training input is always 0 -- there is no input at all.
    assert report["statistics"]["pretrain_input_tokens"] == {
        "minimum": 0,
        "maximum": 0,
        "mean": 0,
        "median": 0.0,
        "mode": 0,
        "mode_frequency": 2,
        "all_modes": [0],
        "sample_count": 2,
    }
    # Fine-tuning: p1_content (1) + p2_content (1) + timesteps (1) = 3, one
    # delimiter per timestep (not two, per the old grammar).
    assert report["statistics"]["finetune_input_tokens"] == {
        "minimum": 3,
        "maximum": 3,
        "mean": 3,
        "median": 3.0,
        "mode": 3,
        "mode_frequency": 2,
        "all_modes": [3],
        "sample_count": 2,
    }
    assert report["statistics"]["output_tokens"]["minimum"] == 3
    assert report["statistics"]["output_tokens"]["maximum"] == 3
