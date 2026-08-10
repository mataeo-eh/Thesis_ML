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
    # Both modes interleave [self][enemy] per timestep with exactly one
    # delimiter; the estimator reports the clean/zero-fog upper bound:
    # p1_content (7) + p2_content (4) + timesteps (3) + EOS (1) = 15.
    assert estimate.pretrain_input_tokens == 15
    assert estimate.finetune_input_tokens == 15
    # Output adds BOS + outcome, enemy content, delimiters, and END.
    assert estimate.p1_perspective_output_tokens == 10
    assert estimate.p2_perspective_output_tokens == 13


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

    # Schema 4 accounts for EOS and the BOS/outcome canvas prefix.
    assert report["schema_version"] == 4
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
    # Both input fields use p1 + p2 content + one delimiter + EOS = 4.
    assert report["statistics"]["pretrain_input_tokens"] == {
        "minimum": 4,
        "maximum": 4,
        "mean": 4,
        "median": 4.0,
        "mode": 4,
        "mode_frequency": 2,
        "all_modes": [4],
        "sample_count": 2,
    }
    # Fine-tuning: p1_content (1) + p2_content (1) + delimiter (1) + EOS (1).
    assert report["statistics"]["finetune_input_tokens"] == {
        "minimum": 4,
        "maximum": 4,
        "mean": 4,
        "median": 4.0,
        "mode": 4,
        "mode_frequency": 2,
        "all_modes": [4],
        "sample_count": 2,
    }
    assert report["statistics"]["output_tokens"]["minimum"] == 5
    assert report["statistics"]["output_tokens"]["maximum"] == 5
