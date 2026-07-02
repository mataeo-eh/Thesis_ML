from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_ENEMY_FOGGED,
    CLASS_PAD,
    SC2DiffusionDataset,
    build_input_records,
    build_target_canvas,
)
from thesis_ml.data.windowing import load_window_manifest, preprocess_replays
from thesis_ml.serialize import serialize_snapshot
from thesis_ml.vocab.content_vocab import load_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "match_4745722_game_state.parquet"
CONFIG = ROOT / "config" / "default.yaml"
TOKEN_DICTIONARY = ROOT / "data" / "Token_Dictionary.json"


def _config(*, window: int = 8, budget: int = 256) -> ProjectConfig:
    config = load_config(CONFIG)
    return replace(
        config,
        data=replace(
            config.data,
            input_budget_tokens=max(64, window * 256),
            canvas_budget_tokens=budget,
        ),
    )


def _vocab():
    return load_content_vocabulary(TOKEN_DICTIONARY)


def _frame() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE).sort_values("game_loop").reset_index(drop=True)


def _enemy_counts(frame: pd.DataFrame, config: ProjectConfig, perspective_player: str) -> dict[tuple[int, str], int]:
    enemy = "p2" if perspective_player == "p1" else "p1"
    vocab = _vocab()
    counts: dict[tuple[int, str], int] = {}
    for timestep, (_, row) in enumerate(frame.iterrows()):
        records = serialize_snapshot(row, config, vocab, perspective_player=perspective_player)
        for record in records:
            if record.owner != enemy:
                continue
            key = (timestep, record.token_name)
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_input_target_asymmetry_and_zero_fog_degenerate_case() -> None:
    frame = _frame().iloc[:8]
    config = _config(window=8, budget=2048)
    vocab = _vocab()
    full_counts = _enemy_counts(frame, config, "p1")

    _, _, observed_counts = build_input_records(
        frame,
        config,
        vocab,
        "p1",
        fog_rate=0.0,
        rng=np.random.default_rng(7),
    )
    assert observed_counts == full_counts

    _, _, fogged_observed_counts = build_input_records(
        frame,
        config,
        vocab,
        "p1",
        fog_rate=0.5,
        rng=np.random.default_rng(7),
    )
    for key, count in fogged_observed_counts.items():
        assert count <= full_counts[key]


def test_canvas_grammar_exact_budget_for_terminated_and_truncated_examples() -> None:
    frame = _frame()
    vocab = _vocab()

    terminated = build_target_canvas(
        frame.tail(2),
        _config(window=2, budget=10000),
        vocab,
        "p2",
        input_timestep_count=2,
        fogged_counts={},
    )
    assert terminated.terminated is True
    assert terminated.truncated is False
    assert len(terminated.token_ids) == 10000
    _assert_canvas_grammar(terminated.token_ids)

    truncated = build_target_canvas(
        frame.head(20),
        _config(window=8, budget=17),
        vocab,
        "p2",
        input_timestep_count=8,
        fogged_counts={},
    )
    assert truncated.truncated is True
    assert truncated.terminated is False
    assert len(truncated.token_ids) == 17
    assert END_ID not in truncated.token_ids
    assert PAD_ID in truncated.token_ids
    _assert_canvas_grammar(truncated.token_ids)


def test_class_label_coverage_and_partially_fogged_group_counts() -> None:
    frame = _frame().iloc[:8]
    config = _config(window=8, budget=2048)
    vocab = _vocab()

    fogged_counts = {}
    for seed in range(100):
        _, fogged_counts, _ = build_input_records(
            frame,
            config,
            vocab,
            "p1",
            fog_rate=0.5,
            rng=np.random.default_rng(seed),
        )
        full_counts = _enemy_counts(frame, config, "p1")
        partial = {
            key: omitted
            for key, omitted in fogged_counts.items()
            if 0 < omitted < full_counts.get(key, 0)
        }
        if partial:
            break
    else:
        raise AssertionError("expected at least one partially fogged repeated-token group")

    target = build_target_canvas(
        frame,
        config,
        vocab,
        "p2",
        input_timestep_count=len(frame),
        fogged_counts=fogged_counts,
    )

    assert len(target.class_labels) == len(target.token_ids)
    labels = set(target.class_labels)
    assert CLASS_ENEMY_FOGGED in labels
    assert CLASS_DELIMITER in labels
    assert CLASS_PAD in labels

    key, omitted = next(iter(partial.items()))
    fogged_positions = [
        index
        for index, metadata in enumerate(target.metadata)
        if metadata.get("timestep_index") == key[0]
        and metadata.get("token_name") == key[1]
        and target.class_labels[index] == CLASS_ENEMY_FOGGED
    ]
    assert len(fogged_positions) == omitted


def test_truncated_target_ends_at_boundary_and_pads_without_end() -> None:
    target = build_target_canvas(
        _frame().head(20),
        _config(window=8, budget=17),
        _vocab(),
        "p2",
        input_timestep_count=8,
        fogged_counts={},
    )
    assert target.truncated is True
    assert END_ID not in target.token_ids
    first_pad = target.token_ids.index(PAD_ID)
    assert target.token_ids[first_pad - 1] == DELIMITER_ID
    assert all(token == PAD_ID for token in target.token_ids[first_pad:])


def test_dataset_and_collate_determinism_under_seed(tmp_path: Path) -> None:
    base = _config(window=8, budget=256)
    config = replace(
        base,
        data=replace(
            base.data,
            tokenized_replay_dir=str(tmp_path / "artifacts"),
            window_manifest_path=str(tmp_path / "manifest.jsonl"),
        ),
    )
    vocab = _vocab()
    preprocess_replays([FIXTURE], config, vocab, perspectives=("p1",))
    windows = load_window_manifest(config.data.window_manifest_path, config=config)
    kwargs = dict(
        windows=windows,
        config=config,
        vocabulary=vocab,
        seed=123,
        fog_rate_override=0.5,
    )
    first = SC2DiffusionDataset(**kwargs)[0]
    second = SC2DiffusionDataset(**kwargs)[0]

    assert torch.equal(first.input_token_ids, second.input_token_ids)
    assert torch.equal(first.target_canvas, second.target_canvas)
    assert torch.equal(first.class_labels, second.class_labels)

    batch = collate_diffusion_examples([first, second])
    assert batch.input_token_ids.shape[0] == 2
    assert batch.target_canvas.shape[1] <= config.data.canvas_budget_tokens
    assert torch.equal(batch.input_lengths, torch.tensor([len(first.input_token_ids), len(second.input_token_ids)]))
    assert torch.equal(batch.canvas_loss_mask, batch.canvas_attention_mask)

    training_batch = collate_diffusion_examples([first, second], retain_metadata=False)
    assert training_batch.input_records == []
    assert training_batch.canvas_metadata == []
    assert training_batch.input_features.map_values.shape[:2] == batch.input_token_ids.shape


def _assert_canvas_grammar(token_ids: list[int]) -> None:
    if PAD_ID in token_ids:
        first_pad = token_ids.index(PAD_ID)
        assert first_pad > 0
        assert token_ids[first_pad - 1] in {END_ID, DELIMITER_ID}
        assert all(token == PAD_ID for token in token_ids[first_pad:])
    if END_ID in token_ids:
        end_index = token_ids.index(END_ID)
        assert end_index > 0 and token_ids[end_index - 1] == DELIMITER_ID
        assert all(token == PAD_ID for token in token_ids[end_index + 1 :])
