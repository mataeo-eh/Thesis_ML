from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from thesis_ml.config import load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import CLASS_ENEMY_FUTURE, SC2DiffusionDataset
from thesis_ml.data.windowing import (
    MANIFEST_VERSION,
    TokenizedReplay,
    load_window_manifest,
    manifest_config_stamp,
    preprocess_replays,
    validate_manifest_budgets,
    validate_manifest_integrity,
)
from thesis_ml.inference.timing import attach_absolute_times
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.model.loss import CLASS_ID_TO_NAME
from thesis_ml.train.train import run_smoke_train
from thesis_ml.vocab.content_vocab import load_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "match_4745722_game_state.parquet"


def _prepared(tmp_path: Path):
    base = load_config(ROOT / "config" / "default.yaml")
    config = replace(
        base,
        data=replace(
            base.data,
            input_budget_tokens=512,
            canvas_budget_tokens=512,
            canvas_recon_fraction=0.5,
            tokenized_replay_dir=str(tmp_path / "tokenized"),
            window_manifest_path=str(tmp_path / "manifest.jsonl"),
        ),
        pipeline=replace(base.pipeline, num_workers=0),
    )
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    preprocess_replays([FIXTURE], config, vocabulary)
    entries = load_window_manifest(config.data.window_manifest_path, config=config)
    return config, vocabulary, entries


def test_manifest_obeys_budgets_and_tiles_single_replays_on_boundaries(
    tmp_path: Path,
    capsys,
) -> None:
    config, _, entries = _prepared(tmp_path)
    output = capsys.readouterr().out
    assert "manifest_budget_compliance=PASS" in output
    assert "manifest_boundary_integrity=PASS" in output
    assert entries
    assert validate_manifest_budgets(entries, config) == []
    assert validate_manifest_integrity(entries) == []
    for perspective in ("p1", "p2"):
        windows = [entry for entry in entries if entry.perspective_player == perspective]
        assert windows[0].start_timestep == 0
        assert windows[-1].end_timestep == windows[-1].replay_timestep_count
        assert all(left.end_timestep == right.start_timestep for left, right in zip(windows, windows[1:]))
        assert len({entry.replay_id for entry in windows}) == 1
        assert all(entry.start_timestep < entry.end_timestep for entry in windows)
        assert all(
            entry.enemy_reconstruction_token_count
            <= config.data.canvas_recon_fraction * config.data.canvas_budget_tokens
            for entry in windows
        )


def test_every_nonterminal_window_has_future_headroom_and_future_labels(tmp_path: Path) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    dataset = SC2DiffusionDataset(entries, config, vocabulary, seed=29, fog_rate_override=0.5)
    minimum_headroom = int((1.0 - config.data.canvas_recon_fraction) * config.data.canvas_budget_tokens)

    for index, entry in enumerate(entries):
        example = dataset[index]
        assert config.data.canvas_budget_tokens - entry.enemy_reconstruction_token_count >= minimum_headroom
        reconstruction_metadata = [
            item
            for item in example.canvas_metadata
            if item.get("timestep_index") is not None
            and int(item["timestep_index"]) < entry.timestep_count
        ]
        assert len(reconstruction_metadata) == entry.enemy_reconstruction_token_count
        if not entry.reaches_replay_end:
            assert int((example.class_labels == CLASS_ENEMY_FUTURE).sum()) > 0


def test_midgame_canvas_contains_no_pre_window_history(tmp_path: Path) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    entry_index = next(index for index, entry in enumerate(entries) if entry.start_timestep > 0)
    entry = entries[entry_index]
    replay = TokenizedReplay(entry.artifact_path)
    example = SC2DiffusionDataset(entries, config, vocabulary, seed=31)[entry_index]
    first_allowed_game_loop = int(replay.game_loops[entry.start_timestep])
    real_metadata = [
        item for item in example.canvas_metadata if item.get("timestep_index") is not None
    ]

    assert real_metadata
    assert min(int(item["game_loop"]) for item in real_metadata) >= first_allowed_game_loop
    assert min(int(item["timestep_index"]) for item in real_metadata) == 0


def test_targets_use_whole_timestep_grammar_and_direct_pad_on_truncation(tmp_path: Path) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    dataset = SC2DiffusionDataset(entries, config, vocabulary, seed=37, fog_rate_override=0.5)
    saw_truncated = False
    saw_terminated = False
    saw_direct_pad_truncation = False
    for index in range(len(dataset)):
        example = dataset[index]
        tokens = example.target_canvas.tolist()
        assert len(tokens) == config.data.canvas_budget_tokens
        first_pad = tokens.index(PAD_ID) if PAD_ID in tokens else len(tokens)
        assert all(token == PAD_ID for token in tokens[first_pad:])
        if example.terminated:
            saw_terminated = True
            assert END_ID in tokens
            assert tokens[first_pad - 1] == END_ID
        else:
            saw_truncated = True
            assert END_ID not in tokens
            assert tokens[first_pad - 1] == DELIMITER_ID
            saw_direct_pad_truncation = saw_direct_pad_truncation or first_pad < len(tokens)
        _assert_metadata_has_only_complete_timesteps(example.canvas_metadata)
    assert saw_truncated and saw_terminated and saw_direct_pad_truncation


def test_stale_manifest_version_and_config_stamp_are_refused(tmp_path: Path) -> None:
    config, _, _ = _prepared(tmp_path)
    manifest = Path(config.data.window_manifest_path)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    assert metadata["config_stamp"] == manifest_config_stamp(config)
    metadata["manifest_version"] = MANIFEST_VERSION - 1
    manifest.write_text("\n".join([json.dumps(metadata), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale window manifest version"):
        load_window_manifest(manifest, config=config)

    metadata["manifest_version"] = MANIFEST_VERSION
    metadata["config_stamp"] = "not-the-current-config"
    manifest.write_text("\n".join([json.dumps(metadata), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale window manifest config stamp"):
        load_window_manifest(manifest, config=config)


def test_short_smoke_logs_all_six_classes_from_first_step(tmp_path: Path) -> None:
    first = run_smoke_train(max_steps=1, seed=41, checkpoint_dir=tmp_path / "smoke")[0]
    assert set(first.per_class) == set(CLASS_ID_TO_NAME.values())


def test_fog_is_resampled_per_serving_while_clean_tokens_stay_fixed(tmp_path: Path) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    dataset = SC2DiffusionDataset(entries, config, vocabulary, seed=91)
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    assert torch.equal(first.clean_input_token_ids, second.clean_input_token_ids)
    assert not torch.equal(first.input_token_ids, second.input_token_ids)


def test_dynamic_padding_masks_loss_and_preserves_real_position_outputs(tmp_path: Path) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    dataset = SC2DiffusionDataset(entries, config, vocabulary, seed=17, fog_rate_override=0.5)
    examples = [dataset[0], dataset[-1]]
    short_index = min(range(2), key=lambda index: examples[index].input_token_ids.numel())
    short = examples[short_index]
    batch = collate_diffusion_examples(examples)
    alone = collate_diffusion_examples([short])

    assert batch.input_token_ids.shape[1] == max(example.input_token_ids.numel() for example in examples)
    assert batch.target_canvas.shape[1] == max(example.target_canvas.numel() for example in examples)
    assert int(batch.input_attention_mask[short_index].sum()) == short.input_token_ids.numel()
    assert int(batch.canvas_loss_mask[short_index].sum()) == short.target_canvas.numel()
    assert not batch.canvas_loss_mask[short_index, short.target_canvas.numel() :].any()

    small = replace(
        config,
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64, self_conditioning=False),
    )
    torch.manual_seed(3)
    model = SC2StrategyDiffusionModel(small, vocab_size=vocabulary.vocab_size).eval()
    with torch.no_grad():
        batched_output = model(
            input_token_ids=batch.input_token_ids,
            canvas_token_ids=batch.target_canvas,
            input_attention_mask=batch.input_attention_mask,
            canvas_attention_mask=batch.canvas_attention_mask,
            input_features=batch.input_features,
        ).logits[short_index]
        alone_output = model(
            input_token_ids=alone.input_token_ids,
            canvas_token_ids=alone.target_canvas,
            input_attention_mask=alone.input_attention_mask,
            canvas_attention_mask=alone.canvas_attention_mask,
            input_features=alone.input_features,
        ).logits[0]

    input_pad = batch.input_token_ids.shape[1] - alone.input_token_ids.shape[1]
    batch_real = torch.cat(
        [
            batched_output[input_pad : batch.input_token_ids.shape[1]],
            batched_output[
                batch.input_token_ids.shape[1] : batch.input_token_ids.shape[1] + short.target_canvas.numel()
            ],
        ]
    )
    alone_real = alone_output[: alone.input_token_ids.shape[1] + short.target_canvas.numel()]
    assert torch.allclose(batch_real, alone_real, atol=2e-5, rtol=2e-5)


def test_local_cadence_matches_timing_recovery() -> None:
    for profile in ("local_overfit.yaml", "local_overfit_v2.yaml", "local_full.yaml"):
        config = load_config(ROOT / "configs" / profile)
        assert config.data.sampling_interval_s == 1
        timed = attach_absolute_times(
            [{"marine": 1}, {"marine": 2}, {"marine": 3}],
            last_input_clock=50.0,
            sampling_interval_s=config.data.sampling_interval_s,
        )
        assert [item.timestamp_seconds for item in timed] == [50.0, 51.0, 52.0]


def test_local_model_parameter_count_is_near_ten_million() -> None:
    config = load_config(ROOT / "configs" / "local_full.yaml")
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    model = SC2StrategyDiffusionModel(config, vocab_size=vocabulary.vocab_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 7_000_000 <= parameter_count <= 13_000_000


def _assert_metadata_has_only_complete_timesteps(metadata: list[dict[str, object]]) -> None:
    real = [item for item in metadata if item.get("timestep_index") is not None]
    assert real
    timestep_indexes = sorted({int(item["timestep_index"]) for item in real})
    assert timestep_indexes == list(range(timestep_indexes[-1] + 1))
    for timestep in timestep_indexes:
        records = [item for item in real if int(item["timestep_index"]) == timestep]
        assert records[-1]["token_id"] == DELIMITER_ID
        assert sum(item["token_id"] == DELIMITER_ID for item in records) == 1
