from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd
import torch

from thesis_ml.config import (
    load_config,
)
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import (
    CLASS_ENEMY_FUTURE,
    PRETRAIN_CLASS_ID_TO_NAME,
    SC2DiffusionDataset,
    _artifact_delimiter,
    _artifact_timestep_records,
    _build_debut_target,
)
from thesis_ml.data.feature_stats import (
    CONTINUOUS_FEATURE_NAMES,
    FeatureStatisticsError,
    compute_feature_statistics,
    load_feature_statistics,
    write_feature_statistics,
)
from thesis_ml.data.features import continuous_feature_is_valid
from thesis_ml.data.windowing import (
    MANIFEST_VERSION,
    _artifact_is_current,
    TokenizedReplay,
    load_window_manifest,
    manifest_config_stamp,
    preprocess_replays,
    read_manifest_metadata,
    replay_source_stamp,
    validate_manifest_budgets,
    validate_manifest_integrity,
    vocabulary_stamp,
)
from thesis_ml.inference.timing import attach_absolute_times
from thesis_ml.model.embedding import build_input_features
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.serialize import serialize_snapshot
from thesis_ml.train.train import run_smoke_train
from thesis_ml.vocab.content_vocab import ContentVocabulary, build_content_vocabulary, load_content_vocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID, WIN_ID


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "match_4745722_game_state.parquet"


def _prepared(tmp_path: Path, *, debut_mode: bool = False):
    base = load_config(ROOT / "config" / "default.yaml")
    config = replace(
        base,
        data=replace(
            base.data,
            input_budget_tokens=512,
            canvas_budget_tokens=512,
            canvas_recon_fraction=0.5,
            debut_mode=debut_mode,
            tokenized_replay_dir=str(tmp_path / "tokenized"),
            window_manifest_path=str(tmp_path / "manifest.jsonl"),
        ),
        pipeline=replace(base.pipeline, num_workers=0),
    )
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    preprocess_replays([FIXTURE], config, vocabulary)
    entries = load_window_manifest(config.data.window_manifest_path, config=config)
    return config, vocabulary, entries


def test_feature_statistics_are_deterministic_frozen_and_strict(tmp_path: Path) -> None:
    config, _vocabulary, entries = _prepared(tmp_path)
    artifact_paths = {entry.artifact_path for entry in entries}
    statistics = compute_feature_statistics(
        artifact_paths,
        source_replay_ids=[FIXTURE.name],
    )
    assert statistics.feature_names == CONTINUOUS_FEATURE_NAMES
    assert all(count > 0 for count in statistics.counts)
    assert all(std > 0 for std in statistics.stds)
    assert statistics.counts[CONTINUOUS_FEATURE_NAMES.index("map_x")] > statistics.counts[
        CONTINUOUS_FEATURE_NAMES.index("energy")
    ]
    for feature_name in statistics.zero_variance_features:
        assert statistics.stds[statistics.feature_names.index(feature_name)] == 1.0

    path = Path(config.data.feature_statistics_path)
    path = tmp_path / path.name
    write_feature_statistics(statistics, path)
    first_bytes = path.read_bytes()
    write_feature_statistics(statistics, path)
    assert path.read_bytes() == first_bytes
    loaded = load_feature_statistics(
        path,
        expected_identity=statistics.identity,
        expected_source_replay_ids=[FIXTURE.name],
    )
    assert loaded == statistics

    with pytest.raises(FeatureStatisticsError, match="training split"):
        load_feature_statistics(path, expected_source_replay_ids=["heldout.parquet"])
    with pytest.raises(FeatureStatisticsError, match="missing"):
        load_feature_statistics(tmp_path / "absent.json")

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["feature_names"] = "not-an-array"
    payload = {key: value for key, value in malformed.items() if key != "identity"}
    malformed["identity"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    malformed_path = tmp_path / "malformed-statistics.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(FeatureStatisticsError, match="malformed"):
        load_feature_statistics(malformed_path)


def test_artifact_preserves_wide_instance_ids_and_fraction_features(tmp_path: Path) -> None:
    source = tmp_path / "wide-instance.parquet"
    pd.DataFrame(
        {
            "game_loop": [0],
            "timestamp_seconds": [0.0],
            "p1_chito_zergling_1000_pos_(X,Y,Z)": ["(1.25, 2.5, 3.75)"],
            "p1_chito_zergling_1000_health": ["6.0/45.0"],
            "p1_chito_zergling_1000_facing": ["1.5707963267948966"],
            "p1_chito_zergling_1000_ideal_harvesters": ["16"],
            "p1_chito_zergling_1000_cloak": ["4"],
            "p1_chito_zergling_1000_buff_ids": ["[7, 27]"],
            "p1_chito_scv_1001_pos_(X,Y,Z)": ["destroyed"],
            "p1_chito_scv_1001_health": ["destroyed"],
        }
    ).to_parquet(source)
    base = load_config(ROOT / "config" / "default.yaml")
    config = replace(
        base,
        data=replace(
            base.data,
            tokenized_replay_dir=str(tmp_path / "tokenized"),
            window_manifest_path=str(tmp_path / "manifest.jsonl"),
        ),
        pipeline=replace(base.pipeline, num_workers=0),
    )
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")

    preprocess_replays([source], config, vocabulary)
    entries = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=[source],
    )
    replay = TokenizedReplay(entries[0].artifact_path)
    health_index = CONTINUOUS_FEATURE_NAMES.index("health")
    facing_sin_index = CONTINUOUS_FEATURE_NAMES.index("facing_sin")
    facing_cos_index = CONTINUOUS_FEATURE_NAMES.index("facing_cos")

    assert replay.token_ids.tolist() == [vocabulary.token_id_for("zergling")]
    assert replay.features[0, :2].tolist() == pytest.approx([1.25, 2.5])
    assert float(replay.features[0, health_index]) == pytest.approx(6.0 / 45.0)
    assert float(replay.features[0, facing_sin_index]) == pytest.approx(1.0)
    assert float(replay.features[0, facing_cos_index]) == pytest.approx(0.0, abs=1e-7)
    assert continuous_feature_is_valid(int(replay.feature_validity[0]), health_index)
    assert int(replay.cloak_states[0]) == 4
    assert replay.buff_counts.tolist() == [2]
    assert replay.buff_ids.tolist() == [7, 27]
    assert _artifact_is_current(Path(entries[0].artifact_path), source, vocabulary)
    reduced_vocabulary = ContentVocabulary(tokens=vocabulary.tokens[:-1])
    assert not _artifact_is_current(
        Path(entries[0].artifact_path),
        source,
        reduced_vocabulary,
    )

    original_source_stamp = read_manifest_metadata(config.data.window_manifest_path)[
        "replay_source_stamp"
    ]
    source.write_bytes(source.read_bytes() + b"source-drift")
    assert not _artifact_is_current(Path(entries[0].artifact_path), source, vocabulary)
    assert original_source_stamp != replay_source_stamp([source])


def test_artifact_and_rich_serializers_have_model_input_parity(tmp_path: Path) -> None:
    """Keep the optimized artifact path equivalent to the source oracle."""

    config, vocabulary, entries = _prepared(tmp_path)
    replay = TokenizedReplay(entries[0].artifact_path)
    frame = pd.read_parquet(FIXTURE).sort_values("game_loop").reset_index(drop=True)
    timesteps = sorted({0, len(frame) // 2, len(frame) - 1})

    for perspective_player in ("p1", "p2"):
        for timestep in timesteps:
            rich_records = serialize_snapshot(
                frame.iloc[timestep],
                config,
                vocabulary,
                perspective_player=perspective_player,
            )
            artifact_records = [
                record
                for _owner_code, record in _artifact_timestep_records(
                    replay,
                    timestep,
                    vocabulary,
                    perspective_player,
                )
            ]
            artifact_records.append(_artifact_delimiter(replay, timestep))

            def model_semantics(records):
                return [
                    (
                        record.token_id,
                        record.token_name,
                        record.token_kind,
                        record.owner,
                        record.allegiance,
                        record.game_loop,
                        record.timestamp_seconds,
                    )
                    for record in records
                ]

            assert model_semantics(artifact_records) == model_semantics(rich_records)

            rich_features = build_input_features([rich_records], len(rich_records))
            artifact_features = build_input_features([artifact_records], len(artifact_records))
            assert torch.equal(
                artifact_features.continuous_values,
                rich_features.continuous_values,
            )
            assert torch.equal(
                artifact_features.continuous_validity,
                rich_features.continuous_validity,
            )
            assert torch.equal(
                artifact_features.categorical_values,
                rich_features.categorical_values,
            )
            assert torch.equal(
                artifact_features.allegiance_values,
                rich_features.allegiance_values,
            )
            assert torch.equal(artifact_features.feature_mask, rich_features.feature_mask)


def test_debut_windows_tile_inputs_by_input_budget_and_allow_overlapping_targets(
    tmp_path: Path,
) -> None:
    config, vocabulary, entries = _prepared(tmp_path, debut_mode=True)
    assert validate_manifest_budgets(entries, config) == []
    assert validate_manifest_integrity(entries) == []

    reconstruction_limit = int(
        config.data.canvas_recon_fraction * config.data.canvas_budget_tokens
    )
    assert any(
        entry.enemy_reconstruction_token_count > reconstruction_limit for entry in entries
    )

    for perspective in ("p1", "p2"):
        indexed = [
            (index, entry)
            for index, entry in enumerate(entries)
            if entry.perspective_player == perspective
        ]
        assert all(
            left.end_timestep == right.start_timestep
            for (_, left), (_, right) in zip(indexed, indexed[1:])
        )

        saw_overlapping_targets = False
        for (left_index, left), (right_index, right) in zip(indexed, indexed[1:]):
            replay = TokenizedReplay(left.artifact_path)
            enemy_player = "p2" if perspective == "p1" else "p1"
            left_target = _build_debut_target(
                replay,
                left,
                vocabulary,
                enemy_player,
                fogged_counts={},
                budget=config.data.canvas_budget_tokens,
                outcome_id=WIN_ID,
            )
            right_target = _build_debut_target(
                replay,
                right,
                vocabulary,
                enemy_player,
                fogged_counts={},
                budget=config.data.canvas_budget_tokens,
                outcome_id=WIN_ID,
            )
            left_timesteps = {
                left.start_timestep + int(item["timestep_index"])
                for item in left_target.metadata
                if item.get("timestep_index") is not None
            }
            right_timesteps = {
                right.start_timestep + int(item["timestep_index"])
                for item in right_target.metadata
                if item.get("timestep_index") is not None
            }
            saw_overlapping_targets = saw_overlapping_targets or bool(
                left_timesteps & right_timesteps
            )
        assert saw_overlapping_targets


def test_manifest_obeys_budgets_and_tiles_single_replays_on_boundaries(
    tmp_path: Path,
    capsys,
) -> None:
    config, vocabulary, entries = _prepared(tmp_path)
    output = capsys.readouterr().out
    assert "manifest_budget_compliance=PASS" in output
    assert "manifest_boundary_integrity=PASS" in output
    assert entries
    assert validate_manifest_budgets(entries, config) == []
    assert validate_manifest_integrity(entries) == []
    metadata = json.loads(Path(config.data.window_manifest_path).read_text(encoding="utf-8").splitlines()[0])
    assert metadata["perspectives"] == ["p1", "p2"]
    assert {entry.perspective_player for entry in entries} == {"p1", "p2"}
    dataset = SC2DiffusionDataset(
        entries,
        config,
        vocabulary,
        seed=0,
        fog_rate_override=0.0,
    )
    assert dataset[0].input_records
    assert dataset[0].input_token_ids.numel() > 0
    for perspective in ("p1", "p2"):
        index = next(
            index
            for index, entry in enumerate(entries)
            if entry.perspective_player == perspective
        )
        example = dataset[index]
        owned_records = [record for record in example.input_records if record.owner is not None]
        assert {record.owner for record in owned_records} == {"p1", "p2"}
        assert all(
            record.allegiance == ("self" if record.owner == perspective else "enemy")
            for record in owned_records
        )
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

    nonterminal_future_label_counts: list[int] = []
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
            nonterminal_future_label_counts.append(
                int((example.class_labels == CLASS_ENEMY_FUTURE).sum())
            )
    assert nonterminal_future_label_counts
    assert any(count > 0 for count in nonterminal_future_label_counts)


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


def test_architecture_ablation_toggles_do_not_change_manifest_or_vocabulary_stamps() -> None:
    """The three architecture ablation toggles must NEVER force a manifest or
    tokenized-artifact rebuild.

    `manifest_config_stamp` and `vocabulary_stamp` key persisted artifacts
    (window manifests, tokenized replays) to the config/vocabulary that
    produced them. `frozen_input_kv`, `segment_embeddings`, and
    `per_segment_positions` change only what the MODEL does with an
    already-built manifest at forward time -- they must never leak into
    either stamp, or flipping one on would force re-preprocessing all 943
    replays for no data-side reason. `manifest_config_stamp` only reads
    `config.data.*` fields (see `src/thesis_ml/data/windowing.py`), and
    `vocabulary_stamp` does not take a config at all, so this test also
    guards against a future refactor accidentally threading `config.model`
    into either function.
    """

    base = load_config(ROOT / "config" / "default.yaml")
    vocabulary = build_content_vocabulary({"1": "marine", "2": "scv"})

    toggle_combinations = [
        {},
        {"frozen_input_kv": True},
        {"segment_embeddings": True},
        {"per_segment_positions": True},
        {"frozen_input_kv": True, "segment_embeddings": True, "per_segment_positions": True},
    ]
    manifest_stamps = set()
    vocabulary_stamps = set()
    for toggles in toggle_combinations:
        config = replace(base, model=replace(base.model, **toggles))
        manifest_stamps.add(manifest_config_stamp(config))
        vocabulary_stamps.add(vocabulary_stamp(vocabulary))

    assert len(manifest_stamps) == 1, manifest_stamps
    assert len(vocabulary_stamps) == 1, vocabulary_stamps


def test_short_smoke_logs_all_pretraining_classes_from_first_step(tmp_path: Path) -> None:
    first = run_smoke_train(max_steps=1, seed=41, checkpoint_dir=tmp_path / "smoke")[0]
    assert set(first.per_class) == set(PRETRAIN_CLASS_ID_TO_NAME.values())


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
    batch = collate_diffusion_examples(examples, debut_mode=False)
    alone = collate_diffusion_examples([short], debut_mode=False)

    assert batch.input_token_ids.shape[1] == max(example.input_token_ids.numel() for example in examples)
    assert batch.target_canvas.shape[1] == max(example.target_canvas.numel() for example in examples)
    assert int(batch.input_attention_mask[short_index].sum()) == short.input_token_ids.numel()
    assert int(batch.canvas_loss_mask[short_index].sum()) == short.target_canvas.numel() - 1
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
    assert parameter_count == 10_995_776
    assert 7_000_000 <= parameter_count <= 13_000_000


def test_small_training_v3_model_parameter_count() -> None:
    config = load_config(ROOT / "configs" / "smallTrainingTestV3.yaml")
    vocabulary = load_content_vocabulary(ROOT / "data" / "Token_Dictionary.json")
    model = SC2StrategyDiffusionModel(config, vocab_size=vocabulary.vocab_size)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 29_318_720


def _assert_metadata_has_only_complete_timesteps(metadata: list[dict[str, object]]) -> None:
    real = [item for item in metadata if item.get("timestep_index") is not None]
    assert real
    timestep_indexes = sorted({int(item["timestep_index"]) for item in real})
    assert timestep_indexes == list(range(timestep_indexes[-1] + 1))
    for timestep in timestep_indexes:
        records = [item for item in real if int(item["timestep_index"]) == timestep]
        assert records[-1]["token_id"] == DELIMITER_ID
        assert sum(item["token_id"] == DELIMITER_ID for item in records) == 1
