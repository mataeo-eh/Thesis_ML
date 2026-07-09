from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thesis_ml.config import (
    ClassLossWeightsConfig,
    FogConfig,
    ProjectConfig,
    UniformDistributionConfig,
    load_config,
)
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import (
    CLASS_CONTENT,
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_PAD,
    CLASS_WINLOSS,
    SC2DiffusionDataset,
    _build_artifact_input,
    build_input_records,
    build_target_canvas,
)
from thesis_ml.data.windowing import (
    ENTITY_CODE,
    P1_CODE,
    P2_CODE,
    UPGRADE_CODE,
    WindowManifestEntry,
    load_window_manifest,
    preprocess_replays,
)
from thesis_ml.model.embedding import STAT_KEYS
from thesis_ml.model.model import SC2StrategyDiffusionModel
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


def _debut_config(*, window: int = 8, budget: int = 256) -> ProjectConfig:
    """Fine-tuning (debut_mode=True) config variant of `_config`.

    Fog and fog-resampling only exist in fine-tuning (see
    `SC2DiffusionDataset.__getitem__`), so any test exercising fog resampling,
    CLASS_ENEMY_FUTURE labels, or per-serving-varying input must build its
    dataset from a debut_mode=True config. `load_config` REQUIRES `fog` and
    `loss.class_loss_weights` to be populated once debut_mode is True (see
    `_validate_debut_mode_sections`), so both are supplied here with simple
    placeholder values -- their exact numbers are not under test.
    """

    base = _config(window=window, budget=budget)
    return replace(
        base,
        data=replace(base.data, debut_mode=True),
        fog=FogConfig(rate_distribution=UniformDistributionConfig(name="uniform", min=0.0, max=0.8)),
        loss=replace(
            base.loss,
            class_loss_weights=ClassLossWeightsConfig(
                enemy_observed_reconstruction=1.0,
                enemy_fogged_reconstruction=1.0,
                enemy_future_prediction=1.0,
                delimiter=1.0,
                end=1.0,
                pad=1.0,
                win_loss=1.0,
            ),
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


# ---------------------------------------------------------------------------
# LEGACY-path tests.
#
# `build_input_records` and `build_target_canvas` are LEGACY, test-only helpers:
# no production code calls them (the production pipeline serves examples through
# `SC2DiffusionDataset.__getitem__` -> `_build_artifact_input` /
# `_build_artifact_target` / `_build_debut_target` over memory-mapped artifacts).
# The orchestrator has flagged the helpers for cleanup; until they are removed,
# these tests document their CURRENT behavior (build_input_records now fogs
# enemy content tokens of every kind -- the entity-only guard was removed to
# stay consistent with the production builder -- but it keeps the OLD
# [all self][all enemy] grammar with one delimiter per player per timestep).
# Production-path coverage of fog application, input grammar, and canvas
# labeling lives in the SC2DiffusionDataset / _build_artifact_input tests
# further down and in tests/test_debut_target.py -- do NOT treat the tests in
# this block as covering the production pipeline.
# ---------------------------------------------------------------------------


def test_input_target_asymmetry_and_zero_fog_degenerate_case() -> None:
    # LEGACY-path test (see block comment above): exercises the dead
    # build_input_records helper, not the production input builder.
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
    # LEGACY-path test (see block comment above): exercises the dead
    # build_target_canvas helper. The production canvas grammar is covered by
    # tests/test_windowing.py::test_targets_use_whole_timestep_grammar_and_direct_pad_on_truncation
    # and tests/test_debut_target.py.
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
    # LEGACY-path test (see block comment above): exercises the dead
    # build_input_records/build_target_canvas pair. Production fog application
    # is covered by the _build_artifact_input / SC2DiffusionDataset tests
    # below; production fog-state labeling by tests/test_debut_target.py.
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
    # LEGACY-path test (see block comment above): exercises the dead
    # build_target_canvas helper; production truncation grammar is covered by
    # tests/test_windowing.py and tests/test_debut_target.py.
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
    # Fog resampling, non-empty input, and CLASS_ENEMY_FUTURE labels are all
    # fine-tuning-only behaviors (see `SC2DiffusionDataset.__getitem__`), so
    # this test's dataset must be built from a debut_mode=True config -- a
    # pre-training (default.yaml) config would produce an EMPTY input and
    # collapse every content label to CLASS_CONTENT, making the assertions
    # below meaningless.
    base = _debut_config(window=8, budget=256)
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

    batch = collate_diffusion_examples([first, second], debut_mode=True)
    assert batch.input_token_ids.shape[0] == 2
    assert batch.target_canvas.shape[1] <= config.data.canvas_budget_tokens
    assert torch.equal(batch.input_lengths, torch.tensor([len(first.input_token_ids), len(second.input_token_ids)]))
    assert torch.equal(batch.canvas_loss_mask, batch.canvas_attention_mask)
    future_mask = batch.class_labels == CLASS_ENEMY_FUTURE
    assert (batch.canvas_prediction_distances[future_mask] > 0).all()
    assert (batch.canvas_prediction_distances[~future_mask] == -1).all()

    training_batch = collate_diffusion_examples([first, second], debut_mode=True, retain_metadata=False)
    assert training_batch.input_records == []
    assert training_batch.canvas_metadata == []
    assert training_batch.input_features.map_values.shape[:2] == batch.input_token_ids.shape
    assert torch.equal(
        training_batch.canvas_prediction_distances,
        batch.canvas_prediction_distances,
    )


def _prepared_windows(tmp_path: Path, config: ProjectConfig):
    """Preprocess the fixture replay under `config` and load its windows."""

    prepared = replace(
        config,
        data=replace(
            config.data,
            tokenized_replay_dir=str(tmp_path / "artifacts"),
            window_manifest_path=str(tmp_path / "manifest.jsonl"),
        ),
    )
    vocab = _vocab()
    preprocess_replays([FIXTURE], prepared, vocab, perspectives=("p1",))
    windows = load_window_manifest(prepared.data.window_manifest_path, config=prepared)
    return prepared, vocab, windows


def test_pretraining_example_has_no_input_and_model_sequence_is_exactly_canvas(
    tmp_path: Path,
) -> None:
    """Pre-training input is LITERALLY ABSENT, end to end.

    Dataset level: zero-length `input_token_ids` AND `clean_input_token_ids`,
    no input records, no fog bookkeeping, and every content canvas position
    labeled `CLASS_CONTENT` (the collapsed pre-training class). Model level:
    the collated batch has a `[B, 0]` input segment, so the backbone sequence
    and its logits are EXACTLY the canvas length -- no separator/BOS token, no
    reserved input columns in the attention mask.
    """

    config, vocab, windows = _prepared_windows(tmp_path, _config(window=8, budget=256))
    assert config.data.debut_mode is False
    dataset = SC2DiffusionDataset(windows, config, vocab, seed=11)
    example = dataset[0]

    # Input is absent at the dataset level: no records, zero-length tensors,
    # and no fog was ever applied (nothing fogged, nothing observed).
    assert example.input_records == []
    assert example.input_token_ids.numel() == 0
    assert example.clean_input_token_ids is not None
    assert example.clean_input_token_ids.numel() == 0
    assert example.fogged_counts == {}
    assert example.observed_counts == {}

    # Every CONTENT canvas position is CLASS_CONTENT; only the structural
    # classes and the leading win/loss token appear besides it. The
    # fine-tuning-only ids (fogged=1 / future=2) never appear -- note
    # CLASS_CONTENT aliases id 0, so asserting per-kind below is what proves
    # the collapse (a bare "no id 1/2" check alone could not).
    labels = example.class_labels.tolist()
    assert set(labels) <= {CLASS_CONTENT, CLASS_DELIMITER, CLASS_END, CLASS_PAD, CLASS_WINLOSS}
    for metadata, label in zip(example.canvas_metadata, labels, strict=True):
        if metadata.get("token_kind") in {"entity", "upgrade"}:
            assert label == CLASS_CONTENT

    # Collated: the input segment has ZERO columns -- the attention mask has no
    # input columns for the model to attend to.
    batch = collate_diffusion_examples([example], debut_mode=False)
    assert batch.input_token_ids.shape == (1, 0)
    assert batch.input_attention_mask.shape == (1, 0)
    assert batch.input_features.team_ids.shape == (1, 0)

    # Model: hidden states / logits have sequence length EXACTLY equal to the
    # canvas length (the input contributes zero sequence positions).
    small = replace(
        config,
        model=replace(config.model, d_model=32, layers=1, heads=4, ffn=64, self_conditioning=False),
    )
    torch.manual_seed(5)
    model = SC2StrategyDiffusionModel(small, vocab_size=vocab.vocab_size).eval()
    with torch.no_grad():
        embeddings = model.embedding(
            batch.input_token_ids,
            batch.target_canvas,
            input_features=batch.input_features,
        )
        logits = model(
            input_token_ids=batch.input_token_ids,
            canvas_token_ids=batch.target_canvas,
            input_attention_mask=batch.input_attention_mask,
            canvas_attention_mask=batch.canvas_attention_mask,
            input_features=batch.input_features,
        ).logits
    assert embeddings.shape[1] == batch.target_canvas.shape[1]
    assert logits.shape[1] == batch.target_canvas.shape[1]


def test_finetune_input_interleaves_self_then_enemy_with_one_delimiter_per_timestep(
    tmp_path: Path,
) -> None:
    """Fine-tuning input grammar: per-timestep [self][enemy][one DELIMITER].

    Within each timestep block, ALL self records come before any enemy record,
    and each timestep is closed by exactly ONE delimiter -- so the total
    delimiter count equals the window's timestep count (the old grammar had
    one delimiter per PLAYER per timestep, i.e. twice as many).
    """

    config, vocab, windows = _prepared_windows(tmp_path, _debut_config(window=8, budget=256))
    dataset = SC2DiffusionDataset(windows, config, vocab, seed=13, fog_rate_override=0.0)
    window = windows[0]
    example = dataset[0]

    records = example.input_records
    delimiter_count = sum(record.token_kind == "delimiter" for record in records)
    assert delimiter_count == window.timestep_count

    # Split the flat record list into per-timestep blocks at each delimiter;
    # the input must END with a delimiter (each block is closed by one).
    assert records[-1].token_kind == "delimiter"
    blocks: list[list] = []
    current: list = []
    for record in records:
        if record.token_kind == "delimiter":
            blocks.append(current)
            current = []
        else:
            current.append(record)
    assert current == []  # nothing dangles after the final delimiter
    assert len(blocks) == window.timestep_count

    for block in blocks:
        allegiances = [record.allegiance for record in block]
        assert set(allegiances) <= {"self", "enemy"}
        # Interleaved order: once an enemy record appears in a timestep block,
        # no self record may follow it (self block strictly precedes enemy).
        if "enemy" in allegiances:
            first_enemy = allegiances.index("enemy")
            assert all(item == "enemy" for item in allegiances[first_enemy:])

    # With zero fog the fogged and clean variants are identical.
    assert torch.equal(example.input_token_ids, example.clean_input_token_ids)


class _FakeReplayWithEnemyUpgrade:
    """In-memory replay whose ENEMY (p2) owns entity AND upgrade tokens.

    Used to prove fog applies to enemy content tokens of EVERY token_kind --
    the fixture parquets carry no cumulative upgrade tokens, so this synthetic
    replay supplies them. Two timesteps, each with one p1 entity (self), one
    p2 entity, and one p2 upgrade.
    """

    def __init__(self) -> None:
        # per timestep: [p1 entity 100, p2 entity 101, p2 upgrade 102]
        token_ids = [100, 101, 102, 100, 101, 102]
        owners = [P1_CODE, P2_CODE, P2_CODE, P1_CODE, P2_CODE, P2_CODE]
        kinds = [
            ENTITY_CODE,
            ENTITY_CODE,
            UPGRADE_CODE,
            ENTITY_CODE,
            ENTITY_CODE,
            UPGRADE_CODE,
        ]
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.owners = np.asarray(owners, dtype=np.uint8)
        self.kinds = np.asarray(kinds, dtype=np.uint8)
        self.offsets = np.asarray([0, 3, 6], dtype=np.int64)
        self.features = np.zeros((len(token_ids), 2 + len(STAT_KEYS)), dtype=np.float32)
        self.game_loops = np.asarray([0, 1], dtype=np.int64)
        self.timestamps = np.asarray([0.0, 1.0], dtype=np.float64)

    def token_slice(self, timestep: int) -> slice:
        return slice(int(self.offsets[timestep]), int(self.offsets[timestep + 1]))

    @property
    def timestep_count(self) -> int:
        return len(self.offsets) - 1


class _FakeUpgradeVocabulary:
    """Names for the fake replay's three token ids."""

    _NAMES = {100: "scv", 101: "marine", 102: "stimpack"}

    def token_name_for(self, token_id: int) -> str:
        return self._NAMES[int(token_id)]


def test_full_fog_omits_every_enemy_content_token_including_upgrades(tmp_path: Path) -> None:
    """fog_rate=1.0 fogs ALL enemy content tokens, upgrades included.

    Fog applies uniformly to every enemy content token_kind now (the old
    entity-only guard is gone). Asserted at two levels:
      1. The production input builder `_build_artifact_input` (which is what
         `SC2DiffusionDataset.__getitem__` calls) over a synthetic replay
         whose enemy owns an UPGRADE token: at fog_rate=1.0 the fogged input
         keeps no enemy record of ANY kind -- the upgrade is omitted too --
         while the clean variant keeps all of them.
      2. The dataset-level `fog_rate_override=1.0` hook over the real fixture:
         the served input contains zero enemy records and observed_counts is
         empty (every enemy token was counted as fogged).
    """

    # --- 1. Production builder, synthetic replay with an enemy upgrade. -----
    window = WindowManifestEntry(
        replay_id="fake",
        replay_path="fake/parquet/match_fake_game_state.parquet",
        artifact_path="fake/artifact",
        perspective_player="p1",
        start_timestep=0,
        end_timestep=2,
        input_token_count=0,
        enemy_reconstruction_token_count=0,
        replay_timestep_count=2,
    )
    fogged_records, clean_records, fogged_counts, observed_counts = _build_artifact_input(
        _FakeReplayWithEnemyUpgrade(),
        window,
        _FakeUpgradeVocabulary(),
        fog_rate=1.0,
        rng=np.random.default_rng(3),
    )
    # The clean variant proves the enemy upgrade exists to be fogged.
    clean_enemy_kinds = {
        record.token_kind for record in clean_records if record.allegiance == "enemy"
    }
    assert clean_enemy_kinds == {"entity", "upgrade"}
    # The fogged variant has NO enemy records at all -- the upgrade included.
    assert all(record.allegiance != "enemy" for record in fogged_records)
    # Both enemy kinds were counted as fogged per timestep; nothing observed.
    assert fogged_counts == {
        (0, "marine"): 1,
        (0, "stimpack"): 1,
        (1, "marine"): 1,
        (1, "stimpack"): 1,
    }
    assert observed_counts == {}
    # Self records and the one-delimiter-per-timestep skeleton survive fog.
    assert sum(record.token_kind == "delimiter" for record in fogged_records) == 2
    assert sum(record.allegiance == "self" for record in fogged_records) == 2

    # --- 2. Dataset-level override hook over the real fixture. --------------
    config, vocab, windows = _prepared_windows(tmp_path, _debut_config(window=8, budget=256))
    dataset = SC2DiffusionDataset(windows, config, vocab, seed=19, fog_rate_override=1.0)
    example = dataset[0]
    assert all(record.allegiance != "enemy" for record in example.input_records)
    assert example.observed_counts == {}
    assert sum(example.fogged_counts.values()) > 0
    # The clean variant still carries the enemy tokens the fog omitted.
    assert example.clean_input_token_ids.numel() > example.input_token_ids.numel()


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
