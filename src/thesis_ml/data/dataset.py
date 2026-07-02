"""Dataset construction for masked-diffusion pretraining examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from thesis_ml.config import ProjectConfig
from thesis_ml.data.windowing import (
    ENTITY_CODE,
    P1_CODE,
    P2_CODE,
    TokenizedReplay,
    WindowManifestEntry,
)
from thesis_ml.model.embedding import STAT_KEYS
from thesis_ml.serialize import TokenRecord, serialize_snapshot
from thesis_ml.vocab.content_vocab import ContentVocabulary
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID

CLASS_ENEMY_OBSERVED = 0
CLASS_ENEMY_FOGGED = 1
CLASS_ENEMY_FUTURE = 2
CLASS_DELIMITER = 3
CLASS_END = 4
CLASS_PAD = 5

CLASS_LABELS: dict[str, int] = {
    "enemy-observed": CLASS_ENEMY_OBSERVED,
    "enemy-fogged": CLASS_ENEMY_FOGGED,
    "enemy-future": CLASS_ENEMY_FUTURE,
    "[DELIMITER]": CLASS_DELIMITER,
    "[END]": CLASS_END,
    "[PAD]": CLASS_PAD,
}


@dataclass(frozen=True)
class ReplayWindow:
    replay_path: Path
    start: int
    perspective_player: str


@dataclass(frozen=True)
class CanvasBuild:
    token_ids: list[int]
    class_labels: list[int]
    metadata: list[dict[str, Any]]
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class DatasetExample:
    input_records: list[TokenRecord]
    input_token_ids: torch.Tensor
    target_canvas: torch.Tensor
    class_labels: torch.Tensor
    terminated: bool
    truncated: bool
    canvas_metadata: list[dict[str, Any]]
    fogged_counts: dict[tuple[int, str], int]
    observed_counts: dict[tuple[int, str], int]
    window_start: int
    perspective_player: str
    replay_path: Path | None = None
    clean_input_token_ids: torch.Tensor | None = None
    window_end: int | None = None
    replay_id: str | None = None


class SC2DiffusionDataset(Dataset[DatasetExample]):
    """Lazy manifest-backed clamped-input and clean-canvas examples."""

    def __init__(
        self,
        windows: Sequence[WindowManifestEntry],
        config: ProjectConfig,
        vocabulary: ContentVocabulary,
        *,
        seed: int | None = None,
        fog_rate_override: float | None = None,
    ) -> None:
        self.windows = tuple(windows)
        self.config = config
        self.vocabulary = vocabulary
        self.seed = seed
        self.fog_rate_override = fog_rate_override
        self._artifact_path: str | None = None
        self._artifact: TokenizedReplay | None = None
        self._serve_counts: dict[int, int] = {}
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> DatasetExample:
        if index < 0:
            index += len(self.windows)
        window = self.windows[index]
        replay = self._replay(window.artifact_path)
        enemy_player = _enemy_player(window.perspective_player)
        rng = self._rng_for_index(index)
        fog_rate = self._sample_fog_rate(rng)

        input_records, clean_records, fogged_counts, observed_counts = _build_artifact_input(
            replay,
            window,
            self.vocabulary,
            fog_rate=fog_rate,
            rng=rng,
        )

        target = _build_artifact_target(
            replay,
            window,
            self.vocabulary,
            enemy_player,
            fogged_counts=fogged_counts,
            budget=self.config.data.canvas_budget_tokens,
        )

        return DatasetExample(
            input_records=input_records,
            input_token_ids=torch.tensor([record.token_id for record in input_records], dtype=torch.long),
            target_canvas=torch.tensor(target.token_ids, dtype=torch.long),
            class_labels=torch.tensor(target.class_labels, dtype=torch.long),
            terminated=target.terminated,
            truncated=target.truncated,
            canvas_metadata=target.metadata,
            fogged_counts=fogged_counts,
            observed_counts=observed_counts,
            window_start=window.start_timestep,
            perspective_player=window.perspective_player,
            replay_path=Path(window.replay_path),
            clean_input_token_ids=torch.tensor([record.token_id for record in clean_records], dtype=torch.long),
            window_end=window.end_timestep,
            replay_id=window.replay_id,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _replay(self, artifact_path: str) -> TokenizedReplay:
        if self._artifact is None or self._artifact_path != artifact_path:
            self._artifact = TokenizedReplay(artifact_path)
            self._artifact_path = artifact_path
        return self._artifact

    def _rng_for_index(self, index: int) -> np.random.Generator:
        serving = self._serve_counts.get(index, 0)
        self._serve_counts[index] = serving + 1
        worker = get_worker_info()
        worker_seed = int(worker.seed) if worker is not None else 0
        base = int(self.seed) if self.seed is not None else int(np.random.SeedSequence().entropy)
        return np.random.default_rng(
            np.random.SeedSequence([base, self._epoch, index, serving, worker_seed & 0xFFFFFFFF])
        )

    def _sample_fog_rate(self, rng: np.random.Generator) -> float:
        if self.fog_rate_override is not None:
            return self.fog_rate_override
        distribution = self.config.fog.rate_distribution
        if distribution.name != "uniform":
            raise ValueError(f"unsupported fog distribution: {distribution.name}")
        return float(rng.uniform(distribution.min, distribution.max))


def _build_artifact_input(
    replay: TokenizedReplay,
    window: WindowManifestEntry,
    vocabulary: ContentVocabulary,
    *,
    fog_rate: float,
    rng: np.random.Generator,
) -> tuple[list[TokenRecord], list[TokenRecord], dict[tuple[int, str], int], dict[tuple[int, str], int]]:
    self_code = P1_CODE if window.perspective_player == "p1" else P2_CODE
    enemy_code = P2_CODE if self_code == P1_CODE else P1_CODE
    self_block: list[TokenRecord] = []
    clean_enemy_block: list[TokenRecord] = []
    fogged_enemy_block: list[TokenRecord] = []
    fogged_counts: dict[tuple[int, str], int] = {}
    observed_counts: dict[tuple[int, str], int] = {}
    for relative_timestep, timestep in enumerate(range(window.start_timestep, window.end_timestep)):
        records = _artifact_timestep_records(replay, timestep, vocabulary, window.perspective_player)
        delimiter = _artifact_delimiter(replay, timestep)
        self_records = [record for code, record in records if code == self_code]
        enemy_records = [record for code, record in records if code == enemy_code]
        self_block.extend(self_records)
        self_block.append(delimiter)
        clean_enemy_block.extend(enemy_records)
        clean_enemy_block.append(delimiter)
        for record in enemy_records:
            key = (relative_timestep, record.token_name)
            if record.token_kind == "entity" and rng.random() < fog_rate:
                _increment(fogged_counts, key)
                continue
            _increment(observed_counts, key)
            fogged_enemy_block.append(record)
        fogged_enemy_block.append(delimiter)
    return (
        self_block + fogged_enemy_block,
        self_block + clean_enemy_block,
        fogged_counts,
        observed_counts,
    )


def _build_artifact_target(
    replay: TokenizedReplay,
    window: WindowManifestEntry,
    vocabulary: ContentVocabulary,
    enemy_player: str,
    *,
    fogged_counts: dict[tuple[int, str], int],
    budget: int,
) -> CanvasBuild:
    enemy_code = P1_CODE if enemy_player == "p1" else P2_CODE
    remaining_fogged = dict(fogged_counts)
    token_ids: list[int] = []
    class_labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    truncated = False
    reached_game_end = False
    for timestep in range(window.start_timestep, replay.timestep_count):
        relative_timestep = timestep - window.start_timestep
        records = [
            record
            for code, record in _artifact_timestep_records(
                replay, timestep, vocabulary, window.perspective_player
            )
            if code == enemy_code
        ]
        records.append(_artifact_delimiter(replay, timestep))
        is_final_game_timestep = timestep == replay.timestep_count - 1
        required = len(records) + (1 if is_final_game_timestep else 0)
        if len(token_ids) + required > budget:
            if relative_timestep < window.timestep_count:
                raise RuntimeError(
                    f"manifest reconstruction does not fit canvas: replay={window.replay_id} "
                    f"start={window.start_timestep} end={window.end_timestep}"
                )
            truncated = True
            break
        for record in records:
            token_ids.append(record.token_id)
            class_labels.append(
                _canvas_label(record, relative_timestep, window.timestep_count, remaining_fogged)
            )
            metadata.append(_canvas_metadata(record, relative_timestep))
        if is_final_game_timestep:
            reached_game_end = True
            break

    terminated = reached_game_end
    if terminated:
        token_ids.append(END_ID)
        class_labels.append(CLASS_END)
        metadata.append({"token_kind": "end", "timestep_index": None, "token_name": "[END]"})
    else:
        truncated = True
    while len(token_ids) < budget:
        token_ids.append(PAD_ID)
        class_labels.append(CLASS_PAD)
        metadata.append({"token_kind": "pad", "timestep_index": None, "token_name": "[PAD]"})
    return CanvasBuild(token_ids, class_labels, metadata, terminated, truncated)


def _artifact_timestep_records(
    replay: TokenizedReplay,
    timestep: int,
    vocabulary: ContentVocabulary,
    perspective_player: str,
) -> list[tuple[int, TokenRecord]]:
    result: list[tuple[int, TokenRecord]] = []
    token_slice = replay.token_slice(timestep)
    for position in range(token_slice.start, token_slice.stop):
        owner_code = int(replay.owners[position])
        owner = "p1" if owner_code == P1_CODE else "p2"
        token_id = int(replay.token_ids[position])
        token_name = vocabulary.token_name_for(token_id)
        values = replay.features[position]
        raw_attributes = {
            key: float(values[2 + stat_index])
            for stat_index, key in enumerate(STAT_KEYS)
            if float(values[2 + stat_index]) != 0.0
        }
        record = TokenRecord(
            token_id=token_id,
            token_name=token_name,
            token_kind="entity" if int(replay.kinds[position]) == ENTITY_CODE else "upgrade",
            owner=owner,
            allegiance="self" if owner == perspective_player else "enemy",
            game_loop=int(replay.game_loops[timestep]),
            timestamp_seconds=_optional_artifact_timestamp(replay.timestamps[timestep]),
            entity_type=token_name,
            raw_position=(float(values[0]), float(values[1]), 0.0),
            raw_attributes=raw_attributes,
        )
        result.append((owner_code, record))
    return result


def _artifact_delimiter(replay: TokenizedReplay, timestep: int) -> TokenRecord:
    return TokenRecord(
        token_id=DELIMITER_ID,
        token_name="[DELIMITER]",
        token_kind="delimiter",
        owner=None,
        allegiance=None,
        game_loop=int(replay.game_loops[timestep]),
        timestamp_seconds=_optional_artifact_timestamp(replay.timestamps[timestep]),
    )


def _optional_artifact_timestamp(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _read_replay_frame(path: Path) -> pd.DataFrame:
    """Read one replay parquet into a game-loop-ordered DataFrame.

    Module-level (not a closure) so it is picklable to DataLoader workers and
    can be passed as the BoundedFrameCache loader callback.
    """

    return pd.read_parquet(path).sort_values("game_loop").reset_index(drop=True)


def build_input_records(
    input_frame: pd.DataFrame,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    perspective_player: str,
    *,
    fog_rate: float,
    rng: np.random.Generator,
) -> tuple[list[TokenRecord], dict[tuple[int, str], int], dict[tuple[int, str], int]]:
    enemy_player = _enemy_player(perspective_player)
    self_block: list[TokenRecord] = []
    enemy_block: list[TokenRecord] = []
    fogged_counts: dict[tuple[int, str], int] = {}
    observed_counts: dict[tuple[int, str], int] = {}

    serialized = [
        serialize_snapshot(row, config, vocabulary, perspective_player=perspective_player)
        for _, row in input_frame.iterrows()
    ]

    for timestep_index, records in enumerate(serialized):
        self_records = _records_for_owner(records, perspective_player)
        enemy_records = _records_for_owner(records, enemy_player)
        delimiter = _delimiter(records)

        self_block.extend(self_records)
        self_block.append(delimiter)

        for record in enemy_records:
            if record.token_kind == "entity" and rng.random() < fog_rate:
                _increment(fogged_counts, (timestep_index, record.token_name))
                continue
            _increment(observed_counts, (timestep_index, record.token_name))
            enemy_block.append(record)
        enemy_block.append(delimiter)

    return self_block + enemy_block, fogged_counts, observed_counts


def build_target_canvas(
    target_frame: pd.DataFrame,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    enemy_player: str,
    *,
    input_timestep_count: int,
    fogged_counts: dict[tuple[int, str], int],
) -> CanvasBuild:
    budget = config.data.canvas_budget_tokens
    remaining_fogged_counts = dict(fogged_counts)
    token_ids: list[int] = []
    class_labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    truncated = False
    terminated = False
    rows = list(target_frame.iterrows())

    for timestep_index, (_, row) in enumerate(rows):
        records = serialize_snapshot(row, config, vocabulary, perspective_player=_enemy_player(enemy_player))
        enemy_records = _records_for_owner(records, enemy_player)
        timestep_records = enemy_records + [_delimiter(records)]
        is_final_game_timestep = timestep_index == len(rows) - 1
        required = len(timestep_records) + (1 if is_final_game_timestep else 0)
        if len(token_ids) + required > budget:
            truncated = True
            break
        for record in timestep_records:
            label = _canvas_label(record, timestep_index, input_timestep_count, remaining_fogged_counts)
            token_ids.append(record.token_id)
            class_labels.append(label)
            metadata.append(_canvas_metadata(record, timestep_index))
        if is_final_game_timestep:
            terminated = True
            break

    if terminated:
        token_ids.append(END_ID)
        class_labels.append(CLASS_END)
        metadata.append({"token_kind": "end", "timestep_index": None, "token_name": "[END]"})
    while len(token_ids) < budget:
        token_ids.append(PAD_ID)
        class_labels.append(CLASS_PAD)
        metadata.append({"token_kind": "pad", "timestep_index": None, "token_name": "[PAD]"})

    return CanvasBuild(
        token_ids=token_ids,
        class_labels=class_labels,
        metadata=metadata,
        terminated=terminated,
        truncated=truncated,
    )
def _canvas_label(
    record: TokenRecord,
    timestep_index: int,
    input_timestep_count: int,
    fogged_counts: dict[tuple[int, str], int],
) -> int:
    if record.token_id == DELIMITER_ID:
        return CLASS_DELIMITER
    if timestep_index >= input_timestep_count:
        return CLASS_ENEMY_FUTURE
    key = (timestep_index, record.token_name)
    fogged = fogged_counts.get(key, 0)
    if fogged > 0:
        fogged_counts[key] = fogged - 1
        return CLASS_ENEMY_FOGGED
    return CLASS_ENEMY_OBSERVED


def _canvas_metadata(record: TokenRecord, timestep_index: int) -> dict[str, Any]:
    return {
        "token_id": record.token_id,
        "token_name": record.token_name,
        "token_kind": record.token_kind,
        "timestep_index": timestep_index,
        "owner": record.owner,
        "instance_id": record.instance_id,
        "game_loop": record.game_loop,
    }


def _records_for_owner(records: Iterable[TokenRecord], owner: str) -> list[TokenRecord]:
    return [record for record in records if record.owner == owner]


def _delimiter(records: Sequence[TokenRecord]) -> TokenRecord:
    delimiter = records[-1]
    if delimiter.token_id != DELIMITER_ID:
        raise ValueError("serialized snapshot must end with [DELIMITER]")
    return delimiter


def _enemy_player(perspective_player: str) -> str:
    if perspective_player == "p1":
        return "p2"
    if perspective_player == "p2":
        return "p1"
    raise ValueError("perspective_player must be 'p1' or 'p2'")


def _increment(counts: dict[tuple[int, str], int], key: tuple[int, str]) -> None:
    counts[key] = counts.get(key, 0) + 1
