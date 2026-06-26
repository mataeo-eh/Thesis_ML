"""Dataset construction for masked-diffusion pretraining examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from thesis_ml.config import ProjectConfig
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


class SC2DiffusionDataset(Dataset[DatasetExample]):
    """PyTorch dataset for clamped-input and clean-canvas examples.

    Window starts are sampled once during construction using the optional seed.
    Fog is sampled per item with a seed derived from the dataset seed and item index,
    making tests reproducible while leaving the unseeded training path random.
    Short games use all available rows from the sampled start.
    """

    def __init__(
        self,
        replay_paths: Sequence[str | Path],
        config: ProjectConfig,
        vocabulary: ContentVocabulary,
        *,
        seed: int | None = None,
        examples_per_replay: int = 1,
        perspectives: Sequence[str] = ("p1", "p2"),
        fog_rate_override: float | None = None,
    ) -> None:
        self.replay_paths = tuple(Path(path) for path in replay_paths)
        self.config = config
        self.vocabulary = vocabulary
        self.seed = seed
        self.fog_rate_override = fog_rate_override
        self._frames: dict[Path, pd.DataFrame] = {}

        rng = np.random.default_rng(seed)
        windows: list[ReplayWindow] = []
        for replay_path in self.replay_paths:
            row_count = len(pd.read_parquet(replay_path, columns=["game_loop"]))
            max_start = max(0, row_count - 1)
            for perspective in perspectives:
                for _ in range(examples_per_replay):
                    start = int(rng.integers(0, max_start + 1)) if max_start else 0
                    windows.append(ReplayWindow(replay_path, start, perspective))
        self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> DatasetExample:
        window = self.windows[index]
        frame = self._frame(window.replay_path)
        end = min(len(frame), window.start + self.config.data.input_window_timesteps)
        input_frame = frame.iloc[window.start:end]
        enemy_player = _enemy_player(window.perspective_player)
        rng = self._rng_for_index(index)
        fog_rate = self._sample_fog_rate(rng)

        input_records, fogged_counts, observed_counts = build_input_records(
            input_frame,
            self.config,
            self.vocabulary,
            window.perspective_player,
            fog_rate=fog_rate,
            rng=rng,
        )

        target = build_target_canvas(
            frame.iloc[window.start:],
            self.config,
            self.vocabulary,
            enemy_player,
            input_timestep_count=len(input_frame),
            fogged_counts=fogged_counts,
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
            window_start=window.start,
            perspective_player=window.perspective_player,
            replay_path=window.replay_path,
        )

    def _frame(self, path: Path) -> pd.DataFrame:
        if path not in self._frames:
            self._frames[path] = pd.read_parquet(path).sort_values("game_loop").reset_index(drop=True)
        return self._frames[path]

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + index)

    def _sample_fog_rate(self, rng: np.random.Generator) -> float:
        if self.fog_rate_override is not None:
            return self.fog_rate_override
        distribution = self.config.fog.rate_distribution
        if distribution.name != "uniform":
            raise ValueError(f"unsupported fog distribution: {distribution.name}")
        return float(rng.uniform(distribution.min, distribution.max))


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

    for timestep_index, (_, row) in enumerate(target_frame.iterrows()):
        records = serialize_snapshot(row, config, vocabulary, perspective_player=_enemy_player(enemy_player))
        enemy_records = _records_for_owner(records, enemy_player)
        timestep_records = enemy_records + [_delimiter(records)]

        for record in timestep_records:
            if len(token_ids) >= budget:
                truncated = True
                break
            label = _canvas_label(record, timestep_index, input_timestep_count, remaining_fogged_counts)
            token_ids.append(record.token_id)
            class_labels.append(label)
            metadata.append(_canvas_metadata(record, timestep_index))
        if len(token_ids) >= budget:
            truncated = True
            break

    if not truncated:
        if len(token_ids) < budget:
            token_ids.append(END_ID)
            class_labels.append(CLASS_END)
            metadata.append({"token_kind": "end", "timestep_index": None, "token_name": "[END]"})
            terminated = True
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


def drop_final_partial_timestep(token_ids: Sequence[int], *, truncated: bool) -> list[int]:
    """Return tokens suitable for eval by dropping a truncated final partial timestep."""

    if not truncated:
        return list(token_ids)
    last_delimiter = -1
    for index, token_id in enumerate(token_ids):
        if token_id == DELIMITER_ID:
            last_delimiter = index
    return list(token_ids[: last_delimiter + 1])


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
