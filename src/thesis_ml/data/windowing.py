"""Tokenized replay artifacts and timestep-aligned, budget-driven manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from thesis_ml.config import ProjectConfig
from thesis_ml.model.embedding import STAT_KEYS, _numeric_feature, _parse_position
from thesis_ml.serialize import parse_entity_columns, parse_upgrades
from thesis_ml.vocab.content_vocab import ContentVocabulary


TOKENIZED_ARTIFACT_VERSION = 1
MANIFEST_VERSION = 2
TARGET_SEMANTICS = "reconstruction-plus-whole-future-timesteps-v2"
P1_CODE = 1
P2_CODE = 2
ENTITY_CODE = 1
UPGRADE_CODE = 2


@dataclass(frozen=True)
class WindowManifestEntry:
    replay_id: str
    replay_path: str
    artifact_path: str
    perspective_player: str
    start_timestep: int
    end_timestep: int
    input_token_count: int
    enemy_reconstruction_token_count: int
    replay_timestep_count: int

    @property
    def timestep_count(self) -> int:
        return self.end_timestep - self.start_timestep

    @property
    def reaches_replay_end(self) -> bool:
        return self.end_timestep == self.replay_timestep_count


@dataclass(frozen=True)
class PreprocessingResult:
    manifest_path: Path
    replay_count: int
    window_count: int


class TokenizedReplay:
    """Memory-mapped arrays for one preprocessed replay."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.path = Path(artifact_path)
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r")
        self.token_ids = np.load(self.path / "token_ids.npy", mmap_mode="r")
        self.owners = np.load(self.path / "owners.npy", mmap_mode="r")
        self.kinds = np.load(self.path / "kinds.npy", mmap_mode="r")
        self.features = np.load(self.path / "features.npy", mmap_mode="r")
        self.game_loops = np.load(self.path / "game_loops.npy", mmap_mode="r")
        self.timestamps = np.load(self.path / "timestamps.npy", mmap_mode="r")
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))

    def token_slice(self, timestep: int) -> slice:
        return slice(int(self.offsets[timestep]), int(self.offsets[timestep + 1]))

    @property
    def timestep_count(self) -> int:
        return len(self.offsets) - 1


def preprocess_replays(
    replay_paths: Sequence[str | Path],
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
    *,
    perspectives: Sequence[str] = ("p1", "p2"),
    force: bool = False,
) -> PreprocessingResult:
    """Tokenize each replay once and persist a complete window manifest.

    Artifacts are one directory per replay containing memory-mappable NumPy
    arrays. Training workers map only the replay needed by the requested
    window; no worker loads the corpus into RAM.
    """

    artifact_root = Path(config.data.tokenized_replay_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.data.window_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(path), str(artifact_root), config, vocabulary, tuple(perspectives), force)
        for path in replay_paths
    ]
    workers = min(max(1, config.data_source.workers), len(tasks)) if tasks else 1
    if workers == 1:
        results = map(_preprocess_one_replay, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_preprocess_one_replay, tasks)

    entries: list[WindowManifestEntry] = []
    for replay_index, (replay_name, replay_entries) in enumerate(results, start=1):
        entries.extend(replay_entries)
        print(
            f"preprocess replay={replay_index}/{len(replay_paths)} name={replay_name} "
            f"windows={len(entries)}",
            flush=True,
        )
    if workers > 1:
        executor.shutdown()

    header = {
        "type": "metadata",
        "manifest_version": MANIFEST_VERSION,
        "tokenized_artifact_version": TOKENIZED_ARTIFACT_VERSION,
        "config_stamp": manifest_config_stamp(config),
        "target_semantics": TARGET_SEMANTICS,
        "sampling_interval_s": config.data.sampling_interval_s,
        "input_budget_tokens": config.data.input_budget_tokens,
        "canvas_budget_tokens": config.data.canvas_budget_tokens,
        "canvas_recon_fraction": config.data.canvas_recon_fraction,
        "replay_count": len(replay_paths),
        "window_count": len(entries),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for entry in entries:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
    budget_violations = validate_manifest_budgets(entries, config)
    boundary_violations = validate_manifest_integrity(entries)
    print(
        f"manifest_budget_compliance={'PASS' if not budget_violations else 'FAIL'} "
        f"windows={len(entries)} violations={len(budget_violations)}",
        flush=True,
    )
    print(
        f"manifest_boundary_integrity={'PASS' if not boundary_violations else 'FAIL'} "
        f"windows={len(entries)} violations={len(boundary_violations)}",
        flush=True,
    )
    if budget_violations or boundary_violations:
        examples = (budget_violations + boundary_violations)[:10]
        raise RuntimeError(f"window manifest validation failed: {examples}")
    return PreprocessingResult(manifest_path, len(replay_paths), len(entries))


def _preprocess_one_replay(
    task: tuple[str, str, ProjectConfig, ContentVocabulary, tuple[str, ...], bool],
) -> tuple[str, list[WindowManifestEntry]]:
    replay_path_value, artifact_root_value, config, vocabulary, perspectives, force = task
    replay_path = Path(replay_path_value)
    replay_id = _replay_id(replay_path)
    artifact_path = Path(artifact_root_value) / replay_id
    if force or not _artifact_is_current(artifact_path, replay_path):
        _write_tokenized_replay(replay_path, artifact_path, config, vocabulary)
    replay = TokenizedReplay(artifact_path)
    entries = build_replay_windows(
        replay_id,
        replay_path,
        replay,
        config,
        perspectives=perspectives,
    )
    return replay_path.name, entries


def build_replay_windows(
    replay_id: str,
    replay_path: Path,
    replay: TokenizedReplay,
    config: ProjectConfig,
    *,
    perspectives: Sequence[str] = ("p1", "p2"),
) -> list[WindowManifestEntry]:
    """Greedily tile a replay in whole timesteps under both token budgets."""

    recon_limit = _reconstruction_limit(config)
    counts = _timestep_owner_counts(replay)
    entries: list[WindowManifestEntry] = []
    for perspective in perspectives:
        enemy_code = P2_CODE if perspective == "p1" else P1_CODE
        start = 0
        while start < replay.timestep_count:
            end = start
            input_count = 0
            enemy_count = 0
            while end < replay.timestep_count:
                p1_count, p2_count = counts[end]
                candidate_input = input_count + int(p1_count + p2_count) + 2
                candidate_enemy = enemy_count + int(p2_count if enemy_code == P2_CODE else p1_count) + 1
                if (
                    candidate_input > config.data.input_budget_tokens
                    or candidate_enemy > recon_limit
                ):
                    break
                input_count = candidate_input
                enemy_count = candidate_enemy
                end += 1
            if end == start:
                p1_count, p2_count = counts[start]
                raise ValueError(
                    f"single timestep exceeds a configured window budget: replay={replay_path.name} "
                    f"perspective={perspective} timestep={start} input={int(p1_count+p2_count)+2} "
                    f"enemy={int(p2_count if enemy_code == P2_CODE else p1_count)+1}"
                )
            entries.append(
                WindowManifestEntry(
                    replay_id=replay_id,
                    replay_path=str(replay_path),
                    artifact_path=str(replay.path),
                    perspective_player=perspective,
                    start_timestep=start,
                    end_timestep=end,
                    input_token_count=input_count,
                    enemy_reconstruction_token_count=enemy_count,
                    replay_timestep_count=replay.timestep_count,
                )
            )
            start = end
    return entries


def load_window_manifest(
    path: str | Path,
    *,
    config: ProjectConfig,
    replay_paths: Iterable[str | Path] | None = None,
) -> tuple[WindowManifestEntry, ...]:
    metadata = read_manifest_metadata(path)
    expected_stamp = manifest_config_stamp(config)
    if metadata.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"stale window manifest version: expected {MANIFEST_VERSION}, "
            f"got {metadata.get('manifest_version')!r}; rebuild preprocessing"
        )
    if metadata.get("config_stamp") != expected_stamp:
        raise ValueError(
            "stale window manifest config stamp; budgets, reconstruction fraction, "
            "cadence, or target semantics changed; rebuild preprocessing"
        )
    allowed = None
    if replay_paths is not None:
        allowed = {_normalized_path(path_value) for path_value in replay_paths}
    entries: list[WindowManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("type") == "metadata":
                continue
            entry = WindowManifestEntry(**raw)
            if allowed is None or _normalized_path(entry.replay_path) in allowed:
                entries.append(entry)
    return tuple(entries)


def read_manifest_metadata(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        metadata = json.loads(handle.readline())
    if metadata.get("type") != "metadata":
        raise ValueError("window manifest is missing its metadata header")
    return metadata


def validate_manifest_budgets(entries: Iterable[WindowManifestEntry], config: ProjectConfig) -> list[str]:
    recon_limit = _reconstruction_limit(config)
    violations = []
    for entry in entries:
        if entry.input_token_count > config.data.input_budget_tokens:
            violations.append(f"{entry.replay_id}:{entry.start_timestep}:input")
        if entry.enemy_reconstruction_token_count > recon_limit:
            violations.append(f"{entry.replay_id}:{entry.start_timestep}:reconstruction")
        if not 0 <= entry.start_timestep < entry.end_timestep <= entry.replay_timestep_count:
            violations.append(f"{entry.replay_id}:{entry.start_timestep}:boundary")
    return violations


def manifest_config_stamp(config: ProjectConfig) -> str:
    stamp_fields = {
        "manifest_version": MANIFEST_VERSION,
        "target_semantics": TARGET_SEMANTICS,
        "sampling_interval_s": config.data.sampling_interval_s,
        "input_budget_tokens": config.data.input_budget_tokens,
        "canvas_budget_tokens": config.data.canvas_budget_tokens,
        "canvas_recon_fraction": config.data.canvas_recon_fraction,
        "within_type_tiebreak": config.data.within_type_tiebreak,
    }
    encoded = json.dumps(stamp_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reconstruction_limit(config: ProjectConfig) -> int:
    fraction = config.data.canvas_recon_fraction
    if not 0.0 < fraction <= 1.0:
        raise ValueError("data.canvas_recon_fraction must be in (0, 1]")
    return int(config.data.canvas_budget_tokens * fraction)


def validate_manifest_integrity(entries: Iterable[WindowManifestEntry]) -> list[str]:
    """Prove that every replay-perspective is tiled by contiguous whole timesteps."""

    groups: dict[tuple[str, str], list[WindowManifestEntry]] = {}
    for entry in entries:
        groups.setdefault((entry.replay_id, entry.perspective_player), []).append(entry)

    violations: list[str] = []
    for (replay_id, perspective), windows in groups.items():
        ordered = sorted(windows, key=lambda item: item.start_timestep)
        expected_start = 0
        replay_paths = {item.replay_path for item in ordered}
        artifact_paths = {item.artifact_path for item in ordered}
        replay_lengths = {item.replay_timestep_count for item in ordered}
        if len(replay_paths) != 1 or len(artifact_paths) != 1 or len(replay_lengths) != 1:
            violations.append(f"{replay_id}:{perspective}:mixed-replay")
            continue
        for window in ordered:
            if window.start_timestep != expected_start or window.end_timestep <= window.start_timestep:
                violations.append(
                    f"{replay_id}:{perspective}:expected-{expected_start}-got-{window.start_timestep}"
                )
            expected_start = window.end_timestep
        replay_length = next(iter(replay_lengths))
        if expected_start != replay_length:
            violations.append(f"{replay_id}:{perspective}:ends-{expected_start}-of-{replay_length}")
    return violations


def _write_tokenized_replay(
    replay_path: Path,
    artifact_path: Path,
    config: ProjectConfig,
    vocabulary: ContentVocabulary,
) -> None:
    frame = pd.read_parquet(replay_path).sort_values("game_loop").reset_index(drop=True)
    offsets = [0]
    token_ids: list[int] = []
    owners: list[int] = []
    kinds: list[int] = []
    features: list[list[float]] = []
    game_loops: list[int] = []
    timestamps: list[float] = []
    column_indexes = {name: index for index, name in enumerate(frame.columns)}
    entity_groups = parse_entity_columns(frame.columns)
    group_specs = sorted(
        (
            vocabulary.source_id_for(group.entity_type),
            int(group.instance_id),
            group.owner,
            vocabulary.token_id_for(group.entity_type),
            group,
        )
        for group in entity_groups
    )
    upgrade_indexes = {
        owner: column_indexes.get(f"{owner}_upgrades")
        for owner in ("p1", "p2")
    }
    rows = frame.itertuples(index=False, name=None)
    for row in rows:
        for _, _, owner, token_id, group in group_specs:
            raw = {
                attribute: row[column_indexes[column]]
                for attribute, column in group.attributes.items()
                if not pd.isna(row[column_indexes[column]])
            }
            if not raw:
                continue
            token_ids.append(token_id)
            owners.append(P1_CODE if owner == "p1" else P2_CODE)
            kinds.append(ENTITY_CODE)
            position = _parse_position(raw.get("pos_(X,Y,Z)")) or (0.0, 0.0, 0.0)
            features.append([position[0], position[1], *(_numeric_feature(raw.get(key)) for key in STAT_KEYS)])

        upgrades: list[tuple[int, str, int]] = []
        for owner, column_index in upgrade_indexes.items():
            if column_index is None:
                continue
            owner_code = P1_CODE if owner == "p1" else P2_CODE
            for upgrade in parse_upgrades(row[column_index]):
                upgrades.append((vocabulary.source_id_for(upgrade), upgrade, owner_code))
        for _, upgrade, owner_code in sorted(upgrades):
            token_ids.append(vocabulary.token_id_for(upgrade))
            owners.append(owner_code)
            kinds.append(UPGRADE_CODE)
            features.append([0.0] * (2 + len(STAT_KEYS)))
        offsets.append(len(token_ids))
        game_loops.append(int(row[column_indexes["game_loop"]]))
        timestamp_index = column_indexes.get("timestamp_seconds")
        timestamp = row[timestamp_index] if timestamp_index is not None else None
        timestamps.append(float(timestamp) if timestamp is not None and not pd.isna(timestamp) else float("nan"))

    artifact_path.mkdir(parents=True, exist_ok=True)
    np.save(artifact_path / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(artifact_path / "token_ids.npy", np.asarray(token_ids, dtype=np.int32))
    np.save(artifact_path / "owners.npy", np.asarray(owners, dtype=np.uint8))
    np.save(artifact_path / "kinds.npy", np.asarray(kinds, dtype=np.uint8))
    np.save(
        artifact_path / "features.npy",
        np.asarray(features, dtype=np.float32).reshape((-1, 2 + len(STAT_KEYS))),
    )
    np.save(artifact_path / "game_loops.npy", np.asarray(game_loops, dtype=np.int64))
    np.save(artifact_path / "timestamps.npy", np.asarray(timestamps, dtype=np.float64))
    stat = replay_path.stat()
    metadata = {
        "artifact_version": TOKENIZED_ARTIFACT_VERSION,
        "source_path": str(replay_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "timestep_count": len(frame),
        "token_count": len(token_ids),
    }
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")


def _artifact_is_current(artifact_path: Path, replay_path: Path) -> bool:
    metadata_path = artifact_path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = replay_path.stat()
    required = ("offsets.npy", "token_ids.npy", "owners.npy", "kinds.npy", "features.npy")
    return (
        metadata.get("artifact_version") == TOKENIZED_ARTIFACT_VERSION
        and metadata.get("source_size") == stat.st_size
        and metadata.get("source_mtime_ns") == stat.st_mtime_ns
        and all((artifact_path / name).exists() for name in required)
    )


def _timestep_owner_counts(replay: TokenizedReplay) -> np.ndarray:
    counts = np.zeros((replay.timestep_count, 2), dtype=np.int64)
    for timestep in range(replay.timestep_count):
        owners = replay.owners[replay.token_slice(timestep)]
        counts[timestep, 0] = np.count_nonzero(owners == P1_CODE)
        counts[timestep, 1] = np.count_nonzero(owners == P2_CODE)
    return counts


def _replay_id(path: Path) -> str:
    digest = hashlib.sha1(_normalized_path(path).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()
