"""Dataset construction for masked-diffusion training examples (both modes).

Two training modes are served from the same manifest-backed dataset class:

- PRE-TRAINING (``config.data.debut_mode`` False): the input is LITERALLY
  ABSENT -- no fog, no self/enemy input blocks, zero-length input tensors.
  The model sequence is 100% output canvas (the published MDLM/LLaDA pure
  reconstruction objective): the leading [WIN]/[LOSS] outcome token, then the
  full enemy reconstruction + future roll-out. Every content token is labeled
  with the single collapsed CLASS_CONTENT class.

- DEBUT FINE-TUNING (``debut_mode`` True): a clamped, fog-filtered input is
  served alongside a sparse debut-event canvas. The input interleaves
  [self records][enemy records][ONE delimiter] per timestep, fog omits enemy
  content tokens of every kind (entities and upgrades), and canvas labels use
  the 7-class debut taxonomy (visible/fogged/future-debut + structural +
  win-loss).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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
from thesis_ml.vocab.special_tokens import (
    DELIMITER_ID,
    END_ID,
    LOSS_ID,
    LOSS_TOKEN,
    PAD_ID,
    SPECIAL_TOKENS,
    WIN_ID,
    WIN_TOKEN,
)

CLASS_ENEMY_OBSERVED = 0
CLASS_ENEMY_FOGGED = 1
CLASS_ENEMY_FUTURE = 2
CLASS_DELIMITER = 3
CLASS_END = 4
CLASS_PAD = 5
# Fine-tuning-only class id for the single win/loss outcome token that sits at
# canvas position 0 in debut mode. Added ALONGSIDE the existing ids above; the
# existing ids are NOT renumbered so pretraining canvases keep the same labels.
CLASS_WINLOSS = 6

# Pre-training-only alias. In pre-training there is no input and no fog, so
# the observed/fogged/future three-way split that fine-tuning uses does not
# apply: every enemy content token collapses into this single class. Reuses
# CLASS_ENEMY_OBSERVED's value (0) rather than allocating a new id -- the two
# names are interchangeable, they just describe the same id in two different
# training modes. CLASS_ENEMY_FOGGED (1) and CLASS_ENEMY_FUTURE (2) are left
# defined above, unused, and un-renumbered, because fine-tuning still needs
# them.
CLASS_CONTENT = CLASS_ENEMY_OBSERVED

CLASS_LABELS: dict[str, int] = {
    "enemy-observed": CLASS_ENEMY_OBSERVED,
    "enemy-fogged": CLASS_ENEMY_FOGGED,
    "enemy-future": CLASS_ENEMY_FUTURE,
    "[DELIMITER]": CLASS_DELIMITER,
    "[END]": CLASS_END,
    "[PAD]": CLASS_PAD,
}

# Shared debut-mode class-id -> human-readable-name map. Exported so the other
# fine-tuning workers (loss weighting, sampling, evaluation) label the same 7
# classes identically. In debut mode the first three ids describe the fog state
# of a DEBUT event (an entity/upgrade's first appearance) rather than a plain
# reconstruction token, hence the "-debut" suffixes.
DEBUT_CLASS_ID_TO_NAME: dict[int, str] = {
    CLASS_ENEMY_OBSERVED: "visible-debut",
    CLASS_ENEMY_FOGGED: "fogged-debut",
    CLASS_ENEMY_FUTURE: "future-debut",
    CLASS_DELIMITER: "delimiter",
    CLASS_END: "end",
    CLASS_PAD: "pad",
    CLASS_WINLOSS: "win-loss",
}

# Pre-training-only class-id -> human-readable-name map (mirrors
# DEBUT_CLASS_ID_TO_NAME's role, but for the pre-training canvas). Pre-training
# never emits ids 1 or 2 (CLASS_ENEMY_FOGGED / CLASS_ENEMY_FUTURE): there is no
# fog and no input/future split in this mode, since the canvas IS the entire
# reconstruction target, so every content token is CLASS_CONTENT. Those two
# ids are intentionally ABSENT here -- this map is sparse (5 entries, ids
# 0/3/4/5/6), not a dense 0..N range. Any consumer that builds an id-indexed
# buffer (e.g. per-class loss weights) MUST size it by max(id) + 1, not
# len(map), or it will under-allocate.
PRETRAIN_CLASS_ID_TO_NAME: dict[int, str] = {
    CLASS_CONTENT: "content",
    CLASS_DELIMITER: "[DELIMITER]",
    CLASS_END: "[END]",
    CLASS_PAD: "[PAD]",
    CLASS_WINLOSS: "win-loss",
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
    """Lazy manifest-backed training examples for both training modes.

    Pre-training (``config.data.debut_mode`` False) serves CANVAS-ONLY
    examples: ``input_token_ids`` / ``clean_input_token_ids`` are zero-length,
    ``input_records`` is empty, and no fog is ever sampled (``config.fog`` is
    None for pre-training configs and is never read here). Debut fine-tuning
    (True) serves the clamped, per-serving fog-filtered interleaved input plus
    the sparse debut canvas. See the module docstring for the two grammars.
    """

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

        if self.config.data.debut_mode:
            # Fine-tuning: sample one fog rate for this served example and use
            # it both to build the fogged input and (via fogged_counts) to
            # drive the debut target's fog/future class split below.
            fog_rate = self._sample_fog_rate(rng)
            input_records, clean_records, fogged_counts, observed_counts = _build_artifact_input(
                replay,
                window,
                self.vocabulary,
                fog_rate=fog_rate,
                rng=rng,
            )
        else:
            # Pre-training: the input is LITERALLY ABSENT -- the model
            # sequence becomes 100% output canvas (no fog paradigm at all).
            # No fog rate is sampled and `config.fog` is never read here,
            # since it may be None/missing for pre-training configs. The
            # empty lists below produce zero-length input tensors below,
            # which contribute zero sequence positions downstream.
            input_records = []
            clean_records = []
            fogged_counts = {}
            observed_counts = {}

        # Both training modes now begin the canvas with the win/loss outcome
        # token (leading position 0, denoised last), so resolve it up front for
        # either target builder. debut_mode selects only the canvas BODY:
        #   - False (pre-training): full enemy reconstruction + future roll-out.
        #   - True  (debut fine-tuning): sparse first-appearance debut events.
        outcome_id = resolve_replay_outcome(window.replay_path, window.perspective_player)
        if self.config.data.debut_mode:
            target = _build_debut_target(
                replay,
                window,
                self.vocabulary,
                enemy_player,
                fogged_counts=fogged_counts,
                budget=self.config.data.canvas_budget_tokens,
                outcome_id=outcome_id,
            )
        else:
            target = _build_artifact_target(
                replay,
                window,
                self.vocabulary,
                enemy_player,
                fogged_counts=fogged_counts,
                budget=self.config.data.canvas_budget_tokens,
                outcome_id=outcome_id,
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
    """Build the fine-tuning input: per-timestep interleaved self+enemy blocks.

    This grammar is fine-tuning (debut_mode) ONLY -- pre-training has no input
    at all, so callers must not invoke this function when
    ``config.data.debut_mode`` is False.

    Layout, walking timesteps from ``window.start_timestep`` to
    ``window.end_timestep``:
        [self records for this timestep]
        [enemy records for this timestep -- fog-filtered for the fogged
         variant, unfiltered for the clean variant]
        [ONE [DELIMITER]]
    This replaces the old ``[all self timesteps][all enemy timesteps]``
    grammar (one delimiter per PLAYER per timestep) with exactly ONE
    delimiter per TIMESTEP, placed after that timestep's enemy records.

    Fog is applied uniformly to enemy content tokens of every ``token_kind``
    (entity AND upgrade) -- there is no longer an entity-only special case.

    Parameters:
        replay: Memory-mapped tokenized replay to read timestep records from.
        window: Manifest entry giving the perspective player and the
            input-window's start/end timesteps.
        vocabulary: Content vocabulary used to name token ids.
        fog_rate: Per-example probability that an enemy content token is
            omitted from the fogged input variant.
        rng: Random generator used for the fog Bernoulli draws (one draw per
            enemy content token, in serialization order).

    Returns:
        A 4-tuple ``(fogged_records, clean_records, fogged_counts,
        observed_counts)``:
            fogged_records: the interleaved input with fog applied.
            clean_records: the interleaved input with zero fog (used for the
                "clean" input variant carried alongside the fogged one).
            fogged_counts / observed_counts: per-(relative timestep, token
                name) counts of enemy content tokens omitted / kept by the
                fog draw, across all token kinds -- consumed by
                ``_build_debut_target`` to label fog/future debut classes.

    Calls:
        ``_artifact_timestep_records``, ``_artifact_delimiter``.
    """

    self_code = P1_CODE if window.perspective_player == "p1" else P2_CODE
    enemy_code = P2_CODE if self_code == P1_CODE else P1_CODE
    fogged_records: list[TokenRecord] = []
    clean_records: list[TokenRecord] = []
    fogged_counts: dict[tuple[int, str], int] = {}
    observed_counts: dict[tuple[int, str], int] = {}
    for relative_timestep, timestep in enumerate(range(window.start_timestep, window.end_timestep)):
        records = _artifact_timestep_records(replay, timestep, vocabulary, window.perspective_player)
        delimiter = _artifact_delimiter(replay, timestep)
        self_records = [record for code, record in records if code == self_code]
        enemy_records = [record for code, record in records if code == enemy_code]

        # Fog every enemy content token regardless of kind (entity or
        # upgrade) -- the old entity-only guard is removed.
        fogged_enemy_records: list[TokenRecord] = []
        for record in enemy_records:
            key = (relative_timestep, record.token_name)
            if rng.random() < fog_rate:
                _increment(fogged_counts, key)
                continue
            _increment(observed_counts, key)
            fogged_enemy_records.append(record)

        fogged_records.extend(self_records)
        fogged_records.extend(fogged_enemy_records)
        fogged_records.append(delimiter)

        clean_records.extend(self_records)
        clean_records.extend(enemy_records)
        clean_records.append(delimiter)
    return (
        fogged_records,
        clean_records,
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
    outcome_id: int,
) -> CanvasBuild:
    """Build the pre-training canvas: win/loss token + full enemy roll-out.

    Layout of the returned canvas:
        position 0: the ``outcome_id`` token (``WIN_ID`` or ``LOSS_ID``) labeled
            ``CLASS_WINLOSS``. It is denoised LAST at inference via the
            ``sampler.outcome_last`` constraint, mirroring the debut target.
        then, walking enemy timesteps from ``window.start_timestep`` to game end:
            the full enemy reconstruction (observed + fogged past/present) plus
            the enemy future continuation, each timestep followed by one
            ``[DELIMITER]``.
        finally: ``[END]`` (if the game end is reached) then ``[PAD]`` padding out
            to ``budget``.

    The outcome token consumes canvas position 0, so the reconstruction + future
    roll-out fits within ``budget - 1``. The in-window reconstruction is bounded
    by ``canvas_recon_fraction`` (see ``windowing._reconstruction_limit``), which
    is far below ``budget``, so the leading outcome token never forces an
    in-window overflow.

    Parameters:
        outcome_id: The win/loss token id to place at position 0, resolved by
            ``resolve_replay_outcome``. Both pre-training and debut fine-tuning
            now carry this token so the frozen-prior model never has to learn it
            fresh at fine-tune time.
    """

    enemy_code = P1_CODE if enemy_player == "p1" else P2_CODE
    remaining_fogged = dict(fogged_counts)
    token_ids: list[int] = []
    class_labels: list[int] = []
    metadata: list[dict[str, Any]] = []

    # Canvas position 0 is always the single win/loss outcome token (identical
    # placement to _build_debut_target so the sampler's outcome_last constraint
    # and the CLASS_WINLOSS loss weighting apply uniformly across both modes).
    token_ids.append(outcome_id)
    class_labels.append(CLASS_WINLOSS)
    metadata.append({"token_kind": "outcome", "timestep_index": None, "token_name": "[WIN/LOSS]"})

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
                _canvas_label(
                    record,
                    relative_timestep,
                    window.timestep_count,
                    remaining_fogged,
                    debut_mode=False,
                )
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


def resolve_replay_outcome(replay_path: str | Path, perspective_player: str) -> int:
    """Resolve the win/loss outcome token id for one replay and perspective.

    The original per-match metadata (with the recorded result) is a sibling of
    the game-state parquet: the parquet lives in a ``parquet/`` directory and the
    metadata lives in a sibling ``json/`` directory, with the basename suffix
    ``_game_state.parquet`` replaced by ``_metadata.json``. The metadata stores
    each player's result under ``players.<player>.result`` as the string
    "Victory" or "Defeat". We map "Victory" -> ``WIN_ID`` and "Defeat" ->
    ``LOSS_ID`` for the requested perspective player.

    Parameters:
        replay_path: Path to the game-state parquet for this replay (typically
            ``window.replay_path`` from the manifest).
        perspective_player: Either "p1" or "p2"; selects whose result is the
            outcome for this training example.

    Returns:
        ``WIN_ID`` (4) if the perspective player won, ``LOSS_ID`` (5) if they
        lost.

    Raises:
        ValueError: If ``perspective_player`` is not "p1"/"p2", if the win/loss
            special tokens are missing from the reserved vocabulary, if the
            metadata file is missing, if the player key is absent, or if the
            recorded result string is neither "Victory" nor "Defeat". This helper
            NEVER silently defaults to a win or a loss.

    Calls:
        Reads the sibling metadata JSON directly; uses ``SPECIAL_TOKENS`` /
        ``WIN_ID`` / ``LOSS_ID`` from the reserved-token module.
    """

    # Fail loudly if the reserved win/loss tokens are not present as expected.
    # Downstream fine-tuning workers rely on these exact ids being available.
    if SPECIAL_TOKENS.get(WIN_TOKEN) != WIN_ID or SPECIAL_TOKENS.get(LOSS_TOKEN) != LOSS_ID:
        raise ValueError(
            f"reserved vocabulary is missing win/loss tokens: expected "
            f"{WIN_TOKEN}={WIN_ID} and {LOSS_TOKEN}={LOSS_ID}"
        )

    if perspective_player not in ("p1", "p2"):
        raise ValueError("perspective_player must be 'p1' or 'p2'")

    # Derive the metadata path from the parquet path: go from ".../parquet/
    # match_<id>_game_state.parquet" to ".../json/match_<id>_metadata.json".
    parquet_path = Path(replay_path)
    metadata_name = parquet_path.name.replace("_game_state.parquet", "_metadata.json")
    if metadata_name == parquet_path.name:
        raise ValueError(
            f"replay path does not look like a game-state parquet: {parquet_path}"
        )
    metadata_path = parquet_path.parent.parent / "json" / metadata_name

    if not metadata_path.exists():
        raise ValueError(f"replay outcome metadata not found: {metadata_path}")

    metadata = _read_replay_metadata(str(metadata_path))
    players = metadata.get("players")
    if not isinstance(players, dict) or perspective_player not in players:
        raise ValueError(
            f"metadata {metadata_path} is missing players.{perspective_player}"
        )
    result = players[perspective_player].get("result")
    if result == "Victory":
        return WIN_ID
    if result == "Defeat":
        return LOSS_ID
    raise ValueError(
        f"unresolvable result {result!r} for {perspective_player} in {metadata_path}"
    )


@lru_cache(maxsize=256)
def _read_replay_metadata(metadata_path: str) -> dict[str, Any]:
    """Read immutable replay metadata once per DataLoader worker process."""

    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def _build_debut_target(
    replay: TokenizedReplay,
    window: WindowManifestEntry,
    vocabulary: ContentVocabulary,
    enemy_player: str,
    *,
    fogged_counts: dict[tuple[int, str], int],
    budget: int,
    outcome_id: int,
) -> CanvasBuild:
    """Build the debut build-order + win/loss canvas for one window (fine-tuning).

    Layout of the returned canvas:
        position 0: the ``outcome_id`` token (``WIN_ID`` or ``LOSS_ID``) labeled
            ``CLASS_WINLOSS``.
        then, walking enemy timesteps from ``window.start_timestep`` to game end:
            only the enemy entities/upgrades making their FIRST appearance at that
            timestep (a "debut"), followed by one ``[DELIMITER]``. A timestep with
            no new debut still emits a bare ``[DELIMITER]``, so back-to-back
            delimiters are legal and expected.
        finally: ``[END]`` (if the game end is reached) then ``[PAD]`` padding out
            to ``budget``.

    "First appearance" mirrors the build-order reference in
    ``eval/buildorder.py``: an entity debuts when a new instance of its type
    appears (its running-max count for that timestep exceeds the max seen so
    far), and an upgrade debuts the first timestep its token is present. Because
    the memory-mapped artifact does not store per-entity instance ids, entity
    debuts are detected by count increase, which is the same notion used by the
    counts-based ``extract_build_order``.

    Fog-state labels reuse the existing ``_canvas_label``: a debut event within
    the input window is labeled visible-debut / fogged-debut using
    ``fogged_counts`` (keyed by relative timestep + token name), and any debut
    at or beyond the input-window boundary (``window.timestep_count``) is labeled
    future-debut.

    Parameters:
        replay: Memory-mapped tokenized replay to read enemy records from.
        window: Manifest entry giving the start timestep, perspective, and
            input-window length used as the future boundary.
        vocabulary: Content vocabulary used to name token ids.
        enemy_player: "p1"/"p2" for the player whose build order is the target
            (the non-perspective player).
        fogged_counts: Per-(relative timestep, token name) counts of enemy
            entities hidden from the input, produced by ``_build_artifact_input``.
        budget: Total canvas length in tokens; the canvas is padded to exactly
            this length.
        outcome_id: The win/loss token id to place at position 0, resolved by
            ``resolve_replay_outcome``.

    Returns:
        A ``CanvasBuild`` (token ids, class labels, metadata, terminated,
        truncated) mirroring ``_build_artifact_target``'s return contract.

    Calls:
        ``_artifact_timestep_records``, ``_artifact_delimiter``, ``_canvas_label``,
        ``_canvas_metadata``.
    """

    enemy_code = P1_CODE if enemy_player == "p1" else P2_CODE
    # _canvas_label consumes fog counts as it labels, so copy to avoid mutating
    # the caller's dict (identical pattern to _build_artifact_target).
    remaining_fogged = dict(fogged_counts)

    token_ids: list[int] = []
    class_labels: list[int] = []
    metadata: list[dict[str, Any]] = []

    # Canvas position 0 is always the single win/loss outcome token.
    token_ids.append(outcome_id)
    class_labels.append(CLASS_WINLOSS)
    metadata.append({"token_kind": "outcome", "timestep_index": None, "token_name": "[WIN/LOSS]"})

    # Cross-timestep first-appearance state, UNIFIED across entity AND upgrade
    # token kinds (the old entity-vs-upgrade special case is removed).
    # running_max maps a token id to the largest per-timestep count of that
    # token seen so far; a token instance "debuts" the first N times its
    # per-timestep count exceeds that running max, N = count_now -
    # running_max_before. For upgrades -- cumulative flags whose per-timestep
    # count is always 0 or 1 -- this fires exactly once, on first appearance,
    # which is provably identical to the old seen_upgrades first-appearance
    # set (verified locally against real and synthetic replay data; see the
    # worker report). Operating on the memory-mapped arrays avoids
    # constructing TokenRecord objects for every non-debut unit in the
    # remainder of the replay. Debut canvases are sparse, so that object
    # construction was the dominant fine-tuning loader cost.
    running_max: dict[int, int] = {}

    truncated = False
    reached_game_end = False
    for timestep in range(window.start_timestep, replay.timestep_count):
        relative_timestep = timestep - window.start_timestep
        token_slice = replay.token_slice(timestep)
        enemy_positions = [
            position
            for position in range(token_slice.start, token_slice.stop)
            if int(replay.owners[position]) == enemy_code
        ]

        # Count how many of each token id (entity instance OR upgrade) are
        # present this timestep so we can tell how many are NEW relative to
        # the running max.
        counts_this_step: dict[int, int] = {}
        for position in enemy_positions:
            token_id = int(replay.token_ids[position])
            counts_this_step[token_id] = counts_this_step.get(token_id, 0) + 1

        # Collect this timestep's debut events in the artifact's natural
        # order. Only the first N new instances of a token type debut, where
        # N = count_now - running_max_before (N is 0 or 1 for upgrades).
        debut_positions: list[int] = []
        emitted_per_token: dict[int, int] = {}
        for position in enemy_positions:
            token_id = int(replay.token_ids[position])
            new_instances = counts_this_step[token_id] - running_max.get(token_id, 0)
            already_emitted = emitted_per_token.get(token_id, 0)
            if already_emitted < new_instances:
                debut_positions.append(position)
                emitted_per_token[token_id] = already_emitted + 1

        # Update running max AFTER deciding debuts for this timestep.
        for token_id, count in counts_this_step.items():
            if count > running_max.get(token_id, 0):
                running_max[token_id] = count

        # Every timestep contributes its debut events plus one delimiter. Empty
        # timesteps therefore produce a bare delimiter (back-to-back delimiters).
        is_final_game_timestep = timestep == replay.timestep_count - 1
        # Reserve one extra slot for [END] on the final timestep.
        required = len(debut_positions) + 1 + (1 if is_final_game_timestep else 0)
        if len(token_ids) + required > budget:
            # Long game overflows the canvas: drop this and all later whole
            # timesteps and mark the example truncated.
            truncated = True
            break
        for position in debut_positions:
            record = _artifact_canvas_record(
                replay,
                timestep,
                position,
                vocabulary,
                window.perspective_player,
            )
            token_ids.append(record.token_id)
            class_labels.append(
                _canvas_label(
                    record,
                    relative_timestep,
                    window.timestep_count,
                    remaining_fogged,
                    debut_mode=True,
                )
            )
            metadata.append(_canvas_metadata(record, relative_timestep))
        token_ids.append(DELIMITER_ID)
        class_labels.append(CLASS_DELIMITER)
        metadata.append(
            {
                "token_id": DELIMITER_ID,
                "token_name": "[DELIMITER]",
                "token_kind": "delimiter",
                "timestep_index": relative_timestep,
                "owner": None,
                "instance_id": None,
                "game_loop": int(replay.game_loops[timestep]),
            }
        )
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


def _artifact_canvas_record(
    replay: TokenizedReplay,
    timestep: int,
    position: int,
    vocabulary: ContentVocabulary,
    perspective_player: str,
) -> TokenRecord:
    """Materialize only the fields consumed by canvas labels and metadata."""

    owner_code = int(replay.owners[position])
    owner = "p1" if owner_code == P1_CODE else "p2"
    token_id = int(replay.token_ids[position])
    token_name = vocabulary.token_name_for(token_id)
    return TokenRecord(
        token_id=token_id,
        token_name=token_name,
        token_kind="entity" if int(replay.kinds[position]) == ENTITY_CODE else "upgrade",
        owner=owner,
        allegiance="self" if owner == perspective_player else "enemy",
        game_loop=int(replay.game_loops[timestep]),
        timestamp_seconds=_optional_artifact_timestamp(replay.timestamps[timestep]),
        entity_type=token_name,
    )


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

        # Fog every enemy content token regardless of kind (entity or
        # upgrade) -- mirrors the same guard removal in _build_artifact_input,
        # kept consistent here since this is a legacy fallback copy of the
        # same fog-application logic.
        for record in enemy_records:
            if rng.random() < fog_rate:
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
    *,
    debut_mode: bool = True,
) -> int:
    """Assign a class id to one non-outcome canvas token.

    Parameters:
        record: The token whose class is being decided.
        timestep_index: Relative timestep index of this record within the
            canvas (0-based from the window start).
        input_timestep_count: Number of timesteps in the input window. Only
            consulted when ``debut_mode`` is True, to tell whether this
            record's timestep lies beyond the input window (the "future").
        fogged_counts: Mutable per-(timestep, token name) remaining-fogged
            counts; one count is consumed each time a fogged token is
            labeled. Only consulted when ``debut_mode`` is True -- pre-training
            has no fog at all.
        debut_mode: True for fine-tuning canvases, which keep the 3-way
            visible/fogged/future split (part of the 7-class debut
            taxonomy). False for pre-training canvases, where every content
            token collapses to the single ``CLASS_CONTENT`` class, because
            pre-training has no input and no fog (the published MDLM/LLaDA
            pure-reconstruction objective: the whole canvas is one
            undifferentiated reconstruction target).

    Returns:
        ``CLASS_DELIMITER`` for delimiter tokens; otherwise ``CLASS_CONTENT``
        when ``debut_mode`` is False, or one of ``CLASS_ENEMY_OBSERVED`` /
        ``CLASS_ENEMY_FOGGED`` / ``CLASS_ENEMY_FUTURE`` when ``debut_mode`` is
        True.
    """

    if record.token_id == DELIMITER_ID:
        return CLASS_DELIMITER
    if not debut_mode:
        # Pre-training: no fog, no input/future split -- every content token
        # collapses to CLASS_CONTENT (alias of CLASS_ENEMY_OBSERVED).
        return CLASS_CONTENT
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
