"""Framework-neutral replay result metadata, shared by all sequence formats."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from thesis_shared.vocab.special_tokens import SPECIAL_TOKENS, WIN_TOKEN, WIN_ID, LOSS_TOKEN, LOSS_ID


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
