"""Fail before parsing if an uploaded source/data file differs from its bundle."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify(root: Path) -> None:
    manifest = json.loads((root / "upload_manifest.json").read_text(encoding="utf-8-sig"))
    expected = set()
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("unsafe upload member")
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        if value.hexdigest() != item["sha256"]:
            raise ValueError(f"upload checksum mismatch: {item['path']}")
        if item["path"].startswith("parquet/"):
            expected.add(path.name)
    observed = {p.name for p in (root / "parquet").glob("*_game_state.parquet")}
    if not expected or observed != expected:
        raise ValueError("uploaded replay set differs from manifest")
    print(f"Verified {len(manifest['files'])} uploaded files, {len(expected)} replays", flush=True)


if __name__ == "__main__":
    verify(ROOT)
