"""Minimal local/S3 storage resolver for pipeline paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


class StorageResolver:
    def __init__(self, *, s3_client: object | None = None) -> None:
        self._s3_client = s3_client

    def is_s3(self, uri: str) -> bool:
        return uri.startswith("s3://")

    def exists(self, uri: str) -> bool:
        if not self.is_s3(uri):
            return Path(uri).exists()
        parsed = parse_s3_uri(uri)
        try:
            self._client().head_object(Bucket=parsed.bucket, Key=parsed.key)
            return True
        except Exception:
            return False

    def ensure_dir(self, uri: str) -> Path | None:
        if self.is_s3(uri):
            return None
        path = Path(uri)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_files(self, uri: str, pattern: str) -> list[str]:
        if not self.is_s3(uri):
            return [str(path) for path in sorted(Path(uri).glob(pattern)) if path.is_file()]
        parsed = parse_s3_uri(uri)
        prefix = parsed.key.rstrip("/") + "/"
        response = self._client().list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)
        keys = [item["Key"] for item in response.get("Contents", [])]
        suffix = pattern.removeprefix("*")
        return [f"s3://{parsed.bucket}/{key}" for key in keys if key.endswith(suffix)]

    def latest_checkpoint(self, uri: str) -> str | None:
        candidates = self.list_files(uri, "*.pt")
        if not candidates:
            return None
        last = [candidate for candidate in candidates if candidate.endswith("last.pt")]
        return sorted(last or candidates)[-1]

    def put_file(self, local_path: str | Path, uri: str) -> None:
        if not self.is_s3(uri):
            destination = Path(uri)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(local_path).read_bytes())
            return
        parsed = parse_s3_uri(uri)
        self._client().upload_file(str(local_path), parsed.bucket, parsed.key)

    def get_file(self, uri: str, local_path: str | Path) -> Path:
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.is_s3(uri):
            destination.write_bytes(Path(uri).read_bytes())
            return destination
        parsed = parse_s3_uri(uri)
        self._client().download_file(parsed.bucket, parsed.key, str(destination))
        return destination

    def put_directory(self, local_dir: str | Path, uri: str) -> None:
        base = Path(local_dir)
        for path in _iter_files(base):
            relative = path.relative_to(base).as_posix()
            target = f"{uri.rstrip('/')}/{relative}"
            self.put_file(path, target)

    def _client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 storage requires boto3 to be installed") from exc
        self._s3_client = boto3.client("s3")
        return self._s3_client


def parse_s3_uri(uri: str) -> S3Uri:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    without_scheme = uri[5:]
    bucket, separator, key = without_scheme.partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"S3 URI must include bucket and key: {uri}")
    return S3Uri(bucket=bucket, key=key)


def _iter_files(directory: Path) -> Iterable[Path]:
    return (path for path in directory.rglob("*") if path.is_file())
