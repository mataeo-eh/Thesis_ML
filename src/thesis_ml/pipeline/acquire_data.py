"""Decoupled data-acquisition entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.pipeline.storage import StorageResolver


class CredentialError(RuntimeError):
    """Raised when required external-service credentials are missing."""


@dataclass(frozen=True)
class AcquisitionResult:
    commands: list[list[str]]
    output_uri: str
    dry_run: bool


def run_acquisition(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    storage: StorageResolver | None = None,
) -> AcquisitionResult:
    config = load_config(config_path)
    resolver = storage or StorageResolver()
    raw_dir = _local_stage_path(config.storage.raw_uri, config.storage.local_cache_dir, "raw")
    output_dir = _local_stage_path(config.storage.data_uri, config.storage.local_cache_dir, "processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    if config.data_source.source == "kaggle":
        _require_env(config.data_source.kaggle_username_env)
        _require_env(config.data_source.kaggle_key_env)
        commands.append(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                config.data_source.kaggle_dataset,
                "-p",
                str(raw_dir),
                "--unzip",
            ]
        )
    elif config.data_source.source not in {"local", "aiarena"}:
        raise ValueError(f"unsupported data source: {config.data_source.source}")

    commands.append(_extractor_command(config, raw_dir, output_dir))
    if not dry_run:
        for command in commands:
            subprocess.run(command, cwd=_extractor_cwd(config), check=True)
        if resolver.is_s3(config.storage.data_uri):
            resolver.put_directory(output_dir, config.storage.data_uri)

    return AcquisitionResult(commands=commands, output_uri=config.storage.data_uri, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and extract SC2 replay data")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_acquisition(args.config, dry_run=args.dry_run)
    for command in result.commands:
        print(shlex.join(command))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise CredentialError(f"required credential environment variable is missing: {name}")
    return value


def _extractor_command(config: ProjectConfig, raw_dir: Path, output_dir: Path) -> list[str]:
    command = shlex.split(config.data_source.extractor_command)
    if not command:
        raise ValueError("data_source.extractor_command must not be empty")
    return command + [
        "--process-replay-directory",
        str(raw_dir),
        "--output",
        str(output_dir),
        "--workers",
        str(config.data_source.workers),
    ]


def _extractor_cwd(config: ProjectConfig) -> Path:
    return Path(config.data_source.extractor_path)


def _local_stage_path(uri: str, cache_dir: str, name: str) -> Path:
    if uri.startswith("s3://"):
        return Path(cache_dir) / name
    return Path(uri)


if __name__ == "__main__":
    main()
