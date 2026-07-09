"""Estimate full-replay input and output token sequence lengths.

The default dataset location is derived from this file's repository position;
no machine-specific absolute path is embedded or written to the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import pyarrow.compute as pc
import pyarrow.parquet as pq

from thesis_ml.serialize import EntityColumnGroup, parse_entity_columns, parse_upgrades


SCRIPT_DIR = Path(__file__).parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = WORKSPACE_DIR / "data" / "quickstart" / "parquet"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "output" / "context_window_estimate.json"
DEFAULT_PATTERN = "*.parquet"
PERSPECTIVES = ("p1", "p2")
END_TOKEN_COUNT = 1


@dataclass(frozen=True)
class ReplayTokenCounts:
    """Unpadded full-replay token lengths for both player perspectives.

    Two separate input token counts are reported because the input grammar is
    now training-mode-dependent (see ``estimate_replay`` and
    ``thesis_ml.data.dataset``):
        pretrain_input_tokens: always 0. Pre-training's model sequence is
            100% output canvas -- there is no input at all (no self/enemy
            blocks, no fog, no delimiters devoted to an input segment).
        finetune_input_tokens: all self tokens plus all zero-fog enemy
            tokens, interleaved per timestep with exactly ONE delimiter per
            timestep (``_build_artifact_input``'s fine-tuning grammar).
    The output/target canvas token counts are unaffected by this mode split
    (the reconstruction-target grammar is unchanged across both modes) and
    stay per-perspective as before.
    """

    replay: str
    timesteps: int
    p1_content_tokens: int
    p2_content_tokens: int
    pretrain_input_tokens: int
    finetune_input_tokens: int
    p1_perspective_output_tokens: int
    p2_perspective_output_tokens: int


def estimate_replay(parquet_path: Path) -> ReplayTokenCounts:
    """Count full-replay model tokens, for both training modes, without loading the wide table."""

    parquet = pq.ParquetFile(parquet_path)
    timesteps = parquet.metadata.num_rows
    groups = parse_entity_columns(parquet.schema_arrow.names)
    leaf_indexes = {
        parquet.metadata.schema.column(index).path: index
        for index in range(parquet.metadata.num_columns)
    }

    entity_counts = {"p1": 0, "p2": 0}
    for group in groups:
        entity_counts[group.owner] += _count_present_entity_rows(parquet, group, leaf_indexes)

    upgrade_counts = _count_upgrade_tokens(parquet)
    p1_content = entity_counts["p1"] + upgrade_counts["p1"]
    p2_content = entity_counts["p2"] + upgrade_counts["p2"]

    # Pre-training: the input is LITERALLY ABSENT -- the model sequence is
    # 100% output canvas (the published MDLM/LLaDA pure-reconstruction
    # objective). No self/enemy blocks, no fog, no delimiters.
    pretrain_input_tokens = 0

    # Fine-tuning: input interleaves [self][enemy] records per timestep with
    # exactly ONE delimiter per timestep (previously: one delimiter per
    # PLAYER per timestep, i.e. 2 per timestep -- that old grammar is gone).
    finetune_input_tokens = p1_content + p2_content + timesteps

    # Output/target canvas grammar is unchanged across both modes: the full
    # enemy reconstruction, one delimiter per timestep, then [END].
    p1_output = p2_content + timesteps + END_TOKEN_COUNT
    p2_output = p1_content + timesteps + END_TOKEN_COUNT
    return ReplayTokenCounts(
        replay=parquet_path.name,
        timesteps=timesteps,
        p1_content_tokens=p1_content,
        p2_content_tokens=p2_content,
        pretrain_input_tokens=pretrain_input_tokens,
        finetune_input_tokens=finetune_input_tokens,
        p1_perspective_output_tokens=p1_output,
        p2_perspective_output_tokens=p2_output,
    )


def build_report(replays: Sequence[ReplayTokenCounts]) -> dict[str, object]:
    """Build auditable aggregate statistics over replay-perspective samples.

    Reports statistics for BOTH training-mode input grammars side by side
    (pre-training's absent input and fine-tuning's interleaved input) so this
    script stays a useful context-window planning tool for either pipeline.
    The output/target canvas statistics are shared (mode-independent).
    """

    if not replays:
        raise ValueError("at least one replay is required")

    pretrain_input_lengths: list[int] = []
    finetune_input_lengths: list[int] = []
    output_lengths: list[int] = []
    pretrain_combined_lengths: list[int] = []
    finetune_combined_lengths: list[int] = []
    timestep_counts = [replay.timesteps for replay in replays]
    for replay in replays:
        for output_length in (
            replay.p1_perspective_output_tokens,
            replay.p2_perspective_output_tokens,
        ):
            pretrain_input_lengths.append(replay.pretrain_input_tokens)
            finetune_input_lengths.append(replay.finetune_input_tokens)
            output_lengths.append(output_length)
            pretrain_combined_lengths.append(replay.pretrain_input_tokens + output_length)
            finetune_combined_lengths.append(replay.finetune_input_tokens + output_length)

    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "parquet_files": len(replays),
            "perspective_samples": len(output_lengths),
            "file_pattern": DEFAULT_PATTERN,
        },
        "token_accounting": {
            "sample_unit": "one replay from one player perspective",
            "entity": "one token per entity instance with a non-null attribute at a timestep",
            "upgrade": "one token per listed cumulative upgrade at a timestep",
            "pretrain_input": (
                "always 0 -- pre-training has no input at all; the model sequence is "
                "100% output canvas (see SC2DiffusionDataset.__getitem__, debut_mode=false)"
            ),
            "finetune_input": (
                "all self tokens plus all zero-fog enemy tokens, interleaved per timestep "
                "([self][enemy] per timestep) with exactly one delimiter per timestep"
            ),
            "output": "all enemy tokens, with one delimiter per timestep and one terminal [END] token",
            "padding": "excluded",
        },
        "statistics": {
            "timesteps": _descriptive_statistics(timestep_counts),
            "pretrain_input_tokens": _descriptive_statistics(pretrain_input_lengths),
            "finetune_input_tokens": _descriptive_statistics(finetune_input_lengths),
            "output_tokens": _descriptive_statistics(output_lengths),
            "pretrain_combined_input_output_tokens": _descriptive_statistics(pretrain_combined_lengths),
            "finetune_combined_input_output_tokens": _descriptive_statistics(finetune_combined_lengths),
        },
        "replays": [asdict(replay) for replay in replays],
    }


def estimate_directory(input_dir: Path, pattern: str = DEFAULT_PATTERN) -> list[ReplayTokenCounts]:
    """Estimate every matching parquet in deterministic filename order."""

    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {pattern!r} in the configured input directory")

    estimates: list[ReplayTokenCounts] = []
    for index, path in enumerate(paths, start=1):
        estimates.append(estimate_replay(path))
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)} parquet files (latest: {path.name})", flush=True)
    return estimates


def write_report(report: dict[str, object], output_path: Path) -> None:
    """Write the JSON report atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def _count_present_entity_rows(
    parquet: pq.ParquetFile,
    group: EntityColumnGroup,
    leaf_indexes: dict[str, int],
) -> int:
    """Count rows where an entity has at least one non-null attribute.

    Current extractor schemas always include ``health`` and populate every
    entity attribute during lifecycle transitions.  Reading that scalar column
    is therefore sufficient and avoids materializing tens of thousands of wide
    columns.  The exact any-attribute calculation remains as a schema fallback.
    """

    health_column = group.attributes.get("health")
    if health_column is not None:
        count = _metadata_non_null_count(parquet, health_column, leaf_indexes)
        if count is not None:
            return count
        health = parquet.read(columns=[health_column]).column(health_column)
        return len(health) - health.null_count

    columns = list(group.attributes.values())
    table = parquet.read(columns=columns)
    present = None
    for column in columns:
        valid = pc.is_valid(table.column(column))
        present = valid if present is None else pc.or_(present, valid)
    return int(pc.sum(present).as_py()) if present is not None else 0


def _metadata_non_null_count(
    parquet: pq.ParquetFile,
    column_name: str,
    leaf_indexes: dict[str, int],
) -> int | None:
    column_index = leaf_indexes.get(column_name)
    if column_index is None:
        return None

    non_null = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        statistics = row_group.column(column_index).statistics
        if statistics is None or not statistics.has_null_count:
            return None
        non_null += row_group.num_rows - statistics.null_count
    return non_null


def _count_upgrade_tokens(parquet: pq.ParquetFile) -> dict[str, int]:
    available = set(parquet.schema_arrow.names)
    columns = [f"{owner}_upgrades" for owner in PERSPECTIVES if f"{owner}_upgrades" in available]
    if not columns:
        return {owner: 0 for owner in PERSPECTIVES}

    table = parquet.read(columns=columns)
    counts = {owner: 0 for owner in PERSPECTIVES}
    for owner in PERSPECTIVES:
        column_name = f"{owner}_upgrades"
        if column_name not in columns:
            continue
        counts[owner] = sum(len(parse_upgrades(value)) for value in table.column(column_name).to_pylist())
    return counts


def _descriptive_statistics(values: Iterable[int]) -> dict[str, int | float | list[int]]:
    sample = list(values)
    if not sample:
        raise ValueError("statistics require at least one value")
    frequencies = Counter(sample)
    mode_frequency = max(frequencies.values())
    modes = sorted(value for value, frequency in frequencies.items() if frequency == mode_frequency)
    return {
        "minimum": min(sample),
        "maximum": max(sample),
        "mean": mean(sample),
        "median": median(sample),
        "mode": modes[0],
        "mode_frequency": mode_frequency,
        "all_modes": modes,
        "sample_count": len(sample),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    replays = estimate_directory(args.input_dir, args.pattern)
    report = build_report(replays)
    report["dataset"]["file_pattern"] = args.pattern
    write_report(report, args.output)
    print(f"Wrote context-window statistics for {len(replays)} parquet files.")


if __name__ == "__main__":
    main()
