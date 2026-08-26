from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_training_report import (
    ReportPreparationError,
    parse_args,
    prepare_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_finished_run(root: Path, *, stop_reason: str = "early_stopping") -> Path:
    run = root / "example-run"
    finished = run / "checkpoints" / "finished"
    best = run / "checkpoints" / "best"
    metrics = run / "metrics"
    finished.mkdir(parents=True)
    best.mkdir(parents=True)
    metrics.mkdir(parents=True)

    metadata = {
        "architecture_identity": "dense-multinomial-SC2-v2+frozen_input_kv",
        "completed_epochs": 3,
        "configured_epochs": 5,
        "created_utc": "2026-08-25T19:46:29+00:00",
        "default_serving_weights": "ema",
        "diffusion_process": "uniform",
        "feature_statistics_identity": "abc123",
        "global_step": 30,
        "self_conditioning": True,
        "stop_reason": stop_reason,
        "vocab_size": 291,
        "weights": {
            "ema": "model.ema.safetensors",
            "raw": "model.raw.safetensors",
        },
    }
    (finished / "finished_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    config = {
        "pipeline": {"batch_size": 6},
        "train": {
            "accumulation_steps": 7,
            "early_stopping_patience_epochs": 2,
            "lr": 0.0003,
            "lr_schedule": "wsd",
            "precision": "bf16",
            "warmup": 500,
        },
        "model": {"d_model": 384, "ffn": 1536, "heads": 6, "layers": 12},
    }
    (finished / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (finished / "model.ema.safetensors").write_bytes(b"ema")
    (finished / "model.raw.safetensors").write_bytes(b"raw")
    (best / "epoch-0002.pt").write_bytes(b"checkpoint payload stays local")

    fieldnames = [
        "epoch",
        "train_loss",
        "dev_loss",
        "total_tokens_ingested",
        "total_unique_tokens_seen",
        "tokens_per_second",
        "wall_clock_elapsed_seconds",
        "average_cuda_device_memory_used_bytes",
        "average_cuda_device_memory_gap_bytes",
    ]
    with (metrics / "epoch_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "train_loss": 1.0,
                "dev_loss": 0.8,
                "total_tokens_ingested": 100,
                "total_unique_tokens_seen": 10,
                "tokens_per_second": 1000,
                "wall_clock_elapsed_seconds": 10,
                "average_cuda_device_memory_used_bytes": 100,
                "average_cuda_device_memory_gap_bytes": 20,
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "train_loss": 0.5,
                "dev_loss": 0.3,
                "total_tokens_ingested": 200,
                "total_unique_tokens_seen": 12,
                "tokens_per_second": 1100,
                "wall_clock_elapsed_seconds": 20,
                "average_cuda_device_memory_used_bytes": 110,
                "average_cuda_device_memory_gap_bytes": 25,
            }
        )
        writer.writerow(
            {
                "epoch": 3,
                "train_loss": 0.4,
                "dev_loss": 0.4,
                "total_tokens_ingested": 300,
                "total_unique_tokens_seen": 13,
                "tokens_per_second": 1200,
                "wall_clock_elapsed_seconds": 30,
                "average_cuda_device_memory_used_bytes": 120,
                "average_cuda_device_memory_gap_bytes": 30,
            }
        )
        writer.writerow({"tokens_per_second": 1150})
    (metrics / "replay_selection.json").write_text(
        json.dumps({"train_replay_ids": ["match_1"], "dev_replay_ids": ["match_2"]}),
        encoding="utf-8",
    )
    (metrics / "pipeline.log").write_text("resumed=True steps=30\n", encoding="utf-8")
    return run


def test_prepare_report_writes_curated_bundle_and_excludes_weights(tmp_path: Path) -> None:
    run = _write_finished_run(tmp_path / "runs")
    output_root = tmp_path / "reports"

    report_dir = prepare_report(
        run,
        output_root=output_root,
        architecture_path=ROOT / "Model_Architecture" / "MODEL_ARCHITECTURE.md",
    )

    assert report_dir.name == "2026-08-25-example-run"
    assert (report_dir / "RUN_FACTS.json").exists()
    assert (report_dir / "TRAINING_CURVES.png").stat().st_size > 0
    assert sorted(path.name for path in (report_dir / "evidence").iterdir()) == [
        "config.json",
        "epoch_metrics.csv",
        "finished_metadata.json",
        "pipeline.log",
        "replay_selection.json",
    ]
    assert not list(report_dir.rglob("*.pt"))
    assert not list(report_dir.rglob("*.safetensors"))

    facts = json.loads((report_dir / "RUN_FACTS.json").read_text(encoding="utf-8"))
    assert facts["run_completion"]["stop_reason"] == "early_stopping"
    assert facts["epoch_metrics"]["valid_epoch_rows"] == 3
    assert facts["epoch_metrics"]["derived"]["best_dev_epoch"] == 2
    assert facts["epoch_metrics"]["best_dev"]["dev_loss"] == 0.3
    assert facts["epoch_metrics"]["final"]["dev_loss"] == 0.4
    assert facts["epoch_metrics"]["anomalous_rows_excluded"] == [
        {
            "line": 5,
            "reason": "missing_or_non_integer_epoch",
            "non_empty_columns": ["tokens_per_second"],
        }
    ]
    assert facts["epoch_metrics"]["wall_clock"]["status"] == "cumulative_monotonic"
    assert facts["epoch_metrics"]["wall_clock"]["recorded_fit_seconds"] == 30.0
    assert facts["checkpoint_observations"]["best_checkpoint_exists_locally"] is True
    assert facts["model"]["architecture_reference"]["matches_finished_run"] is True


def test_prepare_report_rejects_non_finished_stop_reason(tmp_path: Path) -> None:
    run = _write_finished_run(tmp_path / "runs", stop_reason="max_steps")

    with pytest.raises(ReportPreparationError, match="not a publishable finish"):
        prepare_report(run, output_root=tmp_path / "reports")


def test_prepare_report_reconstructs_legacy_wall_clock_segments(tmp_path: Path) -> None:
    run = _write_finished_run(tmp_path / "runs")
    metrics_path = run / "metrics" / "epoch_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows[2]["wall_clock_elapsed_seconds"] = "5"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_dir = prepare_report(run, output_root=tmp_path / "reports")
    facts = json.loads((report_dir / "RUN_FACTS.json").read_text(encoding="utf-8"))
    wall_clock = facts["epoch_metrics"]["wall_clock"]

    assert wall_clock["status"] == "legacy_resume_resets_detected"
    assert wall_clock["exact_total_available"] is False
    assert wall_clock["reset_count"] == 1
    assert wall_clock["recorded_fit_seconds_lower_bound"] == 25.0
    assert [(segment["start_epoch"], segment["end_epoch"]) for segment in wall_clock["segments"]] == [
        (1, 2),
        (3, 3),
    ]


def test_parse_args_accepts_portable_run_path() -> None:
    args = parse_args(["tests/output/example", "--report-id", "trial-01"])
    assert args.run_directory == "tests/output/example"
    assert args.report_id == "trial-01"
