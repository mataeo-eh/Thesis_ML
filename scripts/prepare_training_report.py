"""Prepare a compact, Git-trackable evidence bundle for a finished training run.

Raw launcher output remains local and ignored.  This utility validates a
finished export, derives auditable first/best/final facts from epoch metrics,
copies only a small allowlist of textual evidence, and renders a loss curve.
It never reads or copies checkpoint tensor payloads.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "reports" / "training-runs"
ARCHITECTURE_PATH = ROOT / "Model_Architecture" / "MODEL_ARCHITECTURE.md"
MAX_BUNDLE_FILE_BYTES = 10 * 1024 * 1024
FINISHED_STOP_REASONS = {"completed_all_epochs", "early_stopping"}
EVIDENCE_SOURCES = {
    "config.json": Path("checkpoints/finished/config.json"),
    "finished_metadata.json": Path("checkpoints/finished/finished_metadata.json"),
    "epoch_metrics.csv": Path("metrics/epoch_metrics.csv"),
    "replay_selection.json": Path("metrics/replay_selection.json"),
    "pipeline.log": Path("metrics/pipeline.log"),
}


class ReportPreparationError(RuntimeError):
    """Raised when a run cannot be published as a trustworthy report bundle."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportPreparationError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportPreparationError(f"JSON evidence must be an object: {path}")
    return payload


def _number(value: str) -> int | float | str:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        integer = int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return integer


def _metric_snapshot(row: Mapping[str, str]) -> dict[str, int | float | str]:
    return {key: _number(value) for key, value in row.items() if value.strip()}


def _wall_clock_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Describe cumulative timing and reconstruct legacy resume-reset segments.

    Older checkpoints saved only the wall-clock baseline from process startup,
    so a resumed process restarted this metric near zero.  Summing each
    segment's terminal value recovers completed, recorded fit time, but remains
    a lower bound because work after the last recorded row of a killed process
    may be absent.
    """

    segments: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    segment_start = 0
    values = [float(row["wall_clock_elapsed_seconds"]) for row in rows]
    for index in range(1, len(rows)):
        if values[index] >= values[index - 1]:
            continue
        segments.append(
            {
                "start_epoch": int(rows[segment_start]["epoch"]),
                "end_epoch": int(rows[index - 1]["epoch"]),
                "terminal_recorded_seconds": values[index - 1],
            }
        )
        resets.append(
            {
                "at_epoch": int(rows[index]["epoch"]),
                "previous_recorded_seconds": values[index - 1],
                "new_recorded_seconds": values[index],
            }
        )
        segment_start = index
    segments.append(
        {
            "start_epoch": int(rows[segment_start]["epoch"]),
            "end_epoch": int(rows[-1]["epoch"]),
            "terminal_recorded_seconds": values[-1],
        }
    )
    recorded_seconds = sum(segment["terminal_recorded_seconds"] for segment in segments)
    if resets:
        return {
            "status": "legacy_resume_resets_detected",
            "exact_total_available": False,
            "reset_count": len(resets),
            "resets": resets,
            "segments": segments,
            "recorded_fit_seconds_lower_bound": recorded_seconds,
            "note": (
                "Elapsed time restarted on legacy resumes. The segment sum is "
                "completed recorded fit time, not an exact end-to-end duration."
            ),
        }
    return {
        "status": "cumulative_monotonic",
        "exact_total_available": True,
        "reset_count": 0,
        "resets": [],
        "segments": segments,
        "recorded_fit_seconds": values[-1],
    }


def _read_epoch_metrics(
    path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ReportPreparationError(f"cannot open epoch metrics {path}: {exc}") from exc

    valid: list[dict[str, str]] = []
    anomalies: list[dict[str, Any]] = []
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ReportPreparationError("epoch metrics must contain an epoch column")
        for line_number, row in enumerate(reader, start=2):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            epoch_text = normalized.get("epoch", "")
            if not epoch_text.isdigit():
                anomalies.append(
                    {
                        "line": line_number,
                        "reason": "missing_or_non_integer_epoch",
                        "non_empty_columns": [
                            key for key, value in normalized.items() if value
                        ],
                    }
                )
                continue
            if not normalized.get("train_loss") or not normalized.get("dev_loss"):
                raise ReportPreparationError(
                    f"epoch metrics line {line_number} lacks train_loss or dev_loss"
                )
            valid.append(normalized)

    if not valid:
        raise ReportPreparationError("epoch metrics contain no valid epoch rows")
    epochs = [int(row["epoch"]) for row in valid]
    if epochs != sorted(set(epochs)):
        raise ReportPreparationError(
            "valid epoch rows must have unique, strictly increasing epoch numbers"
        )
    return valid, anomalies


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ReportPreparationError(f"cannot derive a report slug from {value!r}")
    return slug


def _created_date(metadata: Mapping[str, Any]) -> str:
    created = str(metadata.get("created_utc", ""))
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", created)
    if not match:
        raise ReportPreparationError(
            "finished_metadata.json must contain ISO created_utc"
        )
    return match.group(1)


def _architecture_reference(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportPreparationError(f"cannot read architecture reference {path}: {exc}") from exc
    identity_match = re.search(
        r"Derived `architecture_identity`\s*\|\s*`([^`]+)`", text
    )
    parameter_match = re.search(
        r"\*\*Total trainable parameters\*\*\s*\|\s*\*\*([0-9,]+)\*\*", text
    )
    if not identity_match or not parameter_match:
        raise ReportPreparationError(
            "architecture reference lacks the expected identity or parameter total"
        )
    return {
        "path": "Model_Architecture/MODEL_ARCHITECTURE.md",
        "architecture_identity": identity_match.group(1),
        "trainable_parameters": int(parameter_match.group(1).replace(",", "")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_evidence(run_dir: Path, evidence_dir: Path) -> dict[str, dict[str, Any]]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    for destination_name, relative_source in EVIDENCE_SOURCES.items():
        source = run_dir / relative_source
        if not source.exists():
            if destination_name in {"replay_selection.json", "pipeline.log"}:
                continue
            raise ReportPreparationError(f"required run evidence is missing: {source}")
        size = source.stat().st_size
        if size > MAX_BUNDLE_FILE_BYTES:
            raise ReportPreparationError(
                f"refusing oversized evidence file {source} ({size} bytes)"
            )
        destination = evidence_dir / destination_name
        shutil.copyfile(source, destination)
        copied[destination_name] = {
            "source": relative_source.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    return copied


def _render_loss_curve(
    rows: Sequence[Mapping[str, str]], best_epoch: int, destination: Path
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    dev_losses = [float(row["dev_loss"]) for row in rows]
    best_index = epochs.index(best_epoch)

    fig, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    axis.plot(epochs, train_losses, label="Training loss", color="#0068b5", linewidth=2)
    axis.plot(epochs, dev_losses, label="Development loss", color="#d14900", linewidth=2)
    axis.scatter(
        [best_epoch],
        [dev_losses[best_index]],
        color="#00843d",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=f"Best dev (epoch {best_epoch})",
    )
    axis.set_title("Training and development loss by epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(
        destination,
        dpi=180,
        metadata={"Software": "Thesis_ML prepare_training_report.py"},
    )
    plt.close(fig)


def _git_visible(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            relative = path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=ROOT,
            check=False,
        )
        if result.returncode == 0:
            raise ReportPreparationError(
                f"generated report file is still ignored by Git: {relative.as_posix()}"
            )
        if result.returncode != 1:
            raise ReportPreparationError(
                f"git check-ignore failed for {relative.as_posix()}"
            )


def prepare_report(
    run_directory: str | Path,
    *,
    output_root: str | Path = DEFAULT_REPORT_ROOT,
    report_id: str | None = None,
    architecture_path: str | Path = ARCHITECTURE_PATH,
) -> Path:
    """Validate one finished run and write its compact report evidence bundle."""

    run_dir = Path(run_directory).resolve()
    if not run_dir.is_dir():
        raise ReportPreparationError(f"training run directory does not exist: {run_dir}")

    metadata_path = run_dir / EVIDENCE_SOURCES["finished_metadata.json"]
    config_path = run_dir / EVIDENCE_SOURCES["config.json"]
    epoch_path = run_dir / EVIDENCE_SOURCES["epoch_metrics.csv"]
    metadata = _read_json(metadata_path)
    config = _read_json(config_path)
    stop_reason = str(metadata.get("stop_reason", ""))
    if stop_reason not in FINISHED_STOP_REASONS:
        raise ReportPreparationError(
            f"run is not a publishable finish: stop_reason={stop_reason!r}"
        )

    rows, anomalies = _read_epoch_metrics(epoch_path)
    first = rows[0]
    final = rows[-1]
    best = min(rows, key=lambda row: float(row["dev_loss"]))
    completed_epochs = int(metadata.get("completed_epochs", -1))
    if int(final["epoch"]) != completed_epochs:
        raise ReportPreparationError(
            "finished metadata and the final valid epoch row disagree: "
            f"{completed_epochs} != {final['epoch']}"
        )

    architecture = _architecture_reference(Path(architecture_path))
    run_identity = str(metadata.get("architecture_identity", ""))
    architecture["matches_finished_run"] = (
        architecture["architecture_identity"] == run_identity
    )
    if not architecture["matches_finished_run"]:
        raise ReportPreparationError(
            "finished run architecture identity does not match the current reference"
        )

    destination_root = Path(output_root)
    if report_id is None:
        report_id = f"{_created_date(metadata)}-{_safe_slug(run_dir.name)}"
    else:
        report_id = _safe_slug(report_id)
    report_dir = destination_root / report_id
    evidence_dir = report_dir / "evidence"
    report_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_evidence(run_dir, evidence_dir)

    best_epoch = int(best["epoch"])
    curve_path = report_dir / "TRAINING_CURVES.png"
    _render_loss_curve(rows, best_epoch, curve_path)
    if curve_path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
        raise ReportPreparationError("rendered training curve exceeds the bundle limit")

    best_checkpoint = run_dir / "checkpoints" / "best" / f"epoch-{best_epoch:04d}.pt"
    finished_weights: dict[str, dict[str, Any]] = {}
    weights = metadata.get("weights", {})
    if isinstance(weights, Mapping):
        for weight_kind, filename in sorted(weights.items()):
            weight_path = run_dir / "checkpoints" / "finished" / str(filename)
            finished_weights[str(weight_kind)] = {
                "filename": str(filename),
                "exists_locally": weight_path.is_file(),
                "bytes": weight_path.stat().st_size if weight_path.is_file() else None,
                "included_in_git_report": False,
            }

    first_train = float(first["train_loss"])
    final_train = float(final["train_loss"])
    first_dev = float(first["dev_loss"])
    best_dev = float(best["dev_loss"])
    final_dev = float(final["dev_loss"])
    try:
        source_run = run_dir.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        source_run = run_dir.name

    facts = {
        "schema_version": 1,
        "report_id": report_id,
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_directory": source_run,
        "training_source_commit": None,
        "training_source_commit_note": (
            "This run did not record a source commit; do not infer it from the "
            "repository revision used to prepare this report."
        ),
        "run_completion": {
            "stop_reason": stop_reason,
            "configured_epochs": int(metadata.get("configured_epochs", -1)),
            "completed_epochs": completed_epochs,
            "global_step": int(metadata.get("global_step", -1)),
            "created_utc": metadata.get("created_utc"),
        },
        "model": {
            "architecture_identity": run_identity,
            "diffusion_process": metadata.get("diffusion_process"),
            "self_conditioning": metadata.get("self_conditioning"),
            "vocab_size": metadata.get("vocab_size"),
            "default_serving_weights": metadata.get("default_serving_weights"),
            "feature_statistics_identity": metadata.get(
                "feature_statistics_identity"
            ),
            "architecture_reference": architecture,
        },
        "configuration": {
            "batch_size": _nested(config, "pipeline", "batch_size"),
            "accumulation_steps": _nested(config, "train", "accumulation_steps"),
            "effective_windows_per_optimizer_step": (
                int(_nested(config, "pipeline", "batch_size", default=0))
                * int(_nested(config, "train", "accumulation_steps", default=0))
            ),
            "learning_rate": _nested(config, "train", "lr"),
            "lr_schedule": _nested(config, "train", "lr_schedule"),
            "warmup_optimizer_steps": _nested(config, "train", "warmup"),
            "early_stopping_patience_epochs": _nested(
                config, "train", "early_stopping_patience_epochs"
            ),
            "precision": _nested(config, "train", "precision"),
            "model_d_model": _nested(config, "model", "d_model"),
            "model_layers": _nested(config, "model", "layers"),
            "model_heads": _nested(config, "model", "heads"),
            "model_ffn": _nested(config, "model", "ffn"),
        },
        "epoch_metrics": {
            "valid_epoch_rows": len(rows),
            "anomalous_rows_excluded": anomalies,
            "wall_clock": _wall_clock_summary(rows),
            "first": _metric_snapshot(first),
            "best_dev": _metric_snapshot(best),
            "final": _metric_snapshot(final),
            "derived": {
                "train_loss_reduction_first_to_final": first_train - final_train,
                "train_loss_reduction_percent_first_to_final": (
                    100.0 * (first_train - final_train) / first_train
                ),
                "dev_loss_reduction_first_to_best": first_dev - best_dev,
                "dev_loss_reduction_percent_first_to_best": (
                    100.0 * (first_dev - best_dev) / first_dev
                ),
                "dev_loss_increase_best_to_final": final_dev - best_dev,
                "best_dev_epoch": best_epoch,
            },
        },
        "checkpoint_observations": {
            "best_checkpoint_filename": best_checkpoint.name,
            "best_checkpoint_exists_locally": best_checkpoint.is_file(),
            "best_checkpoint_included_in_git_report": False,
            "finished_weights": finished_weights,
        },
        "published_evidence": copied,
        "excluded_runtime_artifacts": [
            "checkpoint tensors (*.pt, *.safetensors)",
            "resume and durable checkpoint state",
            "step_metrics.jsonl",
            "console.log",
            "pipeline caches",
        ],
    }
    facts_path = report_dir / "RUN_FACTS.json"
    facts_path.write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated_paths = [facts_path, curve_path, *evidence_dir.iterdir()]
    for generated_path in generated_paths:
        if generated_path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
            raise ReportPreparationError(
                f"report bundle file exceeds limit: {generated_path}"
            )
    _git_visible(generated_paths)
    return report_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compact tracked evidence bundle for a finished run."
    )
    parser.add_argument("run_directory", help="Finished local training-run directory")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_REPORT_ROOT),
        help="Tracked training-report root",
    )
    parser.add_argument("--report-id", help="Optional stable output directory name")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = prepare_report(
        args.run_directory,
        output_root=args.output_root,
        report_id=args.report_id,
    )
    try:
        display_path = report_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display_path = str(report_dir)
    print(f"report_dir={display_path}")
    print(f"facts={display_path}/RUN_FACTS.json")
    print(f"summary_target={display_path}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
