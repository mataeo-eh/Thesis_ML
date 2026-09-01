"""Restartable, smallest-to-largest driver for the capacity ablation suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from thesis_ml.config import ProjectConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "output" / "SizeAblationTest"
STATUS_JSON = OUTPUT_ROOT / "SWEEP_STATUS.json"
STATUS_MD = OUTPUT_ROOT / "SWEEP_STATUS.md"


@dataclass(frozen=True)
class Arm:
    name: str
    config_path: str
    expected_parameters: int


# Order is a scientific and operational contract: do not reorder by runtime or
# availability. A failed/interrupted arm stops the driver, so a restart returns
# to it before any larger model is attempted.
ARMS = (
    Arm("005m", "configs/size_ablation_005m.yaml", 4_750_016),
    Arm("015m", "configs/size_ablation_015m.yaml", 15_078_080),
    Arm("030m-baseline", "configs/size_ablation_030m_baseline.yaml", 29_318_720),
    Arm("030m-deep", "configs/size_ablation_030m_deep.yaml", 30_213_440),
    Arm("060m", "configs/size_ablation_060m.yaml", 59_359_296),
    Arm("120m", "configs/size_ablation_120m.yaml", 121_465_152),
)


def _checkpoint_root(config: ProjectConfig) -> Path:
    path = Path(config.storage.checkpoint_uri)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _inspect_arm(arm: Arm, config: ProjectConfig) -> dict[str, Any]:
    checkpoint_root = _checkpoint_root(config)
    finished_dir = checkpoint_root / "finished"
    metadata_path = finished_dir / "finished_metadata.json"
    resume_path = checkpoint_root / config.train.resume_checkpoint_subdir / "last.pt"
    state: dict[str, Any] = {
        "name": arm.name,
        "config": arm.config_path,
        "expected_parameters": arm.expected_parameters,
        "state": "pending",
        "note": "not started",
    }
    if finished_dir.exists():
        if not metadata_path.is_file():
            return {
                **state,
                "state": "blocked",
                "note": f"finished directory exists without readable metadata: {metadata_path}",
            }
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_config = json.loads((finished_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {**state, "state": "blocked", "note": f"unreadable finished export: {exc}"}
        try:
            weights = metadata["weights"]
            expected_model = asdict(config.model)
            observed_model = saved_config["model"]
            saved_train = saved_config["train"]
            required_artifacts = (
                weights["raw"],
                weights["ema"],
                metadata["torch_bundle"],
                metadata["config_file"],
            )
            valid = all(
                (
                    metadata.get("stop_reason") == "completed_all_epochs",
                    int(metadata["completed_epochs"]) >= config.train.epochs,
                    int(metadata["configured_epochs"]) == config.train.epochs,
                    observed_model == expected_model,
                    saved_train.get("schedule_horizon_epochs")
                    == config.train.schedule_horizon_epochs,
                    all(
                        isinstance(name, str) and (finished_dir / name).is_file()
                        for name in required_artifacts
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            return {
                **state,
                "state": "blocked",
                "note": "finished export does not match the current arm contract; refusing to archive/retrain it",
            }
        return {
            **state,
            "state": "done",
            "note": f"completed {metadata['completed_epochs']}/{config.train.epochs} epochs",
        }
    if resume_path.exists():
        return {
            **state,
            "state": "resume",
            "note": f"resume checkpoint present: {resume_path}",
        }
    return state


def _validate_arm_contract(arm: Arm, config: ProjectConfig) -> None:
    if config.train.epochs != 3 or config.train.schedule_horizon_epochs != 50:
        raise ValueError(f"{arm.name}: expected 3 run epochs on a 50-epoch schedule horizon")
    if config.train.max_steps != 0:
        raise ValueError(f"{arm.name}: train.max_steps must remain 0")
    if config.model.d_model // config.model.heads != 64:
        raise ValueError(f"{arm.name}: attention head dimension must be 64")
    nominal_windows = config.pipeline.batch_size * config.train.accumulation_steps
    if nominal_windows < 42:
        raise ValueError(f"{arm.name}: nominal effective batch fell below 42 windows")


def _write_status(states: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "order": [arm.name for arm in ARMS],
        "arms": states,
    }
    json_tmp = STATUS_JSON.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    json_tmp.replace(STATUS_JSON)
    lines = [
        "# Size ablation status",
        "",
        f"Last updated: {payload['updated']}",
        "",
        "| arm | state | parameters | note |",
        "|---|---:|---:|---|",
    ]
    for state in states:
        note = str(state["note"]).replace("|", "\\|")
        lines.append(
            f"| {state['name']} | {state['state']} | "
            f"{int(state['expected_parameters']):,} | {note} |"
        )
    md_tmp = STATUS_MD.with_suffix(".md.tmp")
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_tmp.replace(STATUS_MD)


def _run_arm(arm: Arm) -> int:
    arm_output = OUTPUT_ROOT / arm.name
    arm_output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = arm_output / f"console-{stamp}.log"
    command = [
        sys.executable,
        "-m",
        "thesis_ml.pipeline.train_pipeline",
        "--config",
        arm.config_path,
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"launching {arm.name}: {' '.join(command)}", flush=True)
    print(f"console log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configs and print restart decisions without launching training",
    )
    parser.add_argument(
        "--only",
        choices=[arm.name for arm in ARMS],
        help="operate on one named arm (intended for bounded maintenance/verification)",
    )
    args = parser.parse_args(argv)

    selected = [arm for arm in ARMS if args.only is None or arm.name == args.only]
    configs: dict[str, ProjectConfig] = {}
    states: list[dict[str, Any]] = []
    for arm in ARMS:
        config = load_config(PROJECT_ROOT / arm.config_path)
        _validate_arm_contract(arm, config)
        configs[arm.name] = config
        states.append(_inspect_arm(arm, config))
    _write_status(states)

    for state in states:
        print(
            f"{state['name']:>13}  {state['state']:<7}  "
            f"{int(state['expected_parameters']):>11,} params  {state['note']}",
            flush=True,
        )
    if args.dry_run:
        return 0

    for arm in selected:
        state_index = next(index for index, item in enumerate(states) if item["name"] == arm.name)
        state = states[state_index]
        if state["state"] == "done":
            continue
        if state["state"] == "blocked":
            print(f"blocked at {arm.name}: {state['note']}", file=sys.stderr, flush=True)
            return 2
        states[state_index] = {**state, "state": "running", "note": state["note"]}
        _write_status(states)
        exit_code = _run_arm(arm)
        if exit_code != 0:
            states[state_index] = {
                **states[state_index],
                "state": "failed",
                "note": f"trainer exited {exit_code}; rerun resumes this arm before larger arms",
            }
            _write_status(states)
            return exit_code
        refreshed = _inspect_arm(arm, configs[arm.name])
        states[state_index] = refreshed
        _write_status(states)
        if refreshed["state"] != "done":
            print(
                f"{arm.name} returned successfully but has no valid finished export",
                file=sys.stderr,
                flush=True,
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
