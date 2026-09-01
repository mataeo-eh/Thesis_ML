"""Run every inference test in ``Test_Scripts/`` against one trained checkpoint.

Role in the larger system
-------------------------
This is the single entry point for "I finished a training run -- now show me how
the model actually behaves on replays it has never seen." It replaces hunting
down and hand-invoking the individual diagnostics scattered across ``src/``,
``scripts/``, and ``tests/``.

What it does
------------
1. Resolves the checkpoint, the run profile, and the run's recorded replay
   selection, then derives the HELD-OUT TEST SPLIT and cross-checks it against
   what the run actually recorded (see ``inference_test_api.resolve_test_split_replays``).
2. Creates one run directory under ``output/`` named
   ``<model label>__<human-readable date>`` -- e.g.
   ``smallTrainingTestV3-epoch-0033__2026-Aug-26_02-41PM``.
3. Discovers every ``test_*.py`` module in ``Test_Scripts/``, in filename order,
   and runs each one with its own subdirectory inside that run directory.
4. Writes a run-level ``summary.json``, a human-readable ``SUMMARY.md``, and a
   per-test console log.

The checkpoint is loaded ONCE and shared by every test (see
``inference_test_api.SharedResources``), so running six tests costs one model
load, not six.

Nothing outside the run's own output directory is written. No checkpoint,
config, manifest, or source replay is ever modified.

Usage
-----
    .venv\\Scripts\\python.exe Model_Inference_Tests\\run_inference_tests.py \\
        --checkpoint tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt \\
        --config configs/smallTrainingTestV3.yaml

    # only some tests, by substring
    ... --only build_order --only cross_entropy

    # see what would run, load nothing
    ... --list

Depends on: ``inference_test_api`` (shared context) and every module under
``Test_Scripts/``.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import io
from pathlib import Path
import re
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Sequence

# Three directories have to be importable before anything else is touched,
# because this script runs as a plain file (not `python -m`):
#   src/         -> the `thesis_ml` package
#   <repo root>/ -> the `scripts` package (one test wraps a script from there)
#   this dir     -> `inference_test_api`, which the test modules import by name
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
for path_entry in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PACKAGE_DIR)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

import torch  # noqa: E402 -- import after sys.path is prepared

from inference_test_api import (  # noqa: E402
    SharedResources,
    TestContext,
    TestResult,
    portable_path,
    write_json,
    write_text,
)


TEST_SCRIPTS_DIR = PACKAGE_DIR / "Test_Scripts"
OUTPUT_DIR = PACKAGE_DIR / "output"

# Default run profile and checkpoint: the canonical full-corpus V3 run and its
# best-dev-epoch checkpoint. Both are overridable on the command line; they are
# defaults rather than hardcoded paths so the common case is a one-flag command.
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smallTrainingTestV3.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "tests" / "output" / "smallTrainingTestV3" / "checkpoints" / "best" / "epoch-0033.pt"
)


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredTest:
    """One importable test module plus the metadata it declares.

    Every module under ``Test_Scripts/`` named ``test_*.py`` must define:

      * ``TEST_NAME``          -- short slug, also its output subdirectory name
      * ``TEST_TITLE``         -- one-line human title
      * ``TEST_DESCRIPTION``   -- what question the test answers
      * ``TEST_OUTPUTS``       -- sequence of ``"filename -- what it is"`` strings
      * ``USES_MODEL``         -- False for data-only baselines
      * ``REQUIRES_DEBUT_FINETUNE`` -- True when the test is only meaningful for a
        debut/build-order FINE-TUNED checkpoint. The runner SKIPS such a test on
        a pre-training checkpoint rather than emitting a meaningless number.
      * ``run(context) -> TestResult``
    """

    module: ModuleType
    path: Path
    name: str
    title: str
    description: str
    outputs: tuple[str, ...]
    uses_model: bool
    requires_debut_finetune: bool


def discover_tests(directory: Path) -> list[DiscoveredTest]:
    """Import every ``test_*.py`` module in ``directory``, in filename order.

    Filename order is the run order, which is why the shipped tests are prefixed
    ``test_01_``, ``test_02_``, ... -- cheap, broad tests run before expensive,
    narrow ones so a broken setup fails fast.

    Raises:
        AttributeError: a module is missing one of the required metadata names.
    """

    discovered: list[DiscoveredTest] = []
    for path in sorted(directory.glob("test_*.py")):
        module = _import_module_from_path(path)
        missing = [
            name
            for name in ("TEST_NAME", "TEST_TITLE", "TEST_DESCRIPTION", "TEST_OUTPUTS", "run")
            if not hasattr(module, name)
        ]
        if missing:
            raise AttributeError(f"{path.name} is missing required attribute(s): {', '.join(missing)}")
        discovered.append(
            DiscoveredTest(
                module=module,
                path=path,
                name=str(module.TEST_NAME),
                title=str(module.TEST_TITLE),
                description=str(module.TEST_DESCRIPTION),
                outputs=tuple(module.TEST_OUTPUTS),
                uses_model=bool(getattr(module, "USES_MODEL", True)),
                requires_debut_finetune=bool(getattr(module, "REQUIRES_DEBUT_FINETUNE", False)),
            )
        )
    return discovered


def _import_module_from_path(path: Path) -> ModuleType:
    """Import a single .py file as a module without requiring a package."""

    spec = importlib.util.spec_from_file_location(f"inference_tests.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import test module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Run directory naming
# ---------------------------------------------------------------------------


def build_model_label(checkpoint_path: Path, config_path: Path) -> str:
    """Name the weights under test from their location on disk.

    The run directory has to be identifiable at a glance months later, so the
    label combines the RUN name with the CHECKPOINT name:

        tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt
            -> "smallTrainingTestV3-epoch-0033"

    The run name is taken from the checkpoint path's own run directory when one
    is recognisable (``<run>/checkpoints/<subdir>/<file>.pt``); otherwise it
    falls back to the config file's stem, which always exists.

    Parameters:
        checkpoint_path: the ``.pt`` file being evaluated.
        config_path: the run profile YAML, used as the fallback run name.

    Returns:
        A filesystem-safe label with no spaces.
    """

    parts = [part for part in checkpoint_path.resolve().parts]
    run_name = config_path.stem
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        if index > 0:
            run_name = parts[index - 1]
    label = f"{run_name}-{checkpoint_path.stem}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label)


def build_run_directory_name(model_label: str, moment: datetime) -> str:
    """Compose ``<model label>__<human-readable date>``.

    The date format is deliberately readable at a glance and still sorts
    sensibly within a given month: ``2026-Aug-26_02-41PM``. Colons and spaces are
    avoided because Windows rejects them in directory names.
    """

    return f"{model_label}__{moment.strftime('%Y-%b-%d_%I-%M%p')}"


def allocate_run_directory(out_root: Path, model_label: str, moment: datetime) -> Path:
    """Create the run directory, adding a suffix rather than merging into an old one.

    The name is minute-resolution, so two invocations in the same minute (a
    common thing while iterating with ``--only``) would otherwise land in the
    same folder and silently overwrite each other's ``SUMMARY.md``. When the
    intended name is already taken, ``-2``, ``-3``, ... is appended.

    Returns:
        The created, empty-or-new run directory.
    """

    base_name = build_run_directory_name(model_label, moment)
    candidate = out_root / base_name
    attempt = 2
    while candidate.exists():
        candidate = out_root / f"{base_name}-{attempt}"
        attempt += 1
    candidate.mkdir(parents=True)
    return candidate


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class TestOutcome:
    """The runner's record of one test's execution."""

    name: str
    title: str
    status: str  # "ok" | "skipped" | "failed"
    duration_seconds: float
    detail: str = ""
    headline: list[str] | None = None
    artifacts: list[str] | None = None
    metrics: dict[str, Any] | None = None


def run_one_test(
    test: DiscoveredTest,
    context: TestContext,
    *,
    log_path: Path,
) -> TestOutcome:
    """Execute one test, capturing its console output and any failure.

    A failing test never aborts the run: its traceback is written to its own log
    and to the outcome, and the remaining tests still execute. Coming back from a
    long GPU run with five results and one traceback beats coming back with one
    traceback.

    Parameters:
        test: the discovered module.
        context: the per-test context (its ``out_dir`` already exists).
        log_path: file receiving everything the test printed.

    Returns:
        A :class:`TestOutcome` describing what happened.
    """

    started = time.perf_counter()
    buffer = io.StringIO()
    status = "ok"
    detail = ""
    result: TestResult | None = None
    try:
        # Tee: the test's own prints go to the log file, while the runner keeps
        # the console readable with just its own progress lines.
        with redirect_stdout(buffer):
            result = test.module.run(context)
        if not isinstance(result, TestResult):
            raise TypeError(f"{test.name}.run() returned {type(result).__name__}, expected TestResult")
    except Exception as error:  # noqa: BLE001 -- one bad test must not kill the run
        status = "failed"
        detail = f"{type(error).__name__}: {error}"
        buffer.write("\n" + traceback.format_exc())

    duration = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(buffer.getvalue(), encoding="utf-8")

    return TestOutcome(
        name=test.name,
        title=test.title,
        status=status,
        duration_seconds=duration,
        detail=detail,
        headline=list(result.headline) if result is not None else [],
        artifacts=[portable_path(path) for path in result.artifacts] if result is not None else [],
        metrics=dict(result.metrics) if result is not None else {},
    )


def write_run_summary(
    run_dir: Path,
    *,
    outcomes: Sequence[TestOutcome],
    provenance: dict[str, Any],
    tests: Sequence[DiscoveredTest],
) -> list[Path]:
    """Write ``summary.json`` and ``SUMMARY.md`` for the whole run.

    ``summary.json`` is the machine-readable record (provenance + every test's
    metrics). ``SUMMARY.md`` is the thing a human opens first: what ran, what it
    measured, what it produced, and the headline numbers.
    """

    summary_json = {
        "provenance": provenance,
        "tests": [
            {
                "name": outcome.name,
                "title": outcome.title,
                "status": outcome.status,
                "duration_seconds": round(outcome.duration_seconds, 3),
                "detail": outcome.detail,
                "headline": outcome.headline,
                "artifacts": outcome.artifacts,
                "metrics": outcome.metrics,
            }
            for outcome in outcomes
        ],
    }
    json_path = write_json(summary_json, run_dir / "summary.json")

    checkpoint = provenance.get("checkpoint", {})
    split = provenance.get("split", {})
    lines = [
        f"# Inference test run -- {provenance.get('model_label', 'unknown model')}",
        "",
        f"- Run started: {provenance.get('started_at')}",
        f"- Checkpoint: `{checkpoint.get('checkpoint', 'n/a')}`",
        f"- Weights: {checkpoint.get('weights', 'n/a')} "
        f"(epoch {checkpoint.get('completed_epochs', '?')}, step {checkpoint.get('global_step', '?')})",
        f"- Architecture identity: `{checkpoint.get('architecture_identity', 'n/a')}`",
        f"- Config: `{provenance.get('config')}`",
        f"- Split: **test** ({split.get('n_replays_in_split', '?')} held-out replays, "
        f"source `{split.get('source')}`, "
        f"verified against the run's recorded selection: "
        f"{split.get('verified_against_recorded_selection')})",
        f"- Device: {provenance.get('device')}  |  fog rate: {provenance.get('fog_rate')}  "
        f"|  seed: {provenance.get('seed')}",
        "",
        "## Results",
        "",
        "| test | status | seconds | headline |",
        "|---|---|---|---|",
    ]
    for outcome in outcomes:
        headline = "; ".join(outcome.headline or []) or outcome.detail or "-"
        headline = headline.replace("|", "\\|")
        lines.append(
            f"| `{outcome.name}` | {outcome.status} | {outcome.duration_seconds:.1f} | {headline} |"
        )

    lines += ["", "## What each test measures", ""]
    by_name = {test.name: test for test in tests}
    for outcome in outcomes:
        test = by_name.get(outcome.name)
        if test is None:
            continue
        lines += [f"### `{test.name}` -- {test.title}", "", test.description.strip(), ""]
        if test.outputs:
            lines.append("Outputs:")
            lines += [f"- {item}" for item in test.outputs]
            lines.append("")

    md_path = write_text(lines, run_dir / "SUMMARY.md")
    return [json_path, md_path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every test in Test_Scripts/ against one trained checkpoint, on the "
            "run's held-out TEST replay split, writing results under output/."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint .pt written by the training loop (default: smallTrainingTestV3 best epoch-0033)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="run profile YAML the checkpoint was trained under (default: configs/smallTrainingTestV3.yaml)",
    )
    parser.add_argument(
        "--replay-selection",
        type=Path,
        default=None,
        help=(
            "the run's recorded replay_selection.json, used to VERIFY the derived "
            "test split. Default: <run>/metrics/replay_selection.json inferred from "
            "the checkpoint path, when it exists"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUTPUT_DIR,
        help="root for run directories (default: Model_Inference_Tests/output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for every model-bound test",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "score the RAW (final optimizer step) weights. The default is the EMA "
            "weights, which is what the sampler and reported eval metrics serve"
        ),
    )
    parser.add_argument(
        "--n-replays",
        type=int,
        default=8,
        help="how many held-out replays to draw windows from (0 = all in the test split)",
    )
    parser.add_argument(
        "--n-windows-per-replay",
        type=int,
        default=2,
        help="windows per selected replay (0 = all)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help=(
            "global cap on scored windows, applied by even striding across the "
            "selection (0 = no cap). Individual tests apply their own tighter caps "
            "where full iterative sampling makes a large sample impractical"
        ),
    )
    parser.add_argument(
        "--fog-rate",
        type=float,
        default=None,
        help="fixed fog rate for every served example; default = config.eval.fog_rate",
    )
    parser.add_argument("--seed", type=int, default=20260826, help="base seed for every stochastic draw")
    parser.add_argument("--dpi", type=int, default=150, help="raster resolution for PNG figures")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="run only tests whose name contains this substring; repeatable",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="skip tests whose name contains this substring; repeatable",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="free-form option passed through to every test; repeatable",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the discovered tests and exit without loading the checkpoint",
    )
    return parser.parse_args(argv)


def _infer_replay_selection_path(checkpoint_path: Path) -> Path | None:
    """Locate ``<run>/metrics/replay_selection.json`` from a checkpoint path.

    The training pipeline writes the file next to the run's metrics, one level
    up from the checkpoint tree. Returns None when it is not where it should be,
    in which case the split falls back to the config rule alone (unverified).
    """

    parts = checkpoint_path.resolve().parts
    if "checkpoints" not in parts:
        return None
    run_dir = Path(*parts[: parts.index("checkpoints")])
    candidate = run_dir / "metrics" / "replay_selection.json"
    return candidate if candidate.exists() else None


def _parse_options(raw_options: Sequence[str] | None) -> dict[str, str]:
    """Turn ``["a=1", "b=2"]`` into ``{"a": "1", "b": "2"}``."""

    options: dict[str, str] = {}
    for item in raw_options or []:
        if "=" not in item:
            raise SystemExit(f"--option expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        options[name.strip()] = value.strip()
    return options


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    tests = discover_tests(TEST_SCRIPTS_DIR)
    if not tests:
        raise SystemExit(f"no test_*.py modules found in {TEST_SCRIPTS_DIR}")

    if args.only:
        tests = [test for test in tests if any(needle in test.name for needle in args.only)]
    if args.skip:
        tests = [test for test in tests if not any(needle in test.name for needle in args.skip)]
    if not tests:
        raise SystemExit("every discovered test was filtered out by --only/--skip")

    if args.list:
        print(f"{len(tests)} test(s) in {portable_path(TEST_SCRIPTS_DIR)}:\n")
        for test in tests:
            flags = []
            if not test.uses_model:
                flags.append("data-only (no model)")
            if test.requires_debut_finetune:
                flags.append("requires a debut fine-tuned checkpoint")
            suffix = f"   [{'; '.join(flags)}]" if flags else ""
            print(f"  {test.name}  --  {test.title}{suffix}")
            print(f"      {test.description.strip().splitlines()[0]}")
            for item in test.outputs:
                print(f"      * {item}")
            print()
        return 0

    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    replay_selection = args.replay_selection
    if replay_selection is None:
        replay_selection = _infer_replay_selection_path(checkpoint_path)

    from thesis_ml.config import load_config  # local import: after sys.path setup

    user_config = load_config(config_path)
    fog_rate = user_config.eval.fog_rate if args.fog_rate is None else args.fog_rate

    shared = SharedResources(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=torch.device(args.device),
        use_raw_weights=args.raw,
        fog_rate=fog_rate,
        seed=args.seed,
        replay_selection_path=replay_selection,
    )

    model_label = build_model_label(checkpoint_path, config_path)
    started_at = datetime.now()
    run_dir = allocate_run_directory(Path(args.out_root), model_label, started_at)

    split = shared.test_split()
    print(f"model      : {model_label}")
    print(f"checkpoint : {portable_path(checkpoint_path)}")
    print(f"config     : {portable_path(config_path)}")
    print(
        f"test split : {len(split.replay_ids)} replays "
        f"(source={split.source}, verified={split.verified_against_recorded})"
    )
    print(f"device     : {args.device}   fog_rate={fog_rate}   seed={args.seed}")
    print(f"output     : {portable_path(run_dir)}")
    print()

    # Debut-mode gate. A checkpoint trained WITHOUT debut targets cannot be
    # meaningfully scored by a debut/build-order-timeline test; those tests are
    # skipped with a stated reason instead of quietly producing noise.
    checkpoint_is_debut = bool(shared.checkpoint_facts().get("debut_mode"))

    options = _parse_options(args.option)
    outcomes: list[TestOutcome] = []
    for index, test in enumerate(tests, start=1):
        print(f"[{index}/{len(tests)}] {test.name} -- {test.title}", flush=True)
        if test.requires_debut_finetune and not checkpoint_is_debut:
            reason = (
                "requires a debut/build-order fine-tuned checkpoint; this one was "
                "trained with data.debut_mode=false (pre-training objective)"
            )
            print(f"        SKIPPED: {reason}\n", flush=True)
            outcomes.append(
                TestOutcome(
                    name=test.name,
                    title=test.title,
                    status="skipped",
                    duration_seconds=0.0,
                    detail=reason,
                )
            )
            continue

        test_dir = run_dir / test.name
        test_dir.mkdir(parents=True, exist_ok=True)
        context = TestContext(
            shared=shared,
            out_dir=test_dir,
            run_dir=run_dir,
            model_label=model_label,
            device=torch.device(args.device),
            seed=args.seed,
            n_replays=args.n_replays,
            n_windows_per_replay=args.n_windows_per_replay,
            max_examples=args.max_examples,
            dpi=args.dpi,
            fog_rate=fog_rate,
            extra=options,
        )
        outcome = run_one_test(test, context, log_path=test_dir / "console.log")
        outcomes.append(outcome)
        marker = {"ok": "done", "failed": "FAILED", "skipped": "skipped"}[outcome.status]
        print(f"        {marker} in {outcome.duration_seconds:.1f}s")
        for line in outcome.headline or []:
            print(f"        {line}")
        if outcome.detail and outcome.status == "failed":
            print(f"        {outcome.detail}")
            print(f"        traceback: {portable_path(test_dir / 'console.log')}")
        print(flush=True)

    provenance = {
        "model_label": model_label,
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "config": portable_path(config_path),
        "checkpoint": shared.checkpoint_facts(),
        "split": {
            "name": "test",
            "source": split.source,
            "verified_against_recorded_selection": split.verified_against_recorded,
            "n_replays_in_split": len(split.replay_ids),
            "replay_ids": list(split.replay_ids),
        },
        "device": str(args.device),
        "fog_rate": fog_rate,
        "seed": args.seed,
        "sampling_budget": {
            "n_replays": args.n_replays,
            "n_windows_per_replay": args.n_windows_per_replay,
            "max_examples": args.max_examples,
        },
    }
    summary_paths = write_run_summary(
        run_dir, outcomes=outcomes, provenance=provenance, tests=tests
    )

    failed = [outcome.name for outcome in outcomes if outcome.status == "failed"]
    print("=" * 72)
    print(f"wrote {portable_path(summary_paths[1])}")
    print(f"      {portable_path(summary_paths[0])}")
    ok_count = sum(1 for outcome in outcomes if outcome.status == "ok")
    skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    print(f"{ok_count} ok, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
