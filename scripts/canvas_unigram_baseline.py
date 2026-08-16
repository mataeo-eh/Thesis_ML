"""Compute constant-predictor baselines for canvas targets without a model.

The script serves manifest windows through ``SC2DiffusionDataset`` and
``collate_diffusion_examples`` so its scored positions are exactly the live
``canvas_loss_mask`` positions.  It then computes:

* the empirical unigram entropy H(p); and
* the best constant predictor for the live weighted canvas objective.

No checkpoint, model forward pass, or GPU is used.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import numpy as np

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import (
    WindowManifestEntry,
    load_window_manifest,
    read_manifest_metadata,
)
from thesis_ml.model.loss import CanvasCrossEntropyLoss, active_class_id_to_name
from thesis_ml.pipeline.storage import StorageResolver
from thesis_ml.pipeline.train_pipeline import (
    _ensure_window_manifest,
    _explicit_replay_selection,
    _make_dataloader,
    _materialize_file,
    _materialize_replay_paths,
    _select_replays,
    _shutdown_dataloader,
)
from thesis_ml.train.corruption import uniform_noise_support_size
from thesis_ml.vocab.content_vocab import load_content_vocabulary


DEFAULT_OUTPUT_DIR = Path("scripts/output/canvas_unigram_baseline")


class UnigramAccumulator:
    """Fixed-memory target histograms accumulated from collated CPU batches."""

    def __init__(
        self,
        *,
        vocab_size: int,
        canvas_length: int,
        class_ids: Sequence[int],
        include_position_conditional: bool = True,
    ) -> None:
        if vocab_size <= 0 or canvas_length <= 0:
            raise ValueError("vocab_size and canvas_length must be positive")
        if not class_ids:
            raise ValueError("class_ids must not be empty")
        self.vocab_size = vocab_size
        self.canvas_length = canvas_length
        self.class_ids = tuple(sorted(class_ids))
        self._class_row = {
            class_id: row for row, class_id in enumerate(self.class_ids)
        }
        self.token_counts = np.zeros(vocab_size, dtype=np.int64)
        self.class_token_counts = np.zeros(
            (len(self.class_ids), vocab_size), dtype=np.int64
        )
        self.position_token_counts = (
            np.zeros((canvas_length, vocab_size), dtype=np.int64)
            if include_position_conditional
            else None
        )
        self.window_count = 0

    def update(self, batch: DiffusionBatch) -> None:
        """Add exactly the positions selected by ``batch.canvas_loss_mask``."""

        targets = batch.target_canvas.detach().cpu().numpy()
        labels = batch.class_labels.detach().cpu().numpy()
        scored = batch.canvas_loss_mask.detach().cpu().numpy().astype(bool)
        if targets.shape != labels.shape or targets.shape != scored.shape:
            raise ValueError("target, class-label, and scored-mask shapes must match")
        if targets.shape[1] > self.canvas_length:
            raise ValueError(
                f"batch canvas width {targets.shape[1]} exceeds configured "
                f"canvas length {self.canvas_length}"
            )

        active_targets = targets[scored]
        active_labels = labels[scored]
        if active_targets.size:
            if active_targets.min() < 0 or active_targets.max() >= self.vocab_size:
                raise ValueError("a scored target token falls outside the vocabulary")
            unknown = sorted(set(active_labels.tolist()) - set(self.class_ids))
            if unknown:
                raise ValueError(f"scored positions contain unknown class ids: {unknown}")
            self.token_counts += np.bincount(
                active_targets, minlength=self.vocab_size
            ).astype(np.int64, copy=False)
            for class_id, row in self._class_row.items():
                class_targets = active_targets[active_labels == class_id]
                if class_targets.size:
                    self.class_token_counts[row] += np.bincount(
                        class_targets, minlength=self.vocab_size
                    ).astype(np.int64, copy=False)

            if self.position_token_counts is not None:
                batch_rows, positions = np.nonzero(scored)
                del batch_rows
                np.add.at(
                    self.position_token_counts,
                    (positions, active_targets),
                    1,
                )
        self.window_count += int(targets.shape[0])


def summarize_counts(
    accumulator: UnigramAccumulator,
    *,
    class_id_to_name: dict[int, str],
    class_weights: Sequence[float],
) -> dict[str, object]:
    """Derive unweighted and live-weighted constant-predictor baselines."""

    counts = accumulator.token_counts.astype(np.float64)
    total_positions = int(counts.sum())
    if total_positions == 0:
        raise ValueError("no scored canvas positions were selected")

    class_weights_array = np.asarray(class_weights, dtype=np.float64)
    if class_weights_array.ndim != 1:
        raise ValueError("class_weights must be one-dimensional")
    weighted_token_counts = np.zeros(accumulator.vocab_size, dtype=np.float64)
    total_weight = 0.0
    for row, class_id in enumerate(accumulator.class_ids):
        if class_id >= len(class_weights_array):
            raise ValueError(f"class weight missing for class id {class_id}")
        weight = float(class_weights_array[class_id])
        weighted_token_counts += accumulator.class_token_counts[row] * weight
        total_weight += float(accumulator.class_token_counts[row].sum()) * weight
    if total_weight <= 0.0:
        raise ValueError("the live class weights assign zero mass to every position")

    marginal = counts / total_positions
    weighted_optimal = weighted_token_counts / total_weight
    unweighted_entropy = _entropy_from_counts(counts)
    weighted_optimal_ce = _cross_entropy_from_counts(
        weighted_token_counts, weighted_optimal
    )
    weighted_ce_of_unweighted_marginal = _cross_entropy_from_counts(
        weighted_token_counts, marginal
    )

    classes: list[dict[str, object]] = []
    contribution_sum = 0.0
    for row, class_id in enumerate(accumulator.class_ids):
        class_counts = accumulator.class_token_counts[row].astype(np.float64)
        class_positions = int(class_counts.sum())
        weight = float(class_weights_array[class_id])
        if class_positions:
            ce_under_unweighted = _cross_entropy_from_counts(class_counts, marginal)
            ce_under_weighted = _cross_entropy_from_counts(
                class_counts, weighted_optimal
            )
            contribution = (
                weight * class_positions / total_weight * ce_under_weighted
            )
            contribution_sum += contribution
            conditional_entropy: float | None = _entropy_from_counts(class_counts)
            unique_tokens = int(np.count_nonzero(class_counts))
        else:
            ce_under_unweighted = None
            ce_under_weighted = None
            contribution = 0.0
            conditional_entropy = None
            unique_tokens = 0
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_id_to_name[class_id],
                "class_weight": weight,
                "scored_positions": class_positions,
                "fraction_of_scored_positions": class_positions / total_positions,
                "unique_target_tokens": unique_tokens,
                "unweighted_conditional_entropy_nats": conditional_entropy,
                "unweighted_global_constant_ce_nats": ce_under_unweighted,
                "weighted_optimal_global_constant_ce_nats": ce_under_weighted,
                "weighted_objective_contribution_nats": contribution,
            }
        )

    position_entropy = None
    if accumulator.position_token_counts is not None:
        weighted_entropy_sum = 0.0
        position_total = 0
        for position_counts in accumulator.position_token_counts:
            position_count = int(position_counts.sum())
            if position_count:
                weighted_entropy_sum += (
                    position_count * _entropy_from_counts(position_counts)
                )
                position_total += position_count
        if position_total != total_positions:
            raise AssertionError("position and unigram scored-position totals diverged")
        position_entropy = weighted_entropy_sum / position_total

    if not math.isclose(
        contribution_sum, weighted_optimal_ce, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError("per-class weighted contributions do not sum to baseline")

    return {
        "overall": {
            "scored_positions": total_positions,
            "total_live_class_weight": total_weight,
            "unique_target_tokens": int(np.count_nonzero(counts)),
            "unweighted_marginal_entropy_nats": unweighted_entropy,
            "weighted_optimal_constant_ce_nats": weighted_optimal_ce,
            "weighted_ce_of_unweighted_marginal_nats": (
                weighted_ce_of_unweighted_marginal
            ),
            "position_conditional_unweighted_entropy_nats": position_entropy,
        },
        "classes": classes,
    }


def compute_baseline(
    config: ProjectConfig,
    *,
    config_path: Path,
    split_name: str,
    max_windows: int,
    manifest_filters: Sequence[str],
    dataset_epoch: int,
    num_workers: int | None,
    include_position_conditional: bool,
) -> dict[str, object]:
    """Stream the selected live dataset split and return a JSON-ready report."""

    resolver = StorageResolver()
    token_dictionary = _materialize_file(
        config.pipeline.token_dictionary_uri,
        config.storage.local_cache_dir,
        resolver,
    )
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)
    split_replay_paths = _resolve_split_replays(
        replay_paths, config=config, split_name=split_name
    )
    split_windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=split_replay_paths,
    )
    windows = _filter_windows(split_windows, manifest_filters)
    if max_windows > 0:
        windows = windows[:max_windows]
    if not windows:
        raise ValueError("the selected split/filter contains no manifest windows")

    class_id_to_name = active_class_id_to_name(config)
    criterion = CanvasCrossEntropyLoss(config)
    accumulator = UnigramAccumulator(
        vocab_size=vocabulary.vocab_size,
        canvas_length=config.data.canvas_budget_tokens,
        class_ids=tuple(class_id_to_name),
        include_position_conditional=include_position_conditional,
    )
    dataset = SC2DiffusionDataset(
        windows,
        config,
        vocabulary,
        seed=config.pipeline.seed,
        fog_rate_override=None,
    )
    dataset.set_epoch(dataset_epoch)
    effective_num_workers = (
        config.pipeline.num_workers if num_workers is None else num_workers
    )
    loader_config = replace(
        config,
        pipeline=replace(config.pipeline, num_workers=effective_num_workers),
    )
    loader = _make_dataloader(dataset, loader_config, shuffle=False, device="cpu")
    total_windows = len(dataset)
    print(
        f"selected_windows={total_windows} batch_size={config.pipeline.batch_size} "
        f"num_workers={effective_num_workers} "
        f"prefetch_factor={config.pipeline.prefetch_factor if effective_num_workers else 0}",
        file=sys.stderr,
        flush=True,
    )
    started_at = time.perf_counter()
    next_progress = min(100, total_windows)
    try:
        for batch in loader:
            accumulator.update(batch)
            processed = accumulator.window_count
            if processed >= next_progress or processed == total_windows:
                print(
                    _progress_line(
                        processed=processed,
                        total=total_windows,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += 100
                next_progress = min(next_progress, total_windows)
    finally:
        _shutdown_dataloader(loader)

    result = summarize_counts(
        accumulator,
        class_id_to_name=class_id_to_name,
        class_weights=criterion.class_weights.detach().cpu().tolist(),
    )
    uniform_support_size = uniform_noise_support_size(vocabulary.vocab_size)
    result["references"] = {
        "uniform_corruption_support_size": uniform_support_size,
        "ln_uniform_corruption_support_nats": math.log(uniform_support_size),
        "full_vocabulary_width": vocabulary.vocab_size,
        "ln_full_vocabulary_nats": math.log(vocabulary.vocab_size),
    }
    result["provenance"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_profile": config_path.stem,
        "config_path": portable_path(config_path),
        "config_sha256": _sha256(config_path),
        "split": split_name,
        "dataset_epoch": dataset_epoch,
        "manifest_filters": list(manifest_filters),
        "max_windows": max_windows,
        "split_replay_count": len(split_replay_paths),
        "split_window_count_before_filter": len(split_windows),
        "window_count": accumulator.window_count,
        "manifest_path": portable_path(Path(config.data.window_manifest_path)),
        "manifest_sha256": _sha256(Path(config.data.window_manifest_path)),
        "manifest_metadata": read_manifest_metadata(config.data.window_manifest_path),
        "token_dictionary_path": portable_path(token_dictionary),
        "token_dictionary_sha256": _sha256(token_dictionary),
        "canvas_length": config.data.canvas_budget_tokens,
        "batch_size": config.pipeline.batch_size,
        "profile_num_workers": config.pipeline.num_workers,
        "num_workers": effective_num_workers,
        "prefetch_factor": (
            config.pipeline.prefetch_factor if effective_num_workers else 0
        ),
        "vocabulary_width": vocabulary.vocab_size,
        "class_weights": {
            str(class_id): {
                "name": class_id_to_name[class_id],
                "weight": float(criterion.class_weights[class_id].item()),
            }
            for class_id in sorted(class_id_to_name)
        },
    }
    return result


def format_summary(report: dict[str, object]) -> str:
    """Render a compact, provenance-rich human summary."""

    provenance = report["provenance"]
    overall = report["overall"]
    references = report["references"]
    assert isinstance(provenance, dict)
    assert isinstance(overall, dict)
    assert isinstance(references, dict)
    lines = [
        "Canvas target unigram baseline",
        (
            f"config={provenance['config_profile']} split={provenance['split']} "
            f"windows={provenance['window_count']} "
            f"scored_positions={overall['scored_positions']} "
            f"vocab={provenance['vocabulary_width']} "
            f"workers={provenance['num_workers']}"
        ),
        f"manifest={provenance['manifest_path']} sha256={provenance['manifest_sha256']}",
        "class_weights="
        + ", ".join(
            f"{class_id}:{entry['name']}={entry['weight']:.12g}"
            for class_id, entry in provenance["class_weights"].items()
        ),
        "",
        (
            "unweighted marginal entropy H(p): "
            f"{overall['unweighted_marginal_entropy_nats']:.9f} nats"
        ),
        (
            "weighted-optimal constant CE (live objective): "
            f"{overall['weighted_optimal_constant_ce_nats']:.9f} nats"
        ),
        (
            "weighted CE of the unweighted marginal: "
            f"{overall['weighted_ce_of_unweighted_marginal_nats']:.9f} nats"
        ),
    ]
    position_value = overall["position_conditional_unweighted_entropy_nats"]
    if position_value is not None:
        lines.append(
            "position-conditional unweighted entropy: "
            f"{position_value:.9f} nats"
        )
    lines.extend(
        [
            (
                f"ln(uniform corruption support={references['uniform_corruption_support_size']}): "
                f"{references['ln_uniform_corruption_support_nats']:.9f} nats"
            ),
            (
                f"ln(full vocabulary={references['full_vocabulary_width']}): "
                f"{references['ln_full_vocabulary_nats']:.9f} nats"
            ),
            "",
            (
                "class_id  class_name         positions  H(token|class)  "
                "CE(class,q_weighted)  weighted contribution"
            ),
        ]
    )
    for entry in report["classes"]:
        entropy = entry["unweighted_conditional_entropy_nats"]
        weighted_ce = entry["weighted_optimal_global_constant_ce_nats"]
        lines.append(
            f"{entry['class_id']:>8}  {entry['class_name']:<17} "
            f"{entry['scored_positions']:>10}  {_format_optional(entropy):>14}  "
            f"{_format_optional(weighted_ce):>20}  "
            f"{entry['weighted_objective_contribution_nats']:.9f}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute data-only unigram baselines for scored canvas targets",
    )
    parser.add_argument("--config", type=Path, required=True, help="merged run profile YAML")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        default="train",
        help="config-derived replay split to scan (default: train)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="scan only the first N selected manifest windows (0 = all)",
    )
    parser.add_argument(
        "--manifest-filter",
        action="append",
        default=[],
        help=(
            "fnmatch pattern over replay_id, replay filename/stem, perspective, or "
            "'<replay_id>:<perspective>:<start>-<end>'; repeat for OR matching"
        ),
    )
    parser.add_argument(
        "--dataset-epoch",
        type=int,
        default=0,
        help="deterministic per-serving fog epoch used for class labels (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker processes; defaults to pipeline.num_workers from "
            "the selected profile, while 0 runs in the main process"
        ),
    )
    parser.add_argument(
        "--no-position-conditional",
        action="store_true",
        help="skip the optional canvas-index-conditional entropy histogram",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON artifact path; defaults below scripts/output/canvas_unigram_baseline",
    )
    args = parser.parse_args(argv)
    if args.max_windows < 0:
        parser.error("--max-windows must be non-negative")
    if args.dataset_epoch < 0:
        parser.error("--dataset-epoch must be non-negative")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    report = compute_baseline(
        config,
        config_path=config_path,
        split_name=args.split,
        max_windows=args.max_windows,
        manifest_filters=args.manifest_filter,
        dataset_epoch=args.dataset_epoch,
        num_workers=args.num_workers,
        include_position_conditional=not args.no_position_conditional,
    )
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{config_path.stem}-{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = format_summary(report)
    summary_path = output_path.with_suffix(".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"json_artifact={portable_path(output_path)}")
    print(f"summary_artifact={portable_path(summary_path)}")
    return 0


def _resolve_split_replays(
    replay_paths: list[str],
    *,
    config: ProjectConfig,
    split_name: str,
) -> list[str]:
    if split_name == "all":
        return list(replay_paths)
    explicit = _explicit_replay_selection(replay_paths, config)
    if explicit is not None:
        train_replays, dev_replays, test_replays = explicit
    else:
        split = split_replays(
            replay_paths,
            seed=config.pipeline.split_seed,
            test_fraction=config.pipeline.test_fraction,
            dev_fraction=config.pipeline.dev_fraction,
            train_count=config.pipeline.train_replay_count,
            dev_count=config.pipeline.validation_replay_count,
        )
        train_replays, dev_replays = _select_replays(
            list(split.train), list(split.dev), config
        )
        test_replays = list(split.test)
    return {
        "train": train_replays,
        "dev": dev_replays,
        "test": test_replays,
    }[split_name]


def _filter_windows(
    windows: Iterable[WindowManifestEntry], patterns: Sequence[str]
) -> list[WindowManifestEntry]:
    if not patterns:
        return list(windows)
    selected: list[WindowManifestEntry] = []
    for window in windows:
        replay_path = Path(window.replay_path)
        values = (
            window.replay_id,
            replay_path.name,
            replay_path.stem,
            window.perspective_player,
            (
                f"{window.replay_id}:{window.perspective_player}:"
                f"{window.start_timestep}-{window.end_timestep}"
            ),
        )
        if any(
            fnmatch.fnmatchcase(value, pattern)
            for pattern in patterns
            for value in values
        ):
            selected.append(window)
    return selected


def _entropy_from_counts(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if positive.size == 0:
        raise ValueError("entropy is undefined for an empty distribution")
    probabilities = positive / positive.sum()
    return float(-np.dot(probabilities, np.log(probabilities)))


def _progress_line(*, processed: int, total: int, elapsed_seconds: float) -> str:
    rate = processed / max(elapsed_seconds, 1e-9)
    remaining = max(0, total - processed)
    eta_seconds = remaining / rate if rate > 0.0 else math.inf
    return (
        f"processed_windows={processed}/{total} "
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"rate={rate:.2f}_windows_per_s "
        f"eta={_format_duration(eta_seconds)}"
    )


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes}m{seconds_part:02d}s"
    return f"{seconds_part}s"


def _cross_entropy_from_counts(
    counts: np.ndarray, probabilities: np.ndarray
) -> float:
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("cross-entropy is undefined for an empty distribution")
    if bool((probabilities[active] <= 0).any()):
        raise ValueError("predictor assigns zero probability to an observed token")
    empirical = counts[active].astype(np.float64)
    empirical /= empirical.sum()
    return float(-np.dot(empirical, np.log(probabilities[active])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.9f}"


if __name__ == "__main__":
    raise SystemExit(main())
