"""Debut-mode fine-tune pipeline entry point (Worker 5).

Role in the system
------------------
`train_pipeline.py` is the pre-training entry point: it trains a fresh model
from random initialization on the general (non-debut) canvas task. This
module is the SEPARATE fine-tuning entry point: it warm-starts a model from a
*pre-trained* checkpoint's weights, then continues training with a much
smaller learning rate on the debut-mode task (predicting the [WIN]/[LOSS]
outcome token plus each side's first-appearance "debut" events). After
training it calls the Worker-4 evaluator to produce `finetune_report.json`.

This module intentionally imports (rather than copies) as much as possible
from `train_pipeline.py` -- the replay split/selection helpers, the
dataloader builder, the checkpoint/metrics publishing helpers -- so the fine
-tune run uses EXACTLY the same 25 train / 3 dev replays (same seeds) as
pre-training, and so `train_pipeline.py` itself needs no edits at all. Only
three things differ from `_run_real_pipeline`:

  1. Warm start: `TrainingLoop.load_model_weights(...)` copies ONLY the model
     (and EMA model) weights from the pre-trained checkpoint. The optimizer,
     LR schedule, global_step, and epoch counters all stay fresh (start at
     zero) -- this is a NEW training run over the debut task, not a resume of
     the old one. `_try_resume` / `_resume_from_remote` (full resume) are
     deliberately never called here.
  2. Metrics files get a `finetune_` filename prefix so they land in the SAME
     log directory as pre-training (`tests/output/overfitV2`) without ever
     overwriting pre-training's `step_metrics.jsonl` / `epoch_metrics.csv`.
  3. After training, the Worker-4 evaluator (`thesis_ml.eval.finetune_report`)
     scores the fine-tuned model on both the "memorized" set (the 25 replays
     it trained on) and the "test" set (the 3 held-out dev replays) and writes
     `finetune_report.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from thesis_ml.config import load_config
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import load_window_manifest
from thesis_ml.eval.finetune_report import build_and_write_finetune_report
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.pipeline.storage import StorageResolver

# Reuse (do NOT copy) the pre-training pipeline's private helpers. Importing
# these keeps the fine-tune replay selection, manifest handling, dataloader
# construction, and checkpoint/metrics publishing byte-for-byte identical to
# pre-training, and means `train_pipeline.py` requires zero edits.
from thesis_ml.pipeline.train_pipeline import (
    _checkpoint_publisher,
    _dataset_available,
    _ensure_window_manifest,
    _local_checkpoint_dir,
    _local_metrics_dir,
    _make_dataloader,
    _materialize_file,
    _materialize_replay_paths,
    _metrics_publisher,
    _publish_checkpoint,
    _select_replays,
)
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.vocab.content_vocab import load_content_vocabulary


@dataclass(frozen=True)
class FinetunePipelineResult:
    """Summary of one fine-tune run, mirroring `TrainingPipelineResult`."""

    checkpoint_uri: str
    steps: int
    parameter_count: int
    peak_vram_bytes: int
    report_path: str


def _select_eval_examples(dataset, cap: int) -> list:
    """Pick a bounded, evenly-strided sample of examples from a windowed dataset.

    A windowed `SC2DiffusionDataset` turns each replay into many overlapping
    windows, so scoring every one is both slow (one sampler run per example) and
    memory-heavy (every 4096-token debut example materialized in host RAM at
    once). This returns at most `cap` examples, chosen at evenly-spaced indices
    so the sample spans early-game through late-game windows (keeping the
    minute-based win/loss buckets populated) instead of clustering at the start.

    Parameters:
        dataset: A dataset supporting `len(...)` and integer indexing
            (`SC2DiffusionDataset`); building an item lazily constructs its
            debut target.
        cap: Maximum number of examples to return. `cap <= 0` means "no cap"
            (return every example) -- only sensible for tiny datasets.

    Returns:
        A list of materialized dataset examples (length `min(cap, len(dataset))`,
        or the full dataset when `cap <= 0`).

    Calls:
        Only builtins plus the dataset's own `__len__`/`__getitem__`.
    """

    total = len(dataset)
    # cap <= 0 disables the cap; also short-circuit when the dataset already
    # fits under the cap so we never over-select.
    if cap <= 0 or total <= cap:
        return [dataset[index] for index in range(total)]

    # Evenly spread `cap` indices across [0, total): index i maps to
    # floor(i * total / cap). This is a simple, dependency-free way to stride
    # across the whole dataset (start .. near-end) without importing numpy.
    selected_indices = [(position * total) // cap for position in range(cap)]
    return [dataset[index] for index in selected_indices]


def run_finetune_pipeline(
    config_path: str | Path,
    *,
    storage: StorageResolver | None = None,
    max_steps_override: int | None = None,
) -> FinetunePipelineResult:
    """Run the full debut fine-tune: warm-start -> train -> evaluate -> report.

    Parameters:
        config_path: Path to the fine-tune YAML config (e.g.
            `configs/local_overfit_v2_finetune.yaml`). Must set
            `data.debut_mode: true`, `sampler.outcome_last: true`, and a
            non-empty `train.init_from_checkpoint`.
        storage: Optional `StorageResolver` override (tests inject a fake).
        max_steps_override: When set, caps the number of optimizer steps
            regardless of the config's epoch/step budget -- used for a
            bounded smoke run (`--max-steps`).

    Returns:
        A `FinetunePipelineResult` with the checkpoint location, steps taken,
        parameter count, peak VRAM usage, and the written report's path.

    Calls:
        `_select_replays`/`split_replays` (same replay picks as pre-training),
        `TrainingLoop.load_model_weights` (weights-only warm start),
        `TrainingLoop.fit` (the actual training loop),
        `build_and_write_finetune_report` (Worker 4's evaluator).
    """

    config = load_config(config_path)
    resolver = storage or StorageResolver()

    if not config.data.debut_mode:
        raise ValueError(
            "finetune_pipeline requires config.data.debut_mode=true "
            "(the evaluator reads outcome/fog labels off debut-mode targets)"
        )
    if not config.train.init_from_checkpoint:
        raise ValueError(
            "finetune_pipeline requires a non-empty config.train.init_from_checkpoint "
            "(the pretrained checkpoint to warm-start from)"
        )
    if not _dataset_available(config, resolver):
        raise FileNotFoundError(f"dataset not found at configured URI: {config.storage.data_uri}")

    torch.manual_seed(config.pipeline.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.train.require_cuda and device != "cuda":
        raise RuntimeError(
            "this fine-tune profile requires CUDA, but torch.cuda.is_available() is false"
        )

    # Fine-tune checkpoints go to a SEPARATE directory (storage.checkpoint_uri
    # in the fine-tune config, e.g. checkpoints/local-overfitV2-finetune) so
    # they never overwrite the pre-trained checkpoint we are warm-starting
    # from.
    checkpoint_dir = _local_checkpoint_dir(config, resolver)
    token_dictionary = _materialize_file(
        config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver
    )
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)

    # Identical split + selection logic (same seeds) as pre-training, imported
    # directly rather than re-implemented, so fine-tuning trains/evaluates on
    # the SAME 25 train + 3 dev replays that produced the checkpoint we are
    # warm-starting from.
    split = split_replays(
        replay_paths,
        seed=config.pipeline.split_seed,
        test_fraction=config.pipeline.test_fraction,
        dev_fraction=config.pipeline.dev_fraction,
    )
    train_replays, dev_replays = _select_replays(list(split.train), list(split.dev), config)

    train_windows = load_window_manifest(
        config.data.window_manifest_path, config=config, replay_paths=train_replays
    )
    dev_windows = load_window_manifest(
        config.data.window_manifest_path, config=config, replay_paths=dev_replays
    )

    # Because config.data.debut_mode is True here, SC2DiffusionDataset builds
    # the 7-class debut target (outcome token at canvas position 0 plus
    # visible/fogged/future debut events) for every example -- see Worker 1's
    # `_build_debut_target`. This is what makes the resulting examples usable
    # both for training AND, unchanged, as the "examples" the Worker-4
    # evaluator scores below.
    train_dataset = SC2DiffusionDataset(
        train_windows, config, vocabulary, seed=config.pipeline.seed, fog_rate_override=None
    )
    train_loader = _make_dataloader(train_dataset, config, shuffle=True, device=device)

    val_loader = None
    dev_dataset = None
    if dev_windows:
        dev_dataset = SC2DiffusionDataset(
            dev_windows, config, vocabulary, seed=config.pipeline.seed, fog_rate_override=None
        )
        val_loader = _make_dataloader(dev_dataset, config, shuffle=False, device=device)

    planned_steps = config.train.max_steps
    if planned_steps <= 0:
        planned_steps = len(train_loader) * config.train.epochs
    training_config = replace(
        config,
        train=replace(config.train, checkpoint_dir=str(checkpoint_dir), max_steps=planned_steps),
    )
    model = SC2StrategyDiffusionModel(training_config, vocab_size=vocabulary.vocab_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Metrics land in the SAME log directory pre-training uses, but every
    # filename gets a "finetune_" prefix so pre-training's step_metrics.jsonl
    # / epoch_metrics.csv are never touched by a fine-tune run.
    metrics_dir = _local_metrics_dir(config, resolver)
    metrics_path = metrics_dir / "finetune_step_metrics.jsonl"
    epoch_metrics_path = metrics_dir / "finetune_epoch_metrics.csv"

    loop = TrainingLoop(
        model=model,
        config=training_config,
        device=device,
        seed=config.pipeline.seed,
        metrics_path=metrics_path,
        epoch_metrics_path=epoch_metrics_path,
        checkpoint_publisher=_checkpoint_publisher(config, resolver),
        metrics_publisher=_metrics_publisher(config, resolver),
    )

    # WARM START (not resume): copy only the pre-trained model/EMA weights
    # into this fresh loop. Optimizer state, LR schedule position, and
    # global_step/completed_epochs are all left at their fresh __init__
    # values, so this fine-tune run starts a brand new 100-epoch schedule at
    # step 0. We deliberately never call `_try_resume`/`_resume_from_remote`
    # here -- those restore the FULL optimizer/step state and would be wrong
    # for starting a new fine-tune phase (see `load_model_weights` docstring).
    loop.load_model_weights(config.train.init_from_checkpoint)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    requested_steps = training_config.train.max_steps if max_steps_override is None else max_steps_override
    loop.fit(
        train_loader,
        val_dataloader=val_loader,
        max_steps=requested_steps,
        epochs=training_config.train.epochs,
    )
    peak_vram_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    checkpoint_uri = _publish_checkpoint(config, resolver, checkpoint_dir / "last.pt")

    # --- Worker-4 evaluation: build finetune_report.json ---------------------
    # "memorized" = the 25 replays actually fine-tuned on (diagnoses whether
    # the debut/outcome targets were absorbed at all). "test" = the 3 held-out
    # dev replays. Evaluation uses the EMA model (the same weights validation
    # uses during training) because EMA weights are the intended deployment
    # weights, and the evaluator generates canvases via the sampler rather
    # than reading raw logits, so training-mode dropout/etc. does not apply.
    # IMPORTANT (performance/memory): a windowed dataset expands each replay into
    # many overlapping windows -- the 25 memorized replays alone become well over
    # a thousand examples. The evaluator samples the diffusion model ONCE PER
    # EXAMPLE, so scoring every window would fire thousands of sampler runs and
    # hold every materialized 4096-token debut example in host RAM at once. We
    # therefore score only a bounded, evenly-strided subset (config-driven cap).
    # Even striding -- rather than "the first N" -- keeps the sample spread from
    # early-game to late-game windows so the 1/3/5/7/10-minute win/loss buckets
    # stay populated.
    eval_cap = training_config.eval.debut_max_examples
    memorized_examples = _select_eval_examples(train_dataset, eval_cap)
    test_examples = _select_eval_examples(dev_dataset, eval_cap) if dev_dataset is not None else []
    report_path = metrics_dir / "finetune_report.json"
    build_and_write_finetune_report(
        memorized_examples=memorized_examples,
        test_examples=test_examples,
        model=loop.ema_model,
        vocabulary=vocabulary,
        config=training_config,
        path=report_path,
        device=device,
    )

    return FinetunePipelineResult(
        checkpoint_uri=checkpoint_uri,
        steps=loop.global_step,
        parameter_count=parameter_count,
        peak_vram_bytes=peak_vram_bytes,
        report_path=str(report_path),
    )


def main() -> None:
    """CLI entry point, mirroring `train_pipeline.main()`'s argparse shape."""

    parser = argparse.ArgumentParser(description="Run the SC2 debut fine-tune from config")
    parser.add_argument("--config", type=Path, default=Path("configs/local_overfit_v2_finetune.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="bounded verification override")
    args = parser.parse_args()
    try:
        result = run_finetune_pipeline(args.config, max_steps_override=args.max_steps)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(
        f"checkpoint_uri={result.checkpoint_uri} steps={result.steps} "
        f"parameters={result.parameter_count} peak_vram_bytes={result.peak_vram_bytes} "
        f"report_path={result.report_path}"
    )


if __name__ == "__main__":
    main()
