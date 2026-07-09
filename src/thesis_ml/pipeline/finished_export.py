"""Durable 'finished' model export for a properly-completed pre-training run.

Role in the system
------------------
A training *checkpoint* (``last.pt``) and a *finished model* are deliberately
different artifacts, and this module exists to draw a hard line between them:

  * ``last.pt`` is a ROLLING, frequently-overwritten resume file. The training
    loop rewrites it every ``checkpoint_interval`` steps and once more at the
    end, purely so an interrupted run can pick up where it left off. The next
    checkpoint interval -- or the next run pointed at the same directory --
    overwrites it in place. It is machinery, not a deliverable.
  * A *finished model* is written EXACTLY ONCE, only when a pre-training run
    *properly finishes*: it trained through all of its configured epochs, or it
    stopped via early stopping. It is never written for a crash, a Ctrl+C, or a
    bounded ``--max-steps`` smoke/verification run. It is the durable,
    publishable end product.

To make that distinction obvious on disk and to make the finished model hard to
lose or clobber, ``export_finished_model`` below:

  * writes into a dedicated ``finished/`` subdirectory of the checkpoint dir, so
    the durable artifact is physically separated from the churny ``last.pt``;
  * exports the raw model weights and the EMA (evaluation) weights SEPARATELY,
    each as its own ``.safetensors`` file (portable, tensor-only, no pickled
    Python objects), tagged ``raw`` / ``ema`` in both the filename and the
    file's embedded metadata;
  * also writes a combined ``finished.pt`` torch bundle (both weight sets plus
    the config) for convenient single-file loading;
  * writes ``config.json`` (the full ``ProjectConfig``) and
    ``finished_metadata.json`` recording everything needed to rebuild the model
    from the ``.safetensors`` files (vocabulary size, architecture flags, which
    file is which, and which weight set is the default for serving);
  * NEVER destroys a previously finished model: if ``finished/`` already exists
    it is first archived to a uniquely-named ``finished_superseded_<UTC>/``;
  * marks every written file read-only, so it is harder to overwrite by
    accident.

This module is only ever invoked by the pre-training pipeline
(``thesis_ml.pipeline.train_pipeline``); fine-tuning has its own separate
pathway and does not call it.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from thesis_ml.config import ProjectConfig


# Fixed artifact filenames inside the ``finished/`` directory. Exposed as module
# constants so downstream loaders and tests refer to them by name instead of
# re-hardcoding the strings. The ``.raw.``/``.ema.`` infix is the human-visible
# tag distinguishing the two weight sets.
RAW_WEIGHTS_FILENAME = "model.raw.safetensors"
EMA_WEIGHTS_FILENAME = "model.ema.safetensors"
TORCH_BUNDLE_FILENAME = "finished.pt"
CONFIG_FILENAME = "config.json"
METADATA_FILENAME = "finished_metadata.json"

# Which of the two weight sets downstream serving/eval should load by default.
# The EMA (exponential moving average) weights are what validation and the
# sampler use, so they are the canonical "the trained model" for inference.
DEFAULT_SERVING_WEIGHTS = "ema"


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached, CPU, storage-independent copy of a module's tensors.

    ``safetensors`` cannot serialize tensors that share underlying memory (it
    has no way to represent aliasing) and expects plain contiguous tensors.
    Cloning each tensor onto the CPU guarantees BOTH invariants -- no two
    entries share storage, and every entry is contiguous -- no matter how the
    live model happened to lay its parameters out in memory (e.g. views into a
    fused buffer). Used for both the raw and EMA state dicts before writing.

    Parameters:
        module: the model (raw or EMA) whose weights are being exported.

    Returns:
        A ``{name: tensor}`` dict of independent CPU tensors safe to hand to
        both ``safetensors.torch.save_file`` and ``torch.save``.
    """

    return {
        name: tensor.detach().to("cpu").contiguous().clone()
        for name, tensor in module.state_dict().items()
    }


def _make_readonly(path: Path) -> None:
    """Best-effort: drop write permission so ``path`` is harder to overwrite.

    Durability is a nicety here, not a hard requirement, so any platform that
    refuses the permission change must not fail the whole training run over it;
    the ``OSError`` is swallowed deliberately.

    Parameters:
        path: the just-written finished-model file to protect.
    """

    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def _archive_existing(finished_dir: Path) -> Path | None:
    """Move an existing ``finished/`` aside so a new export never clobbers it.

    A previously finished model is a valuable, hard-won artifact; a later run
    that also finishes must not silently overwrite it. If ``finished_dir``
    already exists it is renamed to a uniquely-named sibling
    ``finished_superseded_<UTC timestamp>/`` (with a numeric suffix if two
    exports land in the same second). Renaming the directory succeeds even
    though the files inside were marked read-only, because the read-only bit is
    on the files, not on the directory entry being renamed.

    Parameters:
        finished_dir: the target ``finished/`` directory about to be written.

    Returns:
        The path the old directory was archived to, or ``None`` if there was no
        existing finished model.
    """

    if not finished_dir.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = finished_dir.parent / f"finished_superseded_{stamp}"
    counter = 1
    while candidate.exists():
        candidate = finished_dir.parent / f"finished_superseded_{stamp}_{counter}"
        counter += 1
    finished_dir.rename(candidate)
    return candidate


def export_finished_model(
    *,
    checkpoint_dir: str | Path,
    model: nn.Module,
    ema_model: nn.Module,
    config: ProjectConfig,
    vocab_size: int,
    global_step: int,
    completed_epochs: int,
    configured_epochs: int,
    stop_reason: str,
    publisher: Callable[[Path], None] | None = None,
) -> Path:
    """Write the durable finished-model bundle for a completed pre-training run.

    Writes, into ``<checkpoint_dir>/finished/`` (archiving any prior finished
    model first): the raw and EMA weights as two separately-tagged
    ``.safetensors`` files, a combined ``finished.pt`` torch bundle, a
    ``config.json`` (full ``ProjectConfig``), and a ``finished_metadata.json``.
    Every written file is published (if a ``publisher`` is given) and then made
    read-only.

    Parameters:
        checkpoint_dir: the same local directory the loop writes ``last.pt`` to;
            the finished model lands in its ``finished/`` subdir.
        model: the trained model carrying the raw (final-step) weights.
        ema_model: the EMA shadow model carrying the smoothed serving weights.
        config: the effective ``ProjectConfig`` the run used (serialized whole).
        vocab_size: output vocabulary size, needed to rebuild the model from the
            ``.safetensors`` files.
        global_step: total optimizer steps completed (provenance metadata).
        completed_epochs: epochs actually finished.
        configured_epochs: epochs the config asked for (so a reader can tell an
            early-stopped run from a full one).
        stop_reason: short machine-readable reason the run ended, e.g.
            ``"completed_all_epochs"`` or ``"early_stopping"``.
        publisher: optional callback that mirrors each written file to durable
            remote storage (used by the S3 pipeline); ``None`` for local runs.

    Returns:
        The ``finished/`` directory path that was written.

    Calls:
        ``safetensors.torch.save_file``, ``torch.save``,
        ``_cpu_state_dict``, ``_archive_existing``, ``_make_readonly``.
    """

    # Imported lazily so merely importing the pipeline does not pull in
    # safetensors; it is only needed on the (rare) proper-finish code path.
    from safetensors.torch import save_file

    finished_dir = Path(checkpoint_dir) / "finished"
    _archive_existing(finished_dir)
    finished_dir.mkdir(parents=True, exist_ok=True)

    raw_state = _cpu_state_dict(model)
    ema_state = _cpu_state_dict(ema_model)

    # safetensors metadata values must all be strings. These stamp each weight
    # file with which set it is plus enough provenance to identify the run.
    common_metadata = {
        "vocab_size": str(vocab_size),
        "global_step": str(global_step),
        "completed_epochs": str(completed_epochs),
        "stop_reason": stop_reason,
    }
    raw_path = finished_dir / RAW_WEIGHTS_FILENAME
    ema_path = finished_dir / EMA_WEIGHTS_FILENAME
    save_file(raw_state, str(raw_path), metadata={**common_metadata, "weights": "raw"})
    save_file(ema_state, str(ema_path), metadata={**common_metadata, "weights": "ema"})

    # Combined single-file torch bundle: both weight sets plus the config, for
    # callers that would rather load one .pt than wire up the safetensors files.
    bundle_path = finished_dir / TORCH_BUNDLE_FILENAME
    torch.save(
        {
            "model": raw_state,
            "ema_model": ema_state,
            "config": config,
            "vocab_size": vocab_size,
            "global_step": global_step,
            "completed_epochs": completed_epochs,
            "stop_reason": stop_reason,
        },
        bundle_path,
    )

    # Full config as JSON (asdict recurses the nested frozen dataclasses; every
    # leaf is a str/int/float/bool, so it is directly JSON-serializable).
    config_path = finished_dir / CONFIG_FILENAME
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Self-describing manifest: everything a loader needs to rebuild the model
    # from the .safetensors files without importing this package.
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vocab_size": vocab_size,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "configured_epochs": configured_epochs,
        "stop_reason": stop_reason,
        "self_conditioning": config.model.self_conditioning,
        "weights": {"raw": RAW_WEIGHTS_FILENAME, "ema": EMA_WEIGHTS_FILENAME},
        "torch_bundle": TORCH_BUNDLE_FILENAME,
        "config_file": CONFIG_FILENAME,
        "default_serving_weights": DEFAULT_SERVING_WEIGHTS,
    }
    metadata_path = finished_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for path in (raw_path, ema_path, bundle_path, config_path, metadata_path):
        # Publish while still writable (the publisher only reads), then lock the
        # local copy down against accidental overwrite.
        if publisher is not None:
            publisher(path)
        _make_readonly(path)
    return finished_dir
