"""Canvas-position-1 outcome-token probe: does the model know WHERE, WHAT, or neither?

Role in the system
------------------
Read-only measurement tool. It answers one question that the reported
cross-entropy provably cannot, because CE collapses two very different failure
modes into the same number:

  * **"wrong position"** -- the model never located canvas index 1, so its
    distribution there is essentially the noise marginal; and
  * **"right position, hedged"** -- the model located canvas index 1 AND learned
    that only ``[WIN]``/``[LOSS]`` live there, but is outvoted by a copy prior
    that keeps re-emitting whatever noise token it was shown.

Both produce a per-class CE that sits stubbornly around or above ``ln(2)``. They
imply completely different fixes, so they have to be told apart by measurement
rather than by argument. See ``diagnostics/009-rare-class-position-blindness.md``
§1 and §6 for the hypothesis this probe is testing.

What it measures
----------------
For each example, one single denoising forward pass at a high noise level
``t`` (default: drawn per example from ``[0.9, 1.0]``, i.e. the regime where the
canvas is almost entirely noise and positional identification is the only signal
available). It then reads the model's output distribution at **canvas position
1 only** -- the slot that ``SPEC.md`` §3/§7 guarantees holds the perspective
player's ``[WIN]`` or ``[LOSS]`` token -- and records:

  1. ``pair_mass``          -- total probability on ``{[WIN], [LOSS]}`` (the PAIR,
                              not just the correct one). This is "did the model
                              learn what kind of slot this is".
  2. ``shown_token_mass``   -- probability on whatever token was actually sitting
                              at position 1 in the noised canvas the model was
                              shown. This is the copy prior's weight.
  3. ``true_rank``          -- 1-based rank of the true outcome token in the
                              sorted logits (1 = argmax).
  4. ``shown_token_*``      -- the identity of the position-1 token the model was
                              shown, plus whether it was corrupted and whether it
                              actually differs from the target, so a reader can
                              always tell a noised slot from a clean one.

Reading the result
------------------
The two competing worlds are separated by ``pair_mass`` and ``true_rank``, and
both are reported against the noise marginal the summary computes for the run
(``uniform_pair_mass`` = ``0`` for the outcome pair, because the uniform
corruption support intentionally excludes ``[WIN]`` and ``[LOSS]`` -- see
``train/corruption.sample_uniform_noise``):

  * ``pair_mass`` well above the noise marginal with ``true_rank`` in the single
    digits => the model FOUND the position and LEARNED the class; whatever is
    eating the probability mass (see ``argmax_equals_shown_token_fraction``) is
    the real defect.
  * ``pair_mass`` indistinguishable from ``uniform_pair_mass`` with
    ``true_rank`` in the hundreds => positional addressability really is broken.

WEIGHT SELECTION -- this tool deliberately uses the TRAINED weights
------------------------------------------------------------------
Unlike ``viz/diagnostics.py`` (which defaults to EMA because it is reproducing
what the sampler serves), this probe defaults to the raw ``model`` state dict --
the actual final optimizer step. The question here is what training put into the
weights, not what the deployed averaged model emits, and EMA smoothing is an
extra confound between the measurement and the answer. ``--ema`` opts into the
EMA weights instead; every output JSON records which set was used under
``weights``.

Nothing here mutates a checkpoint, a config, the manifest, or the source
replays. The only write target is the per-arm JSON file.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from torch.utils.data import Subset

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import DiffusionBatch, collate_diffusion_examples
from thesis_ml.data.dataset import SC2DiffusionDataset
from thesis_ml.data.feature_stats import load_feature_statistics
from thesis_ml.data.windowing import load_window_manifest
from thesis_ml.pipeline.storage import StorageResolver

# The probe MUST see exactly the examples the arm trained on, in exactly the
# grouping the arm used, or a difference in the numbers could be a difference in
# the data rather than in the weights. These two helpers are the training
# pipeline's own replay-selection and checkpoint-directory resolution, imported
# rather than re-derived for that reason. They are underscore-private to
# train_pipeline, but duplicating replay selection is the strictly worse risk.
from thesis_ml.pipeline.train_pipeline import (
    _explicit_replay_selection,
    _local_checkpoint_dir,
)
from thesis_ml.train.corruption import corrupt_batch
from thesis_ml.train.loop import move_batch_to_device
from thesis_ml.viz.diagnostics import load_diagnostic_model
from thesis_ml.vocab.content_vocab import ContentVocabulary, load_content_vocabulary
from thesis_ml.vocab.special_tokens import (
    CONTENT_TOKEN_OFFSET,
    LOSS_ID,
    SPECIAL_TOKENS,
    WIN_ID,
)

# arm-name : config-file. Mirrors the ARMS list in tests/run_ablation_sweep.sh,
# plus the all-toggles-false baseline, which is not part of the sweep (it was
# trained separately) but is the reference every arm is read against.
#
# The baseline entry points at configs/ablation_00_baseline.yaml, NOT at
# configs/local_overfit_v2.yaml, and that distinction is load-bearing. Each entry
# here is used to BUILD a model, which is then handed the arm's checkpoint --
# so the config must resolve to the architecture the checkpoint was trained
# under. local_overfit_v2.yaml used to pin all three toggles false and did; since
# `model.frozen_input_kv` was promoted to a default (config/default.yaml,
# 2026-08-09) it resolves to `+frozen_input_kv` and would abort the baseline
# checkpoint load on a fingerprint mismatch. ablation_00_baseline.yaml exists to
# hold the original all-false resolution steady at the same storage paths.
ARMS: tuple[tuple[str, str], ...] = (
    ("00-baseline-all-toggles-false", "configs/ablation_00_baseline.yaml"),
    ("01-frozen-input-kv-only", "configs/ablation_01_frozen_input_kv.yaml"),
    ("02-segment-embeddings-only", "configs/ablation_02_segment_embeddings.yaml"),
    ("03-per-segment-positions-only", "configs/ablation_03_per_segment_positions.yaml"),
    (
        "04-segment-embeddings-plus-per-segment-positions",
        "configs/ablation_04_segment_embeddings_plus_per_segment_positions.yaml",
    ),
    (
        "05-frozen-input-kv-plus-segment-embeddings",
        "configs/ablation_05_frozen_input_kv_plus_segment_embeddings.yaml",
    ),
)

# Reverse lookup for the eight reserved ids, so a noise token that happens to be a
# special token is named rather than printed as a bare integer.
_SPECIAL_ID_TO_TOKEN = {token_id: token for token, token_id in SPECIAL_TOKENS.items()}

# DiffusionBatch carries the perspective player as an integer so it survives
# `.to(device)` (a Python string would not); this maps it back for the report.
# Kept in sync with data/collate.py's _PERSPECTIVE_P1 / _PERSPECTIVE_P2.
_PERSPECTIVE_ID_TO_NAME = {1: "p1", 2: "p2"}


@dataclass(frozen=True)
class OutcomePositionRecord:
    """One example's model distribution at canvas position 1.

    Every field is derived from a single forward pass over one example at one
    noise level. Probabilities are softmax over the full vocabulary of the
    float32 logits at canvas index 1.

    Attributes:
        example_index: 0-based index of the window within the probed split's
            dataset. Deterministic and identical across arms, so the same
            ``example_index`` is the same window in every arm's JSON.
        t: the noise level this example was corrupted at.
        perspective: ``"p1"`` or ``"p2"`` -- which player's view this window was
            built from. Load-bearing here rather than incidental: the outcome
            label is perspective-relative, so the SAME game is ``[WIN]`` from one
            perspective and ``[LOSS]`` from the other. A model emitting a global
            class prior scores the same on both; a model that actually reads the
            game state has to flip with the perspective. See
            ``summarize``'s ``by_perspective`` block.
        true_token: name of the ground-truth outcome token (``[WIN]``/``[LOSS]``).
        true_token_id: its vocabulary id (``WIN_ID`` 4 or ``LOSS_ID`` 5).
        shown_token: name of the token the model was actually SHOWN at canvas
            position 1 (the noised canvas entry).
        shown_token_id: that token's vocabulary id.
        shown_was_corrupted: True when the corruption process took the
            "replace" branch at this position. Under uniform diffusion the drawn
            replacement can coincidentally equal the target, which is why this is
            reported separately from ``shown_differs_from_true``.
        shown_differs_from_true: True when the shown token is genuinely a
            different token from the target -- the honest "this slot was noised"
            flag, and the condition under which ``shown_token_mass`` measures a
            copy prior rather than measuring correctness.
        pair_mass: p([WIN]) + p([LOSS]). The headline number.
        true_mass: probability on the correct one of the two.
        other_pair_mass: probability on the incorrect one of the two.
        win_mass: probability on ``[WIN]`` specifically, regardless of which one
            is correct here. Reported alongside ``loss_mass`` so a standing
            preference for one class can be read directly, independent of the
            sample's outcome balance.
        loss_mass: probability on ``[LOSS]`` specifically.
        p_true_given_pair: ``true_mass / pair_mass`` -- the model's call BETWEEN
            the two outcome tokens, with the "did it find the slot at all"
            question divided out. This is the field that separates the two
            failure modes that ``pair_mass`` alone cannot:

              * ``~0.5`` => the model located the slot and learned that only
                these two tokens live there, but has NOT learned which one this
                game is. The remaining ablation work aimed at positional
                addressability would then be aimed at a problem that is not the
                binding constraint.
              * ``~1.0`` with low ``pair_mass`` => the model knows the answer
                whenever it engages the slot at all; the binding constraint
                really is locating/committing to the slot.

            ``None`` when ``pair_mass`` is exactly zero (no meaningful ratio).
            Note the per-row value is noisy when ``pair_mass`` is negligible,
            which is why ``summarize`` also reports the pooled
            ``sum(true_mass) / sum(pair_mass)``.
        pair_argmax_correct: True when ``true_mass > other_pair_mass`` -- the
            hard version of ``p_true_given_pair``, i.e. did the model pick the
            right class given a forced choice between the two.
        shown_token_mass: probability on ``shown_token_id`` -- the copy prior's
            weight at this position.
        argmax_token: name of the highest-logit token.
        argmax_token_id: its vocabulary id.
        argmax_mass: its probability.
        true_rank: 1-based rank of ``true_token_id`` in the sorted logits.
        win_rank: 1-based rank of ``[WIN]``.
        loss_rank: 1-based rank of ``[LOSS]``.
        shown_token_rank: 1-based rank of the shown token. A rank of 1 here with
            a low ``true_rank`` is the signature of "learned the class, outvoted
            by the copy prior".
        entropy_nats: Shannon entropy of the full position-1 distribution, in
            nats. ``ln(vocab_size)`` would be a uniform hedge over everything.
    """

    example_index: int
    t: float
    perspective: str
    true_token: str
    true_token_id: int
    shown_token: str
    shown_token_id: int
    shown_was_corrupted: bool
    shown_differs_from_true: bool
    pair_mass: float
    true_mass: float
    other_pair_mass: float
    win_mass: float
    loss_mass: float
    p_true_given_pair: float | None
    pair_argmax_correct: bool
    shown_token_mass: float
    argmax_token: str
    argmax_token_id: int
    argmax_mass: float
    true_rank: int
    win_rank: int
    loss_rank: int
    shown_token_rank: int
    entropy_nats: float


# ---------------------------------------------------------------------------
# Data wiring -- the training pipeline's own selection, replayed read-only
# ---------------------------------------------------------------------------


def select_probe_indices(
    dataset_size: int,
    *,
    n_examples: int,
    sample_mode: str,
) -> list[int]:
    """Choose which dataset indices to probe.

    This is not a detail. Window order within a split is grouped by replay and
    perspective, so the FIRST N windows are typically all from one or two games
    and can easily all share the same ``[WIN]``/``[LOSS]`` outcome -- which would
    make "rank of the true outcome token" a measurement of one class on one
    replay. ``"strided"`` (the default) walks evenly across the whole split
    instead, so the sample spans every replay, both perspectives, and both
    outcome classes.

    Parameters:
        dataset_size: number of windows in the split.
        n_examples: how many to probe; ``<= 0`` or larger than the split means
            every window.
        sample_mode: ``"strided"`` (evenly spread, the default) or ``"head"``
            (the first ``n_examples``, i.e. plain dataloader order).

    Returns:
        Sorted dataset indices, deterministic and identical across arms.
    """

    if sample_mode not in {"strided", "head"}:
        raise ValueError(f"sample_mode must be 'strided' or 'head', got {sample_mode!r}")
    if n_examples <= 0 or n_examples >= dataset_size:
        return list(range(dataset_size))
    if sample_mode == "head":
        return list(range(n_examples))
    return [(position * dataset_size) // n_examples for position in range(n_examples)]


def build_probe_dataloader(
    config: ProjectConfig,
    *,
    split: str,
    resolver: StorageResolver,
    n_examples: int,
    sample_mode: str,
) -> tuple[DataLoader, ContentVocabulary, list[int]]:
    """Build a deterministic loader over the arm's own train or dev windows.

    This reuses the training pipeline's explicit replay selection
    (``pipeline.train_replay_ids`` / ``dev_replay_ids``) and the already-built
    window manifest, so the probe sees the identical examples the arm trained or
    validated on. The manifest is READ, never rebuilt: if it were stale the load
    would raise, which is the correct outcome for a measurement tool.

    Shuffling is off and ``num_workers`` is forced to 0. Deterministic selection
    makes a given ``example_index`` mean the same window in every arm's JSON, so
    the six output files line up row for row; single-process loading keeps the
    probe from spawning worker processes that would contend with a running sweep.

    Parameters:
        config: the arm's loaded config (its ``model`` section is irrelevant
            here; only the data/pipeline sections are read).
        split: ``"train"`` (default; the memorization claim is about the train
            subset) or ``"dev"``.
        resolver: storage resolver used to enumerate the replay corpus, exactly
            as ``train_pipeline`` does.
        n_examples: how many windows to probe (see ``select_probe_indices``).
        sample_mode: ``"strided"`` or ``"head"`` (see ``select_probe_indices``).

    Returns:
        ``(dataloader, vocabulary, probed_indices)`` where ``probed_indices``
        are the dataset indices behind the loader's rows, in loader order.

    Raises:
        ValueError: ``split`` is not "train"/"dev", or the config has no explicit
            replay selection (the ablation profiles all do; a profile relying on
            the seeded split would need that path added deliberately rather than
            silently probed on a different subset).

    Calls: ``load_content_vocabulary``, ``_explicit_replay_selection``,
    ``load_window_manifest``, ``load_feature_statistics``,
    ``SC2DiffusionDataset``.
    """

    if split not in {"train", "dev"}:
        raise ValueError(f"split must be 'train' or 'dev', got {split!r}")

    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    replay_paths = resolver.list_files(config.storage.data_uri, config.pipeline.replay_glob)
    selection = _explicit_replay_selection(replay_paths, config)
    if selection is None:
        raise ValueError(
            "this probe requires an explicit pipeline.train_replay_ids selection so it "
            "measures the same named subset the arm trained on; the configured profile "
            f"({config.storage.checkpoint_uri}) has none"
        )
    train_replays, dev_replays, _test_replays = selection
    selected_replays = train_replays if split == "train" else dev_replays
    if not selected_replays:
        raise ValueError(f"the {split} replay selection is empty for this profile")

    windows = load_window_manifest(
        config.data.window_manifest_path,
        config=config,
        replay_paths=selected_replays,
    )
    if not windows:
        raise RuntimeError(f"no {split} windows found in {config.data.window_manifest_path}")

    # Statistics identity is validated against the TRAIN replay ids regardless of
    # which split is being probed, because that is what the artifact was frozen
    # from and what the checkpoint was trained under.
    load_feature_statistics(
        config.data.feature_statistics_path,
        expected_source_replay_ids=[Path(path).name for path in train_replays],
    )

    dataset = SC2DiffusionDataset(
        windows,
        config,
        vocabulary,
        seed=config.pipeline.seed,
        fog_rate_override=None,
    )
    probed_indices = select_probe_indices(
        len(dataset), n_examples=n_examples, sample_mode=sample_mode
    )
    loader = DataLoader(
        Subset(dataset, probed_indices),
        batch_size=config.pipeline.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=partial(
            collate_diffusion_examples,
            retain_metadata=False,
            debut_mode=config.data.debut_mode,
        ),
    )
    return loader, vocabulary, probed_indices


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------


def _token_name(token_id: int, vocabulary: ContentVocabulary) -> str:
    """Resolve a vocabulary id to a readable name (specials included)."""

    if token_id in _SPECIAL_ID_TO_TOKEN:
        return _SPECIAL_ID_TO_TOKEN[token_id]
    try:
        return vocabulary.token_name_for(token_id)
    except KeyError:
        return f"[UNKNOWN:{token_id}]"


def _draw_noise_levels(
    batch_size: int,
    *,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one noise level per example, uniform in ``[t_min, t_max]``.

    Drawn on CPU from a caller-owned seeded generator (rather than letting
    ``corrupt_batch`` sample ``t`` itself) for two reasons: the exact per-example
    ``t`` has to be recorded in the output, and drawing it here makes the values
    identical across arms, since every arm sees the same batches in the same
    order from the same seed. That turns a cross-arm comparison into a paired
    one.
    """

    if t_min > t_max:
        raise ValueError(f"t_min ({t_min}) must not exceed t_max ({t_max})")
    draw = torch.rand(batch_size, generator=generator)
    return t_min + draw * (t_max - t_min)


def probe_batch(
    model,
    batch: DiffusionBatch,
    *,
    vocabulary: ContentVocabulary,
    config: ProjectConfig,
    device: torch.device,
    noise_levels: torch.Tensor,
    corruption_generator: torch.Generator,
    row_indices: Sequence[int],
) -> list[OutcomePositionRecord]:
    """Corrupt one batch at the given noise levels and read canvas position 1.

    Runs ONE denoising forward pass -- the same call the training loop makes in
    ``TrainingLoop.compute_batch_loss``, with the same corruption process and the
    same ``input_lengths`` hand-off the ``per_segment_positions`` ablation needs.
    There is no iterative sampling here on purpose: sampling would let later
    steps overwrite position 1 and would measure the sampler's schedule rather
    than the weights' distribution at a known ``t``.

    Self-conditioning is passed as ``None`` when the arm enables it, which is the
    honest first-denoising-step condition (``embed_canvas`` substitutes zeros);
    feeding it a self-conditioning estimate would require a second forward pass
    and would measure a two-step process instead of the model's raw distribution.

    Parameters:
        model: the loaded model, already in ``eval()`` mode.
        batch: one collated batch, still on CPU.
        vocabulary: for resolving ids to readable names.
        config: the arm's run config (read for ``diffusion.*``).
        device: compute device.
        noise_levels: ``[batch]`` per-example ``t``.
        corruption_generator: seeded generator on ``device`` driving the
            corruption branch and replacement draws.
        row_indices: the DATASET index behind each row of this batch, so
            ``example_index`` identifies the same window across arms even under
            strided sampling.

    Returns:
        One ``OutcomePositionRecord`` per row whose canvas position 1 is a real,
        scored outcome token. Rows failing that invariant are skipped by the
        caller-visible count difference and reported loudly.

    Raises:
        ValueError: a row's canvas position 1 is not ``[WIN]`` or ``[LOSS]``,
            which would mean the canvas grammar assumed here no longer holds.

    Calls: ``move_batch_to_device``, ``corrupt_batch``, ``model.forward``.
    """

    batch = move_batch_to_device(batch, device)
    corruption = corrupt_batch(
        input_token_ids=batch.input_token_ids,
        target_canvas=batch.target_canvas,
        process=config.diffusion.process,
        schedule=config.diffusion.schedule,
        vocab_size=int(model.vocab_size),
        generator=corruption_generator,
        t=noise_levels.to(device=device),
        canvas_noise_mask=batch.canvas_loss_mask,
    )

    forward_kwargs: dict = {
        "input_token_ids": corruption.input_token_ids,
        "canvas_token_ids": corruption.noised_canvas,
        "input_attention_mask": batch.input_attention_mask,
        "canvas_attention_mask": batch.canvas_attention_mask,
        "input_features": batch.input_features,
        "input_lengths": batch.input_lengths,
    }
    if config.model.self_conditioning:
        forward_kwargs["canvas_self_conditioning"] = None

    with torch.no_grad():
        output = model(**forward_kwargs)

    input_len = batch.input_token_ids.shape[1]
    # Canvas position 1 sits one token after the input region in the joint
    # [input | canvas] logits the model always returns (see model.forward).
    outcome_logits = output.logits[:, input_len + 1, :].float()
    probabilities = torch.softmax(outcome_logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    # Strict rank: how many tokens are STRICTLY above this one, plus one. Ties
    # are effectively impossible in float logits, and counting them as "not
    # ahead" is the reading that never overstates a token's position.
    def _rank(token_ids: torch.Tensor) -> torch.Tensor:
        selected = outcome_logits.gather(1, token_ids.unsqueeze(1))
        return (outcome_logits > selected).sum(dim=-1) + 1

    true_ids = batch.target_canvas[:, 1]
    shown_ids = corruption.noised_canvas[:, 1]
    win_ids = torch.full_like(true_ids, WIN_ID)
    loss_ids = torch.full_like(true_ids, LOSS_ID)
    argmax_ids = outcome_logits.argmax(dim=-1)

    win_mass = probabilities[:, WIN_ID]
    loss_mass = probabilities[:, LOSS_ID]
    true_mass = probabilities.gather(1, true_ids.unsqueeze(1)).squeeze(1)
    shown_mass = probabilities.gather(1, shown_ids.unsqueeze(1)).squeeze(1)
    argmax_mass = probabilities.gather(1, argmax_ids.unsqueeze(1)).squeeze(1)

    true_ranks = _rank(true_ids)
    win_ranks = _rank(win_ids)
    loss_ranks = _rank(loss_ids)
    shown_ranks = _rank(shown_ids)

    records: list[OutcomePositionRecord] = []
    for row in range(batch.target_canvas.shape[0]):
        dataset_index = int(row_indices[row])
        true_id = int(true_ids[row])
        if true_id not in (WIN_ID, LOSS_ID):
            raise ValueError(
                f"canvas position 1 of example {dataset_index} holds token id "
                f"{true_id}, not [WIN]/[LOSS]; the canvas grammar this probe assumes "
                "(SPEC.md 3/7) no longer holds and the measurement would be meaningless"
            )
        if not bool(batch.canvas_loss_mask[row, 1]):
            raise ValueError(
                f"canvas position 1 of example {dataset_index} is not scored; "
                "the loss mask no longer covers the outcome slot"
            )
        shown_id = int(shown_ids[row])
        pair = float(win_mass[row] + loss_mass[row])
        correct_mass = float(true_mass[row])
        incorrect_mass = pair - correct_mass
        records.append(
            OutcomePositionRecord(
                example_index=dataset_index,
                t=float(noise_levels[row]),
                perspective=_PERSPECTIVE_ID_TO_NAME.get(
                    int(batch.perspective_ids[row]), "unknown"
                ),
                true_token=_token_name(true_id, vocabulary),
                true_token_id=true_id,
                shown_token=_token_name(shown_id, vocabulary),
                shown_token_id=shown_id,
                shown_was_corrupted=bool(corruption.corrupted_positions[row, 1]),
                shown_differs_from_true=shown_id != true_id,
                pair_mass=pair,
                true_mass=correct_mass,
                other_pair_mass=incorrect_mass,
                win_mass=float(win_mass[row]),
                loss_mass=float(loss_mass[row]),
                # Exact-zero pair mass has no meaningful ratio; None rather than
                # a fabricated 0.5 so it can never be averaged in as a real
                # "the model was undecided" observation.
                p_true_given_pair=(correct_mass / pair) if pair > 0.0 else None,
                pair_argmax_correct=correct_mass > incorrect_mass,
                shown_token_mass=float(shown_mass[row]),
                argmax_token=_token_name(int(argmax_ids[row]), vocabulary),
                argmax_token_id=int(argmax_ids[row]),
                argmax_mass=float(argmax_mass[row]),
                true_rank=int(true_ranks[row]),
                win_rank=int(win_ranks[row]),
                loss_rank=int(loss_ranks[row]),
                shown_token_rank=int(shown_ranks[row]),
                entropy_nats=float(entropy[row]),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """Return mean/median/p10/p90/min/max for one measured quantity."""

    if not values:
        return {}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def summarize(records: Sequence[OutcomePositionRecord], *, vocab_size: int) -> dict:
    """Reduce per-example records to the numbers the two hypotheses differ on.

    The summary is computed three times: over every record, over only the
    records whose position 1 was genuinely noised, and over only the records
    whose position 1 happened to survive clean. The split matters because
    ``shown_token_mass`` only measures a copy prior on genuinely-noised rows --
    on a clean row the "shown token" IS the answer, so mass on it is just
    correctness wearing a different name.

    Every subset block carries two nested breakdowns that answer the follow-on
    question ``pair_mass`` cannot:

      * ``p_true_given_pair`` -- the model's call BETWEEN the two outcome tokens
        with "did it find the slot" divided out (see ``_conditional``). Near 0.5
        means the binding constraint is not positional addressability at all.
      * ``by_perspective`` -- the same behaviour split p1 / p2 (see
        ``_perspective_split``). Because the outcome label is perspective-
        relative, comparing ``prefers_win_fraction`` ACROSS the two sides
        separates "reads the game" from "has a standing class preference".

    Both also appear inside every ``position_1_noised_by_t_bucket`` cell, so the
    perspective behaviour can be read at the highest ``t`` -- where the canvas is
    pure noise and no contextual shortcut is available -- rather than only in
    aggregate over the band.

    Parameters:
        records: the per-example measurements.
        vocab_size: model output width, used for the noise-marginal reference
            values a reader compares ``pair_mass`` against.

    Returns:
        A JSON-ready dict with a ``reference`` block, three subset blocks, and
        the per-``t``-bucket breakdown.
    """

    # Uniform corruption draws PAD, DELIMITER, or content tokens. Outcome tokens
    # are deliberately outside the forward-process support.
    corruption_support = max(1, vocab_size - CONTENT_TOKEN_OFFSET + 2)
    reference = {
        "vocab_size": vocab_size,
        "corruption_support_states": corruption_support,
        "uniform_single_token_mass": 1.0 / corruption_support,
        "uniform_pair_mass": 0.0,
        "uniform_entropy_nats": float(torch.tensor(float(vocab_size)).log()),
        "ln2": float(torch.tensor(2.0).log()),
    }

    def _conditional(subset: Sequence[OutcomePositionRecord]) -> dict:
        """P(true | pair) for one subset, pooled and per-example.

        Two readings are reported because they answer different questions and
        can disagree sharply:

          * ``pooled`` = ``sum(true_mass) / sum(pair_mass)``. Mass-weighted, so
            it is dominated by the examples where the model actually put
            probability on the pair. This is the one to read: rows whose
            ``pair_mass`` is ~1e-4 contribute an essentially random ratio, and
            they would otherwise drown the signal.
          * ``per_example`` = the distribution of the raw per-row ratio,
            unweighted. Useful for seeing spread, but its mean is noise-heavy
            for exactly that reason.

        A ``pooled`` value near 0.5 means the model is calling the game by coin
        flip whenever it engages the outcome slot at all.
        """

        usable = [
            record for record in subset if record.p_true_given_pair is not None
        ]
        if not usable:
            return {"n_examples": 0}
        total_true = sum(record.true_mass for record in usable)
        total_pair = sum(record.pair_mass for record in usable)
        return {
            "n_examples": len(usable),
            "pooled": (total_true / total_pair) if total_pair > 0.0 else None,
            "per_example": _distribution(
                [float(record.p_true_given_pair) for record in usable]
            ),
            # The hard-decision version: ignoring magnitudes, how often is the
            # correct outcome token the larger of the two? 0.5 is a coin flip.
            "pair_argmax_correct_fraction": sum(
                record.pair_argmax_correct for record in usable
            )
            / float(len(usable)),
        }

    def _perspective_split(subset: Sequence[OutcomePositionRecord]) -> dict:
        """Break the pair-level behaviour out by p1 / p2 view.

        The outcome label is perspective-relative: the same game is ``[WIN]``
        from one side and ``[LOSS]`` from the other. So a model that has learned
        a standing preference for one class -- rather than reading the game --
        shows up here as ``prefers_win_fraction`` staying roughly CONSTANT
        across the two perspectives, while a model that actually reads the state
        must flip it. That comparison is the point of this block; the per-side
        ``p_true_given_pair`` is what it costs the model.
        """

        split: dict[str, dict] = {}
        for perspective in ("p1", "p2"):
            side = [record for record in subset if record.perspective == perspective]
            if not side:
                continue
            n_side = float(len(side))
            split[perspective] = {
                "n_examples": len(side),
                "true_is_win_fraction": sum(
                    record.true_token_id == WIN_ID for record in side
                )
                / n_side,
                # Which of the two the model leans toward, independent of which
                # is correct. Compare ACROSS perspectives, not against 0.5.
                "prefers_win_fraction": sum(
                    record.win_mass > record.loss_mass for record in side
                )
                / n_side,
                "mean_win_mass": sum(record.win_mass for record in side) / n_side,
                "mean_loss_mass": sum(record.loss_mass for record in side) / n_side,
                "pair_mass": _distribution([record.pair_mass for record in side]),
                "p_true_given_pair": _conditional(side),
            }
        return split

    def _subset(subset: Sequence[OutcomePositionRecord]) -> dict:
        if not subset:
            return {"n_examples": 0}
        ranks = torch.tensor([record.true_rank for record in subset], dtype=torch.float64)
        n = float(len(subset))
        return {
            "n_examples": len(subset),
            "t": _distribution([record.t for record in subset]),
            "pair_mass": _distribution([record.pair_mass for record in subset]),
            "true_mass": _distribution([record.true_mass for record in subset]),
            "other_pair_mass": _distribution([record.other_pair_mass for record in subset]),
            # The model's call BETWEEN the two outcome tokens, with "did it find
            # the slot" divided out. See _conditional.
            "p_true_given_pair": _conditional(subset),
            "by_perspective": _perspective_split(subset),
            "shown_token_mass": _distribution([record.shown_token_mass for record in subset]),
            "argmax_mass": _distribution([record.argmax_mass for record in subset]),
            "entropy_nats": _distribution([record.entropy_nats for record in subset]),
            "true_rank": {
                **_distribution([float(record.true_rank) for record in subset]),
                "top_1_fraction": float((ranks <= 1).sum()) / n,
                "top_2_fraction": float((ranks <= 2).sum()) / n,
                "top_5_fraction": float((ranks <= 5).sum()) / n,
                "top_10_fraction": float((ranks <= 10).sum()) / n,
                "top_50_fraction": float((ranks <= 50).sum()) / n,
            },
            "shown_token_rank": _distribution(
                [float(record.shown_token_rank) for record in subset]
            ),
            # The decisive behavioural counts. "argmax is the shown token" is the
            # copy prior winning outright; "argmax is in the pair" is the model
            # winning; the gap between "true rank is top-5" and "argmax is true"
            # is exactly the hedging the CE number hides.
            "argmax_equals_true_fraction": sum(
                record.argmax_token_id == record.true_token_id for record in subset
            )
            / n,
            "argmax_in_pair_fraction": sum(
                record.argmax_token_id in (WIN_ID, LOSS_ID) for record in subset
            )
            / n,
            "argmax_equals_shown_token_fraction": sum(
                record.argmax_token_id == record.shown_token_id for record in subset
            )
            / n,
            "pair_mass_over_uniform_ratio": None,
        }

    noised = [record for record in records if record.shown_differs_from_true]
    clean = [record for record in records if not record.shown_differs_from_true]

    # The measured quantities move by two orders of magnitude ACROSS the probed
    # band, not just between the band and lower t -- so a single mean over
    # [0.9, 1.0] would blend "canvas still has a few clean neighbours" with
    # "canvas is pure noise" and hide the trend that distinguishes context-driven
    # localization from genuine positional addressing. Bucketed at 0.025 so the
    # default band yields four cells.
    by_t_bucket: dict[str, dict] = {}
    for lower_edge in (0.900, 0.925, 0.950, 0.975):
        upper_edge = lower_edge + 0.025
        bucket = [
            record
            for record in noised
            # Top bucket is closed on the right so t == 1.0 is never dropped.
            if lower_edge <= record.t < upper_edge or (upper_edge >= 1.0 and record.t == 1.0)
        ]
        if bucket:
            by_t_bucket[f"{lower_edge:.3f}-{upper_edge:.3f}"] = _subset(bucket)

    return {
        "reference": reference,
        "all_examples": _subset(records),
        "position_1_noised": _subset(noised),
        "position_1_clean": _subset(clean),
        # Noised rows only: a clean position 1 is trivially correct and would
        # swamp the trend.
        "position_1_noised_by_t_bucket": by_t_bucket,
    }


# ---------------------------------------------------------------------------
# Per-arm driver
# ---------------------------------------------------------------------------


def probe_arm(
    *,
    arm_name: str,
    config_path: str | Path,
    split: str,
    n_examples: int,
    sample_mode: str,
    t_min: float,
    t_max: float,
    device: str,
    seed: int,
    use_ema: bool,
    checkpoint_name: str,
) -> dict:
    """Load one arm's checkpoint and probe it, returning the JSON payload.

    Parameters mirror the CLI flags. The checkpoint is resolved from the arm's
    OWN config (``storage.checkpoint_uri``) rather than from the arm name, so
    the probe can never disagree with the profile about where the weights live --
    the same rule ``tests/run_ablation_sweep.sh`` follows.

    Returns:
        A dict with ``arm``, ``checkpoint``, ``weights``, the arm's toggle
        settings and ``architecture_identity``, the ``summary`` block, and the
        full per-example ``records`` list.

    Raises:
        FileNotFoundError: the arm has no checkpoint yet (never started).
    """

    config = load_config(config_path)
    checkpoint_dir = _local_checkpoint_dir(config, StorageResolver())
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{arm_name}: no checkpoint at {checkpoint_path}")

    torch_device = torch.device(device)
    model, run_config = load_diagnostic_model(
        checkpoint_path, config, device=torch_device, use_raw=not use_ema
    )
    model.to(torch_device)
    model.eval()

    # Read-only metadata straight off the checkpoint, so each JSON is
    # self-describing about exactly which weights produced it.
    #
    # weights_only=False is REQUIRED, not lazy: save_checkpoint stores the
    # pickled ProjectConfig dataclass alongside the tensors, and weights_only=True
    # rejects the whole payload when any entry is not a plain tensor/primitive.
    # This matches every torch.load in the package (train/loop.py,
    # inference/sampler.py, viz/diagnostics.py, tests/run_ablation_sweep.sh). The
    # file was written by this repository's own training loop to a repo-local
    # path -- never third-party input -- so the unpickling surface is code we
    # already own and run.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(payload.get("global_step", 0))
    completed_epochs = int(payload.get("completed_epochs", 0))
    architecture_identity = payload.get("architecture_identity", "")
    del payload

    resolver = StorageResolver()
    loader, vocabulary, probed_indices = build_probe_dataloader(
        run_config,
        split=split,
        resolver=resolver,
        n_examples=n_examples,
        sample_mode=sample_mode,
    )

    # Two separate seeded generators. The t generator lives on CPU and is shared
    # by construction across arms (same seed, same batch order, same shapes), so
    # every arm is measured at the SAME per-example noise levels. The corruption
    # generator has to live on the compute device because corrupt_batch draws on
    # the target canvas's device.
    t_generator = torch.Generator().manual_seed(seed)
    corruption_generator = torch.Generator(device=torch_device).manual_seed(seed)

    records: list[OutcomePositionRecord] = []
    consumed = 0
    for batch in loader:
        rows = batch.target_canvas.shape[0]
        noise_levels = _draw_noise_levels(
            rows, t_min=t_min, t_max=t_max, generator=t_generator
        )
        records.extend(
            probe_batch(
                model,
                batch,
                vocabulary=vocabulary,
                config=run_config,
                device=torch_device,
                noise_levels=noise_levels,
                corruption_generator=corruption_generator,
                row_indices=probed_indices[consumed : consumed + rows],
            )
        )
        consumed += rows

    return {
        "arm": arm_name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        # Spelled out rather than a bare boolean: which weight set produced these
        # numbers is the single most misreadable thing about this file.
        "weights": "EMA (ema_model)" if use_ema else "TRAINED (raw model, final optimizer step)",
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "architecture_identity": architecture_identity,
        "toggles": {
            "frozen_input_kv": run_config.model.frozen_input_kv,
            "segment_embeddings": run_config.model.segment_embeddings,
            "per_segment_positions": run_config.model.per_segment_positions,
        },
        "probe": {
            "split": split,
            "sample_mode": sample_mode,
            "examples_probed": len(records),
            # Total windows available in the split, so a reader can see what
            # fraction of it the strided sample covers.
            "split_size": len(loader.dataset.dataset),
            "probed_dataset_indices": probed_indices,
            "batch_size": run_config.pipeline.batch_size,
            "t_range": [t_min, t_max],
            "seed": seed,
            "device": str(torch_device),
            "diffusion_process": run_config.diffusion.process,
            "forward_passes_per_example": 1,
            "self_conditioning_input": "zeros (first denoising step)"
            if run_config.model.self_conditioning
            else "disabled",
        },
        "summary": summarize(records, vocab_size=int(model.vocab_size)),
        "records": [asdict(record) for record in records],
    }


def _print_arm_summary(result: dict) -> None:
    """Print the four headline numbers so a run is readable without the JSON."""

    summary = result["summary"]
    noised = summary["position_1_noised"]
    print(f"  weights            : {result['weights']}")
    print(f"  steps / epochs     : {result['global_step']} / {result['completed_epochs']}")
    print(f"  architecture id    : {result['architecture_identity']}")
    if not noised.get("n_examples"):
        print("  (no genuinely-noised position-1 examples in this sample)")
        return
    print(f"  examples (noised@1): {noised['n_examples']}")
    print(
        f"  pair mass          : mean {noised['pair_mass']['mean']:.4f}  "
        f"median {noised['pair_mass']['median']:.4f}  "
        "([WIN]/[LOSS] are excluded from the noise support)"
    )
    print(
        f"  shown-noise mass   : mean {noised['shown_token_mass']['mean']:.4f}  "
        f"median {noised['shown_token_mass']['median']:.4f}"
    )
    print(
        f"  true-token rank    : median {noised['true_rank']['median']:.0f}  "
        f"mean {noised['true_rank']['mean']:.1f}  "
        f"top-5 {100.0 * noised['true_rank']['top_5_fraction']:.0f}%"
    )
    conditional = noised["p_true_given_pair"]
    if conditional.get("n_examples"):
        print(
            f"  P(true | pair)     : pooled {conditional['pooled']:.3f}   "
            f"per-example median {conditional['per_example']['median']:.3f}   "
            f"right-of-two {100.0 * conditional['pair_argmax_correct_fraction']:.0f}%"
            "   [0.5 = coin flip]"
        )
    print(
        f"  argmax == shown    : {100.0 * noised['argmax_equals_shown_token_fraction']:.0f}%   "
        f"argmax in pair: {100.0 * noised['argmax_in_pair_fraction']:.0f}%   "
        f"argmax == true: {100.0 * noised['argmax_equals_true_fraction']:.0f}%"
    )
    # Outcome-class balance, so a reader can immediately see whether the sample
    # actually contained both classes (a single-class sample makes "rank of the
    # true token" much weaker evidence than it looks).
    wins = sum(record["true_token"] == "[WIN]" for record in result["records"])
    print(f"  outcome balance    : {wins} [WIN] / {len(result['records']) - wins} [LOSS]")
    buckets = result["summary"]["position_1_noised_by_t_bucket"]
    if buckets:
        print("  pair mass by t     : " + "   ".join(
            f"{label} {stats['pair_mass']['mean']:.4f} (n={stats['n_examples']})"
            for label, stats in buckets.items()
        ))
    # Perspective is printed as a p1-vs-p2 comparison rather than as two
    # standalone rows, because the diagnostic content is entirely in whether
    # prefers-WIN moves between the sides. Two similar numbers = a standing
    # class preference; a large gap = the model is reading the game.
    for perspective, stats in noised.get("by_perspective", {}).items():
        print(
            f"  {perspective}                 : n={stats['n_examples']:<3} "
            f"truth is [WIN] {100.0 * stats['true_is_win_fraction']:3.0f}%   "
            f"model prefers [WIN] {100.0 * stats['prefers_win_fraction']:3.0f}%   "
            f"P(true|pair) {stats['p_true_given_pair']['pooled']:.3f}"
        )


def run(
    *,
    arms: Sequence[tuple[str, str]],
    out_dir: str | Path,
    split: str = "train",
    n_examples: int = 60,
    sample_mode: str = "strided",
    t_min: float = 0.9,
    t_max: float = 1.0,
    device: str = "cuda",
    seed: int = 20260808,
    use_ema: bool = False,
    checkpoint_name: str = "last.pt",
) -> list[Path]:
    """Probe every requested arm and write one JSON per arm under ``out_dir``.

    An arm without a checkpoint, or one that fails to load, is reported and
    SKIPPED rather than aborting the run -- the point of probing six arms is to
    come back with five results rather than with one traceback.

    Returns:
        The written JSON paths, in arm order.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arm_name, config_path in arms:
        print(f"\n=== {arm_name} ===", flush=True)
        try:
            result = probe_arm(
                arm_name=arm_name,
                config_path=config_path,
                split=split,
                n_examples=n_examples,
                sample_mode=sample_mode,
                t_min=t_min,
                t_max=t_max,
                device=device,
                seed=seed,
                use_ema=use_ema,
                checkpoint_name=checkpoint_name,
            )
        except Exception as error:  # noqa: BLE001 -- one bad arm must not kill the rest
            print(f"  SKIPPED: {type(error).__name__}: {error}", flush=True)
            continue
        target = out_path / f"outcome_position_probe_{arm_name}.json"
        target.write_text(
            json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        _print_arm_summary(result)
        print(f"  wrote {target}", flush=True)
        written.append(target)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the model's output distribution at canvas position 1 (the "
            "[WIN]/[LOSS] outcome slot) at high noise, per ablation arm. Separates "
            "'never found the position' from 'found it, learned the class, outvoted "
            "by the copy prior' -- two worlds with near-identical cross-entropy."
        )
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "arm to probe; repeatable. Default: every arm in ARMS (the five sweep "
            f"arms plus the all-false baseline). Valid: {', '.join(name for name, _ in ARMS)}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/output/ablations/position1_probe"),
        help="directory for the per-arm JSON files; the only write target",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="train",
        help="which replay selection to probe (default: train, the memorized subset)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=60,
        help=(
            "windows to probe per arm (0 = the whole split). Default 60 = 'a "
            "handful of batches' at batch_size 10"
        ),
    )
    parser.add_argument(
        "--sample-mode",
        choices=("strided", "head"),
        default="strided",
        help=(
            "how to pick those windows. 'strided' (default) spreads them evenly "
            "across the split so every replay, both perspectives and both outcome "
            "classes are represented; 'head' takes the first N in dataloader "
            "order, which is grouped by replay and can be single-class"
        ),
    )
    parser.add_argument(
        "--t-min", type=float, default=0.9, help="lower bound of the per-example noise level"
    )
    parser.add_argument(
        "--t-max", type=float, default=1.0, help="upper bound of the per-example noise level"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device; use cpu to stay off a GPU that is busy training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260808,
        help="seeds the noise levels and the corruption draws; identical across arms",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help=(
            "probe the EMA weights instead of the default TRAINED (raw) weights. "
            "The default is deliberately the trained weights: this measures what "
            "training produced, not what the sampler serves"
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last.pt",
        help="checkpoint filename inside each arm's checkpoint dir",
    )
    args = parser.parse_args(argv)

    for bound, name in ((args.t_min, "--t-min"), (args.t_max, "--t-max")):
        if not 0.0 <= bound <= 1.0:
            parser.error(f"{name} must be in [0, 1]; got {bound}")
    if args.t_min > args.t_max:
        parser.error(f"--t-min ({args.t_min}) must not exceed --t-max ({args.t_max})")

    by_name = dict(ARMS)
    if args.arm:
        unknown = [name for name in args.arm if name not in by_name]
        if unknown:
            parser.error(
                f"unknown arm(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(name for name, _ in ARMS)}"
            )
        selected = tuple((name, by_name[name]) for name in args.arm)
    else:
        selected = ARMS

    written = run(
        arms=selected,
        out_dir=args.out_dir,
        split=args.split,
        n_examples=args.n_examples,
        sample_mode=args.sample_mode,
        t_min=args.t_min,
        t_max=args.t_max,
        device=args.device,
        seed=args.seed,
        use_ema=args.ema,
        checkpoint_name=args.checkpoint_name,
    )
    print(f"\nwrote {len(written)} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
