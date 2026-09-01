"""Measure whether positional canvas CE starves the RARE-token learning signal.

Role in the larger system
-------------------------
READ-ONLY training-objective diagnostic, a direct follow-up to
``scripts/timestep_alignment_probe.py`` and
``diagnostics/011-timestep-alignment-loss-investigation.md``. It changes no
production loss, serializer, model, corruption process, or sampler, never writes
into a training run or into ``Model_Inference_Tests/``, and its only durable side
effect is a bounded JSON/text/CSV bundle under
``scripts/output/rare_token_signal_probe/``.

The question it answers, and why 011 could not answer it
--------------------------------------------------------
011 measured the alignment overcount POOLED over every content position and
concluded it was modest (1.5-3.2 wrong coordinates per semantic edit), because
"canonical sort-by-type serialization produces long runs of identical ids, and a
one-position shift across such a run changes nothing at those coordinates."

That damping is not free. A one-position shift mis-scores a coordinate exactly
when the token there differs from its neighbour -- i.e. exactly at RUN
BOUNDARIES. A token type that appears 40 times in a timestep is one long run and
is almost entirely interior; a token type that appears ONCE is entirely boundary.
So the pooled amplification is small precisely BECAUSE the damage is concentrated
on the rare types, and a pooled metric cannot see that concentration.

The rare types are the semantically pivotal ones. SPEC.md sec.5 sorts a timestep's
entities by SC2 source (unit-type) ID, so for Protoss the numerous economy and
gateway structures -- nexus(59), pylon(60), assimilator(61), gateway(62),
forge(63) -- sort BEFORE every tech-unlock building: fleetbeacon(64),
twilightcouncil(65), photoncannon(66), stargate(67), templararchive(68),
darkshrine(69), roboticsbay(70), roboticsfacility(71), cyberneticscore(72). A
single pylon miscount therefore displaces EVERY tech building in that timestep.

Three arms
----------
A. ``run_exposure_experiment`` -- model-independent. On real clean target
   canvases, closed-form per-token-type exposure to a one-position bounded shift:
   what fraction of each type's occurrences sit at a run boundary, the run length
   containing them, and the exact prefix count a model must reproduce to place
   them. No model, no checkpoint, no GPU. This arm alone establishes an
   objective-geometry property and cannot be confounded by the checkpoint having
   been trained under this objective.
B. ``run_signal_experiment`` -- observational. One denoiser forward pass per
   corruption level on real recorded replay windows with a named checkpoint,
   scoring each token type at THREE increasingly forgiving levels:
     1. positional recall  -- argmax equals the target at the exact coordinate
     2. timestep recall    -- argmax emits the type ANYWHERE in the ground-truth
                              timestep span (order-invariant; alignment forgiven)
     3. soft mass          -- mean probability the model assigns to the type at
                              its true coordinate, and the expected count it puts
                              on the type across the whole span
   These three discriminate the competing explanations that a single accuracy
   number confounds:
     positional low + timestep HIGH        -> misplacement (an alignment problem)
     positional low + timestep low + soft mass elevated
                                           -> dilution (knows it, loses argmax)
     positional low + timestep low + soft mass ~0
                                           -> washed out (no signal survived)
C. ``loss_share`` inside arm B -- what fraction of the weighted training
   objective each token type's targets actually account for. This is the
   numerical half of the owner's concern: "numerically the effect is small, but
   semantically the effect could be quite large."

Causal caveat, restated in code because it governs how the numbers may be read
-------------------------------------------------------------------------------
The probed checkpoint was itself trained under positional CE. Arm B is
OBSERVATIONAL. It can show WHICH of the three failure shapes the trained model is
in -- which is what determines what an ablation should change -- but it cannot
prove the objective caused that shape. Only a matched training ablation can.
Arm A is model-independent and is the only arm that supports an objective-geometry
claim on its own.

Reuses ``scripts/timestep_alignment_probe.py`` for canvas segmentation, split
resolution/verification, window selection, batching, coupled corruption, and
provenance rather than reimplementing any of it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# `scripts/` is not an installed package, so the sibling probe is imported by
# path. Everything structural (segmentation, split verification, window
# selection, coupled corruption seeding, provenance) is REUSED from it so the two
# diagnostics can never disagree about what a timestep is.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timestep_alignment_probe import (  # noqa: E402
    WindowIdentity,
    _batch_to_device,
    _forward_canvas_logits,
    _format_duration,
    _ratio,
    _shutdown_dataloader,
    parse_canvas_layout,
    portable_path,
    predicted_structure,
    resolve_split_replays,
    select_windows,
    verify_recorded_split,
)

from thesis_ml.config import load_config  # noqa: E402
from thesis_ml.data.collate import DiffusionBatch  # noqa: E402
from thesis_ml.data.dataset import SC2DiffusionDataset  # noqa: E402
from thesis_ml.data.windowing import load_window_manifest  # noqa: E402
from thesis_ml.model.loss import CanvasCrossEntropyLoss  # noqa: E402
from thesis_ml.pipeline.storage import StorageResolver  # noqa: E402
from thesis_ml.pipeline.train_pipeline import (  # noqa: E402
    _ensure_window_manifest,
    _make_dataloader,
    _materialize_file,
    _materialize_replay_paths,
)
from thesis_ml.train.corruption import corrupt_batch  # noqa: E402
from thesis_ml.viz.diagnostics import load_diagnostic_model  # noqa: E402
from thesis_ml.vocab.content_vocab import (  # noqa: E402
    ContentVocabulary,
    load_content_vocabulary,
)
from thesis_ml.vocab.special_tokens import (  # noqa: E402
    CONTENT_TOKEN_OFFSET,
    DELIMITER_ID,
)

DEFAULT_OUTPUT_DIR = Path("scripts/output/rare_token_signal_probe")
DEFAULT_CONFIG = Path("configs/smallTrainingTestV3.yaml")
DEFAULT_CHECKPOINT = Path(
    "tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt"
)
DEFAULT_NOISE_LEVELS = (0.75, 0.90, 0.99, 1.00)

# Tech-unlock structures: the buildings whose PRESENCE changes what an opponent
# can build next, which is the semantics the owner's concern is about. They are
# named explicitly (rather than derived from a frequency cut) so the "semantically
# pivotal" set is a stated editorial choice and not an artefact of the data.
# Names match `data/Token_Dictionary.json` exactly.
TECH_BUILDING_NAMES = frozenset(
    {
        # Protoss
        "cyberneticscore",
        "twilightcouncil",
        "roboticsfacility",
        "roboticsbay",
        "stargate",
        "fleetbeacon",
        "templararchive",
        "darkshrine",
        "forge",
        # Terran
        "barracks",
        "factory",
        "starport",
        "engineeringbay",
        "armory",
        "ghostacademy",
        "fusioncore",
        "techlab",
        "barrackstechlab",
        "factorytechlab",
        "starporttechlab",
        # Zerg
        "spawningpool",
        "roachwarren",
        "banelingnest",
        "evolutionchamber",
        "hydraliskden",
        "lurkerden",
        "spire",
        "greaterspire",
        "infestationpit",
        "ultraliskcavern",
        "nydusnetwork",
        "lair",
        "hive",
    }
)

# Frequency buckets over a token type's occurrences-per-timestep in the probed
# corpus. Roughly logarithmic, because the population of interest spans three
# orders of magnitude: a probe averages ~7.6 occurrences per timestep while a
# twilight council averages ~0.018. Linear buckets would put every rare type in
# one bin and hide exactly the gradient this probe exists to show.
FREQUENCY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("f1_ultra_rare_lt0_05", 0.0, 0.05),
    ("f2_rare_0_05_to_0_25", 0.05, 0.25),
    ("f3_uncommon_0_25_to_1_5", 0.25, 1.5),
    ("f4_common_1_5_to_4", 1.5, 4.0),
    ("f5_dominant_4_plus", 4.0, float("inf")),
)


# ---------------------------------------------------------------------------
# Arm A -- model-independent shift exposure
# ---------------------------------------------------------------------------


def boundary_flags(content: Sequence[int]) -> np.ndarray:
    """Mark each content coordinate that a one-position LEFT shift would break.

    A deletion at offset ``k`` inside this timestep makes the prediction read
    ``pred[j] = content[j + 1]`` for every ``j >= k``, with the group's
    ``[DELIMITER]`` shifting into the final content slot. Coordinate ``j`` is
    therefore mis-scored if and only if ``content[j + 1] != content[j]`` -- that
    is, if and only if ``j`` ends a run of identical ids.

    This is the exact mechanism by which canonical sort-by-type serialization
    "absorbs" the shift: interior coordinates of a long run cost nothing, and the
    entire cost lands on run boundaries.

    Args:
        content: the timestep's content token ids in canonical serialized order.

    Returns:
        A boolean array of the same length; ``True`` where a left shift would
        change the token at that coordinate. The last content coordinate is
        always ``True`` because the delimiter shifts into it.

    Calls: nothing; a pure elementwise comparison.
    """

    ids = np.asarray(content, dtype=np.int64)
    if ids.size == 0:
        return np.zeros(0, dtype=bool)
    # Append the delimiter as the sentinel that shifts into the final slot.
    following = np.concatenate([ids[1:], np.array([DELIMITER_ID], dtype=np.int64)])
    return following != ids


def run_lengths(content: Sequence[int]) -> np.ndarray:
    """Length of the identical-id run each coordinate belongs to.

    Reported alongside the boundary fraction because it is the intuitive form of
    the same fact: a type occurring 40 times in a timestep sits in a run of 40
    and is 38/40 interior, while a singleton sits in a run of 1 and is 1/1
    boundary.

    Args:
        content: the timestep's content token ids in canonical serialized order.

    Returns:
        An int array of the same length giving each coordinate's run length.

    Calls: nothing.
    """

    ids = np.asarray(content, dtype=np.int64)
    lengths = np.zeros(ids.size, dtype=np.int64)
    start = 0
    for index in range(1, ids.size + 1):
        if index == ids.size or ids[index] != ids[start]:
            lengths[start:index] = index - start
            start = index
    return lengths


def accumulate_exposure(
    content: Sequence[int], totals: dict[int, dict[str, float]]
) -> None:
    """Fold one timestep's per-token-type shift exposure into pooled totals.

    For every occurrence of every token type in this timestep it accumulates:

    - ``occurrences``: the denominator for every ratio below.
    - ``boundary_occurrences``: occurrences that a left shift WOULD mis-score,
      given the shift reaches them. Divided by ``occurrences`` this is the
      conditional probability that this type pays a wrong-coordinate penalty when
      an upstream count error displaces it.
    - ``expected_hits``: the same event marginalized over a deletion offset drawn
      uniformly from the timestep's content slots. An occurrence at coordinate
      ``j`` in a group of length ``n`` is reached by ``j + 1`` of the ``n``
      possible deletion offsets, so its unconditional exposure is
      ``boundary[j] * (j + 1) / n``. Drawing the offset uniformly over slots is
      the same as deleting an occurrence chosen in proportion to token frequency,
      which is the realistic error.
    - ``run_length_sum``: for the interior-versus-boundary reading.
    - ``prefix_sum``: the number of content tokens preceding the occurrence
      inside its own timestep. This is the count the model must reproduce EXACTLY
      for the occurrence to land on its trained coordinate, so it measures how
      much upstream arithmetic the type's learning signal is conditioned on.

    Args:
        content: the timestep's content token ids in canonical serialized order.
        totals: mutable accumulator keyed by token id; updated in place.

    Returns:
        None. ``totals`` is mutated.

    Calls: ``boundary_flags``, ``run_lengths``.
    """

    length = len(content)
    if length == 0:
        return
    boundaries = boundary_flags(content)
    lengths = run_lengths(content)
    for coordinate, token_id in enumerate(content):
        entry = totals.setdefault(
            int(token_id),
            {
                "occurrences": 0.0,
                "boundary_occurrences": 0.0,
                "expected_hits": 0.0,
                "run_length_sum": 0.0,
                "prefix_sum": 0.0,
            },
        )
        entry["occurrences"] += 1.0
        if boundaries[coordinate]:
            entry["boundary_occurrences"] += 1.0
            entry["expected_hits"] += (coordinate + 1) / length
        entry["run_length_sum"] += float(lengths[coordinate])
        entry["prefix_sum"] += float(coordinate)


def run_exposure_experiment(
    batches: Sequence[DiffusionBatch], *, max_canvases: int
) -> dict[str, object]:
    """Arm A: per-token-type exposure to a one-position bounded shift.

    Uses ONLY the clean targets carried by the batches -- no logits, no
    checkpoint, no GPU -- so its result is a property of the serialization and the
    objective, not of anything a model learned.

    Args:
        batches: collated production batches; only ``target_canvas`` and
            ``canvas_attention_mask`` are read.
        max_canvases: bound on how many canvas rows to scan.

    Returns:
        A JSON-ready dict with ``canvases_scored``, ``timesteps_scored``, and
        ``per_token`` (token id -> pooled exposure statistics).

    Calls: ``parse_canvas_layout``, ``accumulate_exposure``.
    """

    totals: dict[int, dict[str, float]] = {}
    canvases = 0
    timesteps = 0
    for batch in batches:
        targets = batch.target_canvas.detach().cpu().numpy()
        active = batch.canvas_attention_mask.detach().cpu().numpy().astype(bool)
        for row in range(targets.shape[0]):
            if canvases >= max_canvases:
                break
            active_length = int(active[row].sum())
            token_row = targets[row, :active_length].tolist()
            layout = parse_canvas_layout(token_row, active_length)
            for span in layout.timesteps:
                if span.content_length == 0:
                    continue
                accumulate_exposure(
                    token_row[span.content_start : span.content_end], totals
                )
                timesteps += 1
            canvases += 1
        if canvases >= max_canvases:
            break

    per_token = {}
    for token_id, entry in totals.items():
        occurrences = entry["occurrences"]
        per_token[token_id] = {
            "occurrences": int(occurrences),
            "boundary_fraction": _ratio(entry["boundary_occurrences"], occurrences),
            "expected_hits_per_occurrence": _ratio(entry["expected_hits"], occurrences),
            "mean_run_length": _ratio(entry["run_length_sum"], occurrences),
            "mean_prefix_count": _ratio(entry["prefix_sum"], occurrences),
        }
    return {
        "canvases_scored": canvases,
        "timesteps_scored": timesteps,
        "per_token": per_token,
    }


# ---------------------------------------------------------------------------
# Arm B -- observational per-token-type signal on real checkpoint logits
# ---------------------------------------------------------------------------

# Accumulator field order for the per-(token, t) matrix. Every field is a POOLED
# sum or count; ratios are formed once at the end, never by averaging per-window
# ratios whose supports differ.
SIGNAL_FIELDS = (
    "target_positions",        # scored content coordinates whose target is this type
    "noised_target_positions", # ... of those, genuinely corrupted ones
    "positional_tp",           # argmax == target at the exact coordinate
    "predicted_positions",     # argmax == this type at any scored content coordinate
    "timestep_overlap",        # sum over spans of min(pred_count, target_count)
    "soft_mass_at_target",     # sum of p(type) at coordinates whose target is this type
    "soft_expected_in_spans",  # expected count of this type over spans that contain it
    "target_count_in_spans",   # target count over those same spans (the calibration pair)
    "ce_sum_at_target",        # unweighted CE charged at this type's target coordinates
    "weighted_ce_sum_at_target",  # the same CE under production class weights
    "spans_with_target",       # spans whose target contains this type at least once
    "spans_with_any_prediction",  # ... of those, spans where argmax emits it at least once
    # The base-rate control. Soft mass inside spans that DO contain the type is
    # only evidence of timestep-level knowledge if the model puts LESS mass on
    # the type inside spans that do NOT contain it. A model that sprays the
    # corpus base rate uniformly would score identically on both.
    "soft_expected_in_absent_spans",
    "absent_spans",
    "absent_span_positions",
    "present_span_positions",
)
SIGNAL_FIELD_INDEX = {name: index for index, name in enumerate(SIGNAL_FIELDS)}


def score_signal_batch(
    *,
    canvas_logits: torch.Tensor,
    batch: DiffusionBatch,
    changed_positions: torch.Tensor,
    class_weights: torch.Tensor,
    identities: Sequence[WindowIdentity],
    noise_level: float,
    accumulators: dict[str, np.ndarray],
    per_window_rows: list[dict[str, object]],
    vocab_size: int,
) -> None:
    """Fold one batch of canvas logits into the per-(token type, t) accumulators.

    Everything is derived from the SAME logits and targets, so the positional,
    timestep-local, and soft views are directly comparable rather than being three
    separately-scoped measurements.

    Only CONTENT coordinates inside ground-truth timestep spans are scored here.
    Structural targets -- the outcome, ``[DELIMITER]``, ``[END]``, and semantic
    ``[PAD]`` -- lie outside every span and are 011's subject, not this probe's.

    Args:
        canvas_logits: ``[rows, positions, vocab]`` clean-state logits.
        batch: the production batch these logits were produced from.
        changed_positions: boolean mask of positions the corruption actually
            replaced, used to separate genuinely noised targets from survivors.
        class_weights: the live ``CanvasCrossEntropyLoss`` per-class weights.
        identities: one ``WindowIdentity`` per row, for the per-window CSV.
        noise_level: the corruption level ``t`` these logits were produced at.
        accumulators: mutable ``{t_key: [len(SIGNAL_FIELDS), vocab]}`` sums.
        per_window_rows: mutable list of per-window CSV rows.
        vocab_size: full vocabulary width, the accumulator's second dimension.

    Returns:
        None. ``accumulators`` and ``per_window_rows`` are mutated.

    Calls: ``parse_canvas_layout``, ``predicted_structure``.
    """

    targets = batch.target_canvas
    scored_mask = batch.canvas_loss_mask
    log_probabilities = torch.log_softmax(canvas_logits, dim=-1)
    per_position_ce = F.nll_loss(
        log_probabilities.transpose(1, 2), targets, reduction="none"
    )
    predicted_tokens = canvas_logits.argmax(dim=-1)
    weights = class_weights.to(canvas_logits.device)[batch.class_labels.clamp_min(0)]

    probabilities_cpu = log_probabilities.exp().detach().cpu().numpy()
    ce_cpu = per_position_ce.detach().cpu().numpy()
    predicted_cpu = predicted_tokens.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    scored_cpu = scored_mask.detach().cpu().numpy().astype(bool)
    changed_cpu = changed_positions.detach().cpu().numpy().astype(bool)
    weights_cpu = weights.detach().cpu().numpy()

    key = f"{noise_level:.2f}"
    matrix = accumulators.setdefault(
        key, np.zeros((len(SIGNAL_FIELDS), vocab_size), dtype=np.float64)
    )
    field = SIGNAL_FIELD_INDEX

    for row in range(targets_cpu.shape[0]):
        active_length = int(batch.canvas_attention_mask[row].sum().item())
        target_row = targets_cpu[row, :active_length].tolist()
        layout = parse_canvas_layout(target_row, active_length)
        structure = predicted_structure(
            predicted_cpu[row, :active_length].tolist(), active_length
        )

        window_target_positions = 0
        window_positional_tp = 0
        window_timestep_overlap = 0
        window_tech_target = 0
        window_tech_positional_tp = 0
        window_tech_timestep_overlap = 0

        for span in layout.timesteps:
            if span.content_length == 0:
                continue
            start, end = span.content_start, span.content_end
            span_scored = scored_cpu[row, start:end]
            if not span_scored.any():
                continue
            span_target = targets_cpu[row, start:end]
            span_predicted = predicted_cpu[row, start:end]
            span_changed = changed_cpu[row, start:end]
            span_ce = ce_cpu[row, start:end]
            span_weight = weights_cpu[row, start:end]
            span_probabilities = probabilities_cpu[row, start:end, :]

            # --- level 1: exact-coordinate scoring, the production view -------
            correct = (span_predicted == span_target) & span_scored
            np.add.at(matrix[field["target_positions"]], span_target[span_scored], 1.0)
            np.add.at(
                matrix[field["noised_target_positions"]],
                span_target[span_scored & span_changed],
                1.0,
            )
            np.add.at(matrix[field["positional_tp"]], span_target[correct], 1.0)
            np.add.at(
                matrix[field["predicted_positions"]], span_predicted[span_scored], 1.0
            )
            np.add.at(
                matrix[field["ce_sum_at_target"]],
                span_target[span_scored],
                span_ce[span_scored],
            )
            np.add.at(
                matrix[field["weighted_ce_sum_at_target"]],
                span_target[span_scored],
                (span_ce * span_weight)[span_scored],
            )
            # p(target) at each coordinate is exp(-CE) by construction.
            np.add.at(
                matrix[field["soft_mass_at_target"]],
                span_target[span_scored],
                np.exp(-span_ce[span_scored]),
            )

            # --- level 2: order-invariant scoring inside the span -------------
            # min(predicted_count, target_count) per type is the multiset overlap:
            # the number of occurrences the model gets right once placement inside
            # the timestep is forgiven entirely.
            target_counts = np.bincount(
                span_target[span_scored], minlength=vocab_size
            ).astype(np.float64)
            predicted_counts = np.bincount(
                span_predicted[span_scored], minlength=vocab_size
            ).astype(np.float64)
            overlap = np.minimum(target_counts, predicted_counts)
            matrix[field["timestep_overlap"]] += overlap

            present = target_counts > 0
            matrix[field["spans_with_target"]][present] += 1.0
            matrix[field["spans_with_any_prediction"]][
                present & (predicted_counts > 0)
            ] += 1.0

            # --- level 3: soft expected count over the whole span -------------
            # Summing p(type | coordinate) across the span gives the model's
            # expected number of that type in the timestep. Compared against the
            # target count it is a calibration check that never depends on argmax.
            expected_counts = span_probabilities[span_scored].sum(axis=0)
            span_positions = float(span_scored.sum())
            matrix[field["soft_expected_in_spans"]][present] += expected_counts[present]
            matrix[field["target_count_in_spans"]][present] += target_counts[present]
            matrix[field["present_span_positions"]][present] += span_positions
            # Base-rate control, over the complement: types the span does not
            # contain. Normalizing each side by its own scored-position total
            # makes the two comparable despite differing span lengths.
            absent = ~present
            matrix[field["soft_expected_in_absent_spans"]][absent] += expected_counts[
                absent
            ]
            matrix[field["absent_spans"]][absent] += 1.0
            matrix[field["absent_span_positions"]][absent] += span_positions

            window_target_positions += int(span_scored.sum())
            window_positional_tp += int(correct.sum())
            window_timestep_overlap += int(overlap.sum())

            tech_mask = _TECH_ID_MASK[: vocab_size]
            window_tech_target += int(target_counts[tech_mask].sum())
            window_tech_positional_tp += int(
                np.bincount(span_target[correct], minlength=vocab_size)[tech_mask].sum()
            )
            window_tech_timestep_overlap += int(overlap[tech_mask].sum())

        per_window_rows.append(
            {
                "t": round(noise_level, 4),
                "window": identities[row].as_key(),
                "replay_id": identities[row].replay_id,
                "content_target_positions": window_target_positions,
                "positional_recall": _ratio(
                    window_positional_tp, window_target_positions
                ),
                "timestep_recall": _ratio(
                    window_timestep_overlap, window_target_positions
                ),
                "tech_target_positions": window_tech_target,
                "tech_positional_recall": _ratio(
                    window_tech_positional_tp, window_tech_target
                ),
                "tech_timestep_recall": _ratio(
                    window_tech_timestep_overlap, window_tech_target
                ),
                "predicted_delimiters": structure["delimiter_count"],
                "target_delimiters": len(layout.delimiter_indices),
            }
        )


# Populated once in `run_probe` from the loaded vocabulary; kept module level so
# `score_signal_batch`'s inner loop does not rebuild it per span.
_TECH_ID_MASK: np.ndarray = np.zeros(0, dtype=bool)


def summarize_token_signal(
    matrix: np.ndarray,
    *,
    vocabulary: ContentVocabulary,
    timesteps_scored: int,
    total_weighted_ce: float,
    min_occurrences: int,
) -> list[dict[str, object]]:
    """Turn one noise level's pooled accumulator into per-token-type rows.

    Every ratio is formed once from pooled numerators and denominators.

    Args:
        matrix: ``[len(SIGNAL_FIELDS), vocab]`` pooled sums for one ``t``.
        vocabulary: content vocabulary, used to name ids and tag tech buildings.
        timesteps_scored: number of scored spans at this ``t``, the denominator
            for occurrences-per-timestep.
        total_weighted_ce: pooled weighted CE over ALL scored positions at this
            ``t``, the denominator for each type's loss share.
        min_occurrences: types with fewer target occurrences are dropped, so a
            handful of samples cannot produce a headline ratio.

    Returns:
        A list of JSON-ready per-token rows, sorted by descending occurrences.

    Calls: ``_ratio``.
    """

    field = SIGNAL_FIELD_INDEX
    id_to_name = vocabulary.id_to_name
    rows: list[dict[str, object]] = []
    for token_id in range(matrix.shape[1]):
        occurrences = matrix[field["target_positions"], token_id]
        if occurrences < min_occurrences or token_id < CONTENT_TOKEN_OFFSET:
            continue
        name = id_to_name.get(token_id, f"id_{token_id}")
        spans_with_target = matrix[field["spans_with_target"], token_id]
        rows.append(
            {
                "token_id": token_id,
                "token_name": name,
                "is_tech_building": name in TECH_BUILDING_NAMES,
                "target_positions": int(occurrences),
                "occurrences_per_timestep": _ratio(occurrences, timesteps_scored),
                "positional_recall": _ratio(
                    matrix[field["positional_tp"], token_id], occurrences
                ),
                "timestep_recall": _ratio(
                    matrix[field["timestep_overlap"], token_id], occurrences
                ),
                "span_presence_recall": _ratio(
                    matrix[field["spans_with_any_prediction"], token_id],
                    spans_with_target,
                ),
                "mean_probability_at_target": _ratio(
                    matrix[field["soft_mass_at_target"], token_id], occurrences
                ),
                "expected_over_target_count": _ratio(
                    matrix[field["soft_expected_in_spans"], token_id],
                    matrix[field["target_count_in_spans"], token_id],
                ),
                # Per-scored-position soft rate inside spans that contain the
                # type, divided by the same rate inside spans that do not.
                # 1.0 means the model is spraying a base rate and has no
                # timestep-level knowledge of this type; >1 means it does.
                "present_absent_rate_ratio": _ratio(
                    _ratio(
                        matrix[field["soft_expected_in_spans"], token_id],
                        matrix[field["present_span_positions"], token_id],
                    )
                    or 0.0,
                    _ratio(
                        matrix[field["soft_expected_in_absent_spans"], token_id],
                        matrix[field["absent_span_positions"], token_id],
                    )
                    or 0.0,
                ),
                "mean_ce_at_target_nats": _ratio(
                    matrix[field["ce_sum_at_target"], token_id], occurrences
                ),
                "weighted_loss_share": _ratio(
                    matrix[field["weighted_ce_sum_at_target"], token_id],
                    total_weighted_ce,
                ),
                "predicted_positions": int(
                    matrix[field["predicted_positions"], token_id]
                ),
            }
        )
    rows.sort(key=lambda row: -row["target_positions"])
    return rows


def bucket_token_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Pool per-token rows into frequency buckets plus the tech-building set.

    Pooling re-derives each ratio from summed numerators and denominators
    reconstructed from the per-token rows, so a bucket is never the mean of its
    members' ratios.

    Args:
        rows: per-token rows from ``summarize_token_signal``.

    Returns:
        bucket name -> pooled statistics.

    Calls: ``_ratio``.
    """

    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rate = row["occurrences_per_timestep"] or 0.0
        for name, low, high in FREQUENCY_BUCKETS:
            if low <= rate < high:
                groups[f"frequency/{name}"].append(row)
                break
        if row["is_tech_building"]:
            groups["set/tech_buildings"].append(row)
        else:
            groups["set/non_tech"].append(row)
        # Pooled over every scored content type, so the bucket table can be tied
        # back to the pooled content rows in diagnostics/011.
        groups["set/all_content"].append(row)

    summary: dict[str, dict[str, object]] = {}
    for name, members in groups.items():
        positions = sum(row["target_positions"] for row in members)
        if not positions:
            continue

        def pooled(metric: str) -> float | None:
            """Rebuild a numerator from per-token ratio * per-token support."""
            total = 0.0
            support = 0.0
            for row in members:
                value = row[metric]
                if value is None:
                    continue
                total += float(value) * row["target_positions"]
                support += row["target_positions"]
            return _ratio(total, support)

        summary[name] = {
            "token_types": len(members),
            "target_positions": positions,
            "positional_recall": pooled("positional_recall"),
            "timestep_recall": pooled("timestep_recall"),
            "span_presence_recall": pooled("span_presence_recall"),
            "mean_probability_at_target": pooled("mean_probability_at_target"),
            "expected_over_target_count": pooled("expected_over_target_count"),
            "present_absent_rate_ratio": pooled("present_absent_rate_ratio"),
            "mean_ce_at_target_nats": pooled("mean_ce_at_target_nats"),
            "weighted_loss_share": sum(
                float(row["weighted_loss_share"] or 0.0) for row in members
            ),
        }
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Execute the configured arms and return the full JSON-ready report.

    Calls: ``load_config``, ``load_content_vocabulary``, ``_materialize_*``,
    ``resolve_split_replays``, ``verify_recorded_split``, ``select_windows``,
    ``SC2DiffusionDataset``, ``_make_dataloader``, ``run_exposure_experiment``,
    ``load_diagnostic_model``, ``corrupt_batch``, ``_forward_canvas_logits``,
    ``score_signal_batch``, ``summarize_token_signal``, ``bucket_token_rows``.
    """

    global _TECH_ID_MASK

    started_at = time.perf_counter()
    config_path = args.config.resolve()
    config = load_config(config_path)
    resolver = StorageResolver()
    token_dictionary = _materialize_file(
        config.pipeline.token_dictionary_uri, config.storage.local_cache_dir, resolver
    )
    vocabulary = load_content_vocabulary(token_dictionary)
    replay_paths = _materialize_replay_paths(config, resolver)
    _ensure_window_manifest(replay_paths, config, vocabulary)

    split_paths = resolve_split_replays(
        replay_paths, config=config, split_name=args.split
    )
    replay_selection_path = (
        args.replay_selection
        if args.replay_selection is not None
        else Path(config.storage.log_uri) / "replay_selection.json"
    )
    split_verification = verify_recorded_split(
        split_paths, split_name=args.split, replay_selection_path=replay_selection_path
    )
    manifest_windows = load_window_manifest(
        config.data.window_manifest_path, config=config, replay_paths=list(split_paths)
    )
    windows = select_windows(
        manifest_windows,
        max_examples=args.max_examples,
        windows_per_replay=args.windows_per_replay,
        position=args.window_position,
    )
    if not windows:
        raise ValueError(f"the {args.split} split contains no manifest windows")

    identities = [
        WindowIdentity(
            replay_id=window.replay_id,
            perspective=window.perspective_player,
            start_timestep=window.start_timestep,
            end_timestep=window.end_timestep,
        )
        for window in windows
    ]

    vocab_size = vocabulary.vocab_size
    _TECH_ID_MASK = np.zeros(vocab_size, dtype=bool)
    for token_id, name in vocabulary.id_to_name.items():
        if name in TECH_BUILDING_NAMES:
            _TECH_ID_MASK[token_id] = True
    missing_tech = TECH_BUILDING_NAMES - set(vocabulary.name_to_id)
    if missing_tech:
        raise ValueError(
            "TECH_BUILDING_NAMES contains names absent from the configured "
            f"dictionary: {sorted(missing_tech)}"
        )

    print(
        "rare_token_signal_probe configuration\n"
        f"  config={portable_path(config_path)}\n"
        f"  checkpoint={portable_path(args.checkpoint)}\n"
        f"  weights={'raw' if args.raw else 'ema'}\n"
        f"  split={args.split} split_replays={len(split_paths)} "
        f"manifest_windows={len(manifest_windows)} selected_windows={len(windows)}\n"
        f"  noise_levels={[round(level, 4) for level in args.noise_level]} "
        f"window_position={args.window_position}\n"
        f"  device={args.device} num_workers={args.num_workers} seed={args.seed} "
        f"batch_size={config.pipeline.batch_size}\n"
        f"  tech_building_ids={int(_TECH_ID_MASK.sum())} "
        f"min_occurrences={args.min_occurrences}",
        file=sys.stderr,
        flush=True,
    )

    dataset = SC2DiffusionDataset(
        windows, config, vocabulary, seed=config.pipeline.seed, fog_rate_override=None
    )
    dataset.set_epoch(args.dataset_epoch)
    loader_config = replace(
        config, pipeline=replace(config.pipeline, num_workers=args.num_workers)
    )
    loader = _make_dataloader(dataset, loader_config, shuffle=False, device="cpu")
    batches: list[DiffusionBatch] = []
    batch_identities: list[list[WindowIdentity]] = []
    try:
        cursor = 0
        for batch in loader:
            size = batch.target_canvas.shape[0]
            batches.append(batch)
            batch_identities.append(identities[cursor : cursor + size])
            cursor += size
    finally:
        _shutdown_dataloader(loader)

    report: dict[str, object] = {}

    # ---- Arm A: model-independent shift exposure --------------------------
    exposure_started = time.perf_counter()
    exposure = run_exposure_experiment(batches, max_canvases=args.exposure_canvases)
    exposure["per_token"] = {
        str(token_id): dict(
            stats,
            token_name=vocabulary.id_to_name.get(token_id, f"id_{token_id}"),
            is_tech_building=vocabulary.id_to_name.get(token_id) in TECH_BUILDING_NAMES,
        )
        for token_id, stats in exposure["per_token"].items()
    }
    report["exposure"] = exposure
    print(
        f"arm_a_exposure canvases={exposure['canvases_scored']} "
        f"timesteps={exposure['timesteps_scored']} "
        f"elapsed={_format_duration(time.perf_counter() - exposure_started)}",
        file=sys.stderr,
        flush=True,
    )

    if args.exposure_only:
        report["provenance"] = _provenance(
            args,
            config_path=config_path,
            windows=windows,
            split_verification=split_verification,
            elapsed=time.perf_counter() - started_at,
            device_name=None,
        )
        return report

    # ---- Arm B/C: observational per-token signal ---------------------------
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but no CUDA device is visible; refusing to "
            "report GPU behavior from a CPU fallback"
        )
    model, run_config = load_diagnostic_model(
        args.checkpoint, config, device=device, use_raw=args.raw
    )
    model.to(device)
    model.eval()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    criterion = CanvasCrossEntropyLoss(config)
    class_weights = criterion.class_weights.detach()
    accumulators: dict[str, np.ndarray] = {}
    per_window_rows: list[dict[str, object]] = []
    # Pooled totals used as ratio denominators: scored spans and the full weighted
    # objective at each level, so a type's loss share is measured against the real
    # objective and not against content alone.
    spans_by_level: dict[str, int] = defaultdict(int)
    weighted_ce_by_level: dict[str, float] = defaultdict(float)

    total_units = len(batches) * len(args.noise_level)
    completed = 0
    forward_started = time.perf_counter()
    for batch_index, batch in enumerate(batches):
        device_batch = _batch_to_device(batch, device)
        for noise_level in args.noise_level:
            # Coupled corruption, identical to the 011 probe: an identically
            # seeded generator is rebuilt before every call so the Bernoulli draw
            # and the replacement tokens are shared across levels and the
            # corrupted sets are nested. The sweep therefore changes which
            # truthful anchors survive rather than swapping in unrelated canvases.
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + batch_index)
            corruption = corrupt_batch(
                input_token_ids=device_batch.input_token_ids,
                target_canvas=device_batch.target_canvas,
                process=run_config.diffusion.process,
                schedule=run_config.diffusion.schedule,
                vocab_size=int(model.vocab_size),
                generator=generator,
                t=noise_level,
                canvas_noise_mask=device_batch.canvas_loss_mask,
            )
            canvas_logits = _forward_canvas_logits(
                model,
                input_token_ids=device_batch.input_token_ids,
                input_attention_mask=device_batch.input_attention_mask,
                input_lengths=device_batch.input_lengths,
                input_features=device_batch.input_features,
                noised_canvas=corruption.noised_canvas,
                canvas_attention_mask=device_batch.canvas_attention_mask,
            )

            key = f"{noise_level:.2f}"
            # Whole-objective weighted CE at this level, the loss-share denominator.
            with torch.no_grad():
                log_probabilities = torch.log_softmax(canvas_logits, dim=-1)
                position_ce = F.nll_loss(
                    log_probabilities.transpose(1, 2),
                    device_batch.target_canvas,
                    reduction="none",
                )
                weights = class_weights.to(device)[
                    device_batch.class_labels.clamp_min(0)
                ] * device_batch.canvas_loss_mask.to(position_ce.dtype)
                weighted_ce_by_level[key] += float((position_ce * weights).sum().item())

            score_signal_batch(
                canvas_logits=canvas_logits,
                batch=device_batch,
                changed_positions=corruption.changed_positions,
                class_weights=class_weights,
                identities=batch_identities[batch_index],
                noise_level=noise_level,
                accumulators=accumulators,
                per_window_rows=per_window_rows,
                vocab_size=vocab_size,
            )
            for row in range(device_batch.target_canvas.shape[0]):
                active_length = int(
                    device_batch.canvas_attention_mask[row].sum().item()
                )
                layout = parse_canvas_layout(
                    device_batch.target_canvas[row, :active_length].tolist(),
                    active_length,
                )
                spans_by_level[key] += sum(
                    1 for span in layout.timesteps if span.content_length
                )

            completed += 1
            elapsed = time.perf_counter() - forward_started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (total_units - completed) / rate if rate else 0.0
            print(
                f"  forward {completed}/{total_units} t={noise_level:.2f} "
                f"rate={rate:.2f}/s elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(remaining)}",
                file=sys.stderr,
                flush=True,
            )

    signal: dict[str, object] = {"per_token_by_t": {}, "buckets_by_t": {}}
    for key, matrix in sorted(accumulators.items()):
        rows = summarize_token_signal(
            matrix,
            vocabulary=vocabulary,
            timesteps_scored=spans_by_level[key],
            total_weighted_ce=weighted_ce_by_level[key],
            min_occurrences=args.min_occurrences,
        )
        signal["per_token_by_t"][key] = rows
        signal["buckets_by_t"][key] = bucket_token_rows(rows)
    signal["spans_by_t"] = dict(spans_by_level)
    signal["weighted_ce_sum_by_t"] = dict(weighted_ce_by_level)
    signal["per_window"] = per_window_rows
    report["signal"] = signal
    report["provenance"] = _provenance(
        args,
        config_path=config_path,
        windows=windows,
        split_verification=split_verification,
        elapsed=time.perf_counter() - started_at,
        device_name=device_name,
    )
    return report


def _provenance(
    args: argparse.Namespace,
    *,
    config_path: Path,
    windows: Sequence[object],
    split_verification: Mapping[str, object],
    elapsed: float,
    device_name: str | None,
) -> dict[str, object]:
    """Repository-relative provenance block; never embeds a workstation path.

    Calls: ``portable_path``.
    """

    return {
        "config": portable_path(config_path),
        "checkpoint": portable_path(args.checkpoint),
        "weights": "raw" if args.raw else "ema",
        "split": args.split,
        "split_verification": dict(split_verification),
        "windows": [
            f"{window.replay_id}:{window.perspective_player}:"
            f"{window.start_timestep}-{window.end_timestep}"
            for window in windows
        ],
        "noise_levels": [round(level, 4) for level in args.noise_level],
        "seed": args.seed,
        "device": device_name,
        "num_workers": args.num_workers,
        "elapsed_seconds": round(elapsed, 3),
        "tech_building_names": sorted(TECH_BUILDING_NAMES),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_summary(report: Mapping[str, object]) -> str:
    """Render the human-readable text summary written beside the JSON artifact.

    Calls: ``_format_optional``.
    """

    lines: list[str] = ["Rare-token learning-signal probe"]
    provenance = report.get("provenance", {})
    lines.append(
        f"config={provenance.get('config')} split={provenance.get('split')} "
        f"weights={provenance.get('weights')} device={provenance.get('device')}"
    )

    exposure = report.get("exposure")
    if exposure:
        lines.append("")
        lines.append(
            "ARM A -- model-independent shift exposure "
            f"({exposure['canvases_scored']} canvases, "
            f"{exposure['timesteps_scored']} timesteps)"
        )
        lines.append(
            "  A one-position left shift mis-scores a coordinate exactly at a run"
        )
        lines.append(
            "  boundary. boundary_fraction is therefore P(this type pays a"
        )
        lines.append("  wrong-coordinate penalty | an upstream shift reaches it).")
        rows = sorted(
            exposure["per_token"].values(),
            key=lambda row: -row["occurrences"],
        )
        header = (
            f"  {'token':<24}{'occ':>9}{'occ/step':>10}{'runlen':>9}"
            f"{'bound_frac':>12}{'exp_hits':>10}{'prefix':>9}  tech"
        )
        lines.append(header)
        steps = max(1, exposure["timesteps_scored"])
        for row in rows[: 40]:
            lines.append(
                f"  {row['token_name']:<24}{row['occurrences']:>9}"
                f"{row['occurrences'] / steps:>10.2f}"
                f"{_format_optional(row['mean_run_length'], 2):>9}"
                f"{_format_optional(row['boundary_fraction'], 4):>12}"
                f"{_format_optional(row['expected_hits_per_occurrence'], 4):>10}"
                f"{_format_optional(row['mean_prefix_count'], 1):>9}"
                f"  {'Y' if row['is_tech_building'] else ''}"
            )

    signal = report.get("signal")
    if signal:
        lines.append("")
        lines.append(
            "ARM B/C -- observational per-token signal (one denoiser pass per t)"
        )
        lines.append(
            "  positional = argmax right at the exact coordinate (production view)"
        )
        lines.append(
            "  timestep   = argmax emits the type anywhere in its ground-truth span"
        )
        lines.append(
            "  p@target   = mean probability on the type at its true coordinate"
        )
        lines.append(
            "  exp/target = model's expected count over the span / true count"
        )
        lines.append(
            "  pres/abs   = soft rate inside spans that HAVE the type / spans that"
        )
        lines.append(
            "               do not. 1.0 = sprayed base rate, no timestep knowledge."
        )
        for key in sorted(signal["buckets_by_t"]):
            lines.append(f"  t={key}")
            lines.append(
                f"    {'bucket':<28}{'types':>7}{'targets':>10}{'position':>10}"
                f"{'timestep':>10}{'presence':>10}{'p@target':>10}"
                f"{'exp/target':>12}{'pres/abs':>10}{'CE':>9}{'lossShare':>11}"
            )
            for name, stats in sorted(signal["buckets_by_t"][key].items()):
                lines.append(
                    f"    {name:<28}{stats['token_types']:>7}"
                    f"{stats['target_positions']:>10}"
                    f"{_format_optional(stats['positional_recall'], 4):>10}"
                    f"{_format_optional(stats['timestep_recall'], 4):>10}"
                    f"{_format_optional(stats['span_presence_recall'], 4):>10}"
                    f"{_format_optional(stats['mean_probability_at_target'], 4):>10}"
                    f"{_format_optional(stats['expected_over_target_count'], 3):>12}"
                    f"{_format_optional(stats['present_absent_rate_ratio'], 2):>10}"
                    f"{_format_optional(stats['mean_ce_at_target_nats'], 3):>9}"
                    f"{_format_optional(stats['weighted_loss_share'], 5):>11}"
                )

        worst_key = max(signal["per_token_by_t"], key=float)
        lines.append("")
        lines.append(f"  tech buildings individually at t={worst_key}")
        lines.append(
            f"    {'token':<24}{'targets':>9}{'position':>10}{'timestep':>10}"
            f"{'presence':>10}{'p@target':>10}{'exp/target':>12}"
            f"{'pres/abs':>10}{'CE':>9}"
        )
        for row in signal["per_token_by_t"][worst_key]:
            if not row["is_tech_building"]:
                continue
            lines.append(
                f"    {row['token_name']:<24}{row['target_positions']:>9}"
                f"{_format_optional(row['positional_recall'], 4):>10}"
                f"{_format_optional(row['timestep_recall'], 4):>10}"
                f"{_format_optional(row['span_presence_recall'], 4):>10}"
                f"{_format_optional(row['mean_probability_at_target'], 4):>10}"
                f"{_format_optional(row['expected_over_target_count'], 3):>12}"
                f"{_format_optional(row['present_absent_rate_ratio'], 2):>10}"
                f"{_format_optional(row['mean_ce_at_target_nats'], 3):>9}"
            )

    lines.append("")
    lines.append(
        "READ CAREFULLY: Arm A is model-independent and proves a property of the "
        "serialization plus objective. Arm B is OBSERVATIONAL -- the probed "
        "checkpoint was itself trained under positional CE, so it identifies WHICH "
        "failure shape the model is in but cannot establish that the objective "
        "caused it. Only a matched training ablation can."
    )
    return "\n".join(lines) + "\n"


def write_per_window_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write the per-window CSV beside the JSON artifact.

    Calls: nothing.
    """

    if not rows:
        return
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: object, digits: int) -> str:
    """Render an optional float, or ``-`` when a slice had no support."""

    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"unserializable value of type {type(value)!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether positional canvas cross-entropy starves the learning "
            "signal for rare, semantically pivotal SC2 tokens, with a "
            "model-independent shift-exposure arm and an observational "
            "per-token-type recall arm."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="load raw weights; the default is the EMA weights the sampler serves",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--replay-selection", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--dataset-epoch", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=48)
    parser.add_argument("--windows-per-replay", type=int, default=1)
    parser.add_argument("--window-position", choices=("first", "last"), default="first")
    parser.add_argument(
        "--noise-level",
        type=float,
        action="append",
        default=None,
        help=f"repeatable corruption level (default: {list(DEFAULT_NOISE_LEVELS)})",
    )
    parser.add_argument(
        "--exposure-canvases",
        type=int,
        default=24,
        help="canvas rows scanned by the model-independent arm (default: 24)",
    )
    parser.add_argument(
        "--exposure-only",
        action="store_true",
        help="run only the model-independent arm; no checkpoint, no GPU",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=20,
        help="drop token types with fewer target occurrences (default: 20)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.noise_level is None:
        args.noise_level = list(DEFAULT_NOISE_LEVELS)
    for level in args.noise_level:
        if not 0.0 <= level <= 1.0:
            parser.error("--noise-level must lie in [0, 1]")
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.windows_per_replay <= 0:
        parser.error("--windows-per-replay must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.min_occurrences < 1:
        parser.error("--min-occurrences must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    output_path = args.output
    if output_path is None:
        suffix = "exposure" if args.exposure_only else args.split
        output_path = DEFAULT_OUTPUT_DIR / f"{args.config.stem}-{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = format_summary(report)
    summary_path = output_path.with_suffix(".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    signal = report.get("signal")
    if signal:
        write_per_window_csv(
            signal["per_window"], output_path.with_suffix(".per_window.csv")
        )
    print(summary, end="")
    print(f"json_artifact={portable_path(output_path)}")
    print(f"summary_artifact={portable_path(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
