"""Shared plumbing every inference test script in ``Test_Scripts/`` builds on.

Role in the larger system
-------------------------
``Model_Inference_Tests`` is a READ-ONLY consumer of an already-finished training
run. It answers one question: "how does a trained checkpoint behave on replays it
has never seen?" Nothing in this package trains, fine-tunes, mutates a
checkpoint, or writes anywhere except its own per-run output directory.

This module is the small shared layer under the individual test scripts. It owns
the three things every test needs and none of them should re-derive:

  1. **Which replays are the test split.** Held-out means held-out: every test
     scores windows from the replays the run put in its ``test`` group, and only
     those. See :func:`resolve_test_split_replays`.
  2. **The loaded model.** A 469 MB checkpoint is loaded ONCE per runner
     invocation and shared across every test through :class:`SharedResources`.
  3. **Deterministic example / batch construction** over those replays, using the
     production dataset + collate path so a test scores exactly what training
     scored. See :meth:`SharedResources.examples` and
     :meth:`SharedResources.dataloader`.

Everything heavier than a path lookup is lazy and memoized, so a runner that
executes only the cheap data-only test never pays for a GPU model load.

Depends on (calls into) the main package rather than reimplementing it:
``thesis_ml.config.load_config``, ``thesis_ml.pipeline.storage.StorageResolver``,
``thesis_ml.pipeline.train_pipeline._explicit_replay_selection``,
``thesis_ml.data.split.split_replays``, ``thesis_ml.data.windowing.load_window_manifest``,
``thesis_ml.data.dataset.SC2DiffusionDataset``,
``thesis_ml.data.collate.collate_diffusion_examples``,
``thesis_ml.viz.diagnostics.load_diagnostic_model``,
``thesis_ml.vocab.content_vocab.load_content_vocabulary``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import DatasetExample, SC2DiffusionDataset
from thesis_ml.data.split import split_replays
from thesis_ml.data.windowing import WindowManifestEntry, load_window_manifest
from thesis_ml.pipeline.storage import StorageResolver

# The training pipeline's own replay-selection helper. Imported (despite the
# leading underscore) rather than duplicated for the same reason
# ``viz/outcome_probe.py`` imports it: a second implementation of "which replays
# were held out" is the one bug that would silently invalidate every number this
# package produces.
from thesis_ml.pipeline.train_pipeline import _explicit_replay_selection
from thesis_ml.viz.diagnostics import load_diagnostic_model
from thesis_ml.vocab.content_vocab import ContentVocabulary, load_content_vocabulary


# Repository root (the Thesis_ML package root), derived from this file's location
# so nothing here embeds a machine-specific absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _UseConfiguredFog:
    """Sentinel: 'use the runner's configured fog rate'.

    A plain ``None`` default cannot express this, because ``None`` is itself a
    meaningful fog setting -- it tells ``SC2DiffusionDataset`` to draw fog per
    serving from the TRAINING distribution instead of pinning one rate. So the
    three possible fog conditions are:

      * ``USE_CONFIGURED_FOG`` -- the runner's ``--fog-rate`` / ``config.eval.fog_rate``
        (the default; deterministic, reproducible, and the configured eval condition);
      * a float -- pin every example to exactly that rate;
      * ``None`` -- draw per serving from ``config.fog.rate_distribution``, i.e.
        the same fog the model actually trained under.
    """

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid only
        return "USE_CONFIGURED_FOG"


USE_CONFIGURED_FOG = _UseConfiguredFog()


# ---------------------------------------------------------------------------
# Test-split resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestSplit:
    """The held-out replay group a run never trained or validated on.

    Attributes:
        replay_paths: parquet paths of the test-split replays, sorted by stem so
            the ordering is stable across machines and runs.
        replay_ids: the same replays as file stems (``match_..._game_state``).
        source: how the split was derived -- ``"recorded"`` when it was read from
            a run's ``replay_selection.json``, ``"config-seeded"`` when it was
            re-derived from the config's seeded split, or
            ``"config-explicit"`` for a config that names its replays outright.
        verified_against_recorded: True when a recorded ``replay_selection.json``
            was found AND matched the derived split exactly. This is the strong
            guarantee that the windows scored below are genuinely unseen.
    """

    replay_paths: tuple[str, ...]
    replay_ids: tuple[str, ...]
    source: str
    verified_against_recorded: bool


def resolve_test_split_replays(
    config: ProjectConfig,
    *,
    replay_selection_path: Path | None = None,
) -> TestSplit:
    """Determine which replays are held out for testing, and prove it.

    Two independent sources exist for the split and this function reconciles
    them rather than trusting either blindly:

      * the run's ``replay_selection.json`` (written by the training pipeline at
        the start of the run -- the authoritative record of what actually
        happened), and
      * the config's own split rule, re-derived here.

    When both are available and disagree, this raises: a mismatch means the
    checkpoint was trained on a different partition than the one about to be
    scored, which would make every "held-out" number in this package a lie.

    Parameters:
        config: the loaded run profile. Read for ``storage.data_uri``,
            ``pipeline.replay_glob`` and the split parameters.
        replay_selection_path: optional path to the run's recorded
            ``replay_selection.json``. When None, only the config rule is used.

    Returns:
        A :class:`TestSplit` describing the held-out replays.

    Raises:
        ValueError: the recorded selection and the config-derived split disagree,
            or the test group is empty.

    Calls: ``StorageResolver.list_files``, ``_explicit_replay_selection``,
    ``split_replays``, ``_select_replays``.
    """

    resolver = StorageResolver()
    corpus_paths = resolver.list_files(config.storage.data_uri, config.pipeline.replay_glob)
    if not corpus_paths:
        raise ValueError(
            f"no replays matched {config.pipeline.replay_glob!r} under {config.storage.data_uri}"
        )

    explicit = _explicit_replay_selection(list(corpus_paths), config)
    if explicit is not None:
        _train, _dev, test_paths = explicit
        source = "config-explicit"
    else:
        seeded = split_replays(
            corpus_paths,
            seed=config.pipeline.split_seed,
            test_fraction=config.pipeline.test_fraction,
            dev_fraction=config.pipeline.dev_fraction,
            train_count=config.pipeline.train_replay_count,
            dev_count=config.pipeline.validation_replay_count,
        )
        # Only train/dev are further trimmed downstream (by _select_replays);
        # the test group is exactly what the seeded split left over, so it is
        # taken straight from the split.
        test_paths = list(seeded.test)
        source = "config-seeded"

    if not test_paths:
        raise ValueError("the resolved test replay group is empty; nothing to evaluate")

    derived_ids = sorted(Path(path).stem for path in test_paths)

    verified = False
    if replay_selection_path is not None and replay_selection_path.exists():
        recorded = json.loads(replay_selection_path.read_text(encoding="utf-8"))
        recorded_ids = sorted(recorded.get("test_replay_ids", []))
        if recorded_ids and recorded_ids != derived_ids:
            raise ValueError(
                "the run's recorded test split does not match the split derived from "
                f"{config.pipeline.split_seed=}: recorded {len(recorded_ids)} replays, "
                f"derived {len(derived_ids)}. Refusing to score a partition the "
                "checkpoint may have trained on."
            )
        verified = bool(recorded_ids)
        source = "recorded" if verified else source

    # Sort by stem so the replay ordering (and therefore every "first N replays"
    # selection built on it) is identical on every machine.
    by_stem = {Path(path).stem: str(path) for path in test_paths}
    ordered_ids = tuple(sorted(by_stem))
    return TestSplit(
        replay_paths=tuple(by_stem[stem] for stem in ordered_ids),
        replay_ids=ordered_ids,
        source=source,
        verified_against_recorded=verified,
    )


def select_windows_per_replay(
    windows: Sequence[WindowManifestEntry],
    *,
    n_replays: int,
    n_windows_per_replay: int,
) -> list[WindowManifestEntry]:
    """Take the first N windows from each of the first M test replays.

    Deliberately deterministic and spread across replays rather than random: a
    "first N windows overall" selection would come entirely from one or two games
    (manifest order is grouped by replay and perspective), which makes any
    per-replay variation invisible.

    Parameters:
        windows: manifest entries already filtered to the test split.
        n_replays: how many distinct replays to draw from (``<= 0`` = all).
        n_windows_per_replay: windows to keep per replay (``<= 0`` = all).

    Returns:
        The selected entries, in manifest order.
    """

    replay_order: list[str] = []
    for window in windows:
        if window.replay_id not in replay_order:
            replay_order.append(window.replay_id)
    if n_replays > 0:
        allowed = set(replay_order[:n_replays])
    else:
        allowed = set(replay_order)

    per_replay_count: dict[str, int] = {}
    selected: list[WindowManifestEntry] = []
    for window in windows:
        if window.replay_id not in allowed:
            continue
        count = per_replay_count.get(window.replay_id, 0)
        if n_windows_per_replay > 0 and count >= n_windows_per_replay:
            continue
        selected.append(window)
        per_replay_count[window.replay_id] = count + 1
    return selected


def stride_indices(total: int, *, limit: int) -> list[int]:
    """Pick ``limit`` indices spread evenly across ``range(total)``.

    Same rule as ``viz.outcome_probe.select_probe_indices``' strided mode, kept
    here so a test can sub-sample any sequence (examples, windows) without
    dragging in a dataloader. ``limit <= 0`` or ``limit >= total`` returns
    everything.
    """

    if limit <= 0 or limit >= total:
        return list(range(total))
    return [(position * total) // limit for position in range(limit)]


# ---------------------------------------------------------------------------
# Shared, lazily-loaded resources
# ---------------------------------------------------------------------------


class SharedResources:
    """Loads the checkpoint, vocabulary, and test windows once for all tests.

    The runner creates exactly one of these and hands it to every test through
    the :class:`TestContext`. Each accessor memoizes, so the 469 MB checkpoint is
    read from disk once no matter how many tests ask for the model, and a run
    consisting only of data-only tests never touches the GPU at all.

    Attributes are intentionally accessed through methods rather than properties
    so it is obvious at the call site that the first call does real work.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        use_raw_weights: bool,
        fog_rate: float,
        seed: int,
        replay_selection_path: Path | None,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_raw_weights = use_raw_weights
        self.fog_rate = fog_rate
        self.seed = seed
        self.replay_selection_path = replay_selection_path

        # The user-supplied config, loaded eagerly: it is cheap and every code
        # path below needs it.
        self.user_config: ProjectConfig = load_config(config_path)

        self._model: Any | None = None
        self._run_config: ProjectConfig | None = None
        self._vocabulary: ContentVocabulary | None = None
        self._test_split: TestSplit | None = None
        self._test_windows: tuple[WindowManifestEntry, ...] | None = None
        self._checkpoint_facts: dict[str, Any] | None = None

    # -- split / windows ---------------------------------------------------

    def test_split(self) -> TestSplit:
        """The held-out replay group (memoized). See :func:`resolve_test_split_replays`."""

        if self._test_split is None:
            self._test_split = resolve_test_split_replays(
                self.user_config, replay_selection_path=self.replay_selection_path
            )
        return self._test_split

    def test_windows(self) -> tuple[WindowManifestEntry, ...]:
        """Every manifest window belonging to a test-split replay (memoized).

        Reads the run's EXISTING window manifest rather than re-tokenizing the
        replays: the manifest already covers the whole corpus, and rebuilding it
        would be both slow and a chance to disagree with what training used. A
        stale manifest raises inside ``load_window_manifest``, which is the
        correct outcome for a measurement tool.
        """

        if self._test_windows is None:
            windows = load_window_manifest(
                self.user_config.data.window_manifest_path,
                config=self.user_config,
                replay_paths=self.test_split().replay_paths,
            )
            if not windows:
                raise RuntimeError(
                    "no windows in "
                    f"{self.user_config.data.window_manifest_path} belong to the test split"
                )
            self._test_windows = windows
        return self._test_windows

    # -- model -------------------------------------------------------------

    def model(self) -> tuple[Any, ProjectConfig]:
        """Load the checkpoint's weights and the config they were trained under.

        Returns:
            ``(model, run_config)``. ``run_config`` is the user config with its
            ``model`` section replaced by the checkpoint's, so the sampler and
            loss see the architecture the weights actually have.

        Calls: ``thesis_ml.viz.diagnostics.load_diagnostic_model``, which also
        validates the checkpoint's stamped architecture identity and feature
        statistics identity before loading any tensor.
        """

        if self._model is None:
            model, run_config = load_diagnostic_model(
                self.checkpoint_path,
                self.user_config,
                device=self.device,
                use_raw=self.use_raw_weights,
            )
            model.to(self.device)
            model.eval()
            self._model = model
            self._run_config = run_config
        assert self._run_config is not None  # set together with _model
        return self._model, self._run_config

    def run_config(self) -> ProjectConfig:
        """The config with the checkpoint's ``model`` section (loads the model)."""

        return self.model()[1]

    def vocabulary(self) -> ContentVocabulary:
        """The content vocabulary the run tokenized with (memoized)."""

        if self._vocabulary is None:
            self._vocabulary = load_content_vocabulary(
                self.user_config.pipeline.token_dictionary_uri
            )
        return self._vocabulary

    def checkpoint_facts(self) -> dict[str, Any]:
        """Read-only provenance straight off the checkpoint file (memoized).

        Every artifact this package writes embeds this block, so a stray JSON or
        PNG can always be traced back to the exact weights that produced it.

        ``weights_only=False`` is required, not lazy: ``save_checkpoint`` pickles
        the whole ``ProjectConfig`` dataclass alongside the tensors. The file was
        written by this repository's own training loop to a repo-local path, so
        the unpickling surface is code we already own -- the same reasoning every
        other ``torch.load`` in this project records.
        """

        if self._checkpoint_facts is None:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self._checkpoint_facts = {
                "checkpoint": portable_path(self.checkpoint_path),
                "weights": "raw (final optimizer step)" if self.use_raw_weights else "EMA",
                "completed_epochs": int(payload.get("completed_epochs", 0)),
                "global_step": int(payload.get("global_step", 0)),
                "best_dev_loss": _optional_float(payload.get("best_dev_loss")),
                "architecture_identity": str(payload.get("architecture_identity", "")),
                "diffusion_process": str(payload.get("diffusion_process", "")),
                "debut_mode": bool(getattr(payload.get("config").data, "debut_mode", False))
                if payload.get("config") is not None
                else None,
            }
            del payload
        return self._checkpoint_facts

    # -- examples / batches -------------------------------------------------

    def resolve_fog(self, fog_rate_override: float | None | _UseConfiguredFog) -> float | None:
        """Turn a test's fog request into the value ``SC2DiffusionDataset`` takes.

        See :class:`_UseConfiguredFog` for the three conditions. Returning
        ``None`` means "draw per serving from the training fog distribution".
        """

        if isinstance(fog_rate_override, _UseConfiguredFog):
            return self.fog_rate
        return fog_rate_override

    def dataset(
        self,
        windows: Sequence[WindowManifestEntry],
        *,
        run_config: ProjectConfig | None = None,
        fog_rate_override: float | None | _UseConfiguredFog = USE_CONFIGURED_FOG,
    ) -> SC2DiffusionDataset:
        """Build the production dataset over the given test-split windows.

        By default fog is pinned to the configured evaluation condition, so every
        test scores the same visibility and re-running reproduces the same
        numbers. A test that needs the model's TRAINING visibility instead --
        because it is being compared against a baseline computed that way, or
        because the fogged loss class only exists when fog is actually drawn --
        passes ``fog_rate_override=None`` to sample per serving from
        ``config.fog.rate_distribution``.
        """

        config = run_config if run_config is not None else self.user_config
        return SC2DiffusionDataset(
            windows,
            config,
            self.vocabulary(),
            seed=config.pipeline.seed,
            fog_rate_override=self.resolve_fog(fog_rate_override),
        )

    def examples(
        self,
        *,
        n_replays: int,
        n_windows_per_replay: int,
        max_examples: int = 0,
        run_config: ProjectConfig | None = None,
        fog_rate_override: float | None | _UseConfiguredFog = USE_CONFIGURED_FOG,
    ) -> list[DatasetExample]:
        """Materialize ``DatasetExample`` objects for a selection of test windows.

        Parameters:
            n_replays: how many test replays to draw from (``<= 0`` = all 23).
            n_windows_per_replay: windows per replay (``<= 0`` = all).
            max_examples: hard cap applied afterwards by even striding, so a
                large selection is thinned without collapsing onto one replay.
            run_config: config to serve with; defaults to the user config.
            fog_rate_override: see :meth:`dataset`.

        Returns:
            The selected examples, in manifest order.
        """

        selected = select_windows_per_replay(
            self.test_windows(),
            n_replays=n_replays,
            n_windows_per_replay=n_windows_per_replay,
        )
        if not selected:
            raise RuntimeError("window selection produced no examples")
        dataset = self.dataset(
            selected, run_config=run_config, fog_rate_override=fog_rate_override
        )
        indices = stride_indices(len(dataset), limit=max_examples)
        return [dataset[index] for index in indices]

    def dataloader(
        self,
        *,
        n_replays: int,
        n_windows_per_replay: int,
        max_examples: int = 0,
        batch_size: int | None = None,
        run_config: ProjectConfig | None = None,
        retain_metadata: bool = False,
        fog_rate_override: float | None | _UseConfiguredFog = USE_CONFIGURED_FOG,
    ) -> tuple[DataLoader, list[int]]:
        """Build a deterministic, single-process loader over test-split windows.

        Shuffling is off and ``num_workers`` is 0 on purpose: a measurement tool
        should be reproducible and should not spawn workers that contend with
        whatever else is on the machine.

        ``retain_metadata`` keeps the per-example Python object graphs (input
        records, canvas metadata) on each batch. It defaults to False -- the
        training/validation setting -- because only a test that needs to read
        raw token records off a BATCH should pay for them.

        Returns:
            ``(loader, dataset_indices)`` where ``dataset_indices`` are the
            underlying dataset positions behind the loader's rows, in loader
            order -- so a per-row record can name the window it came from.
        """

        config = run_config if run_config is not None else self.user_config
        selected = select_windows_per_replay(
            self.test_windows(),
            n_replays=n_replays,
            n_windows_per_replay=n_windows_per_replay,
        )
        dataset = self.dataset(
            selected, run_config=config, fog_rate_override=fog_rate_override
        )
        indices = stride_indices(len(dataset), limit=max_examples)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size or config.pipeline.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=partial(
                collate_diffusion_examples,
                retain_metadata=retain_metadata,
                debut_mode=config.data.debut_mode,
            ),
        )
        return loader, indices


# ---------------------------------------------------------------------------
# Per-test context and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestContext:
    """Everything one test script receives from the runner.

    Attributes:
        shared: the loaded model / split / vocabulary, shared across all tests.
        out_dir: this test's own output directory (already created). It is the
            ONLY location a test may write to.
        run_dir: the parent run directory (``<model label>__<date>``), for tests
            that want to reference sibling output. Read-only by convention.
        model_label: human-readable identity of the weights under test, e.g.
            ``smallTrainingTestV3-epoch-0033``.
        device: the torch device tests should compute on.
        seed: base seed for every stochastic draw a test makes.
        n_replays / n_windows_per_replay / max_examples: the runner's sampling
            budget. A test may request FEWER (a sampler-bound test does), but
            should never silently request more.
        dpi: raster resolution for figures.
        fog_rate: the fixed fog rate every example is served at.
        extra: free-form runner options (``--option name=value``) for one-off
            overrides without changing a test's signature.
    """

    shared: SharedResources
    out_dir: Path
    run_dir: Path
    model_label: str
    device: torch.device
    seed: int
    n_replays: int
    n_windows_per_replay: int
    max_examples: int
    dpi: int
    fog_rate: float
    extra: dict[str, str] = field(default_factory=dict)

    def option_int(self, name: str, default: int) -> int:
        """Read an integer from ``--option name=value``, or return ``default``."""

        raw = self.extra.get(name)
        return default if raw is None else int(raw)

    def option_float(self, name: str, default: float) -> float:
        """Read a float from ``--option name=value``, or return ``default``."""

        raw = self.extra.get(name)
        return default if raw is None else float(raw)

    def provenance(
        self,
        *,
        uses_model: bool,
        fog_rate_override: float | None | _UseConfiguredFog = USE_CONFIGURED_FOG,
    ) -> dict[str, Any]:
        """Build the provenance block every test embeds in its JSON artifact.

        Parameters:
            uses_model: when False the checkpoint is never opened, so a data-only
                test stays free of a 469 MB read.
            fog_rate_override: the fog condition this test actually served its
                examples under. Recorded explicitly because two tests using
                different fog are NOT comparable, and a number without its fog
                condition attached invites exactly that mistake.

        Returns:
            A JSON-ready dict naming the weights, the split, the fog condition,
            and the sampling budget that produced the accompanying numbers.
        """

        split = self.shared.test_split()
        resolved_fog = self.shared.resolve_fog(fog_rate_override)
        block: dict[str, Any] = {
            "model_label": self.model_label,
            "config": portable_path(self.shared.config_path),
            "split": {
                "name": "test",
                "source": split.source,
                "verified_against_recorded_selection": split.verified_against_recorded,
                "n_replays_in_split": len(split.replay_ids),
                "replay_ids": list(split.replay_ids),
            },
            "sampling": {
                "n_replays": self.n_replays,
                "n_windows_per_replay": self.n_windows_per_replay,
                "max_examples": self.max_examples,
                "seed": self.seed,
                "device": str(self.device),
            },
            "fog": {
                # Spelled out rather than left as a bare number/null: which fog
                # condition produced a number is the single most misreadable
                # thing about comparing two of these artifacts.
                "condition": (
                    "training distribution (config.fog.rate_distribution, drawn per serving)"
                    if resolved_fog is None
                    else f"fixed rate {resolved_fog}"
                ),
                "fixed_rate": resolved_fog,
                "runner_configured_rate": self.fog_rate,
            },
        }
        if uses_model:
            block["checkpoint"] = self.shared.checkpoint_facts()
        return block


@dataclass
class TestResult:
    """What a test script hands back to the runner.

    Attributes:
        headline: two or three short lines printed in the runner's console
            summary and written into the run's ``SUMMARY.md``. This is what a
            reader sees without opening any artifact.
        artifacts: paths the test wrote, for the runner's manifest.
        metrics: small JSON-ready dict of the test's key numbers, merged into
            the run-level ``summary.json``.
    """

    headline: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lenient (salvaging) canvas decode -- diagnostic use only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalvagedCanvas:
    """A best-effort decode of a canvas that may violate the SPEC grammar.

    Attributes:
        timesteps: per-timestep ``entity_type -> count`` mappings, positionally
            indexed exactly as ``decode_canvas`` produces them, so
            ``extract_build_order`` assigns identical buckets.
        skipped_special_tokens: content-region positions holding a special token
            that the strict decoder would have aborted on.
        skipped_unknown_tokens: content-region positions holding an id absent
            from the vocabulary.
        trailing_partial_timestep: True when the content region did not end on a
            ``[DELIMITER]`` and a final unterminated group was kept anyway.
    """

    timesteps: list[dict[str, int]]
    skipped_special_tokens: int
    skipped_unknown_tokens: int
    trailing_partial_timestep: bool


def salvage_canvas_timesteps(
    token_ids: Sequence[int],
    vocabulary: ContentVocabulary,
) -> SalvagedCanvas:
    """Decode whatever is decodable from a canvas, ignoring grammar violations.

    **This is not the project grammar and must never be presented as it.**
    ``inference.decode.decode_canvas`` is the authority on what constitutes a
    valid canvas, and it deliberately returns NOTHING when validation fails --
    an ill-formed canvas is not a valid model output and should not be scored as
    one.

    This function exists for a narrower, diagnostic purpose. Because a single
    misplaced structural token (one ``[DELIMITER]`` in the wrong place, one
    stray ``[PAD]``) invalidates an entire window, the strict score cannot
    distinguish "the model produced garbage" from "the model produced a good
    build order wrapped in a malformed envelope". Those are very different
    failures. Salvaging the content lets both be reported side by side.

    The walk mirrors ``decode_canvas``'s content loop exactly, with three
    relaxations, each of which is COUNTED so the caller can report how much was
    forgiven:

      * a special token inside the content region is skipped rather than
        aborting the decode;
      * an id missing from the vocabulary is skipped rather than aborting;
      * a final group not terminated by ``[DELIMITER]`` is kept rather than
        discarded.

    Positions 0 and 1 are skipped unconditionally -- they are the structurally
    reserved ``[BOS]`` and outcome slots regardless of what the model emitted
    there. The content region ends at ``[END]`` or at the first ``[PAD]``,
    whichever comes first (the strict grammar requires that order; leniently we
    just take the earlier).

    Parameters:
        token_ids: the full generated canvas token id sequence.
        vocabulary: content vocabulary for id -> entity-type names.

    Returns:
        A :class:`SalvagedCanvas`. Its ``timesteps`` can be handed straight to
        ``eval.buildorder.extract_build_order``.
    """

    from thesis_ml.vocab.special_tokens import (
        BOS_ID,
        DELIMITER_ID,
        END_ID,
        EOS_ID,
        LOSS_ID,
        MASK_ID,
        PAD_ID,
        WIN_ID,
    )

    specials = {PAD_ID, END_ID, MASK_ID, WIN_ID, LOSS_ID, BOS_ID, EOS_ID}
    ids = list(token_ids)

    # Content region: after the two reserved slots, up to the first terminator.
    stops = [index for index in (_first_index(ids, END_ID), _first_index(ids, PAD_ID)) if index is not None]
    active_end = min(stops) if stops else len(ids)
    active = ids[2:active_end] if active_end > 2 else []

    names = vocabulary.id_to_name
    timesteps: list[dict[str, int]] = []
    current: dict[str, int] = {}
    skipped_special = 0
    skipped_unknown = 0

    for token_id in active:
        if token_id == DELIMITER_ID:
            timesteps.append(current)
            current = {}
            continue
        if token_id in specials:
            skipped_special += 1
            continue
        name = names.get(token_id)
        if name is None:
            skipped_unknown += 1
            continue
        current[name] = current.get(name, 0) + 1

    # Anything still in `current` is a group the canvas never closed with a
    # [DELIMITER]. The strict grammar rejects the whole canvas for this; we keep
    # the group and record that we did.
    trailing_partial = bool(current)
    if current:
        timesteps.append(current)

    return SalvagedCanvas(
        timesteps=timesteps,
        skipped_special_tokens=skipped_special,
        skipped_unknown_tokens=skipped_unknown,
        trailing_partial_timestep=trailing_partial,
    )


def _first_index(values: Sequence[int], target: int) -> int | None:
    """Index of the first ``target`` in ``values``, or None when absent."""

    try:
        return list(values).index(target)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------


def portable_path(path: str | Path) -> str:
    """Render a path relative to the repository root with forward slashes.

    Artifacts are read by other people on other machines, so no absolute
    machine-specific path is ever written into one. Paths outside the repository
    are returned as-is.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a pretty-printed, newline-terminated JSON artifact and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_text(lines: Sequence[str], path: Path) -> Path:
    """Write a newline-joined text artifact and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_csv(rows: Sequence[dict[str, Any]], columns: Sequence[str], path: Path) -> Path:
    """Write a CSV with an explicit column order and return its path.

    Uses ``csv`` rather than pandas so this package adds no dependency the
    training environment does not already have.
    """

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def save_figure(figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    """Write one matplotlib figure as PNG (raster) and SVG (vector).

    Mirrors ``viz.diagnostics._save_figure`` so figures produced here look and
    behave like the ones the existing diagnostics module writes.
    """

    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_" for character in stem
    )
    png_path = out_dir / f"{safe}.png"
    svg_path = out_dir / f"{safe}.svg"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, svg_path]


def _optional_float(value: Any) -> float | None:
    """Coerce a checkpoint field to float, returning None for missing/infinite."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
