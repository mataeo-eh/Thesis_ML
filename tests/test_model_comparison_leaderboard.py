"""Unit coverage for the cross-model leaderboard test.

Role in the larger system
-------------------------
``Model_Inference_Tests/Test_Scripts/test_08_model_comparison_leaderboard.py``
appends one row per scored checkpoint to a single shared CSV that accumulates
across every run of the inference suite. That file is the only place in the
project where a measurement is written NEXT TO earlier measurements rather than
into a fresh directory, so the failure mode is not a wrong number -- it is a
silently corrupted or truncated history of every model scored so far.

These tests pin exactly that:

* the append never loses, reorders, or misaligns an existing row, including when
  the column set changes between two runs;
* ``eval_condition_key`` -- the "are these two rows comparable at all" handle --
  is stable, order-independent, and sensitive to every field it claims to cover;
* ``dedupe_key`` is equal precisely when a row is redundant;
* the run date-time is recovered from the runner's directory name, so a row is
  attributable to the run directory beside it;
* the shared CSV lands at the root of ``output/`` and nowhere else.

Nothing here loads a checkpoint, touches CUDA, or reads replay data, so the whole
file runs in the ordinary CPU suite.
"""

from __future__ import annotations

import csv
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


# The inference suite runs as plain files, not as an installed package, so its
# own directory has to be importable before the test module (which does
# `from inference_test_api import ...`) can be loaded.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "Model_Inference_Tests"
MODULE_PATH = PACKAGE_DIR / "Test_Scripts" / "test_08_model_comparison_leaderboard.py"


def _load_leaderboard_module():
    """Import the leaderboard test module by path, under a non-collectable name.

    The spec name deliberately does NOT start with ``test_``: pytest is already
    collecting this file, and a second module whose name matches the collection
    pattern would be picked up as a test module in its own right.

    Returns:
        The imported module object.
    """

    for entry in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PACKAGE_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location("leaderboard_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


leaderboard = _load_leaderboard_module()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV back as ``(header, rows)`` for assertions."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(item) for item in reader]
    return header, rows


# ---------------------------------------------------------------------------
# Runner contract
# ---------------------------------------------------------------------------


def test_declares_the_metadata_the_runner_requires() -> None:
    """Discovery imports the module and reads these names off it directly."""

    assert leaderboard.TEST_NAME == "model_comparison_leaderboard"
    assert leaderboard.TEST_TITLE
    assert leaderboard.TEST_DESCRIPTION
    assert leaderboard.TEST_OUTPUTS
    assert leaderboard.USES_MODEL is True
    assert leaderboard.REQUIRES_DEBUT_FINETUNE is False
    assert callable(leaderboard.run)


class _StubParameter:
    """Minimal stand-in for a torch parameter: just a size and a grad flag."""

    def __init__(self, count: int, *, trainable: bool = True) -> None:
        self._count = count
        self.requires_grad = trainable

    def numel(self) -> int:
        return self._count


def _build_a_row(tmp_path: Path) -> dict[str, object]:
    """Drive ``_build_row`` with stubs, so the CSV schema can be asserted directly.

    Everything ``_build_row`` reads is a plain attribute or method call, so the
    real model / config / checkpoint can be replaced by namespaces. This exercises
    the actual row assembly -- column names, ordering, and value formatting --
    without a 469 MB checkpoint or a GPU.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = tmp_path / "epoch-0033.pt"
    checkpoint_file.write_bytes(b"checkpoint bytes" * 128)

    context = SimpleNamespace(
        model_label="smallTrainingTestV3-epoch-0033",
        run_dir=tmp_path / "smallTrainingTestV3-epoch-0033__2026-Aug-26_02-41PM",
        seed=20260826,
        shared=SimpleNamespace(
            config_path=tmp_path / "smallTrainingTestV3.yaml",
            checkpoint_path=checkpoint_file,
            test_split=lambda: SimpleNamespace(
                source="recorded", replay_ids=("r1", "r2", "r3")
            ),
            checkpoint_facts=lambda: {
                "checkpoint": "tests/output/run/checkpoints/best/epoch-0033.pt",
                "weights": "EMA",
                "completed_epochs": 33,
                "global_step": 33693,
                "best_dev_loss": 0.189912,
                "architecture_identity": "dense-multinomial-SC2-v2+frozen_input_kv",
                "debut_mode": False,
            },
        ),
    )
    model = SimpleNamespace(
        parameters=lambda: [_StubParameter(1000), _StubParameter(500, trainable=False)]
    )
    run_config = SimpleNamespace(
        model=SimpleNamespace(
            d_model=384,
            layers=12,
            heads=6,
            ffn=1536,
            qk_norm=True,
            self_conditioning=True,
            frozen_input_kv=True,
            segment_embeddings=False,
            per_segment_positions=False,
            rope_theta=500000.0,
        ),
        pipeline=SimpleNamespace(batch_size=6),
        diffusion=SimpleNamespace(
            process="uniform",
            schedule=SimpleNamespace(
                name="linear",
                t_distribution="power",
                t_distribution_power=2.0,
                min=0.0,
                max=1.0,
                t_one_fraction=0.05,
            ),
        ),
    )
    headline_log = SimpleNamespace(
        bits_per_token=2.0,
        perplexity=4.0,
        loss=0.0266,
        accuracy={"noised": 0.985, "ground_truth_preserved": 0.9999},
        macro_f1={"noised": 0.901, "ground_truth_preserved": 0.9997},
    )

    return leaderboard._build_row(
        context=context,
        model=model,
        run_config=run_config,
        headline_log=headline_log,
        grid_bits={0.5: 0.5, 1.0: 3.0},
        uniform_t_bits=1.75,
        t_grid=(0.5, 1.0),
        n_windows=240,
        window_selection_key="abcdef0123",
        fog_label="training distribution (per serving)",
        vocab_size=291,
        uniform_prior_bits=8.1849,
        bits_saved=6.1849,
    )


def test_row_reports_effective_token_choices_and_never_names_it_perplexity(
    tmp_path: Path,
) -> None:
    """The exponent column must not claim to be a perplexity.

    ``2 ** bits_per_token`` is arithmetically what an autoregressive LM's
    perplexity computes, but a discrete-diffusion denoiser assigns no sequence
    likelihood, so publishing the column under that name would invite lining the
    value up against published LM perplexities. The source field on
    ``ValidationLog`` still carries the old name; this pins that the name stops at
    the module boundary and never reaches the CSV.
    """

    row = _build_a_row(tmp_path)

    assert "perplexity" not in row
    assert row["effective_token_choices"] == 4.0
    assert row["effective_token_choices_t_1.00"] == 8.0  # 2 ** 3.0

    # Each bits column is immediately followed by its own effective-choices
    # partner, and the inference-condition pair sits directly after the headline
    # pair. A reader scanning left to right must never meet an exponent before
    # the bits value it is the exponent of.
    columns = list(row)
    assert columns[3:7] == [
        "bits_per_token",
        "effective_token_choices",
        "bits_per_token_t_1.00",
        "effective_token_choices_t_1.00",
    ]


def test_inference_condition_is_reported_once_not_duplicated(tmp_path: Path) -> None:
    """t=1.0 belongs to the headline block, not also to the per-t curve block.

    It is part of the default grid, so without the skip it would appear twice --
    a second column carrying the same number is a second source of truth that can
    drift when one of them is edited.
    """

    row = _build_a_row(tmp_path)

    assert list(row).count("bits_per_token_t_1.00") == 1
    # The other grid level still gets its plain bits column in the curve block.
    assert row["bits_per_token_t_0.50"] == 0.5
    assert "effective_token_choices_t_0.50" not in row


def test_inference_condition_column_is_the_deployed_sampler_start() -> None:
    """The constant must stay at the terminal prior the sampler begins from."""

    assert leaderboard.INFERENCE_T == 1.0
    assert leaderboard._effective_choices(0.0) == 1.0
    assert leaderboard._effective_choices(1.0) == 2.0
    assert leaderboard._effective_choices(3.0) == 8.0


def test_row_carries_the_score_architecture_weights_and_condition(
    tmp_path: Path,
) -> None:
    """One row has to answer "which model, how big, scored what, under what"."""

    row = _build_a_row(tmp_path)

    # when it was run -- recovered from the runner's directory name
    assert row["run_datetime"] == "2026-08-26 14:41:00"
    assert row["model_label"] == "smallTrainingTestV3-epoch-0033"
    # the score, and its per-corruption-level curve
    assert row["bits_per_token"] == 2.0
    assert row["bits_per_token_uniform_t"] == 1.75
    assert row["bits_per_token_t_0.50"] == 0.5
    assert row["bits_per_token_t_1.00"] == 3.0
    # the architecture: shape, size, params, toggles
    assert row["arch_shape"] == "d384-L12-H6-F1536"
    assert row["params_total"] == 1500
    assert row["params_trainable"] == 1000
    assert (row["d_model"], row["layers"], row["heads"], row["ffn"]) == (384, 12, 6, 1536)
    assert row["frozen_input_kv"] == "true"
    assert row["segment_embeddings"] == "false"
    # the weights it came from
    assert row["completed_epochs"] == 33
    assert row["global_step"] == 33693
    assert row["checkpoint_mb"] == 0.0
    # the evaluation condition, and its comparability handle
    assert row["n_windows"] == 240
    assert row["window_selection_key"] == "abcdef0123"
    assert row["vocab_size"] == 291
    assert row["t_grid"] == "0.50/1.00"
    assert row["eval_condition_key"]
    assert row["dedupe_key"]


def test_uniform_prior_comparison_is_measured_at_the_inference_condition(
    tmp_path: Path,
) -> None:
    """The prior gap must be computed from the t=1.0 score, not the headline.

    At the training t-distribution most scored positions already hold the correct
    token, so the fraction reads ~99% for any half-trained model and separates
    nothing. At t=1.0 the canvas carries no information, so the gap to log2(V) is
    the model's real contribution from the input replay alone.
    """

    row = _build_a_row(tmp_path)

    # The stub passes bits_saved = 6.1849 = 8.1849 - 3.0, i.e. computed against
    # the t=1.0 score of 3.0 rather than the headline 2.0.
    assert row["bits_removed_vs_uniform_prior_t_1.00"] == 6.1849
    assert row["fraction_of_uniform_prior_removed_t_1.00"] == round(6.1849 / 8.1849, 4)

    # The unsuffixed names must not exist: they would read as applying to the
    # headline number, which is exactly the misreading the suffix prevents.
    assert "bits_removed_vs_uniform_prior" not in row
    assert "fraction_of_uniform_prior_removed" not in row


def test_window_selection_key_distinguishes_equal_sized_selections() -> None:
    """Two runs can both score 240 windows and score 240 DIFFERENT windows.

    The count alone cannot tell those apart, so without this hash the
    comparability key would call two unrelated measurements comparable.
    """

    def window(replay_id: str, player: str, start: int, end: int) -> SimpleNamespace:
        return SimpleNamespace(
            replay_id=replay_id,
            perspective_player=player,
            start_timestep=start,
            end_timestep=end,
        )

    first = [window("r1", "p1", 0, 10), window("r1", "p1", 10, 20)]
    same = [window("r1", "p1", 0, 10), window("r1", "p1", 10, 20)]
    different_span = [window("r1", "p1", 0, 10), window("r1", "p1", 20, 30)]
    different_replay = [window("r1", "p1", 0, 10), window("r2", "p1", 10, 20)]
    different_player = [window("r1", "p1", 0, 10), window("r1", "p2", 10, 20)]
    reordered = list(reversed(first))

    key = leaderboard._window_selection_key
    assert key(first) == key(same)
    # Same count, every one of these is a different set of scored positions.
    for other in (different_span, different_replay, different_player, reordered):
        assert key(other) != key(first)


def test_window_selection_key_reads_back_what_the_loader_serves() -> None:
    """The key is derived from the loader, never re-derived from the budget.

    ``SharedResources.dataloader`` wraps the dataset in a ``Subset``; reading the
    windows back through it means the hash cannot disagree with what was scored.
    """

    windows = tuple(
        SimpleNamespace(
            replay_id=f"r{index}",
            perspective_player="p1",
            start_timestep=index * 10,
            end_timestep=index * 10 + 10,
        )
        for index in range(5)
    )
    loader = SimpleNamespace(
        dataset=SimpleNamespace(
            dataset=SimpleNamespace(windows=windows), indices=[3, 1]
        )
    )

    served = leaderboard._selected_windows(loader)
    assert [item.replay_id for item in served] == ["r3", "r1"]


def test_two_runs_of_the_same_checkpoint_share_a_dedupe_key(tmp_path: Path) -> None:
    """The redundancy handle is what makes repeat rows deletable in bulk."""

    first = _build_a_row(tmp_path)
    second = _build_a_row(tmp_path)
    assert first["dedupe_key"] == second["dedupe_key"]
    assert first["eval_condition_key"] == second["eval_condition_key"]


def test_shared_csv_resolves_to_the_output_root_not_a_run_directory() -> None:
    """The one place this package writes outside a per-run subdirectory.

    ``run`` builds the path as ``context.run_dir.parent / LEADERBOARD_FILENAME``.
    The runner allocates ``run_dir`` as ``output/<model label>__<date>``, so the
    parent is ``output/`` itself -- this pins that the filename cannot smuggle in
    a subdirectory and that the result is not inside any run directory.
    """

    assert "/" not in leaderboard.LEADERBOARD_FILENAME
    assert "\\" not in leaderboard.LEADERBOARD_FILENAME

    output_root = PACKAGE_DIR / "output"
    run_dir = output_root / "smallTrainingTestV3-epoch-0033__2026-Aug-27_09-14PM"
    resolved = run_dir.parent / leaderboard.LEADERBOARD_FILENAME
    assert resolved.parent == output_root
    assert run_dir.name not in resolved.parts


# ---------------------------------------------------------------------------
# append_leaderboard_row -- the accumulating file
# ---------------------------------------------------------------------------


def test_append_carries_the_effective_token_choices_column(tmp_path: Path) -> None:
    """A realistic row round-trips with the renamed column intact."""

    path = tmp_path / "leaderboard.csv"
    leaderboard.append_leaderboard_row(
        {
            "model_label": "modelA",
            "bits_per_token": 2.0,
            "effective_token_choices": 4.0,
        },
        path,
    )

    header, rows = _read_csv(path)
    assert "perplexity" not in header
    assert header[-1] == "effective_token_choices"
    assert rows[0]["effective_token_choices"] == "4.0"


def test_append_creates_the_file_with_a_header_and_one_row(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.csv"
    returned = leaderboard.append_leaderboard_row(
        {"model_label": "modelA", "bits_per_token": 3.5}, path
    )

    assert returned == path
    header, rows = _read_csv(path)
    assert header == ["model_label", "bits_per_token"]
    assert rows == [{"model_label": "modelA", "bits_per_token": "3.5"}]


def test_append_keeps_every_earlier_row(tmp_path: Path) -> None:
    """Re-running a checkpoint must add history, never replace it."""

    path = tmp_path / "leaderboard.csv"
    for index in range(3):
        leaderboard.append_leaderboard_row(
            {"model_label": f"model{index}", "bits_per_token": index}, path
        )

    _, rows = _read_csv(path)
    assert [row["model_label"] for row in rows] == ["model0", "model1", "model2"]


def test_append_widens_the_header_without_misaligning_old_rows(tmp_path: Path) -> None:
    """A later run with more columns is the case plain append mode gets wrong.

    The old row must keep its own values and receive a BLANK under the new
    column -- not have its values slide left under the wrong headers.
    """

    path = tmp_path / "leaderboard.csv"
    leaderboard.append_leaderboard_row(
        {"model_label": "old", "bits_per_token": 3.5}, path
    )
    leaderboard.append_leaderboard_row(
        {"model_label": "new", "bits_per_token": 2.5, "bits_per_token_t_0.25": 1.5},
        path,
    )

    header, rows = _read_csv(path)
    assert header == ["model_label", "bits_per_token", "bits_per_token_t_0.25"]
    assert rows[0] == {
        "model_label": "old",
        "bits_per_token": "3.5",
        "bits_per_token_t_0.25": "",
    }
    assert rows[1]["bits_per_token_t_0.25"] == "1.5"


def test_append_preserves_an_existing_header_order(tmp_path: Path) -> None:
    """Columns reordered by hand in a spreadsheet must survive the next run."""

    path = tmp_path / "leaderboard.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bits_per_token", "model_label"])
        writer.writeheader()
        writer.writerow({"bits_per_token": "9.0", "model_label": "hand_edited"})

    # The new row's own dict order is the opposite of the file's.
    leaderboard.append_leaderboard_row(
        {"model_label": "fresh", "bits_per_token": 1.0}, path
    )

    header, rows = _read_csv(path)
    assert header == ["bits_per_token", "model_label"]
    assert rows[0]["model_label"] == "hand_edited"
    assert rows[1] == {"bits_per_token": "1.0", "model_label": "fresh"}


def test_append_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "output" / "leaderboard.csv"
    leaderboard.append_leaderboard_row({"model_label": "modelA"}, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# eval_condition_key -- "are these two rows comparable at all"
# ---------------------------------------------------------------------------


def _condition(**overrides) -> dict[str, object]:
    """A complete evaluation condition, with per-test overrides applied."""

    base = {
        "config": "configs/smallTrainingTestV3.yaml",
        "split_source": "recorded",
        "n_replays_in_split": 23,
        "n_windows": 240,
        "window_selection_key": "abcdef0123",
        "fog_condition": "training distribution (per serving)",
        "seed": 1234,
        "batch_size": 8,
        "vocab_size": 512,
        "diffusion_process": "masked",
        "t_schedule": "cosine",
        "t_distribution": "power",
        "t_distribution_power": 2.0,
        "t_min": 0.0,
        "t_max": 1.0,
        "t_one_fraction": 0.05,
        "t_grid": "0.25/0.50/0.75/1.00",
    }
    base.update(overrides)
    return base


def test_condition_key_is_stable_and_order_independent() -> None:
    forward = _condition()
    reversed_order = dict(reversed(list(forward.items())))
    assert leaderboard._eval_condition_key(forward) == leaderboard._eval_condition_key(
        reversed_order
    )


@pytest.mark.parametrize("field", leaderboard.EVAL_CONDITION_FIELDS)
def test_condition_key_responds_to_every_field_it_claims_to_cover(field: str) -> None:
    """A field in the declared list that did not change the key would be a lie.

    The key is what tells the reader two rows may be ranked against each other,
    so a covered field that leaves it untouched would let two incomparable
    measurements share one key.
    """

    baseline = leaderboard._eval_condition_key(_condition())
    mutated = leaderboard._eval_condition_key(_condition(**{field: "CHANGED"}))
    assert mutated != baseline


def test_condition_key_rejects_an_incomplete_condition() -> None:
    """Failing closed beats hashing a silently-missing field into a shared key."""

    partial = _condition()
    del partial["seed"]
    with pytest.raises(KeyError):
        leaderboard._eval_condition_key(partial)


# ---------------------------------------------------------------------------
# dedupe_key -- "is this row redundant"
# ---------------------------------------------------------------------------


def _dedupe(**overrides) -> str:
    base = {
        "model_label": "smallTrainingTestV3-epoch-0033",
        "architecture_identity": "arch-abc123",
        "arch_shape": "d768-L12-H12-F3072",
        "global_step": 41000,
        "condition_key": "0123456789",
        "bits_per_token": 3.141593,
    }
    base.update(overrides)
    return leaderboard._dedupe_key(**base)


def test_dedupe_key_matches_for_an_identical_repeat_measurement() -> None:
    assert _dedupe() == _dedupe()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_label", "someOtherRun-epoch-0100"),
        ("architecture_identity", "arch-def456"),
        ("arch_shape", "d1024-L12-H16-F4096"),
        ("global_step", 42000),
        ("condition_key", "9876543210"),
        ("bits_per_token", 3.141594),
    ],
)
def test_dedupe_key_separates_rows_that_differ(field: str, value: object) -> None:
    assert _dedupe(**{field: value}) != _dedupe()


def test_dedupe_key_ignores_noise_below_the_reported_precision() -> None:
    """The CSV stores six decimals, so the key must not split on the seventh."""

    assert _dedupe(bits_per_token=3.1415930000001) == _dedupe(bits_per_token=3.141593)


# ---------------------------------------------------------------------------
# Row provenance and formatting helpers
# ---------------------------------------------------------------------------


def test_run_moment_parses_the_runner_directory_name() -> None:
    moment = leaderboard._run_moment(
        Path("output/smallTrainingTestV3-epoch-0033__2026-Aug-26_02-41PM")
    )
    assert moment == datetime(2026, 8, 26, 14, 41)


def test_run_moment_ignores_a_collision_suffix() -> None:
    """Two runs in the same minute get ``-2``; both still carry that minute."""

    moment = leaderboard._run_moment(
        Path("output/smallTrainingTestV3-epoch-0033__2026-Aug-26_02-41PM-2")
    )
    assert moment == datetime(2026, 8, 26, 14, 41)


def test_run_moment_falls_back_to_now_for_a_foreign_directory_name() -> None:
    before = datetime.now()
    moment = leaderboard._run_moment(Path("some_scratch_directory"))
    assert before <= moment <= datetime.now()


def test_architecture_shape_reads_the_four_dimensions() -> None:
    shape = leaderboard._architecture_shape(
        SimpleNamespace(d_model=768, layers=12, heads=12, ffn=3072)
    )
    assert shape == "d768-L12-H12-F3072"


def test_round_renders_a_missing_value_as_a_blank_cell() -> None:
    assert leaderboard._round(None) == ""
    assert leaderboard._round(3.14159265) == 3.141593
    assert leaderboard._round(3.14159265, 2) == 3.14


def test_flag_renders_booleans_and_unknowns() -> None:
    assert leaderboard._flag(True) == "true"
    assert leaderboard._flag(False) == "false"
    assert leaderboard._flag(None) == ""


def test_format_grid_is_stable() -> None:
    assert leaderboard._format_grid((0.25, 0.5, 1.0)) == "0.25/0.50/1.00"


# ---------------------------------------------------------------------------
# The t-grid option
# ---------------------------------------------------------------------------


def test_t_grid_defaults_when_the_option_is_absent() -> None:
    context = SimpleNamespace(extra={})
    assert leaderboard._parse_t_grid(context) == leaderboard.DEFAULT_T_GRID


def test_t_grid_accepts_an_override() -> None:
    context = SimpleNamespace(extra={"leaderboard_t_grid": "0.1,0.9"})
    assert leaderboard._parse_t_grid(context) == (0.1, 0.9)


def test_t_grid_rejects_values_outside_the_unit_interval() -> None:
    context = SimpleNamespace(extra={"leaderboard_t_grid": "0.5,1.5"})
    with pytest.raises(ValueError):
        leaderboard._parse_t_grid(context)


def test_t_grid_override_changes_the_condition_key() -> None:
    """An overridden grid must not be silently comparable to the default one."""

    default_key = leaderboard._eval_condition_key(
        _condition(t_grid=leaderboard._format_grid(leaderboard.DEFAULT_T_GRID))
    )
    override_key = leaderboard._eval_condition_key(
        _condition(t_grid=leaderboard._format_grid((0.1, 0.9)))
    )
    assert override_key != default_key
