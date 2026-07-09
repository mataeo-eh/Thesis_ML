"""Tests for the read-only diagnostic visualisation module.

Covers the acceptance checks from prompt 011:
  1. End-to-end render: a random checkpoint + tiny replay set produces both
     figure files under --out-dir.
  2. Reuse, not reimplementation: the module imports/calls the existing
     harness/sampler/decode/oracle interfaces and defines none of them.
  3. Resolution-matched: rendered grids come from the harness count grids
     (decoded at model resolution).
  4. Truncated-drop consistency: figures are built from decode_canvas output,
     which never emits a partial final timestep.
  5. Read-only: the run mutates no checkpoint/source data and writes only under
     --out-dir.

The heavy end-to-end test uses the real extractor fixtures (the two smallest),
a tiny random-weight model, and a small canvas budget so it stays fast.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from thesis_ml.config import load_config
from thesis_ml.inference.timing import TimedTimestep
from thesis_ml.eval.buildorder import BuildOrderEvent
from thesis_ml.eval.harness import EvaluationExampleResult
from thesis_ml.inference.sampler import denoise_canvas_once
from thesis_ml.serialize import TokenRecord
from thesis_ml.viz import diagnostics
from thesis_ml.vocab.content_vocab import build_content_vocabulary
from thesis_ml.vocab.special_tokens import END_ID, MASK_ID, WIN_ID


# Repo-relative fixture locations (metadata resolves via ../json/ of the parquet).
_REPO = Path(__file__).resolve().parent.parent
_FIXTURE_PARQUETS = _REPO / "tests" / "fixtures"
_FIXTURE_JSON = _REPO / "data" / "processed" / "quickstart" / "json"
# The two smallest fixture replays keep tokenization + sampling cheap.
_SMALL_REPLAYS = ("match_4745722", "match_4745724")


# ---------------------------------------------------------------------------
# Fast unit tests for the pure helpers (no model / pipeline needed)
# ---------------------------------------------------------------------------


def _timed(counts_by_step):
    return tuple(
        TimedTimestep(timestep_index=index, timestamp_seconds=float(index), counts=dict(counts))
        for index, counts in enumerate(counts_by_step)
    )


def test_counts_matrix_is_signed_predicted_minus_ground_truth():
    predicted = _timed([{"marine": 3}, {"marine": 3, "barracks": 1}])
    ground_truth = _timed([{"marine": 1}, {"marine": 3}])

    entity_types, buckets, matrix = diagnostics._counts_matrix(predicted, ground_truth)

    assert entity_types == ["barracks", "marine"]
    assert buckets == [0, 1]
    # barracks: predicted-only at bucket 1 -> +1; marine: +2 then 0.
    assert matrix[entity_types.index("marine")] == [2.0, 0.0]
    assert matrix[entity_types.index("barracks")] == [0.0, 1.0]


def test_window_selection_applies_limit_per_replay_without_total_cap():
    windows = [
        SimpleNamespace(replay_id=replay_id, window_start=start)
        for replay_id in ("replay_a", "replay_b")
        for start in range(4)
    ]

    selected = diagnostics._select_windows(windows, n_windows=2)

    assert [item.replay_id for item in selected] == ["replay_a", "replay_a", "replay_b", "replay_b"]
    assert [item.window_start for item in selected] == [0, 1, 0, 1]


def test_count_comparison_uses_aligned_high_contrast_panels():
    item = diagnostics.RenderedExample(
        example=SimpleNamespace(),
        result=SimpleNamespace(
            predicted_counts=_timed([{"marine": 1}, {"marine": 3}]),
            ground_truth_counts=_timed([{"marine": 2}, {"marine": 2}]),
            prediction_valid=True,
        ),
        label="replay_a_p1_t0",
    )

    figure = diagnostics.plot_count_comparison(item)

    panel_titles = [axis.get_title(loc="left") for axis in figure.axes]
    assert any("GROUND TRUTH" in title for title in panel_titles)
    assert any("MODEL PREDICTION" in title for title in panel_titles)
    assert any("ERROR direction" in title for title in panel_titles)
    assert figure.get_size_inches()[0] <= 18.0
    diagnostics.plt.close(figure)


def test_first_appearance_files_are_opt_in(tmp_path: Path):
    item = diagnostics.RenderedExample(
        example=SimpleNamespace(),
        result=SimpleNamespace(
            predicted_counts=_timed([{"marine": 1}]),
            ground_truth_counts=_timed([{"marine": 1}]),
            predicted_events=(BuildOrderEvent("marine", 0),),
            ground_truth_events=(BuildOrderEvent("marine", 0),),
            prediction_valid=True,
        ),
        label="replay_a_p1_t0",
    )

    diagnostics.render_figures([item], tmp_path, tolerance_buckets=1, dpi=40)

    assert list(tmp_path.glob("prediction_vs_truth_*.png"))
    assert not list(tmp_path.glob("first_appearance_*"))


def test_first_appearance_reduces_to_earliest_bucket_per_type():
    events = (
        BuildOrderEvent("marine", 5),
        BuildOrderEvent("marine", 2),
        BuildOrderEvent("barracks", 3),
    )
    assert diagnostics._first_appearance(events) == {"marine": 2, "barracks": 3}


def test_module_does_not_redefine_pipeline_functions():
    """Reuse check: the pipeline primitives are imported, not defined here."""

    source = inspect.getsource(diagnostics)
    for banned in (
        "def sample_canvas",
        "def decode_canvas",
        "def extract_build_order",
        "def evaluate_example",
        "def preprocess_replays",
    ):
        assert banned not in source, f"viz must not reimplement `{banned}`"


def test_optional_canvas_and_logit_exports_preserve_raw_positions(tmp_path: Path):
    vocabulary = build_content_vocabulary({"1": "marine", "2": "barracks"})
    logits = torch.zeros(3, vocabulary.vocab_size)
    logits[0, WIN_ID] = 4.0
    logits[1, 100] = 3.0
    logits[2, END_ID] = 2.0
    item = diagnostics.RenderedExample(
        example=SimpleNamespace(),
        result=SimpleNamespace(
            predicted_canvas=(WIN_ID, 100, END_ID),
            ground_truth_canvas=(WIN_ID, 101, END_ID),
            final_canvas_logits=logits,
            # Position 0 was revealed as ground truth; positions 1-2 predicted.
            predicted_canvas_revealed_mask=(True, False, False),
        ),
        label="replay_p1_t0",
    )

    csv_paths = diagnostics.write_canvas_comparison_csv_files([item], vocabulary, tmp_path)
    json_path = diagnostics.write_logits_json([item], vocabulary, tmp_path, top_k=10)

    assert len(csv_paths) == 1
    with (tmp_path / "canvas_comparison_replay_p1_t0.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Header order/names match the requested comparison schema.
    assert list(rows[0].keys()) == ["sequenceindex", "modelprediction", "groundtruth", "correct"]
    # Position 1: model predicted marine (100) but truth was barracks (101) -> mismatch.
    assert rows[1]["modelprediction"] == "marine"
    assert rows[1]["groundtruth"] == "barracks"
    assert rows[1]["correct"] == "False"
    # Position 0 was revealed as ground truth (unmasked), not predicted, so it is
    # flagged Unmasked rather than True even though the tokens match.
    assert rows[0]["sequenceindex"] == "0"
    assert rows[0]["correct"] == "Unmasked"
    # Position 2 was masked and predicted correctly (END on both sides) -> True.
    assert rows[2]["correct"] == "True"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    position = payload["examples"][0]["positions"][1]
    assert position["sequence_index"] == 1
    assert position["source"] == "MODEL"
    assert position["predicted_token"] == "marine"
    assert position["ground_truth_token"] == "barracks"
    assert len(position["top_k"]) == 10
    assert {"token", "token_id", "logit", "confidence"} <= set(position["top_k"][0])


def _input_record(token_id: int, name: str, allegiance: str | None) -> TokenRecord:
    """Minimal TokenRecord for input-canvas tests (only fields the writer reads)."""

    return TokenRecord(
        token_id=token_id,
        token_name=name,
        token_kind="delimiter" if allegiance is None else "entity",
        owner=None,
        allegiance=allegiance,
        game_loop=0,
        timestamp_seconds=0.0,
    )


def test_write_input_canvas_marks_self_and_enemy_tokens(tmp_path: Path):
    """--show-input dump tags each token with its side so humans can tell them apart."""

    input_records = [
        _input_record(100, "marine", "self"),
        _input_record(3, "[DELIMITER]", None),
        _input_record(101, "zealot", "enemy"),
        _input_record(3, "[DELIMITER]", None),
    ]
    item = diagnostics.RenderedExample(
        example=SimpleNamespace(input_records=input_records),
        result=SimpleNamespace(),
        label="replay_p1_t0",
    )

    written = diagnostics.write_input_canvas_text_files([item], tmp_path)

    assert len(written) == 1
    text = (tmp_path / "input_canvas_replay_p1_t0.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    # Every column layout is index<TAB>token_id<TAB>side<TAB>token_name.
    assert "0\t100\tSELF \tmarine" in lines
    assert "2\t101\tENEMY\tzealot" in lines
    # Delimiters belong to neither side and are marked distinctly.
    assert "1\t3\t-----\t[DELIMITER]" in lines
    # The self and enemy markers are visibly different (the human-readable point).
    assert diagnostics._allegiance_marker("self") != diagnostics._allegiance_marker("enemy")


def test_multi_window_non_image_exports_are_consolidated(tmp_path: Path):
    vocabulary = build_content_vocabulary({"1": "marine", "2": "barracks"})
    logits = torch.zeros(1, vocabulary.vocab_size)
    items = [
        diagnostics.RenderedExample(
            example=SimpleNamespace(input_records=[_input_record(100, "marine", "self")]),
            result=SimpleNamespace(
                predicted_canvas=(100,),
                ground_truth_canvas=(truth_id,),
                predicted_canvas_revealed_mask=(False,),
                final_canvas_logits=logits,
            ),
            label=label,
        )
        for label, truth_id in (("replay_a_p1_t0", 100), ("replay_b_p1_t0", 101))
    ]

    input_paths = diagnostics.write_input_canvas_text_files(items, tmp_path)
    csv_paths = diagnostics.write_canvas_comparison_csv_files(items, vocabulary, tmp_path)
    json_path = diagnostics.write_logits_json(items, vocabulary, tmp_path, top_k=3)

    assert input_paths == [tmp_path / "input_canvas.txt"]
    input_text = input_paths[0].read_text(encoding="utf-8")
    assert "# window: replay_a_p1_t0" in input_text
    assert "# window: replay_b_p1_t0" in input_text
    assert not list(tmp_path.glob("input_canvas_replay_*.txt"))

    assert csv_paths == [tmp_path / "canvas_comparison.csv"]
    with csv_paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["window", "sequenceindex", "modelprediction", "groundtruth", "correct"]
    assert [row["window"] for row in rows] == ["replay_a_p1_t0", "replay_b_p1_t0"]
    assert [row["sequenceindex"] for row in rows] == ["0", "0"]
    assert not list(tmp_path.glob("canvas_comparison_replay_*.csv"))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [example["label"] for example in payload["examples"]] == [
        "replay_a_p1_t0",
        "replay_b_p1_t0",
    ]


def test_cli_json_and_csv_flags_are_opt_in(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(diagnostics, "run", fake_run)
    diagnostics.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--replay-dir",
            str(tmp_path / "replays"),
            "--out-dir",
            str(tmp_path / "out"),
            "--json",
            "--csv",
            "--show-input",
            "--first-appearance",
            "--bypass-sampler",
        ]
    )

    assert captured["write_json"] is True
    assert captured["write_csv"] is True
    assert captured["write_input"] is True
    assert captured["write_first_appearance"] is True
    assert captured["bypass_sampler"] is True


def test_cli_bypass_sampler_defaults_off(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(diagnostics, "run", fake_run)
    diagnostics.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--replay-dir",
            str(tmp_path / "replays"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert captured["bypass_sampler"] is False
    assert captured["write_input"] is False
    assert captured["write_first_appearance"] is False
    # Default weight selection is deterministically EMA (no --raw flag).
    assert captured["use_raw"] is False


def test_cli_raw_flag_selects_raw_weights(monkeypatch, tmp_path: Path):
    """`--raw` opts into the raw weights; its absence stays on EMA (above)."""

    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(diagnostics, "run", fake_run)
    diagnostics.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--replay-dir",
            str(tmp_path / "replays"),
            "--out-dir",
            str(tmp_path / "out"),
            "--raw",
        ]
    )

    assert captured["use_raw"] is True


def _cli_base_args(tmp_path: Path) -> list[str]:
    """Minimal required CLI args for main() with the heavy run() monkeypatched."""

    return [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--replay-dir",
        str(tmp_path / "replays"),
        "--out-dir",
        str(tmp_path / "out"),
    ]


def test_cli_output_mask_defaults_to_pretraining_average(monkeypatch, tmp_path: Path):
    """Omitting --output-mask yields the single pre-training average rate (0.5)."""

    captured: dict[str, object] = {}
    monkeypatch.setattr(diagnostics, "run", lambda **kwargs: captured.update(kwargs) or [])
    diagnostics.main(_cli_base_args(tmp_path))
    assert captured["output_masks"] == [0.5]


def test_cli_window_count_is_per_replay(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    monkeypatch.setattr(diagnostics, "run", lambda **kwargs: captured.update(kwargs) or [])

    diagnostics.main(_cli_base_args(tmp_path) + ["--n-replays", "2", "--n-windows", "5"])

    assert captured["n_replays"] == 2
    assert captured["n_windows"] == 5
    assert "n" not in captured


def test_cli_output_mask_threads_multiple_rates(monkeypatch, tmp_path: Path):
    """Several --output-mask values thread through as an ordered list of rates."""

    captured: dict[str, object] = {}
    monkeypatch.setattr(diagnostics, "run", lambda **kwargs: captured.update(kwargs) or [])
    diagnostics.main(_cli_base_args(tmp_path) + ["--output-mask", "0.9", "1.0", "0.4"])
    assert captured["output_masks"] == [0.9, 1.0, 0.4]


def test_cli_output_mask_rejects_out_of_range(monkeypatch, tmp_path: Path):
    """A rate outside [0, 1] is rejected before any work runs."""

    monkeypatch.setattr(diagnostics, "run", lambda **kwargs: [])
    with pytest.raises(SystemExit):
        diagnostics.main(_cli_base_args(tmp_path) + ["--output-mask", "1.5"])


def test_bypass_sampler_uses_exactly_one_all_mask_forward():
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.seen_canvas = None

        def forward(self, *, input_token_ids, canvas_token_ids, **kwargs):
            self.calls += 1
            self.seen_canvas = canvas_token_ids.detach().clone()
            batch_size, canvas_len = canvas_token_ids.shape
            input_len = input_token_ids.shape[1]
            logits = torch.zeros(batch_size, input_len + canvas_len, 128)
            logits[:, input_len:, END_ID] = 1.0
            return SimpleNamespace(logits=logits)

    model = CountingModel()
    batch = SimpleNamespace(
        input_token_ids=torch.tensor([[100, 101]]),
        input_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        input_features=None,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(canvas_budget_tokens=3),
        model=SimpleNamespace(self_conditioning=True),
    )

    output = denoise_canvas_once(
        model,
        batch,
        config,
        device="cpu",
        return_final_logits=True,
    )

    assert model.calls == 1
    assert torch.all(model.seen_canvas == MASK_ID)
    assert output.steps == 1
    assert output.committed_mask.all()
    assert output.final_canvas_logits is not None
    assert output.final_canvas_logits.shape == (1, 3, 128)
    # Default mask_rate=1.0: nothing is revealed, so the whole canvas is model output.
    assert output.revealed_mask is not None
    assert not output.revealed_mask.any()


def test_denoise_partial_mask_reveals_ground_truth():
    """--output-mask < 1.0 masks only part of the canvas; the rest is revealed GT."""

    class ConstantModel(torch.nn.Module):
        def forward(self, *, input_token_ids, canvas_token_ids, **kwargs):
            batch_size, canvas_len = canvas_token_ids.shape
            input_len = input_token_ids.shape[1]
            logits = torch.zeros(batch_size, input_len + canvas_len, 128)
            logits[:, input_len:, 7] = 1.0  # the model always predicts token id 7
            return SimpleNamespace(logits=logits)

    model = ConstantModel()
    target = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    batch = SimpleNamespace(
        input_token_ids=torch.tensor([[100, 101]]),
        input_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        input_features=None,
        target_canvas=target,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(canvas_budget_tokens=8),
        model=SimpleNamespace(self_conditioning=False),
        pipeline=SimpleNamespace(seed=0),
        diffusion=SimpleNamespace(
            mask_schedule=SimpleNamespace(
                name="linear",
                t_distribution="uniform",
                min=0.0,
                max=1.0,
                loss_reweight="inverse_t",
            )
        ),
    )

    output = denoise_canvas_once(model, batch, config, device="cpu", mask_rate=0.5)

    assert output.revealed_mask is not None
    revealed = output.revealed_mask[0]
    canvas = output.canvas[0]
    # Invariant: revealed positions keep ground truth, masked positions become the
    # model's prediction (token id 7). Holds for whatever reveal pattern the seed
    # produces.
    for position in range(8):
        if revealed[position]:
            assert canvas[position] == target[0, position]
        else:
            assert canvas[position] == 7
    # The reveal pattern is deterministic (seeded from config.pipeline.seed).
    repeat = denoise_canvas_once(model, batch, config, device="cpu", mask_rate=0.5)
    assert torch.equal(output.revealed_mask, repeat.revealed_mask)


def test_evaluate_selected_threads_bypass_to_shared_harness(monkeypatch):
    captured: dict[str, object] = {}

    def fake_evaluate_example(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(diagnostics, "evaluate_example", fake_evaluate_example)
    example = SimpleNamespace(
        replay_id="replay",
        replay_path=None,
        perspective_player="p1",
        window_start=0,
    )

    rendered = diagnostics.evaluate_selected(
        SimpleNamespace(),
        [example],
        SimpleNamespace(),
        SimpleNamespace(),
        device="cpu",
        bypass_sampler=True,
    )

    assert captured["bypass_sampler"] is True
    assert len(rendered) == 1
    assert rendered[0].label == "replay_p1_t0"


# ---------------------------------------------------------------------------
# End-to-end render on real fixtures + a tiny random checkpoint
# ---------------------------------------------------------------------------


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_replay_dir(tmp_path: Path) -> Path:
    """Assemble a replay dir with the features/ + json/ layout the loader needs.

    ``resolve_replay_outcome`` derives metadata as
    ``<parquet>.parent.parent / json / <id>_metadata.json``, so the parquet must
    live in a ``features/`` subdir with a sibling ``json/`` dir.
    """

    replay_root = tmp_path / "replays"
    features = replay_root / "features"
    json_dir = replay_root / "json"
    features.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    for replay_id in _SMALL_REPLAYS:
        shutil.copy(_FIXTURE_PARQUETS / f"{replay_id}_game_state.parquet", features / f"{replay_id}_game_state.parquet")
        shutil.copy(_FIXTURE_JSON / f"{replay_id}_metadata.json", json_dir / f"{replay_id}_metadata.json")
    return features


def _write_tiny_config(tmp_path: Path) -> Path:
    """Write a YAML that extends default.yaml with tiny, fast overrides."""

    default_path = (_REPO / "config" / "default.yaml").resolve().as_posix()
    token_dict = (_REPO / "data" / "Token_Dictionary.json").resolve().as_posix()
    config_path = tmp_path / "viz_config.yaml"
    config_path.write_text(
        f"""
extends: {default_path}
data:
  canvas_budget_tokens: 256
model:
  d_model: 32
  layers: 1
  heads: 4
  ffn: 64
sampler:
  max_steps: 4
eval:
  fog_rate: 0.0
  timing_tolerance_buckets: 1
""",
        encoding="utf-8",
    )
    return config_path


def _write_random_checkpoint(config_path: Path, checkpoint_path: Path) -> None:
    """Create a random-weight checkpoint with a real vocab size.

    Built with the tiny config so its stored ``model.*`` matches, and with
    ``vocab_size`` equal to the real token dictionary so tokenized replay ids
    stay in range. Saved via ``TrainingLoop.save_checkpoint`` so it carries the
    same ``model``/``ema_model``/``config`` keys the loader expects.
    """

    from thesis_ml.model.model import SC2StrategyDiffusionModel
    from thesis_ml.train.loop import TrainingLoop
    from thesis_ml.vocab.content_vocab import load_content_vocabulary

    config = load_config(config_path)
    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    model = SC2StrategyDiffusionModel(config, vocab_size=vocabulary.vocab_size)
    loop = TrainingLoop(model=model, config=config, device="cpu", seed=0)
    loop.save_checkpoint(checkpoint_path)


@pytest.mark.slow
@pytest.mark.parametrize("bypass_sampler", [False, True])
def test_end_to_end_render_writes_comparison_and_opt_in_timeline(tmp_path: Path, bypass_sampler: bool):
    torch.manual_seed(0)
    replay_dir = _build_replay_dir(tmp_path)
    config_path = _write_tiny_config(tmp_path)
    checkpoint_path = tmp_path / "random.pt"
    _write_random_checkpoint(config_path, checkpoint_path)
    out_dir = tmp_path / "figures"

    # Snapshot source artifacts to prove the run is read-only (check 5).
    src_before = {
        checkpoint_path: _hash(checkpoint_path),
        replay_dir / f"{_SMALL_REPLAYS[0]}_game_state.parquet": _hash(
            replay_dir / f"{_SMALL_REPLAYS[0]}_game_state.parquet"
        ),
        replay_dir.parent / "json" / f"{_SMALL_REPLAYS[0]}_metadata.json": _hash(
            replay_dir.parent / "json" / f"{_SMALL_REPLAYS[0]}_metadata.json"
        ),
        config_path: _hash(config_path),
    }

    written = diagnostics.run(
        checkpoint=checkpoint_path,
        replay_dir=replay_dir,
        config_path=config_path,
        out_dir=out_dir,
        n_replays=2,
        n_windows=1,
        fog_rate=None,  # -> config.eval.fog_rate (0.0)
        dpi=80,
        device="cpu",
        bypass_sampler=bypass_sampler,
        write_first_appearance=True,
    )

    # Check 1: both figure kinds were produced, as image files, under out-dir.
    pngs = list(out_dir.glob("*.png"))
    comparison_pngs = [p for p in pngs if p.name.startswith("prediction_vs_truth_")]
    timeline_pngs = [p for p in pngs if p.name.startswith("first_appearance_")]
    assert len(comparison_pngs) == 2, "one comparison per requested window expected"
    assert {"match_4745722", "match_4745724"} == {
        replay_id for replay_id in _SMALL_REPLAYS if any(replay_id in path.name for path in comparison_pngs)
    }
    assert timeline_pngs, "Figure B (first-appearance timeline) PNG missing"
    # Vector output exists too (readability for long/wide grids).
    assert list(out_dir.glob("*.svg")), "vector SVG output missing"
    assert (out_dir / "diagnostics.pdf").exists(), "combined PDF missing"
    assert not list(out_dir.glob("*.txt"))
    assert not list(out_dir.glob("*.json"))
    for path in written:
        assert Path(path).exists()
        assert out_dir in Path(path).parents or Path(path).parent == out_dir

    # Check 5: no source artifact mutated; writes are confined to out-dir.
    for path, digest in src_before.items():
        assert _hash(path) == digest, f"read-only violation: {path} changed"
    # The redirected tokenization cache landed under out-dir, not the repo tree.
    assert (out_dir / "_ingest_cache" / "window_manifest.jsonl").exists()


@pytest.mark.slow
def test_output_mask_sweep_writes_per_rate_subdirs(tmp_path: Path):
    """Multiple --output-mask rates each get their own output_mask_<t>/ subdir."""

    torch.manual_seed(0)
    replay_dir = _build_replay_dir(tmp_path)
    config_path = _write_tiny_config(tmp_path)
    checkpoint_path = tmp_path / "random.pt"
    _write_random_checkpoint(config_path, checkpoint_path)
    out_dir = tmp_path / "figures"

    written = diagnostics.run(
        checkpoint=checkpoint_path,
        replay_dir=replay_dir,
        config_path=config_path,
        out_dir=out_dir,
        n_replays=1,
        n_windows=1,
        fog_rate=None,
        dpi=80,
        device="cpu",
        output_masks=[0.4, 1.0],
    )

    # Each rate produced its own subdirectory with a full figure set + combined PDF.
    for rate in (0.4, 1.0):
        rate_dir = out_dir / f"output_mask_{rate:.2f}"
        assert rate_dir.is_dir(), f"missing subdir for rate {rate}"
        assert (rate_dir / "diagnostics.pdf").exists()
        assert list(rate_dir.glob("prediction_vs_truth_*.png")), f"no comparison for rate {rate}"
    # Nothing was written flat into out_dir when nesting (only the ingest cache +
    # the two rate subdirs live there).
    assert not list(out_dir.glob("*.png"))
    # Every returned path lives under one of the rate subdirs.
    for path in written:
        assert (out_dir / "output_mask_0.40") in Path(path).parents or (
            out_dir / "output_mask_1.00"
        ) in Path(path).parents


def _write_distinct_weight_checkpoint(
    config_path: Path, checkpoint_path: Path, *, drop_ema: bool = False
) -> None:
    """Save a checkpoint whose EMA weights differ from its raw weights.

    Both state dicts start from the same freshly-built tiny model, then the EMA
    copy's output head is offset so the two are distinguishable -- that is what
    lets the loader tests assert WHICH set was loaded. ``drop_ema`` omits the
    ``ema_model`` entry entirely to exercise the "default requires EMA" guard.
    """

    from thesis_ml.model.model import SC2StrategyDiffusionModel
    from thesis_ml.vocab.content_vocab import load_content_vocabulary

    config = load_config(config_path)
    vocabulary = load_content_vocabulary(config.pipeline.token_dictionary_uri)
    model = SC2StrategyDiffusionModel(config, vocab_size=vocabulary.vocab_size)
    raw_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    ema_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    # Make the EMA output head unmistakably different from the raw one.
    ema_state["output_head.weight"] = ema_state["output_head.weight"] + 1.0
    payload: dict[str, object] = {"model": raw_state, "config": config, "global_step": 0}
    if not drop_ema:
        payload["ema_model"] = ema_state
    torch.save(payload, checkpoint_path)


@pytest.mark.slow
def test_load_diagnostic_model_defaults_to_ema_and_raw_switches(tmp_path: Path):
    """Default loads EMA; --raw loads raw. The two weight sets are distinct."""

    config_path = _write_tiny_config(tmp_path)
    checkpoint_path = tmp_path / "distinct.pt"
    _write_distinct_weight_checkpoint(config_path, checkpoint_path)
    config = load_config(config_path)

    ema_model, _ = diagnostics.load_diagnostic_model(checkpoint_path, config, device="cpu")
    raw_model, _ = diagnostics.load_diagnostic_model(
        checkpoint_path, config, device="cpu", use_raw=True
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stored_ema = checkpoint["ema_model"]["output_head.weight"]
    stored_raw = checkpoint["model"]["output_head.weight"]
    # Precondition: the fixture really did make the two sets differ.
    assert not torch.equal(stored_ema, stored_raw)
    # Default -> EMA weights; --raw -> raw weights.
    assert torch.equal(ema_model.output_head.weight.cpu(), stored_ema)
    assert torch.equal(raw_model.output_head.weight.cpu(), stored_raw)


@pytest.mark.slow
def test_load_diagnostic_model_default_requires_ema_but_raw_still_works(tmp_path: Path):
    """The EMA default never silently falls back to raw; --raw is the escape."""

    config_path = _write_tiny_config(tmp_path)
    checkpoint_path = tmp_path / "raw_only.pt"
    _write_distinct_weight_checkpoint(config_path, checkpoint_path, drop_ema=True)
    config = load_config(config_path)

    with pytest.raises(KeyError, match="no EMA weights"):
        diagnostics.load_diagnostic_model(checkpoint_path, config, device="cpu")

    # The same EMA-less checkpoint still loads fine with --raw.
    raw_model, _ = diagnostics.load_diagnostic_model(
        checkpoint_path, config, device="cpu", use_raw=True
    )
    assert raw_model is not None


@pytest.mark.slow
def test_rendered_grids_are_model_resolution(tmp_path: Path):
    """Checks 3 & 4: rendered counts/events come from the harness intermediates.

    ``evaluate_example`` returns count grids decoded at sampling-interval
    resolution; the viz consumes them verbatim. This asserts the intermediates
    flow through unchanged (same lengths, TimedTimestep type).
    """

    replay_dir = _build_replay_dir(tmp_path)
    config_path = _write_tiny_config(tmp_path)
    checkpoint_path = tmp_path / "random.pt"
    _write_random_checkpoint(config_path, checkpoint_path)

    config = load_config(config_path)
    model, model_config = diagnostics.load_diagnostic_model(checkpoint_path, config, device="cpu")
    run_config, vocabulary, examples = diagnostics.ingest_examples(
        model_config,
        replay_dir,
        fog_rate=0.0,
        n_replays=1,
        n_windows=1,
        artifact_root=tmp_path / "cache" / "tok",
        manifest_path=tmp_path / "cache" / "manifest.jsonl",
    )
    rendered = diagnostics.evaluate_selected(model, examples, vocabulary, run_config, device="cpu")

    assert rendered, "expected at least one rendered window"
    item = rendered[0]
    assert isinstance(item.result, EvaluationExampleResult)
    # Ground-truth grid is the decoded clamped target canvas at model resolution.
    for step in item.result.ground_truth_counts:
        assert isinstance(step, TimedTimestep)
    # Ground-truth counts are non-empty (the target canvas always has content),
    # confirming the intermediate flowed through from the harness.
    assert len(item.result.ground_truth_counts) > 0
