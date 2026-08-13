from dataclasses import replace
from pathlib import Path
import sys

import pytest
import yaml

from thesis_ml.pipeline.acquire_data import CredentialError, run_acquisition
from thesis_ml.pipeline.storage import StorageResolver, parse_s3_uri
from thesis_ml.config import load_config
from thesis_ml.pipeline.train_pipeline import (
    _explicit_replay_selection,
    _perspectives,
    _select_replays,
    run_training_pipeline,
)


def test_training_and_finetune_perspectives_require_both_players() -> None:
    assert _perspectives("p1,p2") == ("p1", "p2")
    assert _perspectives("p2, p1") == ("p1", "p2")
    for invalid in ("", "p1", "p2", "p1,p1", "p1,p2,p1", "observer,p1"):
        with pytest.raises(ValueError, match="exactly p1,p2"):
            _perspectives(invalid)


def test_master_pipeline_smoke_run_writes_checkpoint_and_resumes(tmp_path: Path) -> None:
    config_path = _pipeline_config(tmp_path)

    first = run_training_pipeline(config_path)
    second = run_training_pipeline(config_path)

    checkpoint = tmp_path / "checkpoints" / "resume" / "last.pt"
    assert checkpoint.exists()
    assert first.resumed is False
    assert second.resumed is True
    assert second.steps == first.steps


def test_seeded_subset_selection_is_exact_reproducible_and_disjoint() -> None:
    """The seeded subset path (profiles that do NOT name replays explicitly).

    Uses local_full.yaml, which still draws its replays from the seeded split.
    Two calls with the same config must return the identical selection, of the
    configured sizes, with no replay in both groups.
    """

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "local_full.yaml")
    config = replace(
        config,
        pipeline=replace(
            config.pipeline, replay_subset_size=25, validation_replay_count=3
        ),
    )
    train_candidates = [f"train_{index}.parquet" for index in range(100)]
    dev_candidates = [f"dev_{index}.parquet" for index in range(20)]

    first = _select_replays(train_candidates, dev_candidates, config)
    second = _select_replays(train_candidates, dev_candidates, config)

    assert first == second
    assert len(first[0]) == 25
    assert len(first[1]) == 3
    assert set(first[0]).isdisjoint(first[1])


def test_overfit_profile_selects_its_named_replays_and_tests_on_the_rest() -> None:
    """The explicit path: named replays win, everything else becomes test.

    The overfit profile must train on EXACTLY the replays it names, in the order
    it names them, regardless of seeds or corpus size -- that reproducibility is
    the whole reason the named list exists. Every unnamed replay falls through
    to test, so each replay still lands in exactly one group.
    """

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "local_overfit.yaml")
    train_ids = config.pipeline.train_replay_ids.split(",")
    dev_ids = config.pipeline.dev_replay_ids.split(",")
    # A corpus containing every named replay plus unrelated ones.
    corpus = [f"/corpus/{stem}.parquet" for stem in (*train_ids, *dev_ids)]
    corpus += [f"/corpus/other_{index}.parquet" for index in range(5)]

    train, dev, test = _explicit_replay_selection(corpus, config)

    assert [Path(path).stem for path in train] == train_ids
    assert [Path(path).stem for path in dev] == dev_ids
    assert len(test) == 5
    assert set(train).isdisjoint(dev)
    assert set(train + dev).isdisjoint(test)
    assert len(train) + len(dev) + len(test) == len(corpus)


def test_explicit_selection_is_off_by_default_and_fails_loudly_when_misconfigured() -> None:
    """No named ids -> seeded split; bad named ids -> an error, never a silent shrink.

    Training on fewer replays than intended because a name was misspelled would
    quietly change what the run measures, so every failure mode raises.
    """

    root = Path(__file__).resolve().parents[1]
    default_config = load_config(root / "config" / "default.yaml")
    corpus = ["/corpus/a.parquet", "/corpus/b.parquet"]

    # Default profiles opt out entirely: None tells the caller to use the split.
    assert _explicit_replay_selection(corpus, default_config) is None

    def with_ids(train_ids: str, dev_ids: str = ""):
        return replace(
            default_config,
            pipeline=replace(
                default_config.pipeline,
                train_replay_ids=train_ids,
                dev_replay_ids=dev_ids,
            ),
        )

    # A name that is not in the corpus.
    with pytest.raises(ValueError, match="not found in the corpus"):
        _explicit_replay_selection(corpus, with_ids("a,missing"))
    # The same replay in both groups would leak train data into dev.
    with pytest.raises(ValueError, match="both train and dev"):
        _explicit_replay_selection(corpus, with_ids("a,b", "b"))
    # A repeated name silently shrinks the intended selection.
    with pytest.raises(ValueError, match="duplicate replay ids"):
        _explicit_replay_selection(corpus, with_ids("a,a"))
    # Naming dev replays without train replays is a half-configured selection.
    with pytest.raises(ValueError, match="train_replay_ids is empty"):
        _explicit_replay_selection(corpus, with_ids("", "a"))

    # Whitespace and trailing commas are tolerated so the YAML can be readable.
    train, dev, test = _explicit_replay_selection(corpus, with_ids(" a , ", "b"))
    assert [Path(path).stem for path in train] == ["a"]
    assert [Path(path).stem for path in dev] == ["b"]
    assert test == []


def test_real_manifest_pipeline_uses_workers_and_resumes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["storage"].update(
        {
            "data_uri": str(root / "tests" / "fixtures"),
            "checkpoint_uri": str(tmp_path / "real-checkpoints"),
            "log_uri": str(tmp_path / "real-logs"),
            "local_cache_dir": str(tmp_path / "real-cache"),
        }
    )
    raw["data"].update(
        {
            "input_budget_tokens": 512,
            "canvas_budget_tokens": 256,
                "tokenized_replay_dir": str(tmp_path / "tokenized"),
                "window_manifest_path": str(tmp_path / "manifest.jsonl"),
                "feature_statistics_path": str(tmp_path / "feature_statistics.json"),
        }
    )
    raw["pipeline"].update(
        {
            "smoke": False,
            "batch_size": 2,
            "replay_glob": "match_4745722_game_state.parquet",
            "token_dictionary_uri": str(root / "data" / "Token_Dictionary.json"),
            "num_workers": 2,
            "prefetch_factor": 1,
            "test_fraction": 0.0,
                "dev_fraction": 0.0,
                "prepare_feature_statistics": True,
        }
    )
    raw["data_source"]["workers"] = 1
    raw["model"].update({"d_model": 32, "layers": 1, "heads": 4, "ffn": 64})
    raw["train"].update(
        {
            "max_steps": 1,
            "epochs": 1,
            "precision": "fp32",
            "warmup": 1,
            "target_effective_batch_tokens": 0,
            "val_interval": 0,
            "checkpoint_interval": 1,
        }
    )
    config_path = tmp_path / "real.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    first = run_training_pipeline(config_path)
    second = run_training_pipeline(config_path)

    assert first.resumed is False
    assert second.resumed is True
    assert second.steps == 1
    assert (tmp_path / "manifest.jsonl").exists()
    assert (tmp_path / "real-checkpoints" / "resume" / "last.pt").exists()
    assert (tmp_path / "real-logs" / "epoch_metrics.csv").exists()
    assert (tmp_path / "real-logs" / "replay_selection.json").exists()


def test_proper_finish_writes_durable_finished_model(tmp_path: Path) -> None:
    """A real full-epoch finish writes the separated raw/EMA finished bundle.

    Uses ``max_steps: 0`` so the run trains a genuine full epoch (a proper
    finish), not a bounded ``--max-steps`` verification -- the latter must NOT
    produce a finished model. Asserts the finished/ layout, that the EMA
    safetensors is loadable and matches the recorded vocab size, and that a
    SECOND proper finish archives (never destroys) the prior finished model.
    """

    import json

    from safetensors.torch import load_file

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config" / "default.yaml").read_text(encoding="utf-8"))
    checkpoint_root = tmp_path / "real-checkpoints"
    raw["storage"].update(
        {
            "data_uri": str(root / "tests" / "fixtures"),
            "checkpoint_uri": str(checkpoint_root),
            "log_uri": str(tmp_path / "real-logs"),
            "local_cache_dir": str(tmp_path / "real-cache"),
        }
    )
    raw["data"].update(
        {
            "input_budget_tokens": 512,
            "canvas_budget_tokens": 256,
                "tokenized_replay_dir": str(tmp_path / "tokenized"),
                "window_manifest_path": str(tmp_path / "manifest.jsonl"),
                "feature_statistics_path": str(tmp_path / "feature_statistics.json"),
        }
    )
    raw["pipeline"].update(
        {
            "smoke": False,
            "batch_size": 2,
            "replay_glob": "match_4745722_game_state.parquet",
            "token_dictionary_uri": str(root / "data" / "Token_Dictionary.json"),
            "num_workers": 0,
            "test_fraction": 0.0,
                "dev_fraction": 0.0,
                "prepare_feature_statistics": True,
        }
    )
    raw["data_source"]["workers"] = 1
    raw["model"].update({"d_model": 32, "layers": 1, "heads": 4, "ffn": 64})
    raw["train"].update(
        {
            "max_steps": 0,  # 0 -> train a genuine full epoch (proper finish)
            "epochs": 1,
            "precision": "fp32",
            "warmup": 1,
            "target_effective_batch_tokens": 0,
            "val_interval": 0,
            "checkpoint_interval": 100,
        }
    )
    config_path = tmp_path / "finish.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_training_pipeline(config_path)

    finished = checkpoint_root / "finished"
    for name in (
        "model.raw.safetensors",
        "model.ema.safetensors",
        "finished.pt",
        "config.json",
        "finished_metadata.json",
    ):
        assert (finished / name).exists(), f"missing finished artifact: {name}"

    metadata = json.loads((finished / "finished_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stop_reason"] == "completed_all_epochs"
    assert metadata["default_serving_weights"] == "ema"
    assert metadata["weights"] == {
        "raw": "model.raw.safetensors",
        "ema": "model.ema.safetensors",
    }

    ema_state = load_file(str(finished / "model.ema.safetensors"))
    assert "output_head.weight" in ema_state
    assert int(metadata["vocab_size"]) == ema_state["output_head.weight"].shape[0]

    # config.json is valid JSON carrying the model architecture the run used.
    config_json = json.loads((finished / "config.json").read_text(encoding="utf-8"))
    assert config_json["model"]["d_model"] == 32

    # Durability: a second proper finish archives the previous finished model to
    # a finished_superseded_* sibling instead of destroying it, and writes a new
    # finished/ in place.
    run_training_pipeline(config_path)
    superseded = list(checkpoint_root.glob("finished_superseded_*"))
    assert superseded, "prior finished model was not archived on re-finish"
    assert (finished / "model.ema.safetensors").exists()


def test_storage_resolver_routes_local_and_s3(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    fake_s3 = FakeS3Client()
    resolver = StorageResolver(s3_client=fake_s3)

    assert resolver.exists(str(local)) is True
    assert parse_s3_uri("s3://bucket/path/file.pt").bucket == "bucket"
    assert resolver.exists("s3://bucket/path/file.pt") is True
    assert fake_s3.head_calls == [("bucket", "path/file.pt")]

    source = tmp_path / "source.txt"
    source.write_text("ok", encoding="utf-8")
    resolver.put_file(source, "s3://bucket/out/source.txt")
    assert fake_s3.upload_calls == [(str(source), "bucket", "out/source.txt")]


def test_pipeline_code_has_no_machine_specific_absolute_paths() -> None:
    pipeline_dir = Path(__file__).resolve().parents[1] / "src" / "thesis_ml" / "pipeline"
    text = "\n".join(path.read_text(encoding="utf-8") for path in pipeline_dir.glob("*.py"))
    forbidden = ("C:\\", "C:/Users", "/Users/", "/home/matae")
    assert not any(pattern in text for pattern in forbidden)


def test_acquisition_entrypoint_runs_standalone_dry_run(tmp_path: Path) -> None:
    config_path = _pipeline_config(tmp_path, source="local")

    result = run_acquisition(config_path, dry_run=True)

    assert result.dry_run is True
    assert result.commands
    assert result.commands[-1][0:2] == ["python", "quickstart.py"]


def test_kaggle_credentials_are_required_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _pipeline_config(tmp_path, source="kaggle")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with pytest.raises(CredentialError, match="KAGGLE_USERNAME"):
        run_acquisition(config_path, dry_run=True)

    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    result = run_acquisition(config_path, dry_run=True)
    assert result.commands[0][:4] == ["kaggle", "datasets", "download", "-d"]
    assert "secret" not in " ".join(result.commands[0])


def test_run_documentation_matches_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    run_md = (root / "RUN.md").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert "uv sync" in run_md
    assert "uv run thesis-ml-train" in run_md
    assert "uv run thesis-ml-acquire" in run_md
    assert 'thesis-ml-train = "thesis_ml.pipeline.train_pipeline:main"' in pyproject
    assert 'thesis-ml-acquire = "thesis_ml.pipeline.acquire_data:main"' in pyproject


class FakeS3Client:
    def __init__(self) -> None:
        self.head_calls: list[tuple[str, str]] = []
        self.upload_calls: list[tuple[str, str, str]] = []

    def head_object(self, *, Bucket: str, Key: str):
        self.head_calls.append((Bucket, Key))
        return {}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.upload_calls.append((Filename, Bucket, Key))

    def list_objects_v2(self, *, Bucket: str, Prefix: str):
        return {"Contents": [{"Key": Prefix + "last.pt"}]}

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_text("downloaded", encoding="utf-8")


def _pipeline_config(tmp_path: Path, *, source: str = "local") -> Path:
    default_config = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    raw = yaml.safe_load(default_config.read_text(encoding="utf-8"))
    raw["storage"]["data_uri"] = str(tmp_path / "data")
    raw["storage"]["raw_uri"] = str(tmp_path / "raw")
    raw["storage"]["checkpoint_uri"] = str(tmp_path / "checkpoints")
    raw["storage"]["log_uri"] = str(tmp_path / "logs")
    raw["storage"]["local_cache_dir"] = str(tmp_path / "cache")
    raw["data_source"]["source"] = source
    raw["data_source"]["extractor_path"] = "."
    raw["pipeline"]["smoke"] = True
    raw["pipeline"]["smoke_steps"] = 1
    raw["pipeline"]["seed"] = 7
    raw["pipeline"]["batch_size"] = 2
    raw["train"]["checkpoint_interval"] = 100
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path
