from pathlib import Path

import pytest
import yaml

from thesis_ml.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_valid_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.data.sampling_interval_s == 5
    assert config.data.input_window_timesteps == 60
    assert config.data.canvas_budget_tokens == 2048
    assert config.data.within_type_tiebreak == "unit_id"
    assert config.fog.rate_distribution.name == "uniform"
    assert config.fog.rate_distribution.min == 0.0
    assert config.fog.rate_distribution.max == 0.8
    assert config.model.d_model == 1536
    assert config.model.layers == 16
    assert config.model.heads == 12
    assert config.model.ffn == 4096
    assert config.diffusion.mask_schedule.name == "linear"
    assert config.diffusion.mask_schedule.t_distribution == "uniform"
    assert config.diffusion.mask_schedule.min == 0.0
    assert config.diffusion.mask_schedule.max == 1.0
    assert config.diffusion.mask_schedule.loss_reweight == "inverse_t"
    assert config.storage.data_uri == "data/processed/quickstart"
    assert config.storage.raw_uri == "data/raw/replays"
    assert config.storage.checkpoint_uri == "checkpoints"
    assert config.storage.log_uri == "logs"
    assert config.storage.local_cache_dir == ".pipeline_cache"
    assert config.data_source.source == "kaggle"
    assert config.data_source.kaggle_dataset == "mataeoanderson/sc2-replay-data"
    assert config.data_source.kaggle_username_env == "KAGGLE_USERNAME"
    assert config.data_source.kaggle_key_env == "KAGGLE_KEY"
    assert config.data_source.extractor_path == "../SC2-gamestate-extractor"
    assert config.data_source.extractor_command == "python quickstart.py"
    assert config.data_source.workers == 4
    assert config.pipeline.auto_acquire is False
    assert config.pipeline.smoke is False
    assert config.pipeline.smoke_steps == 2
    assert config.pipeline.seed == 123
    assert config.pipeline.batch_size == 2
    assert config.pipeline.examples_per_replay == 1
    assert config.pipeline.replay_glob == "*.parquet"
    assert config.pipeline.token_dictionary_uri == "data/Token_Dictionary.json"
    assert config.pipeline.perspectives == "p1,p2"
    assert config.train.lr == 0.0003
    assert config.train.beta1 == 0.9
    assert config.train.beta2 == 0.95
    assert config.train.weight_decay == 0.1
    assert config.train.adam_eps == 1e-8
    assert config.train.warmup == 2000
    assert config.train.lr_floor_ratio == 0.1
    assert config.train.grad_clip == 1.0
    assert config.train.accum == "as-needed"
    assert config.train.accumulation_steps == 1
    assert config.train.target_effective_batch_tokens == 524288
    assert config.train.max_steps == 100000
    assert config.train.val_interval == 1000
    assert config.train.checkpoint_interval == 1000
    assert config.train.checkpoint_dir == "checkpoints"
    assert config.train.ema_decay == 0.9999
    assert config.train.confidence_loss_weight == 0.1
    assert config.train.precision == "bf16"
    assert config.sampler.max_steps == 48
    assert config.sampler.temperature.start == 0.8
    assert config.sampler.temperature.end == 0.4
    assert config.sampler.entropy_bound == 0.1
    assert config.sampler.confidence_threshold == 0.0
    assert config.sampler.min_commit_per_step == 1
    assert config.eval.heldout_split == "validation"
    assert config.eval.timing_tolerance_buckets == 1
    assert config.eval.fog_rate == 0.0
    assert config.loss.use_fused_cross_entropy is False
    assert config.loss.class_loss_weights.enemy_observed_reconstruction == 1.0
    assert config.loss.class_loss_weights.enemy_fogged_reconstruction == 1.0
    assert config.loss.class_loss_weights.enemy_future_prediction == 1.0
    assert config.loss.class_loss_weights.delimiter == 1.0
    assert config.loss.class_loss_weights.end == 1.0
    assert config.loss.class_loss_weights.pad == 1.0


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["unexpected"] = True
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="unexpected"):
        load_config(config_path)


def test_wrong_typed_value_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["layers"] = "sixteen"
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="model.layers"):
        load_config(config_path)
