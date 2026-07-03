from pathlib import Path

import pytest
import yaml

from thesis_ml.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_valid_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.data.sampling_interval_s == 1
    assert config.data.input_budget_tokens == 4096
    assert config.data.canvas_budget_tokens == 4096
    assert config.data.canvas_recon_fraction == 0.5
    assert config.data.within_type_tiebreak == "unit_id"
    assert config.fog.rate_distribution.name == "uniform"
    assert config.fog.rate_distribution.min == 0.0
    assert config.fog.rate_distribution.max == 0.8
    assert config.model.d_model == 256
    assert config.model.layers == 10
    assert config.model.heads == 4
    assert config.model.ffn == 1024
    assert config.model.qk_norm is True
    assert config.model.self_conditioning is False
    assert config.model.gradient_checkpointing is False
    assert config.model.rope_theta == 500000.0
    assert config.model.rope_scaling.rope_type == "llama3"
    assert config.model.rope_scaling.factor == 8.0
    assert config.model.rope_scaling.low_freq_factor == 1.0
    assert config.model.rope_scaling.high_freq_factor == 4.0
    assert config.model.rope_scaling.original_max_position_embeddings == 8192
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
    assert config.pipeline.batch_size == 8
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
    assert config.train.epochs == 6
    assert config.train.early_stopping_patience_epochs == 0
    assert config.train.early_stopping_min_relative_improvement == 0.001
    assert config.train.val_interval == 1000
    assert config.train.checkpoint_interval == 1000
    assert config.train.checkpoint_dir == "checkpoints"
    assert config.train.ema_decay == 0.9999
    assert config.train.confidence_loss_weight == 0.1
    assert config.train.self_cond_prob == 0.0
    assert config.train.precision == "bf16"
    assert config.train.require_cuda is False
    assert config.train.max_cuda_reserved_gb == 0.0
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


def test_local_profiles_extend_default_and_disable_self_conditioning() -> None:
    root = DEFAULT_CONFIG.parents[1]
    for name in ("local_overfit.yaml", "local_overfit_v2.yaml", "local_full.yaml"):
        config = load_config(root / "configs" / name)
        assert config.data.sampling_interval_s == 1
        assert config.data.input_budget_tokens == 4096
        assert config.data.canvas_budget_tokens == 4096
        assert config.data.canvas_recon_fraction == 0.5
        assert config.model.self_conditioning is False
        assert config.train.self_cond_prob == 0.0
        assert config.train.require_cuda is True

    overfit = load_config(root / "configs" / "local_overfit.yaml")
    overfit_v2 = load_config(root / "configs" / "local_overfit_v2.yaml")
    full = load_config(root / "configs" / "local_full.yaml")
    assert overfit.pipeline.replay_subset_size == 25
    assert overfit.pipeline.validation_replay_count == 3
    assert overfit.pipeline.batch_size == 10
    assert overfit.pipeline.num_workers == 4
    assert overfit.pipeline.prefetch_factor == 4
    assert overfit.train.epochs == 200
    assert overfit.train.early_stopping_patience_epochs == 5
    assert overfit.loss.class_loss_weights.pad == 1.0
    assert overfit.train.max_cuda_reserved_gb == 7.5
    assert overfit.model.gradient_checkpointing is True
    assert overfit_v2.train.epochs == 200
    assert overfit_v2.train.early_stopping_patience_epochs == 0
    assert overfit_v2.loss.class_loss_weights.pad == 0.2
    assert overfit_v2.storage.checkpoint_uri == "checkpoints/local-overfitV2"
    assert overfit_v2.storage.log_uri == "tests/output/overfitV2"
    assert overfit_v2.storage.local_cache_dir == ".pipeline_cache/local-overfitV2"
    assert full.train.epochs == 8


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["unexpected"] = True
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="unexpected"):
        load_config(config_path)


def test_nested_extends_and_cycles_are_handled(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    middle.write_text("extends: base.yaml\ntrain:\n  epochs: 20\n", encoding="utf-8")
    child.write_text("extends: middle.yaml\ntrain:\n  epochs: 30\n", encoding="utf-8")

    assert load_config(child).train.epochs == 30

    base.write_text("extends: child.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="extends cycle"):
        load_config(child)


def test_wrong_typed_value_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["layers"] = "sixteen"
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="model.layers"):
        load_config(config_path)
