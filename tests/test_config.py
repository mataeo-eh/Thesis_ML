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
    # config/default.yaml is a PRE-TRAINING config (data.debut_mode=false), so
    # `fog` (the fine-tuning-only fog paradigm) must be absent -- it is
    # validated to None by `_validate_debut_mode_sections`, not populated.
    assert config.fog is None
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
    # Fraction of examples per epoch oversampled to t=1.0 exactly (Worker 2);
    # every configs/*.yaml now states this explicitly rather than relying on
    # silent inheritance.
    assert config.diffusion.mask_schedule.t_one_fraction == 0.1
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
    # Per-class loss weighting is likewise fine-tuning-only; pre-training uses
    # fully uniform published-MDLM-style weighting instead (see
    # CanvasCrossEntropyLoss.__init__), so this section must be absent too.
    assert config.loss.class_loss_weights is None


def test_local_profiles_extend_default_with_profile_specific_self_conditioning() -> None:
    root = DEFAULT_CONFIG.parents[1]
    for name in ("local_overfit.yaml", "local_overfit_v2.yaml", "local_full.yaml"):
        config = load_config(root / "configs" / name)
        assert config.data.sampling_interval_s == 1
        assert config.data.input_budget_tokens == 4096
        assert config.data.canvas_budget_tokens == 4096
        assert config.data.canvas_recon_fraction == 0.5
        assert config.train.require_cuda is True

    overfit = load_config(root / "configs" / "local_overfit.yaml")
    overfit_v2 = load_config(root / "configs" / "local_overfit_v2.yaml")
    full = load_config(root / "configs" / "local_full.yaml")
    assert overfit.pipeline.replay_subset_size == 25
    assert overfit.pipeline.validation_replay_count == 3
    assert overfit.pipeline.batch_size == 10
    assert overfit.pipeline.num_workers == 4
    assert overfit.pipeline.prefetch_factor == 4
    assert overfit.pipeline.persistent_workers is True
    assert overfit.train.epochs == 200
    assert overfit.train.early_stopping_patience_epochs == 5
    # overfit / overfit_v2 are PRE-TRAINING profiles (data.debut_mode=false,
    # inherited from config/default.yaml): class_loss_weights is a
    # fine-tuning-only section and must be None here, not populated.
    assert overfit.loss.class_loss_weights is None
    assert overfit.fog is None
    assert overfit.train.max_cuda_reserved_gb == 7.5
    assert overfit.model.gradient_checkpointing is True
    assert overfit_v2.train.epochs == 200
    assert overfit_v2.train.early_stopping_patience_epochs == 0
    assert overfit_v2.loss.class_loss_weights is None
    assert overfit_v2.fog is None
    assert overfit_v2.storage.checkpoint_uri == "checkpoints/local-overfitV2"
    assert overfit_v2.storage.log_uri == "tests/output/overfitV2"
    assert overfit_v2.storage.local_cache_dir == ".pipeline_cache/local-overfitV2"
    assert full.train.epochs == 8
    assert full.pipeline.train_replay_count == 870
    assert full.pipeline.validation_replay_count == 50
    assert full.pipeline.batch_size == 9
    assert full.pipeline.num_workers == 10
    assert full.pipeline.prefetch_factor == 4
    assert full.pipeline.persistent_workers is True
    # local_full is the PRE-TRAINING profile: debut_mode is False (full roll-out
    # target). The outcome token is still present at canvas position 0 and
    # denoised last (sampler.outcome_last True, asserted below).
    assert full.data.debut_mode is False
    assert full.model.self_conditioning is True
    assert full.model.gradient_checkpointing is True
    assert full.train.self_cond_prob == 0.5
    assert full.train.early_stopping_patience_epochs == 0
    assert full.train.max_cuda_reserved_gb == 7.5
    assert full.train.empty_cuda_cache_after_epoch is True
    assert full.sampler.outcome_last is True
    assert full.eval.heldout_split == "test"
    assert full.eval.debut_max_examples == 0
    for config in (overfit, overfit_v2):
        assert config.model.self_conditioning is False
        assert config.train.self_cond_prob == 0.0

    # local_overfit_v2_finetune.yaml is the FINE-TUNING profile
    # (data.debut_mode=true): the opposite requirement applies -- `fog` and
    # `loss.class_loss_weights` must be POPULATED (required, not merely
    # allowed), carrying forward the old pre-refactor effective values
    # (including the pad: 0.2 override that used to live in
    # local_overfit_v2.yaml before pre-training dropped class weighting).
    finetune = load_config(root / "configs" / "local_overfit_v2_finetune.yaml")
    assert finetune.data.debut_mode is True
    assert finetune.fog is not None
    assert finetune.fog.rate_distribution.name == "uniform"
    assert finetune.fog.rate_distribution.min == 0.0
    assert finetune.fog.rate_distribution.max == 0.8
    assert finetune.loss.class_loss_weights is not None
    assert finetune.loss.class_loss_weights.pad == 0.2
    assert finetune.loss.class_loss_weights.win_loss == 1.0
    assert finetune.diffusion.mask_schedule.t_one_fraction == 0.1
    assert finetune.sampler.outcome_last is True


def test_pretraining_config_rejects_fog_and_class_loss_weights_sections(tmp_path: Path) -> None:
    """A pre-training config carrying fine-tuning-only knobs must be REJECTED.

    `fog` and `loss.class_loss_weights` are fine-tuning-only sections: a
    debut_mode=false config that carries either is refused by `load_config`
    (no dead knobs sitting in a config nothing reads), with the exact
    validation messages from `_validate_debut_mode_sections`. Conversely a
    debut_mode=true config REQUIRES both sections.
    """

    base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert base["data"]["debut_mode"] is False

    # Pre-training config carrying `fog` -> rejected.
    with_fog = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with_fog["fog"] = {"rate_distribution": {"name": "uniform", "min": 0.0, "max": 0.8}}
    fog_path = tmp_path / "pretrain_with_fog.yaml"
    fog_path.write_text(yaml.safe_dump(with_fog), encoding="utf-8")
    with pytest.raises(
        ConfigError, match=r"config\.fog must not be set when data\.debut_mode=false"
    ):
        load_config(fog_path)

    # Pre-training config carrying `loss.class_loss_weights` -> rejected.
    with_weights = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with_weights["loss"]["class_loss_weights"] = {
        "enemy_observed_reconstruction": 1.0,
        "enemy_fogged_reconstruction": 1.0,
        "enemy_future_prediction": 1.0,
        "delimiter": 1.0,
        "end": 1.0,
        "pad": 0.2,
        "win_loss": 1.0,
    }
    weights_path = tmp_path / "pretrain_with_weights.yaml"
    weights_path.write_text(yaml.safe_dump(with_weights), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"config\.loss\.class_loss_weights must not be set when data\.debut_mode=false",
    ):
        load_config(weights_path)

    # The mirror-image requirement: debut_mode=true REQUIRES both sections.
    debut_missing_both = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    debut_missing_both["data"]["debut_mode"] = True
    debut_path = tmp_path / "debut_missing_fog.yaml"
    debut_path.write_text(yaml.safe_dump(debut_missing_both), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.fog is required when data\.debut_mode=true"):
        load_config(debut_path)

    debut_missing_weights = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    debut_missing_weights["data"]["debut_mode"] = True
    debut_missing_weights["fog"] = {
        "rate_distribution": {"name": "uniform", "min": 0.0, "max": 0.8}
    }
    debut_weights_path = tmp_path / "debut_missing_weights.yaml"
    debut_weights_path.write_text(yaml.safe_dump(debut_missing_weights), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"config\.loss\.class_loss_weights is required when data\.debut_mode=true",
    ):
        load_config(debut_weights_path)


def test_t_one_fraction_out_of_range_is_rejected(tmp_path: Path) -> None:
    """`diffusion.mask_schedule.t_one_fraction` must be a probability in [0, 1]."""

    for bad_value in (-0.1, 1.5):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        raw["diffusion"]["mask_schedule"]["t_one_fraction"] = bad_value
        config_path = tmp_path / f"bad_fraction_{bad_value}.yaml"
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(
            ConfigError,
            match=r"diffusion\.mask_schedule\.t_one_fraction must be in \[0, 1\]",
        ):
            load_config(config_path)


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
