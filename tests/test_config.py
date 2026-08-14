from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from thesis_ml.config import ConfigError, load_config, toggle_fingerprint


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_valid_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.data.sampling_interval_s == 1
    assert config.data.input_budget_tokens == 4096
    assert config.data.canvas_budget_tokens == 4096
    assert config.data.canvas_recon_fraction == 0.5
    assert config.data.within_type_tiebreak == "unit_id"
    assert config.data.feature_statistics_path == "data/processed/feature_statistics.v1.json"
    assert config.fog.rate_distribution.name == "power"
    assert config.fog.rate_distribution.min == 0.0
    assert config.fog.rate_distribution.max == 0.8
    assert config.fog.rate_distribution.power == 2.0
    assert config.model.d_model == 384
    assert config.model.layers == 12
    assert config.model.heads == 6
    assert config.model.ffn == 1536
    assert config.model.qk_norm is True
    assert config.model.self_conditioning is True
    assert config.model.gradient_checkpointing is False
    # Architecture toggles: all three parse from YAML. `frozen_input_kv` was
    # PROMOTED to a default on 2026-08-09 (owner decision on ablation arm 01's
    # evidence: no meaningful loss difference, much faster inference), so the
    # default-derived architecture identity is now
    # `dense-multinomial-SC2-v2+frozen_input_kv`. The other two remain
    # experiments and stay false.
    assert config.model.frozen_input_kv is True
    assert config.model.segment_embeddings is False
    assert config.model.per_segment_positions is False
    assert config.model.rope_theta == 500000.0
    assert config.model.rope_scaling.rope_type == "llama3"
    assert config.model.rope_scaling.factor == 8.0
    assert config.model.rope_scaling.low_freq_factor == 1.0
    assert config.model.rope_scaling.high_freq_factor == 4.0
    assert config.model.rope_scaling.original_max_position_embeddings == 8192
    assert config.diffusion.process == "uniform"
    assert config.diffusion.schedule.name == "linear"
    assert config.diffusion.schedule.t_distribution == "power"
    assert config.diffusion.schedule.t_distribution_power == 2.0
    assert config.diffusion.schedule.min == 0.0
    assert config.diffusion.schedule.max == 1.0
    assert config.diffusion.schedule.t_one_fraction == 0.05
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
    assert config.pipeline.prepare_feature_statistics is False
    assert config.train.lr == 0.0003
    assert config.train.beta1 == 0.9
    assert config.train.beta2 == 0.95
    assert config.train.weight_decay == 0.1
    assert config.train.adam_eps == 1e-8
    assert config.train.warmup == 500
    assert config.train.lr_floor_ratio == 0.01
    assert config.train.lr_schedule == "wsd"
    assert config.train.lr_decay_ratio == 0.20
    assert config.train.grad_clip == 1.0
    assert config.train.accum == "as-needed"
    assert config.train.accumulation_steps == 5
    assert config.train.target_effective_batch_tokens == 0
    assert config.train.max_steps == 100000
    assert config.train.epochs == 6
    assert config.train.early_stopping_patience_epochs == 0
    assert config.train.early_stopping_min_relative_improvement == 0.001
    assert config.train.val_interval == 1000
    assert config.train.checkpoint_interval == 1000
    assert config.train.checkpoint_dir == "checkpoints"
    assert config.train.resume_checkpoint_subdir == "resume"
    assert config.train.best_checkpoint_subdir == "best"
    assert config.train.durable_checkpoint_subdir == "durable"
    assert config.train.save_best_checkpoint is True
    assert config.train.durable_checkpoint_interval_epochs == 5
    # ema_decay is now a CEILING on a run-derived decay, and ema_horizon_ratio is
    # what sizes the EMA window to the run. 0.9999 is a 10,000-step window, so at
    # ratio 0.1 a run of 100,000+ steps sits exactly at the old fixed behavior.
    assert config.train.ema_decay == 0.9999
    assert config.train.ema_horizon_ratio == 0.1
    assert config.train.confidence_loss_weight == 0.0
    assert config.train.self_cond_prob == 0.5
    assert config.train.precision == "bf16"
    assert config.train.require_cuda is False
    assert config.train.max_cuda_reserved_gb == 0.0
    assert config.sampler.max_steps == 64
    assert config.sampler.temperature.start == 0.8
    assert config.sampler.temperature.end == 0.4
    assert config.sampler.temperature.exponent == 1.0
    assert config.sampler.entropy_bound == 0.1
    assert config.sampler.adaptive_stop is True
    assert config.sampler.entropy_threshold == 0.005
    assert config.sampler.stability_steps == 2
    assert not hasattr(config.sampler, "confidence_threshold")
    assert not hasattr(config.sampler, "min_commit_per_step")
    assert not hasattr(config.sampler, "outcome_last")
    assert config.eval.heldout_split == "validation"
    assert config.eval.timing_tolerance_buckets == 1
    assert config.eval.fog_rate == 0.0
    assert config.loss.use_fused_cross_entropy is False
    # Per-class loss weighting is shared by both modes.
    assert config.loss.class_loss_weights is not None
    assert config.loss.class_loss_weights.pad == 0.1
    assert config.loss.class_loss_weights.end == pytest.approx(24.633333333333333)
    assert config.loss.class_loss_weights.win_loss == 1.0


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
    # The overfit profile names its replays explicitly (10 train + 3 dev chosen
    # at the corpus median token count) instead of drawing a seeded subset, so
    # the seeded-subset knobs must be OFF -- leaving either non-zero would mean
    # two selection mechanisms fighting over the same run.
    assert overfit.pipeline.replay_subset_size == 0
    assert overfit.pipeline.validation_replay_count == 0
    assert len(overfit.pipeline.train_replay_ids.split(",")) == 10
    assert len(overfit.pipeline.dev_replay_ids.split(",")) == 3
    # Feature statistics are keyed to that exact train list, so this profile
    # rebuilds them rather than loading a stale frozen artifact.
    assert overfit.pipeline.prepare_feature_statistics is True
    assert overfit.pipeline.batch_size == 10
    assert overfit.pipeline.num_workers == 4
    assert overfit.pipeline.prefetch_factor == 4
    assert overfit.pipeline.persistent_workers is True
    assert overfit.train.epochs == 150
    # Overfitting is the point; early stopping must never cut the run short.
    assert overfit.train.early_stopping_patience_epochs == 0
    # 34 steps/epoch x 150 epochs = 5100 steps. Warmup must stay far below that
    # or the run ends while still ramping and never trains at the configured lr.
    assert overfit.train.warmup == 40
    assert overfit.train.warmup < 34 * overfit.train.epochs
    # max_steps 0 is what makes the cosine horizon DERIVED rather than fixed:
    # train_pipeline injects optimizer_steps_per_epoch * epochs, which
    # is the horizon _lr_multiplier decays over. A non-zero value here would pin
    # the schedule to a step count unrelated to the configured epoch budget.
    assert overfit.train.max_steps == 0
    assert overfit_v2.train.max_steps == 0
    # This profile evaluates dev once per epoch, not at each of the ten interval
    # reports: a dev pass costs more than the training slice it follows here.
    # The DEFAULT stays true, for runs that converge in a single epoch.
    assert overfit.train.interval_dev_evaluation is False
    assert overfit_v2.train.interval_dev_evaluation is False
    assert load_config(DEFAULT_CONFIG).train.interval_dev_evaluation is True
    assert full.train.interval_dev_evaluation is True
    # The train side of those interval rows is switched off too, so this profile
    # reports BOTH train and dev once per epoch and writes no interval rows at
    # all. Same default-true contract as the dev knob.
    assert overfit.train.interval_train_evaluation is False
    assert overfit_v2.train.interval_train_evaluation is False
    assert load_config(DEFAULT_CONFIG).train.interval_train_evaluation is True
    assert full.train.interval_train_evaluation is True
    # Historical profiles explicitly pin their original unit pretraining
    # weights so the new canonical defaults do not rewrite old experiments.
    assert overfit.loss.class_loss_weights is not None
    assert overfit.loss.class_loss_weights.pad == 1.0
    assert overfit.loss.class_loss_weights.end == 1.0
    assert overfit.fog is not None
    assert overfit.train.max_cuda_reserved_gb == 7.5
    assert overfit.model.gradient_checkpointing is True
    # V2 overrides the inherited 150 down to 100 so it and the five
    # configs/ablation_*.yaml arms that extend it share one step budget:
    # 34 steps/epoch x 100 = 3400. tests/run_ablation_sweep.sh hardcodes those
    # same two numbers to decide skip / resume, so they are pinned here.
    assert overfit_v2.train.epochs == 100
    assert overfit_v2.train.early_stopping_patience_epochs == 0
    # The V2 / ablation LR schedule: leave the peak immediately on a constant,
    # shallower-than-cosine slope and end at a lower floor (0.03 * 3.0e-4 =
    # 9.0e-6 instead of 3.0e-5), to stop the late-training loss curve hovering
    # around a minimum. Warmup is unchanged, so only the decay differs.
    assert overfit_v2.train.lr_schedule == "linear"
    assert overfit_v2.train.lr_floor_ratio == 0.03
    assert overfit_v2.train.warmup == overfit.train.warmup
    # local_overfit.yaml (V1) and local_full.yaml keep the inherited cosine/0.1.
    assert overfit.train.lr_schedule == "cosine"
    assert overfit.train.lr_floor_ratio == 0.1
    assert full.train.lr_schedule == "cosine"
    assert full.train.lr_floor_ratio == 0.1
    # V2 inherits the explicit selection unchanged; only output paths differ.
    assert overfit_v2.pipeline.train_replay_ids == overfit.pipeline.train_replay_ids
    assert overfit_v2.pipeline.dev_replay_ids == overfit.pipeline.dev_replay_ids
    assert overfit_v2.loss.class_loss_weights is not None
    assert overfit_v2.loss.class_loss_weights.pad == 1.0
    assert overfit_v2.fog is not None
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
    # target). The outcome token remains at canvas position 1 after clamped BOS.
    assert full.data.debut_mode is False
    assert full.model.self_conditioning is True
    assert full.model.gradient_checkpointing is True
    assert full.train.self_cond_prob == 0.5
    assert full.train.early_stopping_patience_epochs == 0
    assert full.train.max_cuda_reserved_gb == 7.5
    assert full.train.empty_cuda_cache_after_epoch is True
    assert full.eval.heldout_split == "test"
    assert full.eval.debut_max_examples == 0
    for config in (overfit, overfit_v2, full):
        assert config.diffusion.process == "uniform"
        assert config.diffusion.schedule.t_distribution == "uniform"
        assert config.diffusion.schedule.t_distribution_power == 1.0
        assert config.diffusion.schedule.t_one_fraction == 0.0
        assert config.fog.rate_distribution.name == "uniform"
        assert config.fog.rate_distribution.power == 1.0
        assert config.model.self_conditioning is True
        assert config.train.self_cond_prob == 0.5
        assert config.train.confidence_loss_weight == 0.0
    # local_overfit_v2_finetune.yaml is the FINE-TUNING profile
    # (data.debut_mode=true), preserving its historical pad: 0.2 override.
    finetune = load_config(root / "configs" / "local_overfit_v2_finetune.yaml")
    assert finetune.data.debut_mode is True
    assert finetune.fog is not None
    assert finetune.fog.rate_distribution.name == "uniform"
    assert finetune.fog.rate_distribution.min == 0.0
    assert finetune.fog.rate_distribution.max == 0.8
    assert finetune.loss.class_loss_weights is not None
    assert finetune.loss.class_loss_weights.pad == 0.2
    assert finetune.loss.class_loss_weights.win_loss == 1.0
    assert finetune.diffusion.process == "uniform"
    assert finetune.diffusion.schedule.t_one_fraction == 0.0

    # Every profile transitively extends config/default.yaml, so the model
    # toggles reach all four through _deep_merge without a per-profile edit.
    #
    # `frozen_input_kv` is now a DEFAULT (config/default.yaml), and no profile in
    # this group opts back out -- this loop is what proves the promotion actually
    # propagated by inheritance rather than needing to be restated file by file.
    # local_overfit_v2.yaml deliberately DROPPED its old `false` pin so it and
    # everything extending it (including the fine-tune profile, which warm-starts
    # from a pre-training checkpoint and so must match its architecture) inherit
    # the new value. The other two toggles are still experiments and stay false;
    # local_overfit_v2.yaml restates those as its ablation control surface.
    for config in (overfit, overfit_v2, full, finetune):
        assert config.model.frozen_input_kv is True
        assert config.model.segment_embeddings is False
        assert config.model.per_segment_positions is False

    # configs/ablation_00_baseline.yaml is the one profile that opts back OUT. It
    # pins all three false so the completed prompt-009 sweep's baseline arm still
    # resolves to the unsuffixed `dense-multinomial-SC2-v2` identity its recorded
    # checkpoint was trained under -- see the ARMS table in
    # src/thesis_ml/viz/outcome_probe.py, which builds a model from this config
    # and loads that checkpoint into it.
    ablation_baseline = load_config(root / "configs" / "ablation_00_baseline.yaml")
    assert ablation_baseline.model.frozen_input_kv is False
    assert ablation_baseline.model.segment_embeddings is False
    assert ablation_baseline.model.per_segment_positions is False
    assert toggle_fingerprint(ablation_baseline.model) == ""
    # It must keep reading the baseline's existing artifacts, so its storage
    # paths are the inherited V2 ones rather than a private namespace.
    assert ablation_baseline.storage.checkpoint_uri == overfit_v2.storage.checkpoint_uri
    assert ablation_baseline.storage.log_uri == overfit_v2.storage.log_uri
    # Everything else about the baseline arm is V2's, unchanged.
    assert ablation_baseline.train.epochs == overfit_v2.train.epochs
    assert ablation_baseline.train.lr_schedule == overfit_v2.train.lr_schedule
    assert ablation_baseline.pipeline.train_replay_ids == overfit_v2.pipeline.train_replay_ids


def test_small_training_v3_owns_the_full_run_contract() -> None:
    config = load_config(DEFAULT_CONFIG.parents[1] / "configs" / "smallTrainingTestV3.yaml")

    assert (config.model.d_model, config.model.layers, config.model.heads, config.model.ffn) == (
        384,
        12,
        6,
        1536,
    )
    assert config.model.d_model // config.model.heads == 64
    assert config.train.lr == pytest.approx(3e-4)
    assert config.train.lr_floor_ratio == pytest.approx(0.01)
    assert config.train.lr_schedule == "wsd"
    assert config.train.warmup == 500
    assert config.train.lr_decay_ratio == pytest.approx(0.20)
    assert config.train.epochs == 50
    assert config.train.early_stopping_patience_epochs == 10
    assert config.train.accumulation_steps == 5
    assert config.train.target_effective_batch_tokens == 0
    assert config.train.checkpoint_interval == 100
    assert config.train.resume_checkpoint_subdir == "resume"
    assert config.train.best_checkpoint_subdir == "best"
    assert config.train.durable_checkpoint_subdir == "durable"
    assert config.train.durable_checkpoint_interval_epochs == 5
    assert config.train.interval_dev_evaluation is False
    assert config.train.interval_train_evaluation is False
    assert config.fog.rate_distribution.name == "power"
    assert config.fog.rate_distribution.power == 2.0
    assert config.diffusion.schedule.t_distribution == "power"
    assert config.diffusion.schedule.t_distribution_power == 2.0
    assert config.diffusion.schedule.t_one_fraction == pytest.approx(0.05)
    assert config.loss.class_loss_weights.pad == pytest.approx(0.1)
    assert config.loss.class_loss_weights.end == pytest.approx(42862 / 1740)
    assert config.storage.checkpoint_uri == "tests/output/smallTrainingTestV3/checkpoints"
    assert config.storage.log_uri == "tests/output/smallTrainingTestV3/metrics"


#: The five ablation arm profiles, in sweep order. Each extends
#: configs/local_overfit_v2.yaml and must differ from it ONLY by its toggles and
#: its redirected storage paths.
ABLATION_CONFIGS = (
    "ablation_01_frozen_input_kv.yaml",
    "ablation_02_segment_embeddings.yaml",
    "ablation_03_per_segment_positions.yaml",
    "ablation_04_segment_embeddings_plus_per_segment_positions.yaml",
    "ablation_05_frozen_input_kv_plus_segment_embeddings.yaml",
)


def test_ablation_arms_inherit_the_overfit_v2_run_length_and_lr_schedule() -> None:
    """Every arm must share V2's schedule so only the toggle differs.

    The arms are compared to each other and to the V2 baseline epoch for epoch,
    which is only meaningful if the run length, the LR trajectory, and the EMA
    window are byte-identical across arms. Each arm inherits all of that from
    configs/local_overfit_v2.yaml, so this test asserts the inheritance actually
    holds rather than that each file restates the values.
    """

    root = DEFAULT_CONFIG.parents[1]
    baseline = load_config(root / "configs" / "local_overfit_v2.yaml")
    for name in ABLATION_CONFIGS:
        arm = load_config(root / "configs" / name)
        assert arm.train.epochs == baseline.train.epochs, name
        # max_steps 0 is what keeps BOTH schedule horizons derived from `epochs`;
        # a non-zero value in an arm would pin its LR decay and EMA window to a
        # step count unrelated to the epoch budget the other arms use.
        assert arm.train.max_steps == 0, name
        assert arm.train.lr == baseline.train.lr, name
        assert arm.train.warmup == baseline.train.warmup, name
        assert arm.train.lr_schedule == baseline.train.lr_schedule, name
        assert arm.train.lr_floor_ratio == baseline.train.lr_floor_ratio, name
        assert arm.train.ema_decay == baseline.train.ema_decay, name
        assert arm.train.ema_horizon_ratio == baseline.train.ema_horizon_ratio, name
        assert arm.pipeline.batch_size == baseline.pipeline.batch_size, name
        assert arm.pipeline.train_replay_ids == baseline.pipeline.train_replay_ids, name
        assert arm.pipeline.dev_replay_ids == baseline.pipeline.dev_replay_ids, name
        # Private storage per arm: a shared checkpoint_uri would let the pipeline
        # auto-resume one arm from another's last.pt.
        assert arm.storage.checkpoint_uri != baseline.storage.checkpoint_uri, name
        assert arm.storage.log_uri != baseline.storage.log_uri, name
        assert arm.storage.local_cache_dir != baseline.storage.local_cache_dir, name


def test_ablation_sweep_driver_step_constants_match_the_configured_run_length() -> None:
    """Pin the sweep driver's skip/resume arithmetic to the actual config.

    tests/run_ablation_sweep.sh no longer caps the arms with `--max-steps`; it
    still needs STEPS_PER_EPOCH x EXPECTED_EPOCHS to recognize a finished arm from
    its checkpoint's `global_step`. Those two numbers are shell constants that
    cannot read the YAML, so a change to `train.epochs` would silently desync
    them -- the driver would either re-run finished arms or skip unfinished ones.
    This test is the link between the two.
    """

    root = DEFAULT_CONFIG.parents[1]
    script = (root / "tests" / "run_ablation_sweep.sh").read_text(encoding="utf-8")
    declared = {
        key: int(value)
        for key, value in (
            line.split("=", 1)
            for line in script.splitlines()
            if line.startswith(("STEPS_PER_EPOCH=", "EXPECTED_EPOCHS="))
        )
    }
    assert declared["EXPECTED_EPOCHS"] == load_config(
        root / "configs" / "local_overfit_v2.yaml"
    ).train.epochs
    # 334 train windows at batch_size 10 -> 34 optimizer steps per epoch. See the
    # replay-selection comment in configs/local_overfit.yaml.
    assert declared["STEPS_PER_EPOCH"] == 34
    # The driver must not reintroduce a step cap: the arms are meant to run their
    # LR decay and EMA window to completion, and a --max-steps run additionally
    # skips export_finished_model(). Only the train_pipeline invocation itself is
    # inspected -- the header comments and the status-file `echo`s both mention the
    # flag they no longer pass, and matching on the whole file would flag those.
    lines = script.splitlines()
    launch_start = next(
        index for index, line in enumerate(lines) if "thesis_ml.pipeline.train_pipeline" in line
    )
    # The invocation is line-continued with trailing backslashes; take it plus
    # every continuation line that follows.
    launch = [lines[launch_start]]
    while launch[-1].rstrip().endswith("\\"):
        launch.append(lines[launch_start + len(launch)])
    assert not any("--max-steps" in line for line in launch), "\n".join(launch)


#: The three memorization-probe profiles, in sweep order. Each extends
#: configs/local_overfit_v2.yaml and must differ from it ONLY by the training /
#: data knob(s) under test and by its redirected storage paths.
MEMORIZATION_CONFIGS = (
    "memorization_01_no_regularization.yaml",
    "memorization_02_t_one_oversample.yaml",
    "memorization_03_no_regularization_plus_t_one_oversample.yaml",
)

#: The exact values the probe family is defined by. Restated here rather than
#: read from the YAML so a stray edit to a profile fails a test instead of
#: silently changing what the experiment measures.
NO_REGULARIZATION_WEIGHT_DECAY = 0.0
NO_REGULARIZATION_FOG_MAX = 0.0
OVERSAMPLED_T_ONE_FRACTION = 0.25


def test_memorization_arms_isolate_their_knobs_against_the_shared_v2_schedule() -> None:
    """Each probe arm must change only what it claims to change.

    The three arms are read against each other and against ablation arm 01, so
    the run length, LR trajectory, EMA window, batch size, and replay subset have
    to be identical across all of them -- every one of those is inherited from
    configs/local_overfit_v2.yaml, and this test asserts the inheritance actually
    holds rather than that each file restates it. Storage must NOT be shared:
    these arms are architecturally identical to one another, so unlike the
    ablation arms nothing in the checkpoint loader would catch one resuming from
    another's weights. The private paths are the only guard.
    """

    root = DEFAULT_CONFIG.parents[1]
    baseline = load_config(root / "configs" / "local_overfit_v2.yaml")
    seen_checkpoint_uris: set[str] = set()
    for name in MEMORIZATION_CONFIGS:
        arm = load_config(root / "configs" / name)

        # Shared schedule and data: inherited, never restated.
        assert arm.train.epochs == baseline.train.epochs, name
        assert arm.train.max_steps == 0, name
        assert arm.train.lr == baseline.train.lr, name
        assert arm.train.warmup == baseline.train.warmup, name
        assert arm.train.lr_schedule == baseline.train.lr_schedule, name
        assert arm.train.lr_floor_ratio == baseline.train.lr_floor_ratio, name
        assert arm.train.ema_decay == baseline.train.ema_decay, name
        assert arm.train.ema_horizon_ratio == baseline.train.ema_horizon_ratio, name
        assert arm.pipeline.batch_size == baseline.pipeline.batch_size, name
        assert arm.pipeline.train_replay_ids == baseline.pipeline.train_replay_ids, name
        assert arm.pipeline.dev_replay_ids == baseline.pipeline.dev_replay_ids, name

        # Architecture: inherited from config/default.yaml, matching ablation arm
        # 01. If this drifts the arms stop being comparable to arm 01 AND stop
        # being comparable to each other.
        assert arm.model.frozen_input_kv is True, name
        assert arm.model.segment_embeddings is False, name
        assert arm.model.per_segment_positions is False, name

        # Knobs the probe family explicitly does NOT touch. Named individually
        # because "regularization off" is a judgement call, and the call made
        # here was: only weight decay and fog. grad_clip is a stability guard,
        # self_cond_prob changes what the model IS (the sampler feeds the
        # prediction back), and EMA only affects served/eval weights while the
        # memorization signal is read off raw-weight train loss.
        assert arm.train.grad_clip == baseline.train.grad_clip, name
        assert arm.train.self_cond_prob == baseline.train.self_cond_prob, name
        assert arm.train.early_stopping_patience_epochs == 0, name
        assert arm.data.debut_mode is False, name

        # Private storage per arm, and distinct from the V2 baseline's.
        assert arm.storage.checkpoint_uri != baseline.storage.checkpoint_uri, name
        assert arm.storage.log_uri != baseline.storage.log_uri, name
        assert arm.storage.local_cache_dir != baseline.storage.local_cache_dir, name
        assert arm.storage.checkpoint_uri not in seen_checkpoint_uris, name
        seen_checkpoint_uris.add(arm.storage.checkpoint_uri)


def test_memorization_arm_3_is_exactly_the_union_of_arms_1_and_2() -> None:
    """Arm 3 only means something if it is arms 1 and 2 applied together.

    YAML `extends` is single-parent, so arm 3 has to RESTATE arm 1's weight
    decay / fog and arm 2's t_one_fraction rather than inherit them. That
    duplication is the failure mode this test exists for: edit arm 1's fog and
    arm 3 silently stops being the combined condition, and the whole
    "do the two changes compose?" reading of the sweep becomes unsupportable
    while every file still looks fine.
    """

    root = DEFAULT_CONFIG.parents[1]
    baseline = load_config(root / "configs" / "local_overfit_v2.yaml")
    arm_1 = load_config(root / "configs" / "memorization_01_no_regularization.yaml")
    arm_2 = load_config(root / "configs" / "memorization_02_t_one_oversample.yaml")
    arm_3 = load_config(
        root / "configs" / "memorization_03_no_regularization_plus_t_one_oversample.yaml"
    )

    # Arm 1: regularization off, t schedule untouched.
    assert arm_1.train.weight_decay == NO_REGULARIZATION_WEIGHT_DECAY
    assert arm_1.fog is not None
    assert arm_1.fog.rate_distribution.min == 0.0
    assert arm_1.fog.rate_distribution.max == NO_REGULARIZATION_FOG_MAX
    assert arm_1.diffusion.schedule.t_one_fraction == baseline.diffusion.schedule.t_one_fraction

    # Arm 2: t=1 oversampled, regularization untouched.
    assert arm_2.diffusion.schedule.t_one_fraction == OVERSAMPLED_T_ONE_FRACTION
    assert arm_2.train.weight_decay == baseline.train.weight_decay
    assert arm_2.fog is not None
    assert arm_2.fog.rate_distribution.max == baseline.fog.rate_distribution.max

    # Arm 3: both, and byte-for-byte the same values the other two used.
    assert arm_3.train.weight_decay == arm_1.train.weight_decay
    assert arm_3.fog is not None
    assert arm_3.fog.rate_distribution.name == arm_1.fog.rate_distribution.name
    assert arm_3.fog.rate_distribution.min == arm_1.fog.rate_distribution.min
    assert arm_3.fog.rate_distribution.max == arm_1.fog.rate_distribution.max
    assert arm_3.diffusion.schedule.t_one_fraction == arm_2.diffusion.schedule.t_one_fraction

    # The baseline itself must still be the un-modified condition, or "off" and
    # "on" stop being distinguishable.
    assert baseline.train.weight_decay > 0.0
    assert baseline.fog is not None
    assert baseline.fog.rate_distribution.max > 0.0
    assert baseline.diffusion.schedule.t_one_fraction == 0.0


def test_memorization_sweep_driver_step_constants_match_the_configured_run_length() -> None:
    """Pin the memorization driver's skip/resume arithmetic to the actual config.

    Same contract as the ablation sweep driver: tests/run_memorization_sweep.sh
    needs STEPS_PER_EPOCH x EXPECTED_EPOCHS to recognize a finished arm from its
    checkpoint's `global_step`, those are shell constants that cannot read the
    YAML, and a change to `train.epochs` would desync them -- the driver would
    either re-run finished arms or skip unfinished ones.
    """

    root = DEFAULT_CONFIG.parents[1]
    script = (root / "tests" / "run_memorization_sweep.sh").read_text(encoding="utf-8")
    declared = {
        key: int(value)
        for key, value in (
            line.split("=", 1)
            for line in script.splitlines()
            if line.startswith(("STEPS_PER_EPOCH=", "EXPECTED_EPOCHS="))
        )
    }
    assert declared["EXPECTED_EPOCHS"] == load_config(
        root / "configs" / "local_overfit_v2.yaml"
    ).train.epochs
    # 334 train windows at batch_size 10 -> 34 optimizer steps per epoch. See the
    # replay-selection comment in configs/local_overfit.yaml.
    assert declared["STEPS_PER_EPOCH"] == 34

    # Every arm the driver lists must exist and be one of the three profiles, so
    # a renamed config cannot leave the driver pointing at nothing. The driver
    # fails per-arm rather than up front, so a typo would otherwise surface as a
    # crashed arm hours into an overnight sweep.
    for name in MEMORIZATION_CONFIGS:
        assert f"configs/{name}" in script, name

    # The driver must not reintroduce a step cap: the arms are meant to run their
    # LR decay and EMA window to completion, and a --max-steps run additionally
    # skips export_finished_model(). Only the train_pipeline invocation itself is
    # inspected -- the header comments mention the flag they no longer pass.
    lines = script.splitlines()
    launch_start = next(
        index for index, line in enumerate(lines) if "thesis_ml.pipeline.train_pipeline" in line
    )
    launch = [lines[launch_start]]
    while launch[-1].rstrip().endswith("\\"):
        launch.append(lines[launch_start + len(launch)])
    assert not any("--max-steps" in line for line in launch), "\n".join(launch)


def test_lr_schedule_wsd_decay_and_ema_horizon_are_range_checked(tmp_path: Path) -> None:
    """An unusable schedule value must fail at load, not mid-run."""

    # The whole default config is dumped standalone (no `extends`), so the probe
    # file needs no particular location relative to the repo.
    base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def load_with(**train_overrides: object) -> None:
        raw = {**base, "train": {**base["train"], **train_overrides}}
        path = tmp_path / "schedule_probe.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        load_config(path)

    with pytest.raises(ConfigError, match="lr_schedule"):
        load_with(lr_schedule="exponential")
    with pytest.raises(ConfigError, match="ema_horizon_ratio"):
        load_with(ema_horizon_ratio=0.0)
    with pytest.raises(ConfigError, match="ema_horizon_ratio"):
        load_with(ema_horizon_ratio=1.5)
    with pytest.raises(ConfigError, match="lr_floor_ratio"):
        load_with(lr_floor_ratio=-0.1)
    with pytest.raises(ConfigError, match="train.warmup"):
        load_with(warmup=0)
    with pytest.raises(ConfigError, match="lr_decay_ratio"):
        load_with(lr_decay_ratio=1.1)
    for schedule in ("linear", "cosine", "wsd"):
        load_with(lr_schedule=schedule)


def test_toggle_fingerprint_is_empty_at_baseline_and_sorted_when_enabled() -> None:
    """`toggle_fingerprint` must be empty at baseline and deterministic otherwise.

    The empty-string case is the one that matters most: the model stamps its
    identity as `ARCHITECTURE_ID + toggle_fingerprint(...)`, so any non-empty
    return with every toggle off would change the stored identity and break
    every matching vocabulary-v2 baseline checkpoint.
    """

    # The all-off ModelConfig is now constructed EXPLICITLY rather than read
    # straight off config/default.yaml, because the default no longer is all-off:
    # `frozen_input_kv` was promoted to true on 2026-08-09. The empty-string
    # contract is about the fingerprint FUNCTION, not about what the default
    # happens to be, and it must keep holding so the pre-promotion vocabulary-v2
    # checkpoints (and configs/ablation_00_baseline.yaml, which pins all three
    # false to reproduce them) still resolve to the unsuffixed identity.
    baseline = replace(load_config(DEFAULT_CONFIG).model, frozen_input_kv=False)
    assert toggle_fingerprint(baseline) == ""

    # And the promotion must NOT have rebased the fingerprint: a default-derived
    # model has to keep stamping `+frozen_input_kv`, so a pre-promotion all-off
    # checkpoint fails closed on load instead of silently entering a two-pass
    # model. This is the assertion that would catch someone "simplifying" the
    # suffix away now that the toggle is on by default.
    assert toggle_fingerprint(load_config(DEFAULT_CONFIG).model) == "+frozen_input_kv"

    # Single-toggle cases: each name appears verbatim, preceded by exactly one
    # '+'. These strings are persisted in checkpoint identities, so they are
    # asserted literally rather than rebuilt from the field names.
    assert toggle_fingerprint(replace(baseline, frozen_input_kv=True)) == "+frozen_input_kv"
    assert toggle_fingerprint(replace(baseline, segment_embeddings=True)) == "+segment_embeddings"
    assert (
        toggle_fingerprint(replace(baseline, per_segment_positions=True))
        == "+per_segment_positions"
    )

    # A pair, to pin the ALPHABETICAL ordering rather than declaration order:
    # per_segment_positions is declared after frozen_input_kv but sorts before
    # segment_embeddings.
    assert (
        toggle_fingerprint(replace(baseline, frozen_input_kv=True, per_segment_positions=True))
        == "+frozen_input_kv+per_segment_positions"
    )

    # All on.
    assert (
        toggle_fingerprint(
            replace(
                baseline,
                frozen_input_kv=True,
                segment_embeddings=True,
                per_segment_positions=True,
            )
        )
        == "+frozen_input_kv+per_segment_positions+segment_embeddings"
    )

    # The concatenation itself is asserted against the real ARCHITECTURE_ID in
    # the model tests; importing it here would pull torch into a pure-config
    # test module, so this file only pins the empty-suffix half of the contract.
    assert "baseline" + toggle_fingerprint(baseline) == "baseline"


def test_fog_and_class_weights_are_required_in_both_modes(
    tmp_path: Path,
) -> None:
    """Fog and class weighting are shared by pretraining and fine-tuning."""

    base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert base["data"]["debut_mode"] is False

    missing_fog = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    missing_fog.pop("fog")
    fog_path = tmp_path / "pretrain_missing_fog.yaml"
    fog_path.write_text(yaml.safe_dump(missing_fog), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.fog is required"):
        load_config(fog_path)

    missing_weights = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    missing_weights["loss"].pop("class_loss_weights")
    weights_path = tmp_path / "pretrain_missing_weights.yaml"
    weights_path.write_text(yaml.safe_dump(missing_weights), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"class_loss_weights is required"):
        load_config(weights_path)

    # Debut mode also requires fog plus its class weights.
    debut_missing_both = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    debut_missing_both["data"]["debut_mode"] = True
    debut_missing_both.pop("fog")
    debut_path = tmp_path / "debut_missing_fog.yaml"
    debut_path.write_text(yaml.safe_dump(debut_missing_both), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.fog is required"):
        load_config(debut_path)

    debut_missing_weights = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    debut_missing_weights["data"]["debut_mode"] = True
    debut_missing_weights["loss"].pop("class_loss_weights")
    debut_missing_weights["fog"] = {
        "rate_distribution": {"name": "uniform", "min": 0.0, "max": 0.8, "power": 1.0}
    }
    debut_weights_path = tmp_path / "debut_missing_weights.yaml"
    debut_weights_path.write_text(yaml.safe_dump(debut_missing_weights), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"class_loss_weights is required",
    ):
        load_config(debut_weights_path)


def test_t_one_fraction_out_of_range_is_rejected(tmp_path: Path) -> None:
    """`diffusion.schedule.t_one_fraction` must be a probability in [0, 1]."""

    for bad_value in (-0.1, 1.5):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        raw["diffusion"]["schedule"]["t_one_fraction"] = bad_value
        config_path = tmp_path / f"bad_fraction_{bad_value}.yaml"
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(
            ConfigError,
            match=r"diffusion\.schedule\.t_one_fraction must be in \[0, 1\]",
        ):
            load_config(config_path)


@pytest.mark.parametrize("process", ["uniform", "absorbing"])
def test_supported_diffusion_processes_load(tmp_path: Path, process: str) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["diffusion"]["process"] = process
    config_path = tmp_path / f"{process}.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(config_path).diffusion.process == process


def test_unknown_diffusion_process_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["diffusion"]["process"] = "masked"
    config_path = tmp_path / "bad_process.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="uniform.*absorbing"):
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
