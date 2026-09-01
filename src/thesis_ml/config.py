"""Typed project configuration loaded from YAML."""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when configuration input does not match the dataclass contract."""


@dataclass(frozen=True)
class RateDistributionConfig:
    name: str
    min: float
    max: float
    # Shape parameter for the configurable high-end-skewed ``power`` law.
    # ``power=2`` is Beta(2, 1): x = U ** (1 / power). Uniform sampling ignores
    # this field, but it remains explicit so every merged YAML has one schema.
    power: float = 1.0


# Backward-compatible import name for direct test/utility construction. The
# runtime type is no longer uniform-only, but external callers from before the
# configurable power law should not break merely because the schema widened.
UniformDistributionConfig = RateDistributionConfig


@dataclass(frozen=True)
class DataConfig:
    sampling_interval_s: int
    input_budget_tokens: int
    canvas_budget_tokens: int
    canvas_recon_fraction: float
    within_type_tiebreak: str
    tokenized_replay_dir: str
    window_manifest_path: str
    feature_statistics_path: str
    # When True, the artifact target builder produces a "debut build-order +
    # win/loss outcome" canvas (fine-tuning mode) instead of the default
    # full-reconstruction target. When False (default), the pretraining target
    # path is used and behaves exactly as before. This flag ONLY switches the
    # target builder; it does not change the input manifest or any budget.
    debut_mode: bool


@dataclass(frozen=True)
class FogConfig:
    rate_distribution: RateDistributionConfig


@dataclass(frozen=True)
class RopeScalingConfig:
    rope_type: str
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    layers: int
    heads: int
    ffn: int
    qk_norm: bool
    self_conditioning: bool
    gradient_checkpointing: bool
    # --- Architecture toggles -------------------------------------------------
    # Three independent switches over how the input region and the canvas region
    # relate to each other inside the backbone. `segment_embeddings` and
    # `per_segment_positions` are still ablation experiments and default to
    # false; `frozen_input_kv` was PROMOTED to default true on 2026-08-09
    # (SPEC.md 14b). `toggle_fingerprint` was deliberately not rebased by that
    # promotion -- it still returns the empty string only for the ALL-off case,
    # so the unsuffixed architecture ID keeps meaning exactly what it always
    # meant and pre-promotion checkpoints fail closed against the new default.
    # The vocabulary-v2 baseline itself intentionally retires v1 checkpoints.
    #
    # Split the single joint bidirectional forward into two passes: the input
    # region alone through all L blocks (attending only to itself, caching each
    # layer's K/V), then the canvas region attending to
    # `concat(cached_input_K, canvas_K)`. At inference the input KV is computed
    # once and reused across every denoising step instead of being recomputed.
    frozen_input_kv: bool
    # Add a learned `nn.Embedding(2, d_model)` (0 = input segment, 1 = canvas
    # segment) to the final per-region embedding, so the same token id appearing
    # in the input and in the canvas becomes genuinely different to the model.
    segment_embeddings: bool
    # Compute RoPE position ids PER SEGMENT: the input's real content gets
    # 0..L_i-1 in its left-padded slots and the canvas restarts at 0 at canvas
    # index 0. Canvas positions then stop shifting with however much left
    # padding a particular batch happened to need.
    per_segment_positions: bool
    rope_theta: float
    rope_scaling: RopeScalingConfig


@dataclass(frozen=True)
class DiffusionScheduleConfig:
    name: str
    t_distribution: str
    min: float
    max: float
    # Fraction of training examples that get OVERSAMPLED to t=1.0 exactly each
    # epoch, applied per-example as an independent Bernoulli draw (so this is
    # the fraction "in expectation", not an exact per-batch count). The
    # remaining (1 - t_one_fraction) of examples keep the configured continuous
    # draw over [min, max]. Must be a probability, so it is range-checked to
    # [0, 1] by `_validate_diffusion` after the config tree is built.
    t_one_fraction: float
    t_distribution_power: float = 1.0


@dataclass(frozen=True)
class DiffusionConfig:
    process: str
    schedule: DiffusionScheduleConfig


@dataclass(frozen=True)
class StorageConfig:
    data_uri: str
    raw_uri: str
    checkpoint_uri: str
    log_uri: str
    local_cache_dir: str


@dataclass(frozen=True)
class DataSourceConfig:
    source: str
    kaggle_dataset: str
    kaggle_username_env: str
    kaggle_key_env: str
    extractor_path: str
    extractor_command: str
    workers: int


@dataclass(frozen=True)
class PipelineConfig:
    auto_acquire: bool
    smoke: bool
    smoke_steps: int
    seed: int
    batch_size: int
    replay_glob: str
    token_dictionary_uri: str
    perspectives: str
    # DataLoader throughput knobs for the real (non-smoke) training pipeline.
    # num_workers spawns that many background loader processes; prefetch_factor
    # is how many batches each worker pre-loads ahead of the GPU. Both keep the
    # GPU from starving while CPU-bound serialization/parsing happens off-thread.
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    # Fraction of total host RAM the per-process replay-frame cache may use,
    # shared (divided) across DataLoader workers so the aggregate stays bounded.
    # Prevents loading a hundreds-of-GB dataset fully into memory.
    cache_ram_fraction: float
    # Reproducible train/dev/test split over REPLAYS (not windows, to avoid
    # leakage). split_seed is independent of the training seed so re-seeding a
    # run does not reshuffle which replays are held out.
    split_seed: int
    test_fraction: float
    dev_fraction: float
    # When positive, partition the complete corpus by exact replay counts:
    # this many train replays, validation_replay_count dev replays, and every
    # remaining replay in test. Zero preserves fraction-based splitting.
    train_replay_count: int
    replay_subset_size: int
    validation_replay_count: int
    # Explicit, named replay selection. Comma-separated replay FILE STEMS (e.g.
    # "match_4746821_game_state,match_4746300_game_state"), NOT the hashed
    # tokenized-artifact ids. When train_replay_ids is non-empty it REPLACES the
    # seeded shuffle in split_replays/_select_replays entirely: the named
    # replays become train, dev_replay_ids become dev, and every other replay in
    # the corpus becomes test.
    #
    # WHY this exists: the seeded split answers "give me N arbitrary replays",
    # which is right for a real run but wrong for a diagnostic overfit run where
    # the subset must be a KNOWN, reproducible, deliberately-chosen sample (for
    # example replays whose token counts sit at the corpus median, so the run is
    # neither trivially short nor unrepresentatively long). Naming the replays
    # also makes the subset survive changes to seeds, corpus size, or split
    # fractions, none of which should silently re-roll an overfit subset.
    #
    # Both default to "" (empty), which preserves the seeded-split behavior
    # exactly for every profile that does not opt in.
    train_replay_ids: str
    dev_replay_ids: str
    preprocess_if_missing: bool
    rebuild_manifest: bool
    # Explicit preprocessing switch. False makes every normal consumer load
    # the frozen artifact and fail if it is absent or incompatible.
    prepare_feature_statistics: bool


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    adam_eps: float
    warmup: int
    lr_floor_ratio: float
    # Complete scheduler choice. Every schedule uses the same fixed optimizer-
    # step warmup. ``cosine`` and ``linear`` decay immediately afterwards;
    # ``wsd`` holds the peak until the final configured decay fraction.
    lr_schedule: str
    lr_decay_ratio: float
    grad_clip: float
    accum: str
    accumulation_steps: int
    target_effective_batch_tokens: int
    max_steps: int
    epochs: int
    # Optional optimizer-step horizon for LR/EMA schedules, expressed in
    # epochs of the current dataloader. Zero preserves the ordinary contract:
    # schedules are fitted to the run's actual max_steps/epochs. A positive
    # value deliberately decouples the schedule horizon from the stop epoch,
    # which is useful for short learning-curve ablations that must reproduce
    # the opening of a longer run without pretending the short run completed
    # the schedule.
    schedule_horizon_epochs: int
    early_stopping_patience_epochs: int
    early_stopping_min_relative_improvement: float
    val_interval: int
    # Whether the ten-per-epoch interval reports each run a full dev pass.
    #
    # True (default): every interval row carries dev values alongside its train
    # values, giving ten dev observations per epoch. This is what makes the
    # diagnostics usable on a corpus large enough to converge in ONE epoch,
    # where epoch-end validation would yield a single dev point and no trend.
    #
    # False: interval rows carry train values only and their dev columns stay
    # blank; dev is evaluated once per epoch and lands in the epoch CSV. Choose
    # this when the dev pass costs more than the training it is interleaved with
    # -- on a small memorization run a full dev pass can take longer than the
    # ~10% slice of training that preceded it, so ten of them per epoch can more
    # than double wall-clock time for signal that a per-epoch dev point already
    # provides on a run measured in tens of epochs.
    #
    # Independent of `val_interval`, which drives the separate step-cadence
    # validation written into the per-step JSONL.
    interval_dev_evaluation: bool
    # Whether the ten-per-epoch interval reports carry TRAIN values.
    #
    # The exact counterpart of `interval_dev_evaluation` above, for the other
    # half of each interval row. True (default): the train-side breakdown is
    # reported ten times per epoch, which is what keeps the diagnostics readable
    # on a corpus large enough to converge in one epoch.
    #
    # False: the interval rows' train columns stay blank and train loss is
    # reported once per epoch in `epoch_metrics.csv`, exactly as dev already is
    # under `interval_dev_evaluation: false`. Choose this on a run measured in
    # many epochs, where a ~10%-of-epoch slice spans only a handful of batches:
    # those rows are then noise around a number the per-epoch row already
    # reports, and the epoch series is long enough to show the trend on its own.
    #
    # Unlike the dev knob this saves no compute -- the train values are already
    # accumulated by the training pass -- so it is purely about signal.
    #
    # When BOTH this and `interval_dev_evaluation` are false, no interval row is
    # written at all rather than an all-blank one. The accumulation wiring stays
    # live either way, so re-enabling either side needs no other change.
    interval_train_evaluation: bool
    checkpoint_interval: int
    checkpoint_dir: str
    resume_checkpoint_subdir: str
    best_checkpoint_subdir: str
    durable_checkpoint_subdir: str
    save_best_checkpoint: bool
    durable_checkpoint_interval_epochs: int
    # When False (default), each periodic checkpoint overwrites a single
    # `last.pt` so disk/S3 usage stays flat over a multi-day run. When True,
    # every interval also keeps a timestamped `step-N.pt` snapshot.
    keep_step_checkpoints: bool
    # CEILING on the EMA decay, not the decay actually used. See
    # `ema_horizon_ratio` directly below and `TrainingLoop._resolve_ema_decay`:
    # the effective decay is derived from the run's own step count and then
    # clamped to at most this value, so a very long run still tops out at a
    # sane averaging window instead of growing one without bound.
    ema_decay: float
    # The EMA averaging window, expressed as a FRACTION of the run's total
    # optimizer steps. This is what makes the EMA schedule fit the run.
    #
    # An EMA with decay d has an effective window of 1/(1-d) steps: the fixed
    # 0.9999 that used to be applied unconditionally averages over ~10,000
    # steps, so a 3,400-step run finished with an EMA that had only traversed a
    # third of its own window and was still dragging early-training weights into
    # the served weights. Deriving the decay as `1 - 1/(ratio * total_steps)`
    # instead makes the window scale with the run: at 0.1 the EMA always averages
    # over the final ~10% of training, whatever the epoch budget is, and always
    # completes several window lengths before the last step.
    ema_horizon_ratio: float
    confidence_loss_weight: float
    self_cond_prob: float
    precision: str
    require_cuda: bool
    max_cuda_reserved_gb: float
    empty_cuda_cache_after_epoch: bool
    # Weights-only warm-start source for fine-tuning (Worker 5). When this is
    # a non-empty path, the fine-tune pipeline loads ONLY the model weights
    # from this checkpoint (see `TrainingLoop.load_model_weights`) before
    # training begins, rather than doing a full optimizer/step/epoch resume.
    # Empty string "" (the default) disables warm-start entirely, which
    # keeps the pre-training path (`train_pipeline.py`) fully unaffected.
    init_from_checkpoint: str


@dataclass(frozen=True)
class TemperatureScheduleConfig:
    start: float
    end: float
    exponent: float


@dataclass(frozen=True)
class SamplerConfig:
    max_steps: int
    temperature: TemperatureScheduleConfig
    entropy_bound: float
    adaptive_stop: bool
    entropy_threshold: float


@dataclass(frozen=True)
class EvalConfig:
    heldout_split: str
    timing_tolerance_buckets: int
    fog_rate: float
    # Fine-tune debut evaluation buckets (Worker 4). Config validation only
    # supports scalar field types (int/float/str/bool), so these list-shaped
    # bucket definitions are stored as comma-separated strings and parsed by the
    # eval code. `debut_minute_buckets` = cumulative win/loss accuracy checkpoints
    # in minutes; `debut_fog_bucket_edges` = the two fog-rate boundaries that
    # split examples into <low / mid / >high fogged buckets.
    debut_minute_buckets: str
    debut_fog_bucket_edges: str
    # Fine-tune debut evaluation SIZE CAP (Worker 4 / Worker 5). A windowed
    # dataset expands each replay into many input-tiled examples whose output
    # horizons can overlap. The debut evaluator samples the diffusion model once per
    # example, so scoring every window would sample thousands of times and hold
    # every materialized example in host RAM. This caps how many examples each
    # report section ("memorized"/"test") scores. The pipeline picks them by
    # EVEN STRIDING across the dataset so the sample still spans early->late
    # input reach (needed for the 1/3/5/7/10-minute win/loss buckets). 0 = score
    # every window (only sensible for very small datasets).
    debut_max_examples: int


@dataclass(frozen=True)
class ClassLossWeightsConfig:
    enemy_observed_reconstruction: float
    enemy_fogged_reconstruction: float
    enemy_future_prediction: float
    delimiter: float
    end: float
    pad: float
    # Win/loss outcome class weight (class id 6) in both training modes.
    win_loss: float


@dataclass(frozen=True)
class LossConfig:
    use_fused_cross_entropy: bool
    # Shared pretraining/fine-tuning class weights.
    class_loss_weights: ClassLossWeightsConfig


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    fog: FogConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    storage: StorageConfig
    data_source: DataSourceConfig
    pipeline: PipelineConfig
    train: TrainConfig
    sampler: SamplerConfig
    eval: EvalConfig
    loss: LossConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML config file."""

    raw = _load_config_mapping(Path(path).resolve(), stack=())
    config = _build_dataclass(ProjectConfig, raw, "config")
    _validate_train(config)
    _validate_diffusion(config)
    _validate_sampler(config)
    _validate_shared_training_sections(config)
    return config


def toggle_fingerprint(model_config: ModelConfig) -> str:
    """Build the architecture-identity suffix naming the enabled ablation toggles.

    Downstream model code stamps its identity as
    `ARCHITECTURE_ID + toggle_fingerprint(model_config)` so a checkpoint trained
    with an ablation enabled can never be silently loaded into a model built
    without it. The all-off case is load-bearing: it must return the EMPTY
    string so the identity stays the unsuffixed current `ARCHITECTURE_ID`.
    Compatibility across baseline revisions is owned by `ARCHITECTURE_ID`
    itself; vocabulary-v1 checkpoints are intentionally retired.

    Parameters:
        model_config: the built `ModelConfig` whose three ablation toggles
            (`frozen_input_kv`, `segment_embeddings`, `per_segment_positions`)
            are read.

    Returns:
        `""` when all three toggles are false. Otherwise a deterministic,
        alphabetically sorted suffix of the enabled toggles' config field names,
        each preceded by `+` -- for example
        `"+frozen_input_kv+per_segment_positions"`.

    Calls: nothing. It only reads attributes off the passed dataclass.
    """

    # Each toggle is appended under its own explicit `if` rather than looped
    # over via getattr: these strings are persisted inside checkpoint identities,
    # so the exact name that goes into the fingerprint is spelled out literally
    # here and cannot drift with a refactor of the dataclass.
    enabled: list[str] = []
    if model_config.frozen_input_kv:
        enabled.append("frozen_input_kv")
    if model_config.segment_embeddings:
        enabled.append("segment_embeddings")
    if model_config.per_segment_positions:
        enabled.append("per_segment_positions")

    if not enabled:
        # Explicit early return for the all-off baseline. The join below would
        # yield "" anyway, but matching vocabulary-v2 baseline checkpoints
        # depend on this exact unsuffixed identity.
        return ""

    # Sorted so the fingerprint depends only on WHICH toggles are on, never on
    # the order they happen to be declared or checked in.
    enabled.sort()
    return "".join(f"+{name}" for name in enabled)


def _load_config_mapping(
    config_path: Path,
    *,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise ConfigError(f"config.extends cycle: {cycle}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")

    extends = raw.pop("extends", None)
    if extends is not None:
        if not isinstance(extends, str):
            raise ConfigError("config.extends must be str")
        base_path = (config_path.parent / extends).resolve()
        base_raw = _load_config_mapping(base_path, stack=(*stack, config_path))
        raw = _deep_merge(base_raw, raw)

    return raw


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_dataclass(cls: type[T], raw: Any, path: str) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")

    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise ConfigError(f"{path} has unknown key: {unknown[0]}")

    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field in fields(cls):
        field_path = f"{path}.{field.name}"
        expected_type = hints[field.name]
        if field.name not in raw:
            raise ConfigError(f"{field_path} is required")
        values[field.name] = _validate_value(expected_type, raw[field.name], field_path)

    return cls(**values)


# Plain dataclasses plus manual validation keeps this early config contract stable.
def _validate_value(expected_type: type[Any], value: Any, path: str) -> Any:
    if is_dataclass(expected_type):
        return _build_dataclass(expected_type, value, path)

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be int")
        return value

    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be float")
        return float(value)

    if expected_type is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be str")
        return value

    if expected_type is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be bool")
        return value

    raise TypeError(f"unsupported config field type at {path}: {expected_type!r}")


#: Complete learning-rate schedules `TrainingLoop._lr_multiplier` implements.
LR_SCHEDULES = ("cosine", "linear", "wsd")


def _validate_train(config: ProjectConfig) -> None:
    """Range-check the learning-rate schedule, accumulation, and run controls.

    These settings are read once per run by `TrainingLoop` and have no meaningful
    fallback, so invalid values must stop the launch instead of quietly changing
    what the run trains at.

    Parameters:
        config: the fully built `ProjectConfig` to check.

    Returns:
        None. Raises `ConfigError` on the first invalid value.

    Calls: nothing. It only reads fields off `config.train`.
    """

    train = config.train
    if train.lr_schedule not in LR_SCHEDULES:
        raise ConfigError(
            "train.lr_schedule must be one of "
            f"{list(LR_SCHEDULES)}, got {train.lr_schedule!r}"
        )
    if not (0.0 <= train.lr_floor_ratio <= 1.0):
        raise ConfigError(
            f"train.lr_floor_ratio must be in [0, 1], got {train.lr_floor_ratio}"
        )
    if not (0.0 < train.ema_horizon_ratio <= 1.0):
        raise ConfigError(
            "train.ema_horizon_ratio must be in (0, 1] -- it is the EMA averaging "
            f"window as a fraction of the run's steps, got {train.ema_horizon_ratio}"
        )
    if not (0.0 <= train.ema_decay <= 1.0):
        raise ConfigError(f"train.ema_decay must be in [0, 1], got {train.ema_decay}")
    if train.warmup < 1:
        raise ConfigError("train.warmup must be >= 1")
    if not (0.0 <= train.lr_decay_ratio <= 1.0):
        raise ConfigError(
            "train.lr_decay_ratio must be in [0, 1], got "
            f"{train.lr_decay_ratio}"
        )
    if train.accumulation_steps < 1:
        raise ConfigError("train.accumulation_steps must be >= 1")
    if train.target_effective_batch_tokens < 0:
        raise ConfigError("train.target_effective_batch_tokens must be >= 0")
    if train.schedule_horizon_epochs < 0:
        raise ConfigError("train.schedule_horizon_epochs must be >= 0")
    if train.early_stopping_patience_epochs < 0:
        raise ConfigError("train.early_stopping_patience_epochs must be >= 0")
    if not (0.0 <= train.early_stopping_min_relative_improvement < 1.0):
        raise ConfigError(
            "train.early_stopping_min_relative_improvement must be in [0, 1)"
        )
    if train.durable_checkpoint_interval_epochs < 0:
        raise ConfigError("train.durable_checkpoint_interval_epochs must be >= 0")
    for field_name in (
        "resume_checkpoint_subdir",
        "best_checkpoint_subdir",
        "durable_checkpoint_subdir",
    ):
        value = getattr(train, field_name)
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigError(f"train.{field_name} must be a safe relative path")
    active_checkpoint_dirs = [train.resume_checkpoint_subdir.strip()]
    if train.save_best_checkpoint:
        if not train.best_checkpoint_subdir.strip():
            raise ConfigError("train.best_checkpoint_subdir is required when saving best checkpoints")
        active_checkpoint_dirs.append(train.best_checkpoint_subdir.strip())
    if train.durable_checkpoint_interval_epochs > 0:
        if not train.durable_checkpoint_subdir.strip():
            raise ConfigError(
                "train.durable_checkpoint_subdir is required when retaining durable checkpoints"
            )
        active_checkpoint_dirs.append(train.durable_checkpoint_subdir.strip())
    nonempty_checkpoint_dirs = [value for value in active_checkpoint_dirs if value]
    if len(set(nonempty_checkpoint_dirs)) != len(nonempty_checkpoint_dirs):
        raise ConfigError("active train checkpoint subdirectories must be distinct")


def _validate_diffusion(config: ProjectConfig) -> None:
    """Validate the process-neutral linear diffusion schedule."""

    if config.diffusion.process not in {"uniform", "absorbing"}:
        raise ConfigError(
            "diffusion.process must be exactly 'uniform' or 'absorbing', "
            f"got {config.diffusion.process!r}"
        )
    schedule = config.diffusion.schedule
    if schedule.name != "linear":
        raise ConfigError(f"diffusion.schedule.name must be 'linear', got {schedule.name!r}")
    if schedule.t_distribution not in {"uniform", "power"}:
        raise ConfigError(
            "diffusion.schedule.t_distribution must be 'uniform' or 'power', "
            f"got {schedule.t_distribution!r}"
        )
    if schedule.t_distribution_power <= 0.0:
        raise ConfigError("diffusion.schedule.t_distribution_power must be > 0")
    if not (0.0 <= schedule.min <= schedule.max <= 1.0):
        raise ConfigError(
            "diffusion.schedule min/max must satisfy 0 <= min <= max <= 1, "
            f"got min={schedule.min}, max={schedule.max}"
        )
    fraction = schedule.t_one_fraction
    if not (0.0 <= fraction <= 1.0):
        raise ConfigError(
            f"diffusion.schedule.t_one_fraction must be in [0, 1], got {fraction}"
        )


def _validate_sampler(config: ProjectConfig) -> None:
    """Range-check entropy-bounded sampler settings."""

    sampler = config.sampler
    if not (1 <= sampler.max_steps <= 64):
        raise ConfigError("sampler.max_steps must be in [1, 64]")
    if sampler.temperature.start <= 0 or sampler.temperature.end <= 0:
        raise ConfigError("sampler temperatures must be positive")
    if sampler.temperature.exponent <= 0:
        raise ConfigError("sampler.temperature.exponent must be positive")
    if sampler.entropy_bound < 0:
        raise ConfigError("sampler.entropy_bound must be non-negative")
    if sampler.entropy_threshold < 0:
        raise ConfigError("sampler.entropy_threshold must be non-negative")


def _validate_shared_training_sections(config: ProjectConfig) -> None:
    """Validate shared fog and class-weight distributions for both modes."""

    distribution = config.fog.rate_distribution
    if distribution.name not in {"uniform", "power"}:
        raise ConfigError("fog.rate_distribution.name must be 'uniform' or 'power'")
    if not (0.0 <= distribution.min <= distribution.max <= 1.0):
        raise ConfigError("fog.rate_distribution must satisfy 0 <= min <= max <= 1")
    if distribution.power <= 0.0:
        raise ConfigError("fog.rate_distribution.power must be > 0")
    weights = config.loss.class_loss_weights
    for field in fields(weights):
        if getattr(weights, field.name) < 0.0:
            raise ConfigError(f"loss.class_loss_weights.{field.name} must be >= 0")
