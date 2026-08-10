"""Typed project configuration loaded from YAML."""

import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when configuration input does not match the dataclass contract."""


@dataclass(frozen=True)
class UniformDistributionConfig:
    name: str
    min: float
    max: float


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
    rate_distribution: UniformDistributionConfig


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
    # remaining (1 - t_one_fraction) of examples keep the existing uniform
    # draw over [min, max]. Must be a probability, so it is range-checked to
    # [0, 1] by `_validate_diffusion` after the config tree is built.
    t_one_fraction: float


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
    # Shape of the post-warmup learning-rate decay, from the peak `lr` down to
    # `lr * lr_floor_ratio` over the run's derived step horizon (see
    # `TrainingLoop._lr_multiplier`). Both shapes start at the peak and land on
    # the floor at exactly the last step; they differ in HOW they get there.
    #
    #   "cosine" -- half-cosine. Lingers near the peak for the first ~15% of the
    #               decay, drops fastest through the middle (its steepest slope
    #               is pi/2 ~= 1.57x the straight-line slope), then flattens into
    #               a long tail at the floor.
    #   "linear" -- one constant slope. Leaves the peak immediately, so less of
    #               the run is spent at the highest rate, and that same constant
    #               slope is SHALLOWER through the middle of training than the
    #               cosine's steepest section. Pair it with a lower
    #               `lr_floor_ratio` to also end the run at a smaller rate; the
    #               combination is the intended remedy for a loss curve that
    #               hovers around a minimum instead of settling into it, because
    #               the end-of-run gradient-noise floor scales with the final lr.
    lr_decay_shape: str
    grad_clip: float
    accum: str
    accumulation_steps: int
    target_effective_batch_tokens: int
    max_steps: int
    epochs: int
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
    stability_steps: int


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
    # Fine-tune win/loss outcome class weight (class id 6). Pretraining also
    # emits class id 6 but uses its fixed uniform weighting instead.
    win_loss: float


@dataclass(frozen=True)
class LossConfig:
    use_fused_cross_entropy: bool
    # Per-class loss weighting is a FINE-TUNING-ONLY concern (pre-training uses
    # uniform published-style MDLM loss with no dead knobs). Optional here so a
    # pre-training config can omit the `class_loss_weights` section entirely;
    # `_validate_debut_mode_sections` enforces presence/absence based on
    # `data.debut_mode` after the config tree is built.
    class_loss_weights: ClassLossWeightsConfig | None


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
    _validate_debut_mode_sections(config)
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
        # Fine-tuning-only sections (e.g. `fog`, `loss.class_loss_weights`) are
        # typed `X | None` so a config file may omit the key entirely (missing
        # from `raw`) instead of the caller being forced to write an unused
        # placeholder. A field is only ALLOWED to be missing when its type
        # says so; every other field keeps the old strict "required" behavior.
        # Presence/absence relative to `data.debut_mode` is enforced
        # separately by `_validate_debut_mode_sections` after the whole
        # config tree is built (that check needs sibling fields, which are
        # not available yet at this per-field build step).
        is_optional, inner_type = _optional_inner_type(expected_type)
        if field.name not in raw:
            if is_optional:
                values[field.name] = None
                continue
            raise ConfigError(f"{field_path} is required")
        value = raw[field.name]
        if is_optional and value is None:
            values[field.name] = None
            continue
        values[field.name] = _validate_value(inner_type, value, field_path)

    return cls(**values)


def _optional_inner_type(expected_type: Any) -> tuple[bool, Any]:
    """Detect an `X | None` (i.e. Optional[X]) type hint.

    Dataclass field annotations written as `X | None` are resolved by
    `get_type_hints` to a `types.UnionType` (PEP 604 syntax), which is a
    DIFFERENT object from `typing.Union` (what `Optional[X]` resolves to).
    Both spellings are checked here so either would work; this project uses
    the `X | None` spelling.

    Returns (True, X) if expected_type is exactly a two-armed union with
    NoneType as one arm; otherwise returns (False, expected_type) unchanged so
    every other field type is validated exactly as before.
    """
    origin = get_origin(expected_type)
    if origin is Union or origin is types.UnionType:
        args = get_args(expected_type)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(args) == 2 and len(non_none) == 1:
            return True, non_none[0]
    return False, expected_type


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


#: The learning-rate decay shapes `TrainingLoop._lr_multiplier` knows how to
#: build. Kept here so a typo in YAML fails at config load with the list of
#: legal values, rather than silently falling through to a default shape
#: hundreds of optimizer steps into a run.
LR_DECAY_SHAPES = ("cosine", "linear")


def _validate_train(config: ProjectConfig) -> None:
    """Range-check the learning-rate decay shape and the EMA horizon knobs.

    Both settings are read once per run by `TrainingLoop` -- the decay shape by
    `_lr_multiplier` and the EMA horizon by `_resolve_ema_decay` -- and neither
    has a meaningful fallback, so an out-of-range value must stop the launch
    instead of quietly changing what the run trains at.

    Parameters:
        config: the fully built `ProjectConfig` to check.

    Returns:
        None. Raises `ConfigError` on the first invalid value.

    Calls: nothing. It only reads fields off `config.train`.
    """

    train = config.train
    if train.lr_decay_shape not in LR_DECAY_SHAPES:
        raise ConfigError(
            "train.lr_decay_shape must be one of "
            f"{list(LR_DECAY_SHAPES)}, got {train.lr_decay_shape!r}"
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
    if schedule.t_distribution != "uniform":
        raise ConfigError(
            "diffusion.schedule.t_distribution must be 'uniform', "
            f"got {schedule.t_distribution!r}"
        )
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
    if sampler.stability_steps < 1:
        raise ConfigError("sampler.stability_steps must be at least 1")


def _validate_debut_mode_sections(config: ProjectConfig) -> None:
    """Enforce mode-specific loss weights; fog is shared by both modes.

    Pre-training uses the shared fog distribution but keeps uniform loss
    weighting. Fine-tuning additionally requires per-class weights.

    This is a small, explicit, mode-conditional cross-field check -- there is
    no existing conditional validation machinery in this module to extend, so
    it is added as its own post-construction step run once from
    `load_config`, after `fog` and `class_loss_weights` have already been
    parsed as Optional (see `_optional_inner_type`).
    """

    debut_mode = config.data.debut_mode
    if debut_mode:
        if config.loss.class_loss_weights is None:
            raise ConfigError(
                "config.loss.class_loss_weights is required when data.debut_mode=true"
            )
    else:
        if config.loss.class_loss_weights is not None:
            raise ConfigError(
                "config.loss.class_loss_weights must not be set when data.debut_mode=false"
            )
