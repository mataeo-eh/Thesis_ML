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
    rope_theta: float
    rope_scaling: RopeScalingConfig


@dataclass(frozen=True)
class MaskScheduleConfig:
    name: str
    t_distribution: str
    min: float
    max: float
    loss_reweight: str
    # Fraction of training examples that get OVERSAMPLED to t=1.0 exactly each
    # epoch, applied per-example as an independent Bernoulli draw (so this is
    # the fraction "in expectation", not an exact per-batch count). The
    # remaining (1 - t_one_fraction) of examples keep the existing uniform
    # draw over [min, max]. Must be a probability, so it is range-checked to
    # [0, 1] by `_validate_mask_schedule` after the config tree is built.
    t_one_fraction: float


@dataclass(frozen=True)
class DiffusionConfig:
    mask_schedule: MaskScheduleConfig


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
    preprocess_if_missing: bool
    rebuild_manifest: bool


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    adam_eps: float
    warmup: int
    lr_floor_ratio: float
    grad_clip: float
    accum: str
    accumulation_steps: int
    target_effective_batch_tokens: int
    max_steps: int
    epochs: int
    early_stopping_patience_epochs: int
    early_stopping_min_relative_improvement: float
    val_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    # When False (default), each periodic checkpoint overwrites a single
    # `last.pt` so disk/S3 usage stays flat over a multi-day run. When True,
    # every interval also keeps a timestamped `step-N.pt` snapshot.
    keep_step_checkpoints: bool
    ema_decay: float
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


@dataclass(frozen=True)
class SamplerConfig:
    max_steps: int
    temperature: TemperatureScheduleConfig
    entropy_bound: float
    confidence_threshold: float
    min_commit_per_step: int
    # Fine-tune sampler constraint (Worker 2). When True, the leading canvas
    # position (index 0, the [WIN]/[LOSS] outcome token) is denoised LAST: it may
    # only be committed after every other canvas position has been committed.
    # Default False reproduces the pre-training sampler exactly.
    outcome_last: bool


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
    # Fine-tune win/loss outcome class weight (Worker 3). Class id 6
    # (CLASS_WINLOSS). Only used when the debut taxonomy is active; harmless in
    # pre-training (that path never emits class-id-6 labels).
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
    # The fog paradigm is a FINE-TUNING-ONLY concern (pre-training drops fog
    # entirely). Optional here so a pre-training config can omit the `fog`
    # section entirely; `_validate_debut_mode_sections` enforces
    # presence/absence based on `data.debut_mode` after the config tree is
    # built.
    fog: FogConfig | None
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
    _validate_mask_schedule(config)
    _validate_debut_mode_sections(config)
    return config


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


def _validate_mask_schedule(config: ProjectConfig) -> None:
    """Range-check `diffusion.mask_schedule.t_one_fraction`.

    `_build_dataclass`/`_validate_value` only type-check scalar fields (int /
    float / str / bool); they do not range-check values. t_one_fraction is a
    per-example Bernoulli probability (the fraction of training examples that
    get oversampled to t=1.0 each epoch), so it must land in [0, 1]. This is a
    small, separate post-construction step -- not a general-purpose range-check
    system -- kept deliberately minimal to match the existing validation style.
    """

    fraction = config.diffusion.mask_schedule.t_one_fraction
    if not (0.0 <= fraction <= 1.0):
        raise ConfigError(
            f"diffusion.mask_schedule.t_one_fraction must be in [0, 1], got {fraction}"
        )


def _validate_debut_mode_sections(config: ProjectConfig) -> None:
    """Enforce that `fog` and `loss.class_loss_weights` are fine-tuning-only.

    Pre-training (`data.debut_mode=false`) drops the fog paradigm and
    per-class loss weighting entirely in favor of uniform published-style
    MDLM loss, so pre-training configs must not carry those sections (no dead
    knobs sitting in a config that nothing reads). Fine-tuning
    (`data.debut_mode=true`) is the debut build-order + win/loss pathway,
    which still needs both, so they are required there.

    This is a small, explicit, mode-conditional cross-field check -- there is
    no existing conditional validation machinery in this module to extend, so
    it is added as its own post-construction step run once from
    `load_config`, after `fog` and `class_loss_weights` have already been
    parsed as Optional (see `_optional_inner_type`).
    """

    debut_mode = config.data.debut_mode
    if debut_mode:
        if config.fog is None:
            raise ConfigError("config.fog is required when data.debut_mode=true")
        if config.loss.class_loss_weights is None:
            raise ConfigError(
                "config.loss.class_loss_weights is required when data.debut_mode=true"
            )
    else:
        if config.fog is not None:
            raise ConfigError("config.fog must not be set when data.debut_mode=false")
        if config.loss.class_loss_weights is not None:
            raise ConfigError(
                "config.loss.class_loss_weights must not be set when data.debut_mode=false"
            )
