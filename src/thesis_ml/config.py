"""Typed project configuration loaded from YAML."""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

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


@dataclass(frozen=True)
class EvalConfig:
    heldout_split: str
    timing_tolerance_buckets: int
    fog_rate: float


@dataclass(frozen=True)
class ClassLossWeightsConfig:
    enemy_observed_reconstruction: float
    enemy_fogged_reconstruction: float
    enemy_future_prediction: float
    delimiter: float
    end: float
    pad: float


@dataclass(frozen=True)
class LossConfig:
    use_fused_cross_entropy: bool
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

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")

    extends = raw.pop("extends", None)
    if extends is not None:
        if not isinstance(extends, str):
            raise ConfigError("config.extends must be str")
        base_path = (config_path.parent / extends).resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base_raw = yaml.safe_load(handle)
        if not isinstance(base_raw, dict):
            raise ConfigError("extended config must be a mapping")
        raw = _deep_merge(base_raw, raw)

    return _build_dataclass(ProjectConfig, raw, "config")


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
        if field.name not in raw:
            raise ConfigError(f"{field_path} is required")
        value = raw[field.name]
        expected_type = hints[field.name]
        values[field.name] = _validate_value(expected_type, value, field_path)

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
