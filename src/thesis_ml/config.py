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
    input_window_timesteps: int
    canvas_budget_tokens: int
    within_type_tiebreak: str


@dataclass(frozen=True)
class FogConfig:
    rate_distribution: UniformDistributionConfig


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    layers: int
    heads: int
    ffn: int


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
    examples_per_replay: int
    replay_glob: str
    token_dictionary_uri: str
    perspectives: str


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
    val_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    ema_decay: float
    confidence_loss_weight: float
    precision: str


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

    return _build_dataclass(ProjectConfig, raw, "config")


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
