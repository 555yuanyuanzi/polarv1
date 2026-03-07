from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml


@dataclass
class ExperimentConfig:
    name: str = "polarformer_v1"
    output_root: str = "v1/outputs"
    seed: int = 42


@dataclass
class DataConfig:
    root_dir: str = ""
    train_crop_size: int = 256
    batch_size: int = 4
    val_batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    inp_channels: int = 3
    out_channels: int = 3
    dim: int = 48
    enc_blocks: list[int] = field(default_factory=lambda: [3, 4, 6])
    bottleneck_base_blocks: int = 3
    dec3_base_blocks: int = 2
    dec2_base_blocks: int = 2
    dec1_base_blocks: int = 3
    polar_window: int = 8
    n_theta: int = 16
    n_r: int = 8
    polar_proj_dim: int = 32
    router_hidden: int = 32
    router_topk: int = 2
    restormer_ffn_expansion: float = 2.0


@dataclass
class LossConfig:
    charbonnier_weight: float = 1.0
    use_frequency_loss: bool = False
    frequency_weight: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    epochs: int = 300
    grad_clip: float = 1.0
    ema_decay: float = 0.999


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    eta_min: float = 1e-7


@dataclass
class RuntimeConfig:
    amp: bool = True
    distributed: bool = True
    resume: str = ""
    val_interval: int = 10
    save_interval: int = 50


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "polarformer-v1"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    log_router_stats: bool = True
    save_latest_every_epoch: bool = True


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _convert_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if origin is list:
        item_type = get_args(field_type)[0]
        return [_convert_value(item_type, item) for item in value]
    if origin is tuple:
        item_types = get_args(field_type)
        return tuple(_convert_value(tp, item) for tp, item in zip(item_types, value))
    if is_dataclass(field_type):
        return _from_mapping(field_type, value)
    return value


def _from_mapping(cls: type[T], mapping: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in mapping:
            continue
        kwargs[item.name] = _convert_value(item.type, mapping[item.name])
    return cls(**kwargs)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _from_mapping(AppConfig, raw)
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if not config.data.root_dir:
        raise ValueError("`data.root_dir` must be set in the YAML config.")
    if config.model.dim != 48:
        raise ValueError("`model.dim` must stay fixed at 48 for V1.")
    if config.model.enc_blocks != [3, 4, 6]:
        raise ValueError("`model.enc_blocks` must stay fixed at [3, 4, 6] for V1.")
    if config.model.bottleneck_base_blocks != 3:
        raise ValueError("`model.bottleneck_base_blocks` must stay fixed at 3 for V1.")
    if config.model.dec3_base_blocks != 2 or config.model.dec2_base_blocks != 2 or config.model.dec1_base_blocks != 3:
        raise ValueError("Decoder/base block counts must stay fixed at [2, 2, 3] for V1.")
    if config.model.router_topk != 2:
        raise ValueError("`model.router_topk` must stay fixed at 2 for V1.")
    if config.model.polar_window != 8:
        raise ValueError("`model.polar_window` must stay fixed at 8 for V1.")
    if config.model.n_theta != 16 or config.model.n_r != 8 or config.model.polar_proj_dim != 32:
        raise ValueError("Local polar hyperparameters must stay fixed at n_theta=16, n_r=8, polar_proj_dim=32.")
    if config.scheduler.type.lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported in V1.")


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return asdict(config)


def save_resolved_config(config: AppConfig, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
