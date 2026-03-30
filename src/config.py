from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

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
    train_subset_size: int = 0
    val_subset_size: int = 0
    subset_seed: int = -1


@dataclass
class ModelConfig:
    inp_channels: int = 3
    out_channels: int = 3
    dim: int = 48
    enc_blocks: list[int] = field(default_factory=lambda: [3, 4, 6])
    bottleneck_base_blocks: int = 6
    dec3_base_blocks: int = 3
    dec2_base_blocks: int = 3
    dec1_base_blocks: int = 4
    restormer_ffn_expansion: float = 2.0
    naf_dw_expand: int = 2
    naf_ffn_expand: int = 2
    fbeb_enabled: bool = False
    fbeb_stages: list[str] = field(default_factory=list)
    local_refine_enabled: bool = True
    local_refine_stages: list[str] = field(default_factory=lambda: ["decoder3", "decoder2"])
    importance_supervision_enabled: bool = False
    fbeb_init_r1: float = 0.22
    fbeb_init_r2: float = 0.58
    fbeb_init_tau: float = 0.05


@dataclass
class LossConfig:
    charbonnier_weight: float = 1.0
    use_frequency_loss: bool = False
    frequency_weight: float = 0.1
    importance_supervision_weight: float = 0.05
    importance_aux_weight: float = 0.02
    importance_prior_weight: float = 0.25


@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    total_iterations: int = 600000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    epochs: int = 0


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    eta_min: float = 1e-6
    warmup_iterations: int = 0
    warmup_start_lr: float = 1e-6


@dataclass
class RuntimeConfig:
    amp: bool = True
    distributed: bool = True
    num_gpu: int = 1
    gpu_ids: list[int] = field(default_factory=list)
    resume: str = ""
    val_interval_steps: int = 10000
    save_interval_steps: int = 50000
    max_steps: int = 0
    val_interval: int = 0
    save_interval: int = 0


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "polarformer-v1"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_upload_files: bool = True
    wandb_upload_checkpoints: bool = True
    log_fbeb_stats: bool = True
    log_importance_stats: bool = True
    log_visual_maps: bool = True
    log_interval_steps: int = 50
    visual_log_interval_steps: int = 200
    save_latest_interval_steps: int = 5000
    save_latest_every_epoch: bool = False


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
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in mapping:
            continue
        field_type = type_hints.get(item.name, item.type)
        kwargs[item.name] = _convert_value(field_type, mapping[item.name])
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
    if config.data.train_subset_size < 0 or config.data.val_subset_size < 0:
        raise ValueError("`data.train_subset_size` and `data.val_subset_size` must be >= 0.")
    if config.data.subset_seed < -1:
        raise ValueError("`data.subset_seed` must be >= -1.")
    if config.model.dim <= 0:
        raise ValueError("`model.dim` must be positive.")
    if len(config.model.enc_blocks) != 3:
        raise ValueError("`model.enc_blocks` must contain exactly 3 stage depths.")
    if any(blocks <= 0 for blocks in config.model.enc_blocks):
        raise ValueError("`model.enc_blocks` must contain positive integers.")
    if (
        config.model.bottleneck_base_blocks <= 0
        or config.model.dec3_base_blocks <= 0
        or config.model.dec2_base_blocks <= 0
        or config.model.dec1_base_blocks <= 0
    ):
        raise ValueError("All decoder/bottleneck block counts must be positive.")
    if config.model.naf_dw_expand < 1 or config.model.naf_ffn_expand < 1:
        raise ValueError("`model.naf_dw_expand` and `model.naf_ffn_expand` must be positive integers.")
    if config.model.restormer_ffn_expansion <= 0:
        raise ValueError("`model.restormer_ffn_expansion` must be positive.")
    if config.model.fbeb_stages:
        invalid_stages = sorted(set(config.model.fbeb_stages) - {"bottleneck", "decoder3", "decoder2"})
        if invalid_stages:
            raise ValueError(f"`model.fbeb_stages` contains invalid stages: {invalid_stages}")
    if config.model.fbeb_enabled and not config.model.fbeb_stages:
        raise ValueError("`model.fbeb_enabled=true` requires at least one stage in `model.fbeb_stages`.")
    if config.model.local_refine_stages:
        invalid_stages = sorted(set(config.model.local_refine_stages) - {"decoder3", "decoder2"})
        if invalid_stages:
            raise ValueError(f"`model.local_refine_stages` contains invalid stages: {invalid_stages}")
    if config.model.local_refine_enabled and not config.model.local_refine_stages:
        raise ValueError("`model.local_refine_enabled=true` requires at least one stage in `model.local_refine_stages`.")
    if config.model.importance_supervision_enabled and not config.model.local_refine_enabled:
        raise ValueError("`model.importance_supervision_enabled=true` requires `model.local_refine_enabled=true`.")
    if not (0.0 <= config.loss.importance_prior_weight <= 1.0):
        raise ValueError("`loss.importance_prior_weight` must be in [0, 1].")
    if config.loss.importance_supervision_weight < 0.0 or config.loss.importance_aux_weight < 0.0:
        raise ValueError("`loss.importance_supervision_weight` and `loss.importance_aux_weight` must be >= 0.")
    if config.optim.total_iterations <= 0:
        raise ValueError("`optim.total_iterations` must be positive.")
    if config.runtime.num_gpu <= 0:
        raise ValueError("`runtime.num_gpu` must be positive.")
    if config.runtime.gpu_ids:
        raise ValueError("`runtime.gpu_ids` is deprecated. Use `runtime.num_gpu` and set `CUDA_VISIBLE_DEVICES` at launch.")
    if config.runtime.distributed and config.runtime.num_gpu < 2:
        raise ValueError("`runtime.distributed=true` requires `runtime.num_gpu >= 2`.")
    if not config.runtime.distributed and config.runtime.num_gpu != 1:
        raise ValueError("Single-process training requires `runtime.num_gpu = 1`.")
    if config.optim.epochs > 0:
        raise ValueError("`optim.epochs` is deprecated. Use `optim.total_iterations` instead.")
    if config.runtime.max_steps > 0:
        raise ValueError("`runtime.max_steps` is deprecated. Use `optim.total_iterations` instead.")
    if config.runtime.val_interval > 0:
        raise ValueError("`runtime.val_interval` is deprecated. Use `runtime.val_interval_steps` instead.")
    if config.runtime.save_interval > 0:
        raise ValueError("`runtime.save_interval` is deprecated. Use `runtime.save_interval_steps` instead.")
    if config.runtime.val_interval_steps < 0 or config.runtime.save_interval_steps < 0:
        raise ValueError("`runtime.val_interval_steps` and `runtime.save_interval_steps` must be >= 0.")
    if config.scheduler.type.lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported in V1.")
    if config.scheduler.eta_min < 0.0:
        raise ValueError("`scheduler.eta_min` must be >= 0.")
    if config.scheduler.eta_min > config.optim.lr:
        raise ValueError("`scheduler.eta_min` must be <= `optim.lr`.")
    if config.scheduler.warmup_iterations < 0:
        raise ValueError("`scheduler.warmup_iterations` must be >= 0.")
    if config.scheduler.warmup_iterations >= config.optim.total_iterations:
        raise ValueError("`scheduler.warmup_iterations` must be smaller than `optim.total_iterations`.")
    if config.scheduler.warmup_start_lr < 0.0:
        raise ValueError("`scheduler.warmup_start_lr` must be >= 0.")
    if config.scheduler.warmup_start_lr > config.optim.lr:
        raise ValueError("`scheduler.warmup_start_lr` must be <= `optim.lr`.")
    if config.logging.save_latest_every_epoch:
        raise ValueError("`logging.save_latest_every_epoch` is deprecated. Use `logging.save_latest_interval_steps`.")
    if config.logging.save_latest_interval_steps < 0:
        raise ValueError("`logging.save_latest_interval_steps` must be >= 0.")
    if config.logging.log_interval_steps <= 0:
        raise ValueError("`logging.log_interval_steps` must be positive.")
    if config.logging.visual_log_interval_steps <= 0:
        raise ValueError("`logging.visual_log_interval_steps` must be positive.")


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return asdict(config)


def save_resolved_config(config: AppConfig, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
