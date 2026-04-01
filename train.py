from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config_to_dict, load_config, save_resolved_config
from src.data.gopro import GoProDataset
from src.engine.checkpoint import load_checkpoint
from src.engine.ema import ModelEMA
from src.engine.optim import build_optimizer, build_scheduler
from src.engine.trainer import Trainer
from src.models import PolarFormer
from src.utils.distributed import (
    DistributedEvalSampler,
    broadcast_object,
    cleanup_distributed,
    init_distributed_mode,
    is_main_process,
)
from src.utils.experiment import create_experiment_dirs, load_experiment_dirs
from src.utils.logging import JsonlWriter, create_logger, create_tensorboard_writer, create_wandb_run, register_wandb_files
from src.utils.seed import build_worker_init_fn, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid Decoder-Heavy PolarFormer V1.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def build_model(config) -> PolarFormer:
    return PolarFormer(
        inp_channels=config.model.inp_channels,
        out_channels=config.model.out_channels,
        dim=config.model.dim,
        enc_blocks=tuple(config.model.enc_blocks),
        bottleneck_base_blocks=config.model.bottleneck_base_blocks,
        dec3_base_blocks=config.model.dec3_base_blocks,
        dec2_base_blocks=config.model.dec2_base_blocks,
        dec1_base_blocks=config.model.dec1_base_blocks,
        restormer_ffn_expansion=config.model.restormer_ffn_expansion,
        naf_dw_expand=config.model.naf_dw_expand,
        naf_ffn_expand=config.model.naf_ffn_expand,
        fbeb_enabled=config.model.fbeb_enabled,
        fbeb_stages=tuple(config.model.fbeb_stages),
        local_refine_enabled=config.model.local_refine_enabled,
        local_refine_stages=tuple(config.model.local_refine_stages),
        importance_supervision_enabled=config.model.importance_supervision_enabled,
        fbeb_init_r1=config.model.fbeb_init_r1,
        fbeb_init_r2=config.model.fbeb_init_r2,
        fbeb_init_tau=config.model.fbeb_init_tau,
    )


def maybe_build_subset(dataset, *, subset_size: int, seed: int, split_name: str, logger):
    if subset_size <= 0:
        return dataset

    dataset_size = len(dataset)
    if subset_size > dataset_size:
        raise ValueError(
            f"`data.{split_name}_subset_size={subset_size}` exceeds {split_name} dataset size {dataset_size}."
        )
    if subset_size == dataset_size:
        logger.info(
            "Using full %s split because subset size equals dataset size: %d.",
            split_name,
            dataset_size,
        )
        return dataset

    indices = sorted(random.Random(seed).sample(range(dataset_size), subset_size))
    logger.info(
        "Using %d/%d samples from %s split with subset_seed=%d.",
        subset_size,
        dataset_size,
        split_name,
        seed,
    )
    return Subset(dataset, indices)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    state = init_distributed_mode(config.runtime.distributed)
    set_seed(config.experiment.seed)

    resume_path = Path(config.runtime.resume) if config.runtime.resume else None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        experiment_dirs = load_experiment_dirs(resume_path.resolve().parents[1])
    else:
        output_root = Path(config.experiment.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        if is_main_process(state):
            experiment_dirs = create_experiment_dirs(str(output_root), config.experiment.name)
            experiment_root = str(experiment_dirs.root)
        else:
            experiment_root = None
        experiment_root = broadcast_object(experiment_root, src=0)
        experiment_dirs = load_experiment_dirs(experiment_root)

    logger = create_logger(experiment_dirs.train_log, is_main=is_main_process(state))
    metrics_writer = JsonlWriter(experiment_dirs.metrics_jsonl)
    tensorboard_writer = create_tensorboard_writer(
        experiment_dirs.tensorboard,
        enabled=config.logging.tensorboard,
        is_main=is_main_process(state),
    )
    if is_main_process(state):
        save_resolved_config(config, experiment_dirs.resolved_config)

    logger.info("Experiment directory: %s", experiment_dirs.root)
    logger.info("Loading datasets from %s", config.data.root_dir)

    train_dataset = GoProDataset(
        root_dir=config.data.root_dir,
        split="train",
        crop_size=config.data.train_crop_size,
        random_rot90=config.data.random_rot90,
    )
    val_dataset = GoProDataset(
        root_dir=config.data.root_dir,
        split="test",
        crop_size=config.data.train_crop_size,
    )
    subset_seed = config.experiment.seed if config.data.subset_seed < 0 else config.data.subset_seed
    train_dataset = maybe_build_subset(
        train_dataset,
        subset_size=config.data.train_subset_size,
        seed=subset_seed,
        split_name="train",
        logger=logger,
    )
    val_dataset = maybe_build_subset(
        val_dataset,
        subset_size=config.data.val_subset_size,
        seed=subset_seed + 1,
        split_name="val",
        logger=logger,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=True,
            seed=config.experiment.seed,
            drop_last=True,
        )
        if state.distributed
        else None
    )
    val_sampler = DistributedEvalSampler(val_dataset, num_replicas=state.world_size, rank=state.rank) if state.distributed else None
    train_worker_init_fn = build_worker_init_fn(config.experiment.seed, rank=state.rank)
    val_worker_init_fn = build_worker_init_fn(config.experiment.seed + 100000, rank=state.rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
        worker_init_fn=train_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
        worker_init_fn=val_worker_init_fn,
    )

    model = build_model(config).to(state.device)
    if state.distributed:
        if state.device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[state.local_rank], output_device=state.local_rank)
        else:
            model = DistributedDataParallel(model)

    ema = ModelEMA(model, decay=config.optim.ema_decay).to(state.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config.runtime.amp and state.device.type == "cuda")

    start_data_pass = 0
    start_step_in_pass = 0
    global_step = 0
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    wandb_run_id = ""
    if resume_path is not None:
        logger.info("Resuming from %s", resume_path)
        resume_state = load_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=state.device,
        )
        start_data_pass = resume_state["data_pass"]
        start_step_in_pass = resume_state["step_in_pass"]
        global_step = resume_state["global_step"]
        best_psnr = resume_state["best_psnr"]
        best_ssim = resume_state["best_ssim"]
        wandb_run_id = resume_state["wandb_run_id"]

    if is_main_process(state):
        logger.info(
            "Distributed training: enabled=%s world_size=%d rank=%d local_rank=%d device=%s global_batch=%d",
            state.distributed,
            state.world_size,
            state.rank,
            state.local_rank,
            state.device,
            config.data.batch_size * state.world_size,
        )
        logger.info(
            "Scheduler: type=%s warmup_iterations=%d warmup_start_lr=%.6e eta_min=%.6e total_iterations=%d",
            config.scheduler.type,
            config.scheduler.warmup_iterations,
            config.scheduler.warmup_start_lr,
            config.scheduler.eta_min,
            config.optim.total_iterations,
        )

    wandb_run = create_wandb_run(
        enabled=config.logging.wandb,
        is_main=is_main_process(state),
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        mode=config.logging.wandb_mode,
        config=config_to_dict(config),
        run_name=f"{config.experiment.name}-{experiment_dirs.root.name}",
        run_dir=experiment_dirs.root,
        run_id=wandb_run_id,
    )
    if is_main_process(state) and wandb_run is not None:
        wandb_run_id = wandb_run.id
        wandb_run.config.update({"resolved_config_path": str(experiment_dirs.resolved_config)}, allow_val_change=True)
        wandb_run.config.update({"experiment": config.experiment.name}, allow_val_change=True)
        register_wandb_files(
            enabled=config.logging.wandb_upload_files,
            file_paths=[experiment_dirs.resolved_config],
            base_path=experiment_dirs.root,
            policy="now",
        )
        register_wandb_files(
            enabled=config.logging.wandb_upload_files,
            file_paths=[experiment_dirs.train_log, experiment_dirs.metrics_jsonl],
            base_path=experiment_dirs.root,
            policy="live",
        )

    trainer = Trainer(
        config=config,
        state=state,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        metrics_writer=metrics_writer,
        tensorboard_writer=tensorboard_writer,
        experiment_dirs=experiment_dirs,
        wandb_run=wandb_run,
        start_data_pass=start_data_pass,
        start_step_in_pass=start_step_in_pass,
        global_step=global_step,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
        wandb_run_id=wandb_run_id,
    )

    try:
        trainer.train()
    finally:
        if wandb_run is not None:
            register_wandb_files(
                enabled=config.logging.wandb_upload_files,
                file_paths=[experiment_dirs.train_log, experiment_dirs.metrics_jsonl, experiment_dirs.resolved_config],
                base_path=experiment_dirs.root,
                policy="now",
            )
            wandb_run.finish()
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
