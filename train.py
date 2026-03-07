from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, save_resolved_config
from src.data.gopro import GoProDataset
from src.engine.checkpoint import load_checkpoint
from src.engine.ema import ModelEMA
from src.engine.optim import build_optimizer, build_scheduler
from src.engine.trainer import Trainer
from src.models import PolarFormer
from src.utils.distributed import cleanup_distributed, init_distributed_mode, is_main_process
from src.utils.experiment import create_experiment_dirs, load_experiment_dirs
from src.utils.logging import JsonlWriter, create_logger, create_tensorboard_writer
from src.utils.seed import set_seed


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
        polar_window=config.model.polar_window,
        n_theta=config.model.n_theta,
        n_r=config.model.n_r,
        polar_proj_dim=config.model.polar_proj_dim,
        router_hidden=config.model.router_hidden,
        router_topk=config.model.router_topk,
        restormer_ffn_expansion=config.model.restormer_ffn_expansion,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    state = init_distributed_mode(config.runtime.distributed)
    set_seed(config.experiment.seed + state.rank)

    resume_path = Path(config.runtime.resume) if config.runtime.resume else None
    if resume_path is not None:
        experiment_dirs = load_experiment_dirs(resume_path.resolve().parents[1])
    else:
        experiment_dirs = create_experiment_dirs(config.experiment.output_root, config.experiment.name)

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
    )
    val_dataset = GoProDataset(
        root_dir=config.data.root_dir,
        split="test",
        crop_size=config.data.train_crop_size,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if state.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if state.distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )

    model = build_model(config).to(state.device)
    ema = ModelEMA(model, decay=config.optim.ema_decay).to(state.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config.runtime.amp and state.device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_psnr = float("-inf")
    best_ssim = float("-inf")
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
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        best_psnr = resume_state["best_psnr"]
        best_ssim = resume_state["best_ssim"]

    if state.distributed:
        if state.device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[state.local_rank], output_device=state.local_rank)
        else:
            model = DistributedDataParallel(model)

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
        start_epoch=start_epoch,
        global_step=global_step,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
    )

    try:
        trainer.train()
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
