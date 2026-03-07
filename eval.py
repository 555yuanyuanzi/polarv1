from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.gopro import GoProDataset
from src.engine.checkpoint import load_checkpoint
from src.engine.evaluator import evaluate_model
from src.engine.ema import ModelEMA
from src.engine.optim import build_optimizer, build_scheduler
from src.models import PolarFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Decoder-Heavy PolarFormer V1.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--use-ema", action="store_true", help="Evaluate EMA weights.")
    parser.add_argument("--use-raw", action="store_true", help="Evaluate raw model weights.")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = GoProDataset(root_dir=config.data.root_dir, split="test", crop_size=config.data.train_crop_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )

    model = build_model(config).to(device)
    ema = ModelEMA(model, decay=config.optim.ema_decay).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config.runtime.amp and device.type == "cuda")

    load_checkpoint(
        checkpoint_path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
    )

    use_raw = args.use_raw
    use_ema = args.use_ema or not use_raw
    results: dict[str, float] = {}

    if use_raw:
        raw_metrics = evaluate_model(model, dataloader, device=device, amp=config.runtime.amp)
        results["raw_psnr"] = raw_metrics["psnr"]
        results["raw_ssim"] = raw_metrics["ssim"]

    if use_ema:
        ema_metrics = evaluate_model(ema.module, dataloader, device=device, amp=config.runtime.amp)
        results["ema_psnr"] = ema_metrics["psnr"]
        results["ema_ssim"] = ema_metrics["ssim"]

    print(json.dumps(results, ensure_ascii=True, indent=2))

    output_path = checkpoint_path.parent / f"eval_{checkpoint_path.stem}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
