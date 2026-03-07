from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from src.config import AppConfig
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluator import evaluate_model
from src.engine.ema import ModelEMA
from src.losses import CharbonnierLoss, FrequencyLoss
from src.utils.distributed import DistributedState, is_main_process, reduce_dict, unwrap_model


class Trainer:
    def __init__(
        self,
        config: AppConfig,
        state: DistributedState,
        model: torch.nn.Module,
        ema: ModelEMA,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        logger,
        metrics_writer,
        tensorboard_writer,
        experiment_dirs,
        start_epoch: int = 0,
        global_step: int = 0,
        best_psnr: float = float("-inf"),
        best_ssim: float = float("-inf"),
    ) -> None:
        self.config = config
        self.state = state
        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.metrics_writer = metrics_writer
        self.tensorboard_writer = tensorboard_writer
        self.experiment_dirs = experiment_dirs
        self.start_epoch = start_epoch
        self.global_step = global_step
        self.best_psnr = best_psnr
        self.best_ssim = best_ssim
        self.pixel_loss = CharbonnierLoss()
        self.frequency_loss = FrequencyLoss()

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config.optim.epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "lr": current_lr,
            }
            if self.config.logging.log_router_stats:
                record.update(train_metrics["router"])

            should_validate = (
                (epoch + 1) % self.config.runtime.val_interval == 0 or epoch + 1 == self.config.optim.epochs
            )
            if should_validate:
                raw_metrics = evaluate_model(
                    unwrap_model(self.model),
                    self.val_loader,
                    device=self.state.device,
                    amp=self.config.runtime.amp,
                )
                ema_metrics = evaluate_model(
                    self.ema.module,
                    self.val_loader,
                    device=self.state.device,
                    amp=self.config.runtime.amp,
                )
                record.update(
                    {
                        "raw_psnr": raw_metrics["psnr"],
                        "raw_ssim": raw_metrics["ssim"],
                        "ema_psnr": ema_metrics["psnr"],
                        "ema_ssim": ema_metrics["ssim"],
                    }
                )

                if ema_metrics["psnr"] > self.best_psnr:
                    self.best_psnr = ema_metrics["psnr"]
                    save_checkpoint(
                        self.experiment_dirs.checkpoints / "best_psnr.pth",
                        self.config,
                        self.model,
                        self.ema,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        epoch + 1,
                        self.global_step,
                        self.best_psnr,
                        self.best_ssim,
                        main_process=is_main_process(self.state),
                    )
                if ema_metrics["ssim"] > self.best_ssim:
                    self.best_ssim = ema_metrics["ssim"]
                    save_checkpoint(
                        self.experiment_dirs.checkpoints / "best_ssim.pth",
                        self.config,
                        self.model,
                        self.ema,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        epoch + 1,
                        self.global_step,
                        self.best_psnr,
                        self.best_ssim,
                        main_process=is_main_process(self.state),
                    )

            if self.config.logging.save_latest_every_epoch:
                save_checkpoint(
                    self.experiment_dirs.checkpoints / "latest.pth",
                    self.config,
                    self.model,
                    self.ema,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch + 1,
                    self.global_step,
                    self.best_psnr,
                    self.best_ssim,
                    main_process=is_main_process(self.state),
                )
            if (epoch + 1) % self.config.runtime.save_interval == 0:
                save_checkpoint(
                    self.experiment_dirs.checkpoints / f"epoch_{epoch + 1:04d}.pth",
                    self.config,
                    self.model,
                    self.ema,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch + 1,
                    self.global_step,
                    self.best_psnr,
                    self.best_ssim,
                    main_process=is_main_process(self.state),
                )

            self._log_epoch(record)

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        del epoch
        self.model.train()
        loss_sum = torch.zeros(1, device=self.state.device)
        sample_count = torch.zeros(1, device=self.state.device)
        router_sums = {
            "mean_confidence": torch.zeros(1, device=self.state.device),
            "top2_entropy": torch.zeros(1, device=self.state.device),
            "expert_usage": torch.zeros(4, device=self.state.device),
        }
        router_count = torch.zeros(1, device=self.state.device)

        for blur, sharp in self.train_loader:
            blur = blur.to(self.state.device, non_blocking=True)
            sharp = sharp.to(self.state.device, non_blocking=True)
            batch_size = blur.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.cuda.amp.autocast(enabled=self.config.runtime.amp)
                if self.state.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                output = self.model(blur)
                loss = self.config.loss.charbonnier_weight * self.pixel_loss(output, sharp)
                if self.config.loss.use_frequency_loss:
                    loss = loss + self.config.loss.frequency_weight * self.frequency_loss(output, sharp)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
                self.optimizer.step()

            self.ema.update(self.model)
            self.global_step += 1

            loss_sum += loss.detach() * batch_size
            sample_count += batch_size

            if self.config.logging.log_router_stats:
                stats = unwrap_model(self.model).get_last_router_stats()
                if stats:
                    router_sums["mean_confidence"] += stats["mean_confidence"] * batch_size
                    router_sums["top2_entropy"] += stats["top2_entropy"] * batch_size
                    router_sums["expert_usage"] += stats["expert_usage"] * batch_size
                    router_count += batch_size

        reduced = reduce_dict(
            {
                "loss_sum": loss_sum,
                "sample_count": sample_count,
                "router_mean_confidence_sum": router_sums["mean_confidence"],
                "router_top2_entropy_sum": router_sums["top2_entropy"],
                "router_expert_usage_sum": router_sums["expert_usage"],
                "router_count": router_count,
            },
            average=False,
        )

        total_samples = max(reduced["sample_count"].item(), 1.0)
        metrics: dict[str, Any] = {"loss": reduced["loss_sum"].item() / total_samples}
        if self.config.logging.log_router_stats and reduced["router_count"].item() > 0:
            router_total = reduced["router_count"].item()
            expert_usage = reduced["router_expert_usage_sum"] / router_total
            metrics["router"] = {
                "router/mean_confidence": reduced["router_mean_confidence_sum"].item() / router_total,
                "router/top2_entropy": reduced["router_top2_entropy_sum"].item() / router_total,
                "router/expert_usage_e1": expert_usage[0].item(),
                "router/expert_usage_e2": expert_usage[1].item(),
                "router/expert_usage_e3": expert_usage[2].item(),
                "router/expert_usage_e4": expert_usage[3].item(),
            }
        else:
            metrics["router"] = {}
        return metrics

    def _log_epoch(self, record: dict[str, Any]) -> None:
        if not is_main_process(self.state):
            return

        parts = [
            f"epoch={record['epoch']}",
            f"train_loss={record['train_loss']:.6f}",
            f"lr={record['lr']:.6e}",
        ]
        if "raw_psnr" in record:
            parts.append(f"raw_psnr={record['raw_psnr']:.4f}")
            parts.append(f"raw_ssim={record['raw_ssim']:.4f}")
            parts.append(f"ema_psnr={record['ema_psnr']:.4f}")
            parts.append(f"ema_ssim={record['ema_ssim']:.4f}")
        self.logger.info(" | ".join(parts))
        self.metrics_writer.write(record)

        if self.tensorboard_writer is None:
            return
        self.tensorboard_writer.add_scalar("train/loss", record["train_loss"], record["epoch"])
        self.tensorboard_writer.add_scalar("train/lr", record["lr"], record["epoch"])
        if "raw_psnr" in record:
            self.tensorboard_writer.add_scalar("val/raw_psnr", record["raw_psnr"], record["epoch"])
            self.tensorboard_writer.add_scalar("val/raw_ssim", record["raw_ssim"], record["epoch"])
            self.tensorboard_writer.add_scalar("val/ema_psnr", record["ema_psnr"], record["epoch"])
            self.tensorboard_writer.add_scalar("val/ema_ssim", record["ema_ssim"], record["epoch"])
        if self.config.logging.log_router_stats:
            if "router/mean_confidence" in record:
                self.tensorboard_writer.add_scalar(
                    "router/mean_confidence", record["router/mean_confidence"], record["epoch"]
                )
                self.tensorboard_writer.add_scalar("router/top2_entropy", record["router/top2_entropy"], record["epoch"])
                self.tensorboard_writer.add_scalar(
                    "router/expert_usage_e1", record["router/expert_usage_e1"], record["epoch"]
                )
                self.tensorboard_writer.add_scalar(
                    "router/expert_usage_e2", record["router/expert_usage_e2"], record["epoch"]
                )
                self.tensorboard_writer.add_scalar(
                    "router/expert_usage_e3", record["router/expert_usage_e3"], record["epoch"]
                )
                self.tensorboard_writer.add_scalar(
                    "router/expert_usage_e4", record["router/expert_usage_e4"], record["epoch"]
                )
        self.tensorboard_writer.flush()
