from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from src.config import AppConfig
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluator import evaluate_model
from src.engine.ema import ModelEMA
from src.losses import CharbonnierLoss, FrequencyLoss
from src.utils.distributed import DistributedState, is_main_process, reduce_dict, unwrap_model
from src.utils.logging import log_wandb_checkpoint_artifact


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
        wandb_run,
        start_epoch: int = 0,
        global_step: int = 0,
        best_psnr: float = float("-inf"),
        best_ssim: float = float("-inf"),
        wandb_run_id: str = "",
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
        self.wandb_run = wandb_run
        self.start_epoch = start_epoch
        self.global_step = global_step
        self.best_psnr = best_psnr
        self.best_ssim = best_ssim
        self.wandb_run_id = wandb_run_id
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
            reached_max_steps = self.config.runtime.max_steps > 0 and self.global_step >= self.config.runtime.max_steps

            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "lr": current_lr,
            }
            if self.config.logging.log_fbeb_stats:
                record.update(train_metrics["fbeb"])
            if self.config.logging.log_importance_stats:
                record.update(train_metrics["importance"])

            should_validate = (
                (epoch + 1) % self.config.runtime.val_interval == 0 or epoch + 1 == self.config.optim.epochs
            )
            if reached_max_steps:
                should_validate = True
            if should_validate:
                if is_main_process(self.state):
                    self.logger.info(
                        "Starting validation for epoch=%d at global_step=%d.",
                        epoch + 1,
                        self.global_step,
                    )
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
                    best_psnr_path = self.experiment_dirs.checkpoints / "best_psnr.pth"
                    save_checkpoint(
                        best_psnr_path,
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
                        wandb_run_id=self.wandb_run_id,
                    )
                    self._log_checkpoint_artifact(
                        checkpoint_path=best_psnr_path,
                        checkpoint_kind="best_psnr",
                        epoch=epoch + 1,
                        aliases=["best_psnr"],
                    )
                if ema_metrics["ssim"] > self.best_ssim:
                    self.best_ssim = ema_metrics["ssim"]
                    best_ssim_path = self.experiment_dirs.checkpoints / "best_ssim.pth"
                    save_checkpoint(
                        best_ssim_path,
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
                        wandb_run_id=self.wandb_run_id,
                    )
                    self._log_checkpoint_artifact(
                        checkpoint_path=best_ssim_path,
                        checkpoint_kind="best_ssim",
                        epoch=epoch + 1,
                        aliases=["best_ssim"],
                    )

            if self.config.logging.save_latest_every_epoch:
                latest_path = self.experiment_dirs.checkpoints / "latest.pth"
                save_checkpoint(
                    latest_path,
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
                    wandb_run_id=self.wandb_run_id,
                )
                self._log_checkpoint_artifact(
                    checkpoint_path=latest_path,
                    checkpoint_kind="latest",
                    epoch=epoch + 1,
                    aliases=["latest", f"epoch_{epoch + 1:04d}"],
                )
            if (epoch + 1) % self.config.runtime.save_interval == 0:
                epoch_path = self.experiment_dirs.checkpoints / f"epoch_{epoch + 1:04d}.pth"
                save_checkpoint(
                    epoch_path,
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
                    wandb_run_id=self.wandb_run_id,
                )
                self._log_checkpoint_artifact(
                    checkpoint_path=epoch_path,
                    checkpoint_kind="epoch",
                    epoch=epoch + 1,
                    aliases=[f"epoch_{epoch + 1:04d}"],
                )

            self._log_epoch(record)
            if reached_max_steps:
                if is_main_process(self.state):
                    self.logger.info("Reached runtime.max_steps=%d, stopping training.", self.config.runtime.max_steps)
                break

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        del epoch
        self.model.train()
        loss_sum = torch.zeros(1, device=self.state.device)
        sample_count = torch.zeros(1, device=self.state.device)
        fbeb_sums = {
            "fbeb/r1": torch.zeros(1, device=self.state.device),
            "fbeb/r2": torch.zeros(1, device=self.state.device),
            "fbeb/tau": torch.zeros(1, device=self.state.device),
            "fbeb/low_energy": torch.zeros(1, device=self.state.device),
            "fbeb/mid_energy": torch.zeros(1, device=self.state.device),
            "fbeb/high_energy": torch.zeros(1, device=self.state.device),
        }
        fbeb_count = torch.zeros(1, device=self.state.device)
        importance_sums = {
            "importance/mean": torch.zeros(1, device=self.state.device),
            "importance/std": torch.zeros(1, device=self.state.device),
            "importance/high_ratio": torch.zeros(1, device=self.state.device),
            "importance/raw_gate_mean": torch.zeros(1, device=self.state.device),
        }
        importance_count = torch.zeros(1, device=self.state.device)

        for step_idx, (blur, sharp) in enumerate(self.train_loader, start=1):
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

            model_unwrapped = unwrap_model(self.model)
            if self.config.logging.log_fbeb_stats:
                stats = model_unwrapped.get_last_fbeb_stats()
                if stats:
                    for key in fbeb_sums:
                        fbeb_sums[key] += stats[key] * batch_size
                    fbeb_count += batch_size
            else:
                stats = None
            if self.config.logging.log_importance_stats:
                importance_stats = model_unwrapped.get_last_importance_stats()
                if importance_stats:
                    for key in importance_sums:
                        importance_sums[key] += importance_stats[key] * batch_size
                    importance_count += batch_size
            else:
                importance_stats = None

            should_log_step = (
                is_main_process(self.state)
                and self.config.logging.log_interval_steps > 0
                and (
                    self.global_step % self.config.logging.log_interval_steps == 0
                    or (
                        self.config.runtime.max_steps > 0
                        and self.global_step >= self.config.runtime.max_steps
                    )
                )
            )
            if should_log_step:
                self._log_train_step(
                    step=step_idx,
                    batch_loss=loss.detach().item(),
                    stats=stats,
                    importance_stats=importance_stats,
                )

            should_log_visuals = (
                is_main_process(self.state)
                and self.tensorboard_writer is not None
                and self.config.logging.log_visual_maps
                and self.config.logging.visual_log_interval_steps > 0
                and self.global_step % self.config.logging.visual_log_interval_steps == 0
            )
            if should_log_visuals:
                visual_getter = getattr(model_unwrapped, "get_last_visuals", None)
                if visual_getter is not None:
                    self._log_visual_maps(visual_getter())

            if self.config.runtime.max_steps > 0 and self.global_step >= self.config.runtime.max_steps:
                break

        reduced = reduce_dict(
            {
                "loss_sum": loss_sum,
                "sample_count": sample_count,
                "fbeb_r1_sum": fbeb_sums["fbeb/r1"],
                "fbeb_r2_sum": fbeb_sums["fbeb/r2"],
                "fbeb_tau_sum": fbeb_sums["fbeb/tau"],
                "fbeb_low_energy_sum": fbeb_sums["fbeb/low_energy"],
                "fbeb_mid_energy_sum": fbeb_sums["fbeb/mid_energy"],
                "fbeb_high_energy_sum": fbeb_sums["fbeb/high_energy"],
                "fbeb_count": fbeb_count,
                "importance_mean_sum": importance_sums["importance/mean"],
                "importance_std_sum": importance_sums["importance/std"],
                "importance_high_ratio_sum": importance_sums["importance/high_ratio"],
                "importance_raw_gate_mean_sum": importance_sums["importance/raw_gate_mean"],
                "importance_count": importance_count,
            },
            average=False,
        )

        total_samples = max(reduced["sample_count"].item(), 1.0)
        metrics: dict[str, Any] = {"loss": reduced["loss_sum"].item() / total_samples}
        if self.config.logging.log_fbeb_stats and reduced["fbeb_count"].item() > 0:
            fbeb_total = reduced["fbeb_count"].item()
            metrics["fbeb"] = {
                "fbeb/r1": reduced["fbeb_r1_sum"].item() / fbeb_total,
                "fbeb/r2": reduced["fbeb_r2_sum"].item() / fbeb_total,
                "fbeb/tau": reduced["fbeb_tau_sum"].item() / fbeb_total,
                "fbeb/low_energy": reduced["fbeb_low_energy_sum"].item() / fbeb_total,
                "fbeb/mid_energy": reduced["fbeb_mid_energy_sum"].item() / fbeb_total,
                "fbeb/high_energy": reduced["fbeb_high_energy_sum"].item() / fbeb_total,
            }
        else:
            metrics["fbeb"] = {}
        if self.config.logging.log_importance_stats and reduced["importance_count"].item() > 0:
            importance_total = reduced["importance_count"].item()
            metrics["importance"] = {
                "importance/mean": reduced["importance_mean_sum"].item() / importance_total,
                "importance/std": reduced["importance_std_sum"].item() / importance_total,
                "importance/high_ratio": reduced["importance_high_ratio_sum"].item() / importance_total,
                "importance/raw_gate_mean": reduced["importance_raw_gate_mean_sum"].item() / importance_total,
            }
        else:
            metrics["importance"] = {}
        return metrics

    def _log_train_step(
        self,
        step: int,
        batch_loss: float,
        stats: dict[str, torch.Tensor] | None,
        importance_stats: dict[str, torch.Tensor] | None,
    ) -> None:
        if not is_main_process(self.state):
            return

        record: dict[str, Any] = {
            "type": "train_step",
            "global_step": self.global_step,
            "step_in_epoch": step,
            "train_loss": batch_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        parts = [
            f"global_step={self.global_step}",
            f"step_in_epoch={step}",
            f"train_loss={batch_loss:.6f}",
            f"lr={record['lr']:.6e}",
        ]
        if self.config.logging.log_fbeb_stats and stats:
            r1 = float(stats["fbeb/r1"].item())
            r2 = float(stats["fbeb/r2"].item())
            tau = float(stats["fbeb/tau"].item())
            low_energy = float(stats["fbeb/low_energy"].item())
            mid_energy = float(stats["fbeb/mid_energy"].item())
            high_energy = float(stats["fbeb/high_energy"].item())
            record.update(
                {
                    "fbeb/r1": r1,
                    "fbeb/r2": r2,
                    "fbeb/tau": tau,
                    "fbeb/low_energy": low_energy,
                    "fbeb/mid_energy": mid_energy,
                    "fbeb/high_energy": high_energy,
                }
            )
            parts.append(f"fbeb_r1={r1:.4f}")
            parts.append(f"fbeb_r2={r2:.4f}")
            parts.append(f"fbeb_tau={tau:.4f}")
        if self.config.logging.log_importance_stats and importance_stats:
            imp_mean = float(importance_stats["importance/mean"].item())
            imp_high = float(importance_stats["importance/high_ratio"].item())
            raw_gate_mean = float(importance_stats["importance/raw_gate_mean"].item())
            record.update(
                {
                    "importance/mean": imp_mean,
                    "importance/high_ratio": imp_high,
                    "importance/raw_gate_mean": raw_gate_mean,
                }
            )
            parts.append(f"imp_mean={imp_mean:.4f}")
            parts.append(f"imp_high={imp_high:.4f}")

        self.logger.info(" | ".join(parts))
        self.metrics_writer.write(record)

        if self.tensorboard_writer is None:
            return
        self.tensorboard_writer.add_scalar("train/step_loss", batch_loss, self.global_step)
        self.tensorboard_writer.add_scalar("train/lr_step", record["lr"], self.global_step)
        if self.config.logging.log_fbeb_stats and stats:
            self.tensorboard_writer.add_scalar("fbeb/r1_step", record["fbeb/r1"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/r2_step", record["fbeb/r2"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/tau_step", record["fbeb/tau"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/low_energy_step", record["fbeb/low_energy"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/mid_energy_step", record["fbeb/mid_energy"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/high_energy_step", record["fbeb/high_energy"], self.global_step)
        if self.config.logging.log_importance_stats and importance_stats:
            self.tensorboard_writer.add_scalar("importance/mean_step", record["importance/mean"], self.global_step)
            self.tensorboard_writer.add_scalar(
                "importance/high_ratio_step",
                record["importance/high_ratio"],
                self.global_step,
            )
            self.tensorboard_writer.add_scalar(
                "importance/raw_gate_mean_step",
                record["importance/raw_gate_mean"],
                self.global_step,
            )
        self.tensorboard_writer.flush()

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
        if self.config.logging.log_fbeb_stats and "fbeb/r1" in record:
            self.tensorboard_writer.add_scalar("fbeb/r1", record["fbeb/r1"], record["epoch"])
            self.tensorboard_writer.add_scalar("fbeb/r2", record["fbeb/r2"], record["epoch"])
            self.tensorboard_writer.add_scalar("fbeb/tau", record["fbeb/tau"], record["epoch"])
            self.tensorboard_writer.add_scalar("fbeb/low_energy", record["fbeb/low_energy"], record["epoch"])
            self.tensorboard_writer.add_scalar("fbeb/mid_energy", record["fbeb/mid_energy"], record["epoch"])
            self.tensorboard_writer.add_scalar("fbeb/high_energy", record["fbeb/high_energy"], record["epoch"])
        if self.config.logging.log_importance_stats and "importance/mean" in record:
            self.tensorboard_writer.add_scalar("importance/mean", record["importance/mean"], record["epoch"])
            self.tensorboard_writer.add_scalar("importance/std", record["importance/std"], record["epoch"])
            self.tensorboard_writer.add_scalar(
                "importance/high_ratio",
                record["importance/high_ratio"],
                record["epoch"],
            )
            self.tensorboard_writer.add_scalar(
                "importance/raw_gate_mean",
                record["importance/raw_gate_mean"],
                record["epoch"],
            )
        self.tensorboard_writer.flush()

    def _normalize_visual_map(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected visual map with 2/3/4 dims, got {tuple(tensor.shape)}.")
        min_val = tensor.min()
        max_val = tensor.max()
        if float((max_val - min_val).item()) < 1e-8:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

    def _log_visual_maps(self, visuals: dict[str, torch.Tensor]) -> None:
        if not visuals or self.tensorboard_writer is None:
            return
        for name, tensor in visuals.items():
            self.tensorboard_writer.add_image(f"visuals/{name}", self._normalize_visual_map(tensor), self.global_step)
        self.tensorboard_writer.flush()

    def _log_checkpoint_artifact(
        self,
        *,
        checkpoint_path,
        checkpoint_kind: str,
        epoch: int,
        aliases: list[str],
    ) -> None:
        if not is_main_process(self.state):
            return
        if self.wandb_run is None or not self.config.logging.wandb_upload_checkpoints:
            return

        log_wandb_checkpoint_artifact(
            enabled=True,
            checkpoint_path=checkpoint_path,
            artifact_name=f"run-{self.wandb_run_id or self.wandb_run.id}-{checkpoint_kind}",
            aliases=aliases,
            metadata={
                "checkpoint_kind": checkpoint_kind,
                "epoch": epoch,
                "global_step": self.global_step,
                "best_psnr": self.best_psnr,
                "best_ssim": self.best_ssim,
                "run_id": self.wandb_run_id or self.wandb_run.id,
            },
            file_name=checkpoint_path.name,
        )
