from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from src.config import AppConfig
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluator import evaluate_model
from src.engine.ema import ModelEMA
from src.losses import CharbonnierLoss, FrequencyLoss, ImportanceSupervisionLoss, PolarSpectralLoss
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
        start_data_pass: int = 0,
        start_step_in_pass: int = 0,
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
        self.data_pass = start_data_pass
        self.step_in_pass = start_step_in_pass
        self.global_step = global_step
        self.best_psnr = best_psnr
        self.best_ssim = best_ssim
        self.wandb_run_id = wandb_run_id
        self.pixel_loss = CharbonnierLoss()
        self.frequency_loss = FrequencyLoss()
        self.polar_spectral_loss = PolarSpectralLoss()
        self.importance_supervision_loss = ImportanceSupervisionLoss(
            prior_weight=config.loss.importance_prior_weight
        )

    def _compute_importance_supervision(
        self,
        model_unwrapped: torch.nn.Module,
        blur: torch.Tensor,
        sharp: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if not self.config.model.importance_supervision_enabled:
            return None
        if (
            self.config.loss.importance_supervision_weight <= 0.0
            and self.config.loss.importance_aux_weight <= 0.0
        ):
            return None
        getter = getattr(model_unwrapped, "get_last_importance_supervision", None)
        if getter is None:
            return None
        stage_items = getter()
        if not stage_items:
            return None

        map_loss = torch.zeros((), device=self.state.device)
        aux_loss = torch.zeros((), device=self.state.device)
        stage_count = 0
        for item in stage_items.values():
            importance_map = item["importance_map"]
            stage_prediction = item["stage_prediction"]
            stage_map_loss, _target, sharp_small = self.importance_supervision_loss(
                importance_map=importance_map,
                stage_prediction=stage_prediction,
                blur=blur,
                sharp=sharp,
            )
            map_loss = map_loss + stage_map_loss
            aux_loss = aux_loss + self.pixel_loss(stage_prediction, sharp_small.to(dtype=stage_prediction.dtype))
            stage_count += 1
        if stage_count == 0:
            return None
        map_loss = map_loss / stage_count
        aux_loss = aux_loss / stage_count
        total = (
            self.config.loss.importance_supervision_weight * map_loss
            + self.config.loss.importance_aux_weight * aux_loss
        )
        return {
            "total": total,
            "map_loss": map_loss,
            "aux_loss": aux_loss,
        }

    def train(self) -> None:
        total_iterations = self.config.optim.total_iterations
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")
        if self.global_step >= total_iterations:
            if is_main_process(self.state):
                self.logger.info(
                    "Checkpoint global_step=%d already reached optim.total_iterations=%d. Nothing to do.",
                    self.global_step,
                    total_iterations,
                )
            return

        self.model.train()
        accumulators = self._create_metric_accumulators()
        sampler = getattr(self.train_loader, "sampler", None)

        while self.global_step < total_iterations:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.data_pass)

            completed_pass = True
            for step_idx, (blur, sharp) in enumerate(self.train_loader, start=1):
                if step_idx <= self.step_in_pass:
                    continue

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
                    model_unwrapped = unwrap_model(self.model)
                    loss = self.config.loss.charbonnier_weight * self.pixel_loss(output, sharp)
                    if self.config.loss.use_frequency_loss:
                        loss = loss + self.config.loss.frequency_weight * self.frequency_loss(output, sharp)
                    if self.config.loss.use_polar_spectral_loss:
                        loss = loss + self.config.loss.polar_spectral_weight * self.polar_spectral_loss(output, sharp)
                    importance_supervision = self._compute_importance_supervision(model_unwrapped, blur, sharp)
                    if importance_supervision is not None:
                        loss = loss + importance_supervision["total"]
                    clear_importance_cache = getattr(model_unwrapped, "clear_last_importance_supervision", None)
                    if clear_importance_cache is not None:
                        clear_importance_cache()

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
                self.step_in_pass = step_idx
                self.scheduler.step()

                stats = model_unwrapped.get_last_fbeb_stats() if self.config.logging.log_fbeb_stats else None
                importance_stats = (
                    model_unwrapped.get_last_importance_stats() if self.config.logging.log_importance_stats else None
                )
                self._update_metric_accumulators(
                    accumulators,
                    batch_size=batch_size,
                    loss=loss,
                    stats=stats,
                    importance_stats=importance_stats,
                    importance_supervision=importance_supervision,
                )

                should_log_step = is_main_process(self.state) and (
                    self.global_step % self.config.logging.log_interval_steps == 0 or self.global_step == total_iterations
                )
                if should_log_step:
                    self._log_train_step(
                        total_steps=total_iterations,
                        batch_loss=loss.detach().item(),
                        stats=stats,
                        importance_stats=importance_stats,
                        importance_supervision=importance_supervision,
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

                if self._should_validate(total_iterations):
                    train_metrics = self._finalize_metric_accumulators(accumulators)
                    self._run_validation(train_metrics, total_iterations)
                    accumulators = self._create_metric_accumulators()

                if self._should_save_latest(total_iterations):
                    self._save_training_checkpoint(
                        checkpoint_path=self.experiment_dirs.checkpoints / "latest.pth",
                        checkpoint_kind="latest",
                        aliases=["latest"],
                    )
                if self._should_save_snapshot():
                    step_alias = f"step_{self.global_step:07d}"
                    self._save_training_checkpoint(
                        checkpoint_path=self.experiment_dirs.checkpoints / f"{step_alias}.pth",
                        checkpoint_kind="step",
                        aliases=[step_alias],
                    )

                if self.global_step >= total_iterations:
                    completed_pass = False
                    break

            if completed_pass:
                self.data_pass += 1
                self.step_in_pass = 0

    def _create_metric_accumulators(self) -> dict[str, Any]:
        return {
            "loss_sum": torch.zeros(1, device=self.state.device),
            "sample_count": torch.zeros(1, device=self.state.device),
            "fbeb_sums": {
                "fbeb/r1": torch.zeros(1, device=self.state.device),
                "fbeb/r2": torch.zeros(1, device=self.state.device),
                "fbeb/tau": torch.zeros(1, device=self.state.device),
                "fbeb/low_energy": torch.zeros(1, device=self.state.device),
                "fbeb/mid_energy": torch.zeros(1, device=self.state.device),
                "fbeb/high_energy": torch.zeros(1, device=self.state.device),
                "fbeb/alpha_low": torch.zeros(1, device=self.state.device),
                "fbeb/alpha_mid": torch.zeros(1, device=self.state.device),
                "fbeb/alpha_high": torch.zeros(1, device=self.state.device),
            },
            "fbeb_count": torch.zeros(1, device=self.state.device),
            "importance_sums": {
                "importance/mean": torch.zeros(1, device=self.state.device),
                "importance/std": torch.zeros(1, device=self.state.device),
                "importance/high_ratio": torch.zeros(1, device=self.state.device),
                "importance/raw_gate_mean": torch.zeros(1, device=self.state.device),
            },
            "importance_count": torch.zeros(1, device=self.state.device),
            "importance_supervision_sums": {
                "importance_sup/map_loss": torch.zeros(1, device=self.state.device),
                "importance_sup/aux_loss": torch.zeros(1, device=self.state.device),
            },
            "importance_supervision_count": torch.zeros(1, device=self.state.device),
        }

    def _update_metric_accumulators(
        self,
        accumulators: dict[str, Any],
        *,
        batch_size: int,
        loss: torch.Tensor,
        stats: dict[str, torch.Tensor] | None,
        importance_stats: dict[str, torch.Tensor] | None,
        importance_supervision: dict[str, torch.Tensor] | None,
    ) -> None:
        accumulators["loss_sum"] += loss.detach() * batch_size
        accumulators["sample_count"] += batch_size

        if self.config.logging.log_fbeb_stats and stats:
            for key in accumulators["fbeb_sums"]:
                accumulators["fbeb_sums"][key] += stats[key] * batch_size
            accumulators["fbeb_count"] += batch_size
        if self.config.logging.log_importance_stats and importance_stats:
            for key in accumulators["importance_sums"]:
                accumulators["importance_sums"][key] += importance_stats[key] * batch_size
            accumulators["importance_count"] += batch_size
        if importance_supervision is not None:
            accumulators["importance_supervision_sums"]["importance_sup/map_loss"] += (
                importance_supervision["map_loss"].detach() * batch_size
            )
            accumulators["importance_supervision_sums"]["importance_sup/aux_loss"] += (
                importance_supervision["aux_loss"].detach() * batch_size
            )
            accumulators["importance_supervision_count"] += batch_size

    def _finalize_metric_accumulators(self, accumulators: dict[str, Any]) -> dict[str, Any]:
        reduced = reduce_dict(
            {
                "loss_sum": accumulators["loss_sum"],
                "sample_count": accumulators["sample_count"],
                "fbeb_r1_sum": accumulators["fbeb_sums"]["fbeb/r1"],
                "fbeb_r2_sum": accumulators["fbeb_sums"]["fbeb/r2"],
                "fbeb_tau_sum": accumulators["fbeb_sums"]["fbeb/tau"],
                "fbeb_low_energy_sum": accumulators["fbeb_sums"]["fbeb/low_energy"],
                "fbeb_mid_energy_sum": accumulators["fbeb_sums"]["fbeb/mid_energy"],
                "fbeb_high_energy_sum": accumulators["fbeb_sums"]["fbeb/high_energy"],
                "fbeb_alpha_low_sum": accumulators["fbeb_sums"]["fbeb/alpha_low"],
                "fbeb_alpha_mid_sum": accumulators["fbeb_sums"]["fbeb/alpha_mid"],
                "fbeb_alpha_high_sum": accumulators["fbeb_sums"]["fbeb/alpha_high"],
                "fbeb_count": accumulators["fbeb_count"],
                "importance_mean_sum": accumulators["importance_sums"]["importance/mean"],
                "importance_std_sum": accumulators["importance_sums"]["importance/std"],
                "importance_high_ratio_sum": accumulators["importance_sums"]["importance/high_ratio"],
                "importance_raw_gate_mean_sum": accumulators["importance_sums"]["importance/raw_gate_mean"],
                "importance_count": accumulators["importance_count"],
                "importance_sup_map_loss_sum": accumulators["importance_supervision_sums"]["importance_sup/map_loss"],
                "importance_sup_aux_loss_sum": accumulators["importance_supervision_sums"]["importance_sup/aux_loss"],
                "importance_sup_count": accumulators["importance_supervision_count"],
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
                "fbeb/alpha_low": reduced["fbeb_alpha_low_sum"].item() / fbeb_total,
                "fbeb/alpha_mid": reduced["fbeb_alpha_mid_sum"].item() / fbeb_total,
                "fbeb/alpha_high": reduced["fbeb_alpha_high_sum"].item() / fbeb_total,
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
        if reduced["importance_sup_count"].item() > 0:
            importance_sup_total = reduced["importance_sup_count"].item()
            metrics["importance_supervision"] = {
                "importance_sup/map_loss": reduced["importance_sup_map_loss_sum"].item() / importance_sup_total,
                "importance_sup/aux_loss": reduced["importance_sup_aux_loss_sum"].item() / importance_sup_total,
            }
        else:
            metrics["importance_supervision"] = {}
        return metrics

    def _should_validate(self, total_iterations: int) -> bool:
        return self.global_step == total_iterations or (
            self.config.runtime.val_interval_steps > 0 and self.global_step % self.config.runtime.val_interval_steps == 0
        )

    def _should_save_latest(self, total_iterations: int) -> bool:
        return self.global_step == total_iterations or (
            self.config.logging.save_latest_interval_steps > 0
            and self.global_step % self.config.logging.save_latest_interval_steps == 0
        )

    def _should_save_snapshot(self) -> bool:
        return self.config.runtime.save_interval_steps > 0 and self.global_step % self.config.runtime.save_interval_steps == 0

    def _next_resume_cursor(self) -> tuple[int, int]:
        data_pass = self.data_pass
        step_in_pass = self.step_in_pass
        if step_in_pass >= len(self.train_loader):
            data_pass += 1
            step_in_pass = 0
        return data_pass, step_in_pass

    def _save_training_checkpoint(
        self,
        *,
        checkpoint_path,
        checkpoint_kind: str,
        aliases: list[str],
    ) -> None:
        data_pass, step_in_pass = self._next_resume_cursor()
        save_checkpoint(
            checkpoint_path,
            self.config,
            self.model,
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            data_pass,
            step_in_pass,
            self.global_step,
            self.best_psnr,
            self.best_ssim,
            main_process=is_main_process(self.state),
            wandb_run_id=self.wandb_run_id,
        )
        self._log_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            checkpoint_kind=checkpoint_kind,
            aliases=aliases,
            data_pass=data_pass,
            step_in_pass=step_in_pass,
        )

    def _run_validation(self, train_metrics: dict[str, Any], total_iterations: int) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        record: dict[str, Any] = {
            "type": "validation",
            "global_step": self.global_step,
            "total_iterations": total_iterations,
            "train_loss": train_metrics["loss"],
            "lr": current_lr,
        }
        if self.config.logging.log_fbeb_stats:
            record.update(train_metrics["fbeb"])
        if self.config.logging.log_importance_stats:
            record.update(train_metrics["importance"])
        if train_metrics["importance_supervision"]:
            record.update(train_metrics["importance_supervision"])

        if is_main_process(self.state):
            self.logger.info("Starting validation at global_step=%d.", self.global_step)
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
            self._save_training_checkpoint(
                checkpoint_path=self.experiment_dirs.checkpoints / "best_psnr.pth",
                checkpoint_kind="best_psnr",
                aliases=["best_psnr"],
            )
        if ema_metrics["ssim"] > self.best_ssim:
            self.best_ssim = ema_metrics["ssim"]
            self._save_training_checkpoint(
                checkpoint_path=self.experiment_dirs.checkpoints / "best_ssim.pth",
                checkpoint_kind="best_ssim",
                aliases=["best_ssim"],
            )

        self._log_validation(record)

    def _log_train_step(
        self,
        total_steps: int,
        batch_loss: float,
        stats: dict[str, torch.Tensor] | None,
        importance_stats: dict[str, torch.Tensor] | None,
        importance_supervision: dict[str, torch.Tensor] | None,
    ) -> None:
        if not is_main_process(self.state):
            return

        record: dict[str, Any] = {
            "type": "train_step",
            "global_step": self.global_step,
            "total_iterations": total_steps,
            "train_loss": batch_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        parts = [
            f"step={self.global_step}/{total_steps}",
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
            alpha_low = float(stats["fbeb/alpha_low"].item())
            alpha_mid = float(stats["fbeb/alpha_mid"].item())
            alpha_high = float(stats["fbeb/alpha_high"].item())
            record.update(
                {
                    "fbeb/r1": r1,
                    "fbeb/r2": r2,
                    "fbeb/tau": tau,
                    "fbeb/low_energy": low_energy,
                    "fbeb/mid_energy": mid_energy,
                    "fbeb/high_energy": high_energy,
                    "fbeb/alpha_low": alpha_low,
                    "fbeb/alpha_mid": alpha_mid,
                    "fbeb/alpha_high": alpha_high,
                }
            )
            parts.append(f"fbeb_r1={r1:.4f}")
            parts.append(f"fbeb_r2={r2:.4f}")
            parts.append(f"fbeb_tau={tau:.4f}")
            parts.append(f"fbeb_a=({alpha_low:.2f},{alpha_mid:.2f},{alpha_high:.2f})")
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
        if importance_supervision is not None:
            imp_sup_map = float(importance_supervision["map_loss"].item())
            imp_sup_aux = float(importance_supervision["aux_loss"].item())
            record.update(
                {
                    "importance_sup/map_loss": imp_sup_map,
                    "importance_sup/aux_loss": imp_sup_aux,
                }
            )
            parts.append(f"imp_sup={imp_sup_map:.4f}")

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
            self.tensorboard_writer.add_scalar("fbeb/alpha_low_step", record["fbeb/alpha_low"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/alpha_mid_step", record["fbeb/alpha_mid"], self.global_step)
            self.tensorboard_writer.add_scalar("fbeb/alpha_high_step", record["fbeb/alpha_high"], self.global_step)
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
        if importance_supervision is not None:
            self.tensorboard_writer.add_scalar(
                "importance_sup/map_loss_step",
                record["importance_sup/map_loss"],
                self.global_step,
            )
            self.tensorboard_writer.add_scalar(
                "importance_sup/aux_loss_step",
                record["importance_sup/aux_loss"],
                self.global_step,
            )
        self.tensorboard_writer.flush()

    def _log_validation(self, record: dict[str, Any]) -> None:
        if not is_main_process(self.state):
            return

        parts = [
            f"step={record['global_step']}/{record['total_iterations']}",
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
        self.tensorboard_writer.add_scalar("train/loss", record["train_loss"], record["global_step"])
        self.tensorboard_writer.add_scalar("train/lr", record["lr"], record["global_step"])
        if "raw_psnr" in record:
            self.tensorboard_writer.add_scalar("val/raw_psnr", record["raw_psnr"], record["global_step"])
            self.tensorboard_writer.add_scalar("val/raw_ssim", record["raw_ssim"], record["global_step"])
            self.tensorboard_writer.add_scalar("val/ema_psnr", record["ema_psnr"], record["global_step"])
            self.tensorboard_writer.add_scalar("val/ema_ssim", record["ema_ssim"], record["global_step"])
        if self.config.logging.log_fbeb_stats and "fbeb/r1" in record:
            self.tensorboard_writer.add_scalar("fbeb/r1", record["fbeb/r1"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/r2", record["fbeb/r2"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/tau", record["fbeb/tau"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/low_energy", record["fbeb/low_energy"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/mid_energy", record["fbeb/mid_energy"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/high_energy", record["fbeb/high_energy"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/alpha_low", record["fbeb/alpha_low"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/alpha_mid", record["fbeb/alpha_mid"], record["global_step"])
            self.tensorboard_writer.add_scalar("fbeb/alpha_high", record["fbeb/alpha_high"], record["global_step"])
        if self.config.logging.log_importance_stats and "importance/mean" in record:
            self.tensorboard_writer.add_scalar("importance/mean", record["importance/mean"], record["global_step"])
            self.tensorboard_writer.add_scalar("importance/std", record["importance/std"], record["global_step"])
            self.tensorboard_writer.add_scalar(
                "importance/high_ratio",
                record["importance/high_ratio"],
                record["global_step"],
            )
            self.tensorboard_writer.add_scalar(
                "importance/raw_gate_mean",
                record["importance/raw_gate_mean"],
                record["global_step"],
            )
        if "importance_sup/map_loss" in record:
            self.tensorboard_writer.add_scalar(
                "importance_sup/map_loss",
                record["importance_sup/map_loss"],
                record["global_step"],
            )
            self.tensorboard_writer.add_scalar(
                "importance_sup/aux_loss",
                record["importance_sup/aux_loss"],
                record["global_step"],
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
        aliases: list[str],
        data_pass: int,
        step_in_pass: int,
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
                "data_pass": data_pass,
                "step_in_pass": step_in_pass,
                "global_step": self.global_step,
                "best_psnr": self.best_psnr,
                "best_ssim": self.best_ssim,
                "run_id": self.wandb_run_id or self.wandb_run.id,
            },
            file_name=checkpoint_path.name,
        )
