from __future__ import annotations

from contextlib import nullcontext

import torch

from src.utils.distributed import reduce_dict
from src.utils.metrics import calculate_psnr, calculate_ssim


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_psnr = torch.zeros(1, device=device)
    total_ssim = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)
    for blur, sharp in dataloader:
        blur = blur.to(device, non_blocking=True)
        sharp = sharp.to(device, non_blocking=True)
        autocast_context = torch.cuda.amp.autocast(enabled=amp) if device.type == "cuda" else nullcontext()
        with autocast_context:
            output = model(blur)
        output = output.clamp_(0.0, 1.0)

        for pred, target in zip(output, sharp):
            pred_cpu = pred.detach().cpu()
            target_cpu = target.detach().cpu()
            total_psnr += calculate_psnr(pred_cpu, target_cpu)
            total_ssim += calculate_ssim(pred_cpu, target_cpu)
            total_count += 1.0

    reduced = reduce_dict(
        {"psnr_sum": total_psnr, "ssim_sum": total_ssim, "count": total_count},
        average=False,
    )
    count = max(reduced["count"].item(), 1.0)
    metrics = {
        "psnr": reduced["psnr_sum"].item() / count,
        "ssim": reduced["ssim_sum"].item() / count,
    }

    if was_training:
        model.train()
    return metrics
