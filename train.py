import sys; print('Python %s on %s' % (sys.version, sys.platform))

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler

from utils import barrier, reduce_mean, update_loss_info


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:

    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1
    regression = (model.module.bins is None) if ddp else (model.bins is None)

    # Choose autocast context depending on device
    if device.type == "cuda":
        print("Using CUDA autocast")
        autocast_context = lambda: autocast(enabled=grad_scaler.is_enabled())
    else:
        autocast_context = nullcontext  # no-op

    for image, target_points, target_density in data_iter:
        image = image.to(device, non_blocking=True)
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)

        with torch.set_grad_enabled(True), autocast_context():
            print("torch.set_grad_enabled")
            if not regression:
                pred_class, pred_density = model(image)
                loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
            else:
                pred_density = model(image)
                loss, loss_info = loss_fn(pred_density, target_density, target_points)

        # Backward/update section (optional here)
        if grad_scaler is not None:
            print("Using grad scaler")
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in loss_info.items()}
        # if rank == 0:
            # loss_info = {k: v.item() for k, v in loss_info.items()}
        info = update_loss_info(info, loss_info)

        barrier(ddp)

    return model, optimizer, grad_scaler, {k: np.mean(v) for k, v in info.items()}
