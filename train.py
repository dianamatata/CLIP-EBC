import sys

print("Python %s on %s" % (sys.version, sys.platform))

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from utils import barrier, reduce_mean, update_loss_info
from utils_augmentation import augment_image_and_keypoints


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
    data_augmentation: bool,
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:
    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1
    regression = (model.module.bins is None) if ddp else (model.bins is None)

    # Choose autocast context depending on device
    if device.type == "cuda":
        print("Using CUDA autocast")
        from torch.cuda.amp import autocast

        autocast_context = lambda: autocast(enabled=grad_scaler.is_enabled())
    else:
        autocast_context = nullcontext  # no-op

    for image, target_points, target_density in data_iter:
        if data_augmentation:
            """
            From data_iter we get an image tensor containing many images, in example here: B=3. Indeed: (torch.Size([3, 3, 448, 448])) [B, C, H, W] 
            We want HWC format.
            
            image.cpu() moves the tensor from GPU (CUDA) memory to CPU memory. if already on CPU, it has no effect.
            """
            print("Applying data augmentation")

            image_np = image.cpu().numpy()
            image_np_hwc = np.transpose(image_np, (0, 2, 3, 1))  # [B, H, W, C]
            keypoints_np = [tp.cpu().numpy() for tp in target_points]
            augmented_images = []
            augmented_keypoints = []
            for i in range(image_np_hwc.shape[0]):  # batch of several images
                assert isinstance(keypoints_np[i], np.ndarray), f"Expected numpy.ndarray, got {type(keypoints_np[i])}"
                assert keypoints_np[i].shape[1] == 2, f"Expected keypoints with shape (N, 2), got {keypoints_np[i].shape}"
                assert image_np_hwc[i].ndim == 3 and image_np_hwc[i].shape[2] == 3, f"Expected RGB image with 3 channels, got shape {image_np_hwc[i].shape}"

                img_aug, keypoints_aug = augment_image_and_keypoints(image_np_hwc[i], keypoints_np[i])
                augmented_images.append(img_aug)
                augmented_keypoints.append(keypoints_aug)
            # NumPy array → PyTorch tensor
            augmented_tensor_image = torch.from_numpy(np.transpose(augmented_images, (0, 3, 1, 2)))
            augmented_tensor_keypoints = [torch.from_numpy(keypoints_aug) for keypoints_aug in augmented_keypoints]

            image = augmented_tensor_image
            target_points = augmented_tensor_keypoints

        image = image.to(device, non_blocking=True)
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)

        with torch.set_grad_enabled(True), autocast_context():
            if not regression:
                pred_class, pred_density = model(image)
                loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
            else:
                pred_density = model(image)
                loss, loss_info = loss_fn(pred_density, target_density, target_points)

        # Backward/update section (optional here)
        if grad_scaler is not None:
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
