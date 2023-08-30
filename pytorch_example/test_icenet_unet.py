"""
Taken from Andrew McDonald's https://github.com/ampersandmcd/icenet-gan/blob/main/notebooks/4_forecast.ipynb
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

from icenet_pytorch_dataset import IceNetDataSetPyTorch

def test_icenet_unet(
    configuration_path: str,
    lit_module_unet: pl.LightningModule, 
    seed: int,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> tuple[np.ndarray]:
    pl.seed_everything(seed)

    # configure datasets and dataloaders
    test_dataset = IceNetDataSetPyTorch(configuration_path=configuration_path, mode="test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 persistent_workers=persistent_workers)

    # pass batches through unet and accumulate into list
    y_true = []
    y_hat_unet = []
    with torch.no_grad():
        for batch in test_dataloader:
            x, y, sample_weight = batch
            # save ground truth
            y_true.extend(y)
            # predict using UNet
            pred_unet = lit_module_unet(x.to(lit_module_unet.device)).detach().cpu().numpy()
            # save prediction
            y_hat_unet.extend(pred_unet)

    y_true = np.array(y_true)
    y_hat_unet = np.array(y_hat_unet)

    return y_hat_unet, y_true
