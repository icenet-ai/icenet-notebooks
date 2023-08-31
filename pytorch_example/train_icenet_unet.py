"""
Taken from Andrew McDonald's https://github.com/ampersandmcd/icenet-gan/blob/main/src/train_icenet.py
"""

from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from icenet_unet_small import weighted_mse_loss, UNet, LitUNet
from icenet_pytorch_dataset import IceNetDataSetPyTorch


def train_icenet_unet(
    configuration_path: str,
    learning_rate: float,
    max_epochs: int,
    filter_size: int = 3,
    n_filters_factor: float = 1.0,
    seed: int = 42,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> tuple[LitUNet, UNet]:
    """
    Train IceNet using the arguments specified.
    """
    pl.seed_everything(seed)
    
    # configure datasets and dataloaders
    train_dataset = IceNetDataSetPyTorch(configuration_path, mode="train")
    val_dataset = IceNetDataSetPyTorch(configuration_path, mode="val")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  persistent_workers=persistent_workers)
    # no need to shuffle validation set
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                persistent_workers=persistent_workers)

    # construct unet
    model = UNet(
        input_channels=len(train_dataset._ds._config["channels"]),
        filter_size=filter_size,
        n_filters_factor=n_filters_factor,
        n_forecast_days=train_dataset._ds._config["n_forecast_days"]
    )
    
    criterion = nn.MSELoss(reduction="none")
    
    # configure PyTorch Lightning module
    lit_module = LitUNet(
        model=model,
        criterion=criterion,
        learning_rate=learning_rate
    )

    # set up trainer configuration
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        max_epochs=max_epochs,
        num_sanity_val_steps=1,
    )
    trainer.callbacks.append(ModelCheckpoint(monitor="val_accuracy", mode="max"))

    # train model
    print(
        f"Training {len(train_dataset)} examples / {len(train_dataloader)} "
        f"batches (batch size {batch_size})."
    )
    print(
        f"Validating {len(val_dataset)} examples / {len(val_dataloader)} "
        f"batches (batch size {batch_size})."
    )
    trainer.fit(lit_module, train_dataloader, val_dataloader)
    
    return lit_module, model
