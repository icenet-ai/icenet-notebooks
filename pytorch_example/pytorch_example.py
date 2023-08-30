#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import os

from icenet.data.loaders import IceNetDataLoaderFactory

from train_icenet_unet import train_icenet_unet
from test_icenet_unet import test_icenet_unet

# Quick hack to put us in the icenet-pipeline folder,
# assuming it was created as per 01.cli_demonstration.ipynb
pipeline_directory = os.path.join(os.path.dirname(__file__), "../../notebook-pipeline")
os.chdir(pipeline_directory)
print("Running in {}".format(os.getcwd()))

logging.getLogger().setLevel(logging.DEBUG)

print('A', torch.__version__)
print('B', torch.cuda.is_available())
print('C', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('D', torch.cuda.get_device_properties(device))

# set loader config and dataset names
implementation = "dask"
loader_config = "loader.notebook_api_data.json"
dataset_name = "pytorch_notebook"
lag = 1

# create IceNet dataloader
# (assuming notebook 03 has been ran and the loader config exists)
# and write a dataset config
dl = IceNetDataLoaderFactory().create_data_loader(
    implementation,
    loader_config,
    dataset_name,
    lag,
    n_forecast_days=7,
    north=False,
    south=True,
    output_batch_size=4,
    generate_workers=8,
)

# write dataset config
dl.write_dataset_config_only()
dataset_config = f"dataset_config.{dataset_name}.json"

# train and test model 
seed = 42
batch_size = 4
shuffle = True
num_workers = 0
persistent_workers = False

# train model
lit_unet_module, unet_model = train_icenet_unet(
    configuration_path=dataset_config,
    learning_rate=1e-4,
    max_epochs=100,
    seed=seed,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)

print("Finished training UNET model")
print(f"UNet model:\n{unet_model}")
print(f"UNet (Lightning Module):\n{lit_unet_module}")

# test model
print("Testing model")
y_hat_unet, y_true = test_icenet_unet(
    configuration_path=dataset_config,
    lit_module_unet=lit_unet_module,
    seed=seed,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)

print("Finished testing model")
print(f"y_hat_unet.shape: {y_hat_unet.shape}")
print(f"y_true.shape: {y_true.shape}")
