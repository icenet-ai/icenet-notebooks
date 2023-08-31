#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import logging

from icenet.data.loaders import IceNetDataLoaderFactory
from torch.utils.data import DataLoader
from icenet_pytorch_dataset import IceNetDataSetPyTorch

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
    generate_workers=8)

# write dataset config
dl.write_dataset_config_only()
dataset_config = f"dataset_config.{dataset_name}.json"

# test creation of custom PyTorch dataset and obtaining samples from them
ds_torch = IceNetDataSetPyTorch(configuration_path=dataset_config,
                                mode="train")

logging.info("Inspecting dataset from torch")
logging.info(ds_torch.__len__())
logging.info(ds_torch._dates[0])
#logging.info(ds_torch.__getitem__(0))
logging.info(ds_torch._dl.generate_sample(date=pd.Timestamp(ds_torch._dates[0].replace('_', '-'))))

# create custom PyTorch datasets for train, validation and test
train_dataset = IceNetDataSetPyTorch(configuration_path=dataset_config, mode="train")
val_dataset = IceNetDataSetPyTorch(configuration_path=dataset_config, mode="val")
test_dataset = IceNetDataSetPyTorch(configuration_path=dataset_config, mode="test")

# creating PyTorch dataloaders from the datasets
batch_size = 4
shuffle = True
pworkers = False
num_workers = 0

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              persistent_workers=pworkers,
                              num_workers=num_workers)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            persistent_workers=pworkers,
                            num_workers=num_workers)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             persistent_workers=pworkers,
                             num_workers=num_workers)

logging.info("Inspecting dataloader from torch")
logging.info("Getting next sample from {} training samples".format(len(train_dataloader)))
for i, data in enumerate(iter(train_dataloader)):
    logging.info("Train sample: {}".format(i))
    train_features, train_labels, sample_weights = data
    logging.info(train_features.shape)
    logging.info(train_labels.shape)
    logging.info(sample_weights.shape)



