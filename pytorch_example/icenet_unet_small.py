"""
Taken from Andrew McDonald's https://github.com/ampersandmcd/icenet-gan/blob/main/src/models.py.
"""

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
# from torchmetrics import MetricCollection
# from metrics import IceNetAccuracy, SIEError


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


class UNet(nn.Module):
    """
    A (small) implementation of a UNet for pixelwise classification.
    """
    
    def __init__(self,
                 input_channels: int, 
                 filter_size: int = 3, 
                 n_filters_factor: int = 1, 
                 n_forecast_days: int = 7):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.n_forecast_days = n_forecast_days

        self.conv1a = nn.Conv2d(in_channels=input_channels, 
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv1b = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=int(128*n_filters_factor))

        self.conv2a = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv2b = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(256*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=int(256*n_filters_factor))

        self.conv9a = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv9b = nn.Conv2d(in_channels=int(256*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")
        self.conv9c = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                out_channels=int(128*n_filters_factor),
                                kernel_size=filter_size,
                                padding="same")  # no batch norm on last layer

        self.final_conv = nn.Conv2d(in_channels=int(128*n_filters_factor),
                                    out_channels=n_forecast_days,
                                    kernel_size=filter_size,
                                    padding="same")
        
    def forward(self, x):
        # transpose from shape (b, h, w, c) to (b, c, h, w) for pytorch conv2d layers
        x = torch.movedim(x, -1, 1)  # move c from last to second dim

        # run through network
        conv1 = self.conv1a(x)  # input to 128
        conv1 = F.relu(conv1)
        conv1 = self.conv1b(conv1)  # 128 to 128
        conv1 = F.relu(conv1)
        bn1 = self.bn1(conv1)
        pool1 = F.max_pool2d(bn1, kernel_size=(2, 2))

        conv2 = self.conv2a(pool1)  # 128 to 256
        conv2 = F.relu(conv2)
        conv2 = self.conv2b(conv2)  # 256 to 256
        conv2 = F.relu(conv2)
        bn2 = self.bn2(conv2)

        up9 = F.upsample(bn2, scale_factor=2, mode="nearest")
        up9 = self.conv9a(up9)  # 256 to 128
        up9 = F.relu(up9)
        merge9 = torch.cat([bn1, up9], dim=1) # 128 and 128 to 256 along c dimension
        conv9 = self.conv9b(merge9)  # 256 to 128
        conv9 = F.relu(conv9)
        conv9 = self.conv9c(conv9)  # 128 to 128
        conv9 = F.relu(conv9)  # no batch norm on last layer
 
        final_layer_logits = self.final_conv(conv9)

        # transpose from shape (b, c, h, w) back to (b, h, w, c) to align with training data
        final_layer_logits = torch.movedim(final_layer_logits, 1, -1)  # move c from second to final dim

        # apply sigmoid
        output = F.sigmoid(final_layer_logits)
        
        return output  # shape (b, h, w, c)


class LitUNet(pl.LightningModule):
    """
    A LightningModule wrapping the UNet implementation of IceNet.
    """
    def __init__(self,
                 model: nn.Module,
                 criterion: callable,
                 learning_rate: float):
        """
        Construct a UNet LightningModule.
        Note that we keep hyperparameters separate from dataloaders to prevent data leakage at test time.
        :param model: PyTorch model
        :param criterion: PyTorch loss function for training instantiated with reduction="none"
        :param learning_rate: Float learning rate for our optimiser
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

        # metrics = {
        #     "val_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
        #     "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        # }
        # for i in range(self.model.n_forecast_days):
        #     metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
        #     metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        # self.metrics = MetricCollection(metrics)

        # test_metrics = {
        #     "test_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
        #     "test_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        # }
        # for i in range(self.model.n_forecast_days):
        #     test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
        #     test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        # self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, x):
        """
        Implement forward function.
        :param x: Inputs to model.
        :return: Outputs of model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a pass through a batch of training data.
        Apply pixel-weighted loss by manually reducing.
        See e.g. https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689/5.
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Loss from this batch of data for use in backprop
        """
        print("in training")
        x, y, sample_weight = batch
        y_hat = self.model(x)
        print(f"y.shape: {y.shape}")
        print(f"y[:,:,:,:,0].shape: {y[:,:,:,:,0].shape}")
        print(f"y_hat.shape: {y_hat.shape}")
        print(f"sample_weight[:,:,:,:,0].shape: {sample_weight[:,:,:,:,0].shape}")
        # y and sample_weight have shape (b, h, w, c, 1)
        # y_hat has shape (b, h, w, c)
        loss = self.criterion(y[:,:,:,:,0], y_hat)
        loss = torch.mean(loss * sample_weight[:,:,:,:,0])
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("in validation")
        x, y, sample_weight = batch
        y_hat = self.model(x)
        print(f"y.shape: {y.shape}")
        print(f"y[:,:,:,:,0].shape: {y[:,:,:,:,0].shape}")
        print(f"y_hat.shape: {y_hat.shape}")
        print(f"sample_weight[:,:,:,:,0].shape: {sample_weight[:,:,:,:,0].shape}")
        # y and sample_weight have shape (b, h, w, c, 1)
        # y_hat has shape (b, h, w, c)
        loss = self.criterion(y[:,:,:,:,0], y_hat)
        loss = torch.mean(loss * sample_weight[:,:,:,:,0])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss
        return loss

    # def on_validation_epoch_end(self):
    #     self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
    #     self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.model(x)
        # y and sample_weight have shape (b, h, w, c, 1)
        # y_hat has shape (b, h, w, c)
        loss = self.criterion(y[:,:,:,:,0], y_hat)
        loss = torch.mean(loss * sample_weight[:,:,:,:,0])
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss
        return loss

    # def on_test_epoch_end(self):
    #     # self.log_dict(self.test_metrics.compute(),on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
    #     self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer
        }