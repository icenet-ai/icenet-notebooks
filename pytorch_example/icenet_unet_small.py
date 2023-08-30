"""
Taken from Andrew McDonald's https://github.com/ampersandmcd/icenet-gan/blob/main/src/models.py.
"""

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from metrics import IceNetAccuracy, SIEError


class UNet(nn.Module):
    """
    A (small) implementation of a UNet for pixelwise classification.
    """
    
    def __init__(self,
                 input_channels, 
                 filter_size=3, 
                 n_filters_factor=1, 
                 n_forecast_days=6, 
                 n_output_classes=3,
                **kwargs):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.n_forecast_days = n_forecast_days
        self.n_output_classes = n_output_classes

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
                                    out_channels=n_output_classes*n_forecast_days,
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
        b, h, w, c = final_layer_logits.shape

        final_layer_logits = final_layer_logits.reshape((b, h, w, self.n_output_classes, self.n_forecast_days))

        # transpose from shape (b, h, w, t, c) to (b, h, w, t, c)
        output = F.softmax(final_layer_logits, dim=-1)  # apply over n_output_classes dimension
        
        return output  # shape (b, h, w, t, c)


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
        self.n_output_classes = model.n_output_classes  # this should be a property of the network

        metrics = {
            "val_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        }
        for i in range(self.model.n_forecast_days):
            metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.metrics = MetricCollection(metrics)

        test_metrics = {
            "test_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "test_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        }
        for i in range(self.model.n_forecast_days):
            test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.test_metrics = MetricCollection(test_metrics)

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
        x, y, sample_weight = batch
        y_hat = self.model(x)
        print("in training")
        print(f"y.shape: {y.shape}")
        print(f"y_hat.shape: {y_hat.shape}")
        # y and y_hat are shape (b, h, w, t, c) but loss expects (b, c, h, w, t)
        # note that criterion needs reduction="none" for weighting to work
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # requires int class encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.argmax(-1).long())
        else:
            # requires one-hot encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.movedim(-1, 1))
        loss = torch.mean(loss * sample_weight.movedim(-1, 1))
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.model(x)
        print("in validation")
        print(f"y.shape: {y.shape}")
        print(f"y_hat.shape: {y_hat.shape}")
        # y and y_hat are shape (b, h, w, t, c) but loss expects (b, c, h, w, t)
        # note that criterion needs reduction="none" for weighting to work
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # requires int class encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.argmax(-1).long())
        else:
            # requires one-hot encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.movedim(-1, 1))
        loss = torch.mean(loss * sample_weight.movedim(-1, 1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss
        y_hat_pred = y_hat.argmax(dim=-1).long()  # argmax over c where shape is (b, h, w, t, c)
        self.metrics.update(y_hat_pred, y.argmax(dim=-1).long(), sample_weight.squeeze(dim=-1))  # shape (b, h, w, t)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.model(x)
        # y and y_hat are shape (b, h, w, t, c) but loss expects (b, c, h, w, t)
        # note that criterion needs reduction="none" for weighting to work
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # requires int class encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.argmax(-1).long())
        else:
            # requires one-hot encoding
            loss = self.criterion(y_hat.movedim(-1, 1), y.movedim(-1, 1))
        loss = torch.mean(loss * sample_weight.movedim(-1, 1))
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)  # epoch-level loss
        y_hat_pred = y_hat.argmax(dim=-1)  # argmax over c where shape is (b, h, w, t, c)
        self.test_metrics.update(y_hat_pred, y.argmax(dim=-1).long(), sample_weight.squeeze(dim=-1))  # shape (b, h, w, t)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(),on_step=False, on_epoch=True, sync_dist=True)  # epoch-level metrics
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer
        }