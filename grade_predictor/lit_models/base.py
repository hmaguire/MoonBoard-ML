"""Basic LightningModules on which other modules can be built."""
import argparse
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict

import lightning as L
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class BaseLitModel(L.LightningModule):
    """
    Basic LightningModule for PyTorch Lightning
        :param
            model: PyTorch model to be trained
            optimizer: Optimizer to be used, default: Adam
            learning_rate: Learning rate to be used, default: 1e-3
            loss: Loss function to be used, default: "mse_loss"
            one_cycle_max_lr: Max learning rate for learning_rate scheduler OneCycleLR
            one_cycle_total_steps: Total steps for OneCycleLR, default: 100
    """

    def __init__(self,
                 model,
                 optimizer: str = "Adam",
                 learning_rate: float = 1e-3,
                 loss: str = "mse_loss",
                 one_cycle_max_lr=None,
                 one_cycle_total_steps: int = 100):
        super().__init__()
        self.model = model

        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = learning_rate

        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = one_cycle_max_lr
        self.one_cycle_total_steps = one_cycle_total_steps

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, preds: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        if targets is None:
            targets = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return self.loss_fn(preds, targets)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y , id = batch["data"], batch["target"], batch["id"]
        preds = self(x)

        train_loss = self.loss(preds, y)
        train_mae = self.train_mae(preds, y)
        train_mse = self.train_mse(preds, y)

        self.log("train/loss", train_loss)
        self.log("train/mae", train_mae, on_step=False, on_epoch=True)
        self.log("train/mse", train_mse, on_step=False, on_epoch=True)

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, y , id = batch["data"], batch["target"], batch["id"]
        preds = self(x)

        val_loss = self.loss(preds, y)
        val_mae = self.val_mae(preds, y)
        val_mse = self.val_mse(preds, y)

        self.log("validation/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("validation/mae", val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/mse", val_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y , id = batch["data"], batch["target"], batch["id"]
        preds = self(x)

        test_loss = self.loss(preds, y)
        test_mae = self.test_mae(preds, y)
        test_mse = self.test_mse(preds, y)

        self.log("test/loss", test_loss, on_step=False, on_epoch=True)
        self.log("test/mae", test_mae, on_step=False, on_epoch=True)
        self.log("test/mse", test_mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss"}


    # def add_on_first_batch(self, metrics, outputs, batch_idx):
    #     if batch_idx == 0:
    #         outputs.update(metrics)
    #
    # def add_on_logged_batches(self, metrics, outputs):
    #     if self.is_logged_batch:
    #         outputs.update(metrics)
    #
    # def is_logged_batch(self):
    #     if self.trainer is None:
    #         return False
    #     else:
    #         return self.trainer._logger_connector.should_update_logs

# # Hide lines above until Lab 04
# # Hide lines below until Lab 03
# class BaseImageToTextLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
#     """Base class for ImageToText models in PyTorch Lightning."""
#
#     def __init__(self, model, args: argparse.Namespace = None):
#         super().__init__(model, args)
#         self.model = model
#         self.args = vars(args) if args is not None else {}
#
#         self.inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
#         self.start_index = self.inverse_mapping["<S>"]
#         self.end_index = self.inverse_mapping["<E>"]
#         self.padding_index = self.inverse_mapping["<P>"]
#
#         self.ignore_tokens = [self.start_index, self.end_index, self.padding_index]
#         self.val_cer = CharacterErrorRate(self.ignore_tokens)
#         self.test_cer = CharacterErrorRate(self.ignore_tokens)
#
#
# # Hide lines above until Lab 03
