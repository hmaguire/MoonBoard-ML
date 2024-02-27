"""Basic LightningModules on which other modules can be built."""
import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "mse_loss"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)

        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss"}

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        preds = self.model(x)
        return torch.argmax(preds, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, preds, loss = self._run_on_batch(batch)
        self.train_mae(preds, y)
        self.train_mse(preds, y)

        self.log("train/loss", loss)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)

        outputs = {"loss": loss}
        # Hide lines below until Lab 04
        self.add_on_first_batch({"preds": preds.detach()}, outputs, batch_idx)
        # Hide lines above until Lab 04

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        x = batch["data"]
        y = batch["target"]
        id = batch["id"]
        preds = self(x)
        loss = self.loss_fn(preds, y)

        return x, y, preds, loss

    def validation_step(self, batch, batch_idx):
        x, y, preds, loss = self._run_on_batch(batch)
        self.val_mae(preds, y)
        self.val_mse(preds, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}
        # Hide lines below until Lab 04
        self.add_on_first_batch({"preds": preds.detach()}, outputs, batch_idx)
        # Hide lines above until Lab 04

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, preds, loss = self._run_on_batch(batch)
        self.test_mae(preds, y)
        self.test_mse(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True)

    # Hide lines below until Lab 04
    def add_on_first_batch(self, metrics, outputs, batch_idx):
        if batch_idx == 0:
            outputs.update(metrics)

    def add_on_logged_batches(self, metrics, outputs):
        if self.is_logged_batch:
            outputs.update(metrics)

    def is_logged_batch(self):
        if self.trainer is None:
            return False
        else:
            return self.trainer._logger_connector.should_update_logs

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
