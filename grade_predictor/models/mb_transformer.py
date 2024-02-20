import argparse
from typing import Any, Dict
from typing import Tuple
import pytorch_lightning as pl
import torch
import logging  # import some stdlib components to control what's display
import textwrap
import traceback
from grade_predictor.lit_models.base import BaseLitModel
import numpy as np

FC1_DIM = 1024
FC2_DIM = 128
FC_DROPOUT = 0.5


class MB2016Transformer(pl.LightningModule):
    """MB Transformer"""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()  # just like in torch.nn.Module, we need to call the parent class __init__
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        input_dim = np.prod(self.data_config["input_dims"])

        self.loss_fn = torch.nn.MSELoss()
        fc1_dim = self.args.get("fc1", FC1_DIM)
        fc2_dim = self.args.get("fc2", FC2_DIM)
        dropout_p = self.args.get("fc_dropout", FC_DROPOUT)

        # attach torch.nn.Modules as top level attributes during init, just like in a torch.nn.Module
        n, d_model = 199, 50
        nhead, nlayers = 2, 3
        self.embedding = torch.nn.Embedding(n, d_model, max_norm=True)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        # self.position_embedding = torch.nn.Embedding(n, d, max_norm=True)
        self.linear = torch.nn.Linear(in_features=d_model, out_features=1)
        self.flatten = torch.nn.Flatten()
        self.linear2 = torch.nn.Linear(in_features=15, out_features=1)
        # we like to define the entire model as one torch.nn.Module -- typically in a separate class

    # optionally, define a forward method
    def forward(self, xs):
        # xs = xs[0]
        # print(xs.size())
        # display(xs)
        # print(xs.type())
        xs = xs[:, 0]
        xs = self.embedding(xs)
        xs = self.transformer_encoder(xs)  # we like to just call the model's forward method
        xs = self.linear(xs)
        xs = self.flatten(xs)
        xs = self.linear2(xs)
        return xs

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc1", type=int, default=FC1_DIM)
        parser.add_argument("--fc2", type=int, default=FC2_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        return parser

    # def training_step(self, batch, batch_idx):
    #     xs, ys = batch  # unpack the batch
    #     xs = xs[:, 0]
    #     ys = ys.unsqueeze(1)
    #     preds = self(xs)  # apply the model
    #     loss = self.loss_fn(preds, y[:, 1:])
    #     loss = torch.nn.functional.mse_loss(outs, ys)  # compute the (squared error) loss
    #     self.log("train/loss", loss)
    #     outputs = {"loss": loss}
    #     return loss
    #
    # def validation_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     xs, ys = batch  # unpack the batch
    #     xs = xs[:,0]
    #     ys = ys.unsqueeze(1)
    #     preds = self(xs)  # apply the model
    #     loss = torch.nn.functional.mse_loss(preds, ys)  # compute the (squared error) loss
    #     return loss
    #
    # def test_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     xs, ys = batch  # unpack the batch
    #     xs = xs[:,0]
    #     ys = ys.unsqueeze(1)
    #     preds = self(xs)  # apply the model
    #     loss = torch.nn.functional.mse_loss(preds, ys)  # compute the (squared error) loss
    #     return loss