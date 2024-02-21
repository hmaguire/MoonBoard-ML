import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class MB2016Transformer(pl.LightningModule):
    """MB Transformer with cleaner structure and improved readability."""

    EMBEDDING_SIZE = 32
    TF_NHEADS = 2
    TF_LAYERS = 2
    TF_DROPOUT = 0.2
    TF_FF_SIZE = 128

    def __init__(self, data_config: dict, args: argparse.Namespace = None):
        super().__init__()
        self.data_config = data_config

        self.input_dim = int(np.prod(self.data_config["input_dims"]))
        self.output_dim = int(np.prod(self.data_config["output_dims"]))
        self.token_dict_size = self.data_config["token_dict_size"]
        self.max_sequence = self.data_config["max_sequence"]

        self.embedding_size = self.EMBEDDING_SIZE
        self.tf_nheads = self.TF_NHEADS
        self.tf_nlayers = self.TF_LAYERS
        self.tf_dropout = self.TF_DROPOUT
        self.tf_ff_size = self.TF_FF_SIZE
        self.tf_max_len = self.max_sequence

        self.loss_fn = nn.MSELoss()

        self.embedding = nn.Embedding(self.token_dict_size, self.embedding_size, max_norm=True)
        self.pos_embedding = nn.Embedding(4, self.embedding_size, max_norm=True)

        # self.pos_encoder_order = PositionalEncoding1D(self.embedding_size)
        #
        # position_dim = 3
        # w_dim = 11
        # h_dim = 19
        # pe_blank = torch.zeros(1, position_dim, w_dim, h_dim, self.embedding_size)
        # self.pos_encoder_3d = Summer(PositionalEncoding3D(self.embedding_size))(pe_blank)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.tf_nheads,
                                                   dropout=self.tf_dropout, dim_feedforward=self.tf_ff_size,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.tf_nlayers)
        self.fc1 = nn.Linear(self.embedding_size*self.tf_max_len, 10)
        self.fc2 = nn.Linear(10, self.output_dim)

    def forward(self, xs):

        xs = self.embedding(xs[:, 0]) + self.pos_embedding(xs[:, 1])
        xs = self.transformer_encoder(xs)
        xs = xs.reshape(-1, self.embedding_size*self.tf_max_len)
        xs = torch.tanh(self.fc1(xs))
        xs = self.fc2(xs)
        return xs


    # pe_blank = torch.zeros(128, position_dim, w_dim, h_dim, self.embedding_size)
    # (self.pos_encoder_3d, torch.full(position_indices.shape[0]1,-1), position_indices)
    # torch.vmap(torch.index_select)(self.pos_encoder_3d, torch.full((position_indices.shape[0],0),-1), position_indices)
    @staticmethod
    def add_to_argparse(parser):
        # parser.add_argument("--fc1", type=int, default=MB2016Transformer.DEFAULT_FC1_DIM)
        # parser.add_argument("--fc2", type=int, default=MB2016Transformer.DEFAULT_FC2_DIM)
        # parser.add_argument("--fc_dropout", type=float, default=MB2016Transformer.DEFAULT_FC_DROPOUT)
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

    def select_positional_encodings(self, indices):
        pe_full = self.pos_encoder_3d
        p_indices, w_indices, h_indices = indices
        pe_sequence = torch.zeros((1,self.max_sequence,128))
        for i in range(self.max_sequence):
            pe_sequence[1,i] = pe_full[0,p_indices[i], w_indices[i], h_indices[i]]
        return pe_sequence
