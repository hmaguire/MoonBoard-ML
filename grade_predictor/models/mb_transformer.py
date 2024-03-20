import argparse
from typing import Dict
import logging
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class MB2016Transformer(L.LightningModule):

    """MB Transformer model.
        :parameter
            data_config: Dict[str, int]
            embedding_size: int, default: 128
            tf_nheads: int, default: 2
            tf_ff_size: int, default: 128
            tf_nlayers: int, default: 2
            tf_dropout: float, default: 0.2
            fc2_size: int, default: 10
            output_dim: int, default: 1
            model_complexity: List[str], default: ["order_pos"]
    """

    def __init__(
            self,
            data_config,
            embedding_size: int = 128,
            tf_nheads: int = 2,
            tf_ff_size: int = 128,
            tf_nlayers: int = 2,
            tf_dropout: float = 0.2,
            fc2_size: int = 10,
            output_dim: int = 1,
            model_complexity=None
    ) -> None:

        super().__init__()
        if model_complexity is None:
            model_complexity = ["order_pos"]
        self.data_config = data_config

        self.output_dim = output_dim
        self.token_dict_size = data_config["token_dict_size"]
        self.max_sequence = data_config["max_sequence"]

        self.model_complexity = model_complexity

        self.embedding_size = embedding_size
        self.tf_nheads = tf_nheads
        self.tf_nlayers = tf_nlayers
        self.tf_ff_size = tf_ff_size
        self.tf_dropout = tf_dropout

        self.fc2_size = fc2_size
        self.tf_max_len = self.max_sequence

        # self.loss_fn = nn.MSELoss()

        self.embedding = nn.Embedding(self.token_dict_size, self.embedding_size, max_norm=True)
        if "order_pos" in self.model_complexity:
            self.pos_embedding = nn.Embedding(4, self.embedding_size, max_norm=True)
        if "spacial_pos" in self.model_complexity:
            self.rel_x_embedding = nn.Embedding(22, self.embedding_size // 2, max_norm=True)
            self.rel_y_embedding = nn.Embedding(36, self.embedding_size // 2, max_norm=True)
        # self.pos_encoder_order = PositionalEncoding1D(self.embedding_size)
        #
        # position_dim = 3
        # w_dim = 11
        # h_dim = 19
        # pe_blank = torch.zeros(1, position_dim, w_dim, h_dim, self.embedding_size)
        # self.pos_encoder_3d = Summer(PositionalEncoding3D(self.embedding_size))(pe_blank)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.tf_nheads,
                                                   dropout=self.tf_dropout, dim_feedforward=self.tf_ff_size,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.tf_nlayers)
        self.fc1 = nn.Linear(self.embedding_size*self.tf_max_len, self.fc2_size)
        self.fc2 = nn.Linear(self.fc2_size, self.output_dim)

    def forward(self, data):
        xs = data['xs']
        xs = self.embedding(xs)
        if "order_pos" in self.model_complexity:
            order = data['order']
            xs = xs + self.pos_embedding(order)

        if "spacial_pos" in self.model_complexity:
            rel_x_tokens = data['rel_x_tokens']
            rel_y_tokens = data['rel_y_tokens']
            rel_x = self.rel_x_embedding(rel_x_tokens)
            rel_y = self.rel_y_embedding(rel_y_tokens)
            re_xy = torch.concat([rel_x, rel_y], -1)
            xs = torch.einsum('b t d, b t t d -> b t d', self.embedding(xs), re_xy)

        xs = self.transformer_encoder(xs)
        xs = xs.reshape(-1, self.embedding_size*self.tf_max_len)
        xs = torch.relu(self.fc1(xs))
        xs = self.fc2(xs)
        return xs


    # pe_blank = torch.zeros(128, position_dim, w_dim, h_dim, self.embedding_size)
    # (self.pos_encoder_3d, torch.full(position_indices.shape[0]1,-1), position_indices)
    # torch.vmap(torch.index_select)(self.pos_encoder_3d, torch.full((position_indices.shape[0],0),-1), position_indices)


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

    def relative_to_absolute(self, q, x):
        """
        Converts the dimension that is specified from the axis
        from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
          Input: [bs, heads, length, 2*length - 1]
          Output: [bs, heads, length, length]
        """
        b, l, device, dtype = *q.shape, q.device, q.dtype
        dd = {'device': device, 'dtype': dtype}
        col_pad = torch.zeros((b, l, 1), **dd)
        x = torch.cat((q, col_pad), dim=2)  # zero pad 2l-1 to 2l
        flat_x = rearrange(x, 'b l c -> b (l c)')
        flat_pad = torch.zeros((b, l - 1), **dd)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x_padded.reshape(b, l + 1, 2 * l - 1)
        final_x = final_x[:, :, :l, (l - 1):]
        return final_x


# from grade_predictor.data import MB2016
# from grade_predictor.data import base_data_module
# from grade_predictor.metadata import mb2016 as data_config
# if __name__ == "__main__":
#     base_dataset = base_data_moduleBaseDataModule()
#     dataset = MB2016()
#     dataset.prepare_data()
#     dataset.setup()
#     train_dataloader = dataset.train_dataloader()
#     val_dataloader = dataset.val_dataloader()
#     model = MB2016Transformer({
#         "input_dims": dataset.input_dims,
#         "output_dims": dataset.output_dims,
#         "token_dict_size": dataset.id_token_dict_size,
#         "max_sequence": dataset.max_sequence
#     })
#     trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu")
#
#     trainer.fit(model, train_dataloader, val_dataloader)
