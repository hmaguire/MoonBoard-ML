"""Base DataModule class."""

import argparse
import os
from typing import Collection, Optional, Tuple, Union

from grade_predictor.data.util import BaseDataset
import grade_predictor.metadata.shared as metadata
import lightning as L
from lightning.pytorch.core.datamodule import LightningDataModule
import torch
from torch.utils.data import ConcatDataset, DataLoader


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


BATCH_SIZE = 128
NUM_AVAIL_CPUS = os.cpu_count()
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode,
# we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


class BaseDataModule(LightningDataModule):
    """Base for all of our LightningDataModules.

    Learn more at about LDMs at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self,
                 batch_size: int = 128,
                 num_workers: int = DEFAULT_NUM_WORKERS,
                 # num_workers: int = 0,
                 args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.max_sequence: int
        self.token_dict_size: int
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIRNAME



    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.

        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.

        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            # persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
