"""Base Dataset class."""

from typing import Any, Callable, Dict, Sequence, Tuple, Union, TypedDict

import torch


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: TypedDict,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        xs: SequenceOrTensor
        indexes: SequenceOrTensor
        orders: SequenceOrTensor
        relative_xs: SequenceOrTensor
        relative_ys: SequenceOrTensor

        if len(data["id_tokens"]) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.xs = data["id_tokens"]
        self.idxs = data["index"]
        self.order_tokens = data["order_tokens"]
        self.rel_x_tokens = data["rel_x_tokens"]
        self.rel_y_tokens = data["rel_y_tokens"]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.xs)

    def __getitem__(self, index: int) -> Dict:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        xs, target = self.xs[index], self.targets[index]


        if self.transform is not None:
            xs = self.transform(xs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        tokens = {
            'xs': xs,
            'order': self.order_tokens[index],
            'rel_x_tokens': self.rel_x_tokens[index],
            'rel_y_tokens': self.rel_y_tokens[index]
        }

        sample = {
            'data': tokens,
            'target': target,
            'id': self.idxs[index]
        }
        return sample


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels

def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )
