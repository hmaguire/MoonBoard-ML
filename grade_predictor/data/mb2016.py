import itertools
import pickle
from typing import Optional

from grade_predictor.data.base_data_module import BaseDataModule, load_and_print_info
from grade_predictor.data.util import BaseDataset, split_dataset
import grade_predictor.metadata.mb2016 as metadata
from grade_predictor.util import temporary_working_directory
import natsort
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer
import torch


# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.pipeline import Pipeline


class MB2016(BaseDataModule):
    """MB2016 dataset: Scraped Moonboard 2016 climbs information"""

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.min_repeats = self.args.get("min_repeats", metadata.MIN_REPEATS)
        self.min_grade_count = self.args.get("min_grade_count", metadata.MIN_GRADE_COUNT)
        self.max_start_holds = self.args.get("max_start_holds", metadata.MAX_START_HOLDS)
        self.min_mid_holds = self.args.get("min_mid_holds", metadata.MIN_MID_HOLDS)
        self.max_mid_holds = self.args.get("max_mid_holds", metadata.MAX_MID_HOLDS)
        self.max_end_holds = self.args.get("max_end_holds", metadata.MAX_END_HOLDS)
        self.with_start_mid_end_tokens = self.args.get("with_start_mid_end_tokens", False)
        self.grade_count = None
        self.grades = None

        self.max_sequence = (
            self.max_start_holds + self.max_mid_holds + self.max_end_holds + (4 if self.with_start_mid_end_tokens else 0)
        )
        self.id_token_dict = self._id_token_dict(self.with_start_mid_end_tokens)
        self.rel_x_token_dict = self._rel_token_dict(11)
        self.rel_y_token_dict = self._rel_token_dict(18)

        self.id_token_dict_size = len(self.id_token_dict) + 1     # embedding indexing falls out of range otherwise



        self.test_size = self.args.get("test_size", 0.2)
        self.random_state = self.args.get("random_state", 42)

        self.data_dir = self.args.get("data_dir", metadata.DATA_DIRNAME)
        self.processed_data_dir = self.args.get("data_dir", metadata.PROCESSED_DATA_DIRNAME)

        self.input_dims = (4, self.max_sequence)
        self.output_dims = (1,)

    def prepare_data(self) -> None:

        # check if data has already been prepared
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        if self.processed_data_filename.exists():
            return

        # extract dataset to pandas dataframe
        with temporary_working_directory(self.data_dir):
            with open(self.raw_data_filename, "rb") as pkl_file:
                pickle_data = pickle.load(pkl_file)
                data = pd.DataFrame.from_dict(pickle_data).T
                data.reset_index(inplace=True)

        # map grades to integers
        sorted_grades = natsort.natsorted(data["grade"].unique())
        grade_to_float = {}
        for i, value in enumerate(sorted_grades):
            grade_to_float[value] = np.array([np.float32(i)])
        data["numeric_grade"] = data["grade"].map(grade_to_float)

        # Clean dataset of extraneous climbs
        data = data[data["repeats"] >= self.min_repeats]
        data = data[data.mid.map(len) <= self.max_mid_holds]
        data = data[data.mid.map(len) >= self.min_mid_holds]
        data = data[data.start.map(len) <= self.max_start_holds]
        data = data[data.end.map(len) <= self.max_end_holds]

        self.grade_count = data["grade"].value_counts()
        self.grades = [key for key, value in self.grade_count.items() if value >= self.min_grade_count]
        data = data[data["grade"].isin(self.grades)]

        # Use index for IDs and reset_index
        data.reset_index(drop=True,inplace=True)

        # Tokenize
        data = pd.concat([data, self._extract_from_df(data)], axis=1)

        # Save cleaned data
        with temporary_working_directory(self.processed_data_dir):
            data.to_pickle(self.processed_data_filename)

    def setup(self, stage: Optional[str] = None) -> None:

        with temporary_working_directory(self.processed_data_dir):
            with open(self.processed_data_filename, "rb") as pkl_file:
                pickle_data = pickle.load(pkl_file)
                data = pd.DataFrame.from_dict(pickle_data)

        # test / train split, keep % of grades equal in train and test.
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            data, data["numeric_grade"],
            test_size=self.test_size, stratify=data["grade"],
            random_state=self.random_state
        )

        # Reset indexes to allow dataloader to iterate correctly
        x_trainval.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
        y_trainval.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        if stage == "fit" or stage is None:
            data_trainval = BaseDataset(x_trainval, y_trainval)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=0.8, seed=42)

        if stage == "test" or stage is None:
            self.data_test = BaseDataset(x_test, y_test)

    def __repr__(self):
        basic = "MB2016 Dataset"
        return basic
        # f"\nNum grades: {len(self.grade_count)}\ngrade count: {self.grade_count}\n"
        # if self.data_train is None and self.data_test is None:
        #     return basic
        #
        # x, y = next(iter(self.train_dataloader()))
        # data = (
        #     f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_test)}\n"
        #     f"Batch x stats: {(x.shape, x.dtype)}\n"
        #     f"Batch y stats: {(y.shape, y.dtype)}\n"
        # )
        # return basic + data

    @property
    def processed_data_filename(self):
        return (
                metadata.PROCESSED_DATA_DIRNAME
                / f"minr_{self.min_repeats}_mingc{self.min_grade_count:f}_maxs{self.max_start_holds:f}"
                  f"minm_{self.min_mid_holds}_maxm{self.max_mid_holds:f}_maxe{self.max_end_holds:f}"
                  f"_tsize{self.test_size}_rand{self.random_state}_ptoken{self.with_start_mid_end_tokens}.pkl"
        )

    @property
    def raw_data_filename(self):
        return metadata.RAW_DATA_DIRNAME / metadata.RAW_DATA_FILENAME

    def _id_token_dict(self, position_tokens: bool):
        # Create token dictionary for MoonBoard hold positions
        rows = range(0, 11)
        columns = range(0, 18)
        positions = itertools.product(rows, columns)
        token_dict = {}
        if position_tokens:
            token_dict = {"S": 0, "SM": 1, "ME": 2, "E": 3, "P": 4}
        for i, k in enumerate(positions, start=1 if not position_tokens else 5):
            token_dict[k] = i

        return token_dict

    def _rel_token_dict(self, dim: int):
        # from tokens from min and max diffences of the relative hold positions
        return dict(zip(range(-dim + 1, dim), range(1, 2*dim)))


    def _extract_from_df(self, df):
        def _row_to_arrays(row):

            sequence_len = len(row["start"]) + len(row["mid"]) + len(row["end"])
            id_token_array = torch.zeros(self.max_sequence, dtype=torch.int32)  # Padding for token array
            order_token_array = torch.zeros(self.max_sequence, dtype=torch.int32)
            rel_x_token_array = torch.zeros((self.max_sequence, self.max_sequence), dtype=torch.int32)
            rel_y_token_array = torch.zeros((self.max_sequence, self.max_sequence), dtype=torch.int32)

            abs_xs = np.zeros(self.max_sequence, dtype=np.int32)
            abs_ys = np.zeros(self.max_sequence, dtype=np.int32)
            # relative_x_array = np.zeros((max_sequence, 2 * max_sequence -1), dtype=np.int32())
            # relative_y_array = np.zeros((max_sequence, 2 * max_sequence -1), dtype=np.int32())

            index = 0
            for xys, order_token in [(row["start"], 1), (row["mid"], 2), (row["end"], 3)]:
                for xy in xys:
                    id_token_array[index] = self.id_token_dict[tuple(xy)]
                    order_token_array[index] = order_token
                    abs_xs[index] = tuple(xy)[0]
                    abs_ys[index] = tuple(xy)[1]
                    index += 1

            for i in range(sequence_len):
                for j in range(sequence_len):
                    rel_x_token_array[i][j] = self.rel_x_token_dict[abs_xs[j] - abs_xs[i]]
                    rel_y_token_array[i][j] = self.rel_y_token_dict[abs_ys[j] - abs_ys[i]]


            return id_token_array, order_token_array, rel_x_token_array, rel_y_token_array

        output = df.apply(lambda x: _row_to_arrays(x), axis=1, result_type="expand")
        output.columns = ["id_tokens", "order_tokens", "rel_x_tokens", "rel_y_tokens"]
        return output


# def _grade_labels_to_one_hot_tensor(df_labels):
#     lb = LabelBinarizer()
#     torch_array = torch.FloatTensor(lb.fit_transform(df_labels))
#     return torch_array


def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


if __name__ == "__main__":
    load_and_print_info(MB2016)
