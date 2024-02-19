import itertools
import pickle
from typing import Optional

from grade_predictor.data.base_data_module import BaseDataModule, load_and_print_info
from grade_predictor.data.util import BaseDataset
import grade_predictor.metadata.transformer as metadata
from grade_predictor.util import temporary_working_directory
import natsort
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
        self.token_dict = self._create_hold_dictionary(self.with_start_mid_end_tokens)
        self.test_size = self.args.get("test_size", 0.2)
        self.random_state = self.args.get("random_state", 42)

        self.data_dir = self.args.get("data_dir", metadata.DATA_DIRNAME)
        self.processed_data_dir = self.args.get("data_dir", metadata.PROCESSED_DATA_DIRNAME)

        # self.transform = ClimbStem(self.with_start_mid_end_tokens)

        self.input_dims = (4, self.max_sequence)
        self.output_dims = (1,)

    def prepare_data(self) -> None:

        # check if data has already been prepared
        if self.processed_data_filename.exists():
            return

        # extract dataset to pandas dataframe
        with temporary_working_directory(self.data_dir):
            with open(self.raw_data_filename, "rb") as pkl_file:
                pickle_data = pickle.load(pkl_file)
                data = pd.DataFrame.from_dict(pickle_data).T

        # map grades to integers
        sorted_grades = natsort.natsorted(data["grade"].unique())
        grade_to_float = {}
        for i, value in enumerate(sorted_grades):
            grade_to_float[value] = np.float32(i)
        data['numeric_grade'] = data['grade'].map(grade_to_float)

        # Clean dataset of extraneous climbs
        data = data[data["repeats"] >= self.min_repeats]
        data = data[data.mid.map(len) <= self.max_mid_holds]
        data = data[data.mid.map(len) >= self.min_mid_holds]
        data = data[data.start.map(len) <= self.max_start_holds]
        data = data[data.end.map(len) <= self.max_end_holds]

        # self.grade_count = cleaned_data["grade"].value_counts()
        # self.grades = [key for key, value in self.grade_count.items() if value >= self.min_grade_count]
        # cleaned_data = cleaned_data[cleaned_data["grade"].isin(self.grades)]

        # Use index for IDs and reset_index
        data.reset_index(inplace=True)

        # Tokenize hold positions
        data["tokens_and_positions_array"] = self._dataframe_to_np_token_array(data,
                                                                                   self.max_sequence,
                                                                                   self.token_dict)

        # Save cleaned data
        with temporary_working_directory(self.processed_data_dir):
            data.to_pickle(self.processed_data_filename)

    def setup(self, stage: Optional[str] = None) -> None:

        with temporary_working_directory(self.processed_data_dir):
            with open(self.processed_data_filename, "rb") as pkl_file:
                pickle_data = pickle.load(pkl_file)
                data = pd.DataFrame.from_dict(pickle_data)

        # test / train split, keep % of grades equal in train and test.
        x = data["tokens_and_positions_array"]
        y = data["numeric_grade"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, stratify=data["grade"], random_state=self.random_state
        )

        # Reset indexes to allow dataloader to iterate correctly
        x_train.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        if stage == "train" or stage is None:
            self.data_train = BaseDataset(x_train, y_train)

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

    def _create_hold_dictionary(self, position_tokens: bool):
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

    def _dataframe_to_np_token_array(self, df, max_sequence, token_dict):
        def _row_to_token_matrix(row):
            token_matrix = np.zeros((4, max_sequence), dtype=np.int32())
            i = 0
            for position_column, position_index in [(row["start"], 1), (row["mid"], 2), (row["end"], 3)]:
                for item in position_column:
                    token_matrix[0][i] = token_dict[tuple(item)]
                    token_matrix[1][i] = position_index
                    token_matrix[2][i] = tuple(item)[0]
                    token_matrix[3][i] = tuple(item)[1]
                    i += 1
            return token_matrix

        token_array = df.apply(lambda x: _row_to_token_matrix(x), axis=1)
        return token_array


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
