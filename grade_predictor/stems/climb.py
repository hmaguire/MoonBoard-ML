# import itertools
# import torch
#
#
# class ClimbStem:
#
#     """A climb's data is presumed to be provided in the format of the 2016 MoonBoard MoonGen Project Dataset converted into
#     a pandas DataFrame with an index of start, mid, end and grade which describe the hold positions and the assigned
#     difficulty grade.
#
#     The data is mapped to a torch tensor in the following format of size (4, max_sequence length)
#     - token ids for each hold for start, mid and end concatenated.
#     - position index for each token
#     - x position of each token
#     - y position for each token
#
#     Padding is added to fit the max_sequence dimension requirement.
#
#     """
#
#     def __init__(self, position_tokens=False):
#         self.token_dict = self.create_hold_dictionary(position_tokens)
#
#     @staticmethod
#     def create_hold_dictionary(position_tokens: bool):
#         # Create token dictionary for MoonBoard hold positions
#         rows = range(0, 11)
#         columns = range(0, 18)
#         positions = itertools.product(rows, columns)
#         token_dict = {}
#         if position_tokens:
#             token_dict = {"S": 0, "SM": 1, "ME": 2, "E": 3, "P": 4}
#         for i, k in enumerate(positions, start=0 if not position_tokens else 5):
#             token_dict[k] = i
#         return token_dict
#
#     def __call__(self, df, max_sequence):
#         def _row_to_token_matrix(row):
#             token_matrix = torch.zeros((4, max_sequence), dtype=torch.int)
#             i = 0
#             for position_column, position_index in [(row["start"], 1), (row["mid"], 2), (row["end"], 3)]:
#                 for item in position_column:
#                     token_matrix[0][i] = self.token_dict[tuple(item)]
#                     token_matrix[1][i] = position_index
#                     token_matrix[2][i] = tuple(item)[0]
#                     token_matrix[3][i] = tuple(item)[1]
#                     i += 1
#             return token_matrix
#
#         token_array = df.apply(lambda x: _row_to_token_matrix(x), axis=1)
#         return token_array
