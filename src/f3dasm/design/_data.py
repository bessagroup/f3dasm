#                                                                       Modules
# =============================================================================

# Standard
import os
from io import TextIOWrapper
from typing import Any, Dict, Iterator, List, Tuple, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from .domain import Domain
from .design import Design

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Data:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return self.number_of_datapoints()

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            index = self.data.index[self.current_index]
            current_value = [self.get_inputdata_dict(
                index), self.get_outputdata_dict(index)]
            self.current_index += 1
            return current_value

    def _repr_html_(self) -> str:
        return self.data._repr_html_()

    @classmethod
    def from_design(cls, design: Domain) -> '_Data':
        # input columns
        input_columns = [("input", name) for name, parameter in design.input_space.items()]

        df_input = pd.DataFrame(columns=input_columns).astype(
            design._cast_types_dataframe(design.input_space, label="input")
        )

        # Set the categories tot the categorical input parameters
        for name, categorical_input in design.get_categorical_input_parameters().items():
            df_input[('input', name)] = pd.Categorical(
                df_input[('input', name)], categories=categorical_input.categories)

        # output columns
        output_columns = [("output", name) for name, parameter in design.output_space.items()]

        df_output = pd.DataFrame(columns=output_columns).astype(
            design._cast_types_dataframe(design.output_space, label="output")
        )

        # Set the categories tot the categorical output parameters
        for name, categorical_output in design.get_categorical_output_parameters().items():
            df_output[('output', name)] = pd.Categorical(
                df_input[('output', name)], categories=categorical_output.categories)

        empty_dataframe = pd.concat([df_input, df_output])
        return cls(empty_dataframe)

    @classmethod
    def from_file(cls, filename: str, text_io: TextIOWrapper = None) -> '_Data':
        """Loads the data from a file.

        Parameters
        ----------
        filename : str
            The filename to load the data from.

        text_io: TextIOWrapper, optional
            A text io object to load the data from.
        """
        # Load the data from a csv
        if text_io is None:
            if not filename.endswith('.csv'):
                filename = filename + '.csv'
            file = filename
        else:
            file = text_io

        return cls(pd.read_csv(file, header=[0, 1], index_col=0))

    def reset(self, design: Domain):
        """Resets the data to the initial state.

        Parameters
        ----------
        design : DesignSpace
            The design space of the experiment.
        """
        self.data = self.from_design(design).data

    def store(self, filename: str, text_io: TextIOWrapper = None) -> None:
        """Stores the data to a file.

        Parameters
        ----------
        filename : str
            The filename to store the data to.
        """

        if text_io is not None:
            self.data.to_csv(text_io)
            return

        if not filename.endswith('.csv'):
            filename = filename + '.csv'
        # Store the data

        self.data.to_csv(filename)

    def select(self, indices: List[int]):
        self.data = self.data.loc[indices]

    def remove(self, indices: List[int]):
        self.data = self.data.drop(indices)

    def add(self, data: pd.DataFrame):
        try:
            last_index = self.data.index[-1]
        except IndexError:  # Empty dataframe
            self.data = data
            return

        new_indices = pd.RangeIndex(start=last_index + 1, stop=last_index + len(data) + 1, step=1)

        # set the indices of the data to new_indices
        data.index = new_indices

        self.data = pd.concat([self.data, data], ignore_index=False)

    def add_output(self, output: np.ndarray, label: str = "y"):
        self.data[("output", label)] = output

    def add_numpy_arrays(self, input: np.ndarray, output: Union[np.ndarray, None]):

        if output is None:
            output = np.nan * np.ones((input.shape[0], len(self.data['output'].columns)))

        df = pd.DataFrame(np.hstack((input, output)),
                          columns=self.data.columns)
        self.add(df)

    def get_inputdata(self) -> pd.DataFrame:
        return self.data['input']

    def get_outputdata(self) -> pd.DataFrame:
        return self.data['output']

    def get_inputdata_dict(self, index: int) -> Dict[str, Any]:
        return self.data['input'].loc[index].to_dict()

    def get_outputdata_dict(self, index: int) -> Dict[str, Any]:
        return self.data['output'].loc[index].to_dict()

    def get_design(self, index: int) -> Design:
        return Design(self.get_inputdata_dict(index), self.get_outputdata_dict(index), index)

    def set_design(self, design: Design) -> None:
        for column, value in design._dict_output.items():
            self.data.loc[design._jobnumber, ('output', column)] = value

    def set_inputdata(self, index: int, value: Any, column: str = 'input'):
        # check if the index exists
        if index not in self.data.index:
            raise IndexError(f"Index {index} does not exist in the data.")

        self.data.at[index, column] = value

    def set_outputdata(self, index: int, value: Any):
        # check if the index exists
        if index not in self.data.index:
            raise IndexError(f"Index {index} does not exist in the data.")

        self.data.at[index, 'output'] = value

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data['input'].to_numpy(), self.data['output'].to_numpy()

    def n_best_samples(self, nosamples: int, output_names: List[str]) -> pd.DataFrame:
        return self.data.nsmallest(n=nosamples, columns=[("output", name)
                                                         for name in output_names])

    def number_of_datapoints(self) -> int:
        return len(self.data)

    def plot(self, input_par1: str = "x0", input_par2: str = "x1") -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(self.data[("input", input_par1)],
                   self.data[("input", input_par2)], s=3)

        ax.set_xlabel(input_par1)
        ax.set_ylabel(input_par2)

        return fig, ax
