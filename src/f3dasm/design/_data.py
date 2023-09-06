#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import os
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Local
from .domain import Domain
from .parameter import Parameter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Data:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        if data is None:
            data = pd.DataFrame()

        self.data: pd.DataFrame = data

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            index = self.data.index[self.current_index]
            current_value = self.get_data_dict(index)
            self.current_index += 1
            return current_value

    def _repr_html_(self) -> str:
        return self.data._repr_html_()

#                                                                    Properties
# =============================================================================

    @property
    def indices(self) -> pd.Index:
        return self.data.index

    @property
    def names(self) -> List[str]:
        return self.data.columns.to_list()

#                                                      Alternative constructors
# =============================================================================

    @classmethod
    def from_indices(cls, indices: pd.Index) -> _Data:
        """Create a Data object from a list of indices.

        Parameters
        ----------
        indices : pd.Index
            The indices of the data.

        Returns
        -------
            Empty data object with indices
        """
        return cls(pd.DataFrame(index=indices))

    @classmethod
    def from_domain(cls, domain: Domain) -> _Data:
        """Create a Data object from a domain.

        Parameters
        ----------
        design
            _description_

        Returns
        -------
            _description_
        """
        df = pd.DataFrame(columns=list(domain.input_space.keys())).astype(
            domain._cast_types_dataframe()
        )

        # Set the categories tot the categorical parameters
        for name, categorical_input in domain.get_categorical_parameters().items():
            df[name] = pd.Categorical(
                df[name], categories=categorical_input.categories)

        return cls(df)

    @classmethod
    def from_file(cls, filename: Path, text_io: Optional[TextIOWrapper] = None) -> _Data:
        """Loads the data from a file.

        Parameters
        ----------
        filename : Path
            The filename to load the data from.

        text_io: TextIOWrapper, optional
            A text io object to load the data from.
        """
        # Load the data from a csv
        if text_io is None:
            file = filename.with_suffix('.csv')

        else:
            file = text_io

        return cls(pd.read_csv(file, header=0, index_col=0))

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> _Data:
        """Loads the data from a dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to load the data from.
        """
        return cls(dataframe)

    def reset(self, domain: Optional[Domain] = None):
        """Resets the data to the initial state.

        Parameters
        ----------
        domain : Domain, optional
            The domain of the experiment.

        Note
        ----
        If the domain is None, the data will be reset to an empty dataframe.
        """
        if domain is None:
            self.data = pd.DataFrame()
            return

        self.data = self.from_domain(domain).data

#                                                                        Export
# =============================================================================

    def to_numpy(self) -> np.ndarray:
        return self.data.to_numpy()

    def to_xarray(self, label: str) -> xr.DataArray:
        return xr.DataArray(self.data, dims=['iterations', label], coords={
            'iterations': range(len(self)), label: self.names})

    def combine_data_to_multiindex(self, other: _Data) -> pd.DataFrame:
        return pd.concat([self.data, other.data], axis=1, keys=['input', 'output'])

    def store(self, filename: Path, text_io: TextIOWrapper = None) -> None:
        """Stores the data to a file.

        Parameters
        ----------
        filename : Path
            The filename to store the data to.
        """

        if text_io is not None:
            self.data.to_csv(text_io)
            return

        self.data.to_csv(filename.with_suffix('.csv'))

    def n_best_samples(self, nosamples: int, column_name: List[str] | str) -> pd.DataFrame:
        return self.data.nsmallest(n=nosamples, columns=column_name)

#                                                        Append and remove data
# =============================================================================

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

    def add_empty_rows(self, number_of_rows: int):
        if self.data.index.empty:
            last_index = -1
        else:
            last_index = self.data.index[-1]

        new_indices = pd.RangeIndex(start=last_index + 1, stop=last_index + number_of_rows + 1, step=1)
        empty_data = pd.DataFrame(np.nan, index=new_indices, columns=self.data.columns)
        self.data = pd.concat([self.data, empty_data], ignore_index=False)

    def add_column(self, name: str):
        self.data[name] = np.nan

    def add_numpy_arrays(self, array: np.ndarray):
        df = pd.DataFrame(array,
                          columns=self.data.columns)
        self.add(df)

    def fill_numpy_arrays(self, array: np.ndarray):
        # get the indices of the nan values
        idx, _ = np.where(np.isnan(self.data))
        self.data.loc[np.unique(idx)] = array

    def remove(self, indices: List[int]):
        self.data = self.data.drop(indices)

#                                                           Getters and setters
# =============================================================================

    def select(self, indices: List[int]):
        self.data = self.data.loc[indices]

    def get_data_dict(self, index: int) -> Dict[str, Any]:
        return self.data.loc[index].to_dict()

    def set_data(self, index: int, value: Any, column: Optional[str] = None):
        # check if the index exists
        if index not in self.data.index:
            raise IndexError(f"Index {index} does not exist in the data.")

        if column is None:
            self.data.loc[index] = value
        else:
            self.data.at[index, column] = value
