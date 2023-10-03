#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# Local
from ..design.domain import Domain

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

    def __getitem__(self, index: int | slice | Iterable[int]) -> _Data:
        """Get a subset of the data.

        Parameters
        ----------
        index : int, slice, list
            The index of the data to get.

        Returns
        -------
            A subset of the data.
        """
        if isinstance(index, int):
            index = [index]
        return _Data(self.data.loc[index].copy())

    def __add__(self, other: _Data | Dict[str, Any]) -> _Data:
        """Add two Data objects together.

        Parameters
        ----------
        other : Data
            The Data object to add.

        Returns
        -------
            The sum of the two Data objects.
        """
        # If other is a dictionary, convert it to a _Data object
        if isinstance(other, Dict):
            other = _convert_dict_to_data(other)

        try:
            last_index = self.data.index[-1]
        except IndexError:  # Empty DataFrame
            return _Data(other.data.copy())  # Make a copy of other.data

        # Make a copy of other.data and modify its index
        other_data_copy = other.data.copy()
        other_data_copy.index = other_data_copy.index + last_index + 1
        return _Data(pd.concat([self.data, other_data_copy]))

    def __eq__(self, __o: _Data) -> bool:
        return self.data.equals(__o.data)

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
        df = pd.DataFrame(columns=domain.names).astype(
            domain._cast_types_dataframe()
        )

        # Set the categories tot the categorical parameters
        for name, categorical_input in domain.get_categorical_parameters().items():
            df[name] = pd.Categorical(
                df[name], categories=categorical_input.categories)

        return cls(df)

    @classmethod
    def from_file(cls, filename: Path | str) -> _Data:
        """Loads the data from a file.

        Parameters
        ----------
        filename : Path
            The filename to load the data from.
        """
        file = Path(filename).with_suffix('.csv')
        return cls(pd.read_csv(file, header=0, index_col=0))

    @classmethod
    def from_numpy(cls: Type[_Data], array: np.ndarray) -> _Data:
        """Loads the data from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The numpy array to load the data from.
        data_type : str
        """
        return cls(pd.DataFrame(array))

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
        """Export the _Data object to a numpy array.

        Returns
        -------
        np.ndarray
            numpy array with the data.
        """
        return self.data.to_numpy()

    def to_xarray(self, label: str) -> xr.DataArray:
        """Export the _Data object to a xarray DataArray.

        Parameters
        ----------
        label : str
            The name of the data.

        Returns
        -------
        xr.DataArray
            xarray DataArray with the data.
        """
        return xr.DataArray(self.data, dims=['iterations', label], coords={
            'iterations': range(len(self)), label: self.names})

    def to_dataframe(self) -> pd.DataFrame:
        """Export the _Data object to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            pandas dataframe with the data.
        """
        return self.data

    def combine_data_to_multiindex(self, other: _Data, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Combine the data to a multiindex dataframe.

        Parameters
        ----------
        other : _Data
            The other data to combine.
        jobs : pd.DataFrame
            The jobs dataframe.

        Returns
        -------
        pd.DataFrame
            The combined dataframe.

        Note
        ----
        This function is mainly used to show the combined ExperimentData object in a
        Jupyter Notebook
        """
        return pd.concat([jobs_df, self.data, other.data], axis=1, keys=['jobs', 'input', 'output'])

    def store(self, filename: Path) -> None:
        """Stores the data to a file.

        Parameters
        ----------
        filename : Path
            The filename to store the data to.
        """
        self.data.to_csv(filename.with_suffix('.csv'))

    def n_best_samples(self, nosamples: int, column_name: List[str] | str) -> pd.DataFrame:
        """Returns the n best samples. We consider to be lower values better.

        Parameters
        ----------
        nosamples : int
            The number of samples to return.
        column_name : List[str] | str
            The column name to sort on.

        Returns
        -------
        pd.DataFrame
            The n best samples.
        """
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

    def fill_numpy_arrays(self, array: np.ndarray) -> Iterable[int]:
        # get the indices of the nan values
        idx, _ = np.where(np.isnan(self.data))
        self.data.loc[np.unique(idx)] = array
        return np.unique(idx)

    def remove(self, indices: List[int]):
        self.data = self.data.drop(indices)

#                                                           Getters and setters
# =============================================================================

    def get_data_dict(self, index: int) -> Dict[str, Any]:
        return self.data.loc[index].to_dict()

    def set_data(self, index: int, value: Any, column: Optional[str] = None):
        # check if the index exists
        if index not in self.data.index:
            raise IndexError(f"Index {index} does not exist in the data.")

        if column is None:
            self.data.loc[index] = value
        else:
            try:
                self.data.at[index, column] = value
            except ValueError:
                self.data = self.data.astype(object)
                self.data.at[index, column] = value

    def reset_index(self) -> None:
        """Reset the index of the data."""
        self.data.reset_index(drop=True, inplace=True)

    def is_empty(self) -> bool:
        """Check if the data is empty."""
        return self.data.empty

    def has_columnnames(self, names: Iterable[str]) -> bool:
        return set(names).issubset(self.names)

    def set_columnnames(self, names: Iterable[str]) -> None:
        self.data.columns = names


def _convert_dict_to_data(dictionary: Dict[str, Any]) -> _Data:
    """Converts a dictionary with scalar values to a data object.

    Parameters
    ----------
    dict : Dict[str, Any]
        The dictionary to convert. Note that the dictionary should only have scalar values!

    Returns
    -------
    _Data
        The data object.
    """
    return _Data(pd.DataFrame(dictionary, index=[0]).copy())
