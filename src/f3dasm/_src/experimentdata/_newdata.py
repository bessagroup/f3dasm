"""
Module that contains the underlying data structure for ordering
input and output data of experiments.

Note
----
The data is stored as a list of lists, where each list represents
this is a work in progress and not yet implemented, that's why the name is
_newdata.py and the module is not imported in the __init__.py file.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple,
                    Type, Union)

import numpy as np
import pandas as pd
import xarray as xr

from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'


# =============================================================================
#
# =============================================================================


def dict_to_defaultdict(d: Dict[Any, Any] | Any) -> defaultdict:
    if isinstance(d, dict):
        return defaultdict(dict, {k: dict_to_defaultdict(v)
                                  for k, v in d.items()})
    else:
        # If it's not a dict, just return the value as is
        return d


class _Data:
    def __init__(self, data: Optional[dict] = None):
        if data is None:
            data = defaultdict(dict)

        self.data = dict_to_defaultdict(data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            current_value = self.data[self.current_index]
            self.current_index += 1
            return current_value

    def __getitem__(self, index: int | Iterable[int]) -> _Data:
        if isinstance(index, int):
            index = [index]
        return _Data({i: self.data[i] for i in index})

    def __add__(self, other: _Data | Dict[str, Any]) -> _Data:
        # Find the last key in my_dict
        last_key = max(self.data.keys()) if self.data else -1

        # Update keys of other dict
        other_updated_data = {
            last_key + 1 + k: v for k, v in other.data.items()}

        return _Data(
            {k: v for d in (self.data, other_updated_data)
             for k, v in d.items()})

    def __eq__(self, __o: _Data) -> bool:
        return self.data == __o.data

    def _repr_html_(self) -> str:
        return self.data.__repr__()

    @property
    def names(self) -> Set[str]:
        """
        Names of the columns of the data

        Returns
        -------
        List[str]
            Names of the columns of the data

        Note
        ----
        This is a shortcut for self.columns.names, accessing the private
        object.
        """
        return {key for inner_dict in self.data.values()
                for key in inner_dict.keys()}

    @property
    def indices(self) -> List[int]:
        """
        Indices of the experiments in the data

        Returns
        -------
        pd.Index
            Indices of the experiments in the data

        Note
        ----
        This is a shortcut for self.index.indices, accessing the private
        object.
        """
        return list(self.data.keys())

    @classmethod
    def from_list(cls: Type[_Data], list: List[List[Any]]) -> _Data:
        return _Data(
            {i: {k: v for k, v in enumerate(row)}
             for i, row in enumerate(list)})

    # Is this necessary?
    @classmethod
    def from_indices(cls: Type[_Data], indices: pd.Index) -> _Data:
        return _Data()

    # Is this necessary?
    @classmethod
    def from_domain(cls: Type[_Data], domain: Domain) -> _Data:
        return _Data()

    @classmethod
    def from_file(cls: Type[_Data], filename: Path | str) -> _Data:
        ...

    @classmethod
    def from_numpy(cls: Type[_Data], array: np.ndarray) -> _Data:
        return _Data.from_list(array.tolist())

    @classmethod
    def from_dataframe(cls: Type[_Data], df: pd.DataFrame) -> _Data:
        return _Data(df.to_dict())

    def reset(self, domain: Optional[Domain] = None):
        self.data = defaultdict(dict)

    def to_numpy(self) -> np.ndarray:
        return self.to_dataframe().to_numpy()

    def to_xarray(self, label: str) -> Any:
        # TODO: THIS WILL NOT WORK IF DATA IS INHOMOGENEOUS!
        df = self.to_dataframe()
        return xr.DataArray(df, dims=['iterations', label],
                            coords={'iterations': df.index,
                                    label: df.columns})

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The data as a pandas DataFrame

        Note
        ----
        The resulting dataframe has the indices as rows and the columns as
        column names.
        """
        return pd.DataFrame(self.data, columns=self.indices).T

    def combine_data_to_multiindex(self, other: _Data,
                                   jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the data to a multiindex dataframe.

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
        This function is mainly used to show the combined ExperimentData
        object in a Jupyter Notebook
        """
        return pd.concat([jobs_df, self.to_dataframe(),
                          other.to_dataframe()],
                         axis=1, keys=['jobs', 'input', 'output'])

    def store(self, filename: Path) -> None:
        """Stores the data to a file.

        Parameters
        ----------
        filename : Path
            The filename to store the data to.

        Note
        ----
        The data is stored as a csv file.
        """
        # TODO: Test this function!
        self.to_dataframe().to_csv(filename.with_suffix('.csv'))

    def n_best_samples(self, nosamples: int,
                       column_name: List[str] | str) -> pd.DataFrame:
        return self.to_dataframe().nsmallest(n=nosamples,
                                             columns=column_name)

    def select_columns(self, columns: Iterable[str] | str) -> _Data:
        """Selects columns from the data.

        Parameters
        ----------
        columns : Iterable[str] | str
            The column(s) to select.

        Returns
        -------
        _Data
            The data with the selected columns.

        Note
        ----
        This method returns a new data object with the selected columns.
        """
        if isinstance(columns, str):
            columns = [columns]

        return _Data(
            {i: {k: v for k, v in inner_dict.items()
                 if k in columns}
             for i, inner_dict in self.data.items()})

#                                                        Append and remove data
# =============================================================================

    def add(self, data: pd.DataFrame):
        ...

    # This is not necessary with defaultdict
    def add_empty_rows(self, number_of_rows: int):
        ...

    def add_column(self, name: str):
        ...

    def remove(self, indices: List[int] | int):
        """Removes rows from the data object.

        Parameters
        ----------
        indices : List[int] | int
            The indices of the rows to remove.
        """
        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            del self.data[index]

    def round(self, decimals: int):
        ...

    def overwrite(self, data: _Data, indices: Iterable[int]):
        # TODO: Implement this method!
        ...


#                                                           Getters and setters
# =============================================================================


    def get_data_dict(self, index: int) -> Dict[str, Any]:
        return dict(self.data[index])

    def set_data(self, index: int, value: Any, column: Optional[str] = None):
        self.data[index][column] = value

    def reset_index(self, indices: Optional[Iterable[int]] = None):
        """Resets the index of the data object.

        Parameters
        ----------
        indices : Optional[Iterable[int]], optional
            The indices to reset the index to, by default None

        Note
        ----
        This method resets the index of the data object.

        If no indices are provided, the index will be
        reset to range(len(data)). This means that the index will be
        [0, 1, 2, ..., len(data) - 1].
        """
        if indices is None:
            indices = range(len(self.data))

        self.data = {
            new_index: self.data[old_index]
            for new_index, old_index in zip(indices, self.data.keys())}

    def is_empty(self) -> bool:
        """Checks if the data object is empty.

        Returns
        -------
        bool
            True if the data object is empty, False otherwise.
        """
        return not self.data

    def has_columnnames(self, names: Iterable[str]) -> bool:
        """Checks if the data object has the given column names.

        Parameters
        ----------
        names : Iterable[str]
            The column names to check.

        Returns
        -------
        bool
            True if the data object has the given column names, False
            otherwise.
        """
        return set(names).issubset(self.names)

    def set_columnnames(self, names: Iterable[str]):
        """Sets the column names of the data object.

        Parameters
        ----------
        names : Iterable[str]
            The column names to set.

        Note
        ----
        This method overwrite the column names of the data object.
        The number of names should be equal to the number of columns in the
        data object.
        """
        ...

    def cast_types(self, domain: Domain):
        pass


def _convert_dict_to_data(dictionary: Dict[str, Any]) -> _Data:
    """Converts a dictionary with scalar values to a data object.

    Parameters
    ----------
    dict : Dict[str, Any]
        The dictionary to convert. Note that the dictionary
         should only have scalar values!

    Returns
    -------
    _Data
        The data object.
    """
    return _Data({0: dictionary})


def _data_factory(data: DataTypes) -> _Data:
    if data is None:
        return _Data()

    elif isinstance(data, list):
        return _Data.from_list(data)

    elif isinstance(data, _Data):
        return data

    elif isinstance(data, pd.DataFrame):
        return _Data.from_dataframe(data)

    elif isinstance(data, (Path, str)):
        return _Data.from_file(data)

    elif isinstance(data, np.ndarray):
        return _Data.from_numpy(data)

    else:
        raise TypeError(
            f"Data must be of type _Data, pd.DataFrame, np.ndarray, "
            f"Path or str, not {type(data)}")


DataTypes = Union[pd.DataFrame, np.ndarray, Path, str, _Data]
