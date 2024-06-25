#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

MISSING_VALUE = np.nan

# =============================================================================


class _Data:
    def __init__(self, data: Dict[int, Dict[str, Any]] = None):
        """
        Initialize the _Data object.

        Parameters
        ----------
        data : Dict[int, Dict[str, Any]], optional
            The data dictionary with integer keys and dictionaries as values.
        """
        self.data = data if data is not None else {}

    def __len__(self) -> int:
        """
        Get the number of items in the data.

        Returns
        -------
        int
            Number of items in the data.
        """
        return len(self.data)

    def __iter__(self):
        """
        Get an iterator over the data values.

        Returns
        -------
        iterator
            Iterator over the data values.
        """
        return iter(self.data.values())

    def __getitem__(self, rows: int | slice | Iterable[int]) -> _Data:
        """
        Get a subset of the data.

        Parameters
        ----------
        rows : int or slice or Iterable[int]
            The rows to retrieve.

        Returns
        -------
        _Data
            The subset of the data.
        """
        if isinstance(rows, int):
            rows = [rows]

        return _Data({row: self.data.get(row, {}) for row in rows})

    def __add__(self, __o: _Data) -> _Data:
        """
        Add another _Data object to this one.

        Parameters
        ----------
        __o : _Data
            The other _Data object.

        Returns
        -------
        _Data
            The combined _Data object.
        """
        if self.is_empty():
            return __o

        _data_copy = deepcopy(self)
        other_data_copy = deepcopy(__o)

        new_indices = (np.array(range(len(__o))) + max(self.data) + 1).tolist()

        _data_copy.data.update({row: values for row, values in zip(
            new_indices, other_data_copy.data.values())})
        return _data_copy

    def __eq__(self, __o: _Data) -> bool:
        """
        Check if another _Data object is equal to this one.

        Parameters
        ----------
        __o : _Data
            The other _Data object.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        return self.data == __o.data

    def _repr_html_(self) -> str:
        """
        Get the HTML representation of the data.

        Returns
        -------
        str
            The HTML representation of the data.
        """
        return self.to_dataframe()._repr_html_()

    def __repr__(self) -> str:
        """
        Get the string representation of the data.

        Returns
        -------
        str
            The string representation of the data.
        """
        return self.to_dataframe().__repr__()


#                                                                    Properties
# =============================================================================

    @property
    def indices(self) -> List[int]:
        """
        Get the indices of the data.

        Returns
        -------
        List[int]
            The list of indices.
        """
        return list(self.data.keys())

    @property
    def names(self) -> List[str]:
        """
        Get the column names of the data.

        Returns
        -------
        List[str]
            The list of column names.
        """
        return self.to_dataframe().columns.tolist()

    def is_empty(self) -> bool:
        """
        Check if the data is empty.

        Returns
        -------
        bool
            True if the data is empty, False otherwise.
        """
        return not bool(self.data)


#                                                                Initialization
# =============================================================================

    @classmethod
    def from_indices(cls, rows: Iterable[int]) -> _Data:
        """
        Create a _Data object from a list of indices.

        Parameters
        ----------
        rows : Iterable[int]
            The indices to create the _Data object from.

        Returns
        -------
        _Data
            The created _Data object.
        """
        return cls({row: {} for row in rows})

    @classmethod
    def from_file(cls, filename: Path) -> _Data:
        """
        Create a _Data object from a file.

        Parameters
        ----------
        filename : Path
            The file to read the data from.

        Returns
        -------
        _Data
            The created _Data object.
        """
        ...

    @classmethod
    def from_numpy(cls: Type[_Data], array: np.ndarray,
                   keys: Optional[Iterable[str]] = None) -> _Data:
        """
        Create a _Data object from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The numpy array to create the _Data object from.
        keys : Optional[Iterable[str]], optional
            The keys for the columns of the data.

        Returns
        -------
        _Data
            The created _Data object.
        """
        if keys is not None:
            return _Data(
                {index: {key: col for key, col in zip(keys, row)
                         } for index, row in enumerate(array)})
        else:
            return _Data(
                {index: {i: col for i, col in enumerate(row)
                         } for index, row in enumerate(array)})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> _Data:
        """
        Create a _Data object from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to create the _Data object from.

        Returns
        -------
        _Data
            The created _Data object.
        """
        return _Data(
            {index: row.to_dict() for index, (_, row) in
             enumerate(df.iterrows())})

#                                                                     Exporting
# =============================================================================

    def to_numpy(self) -> np.ndarray:
        """
        Convert the data to a numpy array.

        Returns
        -------
        np.ndarray
            The numpy array representation of the data.
        """
        return self.to_dataframe().to_numpy()

    def to_xarray(self, label: str):
        """
        Convert the data to an xarray DataArray.

        Parameters
        ----------
        label : str
            The label for the xarray DataArray.

        Returns
        -------
        xr.DataArray
            The xarray DataArray representation of the data.
        """
        df = self.to_dataframe()
        return xr.DataArray(
            self.to_dataframe(), dims=['iterations', label], coords={
                'iterations': df.index, label: df.columns})

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame representation of the data.
        """
        return pd.DataFrame(self.data).T

    def store(self, filename: Path):
        """
        Store the data to a file.

        Parameters
        ----------
        filename : Path
            The file to store the data in.
        """
        ...

    def get_data_dict(self, row: int) -> Dict[str, Any]:
        """
        Get the data dictionary for a specific row.

        Parameters
        ----------
        row : int
            The row to retrieve the data from.

        Returns
        -------
        Dict[str, Any]
            The data dictionary for the specified row.
        """
        return self.data[row]

#                                                       Selecting and combining
# =============================================================================

    def select_columns(self, keys: Iterable[str] | str) -> _Data:
        """
        Select specific columns from the data.

        Parameters
        ----------
        keys : Iterable[str] or str
            The keys of the columns to select.

        Returns
        -------
        _Data
            The _Data object with only the selected columns.
        """
        if isinstance(keys, str):
            keys = [keys]

        return _Data(
            {index: {key: row.get(key, MISSING_VALUE) for key in keys}
             for index, row in self.data.items()})

    def drop(self, keys: Iterable[str] | str) -> _Data:
        """
        Drop specific columns from the data.

        Parameters
        ----------
        keys : Iterable[str] or str
            The keys of the columns to drop.

        Returns
        -------
        _Data
            The _Data object with the specified columns removed.
        """
        if isinstance(keys, str):
            keys = [keys]

        for row in self:
            for key in keys:
                if key in row:
                    del row[key]

    def join(self, __o: _Data) -> _Data:
        """
        Join another _Data object with this one.

        Parameters
        ----------
        __o : _Data
            The other _Data object to join with this one.

        Returns
        -------
        _Data
            The combined _Data object.
        """
        _data = deepcopy(self)
        for row, other_row in zip(_data, __o):
            row.update(other_row)

        return _data

#                                                                     Modifying
# =============================================================================

    def n_best_samples(self, nosamples: int, key: str) -> pd.DataFrame:
        """
        Get the top N samples based on a specific key.

        Parameters
        ----------
        nosamples : int
            The number of samples to retrieve.
        key : str
            The key to sort the samples by.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the top N samples.
        """
        df = self.to_dataframe()
        return df.nsmallest(n=nosamples, columns=key)

    def add_column(self, key: str):
        """
        Add a new column to the data with missing values.

        Parameters
        ----------
        key : str
            The key for the new column.
        """
        for row in self.data:
            self.data[row][key] = MISSING_VALUE

    def remove(self, rows: Iterable[int]):
        """
        Remove specific rows from the data.

        Parameters
        ----------
        rows : Iterable[int]
            The rows to remove.
        """
        for row in rows:
            del self.data[row]

    def overwrite(self, rows: Iterable[int], __o: _Data):
        """
        Overwrite specific rows with data from another _Data object.

        Parameters
        ----------
        rows : Iterable[int]
            The rows to overwrite.
        __o : _Data
            The _Data object to overwrite the rows with.
        """
        for index, other_row in zip(rows, __o):
            self.data[index] = other_row

    def set_data(self, row: int, value: Any, key: str):
        """
        Set a specific value in the data.

        Parameters
        ----------
        row : int
            The row to set the value in.
        value : Any
            The value to set.
        key : str
            The key for the value.
        """
        self.data[row][key] = value

    def reset_index(self, rows: Iterable[int] = None):
        """
        Reset the index of the data.

        Parameters
        ----------
        rows : Iterable[int], optional
            The rows to reset the index for.

        """
        self.data = {index: values for index, values in enumerate(self)}

# =============================================================================


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

# =============================================================================


def _data_factory(data: DataTypes) -> _Data:
    if data is None:
        return _Data()

    elif isinstance(data, _Data):
        return data

    elif isinstance(data, pd.DataFrame):
        return _Data.from_dataframe(data)

    elif isinstance(data, (Path, str)):
        return _Data.from_file(Path(data))

    elif isinstance(data, np.ndarray):
        return _Data.from_numpy(data)

    else:
        raise TypeError(
            f"Data must be of type _Data, pd.DataFrame, np.ndarray, "
            f"Path or str, not {type(data)}")

# =============================================================================


DataTypes = Union[pd.DataFrame, np.ndarray, Path, str, _Data]
