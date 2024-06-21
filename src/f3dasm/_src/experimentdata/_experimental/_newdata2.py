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


class _Data:
    def __init__(self, data: Dict[int, Dict[str, Any]] = None):
        self.data = data if data is not None else {}

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data.values())

    def __getitem__(self, rows: int | slice | Iterable[int]) -> _Data:

        if isinstance(rows, int):
            rows = [rows]

        return _Data({row: self.data.get(row, {}) for row in rows})

    def __add__(self, __o: _Data) -> _Data:
        if self.is_empty():
            return __o

        _data_copy = deepcopy(self)
        other_data_copy = deepcopy(__o)

        new_indices = (np.array(range(len(__o))) + max(self.data) + 1).tolist()

        _data_copy.data.update({row: values for row, values in zip(
            new_indices, other_data_copy.data.values())})
        return _data_copy

    def __eq__(self, __o: _Data) -> bool:
        return self.data == __o.data

    def _repr_html_(self) -> str:
        return self.to_dataframe()._repr_html_()

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    @property
    def indices(self) -> List[int]:
        return list(self.data.keys())

    @property
    def names(self) -> List[str]:
        return self.to_dataframe().columns.tolist()

    @classmethod
    def from_indices(cls, rows: Iterable[int]):
        return cls({row: {} for row in rows})

    # @classmethod
    # def from_domain(cls, space: Iterable[str]):
    #     return cls(None)

    @classmethod
    def from_file(cls, filename: Path) -> _Data:
        ...

    @classmethod
    def from_numpy(cls: Type[_Data], array: np.ndarray,
                   keys: Optional[Iterable[str]] = None) -> _Data:
        if keys is not None:
            return _Data(
                {index: {key: col for key, col in zip(keys, row)
                         } for index, row in enumerate(array)})
        else:
            # Look out! i is now an integer key!
            return _Data(
                {index: {i: col for i, col in enumerate(row)
                         } for index, row in enumerate(array)})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> _Data:
        return _Data(
            {index: row.to_dict() for index, (_, row) in
             enumerate(df.iterrows())})

    def to_numpy(self) -> np.ndarray:
        return self.to_dataframe().to_numpy()

    def to_xarray(self, label: str):
        df = self.to_dataframe()
        # Can create the xarray with the information from the domain!
        return xr.DataArray(
            self.to_dataframe(), dims=['iterations', label], coords={
                'iterations': df.index, label: df.columns})

    def to_dataframe(self) -> pd.DataFrame:
        # Can create the dataframe from the numpy array + column names!!
        return pd.DataFrame(self.data).T

    def store(self, filename: Path):
        ...

    def n_best_samples(self, nosamples: int, key: str) -> _Data:
        df = self.to_dataframe()
        return df.nsmallest(
            n=nosamples, columns=key)

    def select_columns(self, keys: Iterable[str] | str) -> _Data:
        # This only works for single ints or slices!!

        if isinstance(keys, str):
            keys = [keys]

        return _Data(
            {index: {key: row.get(key, MISSING_VALUE) for key in keys}
             for index, row in self.data.items()})

    def drop(self, keys: Iterable[str] | str) -> _Data:
        # Might be depreciated?

        if isinstance(keys, str):
            keys = [keys]

        for row in self.data:
            for key in keys:
                if key in row:
                    del self.data[row][key]

    def add_column(self, key: str):
        for row in self.data:
            self.data[row][key] = MISSING_VALUE

    def remove(self, rows: Iterable[int]):
        for row in rows:
            del self.data[row]  # = deleting the row

    def overwrite(self, rows: Iterable[int], __o: _Data):
        for index, other_row in zip(rows, __o):
            self.data[index] = other_row

    def join(self, __o: _Data) -> _Data:
        _data = deepcopy(self)
        for row, other_row in zip(_data, __o):
            row.update(other_row)

        return _Data(_data)

    def get_data_dict(self, row: int) -> Dict[str, Any]:
        return self.data[row]

    def set_data(self, row: int, value: Any, key: str):
        self.data[row][key] = value

    def reset_index(self, rows: Iterable[int] = None):
        self.data = {index: values for index, values in enumerate(self.data)
                     }

    def is_empty(self) -> bool:
        return not bool(self.data)


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
    return _Data({0: {dictionary}})


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


DataTypes = Union[pd.DataFrame, np.ndarray, Path, str, _Data]
