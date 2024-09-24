#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from copy import deepcopy
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type,
                    Union)

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# Local
from ..design.domain import Domain
from ._columns import _Columns

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Data:
    def __init__(self, data: Optional[pd.DataFrame] = None,
                 columns: Optional[_Columns] = None):
        if data is None:
            data = pd.DataFrame()

        if columns is None:
            columns = _Columns({col: None for col in data.columns})

        self.columns: _Columns = columns
        self.data = data.rename(
            columns={name: i for i, name in enumerate(data.columns)})

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

    def __getitem__(self, index: int | Iterable[int]) -> _Data:
        """Get a subset of the data.

        Parameters
        ----------
        index : int, list
            The index of the data to get.

        Returns
        -------
            A subset of the data.
        """
        # If the object is empty, return itself
        if self.is_empty():
            return self

        if isinstance(index, int):
            index = [index]
        return _Data(data=self.data.loc[index].copy(),
                     columns=self.columns)

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
            # Make a copy of other.data
            return _Data(data=other.data.copy(), columns=other.columns)

        # Make a copy of other.data and modify its index
        other_data_copy = other.data.copy()
        other_data_copy.index = other_data_copy.index + last_index + 1
        return _Data(pd.concat(
            [self.data, other_data_copy]), columns=self.columns)

    def __eq__(self, __o: _Data) -> bool:
        try:
            pd.testing.assert_frame_equal(self.data, __o.data)
        except AssertionError:
            return False

        return True

    def _repr_html_(self) -> str:
        return self.to_dataframe()._repr_html_()

#                                                                    Properties
# =============================================================================

    @property
    def indices(self) -> pd.Index:
        return self.data.index

    @property
    def names(self) -> List[str]:
        return self.columns.names

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
        _dtypes = {index: parameter._type for index,
                   (_, parameter) in enumerate(domain.space.items())}

        df = pd.DataFrame(columns=range(len(domain))).astype(_dtypes)

        # Set the categories tot the categorical parameters
        for index, (name, categorical_input) in enumerate(
                domain.categorical.space.items()):
            df[index] = pd.Categorical(
                df[index], categories=categorical_input.categories)

        _columns = {name: None for name in domain.names}
        return cls(df, columns=_Columns(_columns))

    @classmethod
    def from_file(cls, filename: Path | str) -> _Data:
        """Loads the data from a file.

        Parameters
        ----------
        filename : Path
            The filename to load the data from.
        """
        file = Path(filename).with_suffix('.csv')
        df = pd.read_csv(file, header=0, index_col=0)
        _columns = {name: False for name in df.columns.to_list()}
        # Reset the columns to be consistent
        df.columns = range(df.columns.size)
        return cls(df, columns=_Columns(_columns))

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
        _columns = {name: None for name in dataframe.columns.to_list()}
        return cls(dataframe, columns=_Columns(_columns))

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
            self.columns = _Columns()
        else:
            self.data = self.from_domain(domain).data
            self.columns = self.from_domain(domain).columns

#                                                                        Export
# =============================================================================

    def to_numpy(self) -> np.ndarray:
        """Export the _Data object to a numpy array.

        Returns
        -------
        np.ndarray
            numpy array with the data.
        """
        return self.data.to_numpy(dtype=np.float32)

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
            'iterations': self.indices, label: self.names})

    def to_dataframe(self) -> pd.DataFrame:
        """Export the _Data object to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            pandas dataframe with the data.
        """
        df = deepcopy(self.data)
        df.columns = self.names
        return df.astype(object)

    def combine_data_to_multiindex(self, other: _Data,
                                   jobs_df: pd.DataFrame) -> pd.DataFrame:
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
        # TODO: The column information is not saved in the .csv!
        self.to_dataframe().to_csv(filename.with_suffix('.csv'))

    def n_best_samples(self, nosamples: int,
                       column_name: List[str] | str) -> pd.DataFrame:
        """Returns the n best samples. We consider to be lower values better.

        Parameters
        ----------
        nosamples : int
            The number of samples to return.
        column_name : List[str] | str
            The column name(s) to sort on. If this is a list; priority will \
            be given on the first entry.

        Returns
        -------
        pd.DataFrame
            The n best samples.
        """
        return self.data.nsmallest(
            n=nosamples, columns=self.columns.iloc(column_name))

    def select_columns(self, columns: Iterable[str] | str) -> _Data:
        """Filter the data on the selected columns.

        Parameters
        ----------
        columns : Iterable[str] | str
            The columns to select.

        Returns
        -------
        _Data
            The data only with the selected columns
        """
        # This is necessary otherwise self.data[columns] will be a Series
        if isinstance(columns, str):
            columns = [columns]
        _selected_columns = _Columns(
            {column: self.columns.columns[column] for column in columns})
        return _Data(
            self.data[self.columns.iloc(columns)], columns=_selected_columns)

    def drop(self, columns: Iterable[str] | str) -> _Data:
        """Drop the selected columns from the data.

        Parameters
        ----------
        columns : Iterable[str] | str
            The columns to drop.

        Returns
        -------
        _Data
            The data without the selected columns
        """
        if isinstance(columns, str):
            columns = [columns]
        _selected_columns = _Columns(
            {
                name: None for name in self.columns.columns
                if name not in columns})
        return _Data(
            data=self.data.drop(columns=self.columns.iloc(columns)),
            columns=_selected_columns)

#                                                        Append and remove data
# =============================================================================

    def add(self, data: pd.DataFrame):
        try:
            last_index = self.data.index[-1]
        except IndexError:  # Empty dataframe
            self.data = data
            return

        new_indices = pd.RangeIndex(
            start=last_index + 1, stop=last_index + len(data) + 1, step=1)

        # set the indices of the data to new_indices
        data.index = new_indices

        self.data = pd.concat([self.data, data], ignore_index=False)

    def add_empty_rows(self, number_of_rows: int):
        if self.data.index.empty:
            last_index = -1
        else:
            last_index = self.data.index[-1]

        new_indices = pd.RangeIndex(
            start=last_index + 1, stop=last_index + number_of_rows + 1, step=1)
        empty_data = pd.DataFrame(
            np.nan, index=new_indices, columns=self.data.columns)
        self.data = pd.concat([self.data, empty_data], ignore_index=False)

    def add_column(self, name: str, exist_ok: bool = False):
        if name in self.columns.names:
            if not exist_ok:
                raise ValueError(
                    f"Column {name} already exists in the data. "
                    "Set exist_ok to True to allow skipping existing columns.")
            return

        if self.data.columns.empty:
            new_columns_index = 0
        else:
            new_columns_index = self.data.columns[-1] + 1

        self.columns.add(name)
        self.data[new_columns_index] = np.nan

    def remove(self, indices: List[int]):
        self.data = self.data.drop(indices)

    def round(self, decimals: int):
        self.data = self.data.round(decimals=decimals)

    def overwrite(self, indices: Iterable[int], other: _Data | Dict[str, Any]):
        if isinstance(other, Dict):
            other = _convert_dict_to_data(other)

        for other_column in other.columns.names:
            if other_column not in self.columns.names:
                self.add_column(other_column)

        self.data.update(other.data.set_index(pd.Index(indices)))

    def join(self, __o: _Data) -> _Data:
        """Join two Data objects together.

        Parameters
        ----------
        __o : Data
            The Data object to join.

        Returns
        -------
            The joined Data object.
        """
        return _Data(
            pd.concat([self.data, __o.data], axis=1, ignore_index=True),
            columns=self.columns + __o.columns)

#                                                           Getters and setters
# =============================================================================

    def get_data_dict(self, index: int) -> Dict[str, Any]:
        return self.to_dataframe().loc[index].to_dict()

    def set_data(self, index: int, value: Any, column: Optional[str] = None):
        # check if the index exists
        if index not in self.data.index:
            raise IndexError(f"Index {index} does not exist in the data.")

        if column is None:
            # Set the entire row to the values
            self.data.loc[index] = value
            return

        elif column not in self.columns.names:
            self.add_column(column)

        _column_index = self.columns.iloc(column)[0]
        try:
            self.data.at[index, _column_index] = value
        except ValueError:
            self.data = self.data.astype(object)
            self.data.at[index, _column_index] = value

    def reset_index(self, indices: Optional[Iterable[int]] = None) -> None:
        """Reset the index of the data.

        Parameters
        ----------
        indices : Optional[Iterable[int]], optional
            The indices to reset, by default None

        Note
        ----
        If indices is None, the entire index will be reset to a RangeIndex
        with the same length as the data.
        """
        if indices is None:
            self.data.reset_index(drop=True, inplace=True)
        else:
            self.data.index = indices

    def is_empty(self) -> bool:
        """Check if the data is empty."""
        return self.data.empty

    def get_index_with_nan(self) -> pd.Index:
        """Get the indices with NaN values.

        Returns
        -------
        pd.Index
            The indices with NaN values.
        """
        return self.indices[self.data.isna().any(axis=1)]

    def has_columnnames(self, names: Iterable[str]) -> bool:
        return set(names).issubset(self.names)

    def set_columnnames(self, names: Iterable[str]) -> None:
        for old_name, new_name in zip(self.names, names):
            self.columns.rename(old_name, new_name)

    def cast_types(self, domain: Domain):
        """Cast the types of the data to the types of the domain.

        Parameters
        ----------
        domain : Domain
            The domain with specific parameters to cast the types to.

        Raises
        ------
        ValueError
            If the types of the domain and the data do not match.
        """
        _dtypes = {index: parameter._type
                   for index, (_, parameter) in enumerate(
                       domain.space.items())}
        self.data = self.data.astype(_dtypes)


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
    _columns = {name: None for name in dictionary.keys()}
    df = pd.DataFrame(dictionary, index=[0]).copy()
    return _Data(data=df, columns=_Columns(_columns))


def _data_factory(data: DataTypes) -> _Data:
    if data is None:
        return _Data()

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
