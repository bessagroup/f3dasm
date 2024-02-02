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

from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type,
                    Union)

import numpy as np
import pandas as pd
import xarray as xr

from f3dasm._src.experimentdata._columns import _Columns
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


DataType = List[List[Any]]


class _Index:
    def __init__(self, indices: Optional[Iterable[int]] = None):
        if indices is None:
            indices = []

        self.indices: pd.Index = pd.Index(indices)

    def __add__(self, other: _Index) -> _Index:

        if self.is_empty():
            return _Index(other.indices.copy())

        return _Index(
            self.indices.append(other.indices + self.indices[-1] + 1))

    def __repr__(self) -> str:
        return self.indices.__repr__()

    def iloc(self, index: int | Iterable[int]) -> List[int]:

        if isinstance(index, int) or isinstance(index, np.int64):
            index = [index]

        _indices = []
        for n in index:
            _indices.append(self.indices.get_loc(n))
        return _indices

    def is_empty(self) -> bool:
        return self.indices.empty


class _Data:
    def __init__(self, data: Optional[DataType] = None,
                 columns: Optional[_Columns] = None,
                 index: Optional[_Index] = None):
        """
        Class for capturing input or output data in a tabular format

        Parameters
        ----------
        data : Optional[DataType], optional
            Data in tabular format, by default None
        columns : Optional[_Columns], optional
            _Columns object representing the names of the parameters,
            by default None
        index : Optional[_Index], optional
            _Index object representing the indices of the experiments,
            by default

        Note
        ----

        * The data is stored as a list of lists, where each list represents
        a row in the table.
        * The columns are stored as a dict with the
        column names as keys and None as values.
        * The index is stored as a list of integers in a pd.Index object.

        If no column names are given, the are by default number starting from
        zero. If no index is given, the indices are by default numbers starting
        from zero.
        """
        if data is None:
            data = []

        if columns is None:
            try:
                columns = _Columns(
                    {col: None for col, _ in enumerate(data[0])})
            except IndexError:
                columns = _Columns()

        if index is None:
            index = _Index(range(len(data)))

        self.columns = columns
        self.data = data
        self.index = index

    def __len__(self) -> int:
        """
        Calculate the number of rows in the data.

        Returns
        -------
        int
            the number of experiments
        """
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        """
        Iterator for the experiments in the data object

        Yields
        ------
        Iterator[Tuple[Dict[str, Any]]]
            Iterator object
        """
        self.current_index = 0
        return self

    def __next__(self):
        """
        Get the next experiment in the data object

        Returns
        -------
        Tuple[Dict[str, Any]]
            The next experiment

        Raises
        ------
        StopIteration
            If the last experiment has been reached
        """
        if self.current_index >= len(self):
            raise StopIteration
        else:
            current_value = self.data[self.current_index]
            self.current_index += 1
            return current_value

    def __getitem__(self, index: int | Iterable[int]) -> _Data:
        """
        Get the experiment(s) at the given index

        Parameters
        ----------
        index : int | Iterable[int]
            The index of the experiment(s) to get

        Returns
        -------
        _Data
            The experiment(s) at the given index
        """
        _index = self.index.iloc(index)
        if self.is_empty():
            return _Data(columns=self.columns, index=_Index(_index))

        else:
            return _Data(data=[self.data[i] for i in _index],
                         columns=self.columns, index=_Index(_index))

    def __add__(self, other: _Data | Dict[str, Any]) -> _Data:
        """
        Add two data objects together

        Parameters
        ----------
        other : _Data
            The data object to add

        Returns
        -------
        _Data
            The combined data object

        Note
        ----
        * The columns of the two data objects must be the same.
        * The indices of the second object are shifted by the number of
        experiments in the first object.
        """
        # If other is a dictionary, convert it to a _Data object
        if not isinstance(other, _Data):
            other = _convert_dict_to_data(other)

        return _Data(data=self.data + other.data,
                     columns=self.columns,
                     index=self.index + other.index)

    def __eq__(self, __o: _Data) -> bool:
        """
        Check if two data objects are equal

        Parameters
        ----------
        __o : _Data
            The data object to compare with

        Returns
        -------
        bool
            True if the data objects are equal, False otherwise

        Note
        ----
        The data objects will first be converted to a pandas DataFrame and
        then compared.
        """
        return self.to_dataframe().equals(__o.to_dataframe())

    def _repr_html_(self) -> str:
        """
        HTML representation of the data object

        Returns
        -------
        str
            HTML representation of the data object

        Note
        ----
        This method is used by Jupyter Notebook to display the data object.
        """
        return self.to_dataframe()._repr_html_()

#                                                                    Properties
# =============================================================================

    @property
    def names(self) -> List[str]:
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
        return self.columns.names

    @property
    def indices(self) -> pd.Index:
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
        return self.index.indices

#                                                      Alternative constructors
# =============================================================================

    @classmethod
    def from_list(cls: Type[_Data], list: List[List[Any]]) -> _Data:
        """Creates a data object from a list of lists.

        Parameters
        ----------
        list : List[List[Any]]
            The list of lists to create the data object from.

        Returns
        -------
        _Data
            The data object.
        """
        return _Data(data=list)

    @classmethod
    def from_indices(cls: Type[_Data], indices: pd.Index) -> _Data:
        """Creates a data object from a pd.Index object.

        Parameters
        ----------
        indices : pd.Index
            The indices of the experiments.

        Returns
        -------
        _Data
            The data object.

        Note
        ----
        The returned object will have no columns and the indices will be
        the given indices. The data will be an empty list.
        """
        return _Data(index=_Index(indices))

    @classmethod
    def from_domain(cls: Type[_Data], domain: Domain) -> _Data:
        """Creates a data object from a domain.

        Parameters
        ----------
        domain : Domain
            The domain to create the data object from.

        Returns
        -------
        _Data
            The data object.

        Note
        ----
        * The returned object will have no data and empty indices.
        * The columns will be the names of the provided domain.
        """
        _columns = {name: None for name in domain.names}
        return _Data(columns=_Columns(_columns))

    @classmethod
    def from_file(cls: Type[_Data], filename: Path | str) -> _Data:
        # TODO: Fix this method for _newdata
        """Loads the data from a file.

        Parameters
        ----------
        filename : Path
            The filename to load the data from.

        Returns
        -------
        _Data
            The loaded data object.
        """
        file = Path(filename).with_suffix('.csv')
        df = pd.read_csv(file, header=0, index_col=0)
        return cls.from_dataframe(df)

    @classmethod
    def from_numpy(cls: Type[_Data], array: np.ndarray) -> _Data:
        """Loads the data from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The array to load the data from.

        Returns
        -------
        _Data
            The data object.

        Note
        ----
        The returned _Data object will have the default column names and
        indices.
        """
        return cls.from_dataframe(pd.DataFrame(array))

    @classmethod
    def from_dataframe(cls: Type[_Data], df: pd.DataFrame) -> _Data:
        """Loads the data from a pandas dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to load the data from.

        Returns
        -------
        _Data
            The data object.

        Note
        ----
        The returned _Data object will have the column names of the dataframe
        and the indices of the dataframe.
        """
        _columns = {name: None for name in df.columns.to_list()}
        return _Data(data=df.to_numpy().tolist(),
                     columns=_Columns(_columns),
                     index=_Index(df.index))

    def reset(self, domain: Optional[Domain] = None):
        """Resets the data object.

        Parameters
        ----------
        domain : Optional[Domain], optional
            The domain to reset the data object to, by default None

        Note
        ----
        * If domain is None, the data object will be reset to an empty data
        object.
        * If a domain is provided, the data object will be reset to a data
        object with the given domain and no data.
        """
        if domain is None:
            self.data = []
            self.columns = _Columns()

        else:
            _reset_data = self.from_domain(domain)
            self.data = _reset_data.data
            self.columns = _reset_data.columns

        self.index = _Index()

#                                                                        Export
# =============================================================================

    def to_numpy(self) -> np.ndarray:
        """
        Convert the data to a numpy array

        Returns
        -------
        np.ndarray
            The data as a numpy array

        Note
        ----
        This method converts the data to a pandas DataFrame and then to a
        numpy array.
        """
        return self.to_dataframe().to_numpy()

    def to_xarray(self, label: str) -> Any:
        # TODO: THIS WILL NOT WORK IF DATA IS INHOMOGENEOUS!
        return xr.DataArray(self.to_dataframe(), dims=['iterations', label],
                            coords={'iterations': self.indices,
                                    label: self.names})

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
        return pd.DataFrame(self.data, columns=self.names, index=self.indices)

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

        _selected_columns = _Columns(
            {column: None for column in columns})

        return _Data(
            data=self.to_dataframe()[columns].values.tolist(),
            columns=_selected_columns, index=self.index)

#                                                        Append and remove data
# =============================================================================

    def add(self, data: pd.DataFrame):
        """Adds data to the data object.

        Parameters
        ----------
        data : pd.DataFrame
            The data to add.

        Note
        ----
        This method adds the dataframe in-place to the currentdata object.
        The data is added to the end of the data object.

        The + operator will do the same thing but return a new data object.s
        """
        _other = _Data.from_dataframe(data)

        self.data += _other.data
        self.index += _other.index

    def add_empty_rows(self, number_of_rows: int):
        """Adds empty rows to the data object.

        Parameters
        ----------
        number_of_rows : int
            The number of rows to add.

        Note
        ----
        This method adds empty rows to the data object. The value of these
        empty rows is np.nan. The columns are not changed.

        The rows are added to the end of the data object.
        """
        self.data += [[np.nan for _ in self.names]
                      for _ in range(number_of_rows)]
        self.index += _Index(range(number_of_rows))

    def add_column(self, name: str):
        """Adds a column to the data object.

        Parameters
        ----------
        name : str
            The name of the column to add.

        Note
        ----
        The values in the rows of this new column will be set to np.nan.
        """
        self.columns.add(name)

        if self.is_empty():
            self.data = [[np.nan] for _ in self.indices]

        else:
            for row in self.data:
                row.append(np.nan)

    def remove(self, indices: List[int] | int):
        """Removes rows from the data object.

        Parameters
        ----------
        indices : List[int] | int
            The indices of the rows to remove.
        """
        if isinstance(indices, int):
            indices = [indices]

        self.data = [row for i, row in enumerate(self.data)
                     if self.index.iloc(i)[0] not in indices]
        self.index = _Index([i for i in self.indices if i not in indices])

    def round(self, decimals: int):
        """Rounds the data.

        Parameters
        ----------
        decimals : int
            The number of decimals to round to.
        """
        self.data = [[round(value, decimals) for value in row]
                     for row in self.data]

    def overwrite(self, data: _Data, indices: Iterable[int]):
        # TODO: Implement this method!
        ...

#                                                           Getters and setters
# =============================================================================

    def get_data_dict(self, index: int) -> Dict[str, Any]:
        """
        Get the data as a dictionary.

        Parameters
        ----------
        index : int
            Index of the data to get.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the data.

        Note
        ----
        If the data is empty, an empty dictionary is returned.
        """
        if self.is_empty():
            return {}

        _index = self.index.iloc(index)[0]
        return {name: value for name, value in zip(self.names,
                                                   self.data[_index])}

    def set_data(self, index: int, value: Any, column: Optional[str] = None):
        """
        Set the data at the given index.

        Parameters
        ----------
        index : int
            Index of the data to set.
        value : Any
            Value to set.
        column : Optional[str], optional
            Column to set, by default None

        Raises
        ------
        IndexError
            If the index is not in the data.

        Note
        ----
        * If the column is not in the data, it will be added.
        * If the column is None, the value will be set to the whole row. Make
        sure that you provide a list with the same length as the number of
        columns in the data.

        """
        if index not in self.indices:
            raise IndexError(f'Index {index} not in data.')

        if column is None:
            _index = self.index.iloc(index)[0]
            self.data[_index] = value
            return

        elif column not in self.names:
            self.add_column(column)

        _column_index = self.columns.iloc(column)[0]
        _index = self.index.iloc(index)[0]
        self.data[_index][_column_index] = value

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

        self.index = _Index(indices)

    def is_empty(self) -> bool:
        """Checks if the data object is empty.

        Returns
        -------
        bool
            True if the data object is empty, False otherwise.
        """
        return len(self.data) == 0

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

        for old_name, new_name in zip(self.names, names):
            self.columns.rename(old_name, new_name)

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
    df = pd.DataFrame(dictionary, index=[0]).copy()
    return _Data.from_dataframe(df)


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
