"""
The _Columns class is used to order and track the parameter names of the data
columns. This class is not intended to be used directly by the user.
It is used by the _Data class to provide an interface to datatypes that do not
have a column structure, such as numpy arrays.

Note
----

For the default back-end of _Data, this class is obsolete since pandas
DataFrames have a column structure. However, this class is intended to be a
uniform interface to data that does not have a column structure.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Dict, List, Optional

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Columns:
    def __init__(self, columns: Optional[Dict[str, None]] = None):
        """Class that keeps track of the names and order of parameters
         in the raw data.

        Parameters
        ----------
        columns: Dict[str, None], optional
            dictionary with names as column names and None as values
            , by default None

        Note
        ----
        The datatype of a dict with nonsensical values is used to prevent
         duplicate keys. This is because the dict is used as a set.
        """
        if columns is None:
            columns = {}

        self.columns: Dict[str, None] = columns

    def __repr__(self) -> str:
        """Representation of the _Columns object."""
        return self.columns.keys().__repr__()

    def __add__(self, __o: _Columns) -> _Columns:
        """Add two _Columns objects.

        Parameters
        ----------
        __o: _Columns
            _Columns object to add

        Returns
        -------
        _Columns
            _Columns object with the columns of both _Columns objects
        """
        return _Columns({**self.columns, **__o.columns})

    @property
    def names(self) -> List[str]:
        """List of the names of the columns.

        Returns
        -------
        List[str]
            list of the names of the columns
        """
        return list(self.columns.keys())

    def add(self, name: str):
        """Add a column to the _Columns object.

        Parameters
        ----------
        name: str
            name of the column to add
        """
        self.columns[name] = None

    def iloc(self, name: str | List[str]) -> List[int]:
        """Get the index of a column.

        Parameters
        ----------
        name: str | List[str]
            name of the column(s) to get the index of

        Returns
        -------
        List[int]
            list of the indices of the columns
        """
        if isinstance(name, str):
            name = [name]

        _indices = []
        for n in name:
            _indices.append(self.names.index(n))
        return _indices

    def rename(self, old_name: str, new_name: str):
        """Replace the name of a column.

        Parameters
        ----------
        old_name: str
            name of the column to replace
        new_name: str
            name of the column to replace with
        """
        self.columns[new_name] = self.columns.pop(old_name)
