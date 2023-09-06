"""
A Design object contains a single realization of the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# Local
from ..logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#                                                               Storing to disk
# =============================================================================


class _Store(Protocol):
    """Protocol class for storing objects to disk."""
    def __call__(object: Any, path: Path) -> str:
        """Protocol method for storing objects to disk.

        Parameters
        ----------
        object : Any
            object to store to disk
        path : Path
            Path to store the object to

        Returns
        -------
        str
            suffix of the file in string format
        """
        ...


def numpy_store(object: np.ndarray, path: Path) -> str:
    """Numpy store method.

    Parameters
    ----------
    object : np.ndarray
        numpy-array to store
    path : Path
        Path to store the object to

    Returns
    -------
    str
        '.npy' suffix
    """
    np.save(file=path.with_suffix('.npy'), arr=object)
    return '.npy'


def pandas_store(object: pd.DataFrame, path: Path) -> str:
    """Pandas DataFrame store method.

    Parameters
    ----------
    object : pd.DataFrame
        pandas DataFrame to store
    path : Path
        Path to store the object to

    Returns
    -------
    str
        '.csv' suffix (comma-separated values file)
    """
    object.to_csv(path.with_suffix('.csv'))
    return '.csv'


def xarray_store(object: xr.DataArray | xr.Dataset, path: Path) -> str:
    """Xarray store method.

    Parameters
    ----------
    object : xr.DataArray | xr.Dataset
        xarray object to store; either a DataArray or a Dataset
    path : Path
        Path to store the object to

    Returns
    -------
    str
        '.nc' suffix (NetCDF4 file)
    """
    object.to_netcdf(path.with_suffix('.nc'))
    return '.nc'


STORE_TYPE_MAPPING: Dict[Type, _Store] = {
    np.ndarray: numpy_store,
    pd.DataFrame: pandas_store,
    xr.DataArray: xarray_store,
    xr.Dataset: xarray_store
}


def save_object(object: Any, path: Path, store_method: Optional[_Store] = None) -> str:
    """Function to save the object to path, with the appropriate storing method.

    Parameters
    ----------
    object : Any
        Object to store
    path : Path
        Path to store the object to
    store_method : Optional[Store], optional
        Storage method, by default None

    Returns
    -------
    str
        suffix of the storage method

    Raises
    ------
    TypeError
        Raises if the object type is not supported, and you haven't provided a custom store method.
    """
    if store_method is not None:
        store_method(object, path)
        return

    # Check if object type is supported
    object_type = type(object)
    if object_type not in STORE_TYPE_MAPPING:
        raise TypeError(f"Object type {object_type} is not natively supported. "
                        f"You can provide a custom store method to save other object types.")

    # Store object
    suffix = STORE_TYPE_MAPPING[object_type](object, path)
    return suffix

#                                                                        Design
# =============================================================================


class Design:
    def __init__(self, dict_input: Dict[str, Any], dict_output: Dict[str, Any], jobnumber: int):
        """Single realization of a design of experiments.

        Parameters
        ----------
        dict_input : Dict[str, Any]
            Input parameters of one experiment
        dict_output : Dict[str, Any]
            Output parameters of one experiment
        jobnumber : int
            Index of the experiment
        """
        self._dict_input = dict_input
        self._dict_output = dict_output
        self._jobnumber = jobnumber

    @classmethod
    def from_numpy(cls: Type[Design], input_array: np.ndarray,
                   output_value: Optional[float] = None, jobnumber: int = 0) -> Design:
        """Create a Design object from a numpy array.

        Parameters
        ----------
        input_array : np.ndarray
            input 1D numpy array        output_value : Optional[float], optional
            objective value, by default None

        jobnumber : int
            jobnumber of the design

        Returns
        -------
        Design
            Design object
        """
        dict_input = {f"x{i}": val for i, val in enumerate(input_array)}
        if output_value is None:
            dict_output = {}
        else:
            dict_output = {"y": output_value}

        return cls(dict_input=dict_input, dict_output=dict_output, jobnumber=jobnumber)

    def __getitem__(self, item: str):
        return self._dict_input[item]

    def __setitem__(self, key: str, value: Any):
        self._dict_output[key] = value

    def __repr__(self) -> str:
        return f"Design({self.job_number} : {self.input_data} - {self.output_data})"

    @property
    def input_data(self) -> Dict[str, Any]:
        """Retrieve the input data of the design as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The input data of the design as a dictionary.
        """
        return self._dict_input

    @property
    def output_data(self) -> Dict[str, Any]:
        """Retrieve the output data of the design as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The output data of the design as a dictionary.
        """
        return self._dict_output

    @property
    def job_number(self) -> int:
        """Retrieve the job number of the design.

        Returns
        -------
        int
            The job number of the design.
        """
        return self._jobnumber

#                                                                        Export
# =============================================================================

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the design to a tuple of numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of numpy arrays containing the input and output data.
        """
        return np.array(list(self._dict_input.values())), np.array(list(self._dict_output.values()))

    def store(self, object: Any, name: str, store_method: _Store = None) -> None:
        """Store an object to disk.

        Parameters
        ----------

        object : Any
            The object to store.
        name : str
            The name of the file to store the object in.
        store_method : Store, optional
            The method to use to store the object, by default None

        Raises
        ------

        TypeError
            If the object type is not supported and no store_method is provided.
        """
        file_dir = Path().cwd() / name
        file_path = file_dir / str(self.job_number)

        # Check if the file_dir exists
        file_dir.mkdir(parents=True, exist_ok=True)

        # Save the object to disk
        suffix = save_object(object=object, path=file_dir/str(self.job_number), store_method=store_method)

        # Store the path to the object in the output_data
        self._dict_output[f"{name}_path"] = str(file_path.with_suffix(suffix))

        logger.info(f"Stored {name} to {file_path.with_suffix(suffix)}")
