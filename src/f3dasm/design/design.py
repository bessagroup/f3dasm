#                                                                       Modules
# =============================================================================

# Standard
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union

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


class Store(Protocol):
    def __call__(object, path: Path) -> str:
        ...


def numpy_store(object: np.ndarray, path: Path) -> str:
    np.save(file=path.with_suffix('.npy'), arr=object)
    return '.npy'


def pandas_store(object: pd.DataFrame, path: Path) -> str:
    object.to_csv(path.with_suffix('.csv'))
    return '.csv'


def xarray_store(object: Union[xr.DataArray, xr.Dataset], path: Path) -> str:
    object.to_netcdf(path.with_suffix('.nc'))
    return '.nc'


STORE_TYPE_MAPPING: Dict[Type, Store] = {
    np.ndarray: numpy_store,
    pd.DataFrame: pandas_store,
    xr.DataArray: xarray_store,
    xr.Dataset: xarray_store
}


def save_object(object: Any, path: Path, store_method: Union[None, Store]) -> str:
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
        self._dict_input = dict_input
        self._dict_output = dict_output
        self._jobnumber = jobnumber

    def __getitem__(self, item: str):
        return self._dict_input[item]

    def __setitem__(self, key: str, value: Any):
        self._dict_output[key] = value

    def __repr__(self) -> str:
        return f"Design({self.job_number} : {self.input_data} - {self.output_data})"

    @property
    def input_data(self) -> Dict[str, Any]:
        """Retrieve the input data of the design as a dictionary."""
        return self._dict_input

    @property
    def output_data(self) -> Dict[str, Any]:
        """Retrive the output data of the design as a dictionary."""
        return self._dict_output

    @property
    def job_number(self) -> int:
        """Retrieve the job number of the design."""
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

    def store(self, object: Any, name: str, store_method: Store = None) -> None:
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
