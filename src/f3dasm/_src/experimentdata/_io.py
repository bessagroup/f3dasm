"""
Module to load and save output data of experiments \
and other common IO operations
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional, Type

# Third-party
import matplotlib.pyplot as plt
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

#                                                  Global folder and file names
# =============================================================================

EXPERIMENTDATA_SUBFOLDER = "experiment_data"

LOCK_FILENAME = "lock"
DOMAIN_FILENAME = "domain"
INPUT_DATA_FILENAME = "input"
OUTPUT_DATA_FILENAME = "output"
JOBS_FILENAME = "jobs"

RESOLUTION_MATPLOTLIB_FIGURE = 300
MAX_TRIES = 10

#                                                               Storing methods
# =============================================================================


class StoreProtocol:
    """Base class for storing and loading output data from disk"""
    suffix: int

    def __init__(self, object: Any, path: Path):
        """
        Protocol class for storing and loading output data from disk

        Parameters
        ----------
        object : Any
            object to store
        path : Path
            location to store the object to
        """
        self.path = path
        self.object = object

    def store(self) -> None:
        """
        Protocol class for storing objects to disk

        Raises
        ------
        NotImplementedError
            Raises if the method is not implemented
        """
        raise NotImplementedError()

    def load(self) -> Any:
        """
        Protocol class for loading objects to disk

        Returns
        -------
        Any
            The loaded object

        Raises
        ------
        NotImplementedError
            Raises if the method is not implemented
        """
        raise NotImplementedError()


class PickleStore(StoreProtocol):
    """Class to store and load objects using the pickle protocol"""
    suffix: str = '.pkl'

    def store(self) -> None:
        """
        Store the object to disk using the pickle protocol
        """
        with open(self.path.with_suffix(self.suffix), 'wb') as file:
            pickle.dump(self.object, file)

    def load(self) -> Any:
        """
        Load the object from disk using the pickle protocol

        Returns
        -------
        Any
            The loaded object

        """
        with open(self.path.with_suffix(self.suffix), 'rb') as file:
            return pickle.load(file)


class NumpyStore(StoreProtocol):
    """Class to store and load objects using the numpy protocol"""
    suffix: str = '.npy'

    def store(self) -> None:
        """
        Store the object to disk using the numpy protocol
        """
        np.save(file=self.path.with_suffix(self.suffix), arr=self.object)

    def load(self) -> np.ndarray:
        """
        Load the object from disk using the numpy protocol

        Returns
        -------
        np.ndarray
            The loaded object
        """
        return np.load(file=self.path.with_suffix(self.suffix))


class PandasStore(StoreProtocol):
    """Class to store and load objects using the pandas protocol"""
    suffix: str = '.csv'

    def store(self) -> None:
        """
        Store the object to disk using the pandas protocol
        """
        self.object.to_csv(self.path.with_suffix(self.suffix))

    def load(self) -> pd.DataFrame:
        """
        Load the object from disk using the pandas protocol

        Returns
        -------
        pd.DataFrame
            The loaded object
        """
        return pd.read_csv(self.path.with_suffix(self.suffix))


class XarrayStore(StoreProtocol):
    """Class to store and load objects using the xarray protocol"""
    suffix: str = '.nc'

    def store(self) -> None:
        """
        Store the object to disk using the xarray protocol
        """
        self.object.to_netcdf(self.path.with_suffix(self.suffix))

    def load(self) -> xr.DataArray | xr.Dataset:
        """
        Load the object from disk using the xarray protocol

        Returns
        -------
        xr.DataArray | xr.Dataset
            The loaded object
        """
        return xr.open_dataset(self.path.with_suffix(self.suffix))


class FigureStore(StoreProtocol):
    """Class to store and load objects using the matplotlib protocol"""
    suffix: str = '.png'

    def store(self) -> None:
        """
        Store the figure to disk as a png file

        Notes
        -----
        - The figure is saved with a resolution of 300 dpi.
        - The figure is saved with tight bounding boxes.
        """
        self.object.savefig(self.path.with_suffix(
            self.suffix), dpi=RESOLUTION_MATPLOTLIB_FIGURE,
            bbox_inches='tight')

    def load(self) -> np.ndarray:
        """
        Load the image as an numpy array from disk
        using the matplotlib `plt.imread` function.

        Returns
        -------
        np.ndarray
            The loaded image in the form of a numpy array

        Notes
        -----
         The returned array has shape
        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.

        Images are returned as float arrays (0-1).
        """
        return plt.imread(self.path.with_suffix(self.suffix))


STORE_TYPE_MAPPING: Mapping[Type, StoreProtocol] = {
    np.ndarray: NumpyStore,
    pd.DataFrame: PandasStore,
    pd.Series: PandasStore,
    xr.DataArray: XarrayStore,
    xr.Dataset: XarrayStore,
    plt.Figure: FigureStore,
}

#                                                  Loading and saving functions
# =============================================================================


def load_object(path: Path, experimentdata_directory: Path,
                store_method: Type[StoreProtocol] = PickleStore) -> Any:
    """
    Load an object from disk from a given path and storing method

    Parameters
    ----------
    path : Path
        path of the object to load
    experimentdata_directory : Path
        path of the f3dasm project directory
    store_method : Type[_Store], optional
        storage method protocol, by default PickleStore

    Returns
    -------
    Any
        the object loaded from disk

    Raises
    ------
    ValueError
        Raises if no matching store type is found

    Note
    ----
    If no store method is provided, the function will try to find a matching
    store type based on the suffix of the item's path. If no matching store
    type is found, the function will raise a ValueError. By default, the
    function will use the PickleStore protocol to load the object from disk.
    """

    _path = experimentdata_directory / path

    if store_method is not None:
        return store_method(None, _path).load()

    if not _path.exists():
        return None

    # Extract the suffix from the item's path
    item_suffix = _path.suffix

    # Use a generator expression to find the first matching store type,
    #  or None if no match is found
    matched_store_type: StoreProtocol = next(
        (store_type for store_type in STORE_TYPE_MAPPING.values() if
         store_type.suffix == item_suffix), PickleStore)

    if matched_store_type:
        return matched_store_type(None, _path).load()
    else:
        # Handle the case when no matching suffix is found
        raise ValueError(
            f"No matching store type for item type: '{item_suffix}'")


def save_object(object: Any, path: Path, experimentdata_directory: Path,
                store_method: Optional[Type[StoreProtocol]] = None) -> str:
    """Function to save the object to path,
     with the appropriate storing method.

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
        Raises if the object type is not supported,
         and you haven't provided a custom store method.
    """
    _path = experimentdata_directory / path

    if store_method is not None:
        storage = store_method(object, _path)
        return

    # Check if object type is supported
    object_type = type(object)

    if object_type not in STORE_TYPE_MAPPING:
        storage: StoreProtocol = PickleStore(object, _path)
        logger.debug(f"Object type {object_type} is not natively supported. "
                     f"The default pickle storage method will be used.")

    else:
        storage: StoreProtocol = STORE_TYPE_MAPPING[object_type](object, _path)
    # Store object
    storage.store()
    return storage.suffix


def _project_dir_factory(project_dir: Path | str | None) -> Path:
    """Creates a Path object for the project directory from a particular input

    Parameters
    ----------
    project_dir : Path | str | None
        path of the user-defined directory where to create the f3dasm project \
        folder.

    Returns
    -------
    Path
        Path object
    """
    if isinstance(project_dir, Path):
        return project_dir.absolute()

    if project_dir is None:
        return Path().cwd()

    if isinstance(project_dir, str):
        return Path(project_dir).absolute()

    raise TypeError(
        f"project_dir must be of type Path, str or None, \
            not {type(project_dir).__name__}")
