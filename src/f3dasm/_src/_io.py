"""
Module to load and save output data of experiments and other common IO
operations.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Type

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Local
from .logger import logger

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
MAX_TRIES = 20

#                                                               Storing methods
# =============================================================================


def pickle_store(object: Any, path: str) -> str:
    """
    Store an object using pickle.

    Parameters
    ----------
    object : Any
        The object to store.
    path : str
        The path where the object will be stored.

    Returns
    -------
    str
        The path to the stored object.
    """
    _path = Path(path).with_suffix('.pkl')
    with open(_path, 'wb') as file:
        pickle.dump(object, file)

    return str(_path)


def pickle_load(path: str) -> Any:
    """
    Load an object using pickle.

    Parameters
    ----------
    path : str
        The path to the object to load.

    Returns
    -------
    Any
        The loaded object.
    """
    _path = Path(path).with_suffix('.pkl')
    with open(_path, 'rb') as file:
        return pickle.load(file)


def numpy_store(object: np.ndarray, path: str) -> str:
    """
    Store a numpy array.

    Parameters
    ----------
    object : np.ndarray
        The numpy array to store.
    path : str
        The path where the array will be stored.

    Returns
    -------
    str
        The path to the stored array.
    """
    _path = Path(path).with_suffix('.npy')
    np.save(file=_path, arr=object)
    return str(_path)


def numpy_load(path: str) -> np.ndarray:
    """
    Load a numpy array.

    Parameters
    ----------
    path : str
        The path to the array to load.

    Returns
    -------
    np.ndarray
        The loaded array.
    """
    _path = Path(path).with_suffix('.npy')
    return np.load(file=_path)


def pandas_store(object: pd.DataFrame, path: str) -> str:
    """
    Store a pandas DataFrame.

    Parameters
    ----------
    object : pd.DataFrame
        The DataFrame to store.
    path : str
        The path where the DataFrame will be stored.

    Returns
    -------
    str
        The path to the stored DataFrame.
    """
    _path = Path(path).with_suffix('.csv')
    object.to_csv(_path)
    return str(_path)


def pandas_load(path: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame.

    Parameters
    ----------
    path : str
        The path to the DataFrame to load.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    _path = Path(path).with_suffix('.csv')
    return pd.read_csv(_path, index_col=0, header=0)


def xarray_dataset_store(object: xr.DataArray | xr.Dataset, path: str) -> str:
    """
    Store an xarray DataArray or Dataset.

    Parameters
    ----------
    object : xr.DataArray or xr.Dataset
        The xarray object to store.
    path : str
        The path where the object will be stored.

    Returns
    -------
    str
        The path to the stored object.
    """
    _path = Path(path).with_suffix('.ncs')
    object.to_netcdf(_path)
    return str(_path)


def xarray_dataarray_store(object: xr.DataArray | xr.Dataset, path: str
                           ) -> str:
    """
    Store an xarray DataArray or Dataset.

    Parameters
    ----------
    object : xr.DataArray or xr.Dataset
        The xarray object to store.
    path : str
        The path where the object will be stored.

    Returns
    -------
    str
        The path to the stored object.
    """
    _path = Path(path).with_suffix('.nca')
    object.to_netcdf(_path)
    return str(_path)


def xarray_dataset_load(path: str) -> xr.DataArray | xr.Dataset:
    """
    Load an xarray Dataset.

    Parameters
    ----------
    path : str
        The path to the Dataset to load.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The loaded Dataset.
    """
    # TODO: open_dataset and open_dataarray?
    _path = Path(path).with_suffix('.ncs')
    return xr.open_dataset(_path)


def xarray_dataarray_load(path: str) -> xr.DataArray | xr.Dataset:
    """
    Load an xarray DataArray.

    Parameters
    ----------
    path : str
        The path to the DataArray to load.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The loaded DataArray.
    """
    # TODO: open_dataset and open_dataarray?
    _path = Path(path).with_suffix('.nca')
    return xr.open_dataarray(_path)


def figure_store(object: plt.Figure, path: str) -> str:
    """
    Store a matplotlib figure.

    Parameters
    ----------
    object : plt.Figure
        The figure to store.
    path : str
        The path where the figure will be stored.

    Returns
    -------
    str
        The path to the stored figure.
    """
    _path = Path(path).with_suffix('.png')
    object.savefig(_path, dpi=RESOLUTION_MATPLOTLIB_FIGURE,
                   bbox_inches='tight')
    return str(_path)


def figure_load(path: str) -> np.ndarray:
    """
    Load a matplotlib figure.

    Parameters
    ----------
    path : str
        The path to the figure to load.

    Returns
    -------
    np.ndarray
        The loaded figure.
    """
    _path = Path(path).with_suffix('.png')
    return plt.imread(_path)


STORE_FUNCTION_MAPPING: Mapping[Type, Callable] = {
    np.ndarray: numpy_store,
    pd.DataFrame: pandas_store,
    pd.Series: pandas_store,
    xr.DataArray: xarray_dataarray_store,
    xr.Dataset: xarray_dataset_store,
    plt.Figure: figure_store,
}

LOAD_FUNCTION_MAPPING: Mapping[str, Callable] = {
    '.npy': numpy_load,
    '.csv': pandas_load,
    '.ncs': xarray_dataset_load,
    '.nca': xarray_dataarray_load,
    '.png': figure_load,
}

#                                                  Loading and saving functions
# =============================================================================


def _project_dir_factory(project_dir: Path | str | None) -> Path:
    """
    Create a Path object for the project directory.

    Parameters
    ----------
    project_dir : Path or str or None
        The path of the user-defined directory where to create the f3dasm
        project folder.

    Returns
    -------
    Path
        The Path object for the project directory.

    Raises
    ------
    TypeError
        If the project_dir is not of type Path, str, or None.
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


def store_to_disk(project_dir: Path, object: Any,
                  name: str, id: int,
                  store_function: Optional[Callable] = None) -> str:
    """
    Store an object to disk.

    Parameters
    ----------
    project_dir : Path
        The ExperimentData project_dir path.
    object : Any
        The object to store.
    name : str
        The name of the object.
    id : int
        The id of the object.
    store_function : Optional[Callable], optional
        The method to store the object, by default None.

    Returns
    -------
    str
        The path to the stored object.

    Notes
    -----
    If no store method is provided, the function will try to find a matching
    store type based on the object's type. If no matching type is found, the
    function will use pickle to store the object to disk.
    """
    path = project_dir / EXPERIMENTDATA_SUBFOLDER / name / str(id)

    # Check if the storage parent folder exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # If no store method is provided, try to find a matching store type
    if store_function is None:
        # Check if object type is supported
        object_type = type(object)

        if object_type not in STORE_FUNCTION_MAPPING:
            store_function = pickle_store
            logger.debug(
                f"Object type {object_type} is not natively supported. "
                f"The default pickle storage method will be used.")

        else:
            store_function = STORE_FUNCTION_MAPPING[object_type]

    # Store the object
    absolute_path = Path(store_function(object, path))

    # Return the path relative from the the project directory
    return str(absolute_path.relative_to(
        project_dir / EXPERIMENTDATA_SUBFOLDER))


def load_object(project_dir: Path, path: str | Path,
                load_function: Optional[Callable] = None) -> Any:
    """
    Load an object from disk from a given path and storing method.

    Parameters
    ----------
    project_dir : Path
        The ExperimentData project_dir path.
    path : str or Path
        The path to the object.
    load_function : Optional[Callable], optional
        The method to load the object, by default None.

    Returns
    -------
    Any
        The object loaded from disk.

    Raises
    ------
    ValueError
        If no matching store type is found.

    Notes
    -----
    If no store method is provided, the function will try to find a matching
    store type based on the suffix of the item's path. If no matching type
    is found, the function will use pickle to load the object from disk.
    """
    _path = project_dir / EXPERIMENTDATA_SUBFOLDER / path
    suffix = _path.suffix

    # If no store method is provided, try to find a matching store type
    if load_function is None:
        # Check if object type is supported

        if suffix not in LOAD_FUNCTION_MAPPING:
            load_function = pickle_load
            logger.debug(
                f"Object type '{suffix}' is not natively supported. "
                f"The default pickle load method will be used.")

        else:
            load_function = LOAD_FUNCTION_MAPPING[suffix]

    # Store the object and return the storage location
    return load_function(_path)


def copy_object(object_path: Path,
                old_project_dir: Path, new_project_dir: Path) -> str:

    old_location = old_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path
    new_location = new_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path

    # Check if the storage parent folder exists
    new_location.parent.mkdir(parents=True, exist_ok=True)

    # Check if we do not overwrite an object at new_location
    if new_location.exists():

        stem, suffix = object_path.stem, object_path.suffix
        while (new_project_dir / EXPERIMENTDATA_SUBFOLDER
                / object_path.parent / f"{stem}{suffix}").exists():
            try:
                stem = str(int(stem) + 1)  # Increment stem as integer
            except ValueError:
                raise ValueError((
                    f"Filename {object_path.name} cannot be converted "
                    f"to an integer.")
                )

        object_path = object_path.parent / f"{stem}{suffix}"
        new_location = new_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path

    shutil.copy2(old_location, new_location)
    return str(object_path)
