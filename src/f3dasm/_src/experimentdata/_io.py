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


#                                                               Storing methods
# =============================================================================

class _Store:
    suffix: int

    def __init__(self, object: Any, path: Path):
        self.path = path
        self.object = object

    def store(self) -> None:
        raise NotImplementedError()

    def load(self) -> Any:
        raise NotImplementedError()


class PickleStore(_Store):
    suffix: str = '.pkl'

    def store(self) -> None:
        with open(self.path.with_suffix(self.suffix), 'wb') as file:
            pickle.dump(self.object, file)

    def load(self) -> Any:
        with open(self.path.with_suffix(self.suffix), 'rb') as file:
            return pickle.load(file)


class NumpyStore(_Store):
    suffix: int = '.npy'

    def store(self) -> None:
        np.save(file=self.path.with_suffix(self.suffix), arr=self.object)

    def load(self) -> np.ndarray:
        return np.load(file=self.path.with_suffix(self.suffix))


class PandasStore(_Store):
    suffix: int = '.csv'

    def store(self) -> None:
        self.object.to_csv(self.path.with_suffix(self.suffix))

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path.with_suffix(self.suffix))


class XarrayStore(_Store):
    suffix: int = '.nc'

    def store(self) -> None:
        self.object.to_netcdf(self.path.with_suffix(self.suffix))

    def load(self) -> xr.DataArray | xr.Dataset:
        return xr.open_dataset(self.path.with_suffix(self.suffix))


STORE_TYPE_MAPPING: Mapping[Type, _Store] = {
    np.ndarray: NumpyStore,
    pd.DataFrame: PandasStore,
    pd.Series: PandasStore,
    xr.DataArray: XarrayStore,
    xr.Dataset: XarrayStore
}

#                                                  Loading and saving functions
# =============================================================================


def load_object(path: Path, experimentdata_directory: Path,
                store_method: Type[_Store] = PickleStore) -> Any:

    _path = experimentdata_directory / path

    if store_method is not None:
        return store_method(None, _path).load()

    if not _path.exists():
        return None

    # Extract the suffix from the item's path
    item_suffix = _path.suffix

    # Use a generator expression to find the first matching store type,
    #  or None if no match is found
    matched_store_type: _Store = next(
        (store_type for store_type in STORE_TYPE_MAPPING.values() if
         store_type.suffix == item_suffix), PickleStore)

    if matched_store_type:
        return matched_store_type(None, _path).load()
    else:
        # Handle the case when no matching suffix is found
        raise ValueError(
            f"No matching store type for item type: '{item_suffix}'")


def save_object(object: Any, path: Path, experimentdata_directory: Path,
                store_method: Optional[Type[_Store]] = None) -> str:
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
        storage: _Store = PickleStore(object, _path)
        logger.debug(f"Object type {object_type} is not natively supported. "
                     f"The default pickle storage method will be used.")

    else:
        storage: _Store = STORE_TYPE_MAPPING[object_type](object, _path)
    # Store object
    storage.store()
    return storage.suffix
