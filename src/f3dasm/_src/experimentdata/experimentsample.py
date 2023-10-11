"""
A ExperimentSample object contains a single realization of the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type

# Third-party
import autograd.numpy as np
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

PATH_PREFIX = 'path_'


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


def load_object(path: Path, experimentdata_directory: Path, store_method: Type[_Store] = PickleStore) -> Any:

    _path = experimentdata_directory / path

    if store_method is not None:
        return store_method(None, _path).load()

    if not _path.exists():
        return None

    # Extract the suffix from the item's path
    item_suffix = _path.suffix

    # Use a generator expression to find the first matching store type, or None if no match is found
    matched_store_type: _Store = next(
        (store_type for store_type in STORE_TYPE_MAPPING.values() if store_type.suffix == item_suffix), PickleStore)

    if matched_store_type:
        return matched_store_type(None, _path).load()
    else:
        # Handle the case when no matching suffix is found
        raise ValueError(f"No matching store type for item type: '{item_suffix}'")


def save_object(object: Any, path: Path, experimentdata_directory: Path,
                store_method: Optional[Type[_Store]] = None) -> str:
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

#                                                              ExperimentSample
# =============================================================================


class ExperimentSample:
    def __init__(self, dict_input: Dict[str, Any], dict_output: Dict[str, Any],
                 jobnumber: int, experimentdata_directory: Optional[Path] = None):
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

        if experimentdata_directory is None:
            experimentdata_directory = Path.cwd()

        self._experimentdata_directory = experimentdata_directory

    @classmethod
    def from_numpy(cls: Type[ExperimentSample], input_array: np.ndarray,
                   output_value: Optional[float] = None, jobnumber: int = 0) -> ExperimentSample:
        """Create a ExperimentSample object from a numpy array.

        Parameters
        ----------
        input_array : np.ndarray
            input 1D numpy array
            output_value : Optional[float], optional
            objective value, by default None

        jobnumber : int
            jobnumber of the design

        Returns
        -------
        ExperimentSample
            ExperimentSample object
        """
        dict_input = {f"x{i}": val for i, val in enumerate(input_array)}
        if output_value is None:
            dict_output = {}
        else:
            dict_output = {"y": output_value}

        return cls(dict_input=dict_input, dict_output=dict_output, jobnumber=jobnumber)

    def get(self, item: str, load_method: Optional[Type[_Store]] = None) -> Any:
        """Retrieve a sample parameter by its name.

        Parameters
        ----------
        item : str
            name of the parameter
        load_method : Optional[Type[_Store]], optional
            class of defined type to load the data. By default None,
            will try to load the data with the default methods

        Returns
        -------
        Any
            Value of the parameter of the sample
        """
        # Load the value literally (even if it is a reference)
        value = self._load_from_experimentdata(item)

        if item.startswith(PATH_PREFIX):

            if isinstance(value, float):
                # value is NaN
                return item

            # Load the object from the reference
            return load_object(Path(value), self._experimentdata_directory, load_method)
        else:
            # Return the literal value
            return value

    def _load_from_experimentdata(self, item: str) -> Any:
        """Load the data from the experiment data.

        Parameters
        ----------
        item : str
            key of the data to load

        Returns
        -------
        Any
            data
        """
        value = self._dict_input.get(item, None)
        if value is None:
            value = self._dict_output.get(item, None)

        return value

    def __setitem__(self, key: str, value: Any):
        self._dict_output[key] = value

    def __repr__(self) -> str:
        return f"ExperimentSample({self.job_number} : {self.input_data} - {self.output_data})"

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
        # Load all the data from the experiment data
        # return {key: self.get(key) for key in self._dict_output.keys()}
        return self._dict_output

    @property
    def output_data_loaded(self) -> Dict[str, Any]:
        """Retrieve the output data of the design as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The output data of the design as a dictionary.
        """
        # Load all the data from the experiment data
        return {key: self.get(key) for key in self._dict_output.keys()}

    @property
    def job_number(self) -> int:
        """Retrieve the job number of the design.

        Returns
        -------
        int
            The job number of the design.
        """
        return self._jobnumber

    @property
    def jobs(self) -> int:
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

    def to_dict(self) -> Dict[str, Any]:
        """Converts the design to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the input and output data.
        """
        return {**self.input_data, **self.output_data_loaded, 'job_number': self.job_number}

    def store(self, object: Any, name: str, to_disk: bool = False,
              store_method: Optional[Type[_Store]] = None) -> None:
        """Store an object to disk.

        Parameters
        ----------

        object : Any
            The object to store.
        name : str
            The name of the file to store the object in.
        to_disk : bool, optional
            Whether to store the object to disk, by default False
        store_method : Store, optional
            The method to use to store the object, by default None

        Raises
        ------

        TypeError
            If the object type is not supported and no store_method is provided.
        """
        if to_disk:
            self._store_to_disk(object=object, name=name, store_method=store_method)
        else:
            self._store_to_experimentdata(object=object, name=name)

    def _store_to_disk(self, object: Any, name: str, store_method: Optional[Type[_Store]] = None) -> None:
        file_path = Path(name) / str(self.job_number)

        # Check if the file_dir exists
        (self._experimentdata_directory / Path(name)).mkdir(parents=True, exist_ok=True)

        # Save the object to disk
        suffix = save_object(object=object, path=file_path,
                             experimentdata_directory=self._experimentdata_directory,
                             store_method=store_method)

        # Store the path to the object in the output_data
        self._dict_output[f"{PATH_PREFIX}{name}"] = str(file_path.with_suffix(suffix))

        logger.info(f"Stored {name} to {file_path.with_suffix(suffix)}")

    def _store_to_experimentdata(self, object: Any, name: str) -> None:
        self._dict_output[name] = object
