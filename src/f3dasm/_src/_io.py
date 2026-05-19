"""
Module to load and save output data of experiments and other common IO
operations.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import logging
import pickle
import shutil
from collections.abc import Callable, Mapping

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# matplotlib is an optional dependency since issue #301 -- only the
# `figure_store` / `figure_load` helpers need it, and they fail with a
# clear ImportError the moment they're called rather than blocking
# every import of f3dasm.
try:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt  # noqa: F811

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger("f3dasm")


# =============================================================================

#                                                  Global folder and file names
# =============================================================================

EXPERIMENTDATA_SUBFOLDER = "experiment_data"
EXPERIMENTSAMPLE_SUBFOLDER = "experiment_sample"

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
    _path = Path(path).with_suffix(".pkl")
    with open(_path, "wb") as file:
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
    _path = Path(path).with_suffix(".pkl")
    with open(_path, "rb") as file:
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
    _path = Path(path).with_suffix(".npy")
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
    _path = Path(path).with_suffix(".npy")
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
    _path = Path(path).with_suffix(".csv")
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
    _path = Path(path).with_suffix(".csv")
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
    _path = Path(path).with_suffix(".ncs")
    object.to_netcdf(_path)
    return str(_path)


def xarray_dataarray_store(
    object: xr.DataArray | xr.Dataset, path: str
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
    _path = Path(path).with_suffix(".nca")
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
    _path = Path(path).with_suffix(".ncs")
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
    _path = Path(path).with_suffix(".nca")
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

    Raises
    ------
    ImportError
        If matplotlib is not installed -- matplotlib is an optional
        dependency on the `figures` extra (see issue #301).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "figure_store requires matplotlib; install with "
            "`pip install f3dasm[figures]`."
        )
    _path = Path(path)

    object.savefig(
        _path.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=True,
        dpi=RESOLUTION_MATPLOTLIB_FIGURE,
    )

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

    Raises
    ------
    ImportError
        If matplotlib is not installed -- see :func:`figure_store`.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "figure_load requires matplotlib; install with "
            "`pip install f3dasm[figures]`."
        )
    _path = Path(path).with_suffix(".pdf")
    return plt.imread(_path)


STORE_FUNCTION_MAPPING: dict[type, Callable] = {
    np.ndarray: numpy_store,
    pd.DataFrame: pandas_store,
    pd.Series: pandas_store,
    xr.DataArray: xarray_dataarray_store,
    xr.Dataset: xarray_dataset_store,
}
# matplotlib is optional (issue #301): register the figure mapping only
# when it's importable so users without matplotlib still get a working
# `from f3dasm._src._io import store_object` etc.
if _HAS_MATPLOTLIB:
    STORE_FUNCTION_MAPPING[plt.Figure] = figure_store

LOAD_FUNCTION_MAPPING: Mapping[str, Callable] = {
    ".npy": numpy_load,
    ".csv": pandas_load,
    ".ncs": xarray_dataset_load,
    ".nca": xarray_dataarray_load,
    ".png": figure_load,
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
            not {type(project_dir).__name__}"
    )


def store_object(
    project_dir: Path,
    object: Any,
    name: str,
    id: int,
    store_function: Optional[Callable] = None,
) -> str:
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
                f"The default pickle storage method will be used."
            )

        else:
            store_function = STORE_FUNCTION_MAPPING[object_type]

    # Store the object
    absolute_path = Path(store_function(object, path))

    # Return the path relative from the the project directory
    return str(
        absolute_path.relative_to(project_dir / EXPERIMENTDATA_SUBFOLDER)
    )


def load_object(
    project_dir: Path,
    path: str | Path,
    load_function: Optional[Callable] = None,
) -> Any:
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
                f"The default pickle load method will be used."
            )

        else:
            load_function = LOAD_FUNCTION_MAPPING[suffix]

    # Store the object and return the storage location
    return load_function(_path)


def copy_object(
    object_path: Path, old_project_dir: Path, new_project_dir: Path
) -> str:
    """Copy a stored object file from one project directory to another.

    If a file with the same name already exists in the destination, the stem
    is incremented numerically until a free name is found.

    Parameters
    ----------
    object_path : Path
        Path of the object relative to the experiment data subfolder.
    old_project_dir : Path
        Source project directory.
    new_project_dir : Path
        Destination project directory.

    Returns
    -------
    str
        The (possibly renamed) path of the copied file, relative to the
        experiment data subfolder of `new_project_dir`.

    Raises
    ------
    ValueError
        If an existing file's stem cannot be converted to an integer for
        automatic renaming.
    """
    old_location = old_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path
    new_location = new_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path

    # Check if the storage parent folder exists
    new_location.parent.mkdir(parents=True, exist_ok=True)

    # Check if we do not overwrite an object at new_location
    if new_location.exists():
        stem, suffix = object_path.stem, object_path.suffix
        while (
            new_project_dir
            / EXPERIMENTDATA_SUBFOLDER
            / object_path.parent
            / f"{stem}{suffix}"
        ).exists():
            try:
                stem = str(int(stem) + 1)  # Increment stem as integer
            except ValueError as exc:
                raise ValueError(
                    f"Filename {object_path.name} cannot be converted "
                    f"to an integer."
                ) from exc

        object_path = object_path.parent / f"{stem}{suffix}"
        new_location = new_project_dir / EXPERIMENTDATA_SUBFOLDER / object_path

    shutil.copy2(old_location, new_location)
    return str(object_path)


# =============================================================================


@dataclass
class ReferenceValue:
    """A lightweight reference to an object stored on disk.

    Rather than holding the object itself, a ``ReferenceValue`` stores the
    path (relative to the experiment-data subfolder) and the callable needed
    to deserialise it.  The actual object is loaded lazily via :meth:`load`.

    Parameters
    ----------
    reference : Path
        Path to the stored object, relative to the experiment data subfolder.
    load_function : Callable[[Path], Any]
        Callable that accepts the absolute path and returns the loaded object.
    """

    reference: Path
    load_function: Callable[[Path], Any]

    def load(self, project_dir: Path) -> Any:
        """Load and return the referenced object from disk.

        Parameters
        ----------
        project_dir : Path
            Root project directory used to resolve the relative `reference`.

        Returns
        -------
        Any
            The object loaded from disk.
        """
        return load_object(
            project_dir=project_dir,
            path=self.reference,
            load_function=self.load_function,
        )

    def copy_to(
        self, old_project_dir: Path, new_project_dir: Path
    ) -> ReferenceValue:
        """Copy the referenced file to a new project directory.

        Parameters
        ----------
        old_project_dir : Path
            The project directory where this reference currently lives.
        new_project_dir : Path
            The project directory to copy the file into.

        Returns
        -------
        ReferenceValue
            A new ReferenceValue with the updated reference path, suitable
            for use within ``new_project_dir``.
        """
        new_reference = copy_object(
            self.reference, old_project_dir, new_project_dir
        )
        return ReferenceValue(
            reference=Path(new_reference), load_function=self.load_function
        )

    def __hash__(self) -> int:
        return hash(self.reference)

    def __str__(self) -> str:
        return self.reference.__str__()

    def to_json(self) -> dict:
        """Convert this ReferenceValue into a JSON-serializable dict.

        Returns
        -------
        dict
            A dict with ``__type__``, ``reference``, and a hex-encoded
            pickle of ``load_function``.
        """
        return {
            "__type__": "ReferenceValue",
            "reference": str(self.reference),
            "load_function": pickle.dumps(self.load_function).hex(),
        }

    @classmethod
    def from_json(cls, data: dict) -> ReferenceValue:
        """Reconstruct a ReferenceValue from a JSON dict.

        Parameters
        ----------
        data : dict
            Dict as produced by :meth:`to_json`.

        Returns
        -------
        ReferenceValue
            The reconstructed instance.
        """
        return cls(
            reference=Path(data["reference"]),
            load_function=pickle.loads(bytes.fromhex(data["load_function"])),
        )


@dataclass
class ToDiskValue:
    """An object that should be persisted to disk as part of an experiment.

    ``ToDiskValue`` holds the object together with the callables needed to
    serialise and deserialise it.  Calling :meth:`store` writes the object and
    returns a :class:`ReferenceValue` path; the original in-memory object is
    then replaced by that lightweight reference.

    Parameters
    ----------
    object : Any
        The in-memory object to persist.
    name : str
        Parameter name used to construct the storage path.
    store_function : Callable[[Any, Path], Path]
        Callable that writes `object` to disk and returns the path.
    load_function : Callable[[Path], Any]
        Callable that reads and returns the object from disk.
    """

    object: Any
    name: str
    store_function: Callable[[Any, Path], Path]
    load_function: Callable[[Path], Any]

    def store(self, project_dir: Path, idx: int) -> Path:
        """Write the object to disk and return the stored path.

        If the object is already a string or :class:`Path` (i.e. it was
        previously stored), the path is returned as-is without re-writing.

        Parameters
        ----------
        project_dir : Path
            Root project directory used to build the storage path.
        idx : int
            Experiment-sample index used as the file name.

        Returns
        -------
        Path
            Path of the stored file, relative to the experiment data subfolder.
        """
        if isinstance(self.object, str | Path):
            return Path(self.object)

        store_location = store_object(
            project_dir=project_dir,
            object=self.object,
            name=self.name,
            id=idx,
            store_function=self.store_function,
        )

        return Path(store_location)

    def to_reference(self, reference: Path) -> ReferenceValue:
        """Convert this value to a :class:`ReferenceValue` after storing.

        Parameters
        ----------
        reference : Path
            The path returned by :meth:`store`.

        Returns
        -------
        ReferenceValue
            A lightweight reference that can load the object on demand.
        """
        return ReferenceValue(
            reference=reference,
            load_function=self.load_function,
        )
