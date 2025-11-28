"""
A ExperimentSample object contains a single realization of
 the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

# Standard
from enum import Enum
from pathlib import Path
from typing import Any, Literal

# Third-party
import numpy as np

# Local
from ._io import EXPERIMENTSAMPLE_SUBFOLDER, ReferenceValue, ToDiskValue
from .errors import DecodeError

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class JobStatus(Enum):
    OPEN = 0
    IN_PROGRESS = 1
    FINISHED = 2
    ERROR = 3

    def __str__(self) -> str:
        return self.name


@dataclass
class ExperimentSample:
    _input_data: dict[str, Any] | None = field(default_factory=dict)
    _output_data: dict[str, Any] | None = field(default_factory=dict)
    job_status: JobStatus | None | str = None
    project_dir: Path = field(default_factory=Path.cwd)
    """
    Realization of a single experiment in the design-of-experiment.

    Parameters
    ----------
    _input_data : dict[str, Any] | None
        Input parameters of one experiment.
        The key is the name of the parameter.
    _output_data : dict[str, Any] | None
        Output parameters of one experiment.
        The key is the name of the parameter.
    job_status : JobStatus | None
        Job status of the experiment, by default None.
    project_dir : Optional[Path]
        Directory of the project, by default None.

    Examples
    --------
    >>> sample = ExperimentSample(
    ...     _input_data={'param1': 1.0},
    ...     _output_data={'result1': 2.0}
    ... )
    >>> print(sample)
    ExperimentSample(input_data={'param1': 1.0},
    output_data={'result1': 2.0}, job_status=JobStatus.OPEN)
    """

    def __post_init__(self):
        """Handle defaults and consistency checks after dataclass init."""
        # Infer job_status if not provided
        if self.job_status is None:
            self.job_status = (
                JobStatus.FINISHED if self._output_data else JobStatus.OPEN
            )

        if isinstance(self.job_status, str):
            # Convert string job_status to JobStatus enum
            try:
                self.job_status = JobStatus[self.job_status]
            except KeyError as exc:
                raise DecodeError() from exc

        if self._output_data is None:
            self._output_data = {}
        if self._input_data is None:
            self._input_data = {}

    def __repr__(self):
        """
        Return a string representation of the ExperimentSample instance.

        Returns
        -------
        str
            String representation of the ExperimentSample instance.
        """
        return (
            f"ExperimentSample("
            f"input_data={self.input_data}, "
            f"output_data={self.output_data}, "
            f"job_status={self.job_status})"
        )

    def __add__(self, __o: ExperimentSample) -> ExperimentSample:
        """
        Add two ExperimentSample instances.

        Parameters
        ----------
        __o : ExperimentSample
            Another ExperimentSample instance.

        Returns
        -------
        ExperimentSample
            A new ExperimentSample instance with combined input
            and output data.

        Notes
        -----
        The job status of the new ExperimentSample instance will be
        reconstructed from the absence or presence of output data.
        If output data is present, the job status will be 'FINISHED'.
        Otherwise, the job status will be 'OPEN'.

        Examples
        --------
        >>> sample1 = ExperimentSample(input_data={'param1': 1.0})
        >>> sample2 = ExperimentSample(output_data={'result1': 2.0})
        >>> combined_sample = sample1 + sample2
        >>> print(combined_sample)
        ExperimentSample(input_data={'param1': 1.0},
        output_data={'result1': 2.0}, job_status=JobStatus.FINISHED)
        """
        return ExperimentSample(
            _input_data={**self._input_data, **__o._input_data},
            _output_data={**self._output_data, **__o._output_data},
            project_dir=self.project_dir,
        )

    # TODO: the self.project_dir should also be compared, but it
    # breaks some tests
    def __eq__(self, __o: ExperimentSample) -> bool:
        """
        Check if two ExperimentSample instances are equal.

        Parameters
        ----------
        __o : ExperimentSample
            Another ExperimentSample instance.

        Returns
        -------
        bool
            True if the instances are equal, False otherwise.
        """
        return (
            self._input_data == __o._input_data
            and self._output_data == __o._output_data
            and self.job_status == __o.job_status
        )

    def _copy(self) -> ExperimentSample:
        """
        Create a copy of the ExperimentSample instance.

        Returns
        -------
        ExperimentSample
            A new ExperimentSample instance with the same input and
            output data.
        """
        return ExperimentSample(
            _input_data=deepcopy(self._input_data),
            _output_data=deepcopy(self._output_data),
            job_status=self.job_status,
            project_dir=self.project_dir,
        )

    @property
    def input_data(self) -> dict[str, Any]:
        """
        Get the input data of the experiment.

        Returns
        -------
        Dict[str, Any]
            Input data of the experiment.
        """
        return {
            k: _get_value(value=v, project_dir=self.project_dir)
            for k, v in self._input_data.items()
        }

    @property
    def output_data(self) -> dict[str, Any]:
        """
        Get the output data of the experiment.

        Returns
        -------
        Dict[str, Any]
            Output data of the experiment.
        """
        return {
            k: _get_value(value=v, project_dir=self.project_dir)
            for k, v in self._output_data.items()
        }

    @classmethod
    def from_numpy(
        cls: type[ExperimentSample], input_array: np.ndarray
    ) -> ExperimentSample:
        """
        Create an ExperimentSample instance from a numpy array.

        Parameters
        ----------
        input_array : np.ndarray
            Numpy array containing input data.

        Returns
        -------
        ExperimentSample
            A new ExperimentSample instance.

        Notes
        -----
        The default names will be 'x0', 'x1', etc.

        Examples
        --------
        >>> import numpy as np
        >>> sample = ExperimentSample.from_numpy(np.array([1.0, 2.0]))
        >>> print(sample)
        ExperimentSample(input_data={'x0': 1.0, 'x1': 2.0},
        output_data={}, job_status=JobStatus.OPEN)
        """
        return cls(
            _input_data={
                f"x{i}": v for i, v in enumerate(input_array.flatten())
            },
        )

    @classmethod
    def from_json(cls, path: Path) -> ExperimentSample:
        """
        Create an ExperimentSample instance from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Returns
        -------
        ExperimentSample
            A new ExperimentSample instance.

        Examples
        --------
        >>> sample = ExperimentSample.from_json(Path("sample.json"))
        >>> print(sample)
        ExperimentSample(input_data={'param1': 1.0},
        output_data={'result1': 2.0}, job_status=JobStatus.FINISHED)
        """
        with open(path) as f:
            data = json.load(f)

        def restore(obj):
            if (
                isinstance(obj, dict)
                and obj.get("__type__") == "ReferenceValue"
            ):
                return ReferenceValue.from_json(obj)
            return obj

        # Recursively apply restoration
        def walk(d):
            if isinstance(d, dict):
                return {k: walk(restore(v)) for k, v in d.items()}
            else:
                return d

        return cls(
            _input_data=walk(data["input_data"]),
            _output_data=walk(data["output_data"]),
            job_status=data["job_status"],
            project_dir=Path(data["project_dir"]),
        )

    def get(self, name: str) -> Any:
        """
        Get the value of a parameter by name.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        Any
            The value of the parameter.

        Raises
        ------
        KeyError
            If the parameter is not found in input or output data.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': 1.0})
        >>> sample.get('param1')
        1.0
        """
        value = self._input_data.get(name, None)
        if value is None:
            value = self._output_data.get(name, None)

        if value is None:
            raise KeyError(
                f"Parameter '{name}' not found in input or output data."
            )

        return _get_value(value=value, project_dir=self.project_dir)

    def mark(
        self, status: Literal["open", "in_progress", "finished", "error"]
    ):
        """
        Mark the job status of the experiment.

        Parameters
        ----------
        status : Literal['open', 'in_progress', 'finished', 'error']
            The new job status.

        Raises
        ------
        ValueError
            If the status is not valid.

        Examples
        --------
        >>> sample = ExperimentSample()
        >>> sample.mark('finished')
        >>> sample.job_status
        <JobStatus.FINISHED: 2>
        """
        try:
            # Look up enum member
            self.job_status = JobStatus[status.upper()]

        # If the status is invalid, raise ValueError
        except KeyError as exc:
            valid = ", ".join(s.lower() for s in JobStatus.__members__)
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {valid}"
            ) from exc

    def replace_nan(self, replacement_value: Any):
        """
        Replace NaN values in input_data and output_data with a custom value.

        Parameters
        ----------
        replacement_value : Any
            The value to replace NaN values with.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': np.nan})
        >>> sample.replace_nan(0)
        >>> sample.input_data['param1']
        0
        """

        def replace_nan_in_dict(data: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (replacement_value if np.isnan(v) else v)
                for k, v in data.items()
            }

        self._input_data = replace_nan_in_dict(self._input_data)
        self._output_data = replace_nan_in_dict(self._output_data)

    def round(self, decimals: int):
        """
        Round the input and output data to a specified number
        of decimal places.

        Parameters
        ----------
        decimals : int
            The number of decimal places to round to.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': 1.2345})
        >>> sample.round(2)
        >>> sample.input_data['param1']
        1.23
        """

        def round_dict(data: dict[str, Any]) -> dict[str, Any]:
            return {
                k: round(v, decimals) if isinstance(v, int | float) else v
                for k, v in data.items()
            }

        self._input_data = round_dict(self._input_data)
        self._output_data = round_dict(self._output_data)

    def to_multiindex(self) -> dict[tuple[str, str], Any]:
        """
        Convert the experiment sample to a multiindex dictionary.
        Used to display the data prettily as a table in a Jupyter notebook.

        Returns
        -------
        Dict[Tuple[str, str], Any]
            A multiindex dictionary containing the job status, input,
            and output data.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': 1.0})
        >>> sample.to_multiindex()
        {('jobs', ''): 'open', ('input', 'param1'): 1.0}
        """
        return {
            ("jobs", ""): self.job_status.name.lower(),
            **{("input", k): v for k, v in self._input_data.items()},
            **{("output", k): v for k, v in self._output_data.items()},
        }

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the experiment sample to numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing numpy arrays of input and output data.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': 1.0})
        >>> sample.to_numpy()
        (array([1.]), array([]))
        """
        return (
            np.array(list(self.input_data.values())),
            np.array(list(self.output_data.values())),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the experiment sample to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing both input and output data.

        Examples
        --------
        >>> sample = ExperimentSample(input_data={'param1': 1.0})
        >>> sample.to_dict()
        {'param1': 1.0}
        """
        return {**self.input_data, **self.output_data}

    def store(
        self,
        name: str,
        object: Any,
        to_disk: bool = False,
        store_function: Callable | None = None,
        load_function: Callable | None = None,
        which: Literal["input", "output"] = "output",
    ):
        """
        Store an object in the experiment sample.

        Parameters
        ----------
        name : str
            The name of the object to store.
        object : Any
            The object to store.
        to_disk : bool, optional
            If True, the object will be stored on disk, by default False.
        store_function : Optional[Type[Callable]], optional
            The function to use for storing the object on disk
            by default None.
        load_function : Optional[Type[Callable]], optional
            The function to use for loading the object from disk,
            by default None.
        which : Literal['input', 'output'], optional
            Specify whether to store the object in input or output data,
            by default 'output'.

        Notes
        -----
        The object will be stored in the input data if the name is in the
        input space of the domain. Otherwise, the object will be stored in
        the output data if the name is in the output space of the domain.

        If the object is stored on disk, the path to the stored object will
        be stored in the input or output data, depending on where the object
        is stored.

        The store_function should have the following signature:

        .. code-block:: python

            def store_function(object: Any, path: str) -> Path:
                ...

        The load_function should have the following signature:

        .. code-block:: python

            def load_function(path: str) -> Any:
                ...
        """
        value = (
            object
            if not to_disk
            else ToDiskValue(
                object=object,
                name=name,
                store_function=store_function,
                load_function=load_function,
            )
        )

        if which == "input":
            self._input_data[name] = value
        elif which == "output":
            self._output_data[name] = value
        else:
            raise ValueError(
                f"Invalid value for 'which': {which}. "
                f"Expected 'input' or 'output'."
            )

    def store_as_json(self, idx: int):
        def default_serializer(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, ReferenceValue):
                return obj.to_json()
            return str(obj)

        data = {
            "input_data": self._input_data,
            "output_data": self._output_data,
            "job_status": self.job_status.name,
            "project_dir": self.project_dir,
        }

        file_path = (
            self.project_dir / EXPERIMENTSAMPLE_SUBFOLDER / f"{idx}"
        ).with_suffix(".json")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=default_serializer)

    #                                                                Job status
    # =========================================================================

    def is_status(self, status: str) -> bool:
        """
        Check if the job's current status matches the given status.

        Parameters
        ----------
        status : str
            The status to check against the job's current status.

        Returns
        -------
        bool
            True if the job's current status matches the given status,
            False otherwise.

        Examples
        --------
        >>> sample = ExperimentSample()
        >>> sample.is_status('open')
        True
        """
        return self.job_status == JobStatus[status.upper()]


def _get_value(value: Any, project_dir: Path) -> Any:
    """
    Retrieve the actual value, loading from disk if necessary.

    Parameters
    ----------
    value : Any
        The value to retrieve, which may be a ReferenceValue.
    project_dir : Path
        The project directory for loading from disk.
    Returns
    -------
    Any
        The actual value, loaded from disk if it was a ReferenceValue.
    """
    return (
        value
        if not isinstance(value, ReferenceValue)
        else value.load(project_dir)
    )
