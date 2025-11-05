from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._io import load_object, store_to_disk
from .design import Domain
from .errors import DecodeError
from .experimentdata import ExperimentData
from .experimentsample import ExperimentSample, JobStatus

# =============================================================================


@dataclass
class ExperimentSample:
    _input_data: dict[str, Any] = field(default_factory=dict)
    _output_data: dict[str, Any] = field(default_factory=dict)
    job_status: str | None = None
    project_dir: Path = field(default_factory=Path.cwd())

    def __post_init__(self):
        """Handle defaults and consistency checks after dataclass init."""
        # Infer job_status if not provided
        if self.job_status is None:
            self.job_status = "FINISHED" if self._output_data else "OPEN"

        # Convert string job_status to JobStatus enum
        try:
            self.job_status = JobStatus[self.job_status]
        except KeyError as exc:
            raise DecodeError() from exc

    def __repr__(self):
        """
        Return a string representation of the ExperimentSample instance.

        Returns
        -------
        str
            String representation of the ExperimentSample instance.
        """
        return (f"ExperimentSample("
                f"input_data={self.input_data}, "
                f"output_data={self.output_data}, "
                f"job_status={self.job_status})")

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
            input_data={**self._input_data, **__o._input_data},
            output_data={**self._output_data, **__o._output_data},
            project_dir=self.project_dir,
        )

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
        return (self._input_data == __o._input_data
                and self._output_data == __o._output_data
                and self.job_status == __o.job_status)

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
            input_data=deepcopy(self._input_data),
            output_data=deepcopy(self._output_data),
            job_status=self.job_status.name,
            project_dir=self.project_dir)

    @property
    def input_data(self) -> dict[str, Any]:
        return {k: _get_value(value=v, project_dir=self.project_dir)
                for k, v in self._input_data.items()}

    @property
    def output_data(self) -> dict[str, Any]:
        return {k: _get_value(value=v, project_dir=self.project_dir)
                for k, v in self._output_data.items()}

    @classmethod
    def from_numpy(cls: type[ExperimentSample], input_array: np.ndarray,
                   domain: Domain | None = None) -> ExperimentSample:
        raise NotImplementedError()

    def get(self, name: str) -> Any:
        value = self._input_data.get(name, None)
        if value is None:
            value = self._output_data.get(name, None)

        if value is None:
            raise KeyError(
                f"Parameter '{name}' not found in input or output data.")

        return _get_value(value=value, project_dir=self.project_dir)

    def mark(self,
             status: Literal['open', 'in_progress', 'finished', 'error']):
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
        if status.upper() not in JobStatus.__members__:
            raise ValueError(f"Invalid status: {status}")

        self.job_status = JobStatus[status.upper()]

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
            return {k: (replacement_value if np.isnan(v) else v)
                    for k, v in data.items()}

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
            return {k: round(v, decimals) if isinstance(v, int | float)
                    else v for k, v in data.items()}

        self._input_data = round_dict(self._input_data)
        self._output_data = round_dict(self._output_data)

    def copy_project_dir(self, project_path: Path):
        raise NotImplementedError()

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
        return {('jobs', ''): self.job_status.name.lower(),
                **{('input', k): v for k, v in self._input_data.items()},
                **{('output', k): v for k, v in self._output_data.items()},
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
        return (np.array(list(self.input_data.values())),
                np.array(list(self.output_data.values()))
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

    def store(self, name: str, object: Any, to_disk: bool = False,
              store_function: type[Callable] | None = None,
              load_function: type[Callable] | None = None):
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
        value = object if not to_disk else ToDiskValue(
            object=object,
            name=name,
            store_function=store_function,
            load_function=load_function)
        self._output_data[name] = value

    def store_experimentsample_references(self, idx: int):
        raise NotImplementedError()

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


@dataclass
class ReferenceValue:
    reference: Path
    load_function: Callable[[Path], Any]

    def load(self, project_dir: Path) -> Any:
        return load_object(
            project_dir=project_dir,
            path=self.reference,
            load_function=self.load_function,
        )


@dataclass
class ToDiskValue:
    object: Any
    name: str
    store_function: Callable[[Any, Path], Path]
    load_function: Callable[[Path], Any]

    def store(self, project_dir: Path, idx: int) -> Path:
        if isinstance(self.object, str | Path):
            return Path(self.object)

        store_location = store_to_disk(
            project_dir=project_dir,
            object=self,
            name=self.name,
            id=idx,
            store_function=self.store_function,
        )

        return Path(store_location)

    def to_reference(self, reference: Path) -> ReferenceValue:
        return ReferenceValue(
            reference=reference,
            load_function=self.load_function,
        )


def _get_value(value: Any, project_dir: Path) -> Any:
    return value if not isinstance(value, ReferenceValue) else value.load(project_dir)


def _store(
        experiment_sample: ExperimentSample, idx: int, domain: Domain,
        project_dir: Path) -> ExperimentSample:
    for name, value in experiment_sample._output_data.items():
        # If the value is a ToDiskValue, we need to store it
        if isinstance(value, ToDiskValue):
            if name not in domain.output_space:
                domain.add_output(
                    name=name,
                    to_disk=True,
                    store_function=value.store_function,
                    load_function=value.load_function)
            # Store the value on disk
            reference = value.store(
                project_dir=project_dir,
                idx=idx,
            )

            # Update the experiment sample to reference the stored location
            experiment_sample._output_data[name] = value.to_reference(
                reference=reference)

        else:
            if name not in domain.output_space:
                domain.add_output(
                    name=name
                )

    for name, value in experiment_sample._input_data.items():
        if isinstance(value, ToDiskValue):
            if name not in domain.input_space:
                domain.add_parameter(
                    name=name,
                    to_disk=True,
                    store_function=value.store_function,
                    load_function=value.load_function)
            # Store the value on disk
            reference = value.store(
                project_dir=project_dir,
                idx=idx,
            )

            # Update the experiment sample to reference the stored location
            experiment_sample._input_data[name] = value.to_reference(
                reference=reference)

        else:
            if name not in domain.input_space:
                domain.add_parameter(
                    name=name
                )

    return experiment_sample, domain
