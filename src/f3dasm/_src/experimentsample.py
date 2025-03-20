"""
A ExperimentSample object contains a single realization of
 the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from copy import deepcopy
# Standard
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

# Third-party
import autograd.numpy as np

# Local
from ._io import copy_object, load_object, store_to_disk
from .design.domain import Domain
from .errors import DecodeError

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class JobStatus(Enum):
    OPEN = 0
    IN_PROGRESS = 1
    FINISHED = 2
    ERROR = 3


class ExperimentSample:
    def __init__(self, input_data: Optional[Dict[str, Any]] = None,
                 output_data: Optional[Dict[str, Any]] = None,
                 domain: Optional[Domain] = None,
                 job_status: Optional[str] = None,
                 project_dir: Optional[Path] = None):
        """
        Realization of a single experiment in the design-of-experiment.

        Parameters
        ----------
        input_data : Optional[Dict[str, Any]]
            Input parameters of one experiment.
            The key is the name of the parameter.
        output_data : Optional[Dict[str, Any]]
            Output parameters of one experiment.
            The key is the name of the parameter.
        domain : Optional[Domain]
            Domain of the experiment, by default None.
        job_status : Optional[str]
            Job status of the experiment, by default None.
        project_dir : Optional[Path]
            Directory of the project, by default None.

        Examples
        --------
        >>> sample = ExperimentSample(
        ...     input_data={'param1': 1.0},
        ...     output_data={'result1': 2.0}
        ... )
        >>> print(sample)
        ExperimentSample(input_data={'param1': 1.0},
        output_data={'result1': 2.0}, job_status=JobStatus.OPEN)
        """
        if input_data is None:
            input_data = dict()

        if output_data is None:
            output_data = dict()

        if domain is None:
            domain = Domain()

        if job_status is None:
            if output_data:
                job_status = 'FINISHED'
            else:
                job_status = 'OPEN'

        if project_dir is None:
            project_dir = Path.cwd()

        self._input_data = input_data
        self._output_data = output_data
        self.domain = domain

        try:
            self.job_status = JobStatus[job_status]
        # If nan is given as key, there is a problem with the decoding of
        # the jobs.csv file
        except KeyError:
            raise DecodeError()

        self.project_dir = project_dir

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
            domain=self.domain + __o.domain,
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
            domain=self.domain._copy(),
            job_status=self.job_status.name,
            project_dir=self.project_dir)

    @property
    def input_data(self) -> Dict[str, Any]:
        """
        Get the input data of the experiment.

        Returns
        -------
        Dict[str, Any]
            Input data of the experiment.
        """
        return {k: self._get_input(k) for k in self._input_data}

    @property
    def output_data(self) -> Dict[str, Any]:
        """
        Get the output data of the experiment.

        Returns
        -------
        Dict[str, Any]
            Output data of the experiment.
        """
        return {k: self._get_output(k) for k in self._output_data}

    #                                                  Alternative constructors
    # =========================================================================

    @classmethod
    def from_numpy(cls: Type[ExperimentSample], input_array: np.ndarray,
                   domain: Optional[Domain] = None) -> ExperimentSample:
        """
        Create an ExperimentSample instance from a numpy array.
        The input data will be stored in the input space of the domain.

        Parameters
        ----------
        cls : Type[ExperimentSample]
            The class type.
        input_array : np.ndarray
            Numpy array containing input data.
        domain : Optional[Domain]
            Domain of the experiment, by default None.

        Returns
        -------
        ExperimentSample
            A new ExperimentSample instance.

        Notes
        -----
        If no domain is provided, the default names will be 'x0', 'x1', etc.

        Examples
        --------
        >>> import numpy as np
        >>> sample = ExperimentSample.from_numpy(np.array([1.0, 2.0]))
        >>> print(sample)
        ExperimentSample(input_data={'x0': 1.0, 'x1': 2.0},
        output_data={}, job_status=JobStatus.OPEN)
        """
        if domain is None:
            n_dim = input_array.flatten().shape[0]
            domain = Domain()
            for i in range(n_dim):
                domain.add_float(name=f'x{i}')

        return cls(input_data={input_name: v for input_name, v in
                               zip(domain.input_space.keys(),
                                   input_array.flatten())},
                   domain=domain,)
    #                                                                   Getters
    # =========================================================================

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
        value = self._get_input(name)

        if value is not None:
            return value

        value = self._get_output(name)

        if value is not None:
            return value

        raise KeyError(f"Parameter {name} not found in input or output data.")

    def _get_input(self, name: str) -> Any:
        """
        Get the value of an input parameter by name.

        Parameters
        ----------
        name : str
            The name of the input parameter.

        Returns
        -------
        Any
            The value of the input parameter, or None if not found.
        """
        if name not in self.domain.input_names:
            return None

        parameter = self.domain.input_space[name]

        if parameter.to_disk:
            return load_object(project_dir=self.project_dir,
                               path=self._input_data[name],
                               load_function=parameter.load_function)
        else:
            return self._input_data[name]

    def _get_output(self, name: str) -> Any:
        """
        Get the value of an output parameter by name.

        Parameters
        ----------
        name : str
            The name of the output parameter.

        Returns
        -------
        Any
            The value of the output parameter, or None if not found.
        """
        if name not in self.domain.output_names:
            return None

        parameter = self.domain.output_space[name]

        if parameter.to_disk:
            return load_object(project_dir=self.project_dir,
                               path=self._output_data[name],
                               load_function=parameter.load_function)
        else:
            return self._output_data[name]

    #                                                                   Setters
    # =========================================================================

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
        def replace_nan_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
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
        def round_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            return {k: round(v, decimals) if isinstance(v, (int, float))
                    else v for k, v in data.items()}

        self._input_data = round_dict(self._input_data)
        self._output_data = round_dict(self._output_data)

    def copy_project_dir(self, project_dir: Path):
        for key, value in self._input_data.items():
            # If the parameter is stored on disk, update the path
            if isinstance(value, str) and self.domain.\
                    input_space[key].to_disk:
                new_value = copy_object(object_path=Path(value),
                                        old_project_dir=self.project_dir,
                                        new_project_dir=project_dir)
                # Update the path in the input data
                self._input_data[key] = new_value

        for key, value in self._output_data.items():
            # If the parameter is stored on disk, update the path
            if isinstance(value, str) and self.domain.\
                    output_space[key].to_disk:
                new_value = copy_object(object_path=Path(value),
                                        old_project_dir=self.project_dir,
                                        new_project_dir=project_dir)
                # Update the path in the input data

                self._output_data[key] = new_value

        self.project_dir = project_dir

    #                                                                 Exporting
    # =========================================================================

    def to_multiindex(self) -> Dict[Tuple[str, str], Any]:
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

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def to_dict(self) -> Dict[str, Any]:
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

    #                                                                   Storing
    # =========================================================================

    def store(self, name: str, object: Any, to_disk: bool = False,
              store_function: Optional[Type[Callable]] = None,
              load_function: Optional[Type[Callable]] = None):
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
        if name not in self.domain.output_names:
            self.domain.add_output(name=name, to_disk=to_disk,
                                   store_function=store_function,
                                   load_function=load_function)

        self._output_data[name] = object

    def store_experimentsample_references(self, idx: int):
        for name, value in self._output_data.items():

            # # If the output parameter is not in the domain, add it
            # if name not in self.domain.output_names:
            #     self.domain.add_output(name=name, to_disk=True)

            parameter = self.domain.output_space[name]

            # If the parameter is to be stored on disk, store it
            # Also check if the value is not already a reference!
            if parameter.to_disk and not isinstance(value, (Path, str)):
                storage_location = store_to_disk(
                    project_dir=self.project_dir,
                    object=value, name=name,
                    id=idx, store_function=parameter.store_function)

                self._output_data[name] = Path(storage_location)

        for name, value in self._input_data.items():
            parameter = self.domain.input_space[name]

            # If the parameter is to be stored on disk, store it
            # Also check if the value is not already a reference!
            if parameter.to_disk and not isinstance(value, (Path, str)):
                storage_location = store_to_disk(
                    project_dir=self.project_dir,
                    object=value, name=name,
                    id=idx, store_function=parameter.store_function)

                self._input_data[name] = Path(storage_location)

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
