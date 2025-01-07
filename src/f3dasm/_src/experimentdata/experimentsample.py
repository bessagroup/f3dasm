"""
A ExperimentSample object contains a single realization of
 the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type

# Third-party
import autograd.numpy as np

# Local
from ..design.domain import Domain
from ._io import LoadFunction, StoreFunction, load_object

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
        """Single realization of a design of experiments.

        Parameters
        ----------
        input_data : Dict[str, Any]
            Input parameters of one experiment. The key is the name
            of the parameter.
        output_data : Dict[str, Tuple[Any, bool]]
            Output parameters of one experiment. The key is the name
            of the parameter.
        domain : Optional[Domain]
            Domain of the experiment, by default None
        job_status : Optional[str]
            Job status of the experiment, by default None
        project_dir : Optional[Path]
            Directory of the project, by default None
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
        self.job_status = JobStatus[job_status]
        self.project_dir = project_dir

    def __repr__(self):
        return (f"ExperimentSample("
                f"input_data={self.input_data}, "
                f"output_data={self.output_data}, "
                f"job_status={self.job_status})")

    def __add__(self, __o: ExperimentSample) -> ExperimentSample:
        return ExperimentSample(
            input_data={**self._input_data, **__o._input_data},
            output_data={**self._output_data, **__o._output_data},
            domain=self.domain + __o.domain,
            project_dir=self.project_dir,
        )

    def __eq__(self, __o: ExperimentSample) -> ExperimentSample:
        return (self._input_data == __o._input_data
                and self._output_data == __o._output_data
                and self.job_status == __o.job_status)

    @property
    def input_data(self) -> Dict[str, Any]:
        return {k: self._get_input(k) for k in self._input_data}

    @property
    def output_data(self):
        return {k: self._get_output(k) for k in self._output_data}

    #                                                  Alternative constructors
    # =========================================================================

    @classmethod
    def from_numpy(cls: Type[ExperimentSample], input_array: np.ndarray,
                   domain: Optional[Domain] = None) -> ExperimentSample:

        if domain is None:
            n_dim = input_array.flatten().shape[0]
            domain = Domain()
            for i in range(n_dim):
                domain.add_float(name=f'x{i}', to_disk=False)

        return cls(input_data={input_name: v for input_name, v in
                               zip(domain.input_space.keys(),
                                   input_array.flatten())},
                   domain=domain,)
    #                                                                   Getters
    # =========================================================================

    def get(self, name: str) -> Any:
        # Check if the name is in the input data
        value = self._get_input(name)

        if value is not None:
            return value

        # Check if the name is in the output data
        value = self._get_output(name)

        if value is not None:
            return value

        raise KeyError(f"Parameter {name} not found in input or output data.")

    def _get_input(self, name: str):
        if name not in self.domain.input_names:
            return None

        parameter = self.domain.input_space[name]

        if parameter.to_disk:
            return load_object(project_dir=self.project_dir,
                               path=self._input_data[name],
                               load_function=parameter.load_function)

        else:
            return self._input_data[name]

    def _get_output(self, name: str):
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
        # Check if the status is valid
        if status.upper() not in JobStatus.__members__:
            raise ValueError(f"Status {status} not valid.")

        self.job_status = JobStatus[status.upper()]

    def replace_nan(self, replacement_value: Any):
        """Replace NaN values in input_data and output_data with a custom value.

        Parameters
        ----------
        replacement_value : Any
            The value to replace NaN values with.
        """
        def replace_nan_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            return {k: (replacement_value if isinstance(v, float)
                        and np.isnan(v) else v) for k, v in data.items()}

        self._input_data = replace_nan_in_dict(self._input_data)
        self._output_data = replace_nan_in_dict(self._output_data)

    #                                                                 Exporting
    # =========================================================================

    def to_multiindex(self):
        """
        Convert the experiment sample to a multiindex dictionary,
        which can be used for displaying the data in a pandas DataFrame.

        Note
        ----
        If input or output data is stored on disk, the path to the
        stored object will be displayed in the multiindex.
        """
        return {('jobs', ''): self.job_status.name.lower(),
                **{('input', k): v for k, v in self._input_data.items()},
                **{('output', k): v for k, v in self._output_data.items()},
                }

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array(list(self.input_data.values())),
                np.array(list(self.output_data.values()))
                )

    def to_dict(self) -> Dict[str, Any]:
        return {**self.input_data, **self.output_data}

    #                                                                   Storing
    # =========================================================================

    def store(self, name: str, object: Any, to_disk: bool = False,
              store_function: Optional[Type[StoreFunction]] = None,
              load_function: Optional[Type[LoadFunction]] = None):

        # Add the output to the domain if it is not already there
        if name not in self.domain.output_names:
            self.domain.add_output(name=name, to_disk=to_disk,
                                   store_function=store_function,
                                   load_function=load_function)

        self._output_data[name] = object

    #                                                                Job status
    # =========================================================================

    def is_status(self, status: str) -> bool:
        return self.job_status == JobStatus[status.upper()]
