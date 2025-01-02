"""
A ExperimentSample object contains a single realization of
 the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

# Third-party
import autograd.numpy as np

# Local
from ..design.domain import Domain
from ._io import StoreProtocol, convert_refs_to_objects

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
                 job_status: Optional[JobStatus] = None):
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
        job_status : Optional[_Jobs]
            Job status of the experiment, by default None
        """

        if input_data is None:
            input_data = dict()

        if output_data is None:
            output_data = dict()

        if domain is None:
            domain = Domain()

        if job_status is None:
            if output_data:
                job_status = JobStatus.FINISHED
            else:
                job_status = JobStatus.OPEN

        self._input_data = input_data
        self._output_data = output_data
        self.domain = domain
        self.job_status = job_status
        self.registered_keys = {}

    def __repr__(self):
        return (f"ExperimentSample("
                f"input_data={self.input_data}, "
                f"output_data={self.output_data}, "
                f"job_status={self.job_status})")

    def __add__(self, __o: ExperimentSample) -> ExperimentSample:
        # TODO: Job status None is default
        return ExperimentSample(
            input_data={**self._input_data, **__o._input_data},
            output_data={**self._output_data, **__o._output_data},
            domain=self.domain + __o.domain,
        )

    @property
    def input_data(self):
        return self._input_data

    @property
    def output_data(self):
        return self._output_data

    #                                                  Alternative constructors
    # =========================================================================

    @classmethod
    def from_numpy(cls: Type[ExperimentSample], input_array: np.ndarray,
                   output_value: Optional[float] = None,
                   jobnumber: int = 0,
                   domain: Optional[Domain] = None) -> ExperimentSample:
        print('Catched!')

    #                                                                   Getters
    # =========================================================================

    def get(self, item: str,
            load_method: Optional[Type[StoreProtocol]] = None):
        if item not in self.domain.input_names:
            raise KeyError(f"Key {item} not in input domain!")

        parameter = self.domain.input_space[item]

        if parameter.to_disk:
            return self._load_from_disk(name=item, load_method=load_method)

        else:
            return self._load_from_experimentdata(name=item)

    def _load_from_disk(self, name: str, load_method: Type[StoreProtocol]):
        ...

    def _load_from_experimentdata(self, name: str):
        return self._input_data[name]

    def load(self,
             reference_keys: Dict[str, Tuple[Callable, Callable]]
             ) -> ExperimentSample:
        input_data = convert_refs_to_objects(self._input_data, reference_keys)
        output_data = convert_refs_to_objects(
            self._output_data, reference_keys)
        return ExperimentSample(
            input_data=input_data,
            output_data=output_data,
            domain=self.domain,
            job_status=self.job_status)

    # def update_job_status(self):
    #     if self.output_data:
    #         self.job_status = _Jobs.FINISHED
    #     else:
    #         self.job_status = _Jobs.OPEN

    def mark(self,
             status: Literal['open', 'in_progress', 'finished', 'error']):
        # Check if the status is valid
        if status.upper() not in JobStatus.__members__:
            raise ValueError(f"Status {status} not valid.")

        self.job_status = JobStatus[status.upper()]

    #                                                                 Exporting
    # =========================================================================

    def to_multiindex(self):
        return {('jobs', ''): self.job_status.name.lower(),
                **{('input', k): v for k, v in self.input_data.items()},
                **{('output', k): v for k, v in self.output_data.items()},
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
              store_method: Optional[Type[StoreProtocol]] = None) -> None:

        # Add the output to the domain if it is not already there
        if name not in self.domain.output_names:
            self.domain.add_output(name=name, to_disk=to_disk)

        parameter = self.domain.output_space[name]

        if parameter.to_disk:
            self._store_to_disk(object=object, name=name,
                                store_method=parameter.store_protocol)
        else:
            self._store_to_experimentdata(object=object, name=name)

        self.registered_keys[name] = to_disk

    def _store_to_disk(
        self, object: Any, name: str,
            store_method: Optional[Type[StoreProtocol]] = None) -> None:
        ...
        # file_path = Path(name) / str(self.job_number)

        # # Check if the file_dir exists
        # (self._experimentdata_directory / Path(name)
        #  ).mkdir(parents=True, exist_ok=True)

        # # Save the object to disk
        # suffix = save_object(
        #     object=object, path=file_path,
        #     experimentdata_directory=self._experimentdata_directory,
        #     store_method=store_method)

        # # Store the path to the object in the output_data
        # self._dict_output[name] = (str(
        #     file_path.with_suffix(suffix)), True)

        # logger.info(f"Stored {name} to {file_path.with_suffix(suffix)}")

    def _store_to_experimentdata(self, object: Any, name: str) -> None:
        self._output_data[name] = object

    def clean_registered_keys(self):
        self.registered_keys = {}

    #                                                                Job status
    # =========================================================================

    def is_status(self, status: str) -> bool:
        return self.job_status == JobStatus[status.upper()]

    # def _store_to_disk(
    #     self, object: Any, name: str,
    #         store_method: Optional[Type[StoreProtocol]] = None) -> None:
    #     file_path = Path(name) / str(self.job_number)

    #     # Check if the file_dir exists
    #     (self._experimentdata_directory / Path(name)
    #      ).mkdir(parents=True, exist_ok=True)

    #     # Save the object to disk
    #     suffix = save_object(
    #         object=object, path=file_path,
    #         experimentdata_directory=self._experimentdata_directory,
    #         store_method=store_method)

    #     # Store the path to the object in the output_data
    #     self._dict_output[name] = (str(
    #         file_path.with_suffix(suffix)), True)

    #     logger.info(f"Stored {name} to {file_path.with_suffix(suffix)}")

    # def _store_to_experimentdata(self, object: Any, name: str) -> None:
    #     self.output_data[name] = object
