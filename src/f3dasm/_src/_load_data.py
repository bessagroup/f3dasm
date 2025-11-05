from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._io import load_object, store_to_disk
from .design import Domain
from .experimentdata import ExperimentData
from .experimentsample import ExperimentSample


@dataclass
class ReferenceValue:
    reference: Path
    load_function: Callable[[Path], Any]

    def __call__(self) -> Any:
        return self.load_function(self.reference)


def _get_input(
        name: str, value: Any, domain: Domain, project_dir: Path) -> Any:
    if name not in domain.input_names:
        return None

    param = domain.input_space[name]
    if not param.to_disk:
        return value

    return load_object(
        project_dir=project_dir,
        path=value,
        load_function=param.load_function,
    )


def _get_output(
        name: str, value: Any, domain: Domain, project_dir: Path) -> Any:
    if name not in domain.output_names:
        return None

    param = domain.output_space[name]
    if not param.to_disk:
        return value

    return load_object(
        project_dir=project_dir,
        path=value,
        load_function=param.load_function,
    )


def _get_experiment_sample(data: ExperimentData, idx: int,) -> ExperimentSample:
    es = data[idx]
    # load the data or give it the load functionality to eagerly load?


def _store(
        experiment_sample: ExperimentSample, idx: int, domain: Domain,
        project_dir: Path) -> ExperimentSample:
    for name, value in experiment_sample._output_data.items():

        # # If the output parameter is not in the domain, add it
        # if name not in self.domain.output_names:
        #     self.domain.add_output(name=name, to_disk=True)

        parameter = domain.output_space[name]

        # If the parameter is to be stored on disk, store it
        # Also check if the value is not already a reference!
        if parameter.to_disk and not isinstance(value, Path | str):
            storage_location = store_to_disk(
                project_dir=project_dir,
                object=value, name=name,
                id=idx, store_function=parameter.store_function)

            experiment_sample._output_data[name] = Path(storage_location)

    for name, value in experiment_sample._input_data.items():
        parameter = domain.input_space[name]

        # If the parameter is to be stored on disk, store it
        # Also check if the value is not already a reference!
        if parameter.to_disk and not isinstance(value, Path | str):
            storage_location = store_to_disk(
                project_dir=project_dir,
                object=value, name=name,
                id=idx, store_function=parameter.store_function)

            experiment_sample._input_data[name] = Path(storage_location)

    return experiment_sample
    return experiment_sample
