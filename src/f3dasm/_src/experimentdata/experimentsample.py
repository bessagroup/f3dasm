"""
A ExperimentSample object contains a single realization of
 the design-of-experiment in ExperimentData.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type

# Third-party
import autograd.numpy as np

# Local
from ..design.domain import Domain
from ..logger import logger
from ._io import StoreProtocol, load_object, save_object

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class ExperimentSample:
    def __init__(self, dict_input: Dict[str, Any],
                 dict_output: Dict[str, Tuple[Any, bool]],
                 jobnumber: int,
                 experimentdata_directory: Optional[Path] = None):
        """Single realization of a design of experiments.

        Parameters
        ----------
        dict_input : Dict[str, Any]
            Input parameters of one experiment. \
            The key is the name of the parameter.
        dict_output : Dict[str, Tuple[Any, bool]]
            Output parameters of one experiment. \
            The key is the name of the parameter, \
            the first value of the tuple is the actual value and the second \
            if the value is stored to disk or not.
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
                   output_value: Optional[float] = None,
                   jobnumber: int = 0,
                   domain: Optional[Domain] = None) -> ExperimentSample:
        """Create a ExperimentSample object from a numpy array.

        Parameters
        ----------
        input_array : np.ndarray
            input 1D numpy array
        output_value : Optional[float], optional
            objective value, by default None
        jobnumber : int
            jobnumber of the design
        domain : Optional[Domain], optional
            domain of the design, by default None

        Returns
        -------
        ExperimentSample
            ExperimentSample object

        Note
        ----
        If no domain is given, the default parameter names are used.
        These are x0, x1, x2, etc. for input and y for output.
        """
        if domain is None:
            dict_input, dict_output = cls._from_numpy_without_domain(
                input_array=input_array, output_value=output_value)

        else:
            dict_input, dict_output = cls._from_numpy_with_domain(
                input_array=input_array, domain=domain,
                output_value=output_value)

        return cls(dict_input=dict_input,
                   dict_output=dict_output, jobnumber=jobnumber)

    @classmethod
    def _from_numpy_with_domain(
            cls: Type[ExperimentSample],
            input_array: np.ndarray,
            domain: Domain,
            output_value: Optional[float] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        dict_input = {name: val for name,
                      val in zip(domain.names, input_array)}

        if output_value is None:
            dict_output = {name: (np.nan, False)
                           for name in domain.output_space.keys()}
        else:
            dict_output = {
                name: (output_value, False) for
                name in domain.output_space.keys()}

        return dict_input, dict_output

    @classmethod
    def _from_numpy_without_domain(
            cls: Type[ExperimentSample],
            input_array: np.ndarray,
            output_value: Optional[float] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        default_input_names = [f"x{i}" for i in range(len(input_array))]
        default_output_name = "y"

        dict_input = {
            name: val for name, val
            in zip(default_input_names, input_array)}

        if output_value is None:
            dict_output = {name: (np.nan, False)
                           for name in default_output_name}
        else:
            dict_output = {default_output_name: (output_value, False)}

        return dict_input, dict_output

    def get(self, item: str,
            load_method: Optional[Type[StoreProtocol]] = None) -> Any:
        """Retrieve a sample parameter by its name.

        Parameters
        ----------
        item : str
            name of the parameter
        load_method : Optional[Type[_Store]], optional
            class of defined type to load the data. By default None, \
            will try to load the data with the default methods

        Returns
        -------
        Any
            Value of the parameter of the sample
        """
        # Load the value literally (even if it is a reference)
        value, from_disk = self._load_from_experimentdata(item)

        if not from_disk:
            return value

        if isinstance(value, float):
            # value is NaN
            return value

        # Load the object from the reference
        return load_object(Path(value),
                           self._experimentdata_directory, load_method)

    def _load_from_experimentdata(self, item: str) -> Tuple[Any, bool]:
        """Load the data from the experiment data.

        Parameters
        ----------
        item : str
            key of the data to load

        Returns
        -------
        Tuple[Any, bool]
            data and if it is stored to disk or not
        """
        value = self._dict_input.get(item, None)

        if value is None:
            return self._dict_output.get(item, None)
        else:
            return value, False

    def __setitem__(self, key: str, value: Any):
        self._dict_output[key] = (value, False)

    def __repr__(self) -> str:
        return (f"ExperimentSample({self.job_number} ({self.jobs}) :"
                f"{self.input_data} - {self.output_data})")

    @property
    def input_data(self) -> Dict[str, Any]:
        """Retrieve the input data of the design as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The input data of the design as a dictionary.
        """
        return self._dict_input

    _input_data = input_data

    @property
    def output_data(self) -> Dict[str, Any]:
        """Retrieve the output data of the design as a dictionary.

        Returns
        -------
        Dict[str, Any]
            The output data of the design as a dictionary.
        """
        # This is the loaded data !
        return {key: self.get(key) for key in self._dict_output}

    # create an alias for output_data names output_data_loaded
    # this is for backward compatibility
    output_data_loaded = output_data

    _output_data = output_data

    @property
    def output_data_with_references(self) -> Dict[str, Any]:
        """Retrieve the output data of the design as a dictionary, \
           but refrain from loading the data from disk and give the references.

        Note
        ----
        If you want to use the data, you can load it in memory with the \
        :func:`output_data` property.

        Returns
        -------
        Dict[str, Any]
            The output data of the design as a dictionary with references.
        """
        return {key: value for key, (value, _) in self._dict_output.items()}

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
    def jobs(self) -> Literal['finished', 'open']:
        """Retrieve the job status.

        Returns
        -------
        str
            The job number of the design as a tuple.
        """
        # Check if the output contains values or not all nan
        has_all_nan = np.all(np.isnan(list(self._output_data.values())))

        if self._output_data and not has_all_nan:
            status = 'finished'
        else:
            status = 'open'

        return status

    # Alias
    _jobs = jobs

#                                                                        Export
# =============================================================================

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the design to a tuple of numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of numpy arrays containing the input and output data.
        """
        return np.array(list(self.input_data.values())), np.array(
            list(self.output_data.values()))

    def to_dict(self) -> Dict[str, Any]:
        """Converts the design to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the input and output data.
        """
        return {**self.input_data, **self.output_data,
                'job_number': self.job_number}

    def store(self, name: str, object: Any, to_disk: bool = False,
              store_method: Optional[Type[StoreProtocol]] = None) -> None:
        """Store an object to disk.

        Parameters
        ----------

        name : str
            The name of the file to store the object in.
        object : Any
            The object to store.
        to_disk : bool, optional
            Whether to store the object to disk, by default False
        store_method : Store, optional
            The method to use to store the object, by default None

        Note
        ----
        If to_disk is True and no store_method is provided, the default store \
        method will be used.

        The default store method is saving the object as a pickle file (.pkl).
        """
        if to_disk:
            self._store_to_disk(object=object, name=name,
                                store_method=store_method)
        else:
            self._store_to_experimentdata(object=object, name=name)

    def _store_to_disk(
        self, object: Any, name: str,
            store_method: Optional[Type[StoreProtocol]] = None) -> None:
        file_path = Path(name) / str(self.job_number)

        # Check if the file_dir exists
        (self._experimentdata_directory / Path(name)
         ).mkdir(parents=True, exist_ok=True)

        # Save the object to disk
        suffix = save_object(
            object=object, path=file_path,
            experimentdata_directory=self._experimentdata_directory,
            store_method=store_method)

        # Store the path to the object in the output_data
        self._dict_output[name] = (str(
            file_path.with_suffix(suffix)), True)

        logger.info(f"Stored {name} to {file_path.with_suffix(suffix)}")

    def _store_to_experimentdata(self, object: Any, name: str) -> None:
        self._dict_output[name] = (object, False)


def _experimentsample_factory(
    experiment_sample: np.ndarray | ExperimentSample | Dict,
    domain: Domain | None) \
        -> ExperimentSample:
    """Factory function for the ExperimentSample class.

    Parameters
    ----------
    experiment_sample : np.ndarray | ExperimentSample | Dict
        The experiment sample to convert to an ExperimentSample.
    domain: Domain | None
        The domain of the experiment sample.

    Returns
    -------
    ExperimentSample
        The converted experiment sample.
    """
    if isinstance(experiment_sample, np.ndarray):
        return ExperimentSample.from_numpy(input_array=experiment_sample,
                                           domain=domain)

    elif isinstance(experiment_sample, dict):
        return ExperimentSample(dict_input=experiment_sample,
                                dict_output={}, jobnumber=0)

    elif isinstance(experiment_sample, ExperimentSample):
        return experiment_sample

    else:
        raise TypeError(
            f"The experiment_sample should be a numpy array"
            f", dictionary or ExperimentSample, not {type(experiment_sample)}")
