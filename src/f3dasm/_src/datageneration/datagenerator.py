"""
Interface class for data generators
"""

#                                                                       Modules
# =============================================================================

# Standard
import sys
from abc import abstractmethod
from functools import partial
from typing import Any, Callable

if sys.version_info < (3, 8):  # NOQA
    from typing_extensions import Protocol  # NOQA
else:
    from typing import Protocol

from ..logger import time_and_log

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class ExperimentSample(Protocol):
    def get(self, key: str) -> Any:
        ...

    def store(object: Any, name: str, to_disk: bool) -> None:
        ...

    @property
    def job_number(self) -> int:
        ...


class DataGenerator:
    """Base class for a data generator"""

    def pre_process(self, experiment_sample: ExperimentSample, **kwargs) -> None:
        """Interface function that handles the pre-processing of the data generator

        Notes
        -----
        If not implemented the function will be skipped

        The experiment_sample is cached inside the data generator. This
        allows the user to access the experiment_sample in the pre_process, execute
        and post_process functions as a class variable called self.experiment_sample.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """Interface function that handles the execution of the data generator

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the user

        Notes
        -----
        The experiment_sample is cached inside the data generator. This
        allows the user to access the experiment_sample in the pre_process, execute
        and post_process functions as a class variable called self.experiment_sample.
        """

        ...

    def post_process(self, experiment_sample: ExperimentSample, **kwargs) -> None:
        """Interface function that handles the post-processing of the data generator

        Notes
        -----
        If not implemented the function will be skipped

        The experiment_sample is cached inside the data generator. This
        allows the user to access the experiment_sample in the pre_process, execute
        and post_process functions as a class variable called self.experiment_sample.
        """
        ...

    @time_and_log
    def _run(self, experiment_sample: ExperimentSample, **kwargs) -> ExperimentSample:
        """
        Run the data generator
        This function chains the following methods together

        * pre_process(); to combine the experiment_sample and the parameters
        of the data generator to an input file that can be used to run the data generator

        * execute(); to run the data generator and generate the response of the experiment

        * post_process(); to process the response of the experiment and store it back
         in the experiment_sample

        The function also caches the experiment_sample in the data generator. This
        allows the user to access the experiment_sample in the pre_process, execute
        and post_process functions as a class variable called self.experiment_sample.

        Parameters
        ----------
        ExperimentSample : ExperimentSample
            The design to run the data generator on

        kwargs : dict
            The keyword arguments to pass to the pre_process, execute and post_process

        Returns
        -------
        ExperimentSample
            Processed design with the response of the data generator saved in the
            experiment_sample
        """
        # Cache the design
        self.experiment_sample: ExperimentSample = experiment_sample

        self._pre_simulation()

        self.pre_process(self.experiment_sample, **kwargs)
        self.execute(**kwargs)
        self.post_process(self.experiment_sample, **kwargs)

        self._post_simulation()

        return self.experiment_sample

    def _pre_simulation(self) -> None:
        ...

    def _post_simulation(self) -> None:
        ...

    def add_pre_process(self, func: Callable, **kwargs):
        """Add a pre-processing function to the data generator

        Parameters
        ----------
        func : Callable
            The function to add to the pre-processing
        kwargs : dict
            The keyword arguments to pass to the pre-processing function
        """
        self.pre_process = partial(func, **kwargs)

    def add_post_process(self, func: Callable, **kwargs):
        """Add a post-processing function to the data generator

        Parameters
        ----------
        func : Callable
            The function to add to the post-processing
        kwargs : dict
            The keyword arguments to pass to the post-processing function
        """
        self.post_process = partial(func, **kwargs)
