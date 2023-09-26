"""
Interface class for data generators
"""

#                                                                       Modules
# =============================================================================

from functools import partial
from typing import Callable

from ..design.experimentsample import ExperimentSample
from ..logger import time_and_log

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class DataGenerator:
    """Base class for a data generator"""

    def pre_process(self, **kwargs) -> None:
        """Function that handles the pre-processing"""
        ...
        # raise NotImplementedError("No pre-process function implemented!")

    def execute(self, **kwargs) -> None:
        """Function that calls the FEM simulator the pre-processing"""
        raise NotImplementedError("No execute function implemented!")

    def post_process(self, **kwargs) -> None:
        """Function that handles the post-processing"""
        ...

    @time_and_log
    def run(self, experiment_sample: ExperimentSample, **kwargs) -> ExperimentSample:
        """Run the data generator

        Parameters
        ----------
        ExperimentSample : ExperimentSample
            The design to run the data generator on

        Returns
        -------
        ExperimentSample
            Processed design
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
        self.pre_process = partial(func, **kwargs)

    def add_post_process(self, func: Callable, **kwargs):
        self.post_process = partial(func, **kwargs)
