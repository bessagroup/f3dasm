#                                                                       Modules
# =============================================================================

from .._logging import time_and_log
from ..design._data import Trial
#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class Simulator:
    """Base class for a FEM simulator"""
    def __init__(self, trial: Trial, **kwargs):
        self.trial = trial
        self.kwargs = kwargs

    def pre_process(self) -> None:
        """Function that handles the pre-processing"""
        ...
        # raise NotImplementedError("No pre-process function implemented!")

    def execute(self) -> None:
        """Function that calls the FEM simulator the pre-processing"""
        raise NotImplementedError("No execute function implemented!")

    def post_process(self) -> None:
        """Function that handles the post-processing"""
        ...

    @time_and_log
    def run(self) -> None:
        self.pre_process()
        self.execute()
        self.post_process()
