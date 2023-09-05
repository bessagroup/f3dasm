#                                                                       Modules
# =============================================================================

from ..design.design import Design
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

    @time_and_log
    def __call__(self, design: Design, **kwargs):
        self.design = design
        self.run()
