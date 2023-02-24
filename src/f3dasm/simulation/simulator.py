#                                                                       Modules
# =============================================================================

# Third-party
import autograd.numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Simulator:
    """Base class for a FEM simulator"""

    # def __init__(self, data: Data):
    #     self.data = data

    def pre_process(self) -> None:
        """Function that handles the pre-processing"""
        pass
        # raise NotImplementedError("No pre-process function implemented!")

    def execute(self, x: np.ndarray) -> None:
        """Function that calls the FEM simulator the pre-processing"""
        raise NotImplementedError("No execute function implemented!")

    def post_process(self) -> None:
        """Function that handles the post-processing"""
        pass
        # raise NotImplementedError("No post-process function implemented!")

    def run(self, x: np.ndarray) -> None:
        self.pre_process()
        self.execute(x)
        self.post_process()
