#                                                                       Modules
# =============================================================================

# Standard
from typing import Protocol

# Third-party
import tensorflow as tf

# Locals

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'https://d2l.ai/']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Model(Protocol):
    """Base class for all machine learning models"""

    def forward(self, X):
        """Forward pass of the model: calculate an output by giving it an input

        Parameters
        ----------
        X
            Learningdata of the model
        """
        # The model should handle the case when X is None!
        ...

    def get_model_weights(self):
        """Retrieve the model weights as a 1D array"""
        ...

    def set_model_weights(self, weights):
        """Set the model weights with a 1D array"""
        ...
