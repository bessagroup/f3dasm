#                                                                       Modules
# =============================================================================

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class LearningData:
    """Class that holds data used for training or validating machine learning models"""

    def get_input_data(self):
        """Get the input data for the machine learning model"""
        ...

    def get_labels(self):
        """Retrieve the labels of the data"""
        ...
