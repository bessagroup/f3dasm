
from dataclasses import dataclass
from abc import ABC, abstractclassmethod
import numpy

from numpy.core.records import array
from SALib.sample import sobol_sequence


class Sampling(ABC):
    """Represets a generic sampling method for the design of experiments"""

    @abstractclassmethod
    def get_sampling(self, sample_size, range_variables, _type='float') -> array:
        """
        computes N number of samples for the values in the dimensions
        Args:
            sample_size (int): number of samples to be generated
            range_variables (dict): minumun and maximum values for the range
            _type (str): type of output for the sample, interger or float. Default is float
        Returns
        """


@dataclass
class SobolSampling(Sampling):
    """Computes sampling based using a sobol sequence from SALib"""

    sample_size: int
    range_variables: dict

    def get_sampling(self) -> array:
        dimensions = get_dimensions(self.range_variables)
        points = sobol_sequence.sample(self.sample_size, dimensions) 
        
        # Stretch the hypercube towards your bounds
        for i, bound in enumerate(self.range_variables.values()):   
            points[:,i] = points[:,i] * (bound[1] - bound[0]) + bound[0] 
        
        print(points)

        return points


@dataclass
class LinearSampling(Sampling):
    """Computes sampling based using a linear sequence generator from Numpy"""

    sample_size: int
    range_variables: dict

    def get_sampling(self) -> array:
        dimensions = get_dimensions(self.range_variables)
        points = numpy.zeros((self.sample_size, dimensions))
        
        # Stretch the hypercube towards your bounds
        for i, bound in enumerate(self.range_variables.values()):          
            points[:,i] = numpy.linspace(bound[0], bound[1],self.sample_size)   
        return points


# Global methonds:
def get_dimensions(range_variables):
    """computes the number of diension or Keys in a dictionary
    Args:
        range_variables (dict): a dictionary with one or more elemments.

    Returns (int): number of keys in dictionary
    """
    if isinstance(range_variables, dict):
        return len(range_variables.keys())
    else:
        raise TypeError("Variables for the DoE must be of type dictionary")

