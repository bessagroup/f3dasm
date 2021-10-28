
from dataclasses import dataclass
from abc import ABC, abstractclassmethod
import numpy

from numpy.core.records import array
from SALib.sample import sobol_sequence
from torch._C import Size

@dataclass
class SamplingMethod(ABC):
    """Represets a generic sampling method for parmeters with a range of values"""

    size: int
    values: dict
    
    @abstractclassmethod
    def compute_sampling(self, aprox='float') -> array:
        """
        Computes N number of samples for the values represented as ranges. E.g. [min, max]
        Values 
        Args:
            sample_size (int): number of samples to be generated
            values (dic): ranges of values to be sample as name:value pairs
            aprox (int or float): controls if sampled values are aproximated to an integer or to a float. Default is float
        Returns:
            sampling results as array 
        """
    
    def validate_range(self) -> None:
        """Checks that elements in values represent value rantes """

        for value in self.values.values():
            if isinstance(value, list) and len(value) == 2:
                return True
            else:
                raise TypeError("Sampling can only be applied to values representing a range. E.g. [2, 3]")


    def compute_dimensions(self) -> int:
        """Computes the number of dimentions for the sampling methon
        Args:
            values (dict): values the sampling will be applied to
        Returns (int): number of elements in values
        """
        return len(self.values.keys())

    
class Sobol(SamplingMethod):
    """Computes sampling using a sobol sequence from SALib"""

    def compute_sampling(self, aprox='float') -> array:
        super().validate_range()
        self.dimensions = super().compute_dimensions()
        # seeds for the sampling
        samples = sobol_sequence.sample(self.size, self.dimensions) 

        if aprox == 'float':
            for i, bound in enumerate(self.values.values()):   
                samples[:,i] = samples[:,i] * (bound[1] - bound[0]) + bound[0] # TODO: values are not stretch to the upper bound
        else:
            raise NotImplementedError
            #TODO: implement case when samples must be integers

        return samples
       

class Linear(SamplingMethod):
    """Computes sampling using a linear sequence generator from Numpy"""

    def compute_sampling(self, aprox='float') -> array:
        super().validate_range()
        self.dimensions = super().compute_dimensions()
        samples = numpy.zeros((self.size, self.dimensions))
    
        # Streches sampling values toward the bounds given by the original values
        if aprox == 'float':
            for i, bound in enumerate(self.values.values()):          
                samples[:,i] = numpy.linspace(bound[0], bound[1],self.size) 
        else:
            raise NotImplementedError
            #TODO: implement case when samples must be integers
        return samples


def main():

    #TODO: write unit test based on this example

    components= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}
    size = 10

    sobol = Sobol(size, components)
    samples =sobol.compute_sampling()
    print(samples)

    linear = Linear(size, components)
    samples =linear.compute_sampling()
    print(samples)

if __name__ == "__main__":
    main()