
from dataclasses import dataclass
from abc import ABC, abstractclassmethod
import typing
import numpy

from numpy.core.records import array
from SALib.sample import sobol_sequence
from torch._C import Size

@dataclass
class SamplingMethod(ABC):
    """Represets a generic sampling method for parmeters with a range of values"""

    size: int
    values: any # list or dict

    @abstractclassmethod
    def compute_sampling(self, aprox='float') -> array:
        """
        Computes N number of samples for the values represented as ranges. E.g. [min, max]
        Values 
        Args:
            sample_size (int): number of samples to be generated
            values (dic or list): ranges of values to be sample. A dictionary conatig several ranges or a single list of lenth 2
            aprox (int or float): controls if sampled values are aproximated to an integer or to a float. Default is float
        Returns:
            sampling results as array 
        """

    def check_input_type(self):
        """
        Check the data type of values attribute.
        Returns:
            dict or list data type
        """
        if isinstance(self.values, dict):
            input_type = {}
        elif isinstance(self.values, list):
            input_type = []
        else:
            raise TypeError("Input values must be dictionary or list")

        return input_type

    def validate_range(self) -> None:
        """Checks that a range of values contains two numeric values"""
        
        if isinstance(self.check_input_type(), dict):
            for value in self.values.values():
                if isinstance(value, list) and len(value) == 2:
                    if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                        return True
                    else:
                        raise TypeError("Range of values contains one of more values that are not numeric.")
                else:
                    
                    raise TypeError("Input don't contain a valid range of vlaues. E.g. [2.1, 3]")
        else: # when input_type is list
            if len(self.values) == 2 and isinstance(self.values[0], (int, float)) and isinstance(self.values[1], (int, float)):
                print('input is list')
                return True
            else:
                print(self.values)
                raise TypeError("Range of values contains one of more values that are not numeric.")
                

    def compute_dimensions(self) -> int:
        """Computes the number of dimentions for the sampling methon
        Args:
            values (dict): values the sampling will be applied to
        Returns (int): number of elements in values
        """
        if isinstance(self.check_input_type(), dict):
            print('type', type(self.check_input_type))
            return len(self.values.keys())
        else: # for the case of a list, dimentions must always be 1
            return 1



class Sobol(SamplingMethod):
    """Computes sampling using a sobol sequence from SALib"""

    def compute_sampling(self, aprox='float') -> array:
        super().validate_range()
        self.dimensions = super().compute_dimensions()
        print(self.dimensions)

        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        # seeds for the sampling
        samples = sobol_sequence.sample(self.size, self.dimensions) 

        # Streches sampling values toward the bounds given by the original values
        if aprox == 'float' and isinstance(self.check_input_type(), dict):
            for i, bound in enumerate(self.values.values()): 
                samples[:,i] = samples[:,i] * (bound[1] - bound[0]) + bound[0] # TODO: Are values being strechted to the upper bound?
        elif aprox == 'float' and isinstance(self.check_input_type(), list):
            samples = samples * (self.values[1] - self.values[0]) + self.values[0]
        else:
            raise NotImplementedError
            #TODO: implement case when samples must be integers

        return samples
       

class Linear(SamplingMethod):
    """Computes sampling using a linear sequence generator from Numpy"""

    def compute_sampling(self, aprox='float') -> array:
        super().validate_range()
        self.dimensions = super().compute_dimensions()

        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        samples = numpy.zeros((self.size, self.dimensions))

        # Streches sampling values toward the bounds given by the original values
        if aprox == 'float' and isinstance(self.check_input_type(), dict):
            for i, bound in enumerate(self.values.values()): 
                samples[:,i] = numpy.linspace(bound[0], bound[1],self.size)
        elif aprox == 'float' and isinstance(self.check_input_type(), list):
            print('input type is list')
            samples = numpy.linspace(self.values[0], self.values[1],self.size)
            # samples = samples * (self.values[1] - self.values[0]) + self.values[0]
        else:
            raise NotImplementedError
            #TODO: implement case when samples must be integers

        return samples


def main():

    #TODO: write unit test based on this example

    components= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}
    size = 10

    # sobol1 = Sobol(size, components)
    # samples =sobol1.compute_sampling()
    # # print(sobol1.check_input_type())
    # print(samples)

    # print('List Case, dimentions=1')
    # var_range = [5, 10]
    # sobol2 = Sobol(size, var_range)
    # samples = sobol2.compute_sampling()
    # print(samples)

    linear = Linear(size, components)
    samples =linear.compute_sampling()
    print(samples)

    var_range = [5, 10]
    linear2 = Linear(size, var_range)
    samples = linear2.compute_sampling()
    print(samples)



if __name__ == "__main__":
    main()