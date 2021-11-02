"""
Example on how to write custom sampling menthod using the base class
This example will be moved to the documentation
"""

from f3dasm.doe.sampling import SamplingMethod
from numpy.core.records import array

class MyCustomeSampling(SamplingMethod):

# write custome sampling method to the 'compute_sampling' function
    def compute_sampling(self, aprox='float') -> array:
        # validate values represente ranges, this is a list with two elements
        super().validate_range()

        # compute dimentions of the input data if required by your sampling method
        self.dimensions = super().compute_dimensions()
        
        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        # 1. seeds for the sampling:
 
        # 2. cases for streaching values based on input data types, if required:
        # 2.1. Case for inputs as dictionary:

        # 2.1. Case for inputs as list:

        # 2.2. Case for X:
    
        samples = None  # output must be an Numpy array

        # ----------------------------------------------------------

        return samples

# Use:

# define strain components
values= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}
sampling_size = 5
sampling = MyCustomeSampling(sampling_size, values)
print('Sampling results:', sampling.compute_sampling())


