"""
Example on how to write custom sampling menthod using the base class
This example will be moved to the documentation
"""

from f3dasm.doe.sampling import SamplingMethod
from numpy.core.records import array
import numpy as np

class MyCustomeSampling(SamplingMethod):

# write custome sampling method to the 'compute_sampling' function
    def compute_sampling(self, aprox='float') -> array:
        
        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        # 1. seeds for the sampling. Example:

        samples = np.random.randint(10, size=(self.size, self.dimensions )) # output must be an Numpy array
 
        # 2. cases for streaching values based on input data types, if required:
        # 2.1. Case for  output as float, aprox = 'float'

        # 2.1. Case for output as integer, aprox = 'int' 
    
        # ----------------------------------------------------------

        return samples

# Use:

# define DoE parameters
values= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}
sampling_size = 10
sampling = MyCustomeSampling(sampling_size, values)
print('Sampling results: \n', sampling.compute_sampling())


