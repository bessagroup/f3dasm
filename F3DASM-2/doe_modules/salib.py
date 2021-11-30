import numpy as np
from SALib.sample import sobol_sequence
from ..src.doe import DOE
from ..src.data import *
import pandas as pd

class doe_SALib(DOE):
    """

        DOE-Module wrap for SALib 

    """
 
    def __init__(self, num, variable, method='sobol'):

        """ Initialize """

        self.__name__ = "SALib"                         # Name module
        super().__init__(num,variable)                  # Initialize base-class


        self.method = method                            # Initialize method of samping
        func = getattr(self,method)                     # Select your method of sampling
        self.DATA(func(),self.keys)                     # Run your method of sampling
        
   
    def sobol(self):

        """ Method: Sobol sequence generator """

        points = sobol_sequence.sample(self.num,self.dim)                   # Create [0,1] self.dim-dimnesional hypercube
        for i, bound in enumerate(self.variable.values()):                  # Stretch the hypercube towards your bounds
            points[:,i] = points[:,i] * (bound[1] - bound[0]) + bound[0]    
        return points
    
