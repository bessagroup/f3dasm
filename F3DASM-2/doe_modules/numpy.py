import numpy as np
from ..src.doe import DOE
from ..src.data import *
import pandas as pd



class doe_numpy(DOE):
    """

        DOE-Module wrap for NUMPY 

    """
 
    def __init__(self, num, variable, method='linear'):

        """ Initialize """

        self.__name__ = "NUMPY"                         # Name module
        super().__init__(num,variable)                  # Initialize base-class


        self.method = method                            # Initialize method of samping
        func = getattr(self,method)                     # Select your method of sampling
        self.DATA(func(),self.keys)                     # Run your method of sampling
        print(self.DATA)
        
   
    def linear(self):

        """ Method: Linear sequence generator """
        points = self.values

        for i, bound in enumerate(self.variable.values()):                  # Stretch the hypercube towards your bounds
            points[:,i] = np.linspace(bound[0], bound[1],self.num)   
        return points
 
