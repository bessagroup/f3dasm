
#######################################################
# Data class for the manipulation and transformation  #
# of data within F3DASM                               #
#######################################################

"""
A dataclass for storing variables (features) during the DoE
"""

from dataclasses import dataclass, field
import numpy as np, array
from .data import DATA

   # TODO: Decouple data from doe-methods

@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    sample_size: int
    variables: dict  # boundary conditions 
    inperfections: any = None
    feature_names: list = field(init=False)     # Names of features 
    dimensions: int = field(init=False)         # number of dimensions for the feature space of boundary conditions
    values: array = field(init=False)

    def __post_init__(self):
        self.feature_names = list(self.variables.keys())
        self.dimensions = len(self.variables)
        self.values = np.zeros((self.sample_size,self.dimensions))    

    def __str__(self):

        """ Overwrite print function"""

        print('-----------------------------------------------------')
        print('                       DOE INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        print('Module Name          :',self.__name__)
        print('Method               :',self.method)
        print('Feature dimension    :',self.dimensions)     
        print('Feature object count :',self.sample_size)
        return '\n'

    def save(self,filename):

        """ Save experiemet doe points as pickle file
        
        Args:
            filename (string): filename for the pickle file
        
        Returns: 
            None
         """  

        data_frame = DATA(self.values,self.feature_names)       # f3dasm data structure, numpy array
        data_frame.to_pickle(filename)

