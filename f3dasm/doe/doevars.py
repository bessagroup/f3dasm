
#######################################################
# Data class for the manipulation and transformation  #
# of data within F3DASM                               #
#######################################################

"""
A dataclass for storing variables (features) during the DoE
"""

from dataclasses import dataclass, field
import numpy as np, array
from data import DATA
from typing import Optional

# using Optiona
# attritube: Optional[optional-object] = None


@dataclass
class Material:
    """represents a material"""
    parameters: dict  # materials can be represeted a variable list of name:value pairs


@dataclass
class Microsructure:
    """Represents a microstructue"""

    shape: str
    size: float
    material: Material


@dataclass
class Imperfection:
    """Represents imperfections"""

    #TODO: define the class further, can we define subtypes?

    imperfections: dict # a list parameters defining an imperfection as name:value pairs


@dataclass
class RVE:
    """Represents an Representative Volume Element"""
    
    Lc: float # characteristic length
    material: Material
    microstructure: Microsructure
    dimesionality: int = 2 # e.g. 2D


@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    boundary_conditions: dict  # boundary conditions 
    rve: RVE
    imperfections: Optional[Imperfection] = None

    # feature_names: list = field(init=False)     # Names of features 
    # dimensions: int = field(init=False)         # number of dimensions for the feature space of boundary conditions
    # values: array = field(init=False)           # inita value to construct data array
    
    # def __post_init__(self):
    #     self.feature_names = list(self.variables.keys())
    #     self.dimensions = len(self.variables)
    #     self.values = np.zeros((self.sample_size,self.dimensions))    

    #TODO: implement own method to convert to pandas dataframe, use data.py as example
    
    def __str__(self):

        """ Overwrite print function"""

        print('-----------------------------------------------------')
        print('                       DOE INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        # print('Module Name          :',self.__name__)
        # print('Method               :',self.method)
        print('Feature dimension    :',self.dimensions)     
        print('Feature object count :',self.sample_size)
        return '\n'

    # todo: convert values to array
    # todo: collect names for data colums
    # pass them on to data.py

    def save(self,filename):

        """ Save experiemet doe points as pickle file
        
        Args:
            filename (string): filename for the pickle file
        
        Returns: 
            None
         """  

        data_frame = DATA(self.values,self.feature_names)       # f3dasm data structure, numpy array
        data_frame.to_pickle(filename)

