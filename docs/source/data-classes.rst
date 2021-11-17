Implementation of DoE using Data Classes:
=========================================

The code bellow was an attempt to define a common data structure for the inputs for the DoE.


.. code-block::

    #######################################################
    # Data class for the manipulation and transformation  #
    # of data within F3DASM                               #
    #######################################################

    """
    A dataclass for storing variables (features) during the DoE
    """

    from dataclasses import dataclass, asdict
    import numpy as np, array
    # from data import DATA
    from typing import Optional
    from abc import ABC
    import pandas as pd

    # using Optional
    # attritube: Optional[optional-object] = None

    @dataclass
    class Material:
        """represents a material"""
        parameters: dict  # materials can be represeted a variable list of name:value pairs


    @dataclass
    class BaseMicrosructure(ABC):
        """Represents a generic microstructue"""
        
        material: Material


    @dataclass
    class CircleMicrostructure(BaseMicrosructure):
        """Represents a microstructure for a circle"""

        diameter: any # can be a single value or a range of values
        shape: str = 'Circle'


    @dataclass
    class CilinderMicrostructure(BaseMicrosructure):
        """Represents a microstructure for """

        diameter: any # can be a single value or a range of values
        length: float
        shape: str = 'Cilinder'


    @dataclass
    class Imperfection:
        """Represents imperfections"""

        #TODO: define the class further, can we define subtypes?
        imperfections: dict # a list parameters defining an imperfection as name:value pairs


    @dataclass
    class REV:
        """Represents an Representative Elementary Volume"""
        
        Lc: float # characteristic length
        material: Material
        microstructure: BaseMicrosructure
        dimesionality: int = 2 # e.g. 2D


    @dataclass
    class DoeVars:
        """Parameters for the design of experiments"""

        boundary_conditions: dict  # boundary conditions 
        rev: REV
        imperfections: Optional[Imperfection] = None

        def info(self):

            """ Overwrite print function"""

            print('-----------------------------------------------------')
            print('                       DOE INFO                      ')
            print('-----------------------------------------------------')
            print('\n')
            print('Boundary conditions:',self.boundary_conditions)
            print('REV dimensions:',self.rev.dimesionality)
            print('REV Lc:',self.rev.Lc)
            print('REV material:',self.rev.material.parameters)
            print('Microstructure shape:',self.rev.microstructure.shape)
            print('Microstructure material:',self.rev.microstructure.material.parameters)
            print('Imperfections:',self.imperfections)
            return '\n'

        # todo: convert values to array
        # todo: collect names for data colums
        # pass them on to data.py
        #TODO: implement own method to convert to pandas dataframe, use data.py as example
        
        def pandas_df(self, max_level=None):
            """
            Converts DoeVars into a normilized flat table.
            Args:
                max_level: Max number of levels(depth of dict) to normalize. if None, normalizes all levels.
            Returns:
                pandas dataframe
            """
            pd.set_option('display.max_columns', None) # show all colums in the dataframe
            normalized_dataframe = pd.json_normalize(asdict(self), max_level=max_level)
            return normalized_dataframe

        def as_dict(self):
            """
            Convert DoeVars into a nested dictionary
            """
            return asdict(self)


        def save(self,filename):

            """ Save doe-vars as pickle file
            
            Args:
                filename (string): filename for the pickle file
        
            Returns: 
                None
            """  

            data_frame = self.pandas_df()       # f3dasm data structure, numpy array
            data_frame.to_pickle(filename)
