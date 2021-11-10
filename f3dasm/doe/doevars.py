
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
import pprint


## working example:
## Only parameters in the top level will be deserialized into a dataframe.
d = {'F11':[-0.15, 1], 
    'F12':[-0.1,0.15],
    'F22':[-0.15, 1], 
    'radius': [0.3, 5],  
    'material1': {'STEEL': {
                    'E': [0,100], 
                    'u': {0.1, 0.2, 0.3} 
                        }, 
                'CARBON': {
                    'E': 5, 
                    'u': 0.5, 
                    's': 0.1 
                    } 
                },
    'material2': {
                'CARBON': {
                    'x': 2
                    } 
                },
     }


def print_variables(dictionary:dict):
    """Print the top level elements in a dictionary"""

    keys = dictionary.keys()
    for k in keys:
        print(k,':', dictionary[k])

    return None


@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    variables: dict

    def info(self):

        """ Overwrite print function"""

        
        print('-----------------------------------------------------')
        print('                       DOE VARIABLES                     ')
        print('-----------------------------------------------------')
        print_variables(self.variables)
        print('\n')
        return None

    # todo: convert values to array
    # todo: collect names for data colums
    # pass them on to data.py
    #TODO: implement own method to convert to pandas dataframe, use data.py as example
    
    def pandas_df(self, max_level=0):
        """
        Converts DoE variables into a normilized data frame. By default only the elements on the top level are serialized.
        Args:
            max_level: Max number of levels(depth of dict) to normalize. if None, normalizes all levels.
        Returns:
            pandas dataframe
        """
        pd.set_option('display.max_columns', None) # show all colums in the dataframe
        normalized_dataframe = pd.json_normalize(self.variables, max_level=max_level)

        return normalized_dataframe

    def as_dict(self):
        """
        Convert DoeVars into a nested dictionary
        """
        return self.variables


    def save(self,filename):

        """ Save doe-vars as Pandas data frame in a pickle file
        Args:
            filename (string): filename for the pickle file
    
        Returns: 
            None
         """  

        data_frame = self.pandas_df()       # f3dasm data structure, numpy array
        data_frame.to_pickle(filename)


def main():

    from dataclasses import asdict
    import json
    import pandas as pd

    d = {'F11':[-0.15, 1], 
    'F12':[-0.1,0.15],
    'F22':[-0.15, 1], 
    'radius': [0.3, 5],  
    'material1': {'STEEL': {
                    'E': [0,100], 
                    'u': {0.1, 0.2, 0.3} 
                        }, 
                'CARBON': {
                    'E': 5, 
                    'u': 0.5, 
                    's': 0.1 
                    } 
                },
    'material2': {
                'CARBON': {
                    'x': 2
                    } 
                },
     }


    doe = DoeVars(variables=d)

    print(doe)

    doe.info()
    # print(doe.as_dict())
    print(doe.pandas_df())
    doe.save('/home/manuel/Documents/development/f3dasm/F3DASM-1/examples/data-frame.pkl')
    


    # print(doe.info())
    # print(asdict(doe))
    # print(json.dumps(asdict(doe)))
     
    # pd.set_option('display.max_columns', None)
    # norm_pd = pd.json_normalize(asdict(doe), max_level=1)
    # df = norm_pd.rename(columns= {"rev.Lc": "Lc"}, errors="raise")
    # # print(df)
    # print(doe.pandas_df())

    # print(doe.as_dict())



if __name__ == "__main__":
    main()