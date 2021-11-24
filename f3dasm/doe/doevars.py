
#######################################################
# Data class for storing variables (parameters) of
# the design of experiments F3DASM                               #
#######################################################

from dataclasses import dataclass, field
import pandas as pd

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
    sampling_vars: list = field(init=False)
    

    def info(self):

        """ Overwrite print function"""

        
        print('-----------------------------------------------------')
        print('                       DOE VARIABLES                     ')
        print('-----------------------------------------------------')
        print_variables(self.variables)
        print('\n')
        return None
    
    def as_dataframe(self, max_level=0):
        """
        Converts DoE variables into a normilized Pandas data frame. By default only the elements on the top level are serialized.
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
