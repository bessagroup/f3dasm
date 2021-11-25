
#######################################################
# Data class for storing variables (parameters) of
# the design of experiments F3DASM                               #
#######################################################

from dataclasses import dataclass, field
import pandas as pd
from f3dasm.doe.sampling import SamplingMethod, SalibSobol

def print_variables(dictionary:dict):
    """Print the top level elements in a dictionary"""

    keys = dictionary.keys()
    for k in keys:
        print(k,':', dictionary[k])

    return None

def find_sampling_vars(doe_vars: dict):
    """Find names of DoeVars that contain a definiton of a sampling method
    Args:
        doe_vars (dict): variables defining a design of experiment
    Returns:
        list of names
    """
    # expand dictionary
    df = pd.json_normalize(doe_vars)
    vars = df.to_dict(orient='records')[0]

    elements_with_functions = [] # list of names
    [ elements_with_functions.append(var) for var in vars.keys() if isinstance(vars[var], SamplingMethod) ]

    return elements_with_functions


@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    variables: dict
    sampling_vars: list = field(init=False)

    def __post_init__(self):
        self.sampling_vars = find_sampling_vars(self.variables)

    def info(self):

        """ Overwrite print function"""

        
        print('-----------------------------------------------------')
        print('                       DOE VARIABLES                     ')
        print('-----------------------------------------------------')
        print_variables(self.variables)
        print('\n')
        return None
    
    def as_dataframe(self, max_level: int =0):
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

    def sample(self):
        """Apply sampling method to sampling variables"""

        for var in self.sampling_vars:
            inner_vars = var.split('.') 
            if len(inner_vars) == 1:
                self.variables[var] = self.variables[var].compute_sampling()
            elif len(inner_vars) == 2:
                self.variables[inner_vars[0]][inner_vars[1]] = self.variables[inner_vars[0]][inner_vars[1]].compute_sampling()
            elif len(inner_vars) == 3:
                self.variables[inner_vars[0]][inner_vars[1]][inner_vars[2]] = self.variables[inner_vars[0]][inner_vars[1]][inner_vars[2]].compute_sampling()
            else:
                raise SyntaxError("DoeVars definition contains too many nested elements. A max of 3 is allowed")
        
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
    vars = {'Fs': SalibSobol(5, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.2, 1]}),
            'R': SalibSobol(2, {'radius': [0.3, 0.5]}),
            'particle': { 
                'name': 'NeoHookean',
                'E': SalibSobol(5 , {'radius': [0.3, 0.5]}), 
                'nu': 0.4 
                } ,
            'matrix': {  
                'name': 'SaintVenant',  
                'E': [5, 200],
                'nu': 0.3
                }
            }

    doe = DoeVars(vars)
    print(doe.sampling_vars)
    print(doe.sample())

if __name__ == "__main__":
    main()