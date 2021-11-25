
#######################################################
# Data class for storing variables (parameters) of
# the design of experiments F3DASM                               #
#######################################################

from dataclasses import dataclass, field
import pandas as pd
from pandas.core.frame import DataFrame
from f3dasm.doe.sampling import SamplingMethod, SalibSobol
import copy
import numpy
import data

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


def deserialize_dictionary(nested_dict: dict):
    """Deserialize a nested dictionary"""
    
    norm_ = pd.json_normalize(nested_dict)
    return norm_.to_dict(orient='records')[0]

def create_combinations(func, args):
        """wrapper for computing combinations of DoE variables"""
        columns = len(args)
        try: 
            result = func(*args)
        finally:
            return numpy.array(result).T.reshape(-1, columns)


@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    variables: dict
    sampling_vars: list = field(init=False)
    data: DataFrame = None

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
        Convert definition of DoeVars into a nested dictionary
        """
        return self.variables

    def sample_doevars(self) -> DataFrame:
        """Apply sampling method to sampling variables, combines sampled value and fixed-values,
        and produces a pandas data frame with all combinations.
        Returns:
            Dataframe with 
        """

        doe_vars = copy.deepcopy(self.variables)
        
        # sample
        for var in self.sampling_vars:
            inner_vars = var.split('.') 
            if len(inner_vars) == 1:
                doe_vars[var] = doe_vars[var].compute_sampling()
            elif len(inner_vars) == 2:
                doe_vars[inner_vars[0]][inner_vars[1]] = doe_vars[inner_vars[0]][inner_vars[1]].compute_sampling()
            elif len(inner_vars) == 3:
                doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]] = doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]].compute_sampling()
            else:
                raise SyntaxError("DoeVars definition contains too many nested elements. A max of 3 is allowed")
    
        # combinations
        sampled_values = list( deserialize_dictionary(doe_vars).values() )
        combinations = create_combinations(numpy.meshgrid, sampled_values)

        # dataframe
        _columns =list( deserialize_dictionary(doe_vars).keys() )
        self.data = pd.DataFrame(combinations,columns=_columns)
        return self.data

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
            'R': SalibSobol(3, {'radius': [0.3, 0.5]}),
            'particle': { 
                'name': 'NeoHookean',
                'E': [0.3, 0.5], 
                'nu': 0.4 
                } ,
            'matrix': {  
                'name': 'SaintVenant',  
                'E': [5, 200, 300],
                'nu': 0.3
                }
            }

    doe = DoeVars(vars)
    # print(doe.sampling_vars)

    samples =doe.sample()
    print(samples)

    # df = pd.json_normalize(samples)
    # vars = df.to_dict(orient='records')[0]

    # print(vars.keys())

    # # print(vars.values())
    # values_list =list(vars.values())
    # print(values_list)

    # combinations = combine(numpy.meshgrid,values_list)
 
    # print(combinations)



if __name__ == "__main__":
    main()