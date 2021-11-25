
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

    def do_sampling(self) -> DataFrame:
        """Apply sampling method to sampling variables, combines sampled value and fixed-values,
        and produces a pandas data frame with all combinations.
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
        if self.data is None:
            print("There's no data to save. Run DoeVars.sample_doevars() first")
        else:
            self.data.to_pickle(filename)

