
from dataclasses import dataclass, field
from abc import ABC, abstractclassmethod
import numpy

from numpy.core.records import array
from SALib.sample import sobol_sequence
from prettyprinter import pprint
import pandas as pd

def validate_range(range) -> None:
    """Checks that list of values contains two numeric values.
    If a range is passed, it will run the validation for that range.
    The default is to validate data in the values attribute of this class.
    Args:
        range (list): min and max values defining a range
    """
    
    if isinstance(range, list) and len(range) == 2:
        if isinstance(range[0], (int, float)) and isinstance(range[1], (int, float)):
            return True
        else:
            raise TypeError("Range of values contains one of more values that are not numeric.")
    else:        
        raise TypeError("Input doesn't contain a valid range of values. Provide a list with a min and max values. E.g. [2.1, 3]") 


def combine(func, args):
    """wrapper for computing combinations of DoE variables"""
    columns = len(args)
    try: 
        result = func(*args)
    finally:
        return numpy.array(result).T.reshape(-1, columns)

def samples_to_dict(samples, column_names) -> dict:
    """Converts sampled values to a dictionary. Each column in the samples-array becomes
    an element of the dictionary

    Args:
        samples(numpy_array): sampled values
        column_names (dict_keys): list of name for the data columns (elements in the dictionary)
    
    Returns:
        Dictionary of length  equals to the no. of columns in the samples-array
    """

    _dictionary = {}

    if len(column_names) != samples.shape[1]:
        raise RuntimeError("sampled array and column_names must be the same length")
    else:
        samples_list = list(samples.T)

        for name in column_names:
            for values in samples_list:
                _dictionary[name] = list(values)
 
    return _dictionary


@dataclass
class SamplingMethod(ABC):
    """Represets a generic sampling method for parameters with a range of values"""

    size: int
    values: dict
    sampling_ranges: dict = field(init=False)
    dimensions: int = field(init=False)

    def __post_init__(self):
        self.sampling_ranges = self.select_values_for_sampling()
        self.dimensions = len(self.sampling_ranges.keys())

    @abstractclassmethod
    def compute_sampling(self, aprox='float') -> array:
        """
        Computes N number of samples for the values represented as ranges of values. E.g. [min, max]
        Args:
            sample_size (int): number of samples to be generated
            values (dic or list): ranges of values to be sample. 
                A dictionary conatig several ranges or a single list of length 2
            aprox (int or float): controls if sampled values are aproximated to an integer or to a float. 
                Default is float
        Returns:
            sampling results as array 
        """

    def select_values_for_sampling(self) -> dict:
        """Selects elements from the values attribute that contain a valid range for sampling.
        Only top-level elements are collected
        Returns:
            dictionary with selected elements
        """
        selected_values = {}
        for k in self.values.keys():
            
            if isinstance(self.values[k], list) and validate_range(self.values[k]):
                selected_values[k] = self.values[k]
            else:
                continue
        
        return selected_values


    def select_fixed_values(self) -> dict:
        """Selects elements from the values attribute that contain values that won't be subject to sampling.
        Only top-level elements will be selected
        Returns:
            dictionary with selected fixed elements
        """
        fixed_values = {}
        for k in self.values.keys():
            if not isinstance(self.values[k], list):
                fixed_values[k] = self.values[k]
            else:
                continue
        
        return fixed_values

    def create_combinations(self, column_names=False, *args) -> array:
        """
        Computes all possible combinations between sampled values and fixed values defined by the DoE variables. 
        Each comibination resesents a distinct set of values for the DoE varialbes.

        Args:
            column_names (bolean): if true, the names of the top-level elements in the 
            DoE variables will be returned along with the combinations. Default is false.

        Returns:
            If column_names = False: array with combination        
            If column_names = True: tuple with array with combination and a list of column names
        """

        samples = self.compute_sampling(*args)
        columns = self.sampling_ranges.keys() 
        fixed_values_dict = self.select_fixed_values() 

        samples_dict = samples_to_dict(samples, column_names=columns)

        combined_dict = {**samples_dict, **fixed_values_dict}
        values_list = list(combined_dict.values())
        combinations = combine(numpy.meshgrid,values_list)
        
        if column_names:
            return ( combinations, list( combined_dict.keys() ) )
        if not column_names:
            return combinations


class SalibSobol(SamplingMethod):
    """Computes sampling using a sobol sequence from SALib"""

    def compute_sampling(self, aprox='float') -> array:
        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        # seeds for the sampling
        samples = sobol_sequence.sample(self.size, self.dimensions) 

        # Streches sampling values toward the bounds given by the original values
        if aprox == 'float':
            for i, bound in enumerate(self.sampling_ranges.values()): 
                samples[:,i] = samples[:,i] * (bound[1] - bound[0]) + bound[0] 
        else:
            raise NotImplementedError
            #TODO: implement cases when samples must be integers

        return samples
       

class NumpyLinear(SamplingMethod):

    """Computes sampling using a linear sequence generator from Numpy"""
    def compute_sampling(self, aprox='float') -> array:
        #----------------------------------------------------------
        # Implementation of Sampling Method
        # ----------------------------------------------------------
        samples = numpy.zeros((self.size, self.dimensions))

        # Streches sampling values toward the bounds given by the original values
        if aprox == 'float':
            for i, bound in enumerate(self.sampling_ranges.values()): 
                samples[:,i] = numpy.linspace(bound[0], bound[1],self.size)
        else:
            raise NotImplementedError
            #TODO: implement cases when samples must be integers

        return samples


def sample_doevars(doe_vars, sampling_method):
    """
    Computes sampling over DoE variables (at level 0) containing a valid range of values using the given sampling method.
    Args:
        doe_vars (object): instance of the DoeVar dataclass
        sampling_method (object): concrete instance of the SamplingMethod with a concrete sampling method.
    Returns:
        list of keys (data colums) and sampled values (numpy array)   
    """
    # print(doe_vars.as_dict())
    vars_dict = doe_vars.as_dict()
    
    results =[]

    for varx in vars_dict.keys():
        sampling_method.values = vars_dict[varx]
        try:
            samples = sampling_method.compute_sampling()
            results.append(numpy.asarray(samples))
            # print(samples)
        except TypeError:
            # Exeptions are handled by the compute_sampling method
            continue
        else:
            # TODO: make the collection of keys generic for the cases where multiple vars require sampling
            data_colums=(list(vars_dict[varx].keys()))

    return (results, data_colums)


def main():

   
    
    # Check if value contains sampling function:


    #type of variables in doeVars
    # 1 variable that need sampling:
    # 'sample2': SalibSobol(2, {'radius': [0.3, 0.5]})
    # variales with constants
    # E: 5
    # variables with vectors
    # 'u': [0.1, 0.2, 0.3] } 
    # varibles with other variables (directories)
    # 'STEEL': {
                #     'sample3': SalibSobol(4, {'E': [1,100]}), 
                #     'u': [0.1, 0.2, 0.3] } 
                # },

    
    vars = {'Fs': SalibSobol(5, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.2, 1]}),
            'R': SalibSobol(2, {'radius': [0.3, 0.5]}),
            'material1': { 'element':
                {'name': 'STEEL',
                    'E': SalibSobol(4, {'E': [1,100]}), 
                    'u': [0.1, 0.2, 0.3] 
                } ,
                'element':
                {'name': 'CARBON', 
                    'E': 5, 
                    'u': 0.5, 
                    's': 0.1  
                } }, 
            'material2': {  
                'name': 'CARBON',  
                'x': 2
                }
            }
            
    print(type(vars['material1']))

    # pprint(list(vars.items()))

    df = pd.json_normalize(vars)
    df_2 =df.to_dict(orient='records')[0]

    def find_functions(vars):
        #list comprenhension

        # thing for thing in list_of_things

        elements_with_functions = [] # list of names
        [ elements_with_functions.append(var) for var in vars.keys() if isinstance(vars[var], SalibSobol) ]

        return elements_with_functions

    print(find_functions(df_2))

        


    # print(isinstance(vars['sample1'], SalibSobol))



    # print(c3[1].compute_sampling())?

    # single param
    # c3 = {"radius": SALibSobol(5, [0.3, 0.5])} # doesn't work

    # VARS = { "parms": SALibSobol(size, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.2, 1], 'radius': [0.3, 5] }) }
      
    # 'material1': {'STEEL': {SALibSobol(size, [0,100], 'u': {0.1, 0.2, 0.3} }, 
    #             'CARBON': {'E': 5, 'u': 0.5, 's': 0.1 } },
    # 'material2': { 'CARBON': {'x': 2} }
    # }


    size = {5}

    # sobol1 = Sobol(size, VARS)
    # samples = sobol1.compute_sampling()
    # print(samples)
    
    # c = sobol1.create_combinations(column_names=True)


if __name__ == "__main__":
    main()