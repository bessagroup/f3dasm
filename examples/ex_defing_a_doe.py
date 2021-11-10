"""
Example on how to define a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import  DoeVars


# define variables for the DoE as a dictionary, for example
vars = {
    'F11':[-0.15, 1], 
    'F12':[-0.1,0.15],
    'F22':[-0.15, 1], 
    'radius': [0.3, 5],  
    'material1': {'STEEL': {'E': [0,100], 'u': {0.1, 0.2, 0.3} }, 
                'CARBON': {'E': 5, 'u': 0.5, 's': 0.1 } },
    'material2': { 'CARBON': {'x': 2} },
    }

doe = DoeVars(vars)

print('DoEVars definition:')
print(doe)

print('\n DoEVars summary information:')
print(doe.info())

print('\n DoEVars as nested dictionary:')
print(doe.as_dict())

print('\n DoEVars as top-level normilized pandas dataframe:')
print(doe.pandas_df(max_level=0))
