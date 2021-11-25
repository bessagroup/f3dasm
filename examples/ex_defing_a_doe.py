"""
Example on how to define a DoE
This example will be moved to the documentation
"""

from f3dasm.doe.doevars import  DoeVars
from f3dasm.doe.sampling import SalibSobol

# define variables for the DoE as a dictionary, for example
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

print('DoEVars definition:')
print(doe)

print('\n DoEVars summary information:')
print(doe.info())

# Compute sampling and combinations
doe.do_sampling()

print('\n Pandas dataframe with compbined-sampled values:')
print(doe.data)
