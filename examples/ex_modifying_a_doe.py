"""
Example on how to modify  the parameters (definition) of a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import  DoeVars

# -------------------------------------------
# Create DoE with original parameters
# -------------------------------------------
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
print(doe.info())


# -------------------------------------------
# Modifying DoE  parameters
# -------------------------------------------
# You can modify each parameter by calling the 'variables' attribute 
# and the name of the parameter, and assing a new value , 
# in the same way dictionaries can be modified

# Modify existing parameter
doe.variables['radius'] = 1.0
print('Modified radius:')
print(doe.info())

# Adding parameters
doe.variables['F13']= [-.1, 0.8]
print('New parameter F13')
print(doe.info())
