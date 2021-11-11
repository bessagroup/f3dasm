"""
Example on how to define a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import DoeVars, Material, CircleMicrostructure, REV, DoeVars
from f3dasm.doe.sampling import sample_doevars, Sobol
from f3dasm.doe.data import DATA
from dataclasses import asdict


# define DoE parameters:

VARS = {
'F11':[-0.15, 1], 
'F12':[-0.1,0.15],
'F22':[-0.2, 1], 
'radius': [0.3, 5],  
'material1': {'STEEL': {'E': [0,100], 'u': {0.1, 0.2, 0.3} }, 
            'CARBON': {'E': 5, 'u': 0.5, 's': 0.1 } },
'material2': { 'CARBON': {'x': 2} },
}


# -------------------------------
# Sampling of Doe Vars
#--------------------------------

sobol = Sobol(size=5, values=VARS) 
sampling = sobol.compute_sampling()

# pipe data to common Data interface
# data = DATA(sampling[0][0], keys=sampling[1])
# print(data)
