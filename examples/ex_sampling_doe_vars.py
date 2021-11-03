"""
Example on how to define a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import DoeVars, Material, CircleMicrostructure, REV, DoeVars
from f3dasm.doe.sampling import sample_doevars, Sobol
from f3dasm.doe.data import DATA
from dataclasses import asdict


# define strain components
components= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}

mat1 = Material({'elements': [ {'name': 'STEEL', 'params': {'param1': 1, 'param2': 2}},
                    {'name': 'CARBON', 'params': {'param1': 3, 'param2': 4, 'param3': 'value3'} }
                    ]
                })

mat2 = Material({'elements': [{'name': 'CARBON', 'params': {'param1': 3, 'param2': 4, 'param3': 'value3'}}
                    ]
                })


# create a microstructure 
#circle
micro = CircleMicrostructure(material=mat2, diameter=0.3)

# create RVE and DoeVars
rev = REV(Lc=4,material=mat1, microstructure=micro, dimesionality=2)
doe = DoeVars(boundary_conditions=components, rev=rev)

# -------------------------------
# Sampling of Doe Vars
#--------------------------------

sobol = Sobol(size=5, values='') 
sampling =sample_doevars(doe_vars=doe, sampling_method=sobol)

# pipe data to common Data interface
data = DATA(sampling[0][0], keys=sampling[1])
print(data)
