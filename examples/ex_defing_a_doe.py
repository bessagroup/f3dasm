"""
Example on how to define a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import CilinderMicrostructure, DoeVars, Material, CircleMicrostructure, REV, DoeVars
from dataclasses import asdict


# define strain components
components= {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]}

# define material for RVE and microstructure
# material are must be defined as an list of 'elements', which 
# are declared as dictionarinaries containing a 'name' and a list of parameters
# ('params') also declared as dictionaries. However, the names and number of elements 
# and parameters can vary
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

#cilinder
micro2 = CilinderMicrostructure(material=mat1, diameter=0.3, length=1.0)

# create RVE
rev = REV(Lc=4,material=mat1, microstructure=micro, dimesionality=2)
doe = DoeVars(boundary_conditions=components, rev=rev)

print('DoEVars definition:')
print(doe)


print('\n DoEVars summary information:')
print(doe.info())

print('\n DoEVars as nested dictionary:')
print(asdict(doe))

