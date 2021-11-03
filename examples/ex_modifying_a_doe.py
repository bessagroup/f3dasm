"""
Example on how to modify  the parameters (definition) of a DoE
This example will be moved to the documentation
"""
from f3dasm.doe.doevars import CilinderMicrostructure, DoeVars, Material, CircleMicrostructure, RVE, DoeVars


# -------------------------------------------
# Create DoE with original parameters
# -------------------------------------------
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

mat3 = Material({'elements': [{'name': 'BRONZE', 'params': {'param1': 5, 'param2': 6, 'param4': 'value4'}}
                    ]
                })

# create a microstructure 
#circle
micro = CircleMicrostructure(material=mat2, diameter=0.3)

# create RVE
rve = RVE(Lc=4,material=mat1, microstructure=micro, dimesionality=2)
doe = DoeVars(boundary_conditions=components, rve=rve)

print('DoEVars Original parameters:')
print(doe)


# -------------------------------------------
# Modifying DoE  parameters
# -------------------------------------------

#.  1 Change material of microstructure
doe.rve.microstructure.material = mat3
print('\n', 'Changed Material of microstructure:')
print('\n', doe)

#. 3 Change parameters of geometery for DoE
micro.diameter = 0.5
doe.rve.microstructure = micro
print('\n', 'Changed diameter of microstructure:')
print('\n', doe)

#.  2 Change  geometry of microstructure
new_micro = CilinderMicrostructure(material=mat1, diameter=0.3, length=1.0)
doe.rve.microstructure = new_micro
print('\n', 'Changed microstructure geometry:')
print('\n', doe)
