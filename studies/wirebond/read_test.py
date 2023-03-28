# import sys
# print()
# print(sys.path)
# print()
import os
print(os.getcwd())
# print(os.path.dirname(__file__))

import fileinput

WThk = 0.3
FL = 0.5
MeshSize_XY = 1.0

geometric_inputs = 'src/f3dasm/examples/benchmark_multifidelity/resources/wirebond_multifidelity_source/01_Geometric_Inputs.txt'
mesh_parameters = 'src/f3dasm/examples/benchmark_multifidelity/resources/wirebond_multifidelity_source/02_Mesh_Parameters.txt'

geometric_inputs_file = fileinput.FileInput(files=geometric_inputs, inplace=True)
for i, line in enumerate(geometric_inputs_file):
    if "WThk = " in line: # Wire thickness
        print('WThk = ' + str(WThk), end='\n')
    elif "FL = " in line: # Foot length
        print('FL = ' + str(FL), end='\n')
    else:
        print(line, end='')

mesh_parameters_file = fileinput.FileInput(files=mesh_parameters, inplace=True)
for i, line in enumerate(mesh_parameters_file):
    if "MeshSize_XY = " in line: # XY-plane mesh size
        print('MeshSize_XY = ' + str(MeshSize_XY), end='\n')
    else:
        print(line, end='')