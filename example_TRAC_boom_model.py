'''
Created on 2020-04-15 12:09:08
Last modified on 2020-04-15 14:08:54
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use TRAC boom model.
'''

# TODO: user must be able to choose for which datapoints wants to run the simulations


#%% imports

# local library
from src.abaqus.models.TRAC_boom import TRACBoomModel


#%% initialization

model_name = 'TRACBOOM'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
height = 2.48450e-03
radius = 9.46834e-03
theta = 1.50926e+02
thickness = 7.10000e-05
length = 5.04000e-01

# material
material_name = 't800_17GSM_120'
layup = [0., 90., 90., 0.]

# boundary conditions
rotation_axis = 2


#%% create model

# create object
model = TRACBoomModel(model_name, job_name, job_description=job_description)

# assemble puzzle
model.assemble_puzzle(height, radius, theta, thickness, length,
                      material_name, layup=layup, rotation_axis=rotation_axis)

# create model
model.create_model()

# write inp
model.write_inp()
