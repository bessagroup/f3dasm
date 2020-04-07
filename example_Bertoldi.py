'''
Created on 2020-03-24 14:52:25
Last modified on 2020-04-07 16:07:56
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to generate a 2d RVE (in particular, Bertoldi's RVE).
'''


#%% imports

# abaqus library
from abaqus import mdb, backwardCompatibility

# local library
from src.abq.geometry.rve import BertoldiExampleRVE


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'RVE2D'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
length = 4.5
width = 3.5
center = (1.75, 1.75)

# inner shape
r_0 = 1.
c_1 = 2.98642006e-01
c_2 = 1.37136137e-01


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define objects

rve = BertoldiExampleRVE(length, width, center, r_0, c_1, c_2,
                         n_points=100, name='BERTOLDI_RVE')


#%% create part and assembly

# create part and generate mesh
rve.create_part(model)
rve.generate_mesh_for_pbcs()

# create assembly
rve.create_instance(model)

# apply boundary conditions
rve.apply_pbcs_constraints(model)
