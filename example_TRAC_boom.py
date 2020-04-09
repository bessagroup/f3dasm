'''
Created on 2020-04-06 18:34:34
Last modified on 2020-04-09 19:55:28
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use TRAC boom class.
'''

#%% imports

# abaqus library
from abaqus import mdb, backwardCompatibility
from abaqusConstants import (BUCKLING_MODES, OFF)

# local library
from src.abaqus.geometry.structures import TRACBOOM
from src.abaqus.modelling.step import BuckleStep
from src.abaqus.modelling.bcs import DisplacementBC
from src.abaqus.modelling.bcs import Moment
from src.abaqus.material.abaqus_materials import LaminaMaterial


# TODO: put it working for linear buckling
# TODO: add field output
# TODO: create inp


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'TRACBOOM'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
w = 0.00000e+00
h = 2.48450e-03
r = 9.46834e-03
theta = 1.50926e+02
t = 7.10000e-05
length = 5.04000e-01

# material
material_name = 't800_17GSM_120'
layup = [0., 90., 90., 0.]

# boundary conditions
applied_moment = 2.


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define materials

material = LaminaMaterial(material_name, model=model, create_section=False)


#%% define objects

trac_boom = TRACBOOM(h, r, theta, t, length, material, width=w, layup=layup,
                     name='TRACBOOM')


#%% create part and assembly

# create part
trac_boom.create_part(model)

# create assembly
trac_boom.create_instance(model)


#%% create step

step_name = 'BUCKLE_STEP'
buckle_step = BuckleStep(step_name, minEigen=0., model=model)


#%% apply boundary conditions

# fix ref point minus
position = trac_boom.ref_point_positions[0]
region_name = trac_boom._get_ref_point_name(position)
fix_1 = DisplacementBC('BC_ZMINUS', createStepName=step_name,
                       region=region_name, u1=0., u2=0., u3=0., ur2=0,
                       ur3=0, model=model, buckleCase=BUCKLING_MODES)

# fix ref point plus
position = trac_boom.ref_point_positions[-1]
region_name = trac_boom._get_ref_point_name(position)
fix_2 = DisplacementBC('BC_ZPLUS', createStepName=step_name,
                       region=region_name, u1=0., u2=0., ur2=0, ur3=0,
                       model=model, buckleCase=BUCKLING_MODES)

# apply moment
position = trac_boom.ref_point_positions[-1]
region_name = trac_boom._get_ref_point_name(position)
moment = Moment('APPLIED_MOMENT', createStepName=step_name,
                region=region_name, cm1=applied_moment, model=model)


#%% create job and write inp

modelJob = mdb.Job(name=job_name, model=model_name,
                   description=job_description)
modelJob.writeInput(consistencyChecking=OFF)
