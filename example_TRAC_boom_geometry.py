'''
Created on 2020-04-06 18:34:34
Last modified on 2020-04-15 12:29:57
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use TRAC boom class (geometry).
'''


#%% imports

# abaqus
from abaqus import mdb, backwardCompatibility


# local library
from src.abaqus.geometry.structures import TRACBoom
from src.abaqus.material.abaqus_materials import LaminaMaterial


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

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


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define materials

material = LaminaMaterial(material_name, model=model, create_section=False)


#%% define objects

trac_boom = TRACBoom(height, radius, theta, thickness, length, material,
                     layup=layup, name='TRACBOOM')


#%% create part and assembly

# create part
trac_boom.create_part(model)

# create assembly
trac_boom.create_instance(model)
