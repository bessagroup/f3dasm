'''
Created on 2020-04-06 18:34:34
Last modified on 2020-04-09 15:33:57
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

# local library
from src.abaqus.geometry.structures import TRACBOOM


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'TRAC-BOOM'
job_name = 'Sim_' + model_name
job_description = ''

# geometry
w = 0.00000e+00
h = 2.48450e-03
r = 9.46834e-03
theta = 1.50926e+02
t = 7.10000e-05
length = 5.04000e-01


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define materials

material = None


#%% define objects

trac_boom = TRACBOOM(h, r, theta, t, length, material, width=w,
                     name='TRACBOOM')


#%% create part and assembly

# create part
trac_boom.create_part(model)

# create assembly
trac_boom.create_instance(model)
