'''
Created on 2020-04-20 12:38:43
Last modified on 2020-05-07 20:02:06
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use supercompressible (geometry).
'''


#%% imports

# abaqus
from abaqus import mdb, backwardCompatibility

# standard library
import pickle

# local library
from f3das.abaqus.geometry.structures import Supercompressible


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'SUPERCOMPRESSIBLE'

# geometry
n_longerons = 3
bottom_diameter = 100.
top_diameter = 82.4
pitch = 115.22
# cross_section_props = {
#     'type': 'generalized',
#     'Ixx': 6.12244e1, 'Iyy': 1.26357e1, 'J': 2.10974e2, 'area': 1.54038e1}
cross_section_props = {'type': 'circular',
                       'd': 10}
young_modulus = 3.5e3
shear_modulus = 1.38631e3


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define objects

supercompressible = Supercompressible(n_longerons, bottom_diameter,
                                      top_diameter, pitch,
                                      young_modulus, shear_modulus,
                                      cross_section_props)


#%% create part and assembly

# create part
supercompressible.create_part(model)

# create assembly
supercompressible.create_instance(model)


#%% dump object

data = {'supercompressible': supercompressible}
filename = 'supercompressible.pickle'
with open(filename, 'wb') as f:
    pickle.dump(data, f)

# with open(filename, 'rb') as f:
#     data = pickle.load(f)
