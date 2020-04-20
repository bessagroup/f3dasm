'''
Created on 2020-04-20 12:38:43
Last modified on 2020-04-20 22:51:20
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
from src.abaqus.geometry.structures import Supercompressible


#%% initialization

backwardCompatibility.setValues(reportDeprecated=False)

model_name = 'SUPERCOMPRESSIBLE'

# geometry
n_vertices_polygon = 3
mast_diameter = 100.
mast_pitch = 115.223
cone_slope = 1.75806e-01
young_modulus = 3.50000e+03
shear_modulus = 1.38631e+03
density = 0.00124
Ixx = 6.12244e+01
Iyy = 1.26357e+01
J = 2.10974e+02
area = 1.54038e+01


#%% create model

model = mdb.Model(name=model_name)

if 'Model-1' in mdb.models.keys():
    del mdb.models['Model-1']


#%% define objects

supercompressible = Supercompressible(n_vertices_polygon, mast_diameter, mast_pitch,
                                      cone_slope, young_modulus, shear_modulus,
                                      density, Ixx, Iyy, J, area)


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
