'''
Created on 2020-04-20 21:34:22
Last modified on 2020-04-22 22:15:59
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use supercompressible model (Riks analysis).
'''


#%% imports

# standard library
import pickle

# local library
from f3das.abaqus.models.supercompressible import SupercompressibleModel


#%% initialization

model_name = 'SUPERCOMPRESSIBLE'
sim_type = 'riks'
job_name = 'Simul_%s_%s' % (model_name, sim_type)
job_description = ''

previous_model_file = 'Simul_SUPERCOMPRESSIBLE_lin_buckle.pickle'


#%% access previous model

# read pickle
with open(previous_model_file, 'rb') as f:
    data = pickle.load(f)
previous_model = data['model']
mode_amplitude = 7.85114e-02

# geometry
n_vertices_polygon = previous_model.n_vertices_polygon
mast_diameter = previous_model.mast_diameter
mast_pitch = previous_model.mast_pitch
cone_slope = previous_model.cone_slope
young_modulus = previous_model.young_modulus
shear_modulus = previous_model.shear_modulus
density = previous_model.density
Ixx = previous_model.Ixx
Iyy = previous_model.Iyy
J = previous_model.J
area = previous_model.area


#%% create model

# create object
model = SupercompressibleModel(model_name, sim_type, job_name, n_vertices_polygon,
                               mast_diameter, mast_pitch, cone_slope, young_modulus,
                               shear_modulus, density, Ixx, Iyy, J, area,
                               job_description=job_description,
                               previous_model=previous_model,
                               mode_amplitude=mode_amplitude)

# create model
model.create_model()

# write inp
model.write_inp()

# dump model
model.dump()
