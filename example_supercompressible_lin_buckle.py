'''
Created on 2020-04-20 21:34:22
Last modified on 2020-04-22 19:13:14
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use supercompressible model (linear buckle analysis).
'''


#%% imports

# local library
from src.abaqus.models.supercompressible import SupercompressibleModel


#%% initialization

model_name = 'SUPERCOMPRESSIBLE'
sim_type = 'lin_buckle'
job_name = 'Simul_%s_%s' % (model_name, sim_type)
job_description = ''

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

# create object
model = SupercompressibleModel(model_name, sim_type, job_name, n_vertices_polygon,
                               mast_diameter, mast_pitch, cone_slope, young_modulus,
                               shear_modulus, density, Ixx, Iyy, J, area,
                               job_description=job_description)


# create model
model.create_model()

# write inp
model.write_inp(submit=True)

# dump model
model.dump()
