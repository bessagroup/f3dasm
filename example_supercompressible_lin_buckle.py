'''
Created on 2020-04-20 21:34:22
Last modified on 2020-04-26 00:13:10
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Show how to use supercompressible model (linear buckle analysis).
'''


#%% imports

# local library
from f3das.abaqus.models.supercompressible import SupercompressibleModel


#%% initialization

model_name = 'SUPERCOMPRESSIBLE'
sim_type = 'lin_buckle'
job_name = 'Simul_%s_%s' % (model_name, sim_type)
job_description = ''
submit = True

# variable definition
n_longerons = 3
bottom_diameter = 100.
top_diameter = 82.42
pitch = 1.15223e2
young_modulus = 3.5e3
shear_modulus = 1.38631e3
Ixx = 6.12244e1
Iyy = 1.26357e1
J = 2.10974e2
area = 1.54038e1


#%% create model

# create object
model = SupercompressibleModel(model_name, sim_type, job_name, n_longerons,
                               bottom_diameter, top_diameter, pitch, young_modulus,
                               shear_modulus, Ixx, Iyy, J, area,
                               job_description=job_description)


# create model
model.create_model()

# write inp
model.write_inp(submit=submit)

# dump model
model.dump()
