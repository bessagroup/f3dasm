'''
Created on 2020-04-20 21:34:22
Last modified on 2020-05-08 13:54:09
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
submit = True


#%% access previous model

# read pickle
with open(previous_model_file, 'rb') as f:
    data = pickle.load(f)
previous_model = data['model']

# variable definition
n_longerons = previous_model.n_longerons
bottom_diameter = previous_model.bottom_diameter
top_diameter = previous_model.top_diameter
pitch = previous_model.pitch
young_modulus = previous_model.young_modulus
shear_modulus = previous_model.shear_modulus
cross_section_props = previous_model.cross_section_props
imperfection = 7.85114e-02


#%% create model

# create object
model = SupercompressibleModel(model_name, sim_type, job_name, n_longerons,
                               bottom_diameter, top_diameter, pitch, young_modulus,
                               shear_modulus, cross_section_props,
                               job_description=job_description,
                               previous_model=previous_model,
                               imperfection=imperfection)

# create model
model.create_model()

# write inp
model.write_inp(submit=submit)

# dump model
model.dump()
