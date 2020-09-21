'''
Created on 2020-05-07 18:04:04
Last modified on 2020-09-21 17:05:59

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

#  abaqus
from abaqus import session

# local library
from f3das.abaqus.models.supercompressible import SupercompressibleModel


# initialization

model_name = 'SUPERCOMPRESSIBLE'

# variable definition
n_longerons = 3
bottom_diameter = 100.
top_diameter = 82.42
pitch = 1.15223e2
young_modulus = 3.5e3
shear_modulus = 1.38631e3
# cross_section_props = {'type': 'circular',
#                        'd': 10}
cross_section_props = {'area': 1.00004e+01,
                       'Ixx': 5.24157e+01,
                       'Iyy': 7.50000e+01,
                       'J': 2.50000e+02}
imperfection = 7.85114e-02


# create linear buckling model

# create object
sim_type = 'lin_buckle'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
lin_buckle_model = SupercompressibleModel(
    model_name, job_info, sim_type, n_longerons, bottom_diameter, top_diameter,
    pitch, young_modulus, shear_modulus, cross_section_props)


# create model
lin_buckle_model.create_model()

# write inp
lin_buckle_model.write_inp(submit=True)


# create riks model

# create object
sim_type = 'riks'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
riks_model = SupercompressibleModel(
    model_name, job_info, sim_type, n_longerons, bottom_diameter, top_diameter,
    pitch, young_modulus, shear_modulus, cross_section_props,
    previous_model=lin_buckle_model, imperfection=imperfection)

# create model
riks_model.create_model()

# write inp
riks_model.write_inp(submit=True)


# post-processing (gui must be opened)

# buckling results
print('\nLinear buckling results')
print(riks_model.previous_model_results)

# riks results
print('\nRiks results')
odb_name = '%s.odb' % riks_model.job_info['name']
odb = session.openOdb(name=odb_name)
riks_results = riks_model.perform_post_processing(odb)
print(riks_results)
