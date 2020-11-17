'''
Created on 2020-05-07 18:04:04
Last modified on 2020-11-17 11:49:51

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

#  abaqus
from abaqus import session

# third-party
from examples.supercompressible_cls.abaqus_modules.supercompressible import SupercompressibleLinearBuckleModel as LinearBuckleModel
from examples.supercompressible_cls.abaqus_modules.supercompressible import SupercompressibleRiksModel as RiksModel


# initialization

model_name = 'SUPERCOMPRESSIBLE'

# variable definition
geometry_info = {'n_longerons': 3,
                 'bottom_diameter': 100.,
                 'top_diameter': 82.42,
                 'pitch': 1.15223e2,
                 'young_modulus': 3.5e3,
                 'shear_modulus': 1.38631e3,
                 'cross_section_props': {'type': 'circular',
                                         'd': 10}}
imperfection = 7.85114e-02


# create linear buckling model

# create object
sim_type = 'lin_buckle'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
lin_buckle_model = LinearBuckleModel(model_name, job_info, geometry_info)

# create model
lin_buckle_model.create_model()

# write inp
lin_buckle_model.write_inp(submit=True)


# create riks model

# create object
sim_type = 'riks'
job_info = {'name': 'Simul_{}_{}'.format(model_name, sim_type)}
riks_model = RiksModel(model_name, job_info, geometry_info,
                       previous_model=lin_buckle_model, imperfection=imperfection)

# create model
riks_model.create_model()

# write inp
riks_model.write_inp(submit=True)


# post-processing (gui must be opened)

# buckling results
linear_buckle_results = riks_model.previous_model_results

# riks results
odb_name = '%s.odb' % riks_model.job_info['name']
odb = session.openOdb(name=odb_name)
riks_results = riks_model.perform_post_processing(odb)


# TODO: curve?
