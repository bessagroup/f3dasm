'''
Created on 2020-09-30 11:09:12
Last modified on 2020-09-30 11:12:58

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

# standard library
import os
from collections import OrderedDict

# third-party
import numpy as np
from f3das.run.utils import create_main_file
from f3das.run.utils import create_sim_info
from f3das.run.abaqus import run_sims
from f3das.design_of_experiments import create_doe

# local library
from abaqus_modules.get_results import get_results


# initialization
example_name = 'example_0'
n_pts = 10  # number of points

if os.path.exists(example_name):
    raise Exception('Name already exists')


# variable definition

doe_variables = OrderedDict({'ratio_d': [0.004, 0.073],
                             'ratio_pitch': [.25, 1.5],
                             'ratio_top_diameter': [0., 0.8]})
fixed_variables = {'n_longerons': 3,
                   'bottom_diameter': 100.,
                   'young_modulus': 3500.,
                   'shear_modulus': 1287., }
points = create_doe(n_pts, doe_variables, sample_strat='sobol')

# imperfections
seed = 1
deg2rad = np.pi / 180
m = 4. * deg2rad  # mean
s = 1.2 * deg2rad  # std
sigma = np.sqrt(np.log(s**2 / m**2 + 1))
mu = np.log((m**2) / np.sqrt(s**2 + m**2))
imperfection_dist = {'mean': mu, 'sigma': sigma}
imperfections = np.random.lognormal(size=n_pts, **imperfection_dist)

additional_variables = {'imperfection': imperfections}


# simulations metadata

sim_info_buckle = create_sim_info(
    name='SUPERCOMPRESSIBLE_LIN_BUCKLE',
    abstract_model='abaqus_modules.supercompressible_fnc.lin_buckle',
    job_info={'name': 'Simul_supercompressible_lin_buckle'},
    post_processing_fnc='abaqus_modules.supercompressible_fnc.post_process_lin_buckle')

sim_info_riks = create_sim_info(
    name='SUPERCOMPRESSIBLE_RIKS',
    abstract_model='abaqus_modules.supercompressible_fnc.riks',
    job_info={'name': 'Simul_supercompressible_riks'},
    post_processing_fnc='abaqus_modules.supercompressible_fnc.post_process_riks')

sim_info = [sim_info_buckle, sim_info_riks]


# main file creation

additional_info = {'imperfection_dist': imperfection_dist,
                   'seed': seed}

create_main_file(example_name, doe_variables, points, sim_info,
                 fixed_variables=fixed_variables,
                 additional_variables=additional_variables,
                 additional_info=additional_info,)


# run simulations

run_sims(example_name, points=[0], abaqus_path='abaqus',
         keep_odb=True, pp_fnc=get_results,
         raw_data_filename='raw_data.pkl', delete=False)
