'''
Created on 2020-04-25 19:45:52
Last modified on 2020-05-07 21:45:13
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create 3d DoE and folder structure for supercompressible case.
'''

# TODO: make function

#%% imports

# standard library
import os
from collections import OrderedDict
import pickle

# third-party
import numpy as np
from SALib.sample import sobol_sequence
import pandas as pd

# local library
from f3das.design_of_experiments.transform_inputs import transform_inputs_supercompressible


#%% initialization

seed = 1
np.random.seed(seed)
example_name = 'example_supercompressible_3d_circular'
simuls_dir_name = 'analyses'
simul_pkl_name = 'simul'

n = 10000  # number of points
doe_variables = OrderedDict({'ratio_d': [0.004, 0.073],
                             'ratio_pitch': [.25, 1.5],
                             'ratio_top_diameter': [0., 0.8]})
fixed_variables = {'n_longerons': 3,
                   'bottom_diameter': 100.,
                   'young_modulus': 3500.,
                   'shear_modulus': 1287.,
                   'section': 'circular', }

# imperfections
deg2rad = np.pi / 180
m = 4. * deg2rad  # mean
s = 1.2 * deg2rad  # std


#%% create dir and folder structure

os.mkdir(example_name)
simuls_dir = os.path.join(example_name, simuls_dir_name)
os.mkdir(simuls_dir)

for i in range(n):
    os.mkdir(os.path.join(simuls_dir, 'DoE_point%i' % i))


#%% deal with imperfection

sigma = np.sqrt(np.log(s**2 / m**2 + 1))
mu = np.log((m**2) / np.sqrt(s**2 + m**2))
imperfection_dist = {'mean': mu, 'sigma': sigma}
imperfections = np.random.lognormal(size=n, **imperfection_dist)


#%% get sobol sequence

dim = len(doe_variables)
points_sobol = sobol_sequence.sample(n, dim)
for i, lims in enumerate(doe_variables.values()):
    points_sobol[:, i] = points_sobol[:, i] * (lims[1] - lims[0]) + lims[0]


#%% store DoE and related info

doe_variables_ls = list(doe_variables.keys())
points = pd.DataFrame(points_sobol, columns=doe_variables_ls)
data = {'doe_variables': doe_variables,
        'points': points,
        'fixed_variables': fixed_variables,
        'imperfections': {'dist': imperfection_dist,
                          'imperfections': imperfections},
        'seed': seed}
with open(os.path.join(example_name, 'DoE.pkl'), 'wb') as file:
    pickle.dump(data, file)


#%% populate DoE point folders

# shared data
abstract_model = 'f3das.abaqus.models.supercompressible.SupercompressibleModel'
sim_info = OrderedDict({'SUPERCOMPRESSIBLE_LIN_BUCKLE':
                        {'sim_type': 'lin_buckle',
                         'job_name': 'Simul_supercompressible_lin_buckle',
                         'job_description': ''},
                        'SUPERCOMPRESSIBLE_RIKS':
                        {'sim_type': 'riks',
                         'job_name': 'Simul_supercompressible_riks',
                         'job_description': ''}})

# TODO: add information to perform automatic post-processing

# populate folders
for i in range(n):

    # get dir
    dir_name = os.path.join(simuls_dir, 'DoE_point%i' % i)

    # access required variables
    variables = {name: float(value) for name, value in zip(doe_variables_ls, points_sobol[i, :])}
    variables.update(fixed_variables)
    variables['imperfection'] = float(imperfections[i])
    variables = transform_inputs_supercompressible(variables)

    # create dict and dump dict
    data = OrderedDict({'abstract_model': abstract_model,
                        'variables': variables,
                        'sim_info': sim_info})
    with open(os.path.join(dir_name, '%s.pkl' % simul_pkl_name), 'wb') as file:
        pickle.dump(data, file, protocol=2)
