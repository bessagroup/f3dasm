'''
Created on 2020-04-28 00:23:38
Last modified on 2020-05-07 14:54:58
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''


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
from f3das.design_of_experiments.convert_inputs import convert_supercompressible


#%% initialization

seed = 1
np.random.seed(seed)
example_name = 'example_supercompressible_test'
simuls_dir_name = 'analyses'
simul_pkl_name = 'simul'

n = 50000  # number of points
doe_variables = OrderedDict({'ratio_area': [1.17e-5, 4.1e-3],
                             'ratio_Ixx': [1.128e-11, 1.4e-6],
                             'ratio_Iyy': [1.128e-11, 1.4e-6],
                             'ratio_J': [1.353e-11, 7.77e-6],
                             'ratio_pitch': [.25, 1.5],
                             'ratio_top_diameter': [0., .8],
                             'ratio_shear_modulus': [.035, .45]})
fixed_variables = {'n_longerons': 3,
                   'bottom_diameter': 100.,
                   'young_modulus': 3500.}
section = ''

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
        'section': section,
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

# populate folders
for i in range(n):

    # get dir
    dir_name = os.path.join(simuls_dir, 'DoE_point%i' % i)

    # access required variables
    variables = {name: float(value) for name, value in zip(doe_variables_ls, points_sobol[i, :])}
    variables.update(fixed_variables)
    variables['imperfection'] = float(imperfections[i])
    variables = convert_supercompressible(variables, section=section)

    # create dict and dump dict
    data = OrderedDict({'abstract_model': abstract_model,
                        'variables': variables,
                        'sim_info': sim_info})
    with open(os.path.join(dir_name, '%s.pkl' % simul_pkl_name), 'wb') as file:
        pickle.dump(data, file, protocol=2)
