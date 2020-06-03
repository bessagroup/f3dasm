'''
Created on 2020-05-05 20:07:00
Last modified on 2020-05-11 16:38:24
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''


#%% imports

# standard library
import os
import pickle

# third-party
import pandas as pd
import numpy as np

# local library
from f3das.misc.file_handling import collect_folder_names
from f3das.misc.misc import get_int_number_from_str
from f3das.post_processing.common import get_results
from f3das.post_processing.supercompressible import get_results_lin_buckle
from f3das.post_processing.supercompressible import read_and_clean_results_riks


#%% initialization

example_name = 'example_supercompressible_3d'
sim_dir = 'analyses'
pkl_filename = 'DoE.pkl'
pkl_filename_output = 'DoE_results_2.pkl'
max_strain = 0.02


#%% get current pandas

# read file
with open(os.path.join(example_name, pkl_filename), 'rb') as file:
    global_data = pickle.load(file)

points = global_data['points']
n_pts = len(points)


#%% get results

# name of the outputs
var_names = ['coilable', 'sigma_crit', 'energy']

# initialize arrays
results = {}
for name in var_names:
    if name == 'coilable':
        results[name] = [None for _ in range(n_pts)]
    else:
        results[name] = np.empty(n_pts) * np.nan

# colect folder names
dir_name = os.path.join(example_name, sim_dir)
folder_names = collect_folder_names(dir_name)

# get values
for folder_name in folder_names:

    # number
    i = get_int_number_from_str(folder_name)
    print(i)

    # get data
    data = get_results(dir_name, folder_name)

    # get data (linear buckling)
    coilable, sigma_crit = get_results_lin_buckle(data)

    # get data (Riks)
    _, (strain, stress), (energy, (x, y)), E_max = read_and_clean_results_riks(
        data, get_energy=True)

    # update coilability
    if coilable and E_max is not np.nan and E_max > max_strain:
        coilable = 2

    # save data
    results['coilable'][i] = coilable
    results['sigma_crit'][i] = sigma_crit
    results['energy'][i] = energy


#%% store results

# create new panda frame
results = pd.DataFrame(results)

# append results to panda frame
column_names = list(points.columns.values)
for name in var_names:
    if name in column_names:
        points.update(results)
        break
else:
    global_data['points'] = points.join(results)

# create new pickle file
with open(os.path.join(example_name, pkl_filename_output), 'wb') as file:
    pickle.dump(global_data, file)
