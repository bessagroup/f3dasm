'''
Created on 2020-05-05 20:07:00
Last modified on 2020-05-07 15:04:21
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
from f3das.post_processing.common import get_results
from f3das.post_processing.supercompressible import get_results_lin_buckle
from f3das.post_processing.supercompressible import read_and_clean_results_riks


#%% initialization

example_name = 'example_supercompressible_3d'
sim_dir = 'analyses'
pkl_filename = 'DoE.pkl'
n_std_cleaning = None  # None if you don't want to "clean" data


#%% get results

# colect folder names
dir_name = os.path.join(example_name, sim_dir)
folder_names = collect_folder_names(dir_name)

# get values
coilable_values = []
sigma_crit_values = []
energy_values = []
for folder_name in folder_names:

    # get data
    data = get_results(dir_name, folder_name)

    # get data (linear buckling)
    coilable, sigma_crit = get_results_lin_buckle(data)

    # get data (Riks)
    _, (strain, stress), (energy, (x, y)) = read_and_clean_results_riks(
        data, get_energy=True)

    # save data
    coilable_values.append(coilable)
    sigma_crit_values.append(sigma_crit)
    energy_values.append(energy)


#%% post-process data before storing it

# delete outliers
if n_std_cleaning:
    y_temp = np.array([yy for yy in energy_values if yy is not None])
    y_mean = np.mean(y_temp)
    y_std = np.std(y_temp)
    y_thresh = y_mean + n_std_cleaning * y_std
    for i, yy in enumerate(energy_values):
        if yy is not None and yy > y_thresh:
            energy_values[i] = None


#%% store results

# create new panda frame
results = pd.DataFrame({'coilable': coilable_values,
                        'sigma_crit': sigma_crit_values,
                        'energy': energy_values})

# read file
with open(os.path.join(example_name, pkl_filename), 'rb') as file:
    data = pickle.load(file)

# append results to panda frame
data['points'] = data['points'].join(results)

# create new pickle file
new_filename = '%s_results.pkl' % pkl_filename.split('.')[0]
with open(os.path.join(example_name, new_filename), 'wb') as file:
    pickle.dump(data, file)
