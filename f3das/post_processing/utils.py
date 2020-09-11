'''
Created on 2020-05-05 16:28:59
Last modified on 2020-09-11 17:05:08
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

# local library
from f3das.utils.file_handling import get_unique_file_by_ext
from f3das.utils.file_handling import collect_folder_names
from f3das.utils.utils import get_int_number_from_str

# TODO: concatenate dicts


# function definition

def post_process_sims(pp_fnc, output_variables,
                      example_name, sim_dir='analyses',
                      pkl_filename='DoE.pkl', pkl_filename_output='DoE_results.pkl'):
    '''
    Notes
    -----
    1. If the output variables already exist in 'points', their values are
    updated only if the post-processing data for the given point is available
    in `sim_dir`. Otherwise, older values are kept.
    '''

    # get current pandas
    with open(os.path.join(example_name, pkl_filename), 'rb') as file:
        global_data = pickle.load(file)
    points = global_data['points']

    # add outputs
    column_names = list(points.columns.values)
    for variable in output_variables:
        if variable not in column_names:
            points.insert(loc=len(column_names), value=None, column=variable)

    # get available simulations
    dir_name = os.path.join(example_name, sim_dir)
    folder_names = collect_folder_names(dir_name)

    # get results
    for folder_name in folder_names:
        # simulation number
        i = get_int_number_from_str(folder_name)

        # get data
        data = get_data(dir_name, folder_name)

        # get results
        points.loc[i, output_variables] = pp_fnc(data)

    # create new pickle file
    with open(os.path.join(example_name, pkl_filename_output), 'wb') as file:
        pickle.dump(global_data, file)


def get_data(dir_name, folder_name='analyses'):

    # TODO: add possibility of using gzip

    # get filename and read data
    dir_name = os.path.join(dir_name, folder_name)
    filename = get_unique_file_by_ext(dir_name, ext='.pkl')

    with open(os.path.join(dir_name, filename), 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    return data
