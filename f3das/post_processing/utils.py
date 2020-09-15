'''
Created on 2020-05-05 16:28:59
Last modified on 2020-09-14 16:15:22
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
import shutil

# local library
from f3das.utils.file_handling import get_unique_file_by_ext
from f3das.utils.file_handling import collect_folder_names
from f3das.utils.utils import get_int_number_from_str

# TODO: possibility of entering abaqus for post-processing

# function definition


def post_process_sims(pp_fnc, output_variables, example_name, sim_dir='analyses',
                      pkl_filename='DoE.pkl', create_new_file=''):
    '''
    Parameters
    ----------
    create_new_file : str
        If not empty, then a new file is created containing the information
        in `pkl_filename` plus the treated outputs.
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
    for variable in output_variables[::-1]:
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
    pkl_filename_output = create_new_file if create_new_file else pkl_filename
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


def concatenate_data(example_name, pkl_filename='DoE.pkl',
                     dict_filename='raw_data.pkl', sim_dir='analyses',
                     delete=False):
    '''
    Creates an unique file that contains all the information of the problem.

    Notes
    -----
    1. If file already exists, then data in 'analyses' is added to the
    already existing information (overriding pre-existing information).
    '''

    # load doe
    with open(os.path.join(example_name, pkl_filename), 'rb') as file:
        doe = pickle.load(file, encoding='latin1')

    # verify if file already exists
    if os.path.exists(os.path.join(example_name, dict_filename)):
        with open(os.path.join(example_name, dict_filename), 'rb') as file:
            data = pickle.load(file)
        data['doe'] = doe
    else:
        data = {'doe': doe,
                'raw_data': {}}

    # get available simulations
    dir_name = os.path.join(example_name, sim_dir)
    folder_names = collect_folder_names(dir_name)

    # store the information of each available DoE point
    for folder_name in folder_names:
        # simulation number
        i = get_int_number_from_str(folder_name)

        # get data
        data['raw_data'][i] = get_data(dir_name, folder_name)

        # delete folder
        if delete:
            shutil.rmtree(os.path.join(dir_name, folder_name))

    # save file
    with open(os.path.join(example_name, dict_filename), 'wb') as file:
        pickle.dump(data, file)
