'''
Created on 2020-05-05 16:28:59
Last modified on 2020-09-25 09:35:12

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# standard library
import os
import pickle
import gzip
import shutil

# third-party
import pandas as pd

# local library
from f3das.utils.file_handling import get_unique_file_by_ext
from f3das.utils.file_handling import collect_folder_names
from f3das.utils.utils import get_int_number_from_str
from f3das.utils.utils import read_pkl_file

# TODO: possibility of entering abaqus for post-processing


# function definition

def post_process_sims(pp_fnc, output_variables, example_name, sim_numbers=None,
                      sims_dir_name='analyses', data_filename='DoE.pkl',
                      raw_data_filename='raw_data.pkl', create_new_file=''):
    '''
    Parameters
    ----------
    create_new_file : str
        If not empty, then a new file is created containing the information
        in `data_filename` plus the treated outputs.
    Notes
    -----
    1. If the output variables already exist in 'points', their values are
    updated only if the post-processing data for the given point is available
    in `sims_dir_name`. Otherwise, older values are kept.
    '''
    # TODO: pass args to post-processing sim
    # TODO: verify if pp_fnc is callable

    # get current pandas
    with open(os.path.join(example_name, data_filename), 'rb') as file:
        global_data = pickle.load(file)
    points = global_data['points']

    # add outputs
    column_names = list(points.columns.values)
    for variable in output_variables[::-1]:
        if variable not in column_names:
            points.insert(loc=len(column_names), value=None, column=variable)

    # get available simulations
    data = collect_raw_data(example_name, sims_dir_name=sims_dir_name,
                            sim_numbers=sim_numbers, delete=False,
                            raw_data_filename=raw_data_filename)

    for i, data_sim in data.iteritems():
        # get results
        points.loc[i, output_variables] = pp_fnc(data_sim)

    # create new pickle file
    data_filename_output = create_new_file if create_new_file else data_filename
    with open(os.path.join(example_name, data_filename_output), 'wb') as file:
        pickle.dump(global_data, file)


def get_sim_data(dir_name, folder_name):

    # TODO: add possibility of using gzip

    # get filename and read data
    dir_name = os.path.join(dir_name, folder_name)
    filename = get_unique_file_by_ext(dir_name, ext='.pkl')

    with open(os.path.join(dir_name, filename), 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    return data


def concatenate_data(example_name, data_filename='DoE.pkl',
                     raw_data_filename='raw_data.pkl', sims_dir_name='analyses',
                     delete=False, compress=True, sim_numbers=None):
    '''
    Creates an unique file that contains all the information of the problem.

    Parameters
    ----------
    compress : bool
        If True, zips file using gzip. It may take longer. Depending on the data,
        the compression ratio may be huge.

    Notes
    -----
    1. If file already exists, then data in 'analyses' is added to the
    already existing information (overriding pre-existing information).
    '''

    # initialization
    open_file = gzip.open if compress else open

    # load doe
    with open(os.path.join(example_name, data_filename), 'rb') as file:
        doe = pickle.load(file, encoding='latin1')

    # verify if file already exists
    if os.path.exists(os.path.join(example_name, raw_data_filename)):
        data = read_pkl_file(os.path.join(example_name, raw_data_filename))
        data['doe'] = doe
    else:
        data = {'doe': doe,
                'raw_data': pd.Series(dtype=object)}

    # get available simulations
    raw_data = collect_raw_data_from_folders(example_name,
                                             sims_dir_name=sims_dir_name,
                                             sim_numbers=sim_numbers, delete=delete)
    data['raw_data'] = raw_data.combine_first(data['raw_data']).sort_index()

    # save file
    with open_file(os.path.join(example_name, raw_data_filename), 'wb') as file:
        pickle.dump(data, file)


def collect_raw_data(example_name, sims_dir_name='analyses', sim_numbers=None,
                     delete=False, raw_data_filename='raw_data.pkl'):
    if raw_data_filename:
        raw_data = read_pkl_file(os.path.join(example_name, raw_data_filename))['raw_data']
        if sim_numbers is not None:
            raw_data = raw_data.loc[sim_numbers]
    else:
        raw_data = collect_raw_data_from_folders(
            example_name, sims_dir_name=sims_dir_name,
            sim_numbers=sim_numbers, delete=False)

    return raw_data


def collect_raw_data_from_folders(example_name, sims_dir_name='analyses',
                                  sim_numbers=None, delete=False):

    # get available simulations
    dir_name = os.path.join(example_name, sims_dir_name)
    folder_names = collect_folder_names(dir_name, sim_numbers=sim_numbers)

    # store the information of each available DoE point
    raw_data = {}
    for folder_name in folder_names:
        # simulation number
        i = get_int_number_from_str(folder_name)

        # get data
        raw_data[i] = get_sim_data(dir_name, folder_name)

        # delete folder
        if delete:
            shutil.rmtree(os.path.join(dir_name, folder_name))

    return pd.Series(raw_data)
